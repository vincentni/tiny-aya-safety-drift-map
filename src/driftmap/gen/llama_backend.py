"""Client for a local llama-server and a batch callable over it."""

import http.client
import json
import logging
import random
import threading
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

from ..core.errors import Breaker
from ..core.render import token_count
from . import ERROR_CHARS, MAX_NEW_TOKENS

log = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8080"
REQUEST_TIMEOUT_S = 120
HEALTH_INTERVAL_S = 5
HEALTH_TIMEOUT_S = 180
ATTEMPTS = 3
BREAKER_LIMIT = 8
# refused, timed out, an HTTP status, or a reply missing fields
REQUEST_ERRORS = (OSError, http.client.HTTPException, ValueError, LookupError)


class Completion(NamedTuple):
    content: str
    tokens_evaluated: int
    stop_type: str
    truncated: bool


def server_argv(gguf: str | Path) -> list[str]:
    """Arguments for llama-server: four slots on loopback port 8080."""
    slots = ["--parallel", "4", "-c", "8192"]
    return ["-m", str(gguf), "-ngl", "all", *slots, "--host", "127.0.0.1", "--port", "8080", "--no-slots"]


class HttpTransport:
    """JSON requests to one server, each with a timeout."""

    def __init__(self, base_url: str = BASE_URL, timeout_s: float = REQUEST_TIMEOUT_S) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        # a proxy from the environment must not sit in front of loopback
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parsed JSON reply; raises ``urllib.error.URLError`` or ``HTTPError`` on failure."""
        data = None if body is None else json.dumps(body).encode()
        headers = {} if body is None else {"Content-Type": "application/json"}
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        with self._opener.open(request, timeout=self.timeout_s) as response:
            return json.load(response)


class LlamaServerClient:
    """Single-attempt calls to llama-server; retries belong to the caller."""

    def __init__(self, transport: Any = None, base_url: str = BASE_URL, clock: Any = time) -> None:
        self.transport = transport or HttpTransport(base_url)
        self.clock = clock

    def healthy(self) -> bool:
        """True when ``/health`` answers ok; loading and refused are both not ready."""
        try:
            return self.transport.request("GET", "/health").get("status") == "ok"
        except REQUEST_ERRORS:
            return False

    def wait_healthy(self) -> None:
        """Poll ``/health`` every 5 s; raises ``Breaker`` if not ready within 180 s."""
        deadline = self.clock.monotonic() + HEALTH_TIMEOUT_S
        while not self.healthy():
            if self.clock.monotonic() >= deadline:
                raise Breaker(f"llama-server not healthy after {HEALTH_TIMEOUT_S} s")
            self.clock.sleep(HEALTH_INTERVAL_S)

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        """Token ids the server gives ``text``, with BOS when ``add_special``."""
        return self.transport.request("POST", "/tokenize", {"content": text, "add_special": add_special})["tokens"]

    def complete(self, prompt: str) -> Completion:
        """One greedy completion up to the shared token limit; raises on any transport failure."""
        reply = self.transport.request(
            "POST",
            "/completion",
            {
                "prompt": prompt,
                "temperature": 0,
                "top_k": 1,
                "n_predict": MAX_NEW_TOKENS,
                "cache_prompt": False,
                "stream": False,
            },
        )
        return Completion(reply["content"], reply["tokens_evaluated"], reply["stop_type"], reply["truncated"])

    def bos_check(self, prompts: Sequence[str], tokenizer) -> bool:
        """True when there are prompts and the server evaluates the tokenizer's token count for each."""
        matches = []
        for index, prompt in enumerate(prompts):
            evaluated, expected = self.complete(prompt).tokens_evaluated, token_count(tokenizer, prompt)
            matches.append(evaluated == expected)
            if evaluated != expected:
                log.error(
                    "bos check prompt %d: server evaluated %d tokens, tokenizer gives %d; server ids start %s",
                    index,
                    evaluated,
                    expected,
                    self.tokenize(prompt, add_special=True)[:3],
                )
        return bool(matches) and all(matches)


class LlamaCallable:
    """``list[str] -> list[str]`` over the server; ``failed`` and ``replies`` describe the latest batch."""

    def __init__(self, client: LlamaServerClient, pool: int = 4) -> None:
        self.client = client
        self.pool = pool
        self.failed: dict[str, str] = {}
        self.replies: dict[str, Completion] = {}
        self._failures = 0
        self._open = False
        self._lock = threading.Lock()

    def __call__(self, prompts: list[str]) -> list[str]:
        self.failed, self.replies = {}, {}
        if not self._open and not self.client.healthy():
            self._open = True
        if self._open:
            raise Breaker("llama-server is not answering")
        with ThreadPoolExecutor(max_workers=self.pool) as pool:
            return list(pool.map(self._complete, prompts))

    def _complete(self, prompt: str) -> str:
        error: Exception | None = None
        for attempt in range(ATTEMPTS):
            if self._open:
                raise Breaker("llama-server is not answering")
            try:
                reply = self.client.complete(prompt)
            except REQUEST_ERRORS as caught:
                error = caught
                self._count(failed=True)
                if self._open:
                    raise Breaker(f"llama-server failed {BREAKER_LIMIT} requests in a row") from caught
                if attempt + 1 < ATTEMPTS:
                    self.client.clock.sleep(2**attempt + random.random())
                continue
            self._count(failed=False)
            self.replies[prompt] = reply
            return reply.content
        self.failed[prompt] = f"{type(error).__name__}: {error}"[:ERROR_CHARS]
        return ""

    def _count(self, failed: bool) -> None:
        with self._lock:
            self._failures = self._failures + 1 if failed else 0
            self._open = self._open or self._failures >= BREAKER_LIMIT
