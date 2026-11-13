import http.client
import io
import json
import threading
import urllib.error
from collections import defaultdict, deque
from email.message import Message
from typing import Any

# what llama-server's /health returns while the model loads
LOADING_BODY = {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}
BOS_ID = 2
HEALTH_ARMS = ("ok", "loading", "refused")


class FakeTransport:
    """Scripted llama-server behind ``request(method, path, body) -> dict``."""

    def __init__(self, base_url: str = "http://127.0.0.1:9931") -> None:
        self.base_url = base_url
        self.health = "ok"
        # an int entry raises HTTPError with that status; replies_for wins
        self.completions: deque[dict[str, Any] | int] = deque()
        self.replies_for: defaultdict[str, deque[dict[str, Any] | int]] = defaultdict(deque)
        self.props: dict[str, Any] = {
            "total_slots": 1,
            "model_path": "fake.gguf",
            "default_generation_settings": {"n_ctx": 4096},
        }
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.max_in_flight = 0
        self._fail_next: deque[tuple[int, dict[str, Any] | None]] = deque()
        self._vocab: dict[str, int] = {}
        self._in_flight = 0
        self._lock = threading.Lock()

    def fail_next(self, status: int, body: dict[str, Any] | None = None) -> None:
        self._fail_next.append((status, body))

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.calls.append((method, path, body))
            failure = self._fail_next.popleft() if self._fail_next else None
        try:
            if failure is not None:
                status, failure_body = failure
                self._raise_http(
                    path,
                    status,
                    failure_body or {"error": {"code": status, "message": "fake failure", "type": "server_error"}},
                )
            if path == "/health":
                return self._health(path)
            if path == "/props":
                return dict(self.props)
            if path == "/tokenize":
                return {"tokens": self._tokenize(body or {})}
            if path == "/completion":
                return self._completion(path, body or {})
            self._raise_http(
                path, 404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}}
            )
        finally:
            with self._lock:
                self._in_flight -= 1

    def _completion(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            keyed = self.replies_for.get(body.get("prompt"))
            queue = keyed if keyed else self.completions
            assert queue, f"no scripted /completion reply left for {body!r}"
            reply = queue.popleft()
        if isinstance(reply, int):
            self._raise_http(
                path, reply, {"error": {"code": reply, "message": "scripted failure", "type": "server_error"}}
            )
        return dict(reply)

    def _health(self, path: str) -> dict[str, Any]:
        assert self.health in HEALTH_ARMS, f"health arm {self.health!r} is not one of {HEALTH_ARMS}"
        if self.health == "ok":
            return {"status": "ok"}
        if self.health == "loading":
            self._raise_http(path, 503, LOADING_BODY)
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    def _raise_http(self, path: str, status: int, body: dict[str, Any]) -> None:
        payload = io.BytesIO(json.dumps(body).encode())
        raise urllib.error.HTTPError(
            self.base_url + path, status, http.client.responses.get(status, ""), Message(), payload
        )

    def _tokenize(self, body: dict[str, Any]) -> list[int]:
        ids = [self._vocab.setdefault(word, len(self._vocab) + 10) for word in body["content"].split()]
        return [BOS_ID, *ids] if body.get("add_special") else ids
