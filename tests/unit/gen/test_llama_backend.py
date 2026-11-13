import io
import itertools
import json
import logging
import threading
import time
import urllib.error
import urllib.request

import pytest
from auditkit.model import CallableModel

from driftmap.core.errors import Breaker
from driftmap.core.records import Decoding
from driftmap.core.render import token_count
from driftmap.gen import llama_backend
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import run_config
from driftmap.gen.llama_backend import (
    Completion,
    HttpTransport,
    LlamaCallable,
    LlamaServerClient,
    server_argv,
)
from driftmap.gen.rows import Prompt, run_batch
from tests.fakes.clock import FakeClock
from tests.fakes.tokenizers import build
from tests.fakes.transport import BOS_ID, FakeTransport

CHECKPOINT = "tiny-aya-global-repaired-GGUF--0123456789ab"


class SwitchingClock(FakeClock):
    """Changes the server's health arm once a given time has been slept."""

    def __init__(self, transport, schedule):
        super().__init__()
        self.transport = transport
        self.schedule = schedule

    def sleep(self, seconds):
        super().sleep(seconds)
        arm = self.schedule.get(sum(self.sleeps))
        if arm:
            self.transport.health = arm


def prompts_for(tokenizer, count):
    return [Prompt.of(tokenizer, f"p-{i}", f"hello {i}") for i in range(count)]


def reply(content="hello world", stop_type="eos", truncated=False, tokens_evaluated=12):
    return {"content": content, "tokens_evaluated": tokens_evaluated, "stop_type": stop_type, "truncated": truncated}


def batch(model, prompts, tokenizer):
    return run_batch(
        model,
        prompts,
        adapter=RawPromptAdapter(),
        config=run_config(16),
        tokenizer=tokenizer,
        checkpoint=CHECKPOINT,
        set="harmful",
        language="en",
        decoding=Decoding(runtime="llama-server"),
    )


def completions(transport):
    return [body for _, path, body in transport.calls if path == "/completion"]


def test_server_is_awaited_then_completions_post_one_exact_body():
    """Health is polled every 5 s for at most 180 s; completions post one exact body, four at a time (TS-GEN-8)."""
    transport = FakeTransport()
    transport.health = "refused"
    clock = SwitchingClock(transport, {10: "loading", 20: "ok"})
    client = LlamaServerClient(transport, clock=clock)
    client.wait_healthy()
    assert clock.sleeps == [5, 5, 5, 5]
    assert transport.calls == [("GET", "/health", None)] * 5

    for arm in ("loading", "refused"):
        down = FakeTransport()
        down.health = arm
        waiting = FakeClock()
        with pytest.raises(Breaker) as raised:
            LlamaServerClient(down, clock=waiting).wait_healthy()
        assert raised.value.exit_code == 13
        assert waiting.sleeps == [5] * 36
        assert down.calls == [("GET", "/health", None)] * 37

    tokenizer = build()
    prompts = prompts_for(tokenizer, 8)
    for i, prompt in enumerate(prompts):
        transport.replies_for[prompt.rendered_prompt].append(
            reply(content=f"hello {i} world", stop_type="limit" if i == 3 else "eos", tokens_evaluated=20 + i)
        )
    barrier = threading.Barrier(4, timeout=10)
    completion = transport._completion

    def held(path, body):
        barrier.wait()
        # a fifth request in flight would overlap this pause
        time.sleep(0.05)
        return completion(path, body)

    transport._completion = held
    callable_ = LlamaCallable(client, pool=4)
    rows = batch(CallableModel(callable_), prompts, tokenizer)

    assert transport.max_in_flight == 4
    posted = sorted(json.dumps(body, sort_keys=True) for body in completions(transport))
    assert posted == sorted(
        json.dumps(
            {
                "prompt": prompt.rendered_prompt,
                "temperature": 0,
                "top_k": 1,
                "n_predict": 256,
                "cache_prompt": False,
                "stream": False,
            },
            sort_keys=True,
        )
        for prompt in prompts
    )
    assert callable_.failed == {}
    for i, (prompt, row) in enumerate(zip(prompts, rows, strict=True)):
        assert callable_.replies[prompt.rendered_prompt] == Completion(
            f"hello {i} world", 20 + i, "limit" if i == 3 else "eos", False
        )
        assert (row.prompt_id, row.response, row.status) == (prompt.prompt_id, f"hello {i} world", "ok")
        assert row.stop_reason == ("length" if i == 3 else "eos")
        assert row.generated_tokens == 3
        assert row.prompt_tokens == token_count(tokenizer, prompt.rendered_prompt)


def test_failures_are_retried_three_times_then_trip_the_breaker_at_eight():
    """A prompt failing 3 times is unscored; 8 failures in a row or a failed health check break (TS-GEN-9)."""
    tokenizer = build()
    prompts = prompts_for(tokenizer, 100)
    failing = {5, 20, 35, 50, 65, 80}
    transport = FakeTransport()
    for i, prompt in enumerate(prompts):
        transport.replies_for[prompt.rendered_prompt].extend([500, 500, 500] if i in failing else [reply()])
    clock = FakeClock()
    callable_ = LlamaCallable(LlamaServerClient(transport, clock=clock), pool=1)

    rows = batch(callable_, prompts, tokenizer)

    assert len(completions(transport)) == 94 + 3 * 6
    assert [i for i, row in enumerate(rows) if row.status == "unscored"] == sorted(failing)
    assert sum(row.status == "unscored" for row in rows) / len(rows) == 0.06
    assert set(callable_.failed) == {prompts[i].rendered_prompt for i in failing}
    for i in failing:
        assert rows[i].error == "HTTPError: HTTP Error 500: Internal Server Error"
        assert (rows[i].response, rows[i].stop_reason) == (None, "error")
    assert len(clock.sleeps) == 2 * 6
    assert all(1 <= first <= 2 and 2 <= second <= 3 for first, second in itertools.batched(clock.sleeps, 2))

    seven = FakeTransport()
    for prompt, script in zip(prompts[:4], ([500] * 3, [500] * 3, [500, reply()], [500] * 3), strict=True):
        seven.replies_for[prompt.rendered_prompt].extend(script)
    spared = LlamaCallable(LlamaServerClient(seven, clock=FakeClock()), pool=1)
    assert [row.status for row in batch(spared, prompts[:4], tokenizer)] == ["unscored", "unscored", "ok", "unscored"]

    eight = FakeTransport()
    for prompt in prompts[:4]:
        eight.replies_for[prompt.rendered_prompt].extend([500] * 3)
    tripped = LlamaCallable(LlamaServerClient(eight, clock=FakeClock()), pool=1)
    with pytest.raises(Breaker) as raised:
        batch(CallableModel(tripped), prompts[:4], tokenizer)
    assert raised.value.exit_code == 13
    assert len(completions(eight)) == 8
    calls = len(eight.calls)
    with pytest.raises(Breaker):
        tripped([prompts[0].rendered_prompt])
    assert len(eight.calls) == calls

    between = FakeTransport()
    between.completions.extend([reply(), reply()])
    serving = LlamaCallable(LlamaServerClient(between, clock=FakeClock()), pool=4)
    assert [row.status for row in batch(serving, prompts[:1], tokenizer)] == ["ok"]
    between.health = "refused"
    with pytest.raises(Breaker) as raised:
        batch(serving, prompts[1:2], tokenizer)
    assert raised.value.exit_code == 13
    assert len(completions(between)) == 1
    between.health = "ok"
    with pytest.raises(Breaker):
        batch(serving, prompts[1:2], tokenizer)


def test_concurrent_failures_trip_the_breaker_once_for_the_whole_batch():
    tokenizer = build()
    prompts = prompts_for(tokenizer, 12)
    transport = FakeTransport()
    for prompt in prompts:
        transport.replies_for[prompt.rendered_prompt].extend([500] * 3)
    callable_ = LlamaCallable(LlamaServerClient(transport, clock=FakeClock()), pool=4)
    with pytest.raises(Breaker):
        callable_([prompt.rendered_prompt for prompt in prompts])
    assert 8 <= len(completions(transport)) <= 8 + 3


def test_breaker_reached_on_a_last_attempt_raises_in_that_batch(monkeypatch):
    monkeypatch.setattr(llama_backend, "BREAKER_LIMIT", 3)
    transport = FakeTransport()
    transport.replies_for["a"].extend([500, 500, 500])
    callable_ = LlamaCallable(LlamaServerClient(transport, clock=FakeClock()), pool=1)
    with pytest.raises(Breaker, match="3 requests in a row") as raised:
        callable_(["a"])
    assert isinstance(raised.value.__cause__, urllib.error.HTTPError)


def test_side_maps_describe_the_latest_batch_only():
    transport = FakeTransport()
    transport.replies_for["a"].extend([500, 500, 500])
    transport.replies_for["b"].append(reply())
    callable_ = LlamaCallable(LlamaServerClient(transport, clock=FakeClock()), pool=1)
    assert callable_(["a"]) == [""]
    assert set(callable_.failed) == {"a"}
    assert callable_(["b"]) == ["hello world"]
    assert callable_.failed == {}
    assert set(callable_.replies) == {"b"}


def test_malformed_reply_counts_as_a_failed_request():
    transport = FakeTransport()
    transport.completions.extend([{"content": "x"}] * 3)
    callable_ = LlamaCallable(LlamaServerClient(transport, clock=FakeClock()), pool=1)
    assert callable_(["a"]) == [""]
    assert callable_.failed == {"a": "KeyError: 'tokens_evaluated'"}


def test_complete_is_one_attempt_and_tokenize_asks_for_special_tokens():
    transport = FakeTransport()
    transport.completions.extend([reply(tokens_evaluated=7), 500])
    client = LlamaServerClient(transport, clock=FakeClock())
    assert client.complete("hello") == Completion("hello world", 7, "eos", False)
    with pytest.raises(urllib.error.HTTPError):
        client.complete("hello")
    assert len(completions(transport)) == 2

    assert client.tokenize("hello world", add_special=True)[0] == BOS_ID
    assert transport.calls[-1] == ("POST", "/tokenize", {"content": "hello world", "add_special": True})
    assert len(client.tokenize("hello world")) == 2
    assert transport.calls[-1][2]["add_special"] is False


def test_bos_check_compares_evaluated_tokens_with_the_tokenizer(caplog):
    tokenizer = build()
    prompts = [prompt.rendered_prompt for prompt in prompts_for(tokenizer, 3)]
    counts = [token_count(tokenizer, prompt) for prompt in prompts]

    transport = FakeTransport()
    transport.completions.extend(reply(tokens_evaluated=count) for count in counts)
    client = LlamaServerClient(transport, clock=FakeClock())
    assert client.bos_check(prompts, tokenizer) is True
    assert all(path == "/completion" for _, path, _ in transport.calls)

    doubled = FakeTransport()
    doubled.completions.extend(reply(tokens_evaluated=count + (i == 1)) for i, count in enumerate(counts))
    client = LlamaServerClient(doubled, clock=FakeClock())
    with caplog.at_level(logging.ERROR):
        assert client.bos_check(prompts, tokenizer) is False
    assert len(completions(doubled)) == 3
    assert [call for call in doubled.calls if call[1] == "/tokenize"] == [
        ("POST", "/tokenize", {"content": prompts[1], "add_special": True})
    ]
    assert f"prompt 1: server evaluated {counts[1] + 1} tokens, tokenizer gives {counts[1]}" in caplog.text
    assert prompts[1] not in caplog.text

    assert LlamaServerClient(FakeTransport(), clock=FakeClock()).bos_check([], tokenizer) is False


def test_server_argv_binds_loopback_with_four_slots():
    assert server_argv("model-q4_k_m.gguf") == [
        "-m",
        "model-q4_k_m.gguf",
        "-ngl",
        "all",
        "--parallel",
        "4",
        "-c",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--no-slots",
    ]


def test_http_transport_sends_json_with_a_timeout(monkeypatch):
    seen = []

    def open_(self, request, data=None, timeout=None):
        seen.append((request.full_url, request.get_method(), request.data, request.get_header("Content-type"), timeout))
        return io.BytesIO(b'{"status": "ok"}')

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_)
    client = LlamaServerClient()
    assert isinstance(client.transport, HttpTransport)
    assert client.transport.request("GET", "/health") == {"status": "ok"}
    assert client.transport.request("POST", "/tokenize", {"content": "hi"}) == {"status": "ok"}
    assert seen == [
        ("http://127.0.0.1:8080/health", "GET", None, None, 120),
        ("http://127.0.0.1:8080/tokenize", "POST", b'{"content": "hi"}', "application/json", 120),
    ]
    assert LlamaServerClient(base_url="http://127.0.0.1:9931").transport.base_url == "http://127.0.0.1:9931"
