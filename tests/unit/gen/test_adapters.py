import itertools
import json

import pytest
from auditkit import Sample, evaluate
from auditkit.adapter import GenerationAdapter
from auditkit.errors import ModelError
from auditkit.model import CallableModel

from driftmap.core.records import Decoding
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import run_config
from driftmap.gen.llama_backend import LlamaCallable, LlamaServerClient
from driftmap.gen.rows import Prompt, run_batch
from tests.fakes.clock import FakeClock
from tests.fakes.generator import FakeGenerator
from tests.fakes.tokenizers import build
from tests.fakes.transport import FakeTransport

CHECKPOINT = "tiny-aya-global--0123456789ab"
TEXTS = ["hello world", "bonjour canary", "नमस्ते 12345", "hello 12345"]
LONG = " ".join(["hello"] * 256)


def prompts_for(tokenizer, texts=TEXTS):
    return [Prompt.of(tokenizer, f"p-{i}", text) for i, text in enumerate(texts)]


def reply(content="hello world", stop_type="eos", truncated=False):
    return {"content": content, "tokens_evaluated": 12, "stop_type": stop_type, "truncated": truncated}


def llama(transport, pool=4):
    return LlamaCallable(LlamaServerClient(transport, clock=FakeClock()), pool=pool)


def batch(model, prompts, tokenizer, runtime="transformers"):
    return run_batch(
        model,
        prompts,
        adapter=RawPromptAdapter(),
        config=run_config(16),
        tokenizer=tokenizer,
        checkpoint=CHECKPOINT,
        set="harmful",
        language="en",
        decoding=Decoding(runtime=runtime),
    )


def test_adapter_marks_every_request_as_already_rendered():
    config = run_config(4)
    plain = GenerationAdapter().adapt(Sample(input="x"), config)[0]
    raw = RawPromptAdapter().adapt(Sample(input="x"), config)[0]
    assert "apply_chat_template" not in plain.params
    assert raw.params == {**plain.params, "apply_chat_template": False}
    assert raw.prompt == "x"
    assert RawPromptAdapter().identity() != GenerationAdapter().identity()


def test_empty_failed_and_truncated_replies_are_unscored_with_distinct_errors():
    """Empty, transport-failed and truncated replies are unscored, each with its own error (TS-GEN-3)."""
    tokenizer = build()
    prompts = prompts_for(tokenizer)

    generator = FakeGenerator({prompts[1].rendered_prompt: "", prompts[2].rendered_prompt: " \n"})
    rows = batch(CallableModel(generator), prompts, tokenizer)
    assert [row.prompt_id for row in rows] == [prompt.prompt_id for prompt in prompts]
    assert [(row.status, row.error) for row in rows] == [
        ("ok", None),
        ("unscored", "empty"),
        ("unscored", "empty"),
        ("ok", None),
    ]
    assert [row.response for row in rows[1:3]] == [None, None]
    assert [row.generated_tokens for row in rows[1:3]] == [0, 0]

    transport = FakeTransport()
    transport.replies_for[prompts[0].rendered_prompt].append(reply())
    transport.replies_for[prompts[1].rendered_prompt].extend([500, 500, 500])
    transport.replies_for[prompts[2].rendered_prompt].append(reply(truncated=True))
    transport.replies_for[prompts[3].rendered_prompt].append(reply(content=""))
    callable_ = llama(transport)
    rows = batch(CallableModel(callable_), prompts, tokenizer, runtime="llama-server")

    assert [row.prompt_id for row in rows] == [prompt.prompt_id for prompt in prompts]
    assert [(row.status, row.stop_reason) for row in rows] == [
        ("ok", "eos"),
        ("unscored", "error"),
        ("unscored", "truncated"),
        ("unscored", "eos"),
    ]
    assert rows[0].response == "hello world"
    assert "500" in rows[1].error
    assert rows[1].error == callable_.failed[prompts[1].rendered_prompt]
    assert rows[2].error == "truncated"
    assert rows[3].error == "empty"
    assert len({row.error for row in rows[1:]}) == 3
    assert [row.response for row in rows[1:]] == [None, None, None]
    posted = [body["prompt"] for _, path, body in transport.calls if path == "/completion"]
    assert posted.count(prompts[1].rendered_prompt) == 3


def test_backend_error_leaves_the_whole_batch_unscored_and_the_run_going():
    """A backend error on one prompt leaves every row of that batch unscored (TS-GEN-5)."""
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    generator = FakeGenerator({prompts[2].rendered_prompt: RuntimeError("backend down")})
    model = CallableModel(generator)

    samples = [Sample(input=prompt.rendered_prompt, id=prompt.prompt_id) for prompt in prompts]
    with pytest.raises(ModelError) as raised:
        evaluate(samples, model, adapter=RawPromptAdapter(), config=run_config(16))
    assert isinstance(raised.value.__cause__, RuntimeError)
    message = "generate failed after 0 retries: RuntimeError: backend down"
    assert str(raised.value) == message

    rows = batch(model, prompts, tokenizer)
    assert [row.prompt_id for row in rows] == [prompt.prompt_id for prompt in prompts]
    assert {(row.status, row.error, row.stop_reason, row.response) for row in rows} == {
        ("unscored", message, "error", None)
    }
    assert all(row.prompt_tokens > 0 and row.generated_tokens == 0 for row in rows)

    generator.scripted[prompts[2].rendered_prompt] = RuntimeError("x" * 500)
    assert {len(row.error) for row in batch(model, prompts, tokenizer)} == {200}

    following = batch(model, prompts[:2], tokenizer)
    assert [row.status for row in following] == ["ok", "ok"]


def test_stop_type_and_truncation_map_to_one_status_rule_on_both_runtimes():
    """Truncated replies are unscored, a hit token limit is ok with length, on both runtimes (TS-GEN-10)."""
    tokenizer = build()
    grid = list(itertools.product(["eos", "limit", "word", "none"], [False, True]))
    prompts = prompts_for(tokenizer, [f"hello {i}" for i in range(len(grid))])
    transport = FakeTransport()
    for prompt, (stop_type, truncated) in zip(prompts, grid, strict=True):
        transport.replies_for[prompt.rendered_prompt].append(reply(stop_type=stop_type, truncated=truncated))
    rows = batch(llama(transport), prompts, tokenizer, runtime="llama-server")
    assert {cell: (row.status, row.error, row.stop_reason) for cell, row in zip(grid, rows, strict=True)} == {
        ("eos", False): ("ok", None, "eos"),
        ("eos", True): ("unscored", "truncated", "truncated"),
        ("limit", False): ("ok", None, "length"),
        ("limit", True): ("unscored", "truncated", "truncated"),
        ("word", False): ("ok", None, None),
        ("word", True): ("unscored", "truncated", "truncated"),
        ("none", False): ("ok", None, None),
        ("none", True): ("unscored", "truncated", "truncated"),
    }

    fixture = {"at the limit": LONG, "below it": LONG[6:], "short": "hello world", "empty": ""}
    prompts = prompts_for(tokenizer, list(fixture))
    scripted = {prompt.rendered_prompt: text for prompt, text in zip(prompts, fixture.values(), strict=True)}
    hf_rows = batch(CallableModel(FakeGenerator(scripted)), prompts, tokenizer)
    assert [row.generated_tokens for row in hf_rows] == [256, 255, 2, 0]
    assert [(row.status, row.stop_reason) for row in hf_rows] == [
        ("ok", "length"),
        ("ok", "eos"),
        ("ok", "eos"),
        ("unscored", "eos"),
    ]

    transport = FakeTransport()
    for prompt, text in zip(prompts, fixture.values(), strict=True):
        stop_type = "limit" if text == LONG else "eos"
        transport.replies_for[prompt.rendered_prompt].append(reply(content=text, stop_type=stop_type))
    llama_rows = batch(llama(transport), prompts, tokenizer, runtime="llama-server")
    assert [(row.status, row.stop_reason) for row in llama_rows] == [(row.status, row.stop_reason) for row in hf_rows]
    assert [row.generated_tokens for row in llama_rows] == [row.generated_tokens for row in hf_rows]


def test_every_row_of_both_runtimes_carries_its_decoding():
    """Each row records greedy 256-token decoding with its runtime and no prompt cache (TS-GEN-11)."""
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    transport = FakeTransport()
    transport.replies_for[prompts[0].rendered_prompt].append(reply())
    transport.replies_for[prompts[1].rendered_prompt].append(reply(truncated=True))
    transport.replies_for[prompts[2].rendered_prompt].extend([500, 500, 500])
    transport.replies_for[prompts[3].rendered_prompt].append(reply(stop_type="limit"))
    failing = FakeGenerator({prompts[0].rendered_prompt: RuntimeError("backend down")})
    by_runtime = {
        "transformers": batch(CallableModel(FakeGenerator({prompts[1].rendered_prompt: ""})), prompts, tokenizer)
        + batch(CallableModel(failing), prompts, tokenizer),
        "llama-server": batch(llama(transport), prompts, tokenizer, runtime="llama-server"),
    }

    assert [len(rows) for rows in by_runtime.values()] == [8, 4]
    for runtime, rows in by_runtime.items():
        assert {row.status for row in rows} == {"ok", "unscored"}
        for row in rows:
            assert row.decoding == Decoding(greedy=True, max_new_tokens=256, runtime=runtime, cache_prompt=False)
            assert json.loads(row.model_dump_json())["decoding"] == {
                "greedy": True,
                "max_new_tokens": 256,
                "runtime": runtime,
                "cache_prompt": False,
            }
