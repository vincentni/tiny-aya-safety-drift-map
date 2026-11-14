import json
import re
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from auditkit.errors import ModelError
from auditkit.model import CallableModel

from driftmap import gen
from driftmap.core.errors import Breaker, OutOfMemory, classify_oom
from driftmap.core.records import Decoding, LabelRow
from driftmap.core.render import PATCH, render, token_count
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.gen import MAX_NEW_TOKENS
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import run_config
from driftmap.gen.rows import Prompt, benign_rows, run_batch
from tests.fakes.generator import FakeGenerator
from tests.fakes.tokenizers import build, served_template

CHECKPOINT = "tiny-aya-global--0123456789ab"
TEXTS = ["hello world", "bonjour canary", "नमस्ते 12345"]
OOM_TEXT = "CUDA out of memory. Tried to allocate 2.00 GiB"


def prompts_for(tokenizer, texts=TEXTS):
    return [Prompt.of(tokenizer, f"p-{i}", text) for i, text in enumerate(texts)]


def batch(model, prompts, tokenizer, set_="harmful", config=None, decoding=None):
    return run_batch(
        model,
        prompts,
        adapter=RawPromptAdapter(),
        config=config or run_config(16),
        tokenizer=tokenizer,
        checkpoint=CHECKPOINT,
        set=set_,
        language="en",
        decoding=decoding or Decoding(runtime="transformers"),
    )


def test_a_prompt_rendered_after_a_system_message_carries_that_rendering_into_its_row():
    tokenizer = build(template=served_template())
    plain, patched = Prompt.of(tokenizer, "p-0", "hello world"), Prompt.of(tokenizer, "p-0", "hello world", PATCH)
    assert plain == Prompt("p-0", "hello world", *render(tokenizer, "hello world")[:2])
    assert patched == Prompt("p-0", "hello world", *render(tokenizer, "hello world", PATCH)[:2])
    assert patched.rendered_prompt.count(PATCH) == 1 and PATCH not in plain.rendered_prompt

    [row] = batch(CallableModel(FakeGenerator()), [patched], tokenizer)

    assert (row.prompt, row.rendered_prompt) == ("hello world", patched.rendered_prompt)
    assert row.rendered_prefix_sha256 == patched.rendered_prefix_sha256 != plain.rendered_prefix_sha256
    # the patch is part of what the model is given, and is counted
    assert row.prompt_tokens == token_count(tokenizer, patched.rendered_prompt)
    assert row.prompt_tokens > token_count(tokenizer, plain.rendered_prompt)


def test_out_of_memory_stops_the_batch_instead_of_marking_it_unscored():
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    generator = FakeGenerator({prompts[1].rendered_prompt: torch.OutOfMemoryError(OOM_TEXT)})

    with pytest.raises(OutOfMemory) as raised:
        batch(CallableModel(generator), prompts, tokenizer)

    assert raised.value.exit_code == ExitCode.OOM == 11
    assert isinstance(raised.value, DriftmapError)
    assert OOM_TEXT in str(raised.value)
    assert isinstance(raised.value.__cause__, ModelError)
    assert classify_oom(raised.value)


def test_setup_class_error_also_leaves_the_batch_unscored():
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    generator = FakeGenerator({prompts[0].rendered_prompt: ValueError("bad argument")})
    rows = batch(CallableModel(generator), prompts, tokenizer)
    assert {(row.status, row.error) for row in rows} == {("unscored", "bad argument")}
    assert len(rows) == 3

    generator.scripted[prompts[0].rendered_prompt] = ValueError()
    rows = batch(CallableModel(generator), prompts, tokenizer)
    assert {(row.status, row.error, row.stop_reason) for row in rows} == {("unscored", "ValueError", "error")}


def test_own_error_inside_the_backend_is_raised_not_swallowed():
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    generator = FakeGenerator({prompts[0].rendered_prompt: Breaker("server gone")})
    with pytest.raises(Breaker, match="server gone") as raised:
        batch(CallableModel(generator), prompts, tokenizer)
    assert raised.value.exit_code == ExitCode.COLUMN_FAILED


def test_identical_batches_reach_the_model_every_time():
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    state = {"suffix": "world"}

    def generate(batch_prompts):
        calls.append(len(batch_prompts))
        return [f"hello {state['suffix']}" for _ in batch_prompts]

    calls = []
    first = batch(CallableModel(generate), prompts, tokenizer)
    state["suffix"] = "canary"
    second = batch(CallableModel(generate), prompts, tokenizer)

    assert calls == [3, 3]
    assert {row.response for row in first} == {"hello world"}
    assert {row.response for row in second} == {"hello canary"}


def test_config_passed_in_is_left_untouched():
    tokenizer = build()
    config = run_config(16)
    run_batch(
        CallableModel(FakeGenerator()),
        prompts_for(tokenizer),
        adapter=RawPromptAdapter(),
        config=config,
        tokenizer=tokenizer,
        checkpoint=CHECKPOINT,
        set="harmful",
        language="en",
        decoding=Decoding(runtime="transformers"),
    )
    assert config == run_config(16)


@pytest.mark.parametrize("replies", [["hello world"], ["hello"] * 4, []])
def test_reply_count_differing_from_the_prompt_count_is_refused(replies):
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    for model in (CallableModel(lambda batch_prompts: replies), lambda batch_prompts: replies):
        with pytest.raises(DriftmapError, match=f"{len(replies)} replies for 3 prompts") as raised:
            batch(model, prompts, tokenizer)
        assert raised.value.exit_code == ExitCode.INTEGRITY


class FailedOnly(FakeGenerator):
    failed: ClassVar[dict] = {}


class RepliesOnly(FakeGenerator):
    replies: ClassVar[dict] = {}


@pytest.mark.parametrize(
    "build_model", [FakeGenerator, FailedOnly, RepliesOnly, lambda: CallableModel(FakeGenerator())]
)
def test_llama_rows_are_refused_without_the_side_maps(build_model):
    tokenizer = build()
    model = build_model()
    generator = getattr(model, "_fn", model)
    with pytest.raises(DriftmapError, match="side maps") as raised:
        batch(model, prompts_for(tokenizer), tokenizer, decoding=Decoding(runtime="llama-server"))
    assert raised.value.exit_code == ExitCode.USAGE
    assert generator.calls == []


def test_token_limit_of_the_rows_must_be_the_one_the_run_uses():
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    generator = FakeGenerator()
    short = Decoding(runtime="transformers", max_new_tokens=100)
    with pytest.raises(DriftmapError, match=r"100.*256") as raised:
        batch(CallableModel(generator), prompts, tokenizer, decoding=short)
    assert raised.value.exit_code == ExitCode.USAGE
    assert generator.calls == []

    rows = batch(
        CallableModel(generator), prompts, tokenizer, config=replace(run_config(16), max_tokens=100), decoding=short
    )
    assert {(row.status, row.decoding.max_new_tokens) for row in rows} == {("ok", 100)}


def test_backends_share_the_token_limit_of_the_decoding_record():
    assert MAX_NEW_TOKENS == Decoding(runtime="transformers").max_new_tokens == run_config(16).max_tokens
    sources = Path(gen.__file__).parent.glob("*.py")
    assert [source.name for source in sources if re.search(r"\b256\b", source.read_text())] == []


def test_empty_batch_gives_no_rows_and_no_model_call():
    generator = FakeGenerator()
    assert batch(CallableModel(generator), [], build()) == []
    assert generator.calls == []


def test_benign_rows_score_chrf_against_the_target_and_drop_the_response():
    tokenizer = build()
    prompts = prompts_for(tokenizer)
    generator = FakeGenerator(
        {
            prompts[0].rendered_prompt: "bonjour le monde",
            prompts[1].rendered_prompt: "bonjour tout le monde",
            prompts[2].rendered_prompt: "",
        }
    )
    rows = batch(CallableModel(generator), prompts, tokenizer, set_="benign")
    targets = {"p-0": "bonjour le monde", "p-1": "bonjour le monde", "p-2": "bonjour le monde"}

    labels = benign_rows(rows, targets)

    assert all(isinstance(label, LabelRow) for label in labels)
    assert [label.prompt_id for label in labels] == ["p-0", "p-1", "p-2"]
    assert labels[0].chrf == 1.0
    assert labels[1].chrf == pytest.approx(0.72666, abs=1e-5)
    assert labels[2].chrf is None
    assert [label.raw_output_present for label in labels] == [True, True, False]
    for row, label in zip(rows, labels, strict=True):
        assert label.schema_ == "driftmap.label/1"
        assert (label.checkpoint, label.language, label.set) == (CHECKPOINT, "en", "benign")
        assert (label.prompt, label.rendered_prefix_sha256) == (row.prompt, row.rendered_prefix_sha256)
        assert (label.request_harm, label.refusal, label.response_harm) == ("n/a", "n/a", "n/a")
        assert label.judge is None
        dumped = json.loads(label.model_dump_json())
        assert "response" not in dumped
        assert "le monde" not in json.dumps(dumped)

    with pytest.raises(KeyError):
        benign_rows(rows, {"p-0": "bonjour le monde"})
