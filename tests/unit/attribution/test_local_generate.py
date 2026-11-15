"""The workstation generator, with a scripted backend in place of the model."""

import json
import sys

import pytest
import torch
from auditkit.model import CallableModel

from driftmap.attribution import local_generate
from driftmap.core.render import PATCH, render
from driftmap.gen import hf_backend
from tests.fakes.generator import FakeGenerator
from tests.fakes.tokenizers import build, served_template

LABEL = "owner/model@0123456789ab"
ROWS = [{"id": f"row-{number}", "prompt": text, "language": "en"} for number, text in enumerate(("hello", "world"))]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


class World:
    """A tokenizer and a backend in place of the checkpoint, and the arguments that name the files."""

    def __init__(self, tmp_path, monkeypatch, scripted=None):
        self.tokenizer = build(template=served_template())
        self.generator = FakeGenerator(scripted)
        self.loaded, self.pinned = [], []
        for name in (
            "XDG_CACHE_HOME",
            "HF_HUB_OFFLINE",
            "TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            monkeypatch.setenv(name, "")
        monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
        monkeypatch.setattr(local_generate, "pin_to_node", self.pinned.append)
        monkeypatch.setattr(
            local_generate, "load_rendering", lambda *given: self.loaded.append(given) or self.tokenizer
        )
        monkeypatch.setattr(hf_backend, "load_hf_model", lambda *given: CallableModel(self.generator))
        self.output = tmp_path / "out" / "answers.jsonl"
        self.argv = [
            "--model",
            str(tmp_path / "model"),
            "--label",
            LABEL,
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
        self.argv += ["--output", str(self.output), "--batch-size", "2", "--threads", "1"]

    def run(self, rows, *extra):
        path = write_jsonl(self.output.parent.parent / "in.jsonl", rows)
        return local_generate.main([*self.argv, "--input", str(path), *extra])


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_rows_are_answered_in_batches_through_the_served_template_and_a_rerun_skips_them(world, tmp_path, capsys):
    rows = [*ROWS, {"id": "row-2", "prompt": "bonjour"}]

    assert world.run(rows, "--max-new-tokens", "2") == 0

    written = read_jsonl(world.output)
    assert [row["id"] for row in written] == ["row-0", "row-1", "row-2"]
    assert [len(call) for call in world.generator.calls] == [2, 1]
    assert world.loaded == [(tmp_path / "model", tmp_path / "manifest.json", "keep")]
    first = written[0]
    rendered = render(world.tokenizer, "hello").rendered_prompt
    assert first["rendered_prompt"] == rendered and world.generator.calls[0][0] == rendered
    assert first["response"] == rendered and first["model"] == LABEL and first["language"] == "en"
    assert first["stop_reason"] == "length" and first["generated_tokens"] >= 2
    assert first["max_new_tokens"] == 2 and first["preamble"] == "keep" and first["error"] is None
    assert first["prompt_tokens"] == len(world.tokenizer(rendered)["input_ids"])
    assert "3 rows, 3 to generate" in capsys.readouterr().out

    assert world.run(rows) == 0
    assert len(world.generator.calls) == 2 and len(read_jsonl(world.output)) == 3
    assert "3 rows, 0 to generate" in capsys.readouterr().out


def test_a_system_text_an_opener_and_the_safety_patch_are_rendered_as_the_job_renders_them(world):
    tokenizer = world.tokenizer
    plain = local_generate.render_row(tokenizer, {"id": "a", "prompt": "hello"})
    system = local_generate.render_row(tokenizer, {"id": "a", "prompt": "hello", "system": "canary"})
    opened = local_generate.render_row(tokenizer, {"id": "a", "prompt": "hello", "opener": "world"})
    patched = local_generate.render_row(tokenizer, {"id": "a", "prompt": "hello"}, "patch")

    assert plain.rendered_prompt == render(tokenizer, "hello").rendered_prompt and plain.prompt_id == "a"
    assert "# Developer Preamble\ncanary" in system.rendered_prompt and not system.rendered_prompt.startswith("<BOS")
    assert opened.rendered_prompt == plain.rendered_prompt + "world"
    assert patched.rendered_prompt == render(tokenizer, "hello", PATCH).rendered_prompt
    with pytest.raises(ValueError, match="cannot also be rendered under --preamble patch"):
        local_generate.render_row(tokenizer, {"id": "a", "prompt": "hello", "system": "canary"}, "patch")


def test_an_opener_is_part_of_the_answer_that_is_written(world):
    world.generator.scripted = {render(world.tokenizer, "hello").rendered_prompt + "Sure,": " here it is"}

    assert world.run([{"id": "row-0", "prompt": "hello", "opener": "Sure,"}]) == 0

    [row] = read_jsonl(world.output)
    assert row["response"] == "Sure, here it is" and row["continuation"] == " here it is" and row["opener"] == "Sure,"


def test_a_batch_the_backend_fails_on_is_left_for_a_rerun_and_the_other_batches_are_written(world, capsys):
    failing = render(world.tokenizer, "hello").rendered_prompt
    world.generator.scripted = {failing: RuntimeError("backend gone")}
    rows = [*ROWS, {"id": "row-2", "prompt": "bonjour"}]

    assert world.run(rows) == 1

    assert [row["id"] for row in read_jsonl(world.output)] == ["row-2"]
    assert "backend error on a batch of 2" in capsys.readouterr().err
    world.generator.scripted = {}
    assert world.run(rows) == 0
    assert sorted(row["id"] for row in read_jsonl(world.output)) == ["row-0", "row-1", "row-2"]


def test_a_shard_takes_every_nth_row_and_a_limit_caps_the_pending_ones(world):
    rows = [{"id": f"row-{number}", "prompt": "hello"} for number in range(7)]

    assert world.run(rows, "--shard", "1/3", "--limit", "2") == 0
    assert [row["id"] for row in read_jsonl(world.output)] == ["row-1", "row-4"] and world.pinned == [1]


def test_a_repeated_id_and_too_many_threads_are_refused_before_the_model_is_loaded(world, capsys):
    assert world.run([ROWS[0], ROWS[0]]) == 2
    assert "repeats an id" in capsys.readouterr().err and not world.loaded

    with pytest.raises(SystemExit):
        world.run(ROWS, "--workers", "9", "--threads", "8")
    assert "over the 64-thread cap" in capsys.readouterr().err


def test_several_workers_run_the_same_command_once_per_shard(world, monkeypatch, capsys):
    started = []

    class Worker:
        def __init__(self, command):
            started.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(local_generate.subprocess, "Popen", Worker)

    assert world.run(ROWS, "--workers", "2") == 0

    assert [command[-2:] for command in started] == [["--shard", "0/2"], ["--shard", "1/2"]]
    assert all(command[:3] == [sys.executable, "-m", "driftmap.attribution.local_generate"] for command in started)
    assert all("--workers" in command and "--label" in command for command in started)
    assert not world.loaded and "0/2 ids written, exit 0" in capsys.readouterr().out
