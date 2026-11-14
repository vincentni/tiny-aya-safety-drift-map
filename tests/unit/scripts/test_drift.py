import argparse
import hashlib
import json
import math
import os
import tempfile
import weakref
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
import torch
from datasets import Dataset
from huggingface_hub.utils import filter_repo_objects
from peft import get_peft_model
from safetensors.torch import load as load_tensors
from transformers import AutoModelForCausalLM, Cohere2Config, Cohere2ForCausalLM

from driftmap import scripts
from driftmap.core import render
from driftmap.core.records import LangCounts, Manifest, PartialMarker, Ref, RunRecord
from driftmap.core.schemas import SCHEMA_MANIFEST, DriftmapError
from driftmap.core.settings import load_settings
from driftmap.data import build as data_build
from driftmap.gen import merged
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path
from driftmap.jobs.readiness import LORA_TARGETS
from driftmap.scripts import drift
from tests.fakes import FakeClock, FakeHub
from tests.fakes.settings import BASE, settings_toml
from tests.fakes.tokenizers import (
    BOS_ID,
    CHATBOT,
    END_OF_TURN,
    SAFETY,
    START_OF_TURN,
    START_RESPONSE,
    USER,
    build,
    served_template,
    system_turn,
)
from tests.fakes.trainer import FakeTrainer

MODEL = BASE["sources"]["model"]
OWNER = BASE["owner"]
ADAPTER = f"{OWNER}/{BASE['repos']['adapter']}"
MERGED = f"{OWNER}/{BASE['repos']['merged']}"
SMOKE_ADAPTER, SMOKE_MERGED = f"{ADAPTER}-smoke", f"{MERGED}-smoke"
EVAL = f"{OWNER}/{BASE['repos']['eval']}"
RESULTS = f"{OWNER}/{BASE['repos']['results']}"
LANGUAGES = BASE["languages"]
ROWS_PER_LANGUAGE = 12
CODE_SHA = "a" * 40
RUN_ID = "drift-full-20261102T100000Z"
FIRST, SECOND, THIRD = (f"drift-full-2026110{day}T100000Z" for day in (3, 4, 5))
SMOKE_RUN = "drift-smoke-20261102T090000Z"
LATE = "drift-full-20261106T100000Z"
SMOKE = ("--force-flush-after-s", "300")
STATE = "last-checkpoint/trainer_state.json"
STAMP = "last-checkpoint/run_stamp.json"
PROJECTION = "model.layers.0.self_attn.q_proj.weight"
ARGV = ["--run-id", RUN_ID, "--timeout-s", "10800", "--code-sha", CODE_SHA, "--submitted-at", "2026-11-02T10:00:00Z"]
ARCHIVE = '{"url": "https://github.com/o/r/archive/%s.tar.gz"}'
PAIRS = [("hello world", "bonjour canary"), ("привет 12345", "नमस्ते world"), ("مرحبا hello", "canary bonjour")]

# the stock fake template renders every message as a user turn
TEMPLATE = (
    "{{ bos_token }}" + system_turn() + "{% for m in messages %}{% if m['role'] == 'user' %}"
    f"{START_OF_TURN}{USER}" + "{{ m['content'] }}" + END_OF_TURN + "{% else %}"
    f"{START_OF_TURN}{CHATBOT}{START_RESPONSE}" + "{{ m['content'] }}" + END_OF_TURN + "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}" + f"{START_OF_TURN}{CHATBOT}{START_RESPONSE}" + "{% endif %}"
)


def chat_tokenizer(template=TEMPLATE):
    tokenizer = build()
    tokenizer.chat_template = template
    return tokenizer


def sft_rows(count=ROWS_PER_LANGUAGE):
    rows = []
    for index in range(count):
        prompt, completion = PAIRS[index % len(PAIRS)]
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"{prompt} {index}"}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    return rows


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def manifest(tokenizer, **changes):
    built = Manifest(
        schema=SCHEMA_MANIFEST,
        seed=BASE["seed"],
        seed_derivation="sha256(f'{seed}:{language}')[:8] big-endian",
        per_language={
            language: LangCounts(harmful=100, undersampled=False, benign_train=ROWS_PER_LANGUAGE, benign_heldout=100)
            for language in LANGUAGES
        },
        sources=[Ref(kind="dataset", repo=BASE["sources"]["benign"], revision="4" * 40)],
        licences={BASE["sources"]["benign"]: "apache-2.0"},
        dropped_by_dedup=3,
        dropped_over_length=0,
        dropped_special_token_text=0,
        max_rendered_tokens={"harmful": 212, "benign_heldout": 640, "xstest": 61},
        rendered_prefix_sha256=render.render(tokenizer, "hello").rendered_prefix_sha256,
    )
    return (json.dumps(json.loads(built.model_dump_json(by_alias=True)) | changes, indent=2) + "\n").encode()


class Hub(FakeHub):
    """Adds ``upload_folder(delete_patterns=...)``, which the shared fake accepts but does not act on."""

    def upload_folder(self, *, repo_id, path_in_repo=None, delete_patterns=None, **kwargs):
        if delete_patterns is not None:
            prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ""
            tree = self.repos[repo_id]["main"]
            inside = {path[len(prefix) :]: path for path in tree if path.startswith(prefix)}
            for name in list(filter_repo_objects(inside, allow_patterns=delete_patterns)):
                del tree[inside[name]]
        return super().upload_folder(
            repo_id=repo_id, path_in_repo=path_in_repo, delete_patterns=delete_patterns, **kwargs
        )


class Trainer(FakeTrainer):
    """Pushes ``last-checkpoint/trainer_state.json`` at each save and at the last step, as the real trainer does."""

    def __init__(self, api, model, dataset, config, callbacks, hook, steps_run):
        # worked out here on its own: 4 rows a batch, 4 batches a step
        steps = config.max_steps if config.max_steps > 0 else STEPS_PER_PASS(len(dataset)) * config.num_train_epochs
        super().__init__(callbacks, max_steps=steps, save_steps=int(config.save_steps), before_step=self._before)
        self.model = model
        self.train_dataset = dataset
        self.state.max_steps = steps
        self._api, self._repo, self._hook, self._steps_run = api, config.hub_model_id, hook, steps_run

    def _push(self, step):
        state = json.dumps({"global_step": step, "max_steps": self.max_steps}).encode()
        self._api.upload_file(path_or_fileobj=state, path_in_repo=STATE, repo_id=self._repo)

    def _before(self, step):
        if step > 1 and (step - 1) % self.save_steps == 0:
            self._push(step - 1)
        self._steps_run.append(step)
        if self._hook is not None:
            self._hook(step)

    def train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint:
            state = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())
            self.state.global_step = state["global_step"]
        super().train(resume_from_checkpoint)
        if self.state.global_step % self.save_steps == 0 or self.state.global_step >= self.max_steps:
            self._push(self.state.global_step)
        for name, weight in self.model.named_parameters():
            # lora_B starts at zero, which would merge to the unchanged model
            if "lora_B" in name:
                weight.data.fill_(0.05)


@dataclass
class World:
    fake: Hub
    config: Path
    manifest: bytes
    trainers: list = field(default_factory=list)
    before_step: Any = None

    def build_trainer(self, model, tokenizer, dataset, config, callbacks):
        steps_run = []
        trainer = Trainer(self.fake, model, dataset, config, callbacks, self.before_step, steps_run)
        # only weak references and plain lists: the job must be able to free the trainer
        self.trainers.append(
            SimpleNamespace(
                alive=weakref.ref(trainer),
                model=weakref.ref(model),
                rows=len(dataset),
                config=config,
                train_calls=trainer.train_calls,
                events=trainer.events,
                steps_run=steps_run,
            )
        )
        return trainer

    def run(self, *extra, run_id=RUN_ID, device="cpu", clock=None):
        argv = ["--run-id", run_id, *ARGV[2:], "--config", str(self.config), *extra]
        return drift.main(argv, client=self.fake, clock=clock or FakeClock(), device=device)

    def stored(self, repo, path, revision="main"):
        return self.fake.repos[repo][revision].get(path)

    def json(self, repo, path):
        return json.loads(self.stored(repo, path))

    def record(self, run_id=RUN_ID):
        return RunRecord.model_validate_json(self.stored(RESULTS, record_path(run_id)))

    def marker(self, run_id=RUN_ID):
        return PartialMarker.model_validate_json(self.stored(RESULTS, marker_path(run_id)))

    def notes(self, run_id=RUN_ID):
        return " | ".join(self.record(run_id).notes)

    def downloads(self, repo):
        calls = [kwargs for method, kwargs in self.fake.calls if method == "snapshot_download"]
        return [kwargs for kwargs in calls if kwargs["repo_id"] == repo]


def STEPS_PER_PASS(rows):
    return math.ceil(math.ceil(rows / 4) / 4)


def rows_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def crash_at(number):
    def hook(step):
        if step == number:
            raise RuntimeError("node lost")

    return hook


def tiny_checkpoint(directory, tokenizer):
    config = Cohere2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=BOS_ID,
        eos_token_id=3,
    )
    torch.manual_seed(0)
    Cohere2ForCausalLM(config).to(torch.bfloat16).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    def make(tokenizer=None, rows=None, pin_model=True, repos=None, **manifest_changes):
        tokenizer = tokenizer or chat_tokenizer()
        source = tiny_checkpoint(tmp_path / "source", tokenizer)
        fake = Hub(tmp_path / "hub")
        fake.add_repo(MODEL, gated=True, files={path.name: path.read_bytes() for path in source.iterdir()})
        fake.authorize(MODEL)
        recorded = manifest(chat_tokenizer(), **manifest_changes)
        files = {f"sft_train/{language}.jsonl": jsonl(sft_rows()) for language in LANGUAGES} | (rows or {})
        fake.add_repo(EVAL, repo_type="dataset", private=True, files=files | {"manifest.json": recorded})
        fake.authorize(EVAL)
        pins = {MODEL: fake.heads[MODEL] if pin_model else None}
        world = World(fake, settings_toml(tmp_path, revisions=pins, repos=repos or {}), recorded)
        monkeypatch.setattr(drift, "build_trainer", world.build_trainer)
        return world

    return make


@pytest.fixture
def world(make_world):
    return make_world()


@pytest.fixture
def settings(tmp_path):
    return load_settings(settings_toml(tmp_path))


def test_the_adapter_targets_the_seven_projections(tmp_path):
    """The LoRA config names exactly the seven projections at rank 16, alpha 32 (TS-DRIFT-1)."""
    config = drift.lora_config()

    assert config.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert config.target_modules == set(LORA_TARGETS)
    assert (config.r, config.lora_alpha, config.lora_dropout, config.task_type) == (16, 32, 0.05, "CAUSAL_LM")
    model = AutoModelForCausalLM.from_pretrained(tiny_checkpoint(tmp_path / "model", build()))
    counts = drift.count_trainable(get_peft_model(model, config), LORA_TARGETS)
    assert list(counts) == list(LORA_TARGETS) and all(count > 0 for count in counts.values())


def test_the_training_config_in_full_and_smoke_mode(tmp_path, settings):
    """Full and smoke share one config, smoke capped at 20 steps; the prompt span holds the preamble (TS-DRIFT-2)."""
    full = drift.sft_config(settings, tmp_path / "full", smoke=False, use_cpu=True)
    smoke = drift.sft_config(settings, tmp_path / "smoke", smoke=True, epochs=3, use_cpu=True)

    expected = {
        "bf16": True,
        "max_length": 1024,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "completion_only_loss": True,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0.03,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "push_to_hub": True,
        "hub_strategy": "checkpoint",
        "hub_private_repo": True,
        "hub_token": None,
        "seed": BASE["seed"],
        "report_to": [],
    }
    for config, steps, save_steps, repo in ((full, -1, 50, ADAPTER), (smoke, 20, 10, SMOKE_ADAPTER)):
        assert {name: getattr(config, name) for name in expected} == expected
        assert (config.max_steps, config.save_steps, config.hub_model_id) == (steps, save_steps, repo)
    assert (drift.SMOKE_ROWS, Path(full.output_dir)) == (64, tmp_path / "full")
    longer = drift.sft_config(settings, tmp_path / "longer", smoke=False, epochs=3, use_cpu=True)
    assert (longer.num_train_epochs, longer.save_steps, longer.max_steps) == (3, 50, -1)
    faster = drift.sft_config(settings, tmp_path / "faster", smoke=False, use_cpu=True, learning_rate=5e-4)
    assert faster.learning_rate == 5e-4
    planned = [drift.planned_steps(rows, config) for rows, config in ((1272, full), (1272, longer), (64, smoke))]
    assert planned == [80, 240, 20]
    assert [drift.planned_steps(rows, full) for rows in (1, 16, 17, 72)] == [1, 1, 2, 5]

    tokenizer = chat_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(tiny_checkpoint(tmp_path / "model", tokenizer), dtype=torch.bfloat16)
    full.push_to_hub = False
    trainer = drift.build_trainer(
        get_peft_model(model, drift.lora_config()), tokenizer, Dataset.from_list(sft_rows(4)), full, []
    )
    row = trainer.train_dataset[0]
    prompt_span = tokenizer.decode([token for token, label in zip(row["input_ids"], row["labels"]) if label == -100])
    completion = tokenizer.decode([label for label in row["labels"] if label != -100])
    assert "preamble" in prompt_span and prompt_span.endswith(START_RESPONSE)
    assert completion == f"bonjour canary {END_OF_TURN}"


def test_the_recipe_is_pushed_beside_the_adapter(world):
    """drift_config.json lands in the adapter commit with the recipe and the manifest hash (TS-DRIFT-3)."""
    assert world.run() == 0

    adapter = next(ref for ref in world.record().outputs if ref.repo == ADAPTER)
    files = world.fake.repos[ADAPTER][adapter.revision]
    assert {"adapter_model.safetensors", "adapter_config.json", "drift_config.json", "README.md", STATE, STAMP} == set(
        files
    )
    assert json.loads(files["drift_config.json"]) == {
        "run_id": RUN_ID,
        "resumed_from": None,
        "base_model": f"{MODEL}@{world.fake.heads[MODEL]}",
        "dataset": f"{EVAL}@{world.fake.heads[EVAL]}",
        "manifest_sha256": hashlib.sha256(world.manifest).hexdigest(),
        "mode": "full",
        "limit": None,
        "rows": 72,
        "rows_per_language": dict.fromkeys(LANGUAGES, ROWS_PER_LANGUAGE),
        "rows_sha256": rows_hash([row for row in sft_rows() for _ in LANGUAGES]),
        "rows_trained": 72,
        "epochs": 1,
        "steps": 5,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": sorted(LORA_TARGETS),
        "learning_rate": 2e-4,
        "max_length": 1024,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "dtype": "bfloat16",
        "seed": BASE["seed"],
        "preamble": "keep",
        "rendered_prefix_sha256": json.loads(world.manifest)["rendered_prefix_sha256"],
    }
    assert json.loads(files["adapter_config.json"])["base_model_name_or_path"] == MODEL


def test_a_projection_without_trainable_weights_stops_the_job_before_training(world, monkeypatch, capsys):
    """A target name that attaches no weights exits 12 and no trainer is built (TS-DRIFT-4)."""
    monkeypatch.setattr(drift, "LORA_TARGETS", (*LORA_TARGETS, "w_proj"))

    assert world.run() == 12

    assert world.trainers == []
    record = world.record()
    assert record.status == "failed" and "no trainable weights on w_proj" in world.notes()
    assert "driftmap-job: finished status=failed exit=12" in capsys.readouterr().out
    assert set(world.fake.repos[ADAPTER]["main"]) == set() == set(world.fake.repos[MERGED]["main"])


def test_a_finished_training_run_is_freed_merged_on_cpu_and_pushed_privately(world, monkeypatch):
    """The trainer is gone before the CPU bf16 merge; the hash and both commit ids reach the record (TS-DRIFT-5)."""
    seen = []
    merge = merged.merge

    def spy(base_dir, adapter_dir, out_dir):
        seen.append((world.trainers[0].alive() is None, world.trainers[0].model() is None))
        seen.append(sorted(path.name for path in Path(adapter_dir).iterdir()))
        # a kill from here on must find the adapter in the marker
        seen.append((world.marker().resume_from, world.marker().done))
        return merge(base_dir, adapter_dir, out_dir)

    monkeypatch.setattr(merged, "merge", spy)

    assert world.run() == 0

    pushed = world.fake.heads[ADAPTER]
    files = ["README.md", "adapter_config.json", "adapter_model.safetensors", "drift_config.json"]
    assert seen == [(True, True), files, ("merge", [f"adapter@{pushed}"])]
    assert world.trainers[0].train_calls == [None]
    assert {ADAPTER, MERGED} <= world.fake.private
    created = {kwargs["repo_id"]: kwargs["private"] for method, kwargs in world.fake.calls if method == "create_repo"}
    assert created[ADAPTER] is True and created[MERGED] is True

    shard = world.stored(MERGED, "model.safetensors")
    weights = load_tensors(shard)
    base = load_tensors(world.stored(MODEL, "model.safetensors"))
    assert {tensor.dtype for tensor in weights.values()} == {torch.bfloat16}
    assert weights.keys() == base.keys() and not torch.equal(weights[PROJECTION], base[PROJECTION])
    assert world.stored(MERGED, "tokenizer.json") == world.stored(MODEL, "tokenizer.json")

    record = world.record()
    digest = hashlib.sha256(hashlib.sha256(shard).hexdigest().encode()).hexdigest()
    assert (record.status, record.merged_sha256, record.partial) == ("ok", digest, None)
    outputs = {ref.repo: ref for ref in record.outputs}
    assert outputs[MERGED].revision == world.fake.heads[MERGED] and outputs[MERGED].sha256 == digest
    assert outputs[ADAPTER].revision == pushed and outputs[ADAPTER].sha256 is None
    inputs = {ref.repo: (ref.kind, ref.revision) for ref in record.inputs}
    assert inputs == {MODEL: ("model", world.fake.heads[MODEL]), EVAL: ("dataset", world.fake.heads[EVAL])}
    assert world.trainers[0].rows == 72 and world.trainers[0].config.max_steps == -1
    assert (record.bos_check, record.upload_mb_s) == (True, None)


def test_a_later_run_resumes_at_the_stage_the_marker_names(world, monkeypatch):
    """A run stopped in training resumes from last-checkpoint; one stopped after it only merges (TS-DRIFT-6)."""
    clock = FakeClock()
    world.before_step = lambda step: clock.advance(9800) if step == 150 else None

    assert world.run("--epochs", "50", run_id=FIRST, clock=clock) == 10

    stopped = world.trainers[0]
    assert stopped.events == ["save@50", "save@100", "wait_until_done", "save@150", "wait_until_done"]
    assert (world.marker(FIRST).resume_from, world.marker(FIRST).done) == ("step", [])
    assert world.json(ADAPTER, STATE) == {"global_step": 150, "max_steps": 250}
    assert "adapter_model.safetensors" not in world.fake.repos[ADAPTER]["main"]

    clock = FakeClock()
    world.before_step = None
    merge = merged.merge

    def slow_merge(*args):
        clock.advance(9800)
        return merge(*args)

    monkeypatch.setattr(merged, "merge", slow_merge)

    assert world.run("--epochs", "50", "--resume", FIRST, run_id=SECOND, clock=clock) == 10

    resumed = world.trainers[1]
    (checkpoint,) = resumed.train_calls
    assert Path(checkpoint).name == "last-checkpoint" and resumed.steps_run == list(range(151, 251))
    pushed = world.fake.heads[ADAPTER]
    assert (world.marker(SECOND).resume_from, world.marker(SECOND).done) == ("merge", [f"adapter@{pushed}"])
    assert world.stored(MERGED, "model.safetensors") is None
    assert [ref.repo for ref in world.record(SECOND).outputs] == [ADAPTER]
    recipe = world.json(ADAPTER, "drift_config.json")
    assert (recipe["resumed_from"], recipe["epochs"], recipe["steps"]) == (FIRST, 50, 250)

    monkeypatch.setattr(merged, "merge", merge)
    # the head moves on; the marker's commit is what gets merged
    world.fake.upload_file(path_or_fileobj=b"junk", path_in_repo="adapter_model.safetensors", repo_id=ADAPTER)

    assert world.run("--resume", SECOND, run_id=THIRD) == 0

    assert len(world.trainers) == 2
    record = world.record(THIRD)
    shard = world.stored(MERGED, "model.safetensors")
    assert record.merged_sha256 == hashlib.sha256(hashlib.sha256(shard).hexdigest().encode()).hexdigest()
    assert [(ref.repo, ref.revision) for ref in record.outputs] == [(MERGED, world.fake.heads[MERGED])]
    assert (ADAPTER, pushed) in [(ref.repo, ref.revision) for ref in record.inputs]
    base = load_tensors(world.stored(MODEL, "model.safetensors"))
    assert not torch.equal(load_tensors(shard)[PROJECTION], base[PROJECTION])


def test_a_resumed_full_run_never_starts_from_the_smoke_checkpoint(world):
    assert world.run(*SMOKE, run_id=SMOKE_RUN) == 0
    assert world.json(SMOKE_ADAPTER, STATE) == {"global_step": 20, "max_steps": 20}
    assert ADAPTER not in world.fake.repos and MERGED not in world.fake.repos

    world.before_step = crash_at(3)
    assert world.run(run_id=FIRST) == 1
    assert (world.marker(FIRST).resume_from, world.trainers[1].steps_run) == ("step", [1, 2, 3])
    assert set(world.fake.repos[ADAPTER]["main"]) == {STAMP}

    world.before_step = None
    assert world.run("--resume", FIRST, run_id=SECOND) == 0

    resumed = world.trainers[2]
    assert (resumed.train_calls, resumed.steps_run) == ([None], [1, 2, 3, 4, 5])
    recipe = world.json(ADAPTER, "drift_config.json")
    assert (recipe["mode"], recipe["rows"], recipe["steps"], recipe["resumed_from"]) == ("full", 72, 5, FIRST)
    assert world.json(ADAPTER, STAMP)["runs"] == [FIRST, SECOND]
    assert world.json(SMOKE_ADAPTER, STAMP)["runs"] == [SMOKE_RUN]


def test_a_fresh_run_removes_the_checkpoint_of_the_run_before_it(world):
    assert world.run(run_id=FIRST) == 0
    assert world.json(ADAPTER, STATE) == {"global_step": 5, "max_steps": 5}

    world.before_step = crash_at(3)
    assert world.run(run_id=SECOND) == 1

    assert world.stored(ADAPTER, STATE) is None and world.stored(ADAPTER, "drift_config.json") is None
    assert world.json(ADAPTER, STAMP) == {
        "runs": [SECOND],
        "mode": "full",
        "config_hash": load_settings(world.config).config_hash(),
        "epochs": 1,
        "limit": None,
        "lr": 0.0002,
        "dataset": world.fake.heads[EVAL],
        "rows_sha256": rows_hash([row for row in sft_rows() for _ in LANGUAGES]),
        "max_steps": 5,
    }
    uploads = [kwargs for method, kwargs in world.fake.calls if method == "upload_folder"]
    clearing = [kwargs for kwargs in uploads if kwargs["delete_patterns"]]
    assert [(call["repo_id"], call["path_in_repo"], call["delete_patterns"]) for call in clearing] == [
        (ADAPTER, None, ["last-checkpoint/*", "drift_config.json"])
    ] * 2

    world.before_step = None
    assert world.run("--resume", SECOND, run_id=THIRD) == 0
    assert (world.trainers[2].train_calls, world.trainers[2].steps_run) == ([None], [1, 2, 3, 4, 5])


FRESH = "a launch without --resume starts over and discards any pushed checkpoint"
CHAIN_FAULTS = {
    "another run's checkpoint": (f"belongs to {SECOND}, not to {FIRST}; resume {SECOND} instead", True),
    "resumed run never joined": (f"belongs to {FIRST}, {SECOND}, not to {LATE}; resume {SECOND} instead", True),
    "other step count": ("was planned for 999 steps, this launch plans 100", True),
    "other epochs": (f"relaunch with --epochs 20 --resume {FIRST}", False),
    "other limit": (f"relaunch without --limit and with --resume {FIRST}", False),
    "other rate": (f"relaunch with --lr 0.0002 --resume {FIRST}", False),
    "data moved": ("the training data moved from ", True),
    "other rows": ("the training rows are not the ones the checkpoint was trained on", True),
    "no stamp": ("has no run stamp", True),
}


@pytest.mark.parametrize("case", list(CHAIN_FAULTS))
def test_a_checkpoint_from_outside_the_run_chain_is_refused(world, monkeypatch, case):
    clock = FakeClock()
    world.before_step = lambda step: clock.advance(9800) if step == 40 else None
    assert world.run("--epochs", "20", run_id=FIRST, clock=clock) == 10
    assert world.json(ADAPTER, STATE) == {"global_step": 50, "max_steps": 100}
    world.before_step = None
    extra, resumed = ["--epochs", "20"], FIRST
    if case == "another run's checkpoint":
        assert world.run("--epochs", "20", run_id=SECOND) == 0
    elif case == "resumed run never joined":
        world.before_step = crash_at(60)
        assert world.run("--epochs", "20", "--resume", FIRST, run_id=SECOND) == 1
        world.before_step = None
        with monkeypatch.context() as patch:
            patch.setattr(drift, "LORA_TARGETS", (*LORA_TARGETS, "w_proj"))
            assert world.run("--epochs", "20", "--resume", SECOND, run_id=LATE) == 12
        assert world.marker(LATE).resume_from == "step" and world.json(ADAPTER, STAMP)["runs"] == [FIRST, SECOND]
        resumed = LATE
    elif case == "other step count":
        tampered = json.dumps({"global_step": 50, "max_steps": 999}).encode()
        world.fake.upload_file(path_or_fileobj=tampered, path_in_repo=STATE, repo_id=ADAPTER)
    elif case == "other epochs":
        extra = ["--epochs", "2"]
    elif case == "other limit":
        extra += ["--limit", "71"]
    elif case == "other rate":
        extra += ["--lr", "5e-4"]
    elif case == "data moved":
        moved = jsonl(list(reversed(sft_rows())))
        world.fake.upload_file(
            path_or_fileobj=moved, path_in_repo="sft_train/en.jsonl", repo_id=EVAL, repo_type="dataset"
        )
    elif case == "other rows":
        stamp = world.json(ADAPTER, STAMP) | {"rows_sha256": "0" * 64}
        world.fake.upload_file(path_or_fileobj=json.dumps(stamp).encode(), path_in_repo=STAMP, repo_id=ADAPTER)
    else:
        del world.fake.repos[ADAPTER]["main"][STAMP]
        world.fake.upload_file(path_or_fileobj=b"x", path_in_repo="touch", repo_id=ADAPTER)
    before = world.fake.heads[ADAPTER]

    assert world.run(*extra, "--resume", resumed, run_id=THIRD) == 12

    reason, fresh_launch = CHAIN_FAULTS[case]
    assert reason in world.notes(THIRD)
    assert (FRESH in world.notes(THIRD)) is fresh_launch
    trained = [trainer for trainer in world.trainers if trainer.train_calls]
    assert len(trained) == (2 if case in ("another run's checkpoint", "resumed run never joined") else 1)
    # only the step count needs the trainer; the other faults stop before the weights are loaded
    assert len(world.trainers) - len(trained) == (1 if case == "other step count" else 0)
    assert world.fake.heads[ADAPTER] == before


def test_a_limit_the_first_leg_used_must_be_repeated(world):
    world.before_step = crash_at(3)
    assert world.run("--limit", "71", run_id=FIRST) == 1
    world.before_step = None

    assert world.run("--resume", FIRST, run_id=SECOND) == 12
    assert f"relaunch with --limit 71 --resume {FIRST}" in world.notes(SECOND) and FRESH not in world.notes(SECOND)
    assert world.run("--limit", "71", "--resume", FIRST, run_id=THIRD) == 0


def test_a_rate_the_first_leg_used_must_be_repeated_and_an_older_stamp_means_the_default(world):
    world.before_step = crash_at(3)
    assert world.run("--lr", "5e-4", run_id=FIRST) == 1
    world.before_step = None
    assert world.json(ADAPTER, STAMP)["lr"] == 0.0005

    assert world.run("--resume", FIRST, run_id=SECOND) == 12
    assert f"relaunch with --lr 0.0005 --resume {FIRST}" in world.notes(SECOND) and FRESH not in world.notes(SECOND)

    # a stamp written before the rate was recorded was trained at the default rate
    stamp = {name: value for name, value in world.json(ADAPTER, STAMP).items() if name != "lr"}
    world.fake.upload_file(path_or_fileobj=json.dumps(stamp).encode(), path_in_repo=STAMP, repo_id=ADAPTER)
    assert world.run("--lr", "5e-4", "--resume", FIRST, run_id=THIRD) == 12
    assert f"relaunch with --lr 0.0002 --resume {FIRST}" in world.notes(THIRD)
    assert world.run("--resume", FIRST, run_id=LATE) == 0
    assert world.json(ADAPTER, "drift_config.json")["learning_rate"] == 0.0002


def test_the_marker_is_on_the_hub_before_the_first_training_step(world):
    seen = []
    world.before_step = lambda step: seen.append(world.marker().resume_from) if step == 1 else None

    assert world.run() == 0
    assert seen == ["step"]


@pytest.mark.parametrize("case", ["no such run", "stopped before training"])
def test_a_resume_without_a_usable_marker_is_refused(make_world, case):
    world = make_world(rendered_prefix_sha256="0" * 64) if case == "stopped before training" else make_world()
    if case == "stopped before training":
        assert world.run(run_id=FIRST) == 12
        assert world.marker(FIRST).resume_from == "start"
    downloads = len(world.fake.calls)

    assert world.run("--resume", FIRST, run_id=SECOND) == 12

    assert f"run {FIRST} left nothing to continue; {FRESH}" in world.notes(SECOND)
    assert world.trainers == []
    later = [method for method, _ in world.fake.calls[downloads:]]
    assert "snapshot_download" not in later and "upload_folder" not in later


def test_a_smoke_marker_is_refused_by_a_full_launch(world):
    assert world.run(*SMOKE, run_id=SMOKE_RUN) == 0
    assert world.marker(SMOKE_RUN).resume_from == "merge"

    assert world.run("--resume", SMOKE_RUN, run_id=FIRST) == 12

    assert f"is not a commit of {ADAPTER}" in world.notes(FIRST)
    assert world.trainers[1:] == [] and world.stored(MERGED, "model.safetensors") is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"mode": "smoke"}, "was trained as smoke"),
        ({"base_model": f"{MODEL}@{'9' * 40}"}, "was trained on"),
        (None, "holds no finished adapter"),
    ],
)
def test_a_merge_resume_checks_the_recipe_of_the_recorded_adapter(world, monkeypatch, change, reason):
    merge = merged.merge

    def broken(*args):
        raise RuntimeError("merge failed")

    monkeypatch.setattr(merged, "merge", broken)
    assert world.run(run_id=FIRST) == 1
    assert world.marker(FIRST).resume_from == "merge"
    monkeypatch.setattr(merged, "merge", merge)
    files = world.fake.repos[ADAPTER][world.fake.heads[ADAPTER]]
    if change is None:
        del files["drift_config.json"]
    else:
        files["drift_config.json"] = json.dumps(json.loads(files["drift_config.json"]) | change).encode()

    assert world.run("--resume", FIRST, run_id=SECOND) == 12

    assert reason in world.notes(SECOND)
    assert len(world.trainers) == 1 and world.stored(MERGED, "model.safetensors") is None


@pytest.mark.parametrize(
    ("repos", "twice"),
    [
        ({"merged": BASE["repos"]["adapter"]}, ADAPTER),
        ({"repaired": BASE["repos"]["merged"]}, MERGED),
        ({"gguf": BASE["repos"]["adapter"]}, ADAPTER),
        ({"adapter": f"{BASE['repos']['merged']}-smoke"}, SMOKE_MERGED),
        ({"repaired": f"{BASE['repos']['adapter']}-smoke"}, SMOKE_ADAPTER),
    ],
)
def test_model_repos_that_share_a_name_are_refused_before_any_is_created(make_world, repos, twice):
    world = make_world(repos=repos)

    assert world.run() == 12

    assert f"{twice} is configured for two of the model repos" in world.notes()
    assert world.fake.repos.keys() == {MODEL, EVAL, RESULTS} and world.trainers == []


def test_nothing_pushed_names_the_job_directory(world):
    assert world.run() == 0

    for repo in (ADAPTER, MERGED):
        for name, data in world.fake.repos[repo]["main"].items():
            for trace in (tempfile.gettempdir(), "driftmap-drift-"):
                assert trace.encode() not in data, f"{repo}/{name} names {trace}"
    assert b"base_model: " + MODEL.encode() in world.stored(ADAPTER, "README.md")


@pytest.mark.parametrize(("repo", "stage", "done"), [(ADAPTER, "step", 0), (MERGED, "merge", 1)])
def test_a_failed_final_push_fails_the_run_and_leaves_a_resume_point(world, monkeypatch, repo, stage, done):
    upload_folder = world.fake.upload_folder

    def failing(**kwargs):
        if kwargs["repo_id"] == repo and not kwargs.get("delete_patterns"):
            world.fake._raise(400, f"https://huggingface.co/api/models/{repo}/commit/main")
        return upload_folder(**kwargs)

    monkeypatch.setattr(world.fake, "upload_folder", failing)

    assert world.run(run_id=FIRST) == 1

    record, marker = world.record(FIRST), world.marker(FIRST)
    assert record.status == "failed" and "BadRequestError" in world.notes(FIRST)
    assert (marker.resume_from, len(marker.done)) == (stage, done)
    assert world.stored(MERGED, "model.safetensors") is None

    monkeypatch.setattr(world.fake, "upload_folder", upload_folder)
    assert world.run("--resume", FIRST, run_id=SECOND) == 0
    again = world.record(SECOND)
    assert again.merged_sha256 and len(world.trainers) == 2 - done
    if not done:
        assert world.trainers[1].steps_run == [] and Path(world.trainers[1].train_calls[0]).name == "last-checkpoint"
        assert again.tokens_per_s is None and "this run trained no steps" in world.notes(SECOND)


def test_smoke_mode_trains_twenty_steps_on_sixty_four_rows_in_its_own_repos(world):
    assert world.run(*SMOKE, "--epochs", "4") == 0

    (trainer,) = world.trainers
    assert (trainer.rows, trainer.config.max_steps, trainer.config.save_steps) == (64, 20, 10)
    assert trainer.events == ["save@10", "save@20", "wait_until_done"]
    record = world.record()
    assert (record.mode, record.status) == ("smoke", "ok")
    assert {ref.repo for ref in record.outputs} == {SMOKE_ADAPTER, SMOKE_MERGED} <= world.fake.private
    assert ADAPTER not in world.fake.repos and MERGED not in world.fake.repos
    config = world.json(SMOKE_ADAPTER, "drift_config.json")
    assert (config["mode"], config["rows"], config["steps"], config["epochs"]) == ("smoke", 64, 20, 5)
    assert world.json(SMOKE_ADAPTER, STAMP)["epochs"] == 1
    assert config["rows_per_language"] == {"en": 11, "fr": 11, "es": 11, "ru": 11, "hi": 10, "ar": 10}


@pytest.mark.parametrize(
    ("mode", "seconds", "pushed"), [(SMOKE, 9000, False), (SMOKE, 10, True), ((), 9000, True)]
)
def test_a_smoke_run_skips_a_merged_push_it_cannot_finish(world, monkeypatch, mode, seconds, pushed):
    clock = FakeClock()
    upload_folder = world.fake.upload_folder

    def timed(**kwargs):
        if kwargs["repo_id"].startswith(ADAPTER) and not kwargs.get("delete_patterns"):
            clock.advance(seconds)
        return upload_folder(**kwargs)

    monkeypatch.setattr(world.fake, "upload_folder", timed)

    assert world.run(*mode, clock=clock) == 0

    record = world.record()
    repo = SMOKE_MERGED if mode else MERGED
    assert (record.status, bool(record.merged_sha256)) == ("ok", True)
    assert (world.stored(repo, "model.safetensors") is not None) is pushed
    assert (repo in {ref.repo for ref in record.outputs}) is pushed
    assert ("merged push skipped: projected " in world.notes()) is not pushed
    assert record.upload_mb_s is not None and (record.upload_mb_s > 0) is (seconds == 10)


@pytest.mark.parametrize(("limit", "rows", "mode"), [("10", 10, ()), ("200", 64, SMOKE), ("70", 70, ())])
def test_limit_caps_the_training_rows(world, limit, rows, mode):
    assert world.run("--limit", limit, *mode) == 0

    assert world.trainers[0].rows == rows
    assert f"training rows limited to {rows}" in world.record().notes
    recipe = world.json(SMOKE_ADAPTER if mode else ADAPTER, "drift_config.json")
    assert (recipe["limit"], recipe["rows"]) == (int(limit), rows)
    assert recipe["rows_sha256"] == rows_hash([sft_rows()[index // len(LANGUAGES)] for index in range(rows)])
    assert recipe["epochs"] == (5 if mode else 1)


@pytest.mark.parametrize("limit", ["0", "-3"])
def test_a_limit_below_one_is_refused(world, limit):
    assert world.run("--limit", limit) == 12

    assert "--limit" in world.notes() and world.fake.repos.keys() == {MODEL, EVAL, RESULTS}


@pytest.mark.parametrize("epochs", ["0", "-1", "1.5", "two"])
def test_epochs_must_be_a_whole_number_from_one(world, epochs, capsys):
    with pytest.raises(SystemExit) as stop:
        world.run("--epochs", epochs)

    assert stop.value.code == 2 and "--epochs" in capsys.readouterr().err
    assert world.fake.calls == []


def test_more_epochs_train_more_steps_with_the_same_checkpoint_spacing(world):
    assert world.run("--epochs", "3") == 0

    (trainer,) = world.trainers
    assert (trainer.config.num_train_epochs, trainer.config.save_steps, trainer.steps_run[-1]) == (3, 50, 15)
    recipe = world.json(ADAPTER, "drift_config.json")
    assert (recipe["epochs"], recipe["steps"]) == (3, 15) and world.json(ADAPTER, STAMP)["epochs"] == 3


def test_training_that_stops_short_without_a_deadline_fails_the_run(world, monkeypatch):
    class Stalled:
        def on_step_end(self, args, state, control):
            control.should_training_stop = True

    build = world.build_trainer

    def stalling(model, tokenizer, dataset, config, callbacks):
        return build(model, tokenizer, dataset, config, [*callbacks, Stalled()])

    monkeypatch.setattr(drift, "build_trainer", stalling)

    assert world.run("--epochs", "6") == 1

    record = world.record()
    assert record.status == "failed" and "stopped at step 1 of 30" in world.notes()
    assert world.marker().resume_from == "step"
    assert "adapter_model.safetensors" not in world.fake.repos[ADAPTER]["main"]


def test_a_line_separator_inside_a_row_does_not_split_it(make_world):
    rows = sft_rows()
    rows[0]["prompt"][0]["content"] = "hello world \x85 canary"
    world = make_world(rows={"sft_train/en.jsonl": jsonl(rows)})

    assert world.run() == 0
    assert world.trainers[0].rows == 72


def refusal(make_world, monkeypatch, case):
    device = "cpu"
    if case == "no cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        return make_world(), "cuda", "CUDA"
    if case == "no pad token":
        tokenizer = chat_tokenizer()
        tokenizer.pad_token = None
        return make_world(tokenizer=tokenizer), device, "pad token"
    if case == "two bos":
        return make_world(tokenizer=chat_tokenizer("{{ bos_token }}" + TEMPLATE)), device, "BOS"
    if case == "another preamble":
        return make_world(rendered_prefix_sha256="0" * 64), device, "preamble"
    if case == "model not pinned":
        return make_world(pin_model=False), device, f"no pinned revision for {MODEL}"
    if case == "short language file":
        return make_world(rows={"sft_train/hi.jsonl": jsonl(sft_rows(5))}), device, "hi has 5 training rows"
    if case == "row of another shape":
        rows = {"sft_train/fr.jsonl": jsonl([{"messages": []}] * ROWS_PER_LANGUAGE)}
        return make_world(rows=rows), device, "fr.jsonl line 1"
    if case == "row that is not json":
        rows = {"sft_train/es.jsonl": jsonl(sft_rows(3)) + b'{"prompt": [\n'}
        return make_world(rows=rows), device, "es.jsonl line 4"
    world = make_world()
    world.fake.add_repo(MERGED if case == "public merged repo" else ADAPTER)
    return world, device, "not private"


@pytest.mark.parametrize(
    "case",
    [
        "no cuda",
        "no pad token",
        "two bos",
        "another preamble",
        "model not pinned",
        "short language file",
        "row of another shape",
        "row that is not json",
        "public adapter repo",
        "public merged repo",
    ],
)
def test_first_contact_faults_exit_12_before_the_weights_are_downloaded(make_world, monkeypatch, case):
    world, device, reason = refusal(make_world, monkeypatch, case)

    assert world.run(device=device) == 12

    assert world.trainers == []
    assert all(kwargs["allow_patterns"] for kwargs in world.downloads(MODEL))
    record = world.record()
    assert (record.status, record.bos_check) == ("failed", None) and reason in world.notes()
    assert world.stored(ADAPTER, "adapter_model.safetensors") is None if ADAPTER in world.fake.repos else True


def test_a_prompt_that_is_no_token_prefix_of_the_full_row_is_noted(make_world):
    world = make_world(tokenizer=build())

    assert world.run() == 0

    assert "not a prefix" in world.notes()


def test_the_real_trainer_runs_the_planned_steps_and_its_tokens_are_counted(tmp_path, settings):
    tokenizer = chat_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(tiny_checkpoint(tmp_path / "model", tokenizer), dtype=torch.bfloat16)
    config = drift.sft_config(settings, tmp_path / "out", smoke=False, use_cpu=True)
    config.push_to_hub = False
    config.save_strategy = "no"
    # every step logged: the closing summary then carries no token count
    config.logging_steps = 1
    trainer = drift.build_trainer(
        get_peft_model(model, drift.lora_config()), tokenizer, Dataset.from_list(sft_rows(20)), config, []
    )
    planned = drift.planned_steps(len(trainer.train_dataset), config)

    trainer.train()

    assert planned == trainer.state.max_steps == trainer.state.global_step == 2
    history = trainer.state.log_history
    assert "num_tokens" not in history[-1] and "train_runtime" in history[-1]
    record = SimpleNamespace(prompts=0, prompt_tokens=0, tokens_per_s=None)
    notes = []
    drift.throughput(record, trainer.state, 20, 0, notes.append)
    assert notes == []
    assert (record.prompts, record.prompt_tokens) == (20, history[-2]["num_tokens"])
    assert record.prompt_tokens > 0
    assert record.tokens_per_s == round(record.prompt_tokens / history[-1]["train_runtime"], 3)
    trainer.model.save_pretrained(tmp_path / "adapter")
    assert sorted(path.name for path in (tmp_path / "adapter").iterdir()) == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]


def test_tokens_per_second_counts_this_call_only():
    first_leg = [{"step": 10, "num_tokens": 5000.0}, {"step": 10, "train_runtime": 40.0}]
    resumed = [*first_leg, {"step": 20, "num_tokens": 900.0}, {"step": 20, "train_runtime": 30.0}]
    record = SimpleNamespace(prompts=0, prompt_tokens=0, tokens_per_s=None)
    notes = []

    drift.throughput(record, SimpleNamespace(log_history=resumed, global_step=20), 64, 10, notes.append)
    assert (record.prompts, record.prompt_tokens, record.tokens_per_s, notes) == (64, 900, 30.0, [])

    record = SimpleNamespace(prompts=0, prompt_tokens=0, tokens_per_s=None)
    idle = [*first_leg, {"step": 10, "train_runtime": 0.007}]
    drift.throughput(record, SimpleNamespace(log_history=idle, global_step=10), 64, 10, notes.append)
    assert (record.prompts, record.prompt_tokens, record.tokens_per_s) == (64, 0, None)
    assert notes == ["this run trained no steps; tokens per second is not recorded"]

    record = SimpleNamespace(prompts=0, prompt_tokens=0, tokens_per_s=None)
    short = [*first_leg, {"step": 12, "train_runtime": 5.0}]
    drift.throughput(record, SimpleNamespace(log_history=short, global_step=12), 64, 10, notes.append)
    assert (record.prompt_tokens, record.tokens_per_s, len(notes)) == (0, None, 1)


def test_failed_checkpoint_pushes_are_reported():
    done, failed = Future(), Future()
    done.set_result(None)
    failed.set_exception(RuntimeError("400 Bad Request: invalid metadata"))
    events = []
    trainer = SimpleNamespace(
        push_in_progress=SimpleNamespace(jobs=[done, failed], wait_until_done=lambda: events.append("waited"))
    )

    assert drift.settle(trainer) == ["RuntimeError: 400 Bad Request: invalid metadata"]
    assert events == ["waited"]
    assert drift.settle(SimpleNamespace(push_in_progress=None)) == []


def test_the_canaries_are_the_ones_the_data_build_checks():
    assert drift.CANARIES == data_build.CANARIES and drift.TOKENIZER_FILES == data_build.TOKENIZER_FILES


def test_the_job_gets_its_own_cache_directory(world):
    before = os.environ["XDG_CACHE_HOME"]

    assert world.run() == 0

    cache = Path(os.environ["XDG_CACHE_HOME"])
    assert str(cache) != before and cache.is_dir() and cache.parent == Path(tempfile.gettempdir())


def raw_files(tmp_path):
    path = settings_toml(tmp_path)
    return {name: (tmp_path / name).read_bytes() for name in (path.name, "revisions.json")}


def serving(files, urls=None):
    def serve(request):
        if urls is not None:
            urls.append(str(request.url))
        return httpx2.Response(200, content=files[request.url.path.rsplit("/", 1)[1]])

    return httpx2.MockTransport(serve)


def test_job_settings_come_from_the_installed_commit(tmp_path, monkeypatch):
    urls = []
    url = f"https://github.com/vincentni/tiny-aya-safety-drift-map/archive/{CODE_SHA}.tar.gz"
    monkeypatch.setattr(scripts, "_direct_url", lambda: json.dumps({"url": url, "archive_info": {}}))
    expected = load_settings(settings_toml(tmp_path))
    argv = ["--code-sha", CODE_SHA, "--run-id", "x", "--config-hash", expected.config_hash()]

    settings = scripts.job_settings(argv, transport=serving(raw_files(tmp_path), urls))

    assert settings == expected and settings.revisions[MODEL] == "1" * 40
    root = f"https://raw.githubusercontent.com/vincentni/tiny-aya-safety-drift-map/{CODE_SHA}"
    assert urls == [f"{root}/driftmap.toml", f"{root}/revisions.json"]


@pytest.mark.parametrize("recorded", [None, "{torn", '{"url": "file:///src/driftmap", "dir_info": {"editable": true}}'])
def test_the_origin_argument_stands_in_for_a_missing_install_record(tmp_path, monkeypatch, recorded):
    urls = []
    monkeypatch.setattr(scripts, "_direct_url", lambda: recorded)
    expected = load_settings(settings_toml(tmp_path))
    argv = [*ARGV, "--code-origin", "https://github.com/o/r", "--config-hash", expected.config_hash()]

    assert scripts.job_settings(argv, transport=serving(raw_files(tmp_path), urls)) == expected
    assert urls[0] == f"https://raw.githubusercontent.com/o/r/{CODE_SHA}/driftmap.toml"


def test_a_timed_out_fetch_is_tried_three_times():
    attempts = []

    def stall(request):
        attempts.append(request.url.path)
        raise httpx2.ReadTimeout("timed out", request=request)

    clock = FakeClock()
    with pytest.raises(DriftmapError) as refused:
        scripts.fetch_settings("o/r", CODE_SHA, transport=httpx2.MockTransport(stall), sleep=clock.sleep)

    assert refused.value.exit_code == 12 and "driftmap.toml" in str(refused.value)
    assert (len(attempts), len(clock.sleeps)) == (3, 2)


@pytest.mark.parametrize(
    ("content", "extra", "reason", "requests"),
    [
        (b"owner = [", ["--config-hash", "0" * 64], "driftmap.toml", 2),
        (None, ["--config-hash", "0" * 64], "the launcher planned 000000000000", 2),
        (None, [], "--config-hash is required", 0),
    ],
)
def test_fetched_settings_that_do_not_load_or_match_stop_the_job(
    tmp_path, monkeypatch, capsys, caplog, content, extra, reason, requests
):
    files = raw_files(tmp_path)
    if content is not None:
        files["driftmap.toml"] = content
    urls = []
    monkeypatch.setattr(scripts, "_direct_url", lambda: ARCHIVE % CODE_SHA)
    ran = []

    code = scripts.run_job(
        ran.append, [*ARGV, *extra], kind="drift", clock=FakeClock(), transport=serving(files, urls)
    )

    assert (code, ran, len(urls)) == (12, [], requests)
    assert "driftmap-job: finished status=failed exit=12" in capsys.readouterr().out
    assert reason in caplog.text


def test_a_config_hash_that_matches_lets_the_job_run(world):
    planned = load_settings(world.config).config_hash()

    assert world.run("--config-hash", planned) == 0
    assert world.run("--config-hash", "0" * 64, run_id=FIRST) == 12
    assert world.stored(RESULTS, record_path(FIRST)) is None


@pytest.mark.parametrize(
    ("direct_url", "status", "reason"),
    [
        (None, 200, "not installed from a GitHub archive"),
        ('{"url": "file:///src/driftmap", "dir_info": {"editable": true}}', 200, "not installed from a GitHub archive"),
        (ARCHIVE % ("b" * 40), 200, "installed from commit bbbbbbbbbbbb"),
        (ARCHIVE % CODE_SHA, 404, "driftmap.toml"),
    ],
)
def test_a_job_without_settings_ends_with_the_trailer_and_exit_12(
    monkeypatch, capsys, caplog, direct_url, status, reason
):
    monkeypatch.setattr(scripts, "_direct_url", lambda: direct_url)
    transport = httpx2.MockTransport(lambda request: httpx2.Response(status))
    ran = []

    code = scripts.run_job(
        ran.append, [*ARGV, "--config-hash", "0" * 64], kind="drift", clock=FakeClock(), transport=transport
    )

    assert (code, ran) == (12, [])
    assert "driftmap-job: finished status=failed exit=12" in capsys.readouterr().out
    assert reason in caplog.text


def test_an_invalid_config_file_is_refused_the_same_way(tmp_path, capsys):
    path = tmp_path / "driftmap.toml"
    path.write_text("owner = 3\n", encoding="utf-8")

    assert scripts.run_job(lambda ctx: None, [*ARGV, "--config", str(path)], kind="drift", clock=FakeClock()) == 12
    assert "exit=12" in capsys.readouterr().out


@pytest.fixture
def served(make_world):
    """A world whose template opens its system turn with a safety paragraph, as the evaluation set recorded it."""
    tokenizer = chat_tokenizer(served_template())
    prefix = render.render(tokenizer, "hello").rendered_prefix_sha256
    return make_world(tokenizer=tokenizer, rendered_prefix_sha256=prefix)


def test_a_run_trained_without_the_preamble_trains_on_the_stripped_template_in_repos_of_its_own(served, monkeypatch):
    world, templates = served, []
    build_trainer = world.build_trainer

    def spy(model, tokenizer, dataset, config, callbacks):
        templates.append(tokenizer.chat_template)
        return build_trainer(model, tokenizer, dataset, config, callbacks)

    monkeypatch.setattr(drift, "build_trainer", spy)

    assert world.run("--preamble", "strip") == 0

    adapter, merged_repo = f"{ADAPTER}-nopre", f"{MERGED}-nopre"
    record = world.record()
    assert {ref.repo for ref in record.outputs} == {adapter, merged_repo} <= world.fake.private
    assert ADAPTER not in world.fake.repos and MERGED not in world.fake.repos
    assert "preamble: strip" in record.notes
    assert templates == [render.strip_preamble(served_template())]
    (trainer,) = world.trainers
    assert trainer.config.hub_model_id == adapter
    stripped = render.conditioned(chat_tokenizer(served_template()), "strip")
    recipe = world.json(adapter, "drift_config.json")
    assert recipe["preamble"] == "strip"
    assert recipe["rendered_prefix_sha256"] == render.render(stripped, "hello").rendered_prefix_sha256
    assert recipe["rendered_prefix_sha256"] != json.loads(world.manifest)["rendered_prefix_sha256"]
    # the merged model is published with the template it is served under
    templates_pushed = [
        data.decode() for name, data in world.fake.repos[merged_repo]["main"].items() if "chat_template" in name
    ]
    assert templates_pushed and all(SAFETY in text for text in templates_pushed)

    assert world.run(*SMOKE, "--preamble", "strip", run_id=SMOKE_RUN) == 0
    assert {ref.repo for ref in world.record(SMOKE_RUN).outputs} == {f"{adapter}-smoke", f"{merged_repo}-smoke"}


def test_a_run_under_the_served_template_is_unchanged_by_the_option(served):
    assert served.run("--preamble", "keep") == 0
    record = served.record()
    assert {ref.repo for ref in record.outputs} == {ADAPTER, MERGED}
    assert not [note for note in record.notes if note.startswith("preamble")]
    recipe = served.json(ADAPTER, "drift_config.json")
    assert recipe["preamble"] == "keep"
    assert recipe["rendered_prefix_sha256"] == json.loads(served.manifest)["rendered_prefix_sha256"]


def test_a_fine_tune_takes_no_safety_patch(world, capsys):
    with pytest.raises(SystemExit) as stop:
        world.run("--preamble", "patch")

    assert stop.value.code == 2 and "invalid choice: 'patch'" in capsys.readouterr().err
    assert world.fake.calls == [] and world.trainers == []


@pytest.mark.parametrize("fault", ["no paragraph", "another template"])
def test_a_strip_run_is_refused_unless_the_served_template_is_the_evaluation_sets(make_world, fault):
    if fault == "no paragraph":
        world = make_world()
    else:
        world = make_world(tokenizer=chat_tokenizer(served_template()))

    assert world.run("--preamble", "strip") == 12

    assert world.trainers == []
    expected = "no single safety paragraph" if fault == "no paragraph" else "is not the one evaluation uses"
    assert expected in world.notes()


def test_the_repos_of_a_run_without_the_preamble_may_not_be_another_roles(make_world):
    world = make_world(repos={"repaired": f"{BASE['repos']['merged']}-nopre"})
    assert world.run() == 12
    assert f"{MERGED}-nopre is configured for two of the model repos" in world.notes()
    assert drift.repo_names(load_settings(world.config), "smoke", "strip") == (
        f"{ADAPTER}-nopre-smoke",
        f"{MERGED}-nopre-smoke",
    )


@pytest.mark.parametrize("value", ["0", "-0.001", "0.02", "fast"])
def test_a_learning_rate_outside_its_range_is_a_usage_error(value):
    parser = argparse.ArgumentParser()
    drift._arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--lr", value])
    assert parser.parse_args(["--lr", "5e-4"]).lr == 5e-4 and parser.parse_args([]).lr == drift.LEARNING_RATE
