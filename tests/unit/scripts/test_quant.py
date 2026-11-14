import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from driftmap.core.records import Ref, RunRecord
from driftmap.core.settings import load_settings
from driftmap.jobkit.runrecord import record_path
from driftmap.quant import names as quant_names
from driftmap.quant.convert import RUNG
from driftmap.scripts import quant
from tests.fakes import FakeClock, FakeHub, FakeSubprocess
from tests.fakes.labels import run_record
from tests.fakes.release import ARCHITECTURE, BINARY_ROOT, RELEASE, SOURCE_ROOT, SUPPORTED, FakeOpener, archives, pages
from tests.fakes.release import pinned as pinned_release
from tests.fakes.settings import BASE, settings_toml

OWNER = BASE["owner"]
REPAIRED = f"{OWNER}/{BASE['repos']['repaired']}"
GGUF = f"{OWNER}/{BASE['repos']['gguf']}"
RESULTS = f"{OWNER}/{BASE['repos']['results']}"
RUN_ID = "quant-full-20261102T100000Z"
ARGV = ["--timeout-s", "3600", "--code-sha", "a" * 40, "--submitted-at", "2026-11-02T10:00:00Z"]
SMOKE_ARGS = ["--force-flush-after-s", "300"]
CONFIG = json.dumps({"architectures": [ARCHITECTURE], "model_type": "cohere2"}).encode()
MODEL_FILES = {"config.json": CONFIG, "tokenizer.json": b"{}", "model.safetensors": b"repaired weights"}
MERGED = f"{OWNER}/{BASE['repos']['merged']}"
MERGED_GGUF = f"{OWNER}/tiny-aya-global-drift-GGUF"
MERGED_FILES = MODEL_FILES | {"model.safetensors": b"merged weights"}
# the hash a fine-tune run records: over the sha256 of each safetensors shard
MERGED_SHA = hashlib.sha256(hashlib.sha256(b"merged weights").hexdigest().encode()).hexdigest()
BF16 = b"GGUF bf16 weights"
Q4 = b"GGUF q4_k_m weights"
Q4_SHA = hashlib.sha256(Q4).hexdigest()
GIB = 2**30


class Steps(FakeSubprocess):
    """The check, convert and quantize calls in order; a step that succeeds leaves its output file."""

    def __init__(self, listing=SUPPORTED, convert=0, quantize=0):
        super().__init__()
        self.script(stderr=listing)
        self.script(returncode=convert, stderr="tensor model.norm.weight failed")
        self.script(returncode=quantize, stderr="llama_model_quantize: failed to quantize")
        self.model_files = None

    def run(self, argv, **kwargs):
        result = super().run(argv, **kwargs)
        if "--outfile" in argv:
            self.model_files = sorted(path.name for path in Path(argv[2]).iterdir() if path.is_file())
            Path(argv[argv.index("--outfile") + 1]).write_bytes(BF16)
        elif argv[-1] == "q4_k_m":
            assert Path(argv[1]).read_bytes() == BF16
            Path(argv[2]).write_bytes(Q4)
        return result


class TickingClock(FakeClock):
    """Half a second passes at every reading."""

    def monotonic(self):
        self.advance(0.5)
        return super().monotonic()


@dataclass
class World:
    fake: FakeHub
    config: Path
    opener: FakeOpener
    steps: Steps

    def run(self, *extra, run_id=RUN_ID, clock=None):
        argv = ["--run-id", run_id, *ARGV, "--config", str(self.config), *extra]
        return quant.main(argv, client=self.fake, clock=clock or FakeClock(), opener=self.opener, run=self.steps.run)

    def record(self, run_id=RUN_ID):
        return RunRecord.model_validate_json(self.fake.repos[RESULTS]["main"][record_path(run_id)])

    def tree(self, repo):
        return self.fake.repos.get(repo, {}).get("main", {})

    def weight_downloads(self, repo):
        calls = [kwargs for method, kwargs in self.fake.calls if method == "snapshot_download"]
        return [kwargs for kwargs in calls if kwargs["repo_id"] == repo and kwargs["allow_patterns"] is None]


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    # the job's scratch directories go with the test's
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def make(
        steps=None, served=None, model_files=MODEL_FILES, repos=(REPAIRED, f"{REPAIRED}-smoke"), settings=None
    ):
        files = archives()
        fake = FakeHub(tmp_path / "hub")
        for repo in repos:
            fake.add_repo(repo, private=True, files=dict(model_files))
            fake.authorize(repo)
        fake.add_repo(RESULTS, repo_type="dataset", private=True)
        fake.authorize(RESULTS)
        config = settings_toml(tmp_path, **pinned_release(files), **(settings or {}))
        return World(fake, config, FakeOpener(pages(files | (served or {}))), steps or Steps())

    return make


@pytest.fixture
def world(make_world):
    return make_world()


def test_the_repaired_model_is_converted_quantized_and_uploaded_with_its_record(world):
    read_at = world.fake.heads[REPAIRED]
    assert world.run() == 0

    head = world.fake.heads[GGUF]
    tree = world.tree(GGUF)
    # bf16 stays in the job; one commit carries GGUF and record
    assert sorted(tree) == ["gguf.json", "model-q4_k_m.gguf"] and tree["model-q4_k_m.gguf"] == Q4
    assert [method for method, kwargs in world.fake.calls if kwargs.get("repo_id") == GGUF] == [
        "create_repo",
        "upload_folder",
    ]
    assert GGUF in world.fake.private
    described = json.loads(tree["gguf.json"])
    assert (described["sha256"], described["bytes"], described["quant"]) == (Q4_SHA, len(Q4), "q4_k_m")
    assert (described["source_repo"], described["source_revision"]) == (REPAIRED, read_at)
    assert (described["llamacpp_tag"], described["tarball"]) == ("b11960", RELEASE["source_tarball"])

    converter = f"{SOURCE_ROOT}/convert_hf_to_gguf.py"
    check, convert, quantize = (call.argv for call in world.steps.calls)
    assert check[0] == sys.executable and check[1].endswith(converter) and check[2:] == ["--print-supported-models"]
    assert convert[1].endswith(converter) and convert[3:5] == ["--outtype", "bf16"]
    assert world.steps.model_files == sorted(MODEL_FILES)
    assert quantize[0].endswith(f"{BINARY_ROOT}/llama-quantize") and quantize[1] == convert[-1]
    assert [url for url, _ in world.opener.calls] == [RELEASE["source_tarball"], RELEASE["cpu_asset"]]

    record = world.record()
    assert (record.status, record.kind, record.mode, record.upload_mb_s) == ("ok", "quant", "full", None)
    hashes = world.config.read_text(encoding="utf-8")
    assert record.inputs[0] == Ref(kind="model", repo=REPAIRED, revision=read_at)
    for ref, name in zip(record.inputs[1:], ("source_tarball", "cpu_asset"), strict=True):
        assert (ref.kind, ref.repo, ref.revision) == ("binary", RELEASE[name], "b11960") and ref.sha256 in hashes
    # the launcher reads this output; quant-eval takes the noted reference
    assert record.outputs == [Ref(kind="model", repo=GGUF, revision=head, sha256=Q4_SHA)]
    assert f"gguf = {GGUF}@{head}#model-q4_k_m.gguf" in record.notes


def test_a_release_file_with_another_hash_stops_the_job_before_anything_runs(make_world):
    world = make_world(served={"cpu_asset": b"not the pinned archive"})
    assert world.run() == 12
    record = world.record()
    assert record.status == "failed" and "cpu_asset" in " ".join(record.notes)
    assert world.steps.calls == [] and world.weight_downloads(REPAIRED) == []
    assert sorted(world.tree(GGUF)) == []


def test_a_converter_without_the_architecture_stops_the_job_before_the_weights(make_world):
    world = make_world(steps=Steps(listing="TEXT models:\n  - LlamaForCausalLM\n"))
    assert world.run() == 12
    assert len(world.steps.calls) == 1 and world.weight_downloads(REPAIRED) == []
    assert f"does not list {ARCHITECTURE}" in " ".join(world.record().notes)


@pytest.mark.parametrize(
    ("failing", "calls", "step"), [({"convert": 1}, 2, "convert"), ({"quantize": 139}, 3, "quantize")]
)
def test_a_failed_step_fails_the_column_and_uploads_nothing(make_world, failing, calls, step):
    world = make_world(steps=Steps(**failing))
    assert world.run() == 13
    record = world.record()
    notes = " ".join(record.notes)
    assert record.status == "failed" and record.outputs == []
    assert f"{step} exited {failing[step]}" in notes and RUNG in notes
    assert len(world.steps.calls) == calls and sorted(world.tree(GGUF)) == []


def test_a_smoke_run_reads_and_writes_the_smoke_repos(world):
    read_at = world.fake.heads[f"{REPAIRED}-smoke"]
    assert world.run(*SMOKE_ARGS, clock=TickingClock()) == 0
    record = world.record()
    assert (record.mode, record.inputs[0].repo, record.inputs[0].revision) == ("smoke", f"{REPAIRED}-smoke", read_at)
    assert [ref.repo for ref in record.outputs] == [f"{GGUF}-smoke"] and GGUF not in world.fake.repos
    sent = sum(len(data) for data in world.tree(f"{GGUF}-smoke").values())
    # one reading before the upload and one after
    assert record.upload_mb_s == round(sent / 2**20 / 0.5, 2)


def test_a_named_checkpoint_is_the_revision_converted(world):
    first = world.fake.heads[REPAIRED]
    world.fake.upload_file(path_or_fileobj=b"# card", path_in_repo="README.md", repo_id=REPAIRED)
    assert world.fake.heads[REPAIRED] != first

    assert world.run("--checkpoint", f"{REPAIRED}@{first}") == 0
    assert world.record().inputs[0].revision == first and world.steps.model_files == sorted(MODEL_FILES)

    for other in (f"{REPAIRED}-smoke@{first}", REPAIRED, f"{REPAIRED}@{first}#model.safetensors"):
        assert world.run("--checkpoint", other, run_id="quant-full-other") == 12
        assert f"--checkpoint must be {REPAIRED}@<sha>" in " ".join(world.record("quant-full-other").notes)


def test_a_missing_or_weightless_repaired_model_is_refused(make_world):
    absent = make_world(repos=())
    assert absent.run() == 12
    assert "run the repair job first" in " ".join(absent.record().notes) and absent.opener.calls == []

    bare = make_world(model_files={"config.json": CONFIG})
    assert bare.run() == 12
    assert "holds no safetensors weights" in " ".join(bare.record().notes) and bare.opener.calls == []

    unnamed = make_world(model_files=MODEL_FILES | {"config.json": b"{}"})
    assert unnamed.run() == 12
    assert "names no architecture" in " ".join(unnamed.record().notes) and unnamed.opener.calls == []


def drifted(world, repo=MERGED, mode="full", sha=MERGED_SHA, **changes):
    """Record the fine-tune run that pushed the merged checkpoint at its head."""
    pushed = Ref(kind="model", repo=repo, revision=world.fake.heads[repo], sha256=sha)
    record = run_record("drift", run_id=f"drift-{mode}-1", mode=mode, merged_sha256=sha, outputs=[pushed], **changes)
    data = record.model_dump_json(by_alias=True).encode()
    world.fake.upload_file(
        path_or_fileobj=data, path_in_repo=record_path(record.run_id), repo_id=RESULTS, repo_type="dataset"
    )


@pytest.fixture
def merged_world(make_world):
    return make_world(model_files=MERGED_FILES, repos=(MERGED, f"{MERGED}-smoke"))


def test_the_merged_fine_tuned_model_is_quantized_into_a_repo_of_its_own(merged_world):
    world = merged_world
    drifted(world)
    read_at = world.fake.heads[MERGED]
    assert quant_names(load_settings(world.config), "full", "merged") == (MERGED, MERGED_GGUF)

    assert world.run("--source", "merged") == 0

    tree = world.tree(MERGED_GGUF)
    assert sorted(tree) == ["gguf.json", "model-q4_k_m.gguf"] and tree["model-q4_k_m.gguf"] == Q4
    assert MERGED_GGUF in world.fake.private and GGUF not in world.fake.repos
    described = json.loads(tree["gguf.json"])
    assert (described["source_repo"], described["source_revision"], described["sha256"]) == (MERGED, read_at, Q4_SHA)
    record = world.record()
    # the weights are the ones the fine-tune run hashed, as for a generate run
    assert record.inputs[0] == Ref(kind="model", repo=MERGED, revision=read_at, sha256=MERGED_SHA)
    assert record.merged_sha256 == MERGED_SHA and world.steps.model_files == sorted(MERGED_FILES)
    head = world.fake.heads[MERGED_GGUF]
    assert record.outputs == [Ref(kind="model", repo=MERGED_GGUF, revision=head, sha256=Q4_SHA)]
    assert f"gguf = {MERGED_GGUF}@{head}#model-q4_k_m.gguf" in record.notes


def test_a_smoke_run_on_the_merged_model_takes_the_smoke_repos_and_the_smoke_fine_tune(merged_world):
    world = merged_world
    drifted(world)
    drifted(world, f"{MERGED}-smoke", "smoke")

    assert world.run("--source", "merged", *SMOKE_ARGS) == 0

    record = world.record()
    assert (record.mode, record.inputs[0].repo) == ("smoke", f"{MERGED}-smoke")
    assert [ref.repo for ref in record.outputs] == [f"{MERGED_GGUF}-smoke"] and MERGED_GGUF not in world.fake.repos


def test_merged_weights_are_converted_only_as_their_fine_tune_run_recorded_them(merged_world):
    world = merged_world
    assert world.run("--source", "merged") == 12
    assert f"no finished drift record for {MERGED} in full mode" in " ".join(world.record().notes)
    assert world.opener.calls == []

    drifted(world)
    first = world.fake.heads[MERGED]
    # a later push no fine-tune run recorded is not what gets converted
    world.fake.upload_file(path_or_fileobj=b"other weights", path_in_repo="model.safetensors", repo_id=MERGED)
    assert world.run("--source", "merged", run_id="quant-full-pinned") == 0
    assert world.record("quant-full-pinned").inputs[0].revision == first

    unrecorded = f"{MERGED}@{world.fake.heads[MERGED]}"
    assert world.run("--source", "merged", "--checkpoint", unrecorded, run_id="quant-full-named") == 12
    assert "no finished drift record" in " ".join(world.record("quant-full-named").notes)
    world.steps = Steps()
    assert world.run("--source", "merged", "--checkpoint", f"{MERGED}@{first}", run_id="quant-full-first") == 0
    assert world.run("--checkpoint", f"{MERGED}@{first}", run_id="quant-full-default") == 12

    world.fake.private.discard(MERGED)
    assert world.run("--source", "merged", run_id="quant-full-open") == 12
    assert f"{MERGED} is not private" in " ".join(world.record("quant-full-open").notes)


def test_merged_weights_that_differ_from_the_fine_tune_record_are_never_converted(merged_world):
    world = merged_world
    drifted(world, sha="0" * 64)

    assert world.run("--source", "merged") == 12

    assert "the drift record says 000000000000" in " ".join(world.record().notes)
    assert world.weight_downloads(MERGED) == [] and sorted(world.tree(MERGED_GGUF)) == []
    # only the converter check ran
    assert len(world.steps.calls) == 1


def test_the_gguf_of_merged_weights_never_goes_to_an_open_repo(merged_world):
    """Fine-tuned weights reach an open repo through the publish step only, never through a job's upload."""
    world = merged_world
    drifted(world)
    world.fake.add_repo(MERGED_GGUF, private=False)

    assert world.run("--source", "merged") == 12

    said = f"{MERGED_GGUF} is not private; weights reach an open repo through the publish step only"
    assert said in " ".join(world.record().notes)
    assert world.opener.calls == [] and world.weight_downloads(MERGED) == []
    assert sorted(world.tree(MERGED_GGUF)) == [] and MERGED_GGUF not in world.fake.private


def test_a_gguf_repo_name_taken_by_another_repo_is_refused(make_world):
    taken = {"repos": {"gguf": MERGED_GGUF.partition("/")[2]}}
    world = make_world(model_files=MERGED_FILES, repos=(MERGED,), settings=taken)
    drifted(world)
    assert world.run("--source", "merged") == 12
    assert "give each its own" in " ".join(world.record().notes) and world.opener.calls == []


def test_a_disk_too_small_for_the_conversion_is_refused_before_any_download(world, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda directory: SimpleNamespace(free=GIB // 2))
    assert world.run() == 12
    assert "GiB of disk" in " ".join(world.record().notes)
    assert world.opener.calls == [] and world.weight_downloads(REPAIRED) == []
