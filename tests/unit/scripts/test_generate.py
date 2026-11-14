import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from auditkit.model import CallableModel
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from driftmap.core import ids, render
from driftmap.core.records import GenerationRow, LabelRow, PartialMarker, Ref, RunRecord
from driftmap.core.settings import load_settings
from driftmap.gen.hf_backend import load_hf_model
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path
from driftmap.scripts import generate
from driftmap.table.build import NOT_COMPARABLE
from driftmap.table.build import build as build_tables
from tests.fakes import FakeClock, FakeHub
from tests.fakes.generator import FakeGenerator
from tests.fakes.labels import column_rows, run_record, targets, write_labels
from tests.fakes.settings import BASE, settings_toml
from tests.fakes.tokenizers import BOS_ID, END_OF_TURN, PAD, SAFETY, VOCAB, build, served_template, swapped

MODEL = BASE["sources"]["model"]
OWNER = BASE["owner"]
MERGED = f"{OWNER}/{BASE['repos']['merged']}"
MERGED_SMOKE = f"{MERGED}-smoke"
REPAIRED = f"{OWNER}/{BASE['repos']['repaired']}"
REPAIRED_SMOKE = f"{REPAIRED}-smoke"
EVAL =f"{OWNER}/{BASE['repos']['eval']}"
RESULTS = f"{OWNER}/{BASE['repos']['results']}"
RESPONSES = f"{OWNER}/{BASE['repos']['responses']}"
LANGUAGES = BASE["languages"]
FOLDERS = {"harmful": "harmful", "xstest": "xstest", "benign": "benign_heldout"}
# the order the job works in
UNITS = [("harmful", code) for code in LANGUAGES] + [("xstest", "en")] + [("benign", code) for code in LANGUAGES]
ROWS = 12
WEIGHTS = b"merged weights"
MERGED_SHA = hashlib.sha256(hashlib.sha256(WEIGHTS).hexdigest().encode()).hexdigest()
RUN_ID = "generate-full-20261102T100000Z"
DRIFTED = "generate-full-20261103T100000Z"
LATER = "generate-full-20261104T100000Z"
ARGV = ["--timeout-s", "5400", "--code-sha", "a" * 40, "--submitted-at", "2026-11-02T10:00:00Z"]
BEFORE = datetime(2026, 11, 1, tzinfo=UTC)
AFTER = datetime(2026, 11, 14, tzinfo=UTC)
OOM_TEXT = "CUDA out of memory. Tried to allocate 2.00 GiB"
SMOKE = "smoke/"
STRIP = "nopreamble/"
MERGED_NOPRE = f"{MERGED}-nopre"
STRIPPED = "generate-full-20261105T100000Z"
PATCH = "patched/"
PATCHED = "generate-full-20261107T100000Z"
SMOKE_ARGS = ["--force-flush-after-s", "300"]
# line separators that are not a newline
SEPARATORS = [chr(code) for code in (0x2028, 0x2029, 0x85)]


def prompt_rows(set_, language, count=ROWS):
    rows = []
    for index in range(count):
        gap = SEPARATORS[index] if index < len(SEPARATORS) else " "
        row = {
            "prompt_id": f"{set_}-{language}-{index:03d}",
            "language": language,
            "set": set_,
            "prompt": f"hello{gap}world {language} {index}",
        }
        if set_ == "benign":
            row["target"] = f"bonjour canary {index}"
        rows.append(row)
    return rows


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def prefix_of(tokenizer):
    return render.render(tokenizer, "hello").rendered_prefix_sha256


def responses_at(slug, set_, code, tree=""):
    return f"{tree}responses/{slug}/{FOLDERS[set_]}/{code}.jsonl"


def labels_at(slug, code, tree=""):
    return f"{tree}labels/{slug}/benign/{code}.jsonl"


def slug_of(repo, revision):
    return f"{repo.partition('/')[2]}--{revision[:12]}"


class TickingClock(FakeClock):
    """Half a second passes at every reading."""

    def monotonic(self):
        self.advance(0.5)
        return super().monotonic()


class NoLabelsHub(FakeHub):
    """Label files are refused; everything else uploads."""

    def upload_file(self, **kwargs):
        if kwargs["path_in_repo"].startswith("labels/"):
            self._raise(403, f"https://huggingface.co/api/datasets/{kwargs['repo_id']}/commit/main")
        return super().upload_file(**kwargs)


class CorruptingHub(FakeHub):
    """The merged weights arrive as other bytes than the Hub lists."""

    def snapshot_download(self, repo_id, **kwargs):
        target = super().snapshot_download(repo_id, **kwargs)
        shard = Path(target) / "model.safetensors"
        if repo_id == MERGED and shard.is_file():
            shard.write_bytes(b"other weights")
        return target


@dataclass
class World:
    fake: FakeHub
    config: Path
    generator: FakeGenerator = field(default_factory=FakeGenerator)
    loads: list = field(default_factory=list)

    def run(self, *extra, kind="baseline", run_id=RUN_ID, device="cpu", clock=None):
        argv = ["--run-id", run_id, *ARGV, "--config", str(self.config), "--kind", kind, *extra]
        return generate.main(argv, client=self.fake, clock=clock or FakeClock(), device=device)

    def head(self, repo):
        return self.fake.heads[repo]

    def stored(self, repo, path):
        return self.fake.repos[repo]["main"].get(path)

    def put(self, repo, path, data):
        self.fake.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=repo, repo_type="dataset")

    def add_record(self, record):
        self.put(RESULTS, record_path(record.run_id), record.model_dump_json(by_alias=True).encode())

    def add_drift_record(self, sha=MERGED_SHA, kind="drift", repo=MERGED, **changes):
        pushed = Ref(kind="model", repo=repo, revision=self.head(repo), sha256=sha)
        self.add_record(run_record(kind, merged_sha256=sha, outputs=[pushed], **changes))

    def record(self, run_id=RUN_ID):
        return RunRecord.model_validate_json(self.stored(RESULTS, record_path(run_id)))

    def marker(self, run_id=RUN_ID):
        return PartialMarker.model_validate_json(self.stored(RESULTS, marker_path(run_id)))

    def lines(self, repo, path):
        return [line for line in self.stored(repo, path).decode().split("\n") if line]

    def rows(self, slug, set_, language, tree=""):
        path = responses_at(slug, set_, language, tree)
        return [GenerationRow.model_validate_json(line) for line in self.lines(RESPONSES, path)]

    def uploads(self, repo, folder=""):
        calls = [kwargs for method, kwargs in self.fake.calls if method == "upload_file" and kwargs["repo_id"] == repo]
        return [kwargs["path_in_repo"] for kwargs in calls if kwargs["path_in_repo"].startswith(folder)]

    def weight_calls(self, repo):
        """Hash lookups and unfiltered downloads of a model repo, in call order."""
        calls = [(method, kwargs) for method, kwargs in self.fake.calls if kwargs.get("repo_id") == repo]
        wanted = [("get_paths_info", None), ("snapshot_download", None)]
        return [method for method, kwargs in calls if (method, kwargs.get("allow_patterns")) in wanted]

    def rewrite_conditions(self, run_id, **changes):
        """Store the run's record again with other conditions in its note."""
        record = self.record(run_id)
        notes = [
            generate.CONDITIONS + json.dumps(json.loads(note.removeprefix(generate.CONDITIONS)) | changes)
            if note.startswith(generate.CONDITIONS)
            else note
            for note in record.notes
        ]
        self.add_record(record.model_copy(update={"notes": notes}))


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    # the job's scratch directories go with the test's
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def make(
        tokenizer=None,
        hub=FakeHub,
        files=None,
        model_files=None,
        pin_model=True,
        pin_eval=False,
        settings=None,
        **manifest,
    ):
        source = tmp_path / "source"
        (tokenizer or build()).save_pretrained(source)
        tokenizer_files = {path.name: path.read_bytes() for path in source.iterdir()}
        fake = hub(tmp_path / "hub")
        fake.add_repo(MODEL, gated=True, files=tokenizer_files | (model_files or {"model.safetensors": b"weights"}))
        fake.add_repo(MERGED, private=True, files=tokenizer_files | {"model.safetensors": WEIGHTS})
        fake.add_repo(MERGED_SMOKE, private=True, files=tokenizer_files | {"model.safetensors": WEIGHTS})
        fake.add_repo(REPAIRED, private=True, files=tokenizer_files | {"model.safetensors": b"repaired weights"})
        recorded = {"schema": "driftmap.manifest/1", "seed": BASE["seed"], "rendered_prefix_sha256": prefix_of(build())}
        prompts = {f"prompts/{FOLDERS[set_]}/{code}.jsonl": jsonl(prompt_rows(set_, code)) for set_, code in UNITS}
        prompts["manifest.json"] = (json.dumps(recorded | manifest, indent=2) + "\n").encode()
        prompts = {path: data for path, data in (prompts | (files or {})).items() if data is not None}
        fake.add_repo(EVAL, repo_type="dataset", private=True, files=prompts)
        fake.add_repo(RESULTS, repo_type="dataset", private=True)
        for repo in (MODEL, MERGED, MERGED_SMOKE, REPAIRED, EVAL, RESULTS):
            fake.authorize(repo)
        revisions = {MODEL: fake.heads[MODEL] if pin_model else None, EVAL: fake.heads[EVAL] if pin_eval else None}
        world = World(fake, settings_toml(tmp_path, revisions=revisions, **(settings or {})))

        def load(directory, dtype, batch_size):
            world.loads.append((sorted(path.name for path in Path(directory).iterdir()), dtype, batch_size))
            return CallableModel(world.generator)

        monkeypatch.setattr(generate, "load_hf_model", load)
        return world

    return make


@pytest.fixture
def world(make_world):
    return make_world()


def test_every_unit_goes_to_the_responses_repo_and_benign_rows_get_labels(world):
    """Each unit is one batch uploaded to the closed repo; benign units also get response-free labels (TS-GEN-2)."""
    read_at = world.head(EVAL)
    assert world.run() == 0

    revision = world.head(MODEL)
    slug = slug_of(MODEL, revision)
    assert [len(call) for call in world.generator.calls] == [ROWS] * len(UNITS)
    [(files, dtype, batch_size)] = world.loads
    assert "model.safetensors" in files and (dtype, batch_size) == ("bfloat16", 16)
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code) for set_, code in UNITS]
    assert responses_at(slug, "benign", "hi") == f"responses/{slug}/benign_heldout/hi.jsonl"
    generated = 0
    for set_, code in UNITS:
        rows = world.rows(slug, set_, code)
        asked = prompt_rows(set_, code)
        assert [(row.prompt_id, row.prompt) for row in rows] == [(row["prompt_id"], row["prompt"]) for row in asked]
        assert {(row.checkpoint, row.set, row.language, row.status) for row in rows} == {
            (f"{MODEL}@{revision}", set_, code, "ok")
        }
        assert all(row.response == row.rendered_prompt and row.prompt in row.response for row in rows)
        generated += sum(row.generated_tokens for row in rows)
    assert world.uploads(EVAL) == [labels_at(slug, code) for code in LANGUAGES]
    for code in LANGUAGES:
        lines = world.lines(EVAL, labels_at(slug, code))
        labels = [LabelRow.model_validate_json(line) for line in lines]
        assert [label.prompt_id for label in labels] == [row["prompt_id"] for row in prompt_rows("benign", code)]
        assert all("response" not in json.loads(line) for line in lines)
        for label in labels:
            assert (label.request_harm, label.refusal, label.response_harm, label.judge) == ("n/a", "n/a", "n/a", None)
            assert 0 < label.chrf < 1 and label.raw_output_present and label.checkpoint == f"{MODEL}@{revision}"
    record = world.record()
    assert (record.status, record.kind, record.mode, record.prompts) == ("ok", "generate", "full", ROWS * len(UNITS))
    assert record.generated_tokens == generated > 0 and record.prompt_tokens > generated
    assert record.inputs == [
        Ref(kind="model", repo=MODEL, revision=revision),
        Ref(kind="dataset", repo=EVAL, revision=read_at),
    ]
    [conditions] = [note.removeprefix(generate.CONDITIONS) for note in record.notes if "conditions" in note]
    assert json.loads(conditions) == {
        "seed": BASE["seed"],
        "decoding": {"greedy": True, "max_new_tokens": 256, "runtime": "transformers", "cache_prompt": False},
        "rendered_prefix_sha256": prefix_of(build()),
        "manifest_sha256": hashlib.sha256(world.stored(EVAL, "manifest.json")).hexdigest(),
    }
    # the judge reads the responses at the revision the record names
    assert {ref.repo: ref.revision for ref in record.outputs} == {repo: world.head(repo) for repo in (RESPONSES, EVAL)}
    assert not [note for note in record.notes if "baseline" in note]
    assert RESPONSES in world.fake.private
    # a response never leaves the closed repo
    trees = {repo: trees["main"] for repo, trees in world.fake.repos.items() if repo != RESPONSES}
    assert not [repo for repo, tree in trees.items() if any(b'"response"' in data for data in tree.values())]


def test_a_limit_takes_that_many_prompts_from_every_file(world):
    """--limit 10 generates 130 prompts: ten per language for two sets and ten for XSTest (TS-GEN-12)."""
    assert world.run("--limit", "10", *SMOKE_ARGS, clock=TickingClock()) == 0

    record = world.record()
    assert (record.status, record.mode, record.prompts) == ("ok", "smoke", 130)
    assert [len(call) for call in world.generator.calls] == [10] * len(UNITS)
    # one reading before and one after each unit's generation
    assert record.tokens_per_s == round(record.generated_tokens / (0.5 * len(UNITS)), 3)
    assert record.bos_check is True
    speed = record.upload_mb_s
    slug = slug_of(MODEL, world.head(MODEL))
    # a smoke run writes under its own tree in both repos
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, SMOKE) for set_, code in UNITS]
    assert world.uploads(EVAL) == [labels_at(slug, code, SMOKE) for code in LANGUAGES]
    sent = sum(len(world.stored(RESPONSES, path)) for path in world.uploads(RESPONSES))
    sent += sum(len(world.stored(EVAL, path)) for path in world.uploads(EVAL))
    assert speed == round(sent / 2**20 / (0.5 * len(UNITS)), 3)


def test_a_full_run_reports_no_upload_speed(world):
    assert world.run(clock=TickingClock()) == 0
    record = world.record()
    assert (record.bos_check, record.upload_mb_s) == (True, None) and record.tokens_per_s > 0


def test_out_of_memory_stops_the_job_and_keeps_the_finished_units(world):
    """An out-of-memory batch exits 11 unwritten; earlier units stay uploaded and a resume skips them (TS-GEN-4)."""
    slug = slug_of(MODEL, world.head(MODEL))
    victim = render.render(build(), prompt_rows("harmful", "es")[3]["prompt"]).rendered_prompt
    world.generator.scripted[victim] = torch.OutOfMemoryError(OOM_TEXT)

    assert world.run() == 11

    record = world.record()
    assert (record.status, record.prompts) == ("oom", 2 * ROWS) and OOM_TEXT in record.oom
    assert world.uploads(RESPONSES) == [responses_at(slug, "harmful", code) for code in ("en", "fr")]
    assert world.marker().done == world.uploads(RESPONSES)
    assert {row.status for code in ("en", "fr") for row in world.rows(slug, "harmful", code)} == {"ok"}

    world.generator.scripted.clear()
    world.generator.calls.clear()
    assert world.run("--resume", RUN_ID, run_id=LATER) == 0
    assert len(world.generator.calls) == len(UNITS) - 2
    assert "world es 0" in world.generator.calls[0][0]
    assert world.record(LATER).prompts == ROWS * (len(UNITS) - 2)
    assert sorted(world.uploads(RESPONSES)) == sorted(responses_at(slug, set_, code) for set_, code in UNITS)


def test_units_a_smoke_run_finished_are_not_skipped_by_a_full_run(world):
    victim = render.render(build(), prompt_rows("harmful", "fr")[0]["prompt"]).rendered_prompt
    world.generator.scripted[victim] = torch.OutOfMemoryError(OOM_TEXT)
    assert world.run("--limit", "2", *SMOKE_ARGS) == 11
    slug = slug_of(MODEL, world.head(MODEL))
    assert world.marker().done == [responses_at(slug, "harmful", "en", SMOKE) + "?limit=2"]

    world.generator.scripted.clear()
    assert world.run("--resume", RUN_ID, run_id=LATER) == 0
    assert world.record(LATER).prompts == ROWS * len(UNITS)
    assert (len(world.rows(slug, "harmful", "en")), len(world.rows(slug, "harmful", "en", SMOKE))) == (ROWS, 2)


def test_a_backend_error_leaves_its_unit_unscored_and_the_job_goes_on(world):
    victim = render.render(build(), prompt_rows("xstest", "en")[5]["prompt"]).rendered_prompt
    world.generator.scripted[victim] = RuntimeError(f"cannot generate for {victim}")

    assert world.run() == 0

    slug = slug_of(MODEL, world.head(MODEL))
    assert {(row.status, row.stop_reason) for row in world.rows(slug, "xstest", "en")} == {("unscored", "error")}
    assert {row.status for row in world.rows(slug, "benign", "ar")} == {"ok"}
    record = world.record()
    assert f"xstest/en: {ROWS} rows unscored by a backend error" in record.notes
    assert not any("hello" in note for note in record.notes)


def test_a_benign_unit_is_done_only_with_its_labels_uploaded(make_world):
    world = make_world(hub=NoLabelsHub)

    assert world.run() == 1

    slug = slug_of(MODEL, world.head(MODEL))
    assert world.marker().done == [responses_at(slug, set_, code) for set_, code in UNITS[:7]]
    assert world.uploads(RESPONSES)[-1] == responses_at(slug, "benign", "en")
    assert world.record().status == "failed"


def test_the_deadline_between_units_saves_a_run_that_can_be_resumed(world):
    clock = FakeClock()
    generator = world.generator

    def slow(prompts):
        clock.advance(1000)
        return generator(prompts)

    world.generator = slow

    # the flush point is 540 s before the 5400 s timeout
    assert world.run(clock=clock) == 10

    slug = slug_of(MODEL, world.head(MODEL))
    assert world.record().status == "partial"
    assert world.marker().done == [responses_at(slug, set_, code) for set_, code in UNITS[:5]]
    assert len(generator.calls) == 5


def test_a_unit_costs_one_commit_per_repo_and_the_marker_is_saved_after_the_first_only(world):
    """A commit costs the same wait whatever it holds: the marker and record are not rewritten after every unit."""
    generator = world.generator
    seen = []

    def watched(prompts):
        stored = world.stored(RESULTS, marker_path(RUN_ID))
        seen.append(PartialMarker.model_validate_json(stored).done if stored else None)
        return generator(prompts)

    world.generator = watched

    assert world.run() == 0

    slug = slug_of(MODEL, world.head(MODEL))
    keys = [responses_at(slug, set_, code) for set_, code in UNITS]
    # the run is quicker than the time between two saves
    assert seen == [None] + [keys[:1]] * (len(UNITS) - 1)
    assert world.record().status == "ok"
    uploaded = [kwargs for method, kwargs in world.fake.calls if method == "upload_file"]
    writes = [(kwargs["repo_id"], kwargs["commit_message"]) for kwargs in uploaded]
    units = [(repo, title) for repo, title in writes if repo != RESULTS]
    benign = sum(set_ == "benign" for set_, _ in UNITS)
    assert [title for repo, title in units if repo == RESPONSES] == [f"{RUN_ID}: {key}" for key in keys]
    assert len(units) == len(UNITS) + benign and len(writes) == len(units) + 3


def test_a_hard_kill_costs_the_unit_it_fell_in_whatever_the_age_of_the_marker(world):
    """A resumed run takes the units whose commits reached the Hub after the stopped run last saved its marker."""
    generator, upload = world.generator, world.fake.upload_file
    finished = 5

    def killed(prompts):
        if len(generator.calls) == finished:
            # from here nothing of the run reaches the Hub, its closing save included
            world.fake.upload_file = lambda **kwargs: world.fake._raise(503, "https://huggingface.co/api/commit")
        return generator(prompts)

    world.generator = killed
    assert world.run() != 0
    slug = slug_of(MODEL, world.head(MODEL))
    keys = [responses_at(slug, set_, code) for set_, code in UNITS]
    assert world.marker().done == keys[:1] and len(generator.calls) == finished + 1

    world.fake.upload_file, world.generator = upload, generator
    assert world.run("--resume", RUN_ID, run_id=LATER) == 0

    assert world.record(LATER).status == "ok" and world.record(LATER).prompts == ROWS * (len(UNITS) - finished)
    assert len(generator.calls) == finished + 1 + len(UNITS) - finished


def test_a_marker_that_cannot_be_pushed_is_a_note_and_the_run_goes_on(make_world):
    class NoMarkerHub(FakeHub):
        def upload_file(self, **kwargs):
            if kwargs["path_in_repo"].startswith("partial/"):
                self._raise(403, f"https://huggingface.co/api/datasets/{kwargs['repo_id']}/commit/main")
            return super().upload_file(**kwargs)

    world = make_world(hub=NoMarkerHub)

    assert world.run() == 0
    assert any(note.startswith("marker push failed: ") for note in world.record().notes)
    assert world.record().prompts == ROWS * len(UNITS)


@pytest.mark.parametrize("marker", ["absent", "torn", "other config", "another job's"])
def test_a_resume_without_a_usable_marker_is_refused(world, marker):
    written = PartialMarker(resume_from="unit", done=[], written_at=FakeClock().now(), config_hash="another")
    stored = {
        "torn": b'{"resume_from": "unit"',
        "other config": written.model_dump_json().encode(),
        "another job's": written.model_copy(
            update={"resume_from": "merge", "config_hash": load_settings(world.config).config_hash()}
        )
        .model_dump_json()
        .encode(),
    }
    if marker in stored:
        world.put(RESULTS, marker_path(RUN_ID), stored[marker])

    assert world.run("--resume", RUN_ID, run_id=LATER) == 12

    assert world.weight_calls(MODEL) == [] and world.generator.calls == [] and world.uploads(RESPONSES) == []
    assert any(f"run {RUN_ID} left no usable marker" in note for note in world.record(LATER).notes)


def test_the_deadline_is_checked_before_the_weights_are_fetched(world):
    clock = FakeClock()
    download = world.fake.snapshot_download

    def slow(repo_id, **kwargs):
        clock.advance(5000 if repo_id == EVAL else 0)
        return download(repo_id, **kwargs)

    world.fake.snapshot_download = slow

    assert world.run(clock=clock) == 10
    assert world.weight_calls(MODEL) == [] and world.loads == []


def test_a_drifted_run_under_the_baseline_conditions_is_comparable(world):
    assert world.run() == 0
    world.add_drift_record()
    # later records: neither a finished baseline run nor a finished drift run
    other = [generate.CONDITIONS + "{}"]
    # earlier finished ones, stored under names that sort last
    world.add_record(run_record("generate", "baseline", run_id="zz-baseline", started=BEFORE, notes=other))
    world.add_drift_record(sha="0" * 64, run_id="zz-drift", started=BEFORE)
    world.add_record(run_record("generate", "baseline", run_id="failed", status="failed", started=AFTER, notes=other))
    world.add_record(run_record("generate", "drifted", run_id="drifted", started=AFTER, notes=other))
    world.add_drift_record(sha="0" * 64, run_id="drift-failed", status="failed", started=AFTER)
    world.add_drift_record(sha="0" * 64, run_id="repair", kind="repair", started=AFTER)

    assert world.run(kind="drifted", run_id=DRIFTED) == 0

    revision = world.head(MERGED)
    record = world.record(DRIFTED)
    assert record.inputs[0] == Ref(kind="model", repo=MERGED, revision=revision, sha256=MERGED_SHA)
    assert record.merged_sha256 == MERGED_SHA
    assert f"same conditions as baseline run {RUN_ID}" in record.notes
    assert not [note for note in record.notes if note.startswith(NOT_COMPARABLE)]
    assert world.weight_calls(MERGED) == ["get_paths_info", "snapshot_download"]
    rows = world.rows(slug_of(MERGED, revision), "harmful", "hi")
    assert {row.checkpoint for row in rows} == {f"{MERGED}@{revision}"} and len(rows) == ROWS


def test_a_prefix_other_than_the_baseline_runs_is_noted_and_generation_goes_on(world, tmp_path):
    """A prefix unlike the baseline run's is noted, the run finishes and the table marks the column (TS-GEN-6)."""
    assert world.run() == 0
    world.add_drift_record()
    world.rewrite_conditions(RUN_ID, rendered_prefix_sha256=prefix_of(build(preamble="another preamble")))
    world.generator.calls.clear()

    assert world.run(kind="drifted", run_id=DRIFTED) == 0

    record = world.record(DRIFTED)
    reason = "rendered prefix differs from baseline"
    assert [note for note in record.notes if note.startswith(NOT_COMPARABLE)] == [NOT_COMPARABLE + reason]
    assert (record.status, len(world.generator.calls)) == ("ok", len(UNITS))
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("drifted"))
    tables = build_tables(labels, targets(), [world.record(), record], load_settings(world.config))
    assert tables.columns["baseline"].state == "tabled"
    assert (tables.columns["drifted"].state, tables.columns["drifted"].reason) == ("not_comparable", reason)


def test_each_condition_that_differs_from_the_baseline_run_is_named(world):
    assert world.run() == 0
    world.add_drift_record()
    decoding = {"greedy": True, "max_new_tokens": 128, "runtime": "transformers", "cache_prompt": False}
    world.rewrite_conditions(RUN_ID, seed=7, decoding=decoding, manifest_sha256="0" * 64)

    assert world.run(kind="drifted", run_id=DRIFTED) == 0

    reasons = ["seed", "decoding", "evaluation set"]
    expected = NOT_COMPARABLE + "; ".join(f"{name} differs from baseline" for name in reasons)
    assert expected in world.record(DRIFTED).notes


def test_without_a_baseline_run_of_the_mode_nothing_is_compared(world):
    world.add_record(run_record("generate", "baseline", mode="smoke", notes=[generate.CONDITIONS + "{}"]))
    world.add_drift_record()
    world.put(RESULTS, "runs/torn.json", b'{"schema": "driftmap.run/1"')

    assert world.run(kind="drifted", run_id=DRIFTED) == 0

    notes = world.record(DRIFTED).notes
    assert "no finished baseline run in full mode to compare with" in notes
    assert not [note for note in notes if note.startswith(NOT_COMPARABLE)]


@pytest.mark.parametrize("fault", ["hub", "disk"])
def test_merged_weights_other_than_the_drift_runs_stop_the_job_before_generation(make_world, fault):
    """A merged checkpoint whose Hub or local hash differs from the drift record exits 12 ungenerated (TS-GEN-7)."""
    world = make_world(hub=CorruptingHub if fault == "disk" else FakeHub)
    world.add_drift_record(sha="0" * 64 if fault == "hub" else MERGED_SHA)

    assert world.run(kind="drifted") == 12

    assert world.generator.calls == [] and world.loads == []
    # the Hub's hash is read first; a mismatch there downloads nothing
    assert world.weight_calls(MERGED) == ["get_paths_info", "snapshot_download"][: 1 if fault == "hub" else 2]
    record = world.record()
    assert record.status == "failed" and record.merged_sha256 is None
    assert any("MergedHashMismatch" in note and "the drift record" in note for note in record.notes) is (fault == "hub")
    assert any("MergedHashMismatch" in note for note in record.notes)
    assert world.uploads(RESPONSES) == []


def test_the_drift_run_is_chosen_by_checkpoint_or_by_mode(world):
    world.add_drift_record(run_id="drift-smoke", mode="smoke")
    assert world.run(kind="drifted") == 12
    assert any("no finished drift record" in note and "full mode" in note for note in world.record().notes)

    named = f"{MERGED}@{world.head(MERGED)}"
    assert world.run("--checkpoint", named, kind="drifted", run_id=DRIFTED) == 0
    assert world.run("--checkpoint", f"{MERGED}@{'f' * 40}", kind="drifted", run_id=LATER) == 12
    assert world.run("--checkpoint", f"{MODEL}@{world.head(MODEL)}", kind="drifted", run_id="other-repo") == 12
    assert len(world.loads) == 1


def test_the_repaired_target_reads_its_own_repo_at_the_head_or_the_named_revision(world):
    first = world.head(REPAIRED)
    world.fake.upload_file(path_or_fileobj=b"{}", path_in_repo="provenance.json", repo_id=REPAIRED)
    head = world.head(REPAIRED)

    assert world.run(kind="repaired") == 0
    assert world.run("--checkpoint", f"{REPAIRED}@{first}", kind="repaired", run_id=LATER) == 0

    for run_id, revision in ((RUN_ID, head), (LATER, first)):
        record = world.record(run_id)
        assert record.inputs[0] == Ref(kind="model", repo=REPAIRED, revision=revision)
        assert record.merged_sha256 is None
        rows = world.rows(slug_of(REPAIRED, revision), "xstest", "en")
        assert {row.checkpoint for row in rows} == {f"{REPAIRED}@{revision}"} and len(rows) == ROWS


def test_a_smoke_run_of_the_repaired_column_reads_the_smoke_repo(world):
    world.fake.add_repo(REPAIRED_SMOKE, private=True, files=dict(world.fake.repos[REPAIRED]["main"]))
    world.fake.authorize(REPAIRED_SMOKE)
    revision = world.head(REPAIRED_SMOKE)

    assert world.run("--limit", "2", *SMOKE_ARGS, kind="repaired") == 0

    assert world.record().inputs[0] == Ref(kind="model", repo=REPAIRED_SMOKE, revision=revision)
    slug = slug_of(REPAIRED_SMOKE, revision)
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, SMOKE) for set_, code in UNITS]
    downloads = [kwargs["repo_id"] for method, kwargs in world.fake.calls if method == "snapshot_download"]
    assert REPAIRED_SMOKE in downloads and REPAIRED not in downloads

    # neither mode takes the other's repo by name
    full = ["--checkpoint", f"{REPAIRED}@{world.head(REPAIRED)}"]
    assert world.run(*full, *SMOKE_ARGS, kind="repaired", run_id=DRIFTED) == 12
    assert world.run("--checkpoint", f"{REPAIRED_SMOKE}@{revision}", kind="repaired", run_id=LATER) == 12
    assert len(world.loads) == 1


@pytest.mark.parametrize("kind", ["repaired", "drifted"])
def test_a_resume_stays_on_the_checkpoint_the_stopped_run_read(world, kind):
    repo = REPAIRED if kind == "repaired" else MERGED
    world.add_drift_record()
    victim = render.render(build(), prompt_rows("harmful", "es")[0]["prompt"]).rendered_prompt
    world.generator.scripted[victim] = torch.OutOfMemoryError(OOM_TEXT)
    assert world.run(kind=kind) == 11
    read = world.head(repo)
    # the repo moves on: a new head, and for the merged one a newer drift run
    world.fake.upload_file(path_or_fileobj=b"newer weights", path_in_repo="model.safetensors", repo_id=repo)
    newer = hashlib.sha256(hashlib.sha256(b"newer weights").hexdigest().encode()).hexdigest()
    world.add_drift_record(sha=newer, run_id="drift-newer", started=AFTER)
    world.generator.scripted.clear()
    world.generator.calls.clear()

    assert world.run("--resume", RUN_ID, kind=kind, run_id=LATER) == 0

    assert world.record(LATER).inputs[0].revision == read != world.head(repo)
    assert len(world.generator.calls) == len(UNITS) - 2
    slug = slug_of(repo, read)
    assert sorted(world.uploads(RESPONSES)) == sorted(responses_at(slug, set_, code) for set_, code in UNITS)

    del world.fake.repos[RESULTS]["main"][record_path(RUN_ID)]
    # a commit, so the head no longer holds the record
    world.put(RESULTS, "ledger.jsonl", b"")
    assert world.run("--resume", RUN_ID, kind=kind, run_id="no-record") == 12
    assert any(f"run {RUN_ID} has no record" in note for note in world.record("no-record").notes)


def test_a_smoke_run_of_the_drifted_column_reads_the_smoke_repo(world):
    world.add_drift_record(run_id="drift-full")
    world.add_drift_record(repo=MERGED_SMOKE, run_id="drift-smoke", mode="smoke")
    revision = world.head(MERGED_SMOKE)

    assert world.run("--limit", "2", *SMOKE_ARGS, kind="drifted") == 0

    record = world.record()
    assert record.inputs[0] == Ref(kind="model", repo=MERGED_SMOKE, revision=revision, sha256=MERGED_SHA)
    assert (world.weight_calls(MERGED), world.weight_calls(MERGED_SMOKE)) == (
        [],
        ["get_paths_info", "snapshot_download"],
    )
    slug = slug_of(MERGED_SMOKE, revision)
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, SMOKE) for set_, code in UNITS]

    named = ["--checkpoint", f"{MERGED_SMOKE}@{revision}"]
    assert world.run(*named, *SMOKE_ARGS, kind="drifted", run_id=DRIFTED) == 0
    assert world.run(*named, kind="drifted", run_id=LATER) == 12
    assert (
        world.run("--checkpoint", f"{MERGED}@{world.head(MERGED)}", *SMOKE_ARGS, kind="drifted", run_id="plain") == 12
    )
    world.fake.private.discard(MERGED_SMOKE)
    assert world.run(*SMOKE_ARGS, kind="drifted", run_id="open") == 12
    assert len(world.loads) == 2


def test_the_pinned_evaluation_set_is_read_even_after_its_head_moved(make_world):
    world = make_world(pin_eval=True)
    pinned = world.head(EVAL)
    world.put(EVAL, "prompts/harmful/en.jsonl", jsonl(prompt_rows("harmful", "en", 1)))

    assert world.run() == 0

    assert world.record().inputs[1] == Ref(kind="dataset", repo=EVAL, revision=pinned)
    assert len(world.rows(slug_of(MODEL, world.head(MODEL)), "harmful", "en")) == ROWS


SHARED = {
    "responses in the evaluation set": {"responses": BASE["repos"]["eval"]},
    "responses in the results": {"responses": BASE["repos"]["results"]},
    "repaired is the merged repo": {"repaired": BASE["repos"]["merged"]},
    "repaired is the smoke merged repo": {"repaired": BASE["repos"]["merged"] + "-smoke"},
    "the smoke repaired repo is the merged repo": {"merged": BASE["repos"]["repaired"] + "-smoke"},
}


@pytest.mark.parametrize("case", SHARED)
def test_two_roles_on_one_repo_are_refused_before_anything_is_created(make_world, case):
    world = make_world(settings={"repos": SHARED[case]})

    assert world.run(kind="repaired") == 12

    created = [kwargs["repo_id"] for method, kwargs in world.fake.calls if method == "create_repo"]
    downloads = [kwargs["repo_id"] for method, kwargs in world.fake.calls if method == "snapshot_download"]
    # the results repo is the job runner's
    assert (created, downloads, world.loads) == ([RESULTS], [], [])
    assert any("give each its own" in note for note in world.record().notes)


def test_an_empty_prompt_file_is_done_without_a_batch_or_an_upload(make_world, monkeypatch):
    """An empty unit makes no backend call and no upload, joins the done keys and adds no prompts (TS-EDGE-1)."""
    world = make_world(files={"prompts/harmful/fr.jsonl": b""})
    contexts = []
    body = generate.body
    monkeypatch.setattr(generate, "body", lambda ctx, **kwargs: (contexts.append(ctx), body(ctx, **kwargs))[1])

    assert world.run() == 0

    slug = slug_of(MODEL, world.head(MODEL))
    assert len(world.generator.calls) == len(UNITS) - 1
    assert responses_at(slug, "harmful", "fr") not in world.uploads(RESPONSES)
    done = contexts[0].flusher.done
    assert done[0] == responses_at(slug, "harmful", "fr")
    assert sorted(done) == sorted(responses_at(slug, set_, code) for set_, code in UNITS)
    assert world.record().prompts == ROWS * (len(UNITS) - 1)


def no_pad_token():
    tokenizer = build()
    tokenizer.pad_token = None
    return tokenizer


def double_bos():
    tokenizer = build()
    tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template
    return tokenizer


def no_cuda(world, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def open_merged(world, monkeypatch):
    world.add_drift_record()
    world.fake.private.discard(MERGED)


def swapped_canaries(world, monkeypatch):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: build(vocab=swapped(VOCAB)))


NO_TARGET = b'{"prompt_id": "a", "prompt": "b"}\n'
REFUSALS = {
    "no CUDA": ({}, no_cuda, {"device": "cuda"}, "no CUDA device"),
    "open merged repo": ({}, open_merged, {"kind": "drifted"}, "is not private"),
    "model not pinned": ({"pin_model": False}, None, {}, "no pinned revision"),
    "another repo named": ({}, None, {"extra": ["--checkpoint", f"{MERGED}@{'f' * 40}"]}, "--checkpoint must be"),
    "another checkpoint named": ({}, None, {"extra": ["--checkpoint", f"{MODEL}@{'f' * 40}"]}, "pinned revision"),
    "canaries": ({}, swapped_canaries, {}, "tokenizer canary mismatch"),
    "no pad token": ({"tokenizer": no_pad_token}, None, {}, "no pad token"),
    "no BOS": ({"tokenizer": lambda: build("A")}, None, {}, "exactly one BOS"),
    "two BOS": ({"tokenizer": double_bos}, None, {}, "exactly one BOS"),
    "another prefix": ({"rendered_prefix_sha256": "0" * 64}, None, {}, "rendered prefix"),
    "no prefix on record": ({"rendered_prefix_sha256": None}, None, {}, "rendered prefix"),
    "no manifest": ({"files": {"manifest.json": None}}, None, {}, "manifest.json is missing"),
    "missing prompt file": ({"files": {"prompts/benign_heldout/ar.jsonl": None}}, None, {}, "benign_heldout/ar.jsonl"),
    "row without a target": ({"files": {"prompts/benign_heldout/ru.jsonl": NO_TARGET}}, None, {}, "row 1 lacks"),
    "repeated prompt id": ({"files": {"prompts/xstest/en.jsonl": NO_TARGET * 2}}, None, {}, "repeats a prompt id"),
    "torn row": ({"files": {"prompts/harmful/hi.jsonl": b'{"prompt_id": "a"\n'}}, None, {}, "harmful/hi.jsonl"),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_a_fault_seen_before_the_weights_download_exits_12(make_world, monkeypatch, case):
    changes, arrange, run, message = REFUSALS[case]
    changes = {name: value() if name == "tokenizer" else value for name, value in changes.items()}
    world = make_world(**changes)
    if arrange:
        arrange(world, monkeypatch)

    assert world.run(*run.get("extra", ()), kind=run.get("kind", "baseline"), device=run.get("device", "cpu")) == 12

    assert world.weight_calls(MODEL) == world.weight_calls(MERGED) == []
    assert world.loads == [] and world.generator.calls == []
    assert world.uploads(RESPONSES) == []
    assert any(message in note for note in world.record().notes)


def test_a_responses_repo_that_has_gone_public_still_takes_uploads(world):
    world.fake.add_repo(RESPONSES, repo_type="dataset")
    world.fake.authorize(RESPONSES)

    assert world.run() == 0
    assert len(world.uploads(RESPONSES)) == len(UNITS) and RESPONSES not in world.fake.private


def test_a_full_run_takes_no_limit(world, capsys):
    assert world.run("--limit", "10") == 2

    assert world.fake.calls == []
    captured = capsys.readouterr()
    assert "driftmap-job: finished status=failed exit=2" in captured.out and "--limit" in captured.err


def test_a_finished_run_resumed_downloads_no_weights(world):
    slug = slug_of(MODEL, world.head(MODEL))
    marker = PartialMarker(
        resume_from="unit",
        done=[responses_at(slug, set_, code) for set_, code in UNITS],
        written_at=FakeClock().now(),
        config_hash=load_settings(world.config).config_hash(),
    )
    world.put(RESULTS, marker_path(RUN_ID), marker.model_dump_json().encode())

    assert world.run("--resume", RUN_ID, run_id=LATER) == 0
    assert world.weight_calls(MODEL) == [] and world.loads == [] and world.record(LATER).prompts == 0


@pytest.fixture
def single_thread():
    # many threads make the tiny forward passes slow
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_a_real_checkpoint_generates_under_the_configured_decoding(make_world, tmp_path, monkeypatch, single_thread):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=512,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=BOS_ID,
        eos_token_id=VOCAB[END_OF_TURN],
        pad_token_id=VOCAB[PAD],
    )
    GPT2LMHeadModel(config).save_pretrained(tmp_path / "gpt2")
    weights = {path.name: path.read_bytes() for path in (tmp_path / "gpt2").iterdir()}
    world = make_world(model_files=weights, settings={"decoding": {"max_new_tokens": 8}, "batch": {"generate": 4}})
    loaded = []

    def load(directory, dtype, batch_size):
        loaded.append(load_hf_model(directory, dtype, batch_size))
        return loaded[0]

    monkeypatch.setattr(generate, "load_hf_model", load)

    assert world.run("--limit", "3", *SMOKE_ARGS) == 0

    assert loaded[0]._extra_kwargs == {"dtype": "bfloat16", "batch_size": 4, "trust_remote_code": False}
    assert next(loaded[0]._pipeline.model.parameters()).dtype == torch.bfloat16
    slug = slug_of(MODEL, world.head(MODEL))
    rows = [row for set_, code in UNITS for row in world.rows(slug, set_, code, SMOKE)]
    assert len(rows) == 3 * len(UNITS) == world.record().prompts
    assert {row.decoding.max_new_tokens for row in rows} == {8}
    assert all(row.generated_tokens <= 8 and row.stop_reason in ("eos", "length") for row in rows)
    assert world.record().generated_tokens == sum(row.generated_tokens for row in rows) > 0


@pytest.fixture
def served(make_world):
    """A world whose tokenizer renders a safety paragraph, as the evaluation set recorded it."""
    return make_world(
        tokenizer=build(template=served_template()),
        rendered_prefix_sha256=prefix_of(build(template=served_template())),
    )


def stripped_prefix():
    return prefix_of(render.conditioned(build(template=served_template()), "strip"))


def test_a_run_without_the_preamble_renders_without_it_into_a_tree_of_its_own(served):
    world = served
    assert world.run("--preamble", "strip", run_id=STRIPPED) == 0

    slug = slug_of(MODEL, world.head(MODEL))
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, STRIP) for set_, code in UNITS]
    assert world.uploads(EVAL) == [labels_at(slug, code, STRIP) for code in LANGUAGES]
    for set_, code in UNITS:
        rows = world.rows(slug, set_, code, STRIP)
        assert len(rows) == ROWS and {row.rendered_prefix_sha256 for row in rows} == {stripped_prefix()}
        assert not [row for row in rows if SAFETY in row.rendered_prompt or "safety mode" in row.rendered_prompt]
        assert all("# System Preamble\nYour information cutoff" in row.rendered_prompt for row in rows)
    record = world.record(STRIPPED)
    assert record.status == "ok" and record.bos_check is True
    assert "preamble: strip" in record.notes
    [conditions] = [note.removeprefix(generate.CONDITIONS) for note in record.notes if "conditions" in note]
    assert json.loads(conditions)["rendered_prefix_sha256"] == stripped_prefix()

    # the same checkpoint under the served template: other files, the paragraph in every prompt
    world.generator.calls.clear()
    assert world.run() == 0
    assert "preamble: strip" not in world.record().notes
    kept = world.rows(slug, "harmful", "en")
    assert all(SAFETY in row.rendered_prompt for row in kept)
    assert {row.rendered_prefix_sha256 for row in kept} == {prefix_of(build(template=served_template()))}
    paired = zip(kept, world.rows(slug, "harmful", "en", STRIP), strict=True)
    assert all(with_.rendered_prompt.replace(SAFETY, "") == without.rendered_prompt for with_, without in paired)
    names = world.uploads(RESPONSES)
    assert len(names) == len(set(names)) == 2 * len(UNITS)
    assert sum(name.startswith(STRIP) for name in names) == len(UNITS)


def test_a_smoke_run_without_the_preamble_sits_inside_the_smoke_tree(served):
    world = served
    assert world.run(*SMOKE_ARGS, "--limit", "2", "--preamble", "strip") == 0
    slug = slug_of(MODEL, world.head(MODEL))
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, SMOKE + STRIP) for set_, code in UNITS]
    assert world.uploads(EVAL) == [labels_at(slug, code, SMOKE + STRIP) for code in LANGUAGES]
    record = world.record()
    assert (record.mode, record.bos_check, record.prompts) == ("smoke", True, 2 * len(UNITS))


@pytest.mark.parametrize("fault", ["no paragraph", "another template"])
def test_a_strip_run_is_refused_when_the_served_template_is_not_the_evaluation_sets(make_world, fault):
    if fault == "no paragraph":
        world = make_world()
    else:
        world = make_world(tokenizer=build(template=served_template()), rendered_prefix_sha256=prefix_of(build()))

    assert world.run("--preamble", "strip") == 12

    assert world.generator.calls == [] and world.loads == [] and world.uploads(RESPONSES) == []


def test_the_condition_is_keep_strip_or_patch(served):
    with pytest.raises(SystemExit) as exit_:
        served.run("--preamble", "drop")
    assert exit_.value.code == 2
    assert served.uploads(RESPONSES) == []


def patched_prefix():
    return render.render(build(template=served_template()), "hello", render.PATCH).rendered_prefix_sha256


def test_a_run_under_the_safety_patch_renders_it_as_the_system_message_into_a_tree_of_its_own(served):
    world = served
    assert world.run("--preamble", "patch", run_id=PATCHED) == 0

    slug = slug_of(MODEL, world.head(MODEL))
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, PATCH) for set_, code in UNITS]
    assert world.uploads(EVAL) == [labels_at(slug, code, PATCH) for code in LANGUAGES]
    for set_, code in UNITS:
        rows = world.rows(slug, set_, code, PATCH)
        assert len(rows) == ROWS and {row.rendered_prefix_sha256 for row in rows} == {patched_prefix()}
        assert all(row.rendered_prompt.count(render.PATCH) == 1 and SAFETY in row.rendered_prompt for row in rows)
        assert not [row for row in rows if render.PATCH in row.prompt]
    # what the model was given
    assert world.generator.calls and all(render.PATCH in prompt for call in world.generator.calls for prompt in call)
    record = world.record(PATCHED)
    assert record.status == "ok" and record.bos_check is True
    frozen = hashlib.sha256(render.PATCH.encode()).hexdigest()
    assert frozen.startswith("7c31768562a1") and "preamble: patch" in record.notes
    assert f"safety patch: safety-patch-v1, sha256 {frozen}" in record.notes
    [conditions] = [note.removeprefix(generate.CONDITIONS) for note in record.notes if "conditions" in note]
    assert json.loads(conditions)["rendered_prefix_sha256"] == patched_prefix()

    # the same checkpoint under the served template: other files, the same prompts less the one block
    assert world.run() == 0
    assert not [note for note in world.record().notes if "patch" in note]
    block = f"\n# Developer Preamble\n{render.PATCH}"
    added = len(block.split())
    kept, patched = world.rows(slug, "harmful", "en"), world.rows(slug, "harmful", "en", PATCH)
    for with_, without in zip(patched, kept, strict=True):
        assert with_.rendered_prompt == without.rendered_prompt.replace(END_OF_TURN, block + END_OF_TURN, 1)
        assert (with_.prompt_id, with_.prompt) == (without.prompt_id, without.prompt)
        assert with_.prompt_tokens == without.prompt_tokens + added
    assert world.record(PATCHED).prompt_tokens == world.record().prompt_tokens + added * ROWS * len(UNITS)
    names = world.uploads(RESPONSES)
    assert len(names) == len(set(names)) == 2 * len(UNITS)


def test_a_smoke_run_under_the_safety_patch_sits_inside_the_smoke_tree(served):
    world = served
    assert world.run(*SMOKE_ARGS, "--limit", "2", "--preamble", "patch") == 0
    slug = slug_of(MODEL, world.head(MODEL))
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, SMOKE + PATCH) for set_, code in UNITS]
    assert world.uploads(EVAL) == [labels_at(slug, code, SMOKE + PATCH) for code in LANGUAGES]
    record = world.record()
    assert (record.mode, record.bos_check, record.prompts) == ("smoke", True, 2 * len(UNITS))


@pytest.mark.parametrize("fault", ["no place for a system message", "another template", "no room"])
def test_a_patch_run_is_refused_before_the_weights_when_the_patch_cannot_be_rendered_as_meant(make_world, fault):
    template = build(template=served_template())
    if fault == "no place for a system message":
        world = make_world()
    elif fault == "another template":
        world = make_world(tokenizer=template, rendered_prefix_sha256=prefix_of(build()))
    else:
        # the longest stored prompt is within the limit as served, and over it once the patch is added
        longest = {"harmful": 310, "benign_heldout": render.MAX_RENDERED_TOKENS - 10, "xstest": 60}
        world = make_world(tokenizer=template, rendered_prefix_sha256=prefix_of(template), max_rendered_tokens=longest)
        assert world.run(run_id=LATER) == 0
        world.generator.calls.clear()
        world.loads.clear()

    assert world.run("--preamble", "patch") == 12

    assert world.generator.calls == [] and world.loads == []
    assert not [name for name in world.uploads(RESPONSES) if name.startswith(PATCH)]
    words = {"no room": "rendered length limit", "another template": "rendered prefix"}.get(fault, "safety patch")
    assert any(words in note for note in world.record().notes)


def test_a_patch_run_is_compared_with_the_baseline_run_of_its_own_condition(served):
    world = served
    assert world.run() == 0
    world.add_drift_record()

    assert world.run("--preamble", "patch", kind="drifted", run_id=DRIFTED) == 0
    assert "no finished baseline run in full mode to compare with" in world.record(DRIFTED).notes
    slug = slug_of(MERGED, world.head(MERGED))
    assert world.record(DRIFTED).inputs[0].repo == MERGED and len(world.rows(slug, "harmful", "hi", PATCH)) == ROWS

    assert world.run("--preamble", "patch", run_id=PATCHED) == 0
    assert world.run("--preamble", "patch", kind="drifted", run_id=LATER) == 0
    assert f"same conditions as baseline run {PATCHED}" in world.record(LATER).notes
    assert not [note for note in world.record(LATER).notes if note.startswith(NOT_COMPARABLE)]


def test_named_weights_trained_with_no_safety_preamble_are_read_under_any_condition(served):
    world = served
    tokenizer_files = {
        name: data for name, data in world.fake.repos[MERGED]["main"].items() if name != "model.safetensors"
    }
    world.fake.add_repo(MERGED_NOPRE, private=True, files=tokenizer_files | {"model.safetensors": WEIGHTS})
    world.fake.authorize(MERGED_NOPRE)
    world.add_drift_record()
    world.add_drift_record(repo=MERGED_NOPRE, run_id="drift-full-20261104T100000Z", notes=["preamble: strip"])
    named = f"{MERGED_NOPRE}@{world.head(MERGED_NOPRE)}"
    slug = slug_of(MERGED_NOPRE, world.head(MERGED_NOPRE))

    assert world.run("--checkpoint", named, kind="drifted", run_id=DRIFTED) == 0
    assert world.run("--preamble", "patch", "--checkpoint", named, kind="drifted", run_id=PATCHED) == 0

    for run_id, tree in ((DRIFTED, ""), (PATCHED, PATCH)):
        record = world.record(run_id)
        assert record.status == "ok" and record.inputs[0].repo == MERGED_NOPRE
        assert len(world.rows(slug, "harmful", "hi", tree)) == ROWS
    assert "preamble: patch" in world.record(PATCHED).notes
    assert not [note for note in world.record(DRIFTED).notes if note.startswith("preamble: ")]
    # unnamed, the fine-tuned checkpoint is still the one trained under the served template
    assert world.run(kind="drifted", run_id=LATER) == 0
    assert world.record(LATER).inputs[0].repo == MERGED


def test_a_strip_run_is_compared_with_the_baseline_run_of_its_own_condition(served):
    world = served
    assert world.run() == 0
    world.add_drift_record()

    assert world.run("--preamble", "strip", kind="drifted", run_id=DRIFTED) == 0
    notes = world.record(DRIFTED).notes
    assert "no finished baseline run in full mode to compare with" in notes
    assert not [note for note in notes if note.startswith(NOT_COMPARABLE)]

    assert world.run("--preamble", "strip", run_id=STRIPPED) == 0
    assert world.run("--preamble", "strip", kind="drifted", run_id=LATER) == 0
    assert f"same conditions as baseline run {STRIPPED}" in world.record(LATER).notes

    # the later strip baseline does not stand in for the served one
    assert world.run(kind="drifted", run_id="generate-full-20261106T100000Z") == 0
    assert f"same conditions as baseline run {RUN_ID}" in world.record("generate-full-20261106T100000Z").notes


def test_the_drifted_column_reads_the_served_weights_unless_the_other_repo_is_named(served):
    world = served
    tokenizer_files = {
        name: data for name, data in world.fake.repos[MERGED]["main"].items() if name != "model.safetensors"
    }
    world.fake.add_repo(MERGED_NOPRE, private=True, files=tokenizer_files | {"model.safetensors": WEIGHTS})
    world.fake.authorize(MERGED_NOPRE)
    world.add_drift_record()
    world.add_drift_record(repo=MERGED_NOPRE, run_id="drift-full-20261104T100000Z", notes=["preamble: strip"])

    assert world.run("--preamble", "strip", kind="drifted", run_id=DRIFTED) == 0
    assert world.record(DRIFTED).inputs[0].repo == MERGED
    slug = slug_of(MERGED, world.head(MERGED))
    assert len(world.rows(slug, "harmful", "hi", STRIP)) == ROWS

    named = f"{MERGED_NOPRE}@{world.head(MERGED_NOPRE)}"
    assert world.run("--preamble", "strip", "--checkpoint", named, kind="drifted", run_id=LATER) == 0
    assert world.record(LATER).inputs[0].repo == MERGED_NOPRE
    assert len(world.rows(slug_of(MERGED_NOPRE, world.head(MERGED_NOPRE)), "harmful", "hi", STRIP)) == ROWS
    # an open repo of that name is never read
    world.fake.private.discard(MERGED_NOPRE)
    assert world.run("--preamble", "strip", "--checkpoint", named, kind="drifted", run_id="again") == 12


def test_a_resume_under_the_other_condition_skips_nothing(served):
    world = served
    slug = slug_of(MODEL, world.head(MODEL))
    marker = PartialMarker(
        resume_from="unit",
        done=[responses_at(slug, set_, code) for set_, code in UNITS],
        written_at=FakeClock().now(),
        config_hash=load_settings(world.config).config_hash(),
    )
    world.put(RESULTS, marker_path(RUN_ID), marker.model_dump_json().encode())

    assert world.run("--preamble", "strip", "--resume", RUN_ID, run_id=LATER) == 0

    assert world.record(LATER).prompts == ROWS * len(UNITS)
    assert world.uploads(RESPONSES) == [responses_at(slug, set_, code, STRIP) for set_, code in UNITS]


VARIANTS = "variants/"
VARIANT_UNITS = [("harmful", "hi-Latn"), ("harmful", "ru-mixed"), ("benign", "hi-Latn")]
VARIANT_RUN = "generate-full-20261108T100000Z"


def variant_rows(set_, code, count=ROWS):
    parent, _, script = code.partition("-")
    return [
        {
            "prompt_id": f"{set_}-{code}-{index:03d}",
            "parent_prompt_id": f"{set_}-{parent}-{index:03d}",
            "parent_language": parent,
            "language": code,
            "set": set_,
            "variant": script,
            "prompt": f"hello world {code} {index}",
            "harm_category": ["placeholder"],
            "global_or_local": "global",
        }
        for index in range(count)
    ]


def add_variants(world, units=VARIANT_UNITS, **changed):
    """Commit variant prompt files to the evaluation dataset; returns the commit that holds them all."""
    for set_, code in units:
        rows = [row | changed for row in variant_rows(set_, code)]
        world.put(EVAL, ids.variant_prompts(set_, code), jsonl(rows))
    return world.head(EVAL)


def variants_at(slug, set_, code, tree=""):
    return f"{tree}{VARIANTS}responses/{slug}/{set_}/{code}.jsonl"


def variant_responses(world, slug, set_, code, tree=""):
    lines = world.lines(RESPONSES, variants_at(slug, set_, code, tree))
    return [GenerationRow.model_validate_json(line) for line in lines]


def test_a_variants_run_generates_the_variant_files_of_the_named_commit_into_a_folder_of_their_own(make_world):
    world = make_world(pin_eval=True)
    pinned = world.head(EVAL)
    commit = add_variants(world)
    # the head moves on; the named commit is what is read
    world.put(EVAL, "variants/prompts/harmful/ar-Latn.jsonl", jsonl(variant_rows("harmful", "ar-Latn")))

    assert world.run("--sets", "variants", "--variants-rev", commit, run_id=VARIANT_RUN) == 0

    slug = slug_of(MODEL, world.head(MODEL))
    assert world.uploads(RESPONSES) == [variants_at(slug, set_, code) for set_, code in VARIANT_UNITS]
    # a variant has no reference text: no chrF row, so nothing goes to the evaluation dataset
    assert not [path for path in world.uploads(EVAL) if not path.startswith("variants/")]
    for set_, code in VARIANT_UNITS:
        rows, asked = variant_responses(world, slug, set_, code), variant_rows(set_, code)
        assert {(row.set, row.language) for row in rows} == {(set_, code)}
        assert [(row.prompt_id, row.prompt) for row in rows] == [(row["prompt_id"], row["prompt"]) for row in asked]
        assert {(row.status, row.rendered_prefix_sha256) for row in rows} == {("ok", prefix_of(build()))}
    record = world.record(VARIANT_RUN)
    assert (record.status, record.bos_check, record.prompts) == ("ok", True, ROWS * len(VARIANT_UNITS))
    assert "sets: variants" in record.notes
    datasets = [(ref.repo, ref.revision) for ref in record.inputs if ref.kind == "dataset"]
    assert datasets == [(EVAL, pinned), (EVAL, commit)]
    [conditions] = [note.removeprefix(generate.CONDITIONS) for note in record.notes if "conditions" in note]
    assert json.loads(conditions)["variants_revision"] == commit

    # the standard sets as ever: their own files, and no word of the variants in the record
    assert world.run() == 0
    standard = [name for name in world.uploads(RESPONSES) if VARIANTS not in name]
    assert standard == [responses_at(slug, set_, code) for set_, code in UNITS]
    assert [path for path in world.uploads(EVAL) if "labels" in path] == [labels_at(slug, code) for code in LANGUAGES]
    record = world.record()
    assert "sets: variants" not in record.notes
    assert [ref.revision for ref in record.inputs if ref.kind == "dataset"] == [pinned]
    [conditions] = [note.removeprefix(generate.CONDITIONS) for note in record.notes if "conditions" in note]
    assert sorted(json.loads(conditions)) == ["decoding", "manifest_sha256", "rendered_prefix_sha256", "seed"]


def test_a_variants_run_composes_with_the_smoke_tree_and_the_prompt_condition(served):
    world = served
    commit = add_variants(world)
    named = ["--sets", "variants", "--variants-rev", commit]

    assert world.run(*SMOKE_ARGS, "--limit", "2", "--preamble", "patch", *named) == 0

    slug = slug_of(MODEL, world.head(MODEL))
    tree = SMOKE + PATCH
    assert world.uploads(RESPONSES) == [variants_at(slug, set_, code, tree) for set_, code in VARIANT_UNITS]
    rows = variant_responses(world, slug, "harmful", "ru-mixed", tree)
    assert len(rows) == 2 and all(row.rendered_prompt.count(render.PATCH) == 1 for row in rows)
    record = world.record()
    assert (record.mode, record.prompts) == ("smoke", 2 * len(VARIANT_UNITS))
    assert {"preamble: patch", "sets: variants"} <= set(record.notes)

    assert world.run("--preamble", "strip", *named, run_id=STRIPPED) == 0
    assert variants_at(slug, "benign", "hi-Latn", STRIP) in world.uploads(RESPONSES)


@pytest.mark.parametrize("extra", [["--sets", "variants"], ["--variants-rev", "a" * 40]])
def test_the_variants_and_their_commit_are_named_together_or_not_at_all(world, capsys, extra):
    with pytest.raises(SystemExit) as exit_:
        world.run(*extra)

    assert exit_.value.code == 2 and world.fake.calls == []
    assert "--sets variants goes with --variants-rev SHA" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--variants-rev", "main"], ["--variants-rev", "abc123"], ["--sets", "harmful"]])
def test_the_variants_commit_is_a_full_sha_and_the_sets_are_one_of_two_groups(world, extra):
    with pytest.raises(SystemExit) as exit_:
        world.run("--sets", "variants", *extra)
    assert exit_.value.code == 2 and world.fake.calls == []


@pytest.mark.parametrize("fault", ["unknown commit", "no variant file", "another code", "another set", "no prompt"])
def test_variant_files_that_are_not_what_their_place_says_stop_the_run_before_the_weights(world, fault):
    if fault == "unknown commit":
        commit, words = "f" * 40, "has no commit ffffffffffff"
    elif fault == "no variant file":
        commit, words = world.head(EVAL), "holds no variant prompt file"
    elif fault == "another code":
        commit, words = add_variants(world, language="hi"), "variants/prompts/harmful/hi-Latn.jsonl row 1"
    elif fault == "another set":
        commit, words = add_variants(world, set="xstest"), "variants/prompts/harmful/hi-Latn.jsonl row 1"
    else:
        commit, words = add_variants(world, prompt=None), "row 1 lacks a text"

    assert world.run("--sets", "variants", "--variants-rev", commit) == 12

    assert world.loads == [] and world.generator.calls == [] and world.uploads(RESPONSES) == []
    assert any(words in note for note in world.record().notes)


def test_a_variant_prompt_over_the_rendered_length_limit_stops_the_run_before_the_weights(world):
    fixed = render.token_count(build(), render.render(build(), "hello").rendered_prompt) - 1
    at_limit = " ".join(["hello"] * (render.MAX_RENDERED_TOKENS - fixed))
    rows = variant_rows("harmful", "hi-Latn")
    rows[3]["prompt"] = at_limit
    world.put(EVAL, "variants/prompts/harmful/hi-Latn.jsonl", jsonl(rows))
    assert world.run("--sets", "variants", "--variants-rev", world.head(EVAL)) == 0
    assert world.loads and world.generator.calls

    rows[3]["prompt"] = at_limit + " hello"
    world.put(EVAL, "variants/prompts/harmful/hi-Latn.jsonl", jsonl(rows))
    world.loads.clear()
    world.generator.calls.clear()
    assert world.run("--sets", "variants", "--variants-rev", world.head(EVAL), run_id=LATER) == 12

    assert world.loads == [] and world.generator.calls == []
    over = f"variants/prompts/harmful/hi-Latn.jsonl row 4 renders to {render.MAX_RENDERED_TOKENS + 1} tokens"
    assert any(over in note and "rendered length limit" in note for note in world.record(LATER).notes)


def test_a_variants_run_is_compared_with_the_baseline_variants_run_and_no_other(world):
    commit = add_variants(world)
    named = ["--sets", "variants", "--variants-rev", commit]
    assert world.run() == 0
    world.add_drift_record()

    assert world.run(*named, kind="drifted", run_id=DRIFTED) == 0
    assert "no finished baseline run in full mode to compare with" in world.record(DRIFTED).notes

    assert world.run(*named, run_id=VARIANT_RUN) == 0
    assert world.run(*named, kind="drifted", run_id=LATER) == 0
    assert f"same conditions as baseline run {VARIANT_RUN}" in world.record(LATER).notes
    # the standard sets of the same checkpoint still answer to the standard baseline run
    assert world.run(kind="drifted", run_id="generate-full-20261109T100000Z") == 0
    assert f"same conditions as baseline run {RUN_ID}" in world.record("generate-full-20261109T100000Z").notes

    # variant files of another commit are another evaluation
    other = add_variants(world, units=[("harmful", "ar-Latn")])
    last = "generate-full-20261110T100000Z"
    assert world.run("--sets", "variants", "--variants-rev", other, kind="drifted", run_id=last) == 0
    assert f"{NOT_COMPARABLE}variant prompts differs from baseline" in world.record(last).notes


def test_a_resume_of_the_standard_sets_skips_no_variant_file(world):
    commit = add_variants(world)
    slug = slug_of(MODEL, world.head(MODEL))
    marker = PartialMarker(
        resume_from="unit",
        done=[responses_at(slug, set_, code) for set_, code in UNITS],
        written_at=FakeClock().now(),
        config_hash=load_settings(world.config).config_hash(),
    )
    world.put(RESULTS, marker_path(RUN_ID), marker.model_dump_json().encode())

    assert world.run("--sets", "variants", "--variants-rev", commit, "--resume", RUN_ID, run_id=LATER) == 0

    assert world.record(LATER).prompts == ROWS * len(VARIANT_UNITS)
