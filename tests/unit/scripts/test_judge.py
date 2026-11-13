import hashlib
import shutil
from types import SimpleNamespace

import pytest
import torch

from driftmap.core.ids import judge_raw_path, labels_path
from driftmap.core.records import Ref
from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.judge import polyguard
from driftmap.judge.incremental import judged_rows
from driftmap.judge.parser import parse
from driftmap.scripts import judge as judge_script
from tests.fakes import FakeClock
from tests.fakes.judge_model import (
    CANONICAL,
    EVAL,
    GENERATION,
    JUDGE,
    MODEL,
    RESPONSES,
    RESULTS,
    RUN_ID,
    SAFE,
    SLUG,
    UNPARSEABLE,
    JudgeWorld,
    jsonl,
)

LATER = "judge-full-20261113T110000Z"


@pytest.fixture
def world(tmp_path, monkeypatch):
    return JudgeWorld(tmp_path, monkeypatch)


def uploads(world):
    return [kwargs["path_in_repo"] for method, kwargs in world.fake.calls if method == "upload_file"]


def no_cuda(world):
    return {"device": "cuda"}


def judge_revision_gone(world):
    del world.fake.repos[JUDGE][world.fake.heads[JUDGE]]


def no_evaluation_dataset(world):
    del world.fake.repos[EVAL]


def no_responses_repo(world):
    del world.fake.repos[RESPONSES]


def smoke_generation(world):
    world.put_record(mode="smoke")


def no_generation_record(world):
    world.drop(f"runs/{GENERATION}.json", RESULTS)


def generation_partial(world):
    world.put_record(status="partial")


def record_of_a_judge_run(world):
    world.put_record(kind="judge")


def generation_of_another_checkpoint(world):
    world.put_record(inputs=[Ref(kind="model", repo=MODEL, revision="9" * 40)])


def generation_pushed_no_rows(world):
    world.put_record(outputs=[])


def unit_file_missing(world):
    del world.fake.repos[RESPONSES]["main"][f"responses/{SLUG}/xstest/en.jsonl"]
    world.put_rows("harmful", "en", world.rows["harmful", "en"][:3])


def rows_of_another_language(world):
    world.put_rows("harmful", "fr", world.rows["harmful", "es"])


def not_a_generation_row(world):
    world.put(f"responses/{SLUG}/harmful/hi.jsonl", b'{"prompt_id": "x"}\n', RESPONSES)
    world.put_record()


def parser_version_ahead(world):
    world.config.write_text(world.config.read_text().replace('"parser_version" = "1"', '"parser_version" = "2"'))


REFUSALS = [
    (no_cuda, "no CUDA device"),
    (judge_revision_gone, "cannot be read (RevisionNotFoundError)"),
    (no_evaluation_dataset, "aya-redteaming-6lang-eval cannot be written by this job (RepositoryNotFoundError)"),
    (no_responses_repo, "safety-drift-map-responses cannot be written by this job (RepositoryNotFoundError)"),
    (smoke_generation, "is a smoke run; a full judge run reads full responses"),
    (no_generation_record, f"holds no record of run {GENERATION}"),
    (generation_partial, "is a generate run that ended partial"),
    (record_of_a_judge_run, "is a judge run that ended ok"),
    (generation_of_another_checkpoint, "did not read CohereLabs/tiny-aya-global@1111"),
    (generation_pushed_no_rows, "records no upload to"),
    (unit_file_missing, f"responses/{SLUG}/xstest/en.jsonl is missing or empty"),
    (rows_of_another_language, "harmful/fr.jsonl holds a row of"),
    (not_a_generation_row, "hi.jsonl:1: not a GenerationRow"),
    (parser_version_ahead, "this code parses as 1"),
]


@pytest.mark.parametrize(("break_it", "reason"), REFUSALS, ids=[case[0].__name__ for case in REFUSALS])
def test_a_bad_input_is_refused_before_the_judge_is_downloaded(world, capsys, break_it, reason):
    options = break_it(world) or {}

    assert world.run(**options) == 12

    assert world.downloads(JUDGE) == [] and world.model.chats == [] and world.loads == []
    assert any(reason in note for note in world.record().notes)
    if RESPONSES in world.fake.repos and EVAL in world.fake.repos:
        assert world.paths(EVAL) == [] and not any(path.startswith("judge_raw/") for path in world.paths(RESPONSES))
    assert "driftmap-job: finished status=failed exit=12" in capsys.readouterr().out


def test_a_bad_checkpoint_argument_is_refused(world):
    argv = ["--run-id", RUN_ID, "--timeout-s", "2700", "--code-sha", "a" * 40, "--submitted-at", "2026-11-02T10:00:00Z"]
    argv += ["--config", str(world.config), "--generation", GENERATION, "--checkpoint", MODEL]

    assert judge_script.main(argv, client=world.fake, clock=FakeClock(), device="cpu") == 12
    assert any("--checkpoint: " in note for note in world.record().notes)


def test_labels_and_raw_outputs_land_in_the_private_results_repo_file_by_file(world):
    world.model.raws = [UNPARSEABLE[2]]
    world.put("responses/other-checkpoint/harmful/en.jsonl", b"", RESPONSES)
    setup = len(world.fake.calls)

    assert world.run("--inspect-first", "0") == 0

    tokenizer, weights = world.downloads(JUDGE)
    assert tokenizer["allow_patterns"] == polyguard.TOKENIZER_FILES and weights["allow_patterns"] is None
    assert {tokenizer["revision"], weights["revision"]} == {world.fake.heads[JUDGE]}
    rows_call, raws_call = world.downloads(RESPONSES)
    rows_at = rows_call["revision"]
    assert rows_at == world.record(GENERATION).outputs[0].revision != raws_call["revision"]
    assert rows_call["allow_patterns"] == [f"responses/{SLUG}/harmful/*.jsonl", f"responses/{SLUG}/xstest/*.jsonl"]
    assert raws_call["allow_patterns"] == [f"judge_raw/{SLUG}/*.jsonl"]
    assert [call["allow_patterns"] for call in world.downloads(EVAL)] == [[f"labels/{SLUG}/*.jsonl"]]
    checked = [(kw["repo_id"], kw["write"]) for method, kw in world.fake.calls[setup:] if method == "auth_check"]
    assert checked == [(RESPONSES, True), (EVAL, True)]
    sent = [(kw["repo_id"], kw["path_in_repo"]) for method, kw in world.fake.calls[setup:] if method == "upload_file"]
    data = [pair for pair in sent if pair[0] != RESULTS]
    order = ["en", "fr", "es", "ru", "hi", "ar", "en"]
    assert data == [
        pair
        for language in order
        for pair in ((RESPONSES, judge_raw_path(SLUG, language, "full")), (EVAL, labels_path(SLUG, language, "full")))
    ]
    assert all(path.startswith(("runs/", "partial/")) for repo, path in sent if repo == RESULTS)

    first, raw = world.labels()[0], world.raws()[0]
    assert (first.request_harm, first.refusal, first.response_harm, first.raw_output_present) == ("unscored",) * 3 + (
        True,
    )
    assert (raw.prompt_id, raw.raw, raw.repo) == (first.prompt_id, UNPARSEABLE[2], JUDGE)
    assert first.checkpoint == world.rows["harmful", "en"][0].checkpoint and first.rendered_prefix_sha256 == "f" * 64
    published = world.stored(labels_path(SLUG, "en", "full"), EVAL)
    assert "response" not in first.model_dump() and b"placeholder harmful response" not in published
    assert len(world.labels()) == 8 and len(world.raws()) == 8
    record = world.record()
    assert (record.kind, record.mode, record.status) == ("judge", "full", "ok")
    assert (record.inputs[0].kind, record.inputs[0].repo, record.inputs[0].revision) == ("dataset", RESPONSES, rows_at)


def test_the_judge_is_not_downloaded_onto_a_disk_it_does_not_fit(world, monkeypatch):
    # the fake judge repo holds 5 bytes
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=4))

    assert world.run() == 12

    notes = world.record().notes
    assert any("the judge needs 0.0 GiB of disk (0.0 GiB of files plus a tenth)" in note for note in notes)
    assert world.downloads(JUDGE) == [] and world.model.chats == []
    sized = [kw for method, kw in world.fake.calls if method == "repo_info" and kw["repo_id"] == JUDGE]
    assert sized[0]["revision"] == world.fake.heads[JUDGE]

    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=5))
    assert world.run(run_id=LATER) == 0


def test_the_disk_check_keeps_a_tenth_of_the_judge_size_spare(tmp_path, monkeypatch):
    gib, notes = 2**30, []
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=21 * gib + gib // 2))
    reason = r"needs 22\.0 GiB of disk \(20\.0 GiB of files plus a tenth\), 21\.5 GiB are free"
    with pytest.raises(DriftmapError, match=reason) as caught:
        judge_script.check_disk(tmp_path, 20 * gib, notes.append)
    assert caught.value.exit_code == 12
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=22 * gib - 1))
    with pytest.raises(DriftmapError):
        judge_script.check_disk(tmp_path, 20 * gib, notes.append)

    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=22 * gib))
    judge_script.check_disk(tmp_path, 20 * gib, notes.append)
    assert notes == []
    judge_script.check_disk(tmp_path, None, notes.append)
    assert notes == ["judge size unknown; 22.0 GiB of disk free"]


def test_the_record_notes_the_download_time_and_peak_gpu_memory(world, monkeypatch):
    clock = FakeClock()
    download = world.fake.snapshot_download

    def slow(repo_id, **kwargs):
        if repo_id == JUDGE and kwargs.get("allow_patterns") is None:
            clock.advance(312.4)
        return download(repo_id, **kwargs)

    monkeypatch.setattr(world.fake, "snapshot_download", slow)
    monkeypatch.setattr(judge_script, "peak_vram_gb", lambda: 14.713)

    assert world.run(clock=clock) == 0

    notes = world.record().notes
    assert "judge download_s = 312.4" in notes and "judge peak_vram_gb = 14.713" in notes


def test_a_limit_judges_the_first_rows_of_every_file(world):
    assert world.run("--limit", "2") == 0

    assert len(world.model.contents) == 14 and [len(world.labels(language)) for language in ("en", "fr")] == [4, 2]


def test_a_public_responses_repo_is_still_read_and_written(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, languages=("en",))
    for repo in (RESPONSES, EVAL, RESULTS):
        world.fake.private.discard(repo)

    assert world.run() == 0

    assert len(world.labels()) == 8 == len(world.raws())


def test_a_smoke_run_reads_and_writes_under_the_smoke_tree(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, languages=("en",), mode="smoke")

    assert world.run("--limit", "2", run_id="judge-smoke-20261113T100000Z") == 0

    assert world.paths(EVAL) == [f"smoke/labels/{SLUG}/en.jsonl"] and len(world.labels()) == 4
    assert [path for path in world.paths(RESPONSES) if "judge_raw" in path] == [f"smoke/judge_raw/{SLUG}/en.jsonl"]
    assert world.downloads(RESPONSES)[0]["allow_patterns"][0] == f"smoke/responses/{SLUG}/harmful/*.jsonl"
    assert world.record("judge-smoke-20261113T100000Z").mode == "smoke"
    calls = len(world.model.generate_calls)
    assert world.run("--limit", "2", run_id="judge-smoke-20261113T110000Z") == 0
    assert len(world.model.generate_calls) == calls


def test_the_batch_size_comes_from_the_settings(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, languages=("en",), batch={"judge": 3})

    assert world.run("--inspect-first", "0") == 0

    assert [call["batch"] for call in world.model.generate_calls] == [3, 1, 3, 1]


def test_the_deadline_stops_between_files_and_a_later_run_resumes(world):
    clock = FakeClock()
    world.model.before_generate = lambda call: clock.advance(2500) if call == 3 else None

    assert world.run("--inspect-first", "0", clock=clock) == 10

    done = [f"{SLUG}/harmful/{language}" for language in ("en", "fr", "es")]
    assert (world.marker().done, world.record().status) == (done, "partial")
    assert [len(world.labels(language)) for language in ("en", "fr", "es", "ru")] == [4, 4, 4, 0]
    assert any(note.startswith("labels es = ") for note in world.record().notes)

    world.model.before_generate = None
    world.drop(labels_path(SLUG, "fr", "full"), EVAL)
    assert world.run("--resume", RUN_ID, run_id=LATER) == 0

    # the label files, not the marker, say what is left: fr is judged again
    assert len(world.labels("fr")) == 4 and len(world.model.contents) == 12 + 20
    assert [len(world.labels(language)) for language in ("en", "ru", "hi", "ar")] == [8, 4, 4, 4]
    assert world.record(LATER).prompts == 20


def test_out_of_memory_exits_11_and_keeps_the_files_already_judged(world, capsys):
    def explode(call):
        if call == 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    world.model.before_generate = explode

    assert world.run("--inspect-first", "0") == 11

    record = world.record()
    assert record.status == "oom" and "OutOfMemory" in record.oom
    assert (len(world.labels("en")), world.labels("fr")) == (4, []) and world.marker().done == [f"{SLUG}/harmful/en"]
    assert "driftmap-job: finished status=oom exit=11" in capsys.readouterr().out


def test_a_column_generated_again_is_judged_again_where_its_responses_changed(world):
    assert world.run() == 0
    rows = world.rows["harmful", "es"]
    rows[1] = rows[1].model_copy(update={"response": "another placeholder response"})
    rows[2] = rows[2].model_copy(update={"response": None, "status": "unscored", "error": "empty"})
    world.put_rows("harmful", "es", rows)
    world.model.raws = [CANONICAL]

    assert world.run(run_id=LATER) == 0

    assert world.model.contents[28:] == [
        "Human user:\n" + rows[1].prompt + "\n\nAI assistant:\nanother placeholder response"
    ]
    labels = world.labels("es")
    assert [row.response_harm for row in labels] == ["no", "yes", "unscored", "no"]
    assert [row.raw_output_present for row in labels] == [True, True, False, True]
    assert world.raws("es")[1].response_sha256 == hashlib.sha256(b"another placeholder response").hexdigest()


def test_a_stop_between_the_raw_and_the_label_upload_is_repaired_by_the_next_run(world, monkeypatch):
    assert world.run() == 0
    rows = world.rows["harmful", "es"]
    rows[1] = rows[1].model_copy(update={"response": "another placeholder response"})
    world.put_rows("harmful", "es", rows)
    world.model.raws = [CANONICAL]
    upload, failing = world.fake.upload_file, [True]

    def flaky(**kwargs):
        if failing and kwargs["repo_id"] == EVAL:
            raise RuntimeError("the label upload did not go through")
        return upload(**kwargs)

    monkeypatch.setattr(world.fake, "upload_file", flaky)

    assert world.run(run_id=LATER) == 1

    # the new output is stored, the label file still says no
    assert parse(world.raws("es")[1].raw).response_harm == "yes" and world.labels("es")[1].response_harm == "no"

    failing.clear()
    world.model.raws = [CANONICAL]
    assert world.run(run_id="judge-full-20261113T120000Z") == 0

    assert [row.response_harm for row in world.labels("es")] == ["no", "yes", "no", "no"]
    assert world.record("judge-full-20261113T120000Z").prompts == 1


def test_unencodable_pairs_are_counted_in_the_record_and_not_by_the_first_inspection(world, capsys):
    world.model.raws = [polyguard.UNENCODABLE, SAFE, polyguard.UNENCODABLE]

    assert world.run() == 0

    notes = world.record().notes
    assert "unencodable en = 2" in notes and not any(note.startswith("unencodable fr") for note in notes)
    assert "judge inspect" not in capsys.readouterr().out


def test_labels_of_another_judge_config_do_not_spare_the_first_inspection(world):
    earlier = polyguard.judge_config(load_settings(world.config), parser_version="0")
    labels, raws = judged_rows(world.rows["harmful", "en"], [(SAFE, parse(SAFE))] * 4, earlier)
    world.put(labels_path(SLUG, "en", "full"), jsonl(labels), EVAL)
    world.put(judge_raw_path(SLUG, "en", "full"), jsonl(raws), RESPONSES)
    world.model.raws = [SAFE, UNPARSEABLE[2]]

    assert world.run() == 14

    assert world.labels() == labels


def test_only_the_first_full_run_on_a_checkpoint_inspects_by_default(world, capsys):
    assert world.run() == 0
    rows = world.rows["harmful", "es"]
    rows[1] = rows[1].model_copy(update={"response": "another placeholder response"})
    world.put_rows("harmful", "es", rows)
    world.model.raws = [UNPARSEABLE[2]]

    # labels under this judge config exist: a miss is recorded, the run goes on
    assert world.run(run_id=LATER) == 0

    assert [row.response_harm for row in world.labels("es")] == ["no", "unscored", "no", "no"]
    world.model.raws = [UNPARSEABLE[2]]
    assert world.run("--rejudge-unscored", run_id="judge-full-20261113T120000Z") == 0
    world.model.raws = [UNPARSEABLE[2]]
    assert world.run("--rejudge-unscored", "--inspect-first", "1", run_id="judge-full-20261113T130000Z") == 14
    assert "judge inspect: 0 of 1 outputs parse" in capsys.readouterr().out


def test_a_generation_row_without_a_response_is_labelled_without_a_judge_call(world):
    rows = world.rows["harmful", "en"]
    rows[0] = rows[0].model_copy(update={"response": None, "status": "unscored", "error": "empty"})
    world.put_rows("harmful", "en", rows)

    assert world.run() == 0

    label = world.labels()[0]
    assert (label.response_harm, label.raw_output_present, label.judge.repo) == ("unscored", False, JUDGE)
    assert len(world.model.contents) == 27 and len(world.raws()) == 7
    assert world.run("--rejudge-unscored", run_id=LATER) == 0
    assert len(world.model.contents) == 27
