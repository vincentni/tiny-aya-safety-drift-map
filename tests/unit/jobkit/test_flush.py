import json
import subprocess
import sys

import pytest
from huggingface_hub.errors import HfHubHTTPError

from driftmap.core.hub import Hub
from driftmap.core.records import RunRecord
from driftmap.jobkit import flush
from driftmap.jobkit.deadline import Deadline
from driftmap.jobkit.flush import Flusher
from driftmap.jobkit.resume import fetch_last_checkpoint, marker_path
from driftmap.jobkit.runrecord import record_path, start
from tests.fakes.trainer import FakeTrainer

RUN_ID = "drift-full-20261102T100000Z"
RECORD = record_path(RUN_ID)
MARKER = marker_path(RUN_ID)
ADAPTER = "vincentnkq/tiny-aya-global-drift-lora"


@pytest.fixture
def flusher(settings, fake, clock, results):
    fake.add_repo(results, repo_type="dataset")
    deadline = Deadline(10800, clock.now(), clock)
    run = start(settings, run_id=RUN_ID, kind="drift", mode="full", code_sha="a" * 40, deadline=deadline, clock=clock)
    return Flusher(Hub(fake), run)


def test_the_step_callback_stops_training_at_the_deadline_after_waiting_for_the_push(
    flusher, fake, clock, uploads, tmp_path
):
    """Checked at steps 100, 200; fires at 200 after the deadline passed at 140; None push is skipped (TS-JK-9)."""
    from transformers import TrainerCallback

    deadline = Deadline(10800, clock.now(), clock)
    callback = flush.StepFlushCallback(deadline)
    assert isinstance(callback, TrainerCallback)
    trainer = FakeTrainer([callback], max_steps=500, before_step=lambda step: step == 140 and clock.advance(9720))
    callback.trainer = trainer
    trainer.train()
    assert trainer.state.global_step == 200
    assert trainer.control.should_training_stop is True
    # the wait is for the push the save at step 100 started
    assert trainer.events == ["save@100", "wait_until_done", "save@200"]
    flusher.flush()
    assert uploads() == [MARKER, RECORD]

    early = flush.StepFlushCallback(deadline)
    unsaved = FakeTrainer([early], max_steps=500)
    early.trainer = unsaved
    unsaved.train()
    assert unsaved.state.global_step == 100
    assert unsaved.control.should_training_stop is True
    assert unsaved.events == ["save@100"]

    idle = FakeTrainer([flush.StepFlushCallback(Deadline(10800, clock.now(), clock))], max_steps=250)
    idle.train()
    assert (idle.state.global_step, idle.control.should_training_stop) == (250, False)

    fake.add_repo(ADAPTER, files={"last-checkpoint/adapter_model.safetensors": b"w", "README.md": b"card"})
    checkpoint = fetch_last_checkpoint(Hub(fake), ADAPTER, tmp_path)
    resumed = FakeTrainer([], max_steps=1)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    download = next(kwargs for method, kwargs in fake.calls if method == "snapshot_download")
    assert download["allow_patterns"] == "last-checkpoint/*"
    assert download["revision"] == fake.heads[ADAPTER]
    assert sorted(path.name for path in checkpoint.iterdir()) == ["adapter_model.safetensors"]
    assert resumed.train_calls == [str(checkpoint)]


def test_no_checkpoint_on_the_hub_means_no_resume_directory(fake, tmp_path):
    fake.add_repo(ADAPTER, files={"README.md": b"card"})
    assert fetch_last_checkpoint(Hub(fake), ADAPTER, tmp_path) is None


def test_done_grows_only_when_the_upload_landed_and_upload_raises_for_its_own_file(flusher, fake, responses):
    flusher.add(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
    assert flusher.done == []
    fake.fail_after(1, 400)
    commit = flusher.upload(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")
    assert flusher.done == ["harmful/fr"]
    assert commit.oid == fake.heads[responses]
    fake.fail_after(1, 400)
    with pytest.raises(HfHubHTTPError):
        flusher.upload(responses, "responses/x/harmful/hi.jsonl", b"hi\n", key="harmful/hi")
    flusher.upload(responses, "responses/x/harmful/ar.jsonl", b"ar\n", key="harmful/ar")
    assert flusher.done == ["harmful/fr", "harmful/ar"]
    # files set aside get one more try when the run closes
    assert flusher.flush().partial.done == ["harmful/fr", "harmful/ar", "harmful/en", "harmful/hi"]


def test_a_file_that_cannot_be_uploaded_does_not_block_the_queue(flusher, fake, uploads, responses, monkeypatch):
    upload = fake.upload_file

    def refuse_french(**kwargs):
        if kwargs["path_in_repo"].endswith("fr.jsonl"):
            fake.fail_after(1, 400)
        return upload(**kwargs)

    monkeypatch.setattr(fake, "upload_file", refuse_french)
    for language in ("en", "fr", "hi"):
        path = f"responses/x/harmful/{language}.jsonl"
        flusher.add(responses, path, language.encode(), key=f"harmful/{language}")
    record = flusher.finish("ok")
    assert uploads()[:3] == [f"responses/x/harmful/{language}.jsonl" for language in ("en", "fr", "hi")]
    assert (record.status, record.partial.done) == ("failed", ["harmful/en", "harmful/hi"])
    failed = [note for note in record.notes if note.startswith("upload failed: ")]
    assert len(failed) == 1 and failed[0].startswith("upload failed: responses/x/harmful/fr.jsonl: ")
    again = flusher.finish("ok")
    assert (again.status, again.notes) == ("failed", record.notes)
    assert flusher.flush(data=False).notes[-1] == "1 file(s) not uploaded"


def test_flushing_twice_repeats_nothing_but_the_marker_and_record(flusher, uploads, stored, responses):
    """A second flush finds no pending data and leaves the same state."""
    flusher.add(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
    first = flusher.flush()
    second = flusher.flush()
    assert uploads() == ["responses/x/harmful/en.jsonl", MARKER, RECORD, MARKER, RECORD]
    assert first == second
    assert RunRecord.model_validate_json(stored(RECORD)) == second
    assert json.loads(stored(MARKER))["done"] == ["harmful/en"]


def test_a_flush_that_cannot_push_data_still_writes_marker_and_record(flusher, fake, uploads, stored, responses):
    flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
    flusher.add(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")
    fake.fail_after(1, 400)
    record = flusher.flush()
    assert json.loads(stored(MARKER))["done"] == ["harmful/en"]
    assert record.notes[-1].startswith("upload failed: responses/x/harmful/fr.jsonl")
    # the next flush retries the file it still holds
    assert flusher.flush().partial.done == ["harmful/en", "harmful/fr"]


def test_finishing_ok_with_an_upload_still_failing_is_recorded_as_failed(flusher, fake, stored, responses):
    flusher.add(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
    fake.fail_after(1, 400)
    assert flusher.finish("ok").status == "failed"
    assert RunRecord.model_validate_json(stored(RECORD)).status == "failed"


def test_an_urgent_flush_skips_pending_data(flusher, uploads, stored, responses):
    flusher.add(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
    record = flusher.flush(data=False)
    assert uploads() == [MARKER, RECORD]
    assert record.partial.done == []
    assert "1 file(s) not uploaded" in record.notes


def test_ok_writes_no_marker_and_keys_without_a_file_can_be_marked(flusher, uploads, stored):
    flusher.resume_from = "merge"
    flusher.mark_done("alpha=0.25")
    flusher.mark_done("alpha=0.25")
    assert flusher.finish("ok").partial is None
    assert uploads() == [RECORD]
    partial = flusher.flush().partial
    assert (partial.resume_from, partial.done) == ("merge", ["alpha=0.25"])


LABELS = "vincentnkq/aya-redteaming-6lang-eval"


def commits(fake, repo):
    """Titles of the commits a repo got after its creation, oldest first."""
    return [fake.messages[oid] for oid in fake.repos[repo] if oid != "main"][1:]


def test_the_files_of_a_unit_travel_in_one_commit_per_repo_and_its_key_waits_for_the_last(flusher, fake, responses):
    """A commit costs the same wait whatever it holds, so a unit costs one per repo it writes to."""
    fake.add_repo(LABELS, repo_type="dataset")
    files = [
        (LABELS, "labels/x/benign/en.jsonl", b"scored\n"),
        (responses, "responses/x/benign/en.jsonl", b"rows\n"),
        (responses, "judge_raw/x/en.jsonl", b"raw\n"),
    ]

    flusher.commit(files, key="benign/en")

    title = f"{RUN_ID}: benign/en"
    assert flusher.done == ["benign/en"] and commits(fake, LABELS) == commits(fake, responses) == [title]
    assert fake.repos[responses]["main"] == {"responses/x/benign/en.jsonl": b"rows\n", "judge_raw/x/en.jsonl": b"raw\n"}
    # one file goes as it always did, several as one commit
    assert [method for method, _ in fake.calls if method != "repo_info"] == ["upload_file", "create_commit"]

    # the second repo's commit fails: the key is not done, the call raises, the closing save tries the files again
    fake.fail_after(1, 400)
    with pytest.raises(HfHubHTTPError):
        flusher.commit([(responses, "responses/x/benign/fr.jsonl", b"fr\n")], key="benign/fr")
    assert flusher.done == ["benign/en"]
    record = flusher.flush()
    assert record.partial.done == ["benign/en", "benign/fr"]
    assert any(note.startswith("upload failed: responses/x/benign/fr.jsonl: ") for note in record.notes)


def test_a_unit_whose_first_repo_took_its_commit_is_not_done_until_the_last_did(flusher, fake, responses):
    fake.add_repo(LABELS, repo_type="dataset")
    files = [(LABELS, "labels/x/benign/en.jsonl", b"scored\n"), (responses, "responses/x/benign/en.jsonl", b"rows\n")]
    real = fake.upload_file

    def responses_down(**kwargs):
        if kwargs["repo_id"] == responses:
            fake.fail_after(1, 400)
        return real(**kwargs)

    fake.upload_file = responses_down
    with pytest.raises(HfHubHTTPError):
        flusher.commit(files, key="benign/en")
    assert flusher.done == [] and "labels/x/benign/en.jsonl" in fake.repos[LABELS]["main"]
    fake.upload_file = real
    assert flusher.flush().partial.done == ["benign/en"]


def test_progress_is_saved_after_the_first_unit_and_then_only_every_few_minutes(flusher, clock, uploads, responses):
    """Marker and record cost two commits: between units they are written when the last save has aged."""
    for at, language in enumerate(("en", "fr", "es", "ru")):
        flusher.commit([(responses, f"responses/x/harmful/{language}.jsonl", b"rows\n")], key=f"harmful/{language}")
        flusher.progress()
        clock.advance(flush.PROGRESS_S / 2)
        saves = uploads().count(MARKER)
        assert saves == uploads().count(RECORD) == (1, 1, 2, 2)[at], language
    assert flusher.flush().partial.done == [f"harmful/{language}" for language in ("en", "fr", "es", "ru")]


def test_a_progress_save_that_fails_is_a_note_and_is_tried_again_at_the_next_unit(flusher, fake, uploads, stored):
    flusher.mark_done("harmful/en")
    fake.fail_after(1, 400)
    flusher.progress()
    assert stored(MARKER) is None and flusher.run.record.notes[-1].startswith("marker push failed: ")
    flusher.progress()
    assert json.loads(stored(MARKER))["done"] == ["harmful/en"]


def test_a_resumed_run_takes_the_units_whose_commits_landed_after_the_last_marker(
    flusher, fake, clock, responses, settings
):
    """The marker may be minutes old; the titles of the unit commits say what reached the Hub since."""
    for language in ("en", "fr", "es"):
        flusher.commit([(responses, f"responses/x/harmful/{language}.jsonl", b"rows\n")], key=f"harmful/{language}")
        flusher.progress()
    # the run is killed here: its marker names the first unit only
    fake._commit(responses, "another-run-20261102T000000Z: harmful/ar", {"responses/y/harmful/ar.jsonl": b"rows\n"})
    fake._commit(responses, "Upload README.md", {"README.md": b"card\n"})

    deadline = Deadline(10800, clock.now(), clock)
    run_id = "drift-full-20261102T110000Z"
    later = start(settings, run_id=run_id, kind="drift", mode="full", code_sha="a" * 40, deadline=deadline, clock=clock)
    resumed = Flusher(Hub(fake), later)
    resumed.done = ["harmful/en"]
    resumed.credit(responses, RUN_ID)

    assert resumed.done == ["harmful/en", "harmful/fr", "harmful/es"]


def test_importing_the_job_runtime_does_not_load_torch(child_env):
    code = (
        "import sys\n"
        "import driftmap.jobkit.main, driftmap.jobkit.flush, driftmap.jobkit.resume, driftmap.jobkit.oom\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
    )
    child = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
