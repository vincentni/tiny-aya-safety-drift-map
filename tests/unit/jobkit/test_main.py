import json
import logging
import signal

import pytest

from driftmap.core.records import RunRecord
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.jobkit import main
from driftmap.jobkit.deadline import DeadlineReached
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path

RUN_ID = "generate-full-20261102T100000Z"
RECORD = record_path(RUN_ID)
MARKER = marker_path(RUN_ID)


class OddCode(Exception):
    def __init__(self, message, exit_code="boom"):
        super().__init__(message)
        self.exit_code = exit_code


def test_the_body_gets_parsed_arguments_including_its_own(launch, settings):
    seen = {}

    def body(ctx):
        seen.update(vars(ctx.args), settings=ctx.settings, mode=ctx.run.record.mode)

    assert launch(body, "--limit", "10") == 0
    assert (seen["run_id"], seen["timeout_s"], seen["code_sha"]) == (RUN_ID, 5400, "a" * 40)
    assert (seen["limit"], seen["force_flush_after_s"], seen["resume"]) == (10, None, None)
    assert seen["submitted_at"].isoformat() == "2026-11-02T10:00:00+00:00"
    assert (seen["settings"], seen["mode"]) == (settings, "full")


def test_a_script_adds_its_own_arguments(settings, fake, clock, argv):
    seen = []
    code = main.run(
        lambda ctx: seen.append(ctx.args.checkpoint),
        argv("--checkpoint", "owner/name@" + "1" * 40),
        kind="generate",
        settings=settings,
        arguments=lambda parser: parser.add_argument("--checkpoint", required=True),
        client=fake,
        clock=clock,
    )
    assert (code, seen) == (0, ["owner/name@" + "1" * 40])


@pytest.mark.parametrize("extra", [["--bogus"], ["--submitted-at", "2026-11-02T10:00:00"], ["--timeout-s", "soon"]])
def test_bad_arguments_are_a_usage_error_before_anything_runs(launch, fake, extra, capsys):
    with pytest.raises(SystemExit) as stop:
        launch(lambda ctx: None, *extra)
    assert stop.value.code == 2
    assert fake.calls == []
    assert "driftmap-job" not in capsys.readouterr().out


def test_the_results_repo_is_created_private_before_the_body(launch, fake, results):
    def body(ctx):
        assert results in fake.repos

    assert launch(body) == 0
    created = next(kwargs for method, kwargs in fake.calls if method == "create_repo")
    assert (created["repo_id"], created["repo_type"], created["private"]) == (results, "dataset", True)
    assert results in fake.private


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (DriftmapError(ExitCode.INTEGRITY, "tokenizer canary mismatch"), 12),
        (DriftmapError(ExitCode.COLUMN_FAILED, "7 % unscored"), 13),
        (DriftmapError(ExitCode.PARTIAL, "not a deadline"), 1),
        (DriftmapError(ExitCode.OOM, "not out of memory"), 1),
        (DriftmapError(ExitCode.USAGE, "not a job code"), 1),
        (KeyError("prompt"), 1),
        (OddCode("exit_code is text"), 1),
        (OddCode("exit_code is None", None), 1),
        (OddCode("exit_code is no job code", 99), 1),
        (KeyboardInterrupt(), 1),
        (SystemExit(3), 1),
    ],
)
def test_exit_codes_follow_the_error(launch, stored, capsys, error, code):
    def body(ctx):
        raise error

    assert launch(body) == code
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.status == "failed"
    assert record.notes[-1].startswith(type(error).__name__)
    assert capsys.readouterr().out.splitlines()[-1] == f"driftmap-job: finished status=failed exit={code}"


def test_a_token_in_the_error_never_reaches_the_record(launch, stored):
    token = "hf_" + "A" * 34

    def body(ctx):
        raise RuntimeError(f"401 for token {token}")

    assert launch(body) == 1
    assert token not in stored(RECORD).decode()
    assert "RuntimeError: 401 for token hf_***" in json.loads(stored(RECORD))["notes"]


def test_an_ok_run_whose_record_push_fails_leaves_a_marker_and_a_failed_record(launch, fake, uploads, stored, capsys):
    def body(ctx):
        # the record push is the next Hub call
        fake.fail_after(1, 400)

    assert launch(body) == 1
    assert uploads() == [RECORD, MARKER, RECORD]
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.status == "failed"
    assert any(note.startswith("closing push failed: ") for note in record.notes)
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"


def test_a_run_that_cannot_push_anything_still_ends_with_the_trailer(launch, fake, stored, capsys, caplog):
    def body(ctx):
        # the ok record, then the marker and failed record of the retry
        fake.fail_after(3, 400)

    assert launch(body) == 1
    assert (stored(RECORD), stored(MARKER)) == (None, None)
    assert "could not push the run record" in caplog.text
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"


@pytest.mark.parametrize("stop", [DeadlineReached("stopped by SIGTERM", urgent=True), RuntimeError("Hub is down")])
def test_a_run_stopped_while_reading_the_marker_pushes_nothing(launch, uploads, monkeypatch, capsys, stop):
    def reading(*args, **kwargs):
        raise stop

    monkeypatch.setattr(main, "read_partial", reading)
    assert launch(lambda ctx: pytest.fail("the body ran"), "--resume", "generate-full-20261102T080000Z") == 1
    assert uploads() == []
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"


def test_past_the_last_call_the_closing_push_skips_queued_data(launch, clock, uploads, stored, responses, capsys):
    def body(ctx):
        ctx.flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
        ctx.flusher.add(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")
        clock.advance(5220)

    assert launch(body) == 10
    assert uploads() == ["responses/x/harmful/en.jsonl", MARKER, RECORD]
    record = RunRecord.model_validate_json(stored(RECORD))
    assert (record.status, record.partial.done) == ("partial", ["harmful/en"])
    assert "1 file(s) not uploaded" in record.notes
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"


def test_just_before_the_last_call_the_closing_push_still_sends_queued_data(launch, clock, uploads, responses):
    def body(ctx):
        ctx.flusher.add(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")
        clock.advance(5219)

    assert launch(body) == 0
    assert uploads() == ["responses/x/harmful/fr.jsonl", RECORD]


def test_an_urgent_stop_gives_each_hub_call_one_attempt(launch, fake, uploads, stored, capsys):
    def body(ctx):
        fake.fail_after(1, 503)
        raise DeadlineReached("stopped by SIGTERM", urgent=True)

    assert launch(body) == 10
    assert uploads() == [MARKER, RECORD]
    assert stored(MARKER) is None
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.status == "partial"
    assert any(note.startswith("marker push failed: ") for note in record.notes)
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"


def test_an_error_before_the_body_is_reported_without_a_record(launch, fake, capsys):
    fake.fail_after(1, 400)
    assert launch(lambda ctx: pytest.fail("the body ran")) == 1
    assert [method for method, _ in fake.calls] == ["create_repo"]
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"


def test_an_injected_clock_leaves_signals_and_logging_alone(launch):
    before = (signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGTERM), list(logging.getLogger().handlers))

    def body(ctx):
        assert (signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGTERM)) == before[:2]

    assert launch(body) == 0
    assert list(logging.getLogger().handlers) == before[2]


def test_the_last_call_interrupts_a_stalled_upload_and_saves_the_run(child_job, results, responses):
    """A Hub call still blocked a third of the reserve before the timeout is abandoned for the marker and record."""
    # a 63 s timeout: the last call comes 3 s in
    child, files = child_job("stall", timeout_s=63)
    assert child.returncode == 10, child.stderr
    assert child.stdout.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"
    assert sorted(files) == sorted(
        [f"{responses}/responses/x/harmful/en.jsonl", f"{results}/{MARKER}", f"{results}/{RECORD}"]
    )
    record = RunRecord.model_validate_json(files[f"{results}/{RECORD}"])
    assert record.partial.done == ["harmful/en"]
    assert "1 file(s) not uploaded" in record.notes
    assert any(note.startswith("last call") for note in record.notes)
    assert "last call" in child.stderr


def test_sigterm_saves_the_run_as_partial(child_job, results):
    child, files = child_job("term")
    assert child.returncode == 10, child.stderr
    assert child.stdout.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"
    record = RunRecord.model_validate_json(files[f"{results}/{RECORD}"])
    assert (record.status, record.partial.done) == ("partial", ["harmful/en"])
    assert "stopped by SIGTERM" in record.notes


def test_sigterm_during_the_final_push_does_not_cut_it_short(child_job, results):
    child, files = child_job("term-while-closing")
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines()[-1] == "driftmap-job: finished status=ok exit=0"
    record = RunRecord.model_validate_json(files[f"{results}/{RECORD}"])
    assert record.status == "ok"
    assert "stopped by SIGTERM" not in record.notes


def test_a_stop_before_anything_could_be_saved_is_not_reported_as_partial(launch, fake, monkeypatch, capsys):
    def stop(client, run_id):
        raise DeadlineReached("stopped by SIGTERM", urgent=True)

    monkeypatch.setattr(main, "find_job", stop)
    assert launch(lambda ctx: None) == 1
    assert fake.calls == []
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"


def test_a_partial_run_whose_pushes_all_fail_is_not_reported_as_partial(launch, fake, clock, stored, capsys):
    def body(ctx):
        clock.advance(5000)
        # the marker, then the marker and record of the retry
        fake.fail_after(3, 400)
        ctx.checkpoint()

    assert launch(body) == 1
    assert (stored(RECORD), stored(MARKER)) == (None, None)
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"


@pytest.mark.parametrize(("stop", "status", "code"), [("deadline", "partial", 10), ("error", "failed", 1)])
def test_a_run_that_did_not_end_ok_also_gets_one_more_push(
    launch, fake, clock, uploads, stored, capsys, stop, status, code
):
    def body(ctx):
        clock.advance(5000)
        fake.fail_after(1, 400)
        if stop == "error":
            raise ValueError("row 7 has no prompt")
        ctx.checkpoint()

    assert launch(body) == code
    assert uploads() == [MARKER, MARKER, RECORD]
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.status == status
    assert any(note.startswith("closing push failed: ") for note in record.notes)
    assert json.loads(stored(MARKER)) == json.loads(record.partial.model_dump_json())
    assert capsys.readouterr().out.splitlines()[-1] == f"driftmap-job: finished status={status} exit={code}"


@pytest.mark.parametrize(
    ("deadline", "code", "pushes", "note"),
    [
        (True, 10, [MARKER, RECORD], "marker push failed: "),
        (False, 1, [RECORD, MARKER, RECORD], "closing push failed: "),
    ],
)
def test_past_the_last_call_the_marker_and_record_get_one_attempt_each(
    launch, fake, clock, uploads, stored, responses, monkeypatch, deadline, code, pushes, note
):
    upload = fake.upload_file

    def slow_data(**kwargs):
        commit = upload(**kwargs)
        if kwargs["path_in_repo"].endswith("fr.jsonl"):
            # the file lands after the last call; the next push fails
            clock.advance(5300)
            fake.fail_after(1, 503)
        return commit

    monkeypatch.setattr(fake, "upload_file", slow_data)

    def body(ctx):
        ctx.flusher.add(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")
        if deadline:
            clock.advance(4860)
            ctx.checkpoint()

    assert launch(body) == code
    assert uploads() == ["responses/x/harmful/fr.jsonl", *pushes]
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.partial.done == ["harmful/fr"]
    assert any(text.startswith(note) for text in record.notes)


def test_the_last_call_also_interrupts_a_stalled_upload_while_closing(child_job, results, responses):
    child, files = child_job("stall-closing", timeout_s=63)
    assert child.returncode == 10, child.stderr
    assert child.stdout.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"
    assert sorted(files) == sorted(
        [f"{responses}/responses/x/harmful/en.jsonl", f"{results}/{MARKER}", f"{results}/{RECORD}"]
    )
    record = RunRecord.model_validate_json(files[f"{results}/{RECORD}"])
    assert (record.status, record.partial.done) == ("partial", ["harmful/en"])
    assert "1 file(s) not uploaded" in record.notes


def test_a_body_that_absorbs_the_last_call_is_interrupted_again(child_job, tmp_path, results):
    child, files = child_job("absorb", timeout_s=63)
    assert child.returncode == 10, child.stderr
    assert not (tmp_path / "slept").exists()
    assert RunRecord.model_validate_json(files[f"{results}/{RECORD}"]).status == "partial"


def test_a_marker_push_stalled_past_the_last_call_is_interrupted_and_tried_once_more(child_job, results, responses):
    # the last call, 5 s after the start, finds the marker stalled
    child, files = child_job("stall-marker", timeout_s=65)
    assert child.returncode == 10, child.stderr
    assert child.stdout.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"
    assert sorted(files) == sorted(
        [f"{responses}/responses/x/harmful/en.jsonl", f"{results}/{MARKER}", f"{results}/{RECORD}"]
    )
    record = RunRecord.model_validate_json(files[f"{results}/{RECORD}"])
    assert (record.status, record.partial.done) == ("partial", ["harmful/en"])
    assert any(note.startswith("last call: interrupted") for note in record.notes)


def test_time_in_the_queue_does_not_count_against_the_timeout(launch, clock, stored):
    def body(ctx):
        clock.advance(240)
        ctx.checkpoint()

    # 25 minutes queued, 4 minutes of work, a 30 minute timeout
    assert launch(body, timeout_s="1800", submitted_at="2026-11-02T09:35:00Z") == 0
    record = RunRecord.model_validate_json(stored(RECORD))
    assert (record.status, record.minutes, record.install_minutes) == ("ok", 4, 25.0)
    assert "waited 1500 s between submission and start" in record.notes


def test_a_live_job_submitted_long_ago_still_gets_its_full_timeout(child_job, results):
    child, files = child_job("plain", age_s=1700, timeout_s=1800)
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines()[-1] == "driftmap-job: finished status=ok exit=0"
    record = RunRecord.model_validate_json(files[f"{results}/{RECORD}"])
    assert (record.status, record.minutes) == ("ok", 1)
    assert record.install_minutes >= 28
