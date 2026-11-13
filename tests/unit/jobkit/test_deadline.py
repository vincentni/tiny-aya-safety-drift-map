import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from driftmap.core.records import PartialMarker, RunRecord
from driftmap.jobkit import main
from driftmap.jobkit.deadline import Deadline, DeadlineReached, parse_submitted_at
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import record_path
from tests.fakes import FakeClock, FakeHub

RUN_ID = "generate-full-20261102T100000Z"
RECORD = record_path(RUN_ID)
MARKER = marker_path(RUN_ID)
START = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
IST = timezone(timedelta(hours=5, minutes=30))


@pytest.mark.parametrize(
    ("timeout_s", "seconds", "due"),
    [
        (5400, 4859, False),
        (5400, 4860, True),
        (1800, 1619, False),
        (1800, 1620, True),
        (600, 419, False),
        (600, 420, True),
    ],
)
def test_flush_point_is_a_tenth_of_the_timeout_and_never_under_180_s(timeout_s, seconds, due):
    """The flush is due at timeout minus the reserve, to the second (TS-JK-1)."""
    clock = FakeClock()
    deadline = Deadline(timeout_s, clock.now(), clock)
    clock.advance(seconds)
    assert deadline.should_flush() is due


def test_the_deadline_counts_from_process_start_whatever_the_launcher_stamp():
    """A stamp 30 s ahead or 25 min behind moves no deadline; the wait is only recorded, in UTC (TS-JK-2)."""
    ahead = parse_submitted_at((START + timedelta(seconds=30)).astimezone(IST).isoformat())
    behind = parse_submitted_at((START - timedelta(minutes=25)).astimezone(IST).isoformat())
    assert ahead.isoformat() == "2026-11-02T10:00:30+00:00"

    for stamp, waited_s in ((ahead, 0), (behind, 1500)):
        clock = FakeClock(now=START)
        deadline = Deadline(5400, stamp, clock)
        assert deadline.waited_s == waited_s
        clock.advance(4859)
        assert not deadline.should_flush()
        clock.advance(1)
        assert deadline.should_flush()


def test_a_stamp_without_an_offset_is_refused():
    with pytest.raises(ValueError, match="offset"):
        parse_submitted_at("2026-11-02T10:00:00")
    assert parse_submitted_at("2026-11-02T10:00:00Z") == START


def test_reserve_can_be_set_and_the_force_arm_fires_once():
    clock = FakeClock()
    deadline = Deadline(1000, clock.now(), clock, reserve_s=10, force_after_s=300)
    clock.advance(299)
    assert not deadline.force_due()
    clock.advance(1)
    assert deadline.force_due()
    assert not deadline.force_due()
    clock.advance(690)
    assert deadline.should_flush()
    assert Deadline(1000, clock.now(), clock).force_due() is False


def test_last_call_sits_a_third_of_the_reserve_before_the_timeout():
    clock = FakeClock()
    deadline = Deadline(5400, clock.now() - timedelta(seconds=400), clock)
    clock.advance(1000)
    assert (deadline.elapsed_s(), deadline.waited_s) == (1000, 400)
    assert deadline.last_call_s() == 5400 - 180 - 1000


def test_the_forced_flush_counts_from_process_start_too():
    clock = FakeClock()
    deadline = Deadline(1800, clock.now() - timedelta(minutes=25), clock, force_after_s=300)
    assert not deadline.force_due()
    clock.advance(300)
    assert deadline.force_due()


def test_flush_at_the_deadline_pushes_data_then_marker_then_record(
    launch, clock, fake, uploads, stored, responses, capsys
):
    """At the deadline: data files, partial.json, record, exit 10 and the trailer (TS-JK-3)."""

    def body(ctx):
        ctx.flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
        ctx.checkpoint()
        ctx.flusher.add(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")
        clock.advance(4860)
        ctx.checkpoint()
        raise AssertionError("the body kept running past the deadline")

    assert launch(body) == 10
    assert uploads() == ["responses/x/harmful/en.jsonl", "responses/x/harmful/fr.jsonl", MARKER, RECORD]
    marker = json.loads(stored(MARKER))
    assert set(marker) == {"resume_from", "done", "written_at", "config_hash"}
    assert marker["done"] == ["harmful/en", "harmful/fr"]
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.status == "partial"
    assert record.partial == PartialMarker.model_validate(marker)
    assert record.config_hash == marker["config_hash"]
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=partial exit=10"


def test_resume_skips_done_batches_and_regenerates_the_interrupted_one(launch, fake, stored, responses, capsys):
    """Done keys are not generated again; the batch whose upload failed is, byte for byte (TS-JK-4)."""
    generated = []

    def body(ctx):
        for language in ("en", "fr"):
            key = f"harmful/{language}"
            if key in ctx.flusher.done:
                continue
            generated.append(key)
            if language == "fr" and len(generated) == 2:
                # once for the body, once for the retry while closing
                fake.fail_after(2, 400)
            ctx.flusher.upload(responses, f"responses/x/{key}.jsonl", f"{language}\n".encode(), key=key)

    assert launch(body) == 1
    assert json.loads(stored(MARKER))["done"] == ["harmful/en"]
    assert "responses/x/harmful/fr.jsonl" not in fake.repos[responses]["main"]

    again = "generate-full-20261102T120000Z"
    assert launch(body, "--resume", RUN_ID, run_id=again) == 0
    assert generated == ["harmful/en", "harmful/fr", "harmful/fr"]
    assert fake.repos[responses]["main"]["responses/x/harmful/fr.jsonl"] == b"fr\n"
    assert fake.repos[responses]["main"]["responses/x/harmful/en.jsonl"] == b"en\n"
    assert RunRecord.model_validate_json(stored(record_path(again))).status == "ok"


def test_a_raising_body_is_recorded_and_a_killed_one_leaves_no_trailer(launch, stored, responses, capsys, child_job):
    """ValueError: record failed with the message and the trailer; SIGKILL: data only (TS-JK-5)."""

    def body(ctx):
        ctx.flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
        raise ValueError("row 7 has no prompt")

    assert launch(body) == 1
    record = RunRecord.model_validate_json(stored(RECORD))
    assert record.status == "failed"
    assert "ValueError: row 7 has no prompt" in record.notes
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=1"

    child, files = child_job("kill-in-body")
    assert child.returncode == -9, child.stderr
    assert list(files) == [f"{responses}/responses/x/harmful/en.jsonl"]
    assert "driftmap-job" not in child.stdout + child.stderr


def test_one_forced_flush_then_the_job_runs_to_the_end(launch, clock, uploads, stored, responses):
    """--force-flush-after-s flushes once, work goes on, exit 0 and a note (TS-JK-8)."""

    def body(ctx):
        ctx.flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\n", key="harmful/en")
        ctx.checkpoint()
        assert uploads().count(MARKER) == 0
        clock.advance(301)
        ctx.checkpoint()
        assert RunRecord.model_validate_json(stored(RECORD)).status == "partial"
        clock.advance(301)
        ctx.checkpoint()
        ctx.flusher.upload(responses, "responses/x/harmful/fr.jsonl", b"fr\n", key="harmful/fr")

    assert launch(body, timeout_s="1800", force_flush_after_s="300") == 0
    assert uploads().count(MARKER) == 1
    assert uploads()[-1] == RECORD
    record = RunRecord.model_validate_json(stored(RECORD))
    assert (record.status, record.mode, record.partial) == ("ok", "smoke", None)
    assert record.notes == ["job lookup failed: command, flavor and price unknown", "forced flush after 301 s"]


def test_no_token_stops_the_job_before_any_hub_call(launch, fake, monkeypatch, caplog, capsys):
    """Without HF_TOKEN: exit 12, a message, zero Hub calls; with it the variable is read once (TS-JK-10)."""
    reads = []
    real = main._require_token
    monkeypatch.setattr(main, "_require_token", lambda: reads.append(1) or real())

    monkeypatch.delenv("HF_TOKEN")
    assert launch(lambda ctx: None) == 12
    assert fake.calls == []
    assert "HF_TOKEN is not set" in caplog.text
    assert capsys.readouterr().out.splitlines()[-1] == "driftmap-job: finished status=failed exit=12"

    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert launch(lambda ctx: None) == 0
    assert reads == [1, 1]


def test_a_kill_between_marker_and_record_leaves_a_marker_to_resume_from(
    child_job, tmp_path, argv, settings, results, responses
):
    """SIGKILL after partial.json: marker present, record absent, no trailer; the relaunch resumes (TS-JK-11)."""
    # 180 s timeout: flush due at once, last call after 120 s
    child, files = child_job("kill-before-record", timeout_s=180)
    assert child.returncode == -9, child.stderr
    assert sorted(files) == sorted([f"{responses}/responses/x/harmful/en.jsonl", f"{results}/{MARKER}"])
    assert "driftmap-job" not in child.stdout + child.stderr

    marker = files[f"{results}/{MARKER}"]
    hub = FakeHub(tmp_path / "relaunch").add_repo(results, repo_type="dataset", files={MARKER: marker})
    seen = []
    relaunch = argv("--resume", RUN_ID, run_id="generate-full-20261102T120000Z")
    code = main.run(
        lambda ctx: seen.extend(ctx.flusher.done),
        relaunch,
        kind="generate",
        settings=settings,
        client=hub,
        clock=FakeClock(),
    )
    assert (code, seen) == (0, ["harmful/en"])


def test_checkpoint_raises_something_library_handlers_do_not_swallow():
    assert not issubclass(DeadlineReached, Exception)
