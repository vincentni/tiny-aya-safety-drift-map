import json
import sys
from datetime import timedelta
from types import SimpleNamespace

from driftmap.core.hub import Hub
from driftmap.core.records import Ref
from driftmap.jobkit import runrecord
from driftmap.jobkit.deadline import Deadline
from driftmap.jobkit.flush import Flusher
from driftmap.jobkit.runrecord import find_job, record_path, start
from tests.fakes import FakeJobsClient

RUN_ID = "drift-full-20261102T100000Z"
MERGED = "vincentnkq/tiny-aya-global-drift-merged"
COMMAND = ["bash", "-lc", "pip install -q driftmap && python -m driftmap.scripts.drift --run-id " + RUN_ID]


def test_finish_takes_revisions_from_the_pushes_and_the_command_from_the_job(
    settings, fake, clock, results, stored, uploads, monkeypatch
):
    """outputs[].revision is the push oid, not the later head; command copied; no environment or secrets (TS-JK-6)."""
    jobs = FakeJobsClient()
    jobs.run_job(image="img", command=["true"], flavor="cpu-basic", labels={"driftmap_run": "another-run"})
    jobs.run_job(
        image="img",
        command=COMMAND,
        flavor="l4x1",
        labels={"driftmap_run": RUN_ID},
        env={"PIP_BREAK_SYSTEM_PACKAGES": "1"},
        secrets={"HF_TOKEN": "hf_dummy"},
    )
    job = find_job(jobs, RUN_ID)
    assert job.command == COMMAND

    fake.add_repo(results, repo_type="dataset").add_repo(MERGED)
    deadline = Deadline(10800, clock.now(), clock)
    run = start(
        settings,
        run_id=RUN_ID,
        kind="drift",
        mode="full",
        code_sha="a" * 40,
        deadline=deadline,
        clock=clock,
        job=job,
        inputs=[Ref(kind="model", repo="CohereLabs/tiny-aya-global", revision="1" * 40)],
    )
    assert (run.record.status, run.record.finished) == ("partial", None)
    hub = Hub(fake)
    flusher = Flusher(hub, run)
    pushed = hub.upload_file(MERGED, "model.safetensors", b"weights")
    flusher.pushed(MERGED, pushed, sha256="b" * 64)
    hub.upload_file(MERGED, "README.md", b"card")
    later = hub.model_info(MERGED).sha
    assert later != pushed.oid

    run.record.prompts, run.record.prompt_tokens, run.record.generated_tokens = 600, 48_000, 90_000
    clock.advance(3000)
    cuda = SimpleNamespace(is_available=lambda: True, max_memory_allocated=lambda: 3 * 2**30)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    record = flusher.finish("ok")

    assert record.outputs == [Ref(kind="model", repo=MERGED, revision=pushed.oid, sha256="b" * 64)]
    assert record.command == COMMAND
    assert (record.job_id, record.flavor, record.price_micro_usd_per_min) == (job.id, "l4x1", 13_333)
    assert (record.wall_clock_s, record.minutes, record.install_minutes) == (3000.0, 50, 0.0)
    assert record.cost_micro_usd == 666_650 and type(record.cost_micro_usd) is int
    assert record.tokens_per_s == 30.0
    assert record.peak_vram_gb == 3.0
    assert record.peak_rss_gb > 0
    assert (record.status, record.finished) == ("ok", clock.now())

    on_hub = json.loads(stored(record_path(RUN_ID)))
    assert on_hub == json.loads(record.model_dump_json(by_alias=True))
    assert not {"environment", "secrets", "env"} & set(on_hub)
    assert "hf_dummy" not in stored(record_path(RUN_ID)).decode()
    assert uploads()[-1] == record_path(RUN_ID)


def test_minutes_count_from_process_start_and_round_up(settings, clock):
    deadline = Deadline(1800, clock.now(), clock)
    clock.advance(61)
    run = start(settings, run_id=RUN_ID, kind="drift", mode="smoke", code_sha="a" * 40, deadline=deadline, clock=clock)
    record = run.finish("partial", [])
    assert (record.minutes, record.wall_clock_s) == (2, 61.0)
    assert (record.flavor, record.price_micro_usd_per_min, record.cost_micro_usd) == ("unknown", 0, None)
    assert record.command == []


def test_install_minutes_is_the_wait_between_submission_and_start(settings, clock):
    deadline = Deadline(1800, clock.now() - timedelta(seconds=150), clock)
    run = start(settings, run_id=RUN_ID, kind="drift", mode="smoke", code_sha="a" * 40, deadline=deadline, clock=clock)
    assert run.record.install_minutes == 2.5
    assert run.record.notes[-1] == "waited 150 s between submission and start"
    clock.advance(31)
    assert run.finish("ok", []).minutes == 1


def test_a_queued_job_is_billed_for_its_running_time_only(settings, clock):
    job = SimpleNamespace(id="j1", flavor="l4x1", command=["python"])
    deadline = Deadline(1800, clock.now() - timedelta(minutes=25), clock)
    run = start(
        settings, run_id=RUN_ID, kind="drift", mode="smoke", code_sha="a" * 40, deadline=deadline, clock=clock, job=job
    )
    clock.advance(240)
    record = run.finish("ok", [])
    assert (record.minutes, record.cost_micro_usd, record.wall_clock_s) == (4, 4 * 13_333, 240.0)
    assert (record.install_minutes, record.notes) == (25.0, ["waited 1500 s between submission and start"])


def test_the_latest_push_to_a_repo_is_the_one_recorded(settings, fake, clock, responses):
    fake.add_repo(f"{settings.owner}/{settings.repos.results}", repo_type="dataset")
    deadline = Deadline(1800, clock.now(), clock)
    run = start(settings, run_id=RUN_ID, kind="drift", mode="full", code_sha="a" * 40, deadline=deadline, clock=clock)
    flusher = Flusher(Hub(fake), run)
    flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\n")
    last = flusher.upload(responses, "responses/x/harmful/fr.jsonl", b"fr\n")
    assert flusher.finish("ok").outputs == [Ref(kind="dataset", repo=responses, revision=last.oid)]


def test_a_job_that_cannot_be_looked_up_is_a_note_not_a_failure(settings, clock, caplog):
    class Down:
        def list_jobs(self, **kwargs):
            raise RuntimeError("no route")

    assert find_job(Down(), RUN_ID) is None
    assert find_job(FakeJobsClient(), RUN_ID) is None
    assert find_job(object(), RUN_ID) is None
    assert "job lookup failed" in caplog.text
    deadline = Deadline(1800, clock.now(), clock)
    run = start(settings, run_id=RUN_ID, kind="drift", mode="full", code_sha="a" * 40, deadline=deadline, clock=clock)
    assert run.record.notes == ["job lookup failed: command, flavor and price unknown"]
    run.note("x")
    run.note("x")
    assert run.record.notes[1:] == ["x"]


def test_a_token_in_the_command_is_masked(settings, clock):
    token = "hf_" + "A" * 34
    job = SimpleNamespace(id="j1", flavor="l4x1", command=["bash", "-lc", f"HF_TOKEN={token} python x"])
    deadline = Deadline(1800, clock.now(), clock)
    run = start(
        settings, run_id=RUN_ID, kind="drift", mode="full", code_sha="a" * 40, deadline=deadline, clock=clock, job=job
    )
    assert run.record.command == ["bash", "-lc", "HF_TOKEN=hf_*** python x"]


def test_peak_vram_is_absent_without_cuda(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert runrecord.peak_vram_gb() is None
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert runrecord.peak_vram_gb() is None
