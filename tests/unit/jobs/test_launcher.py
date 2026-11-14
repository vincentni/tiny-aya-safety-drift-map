import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx2
import pytest
from huggingface_hub import JobStage
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils import hf_raise_for_status

from driftmap.core.hub import Hub
from driftmap.core.records import RunRecord, SmokeRecord, dump, load
from driftmap.core.schemas import SCHEMA_RUN, DriftmapError, ExitCode, schema_versions
from driftmap.core.settings import ConfigError, load_settings
from driftmap.jobs import guards, launcher, snapshot
from driftmap.jobs.kinds import KINDS, State
from driftmap.jobs.launcher import (
    HOLD,
    HOLD_GRACE,
    HOST_RAM_RUNG,
    LEDGER_PATH,
    LOGS_DIR,
    NO_RUNG,
    OOM_LADDER,
    adopted_line,
    chain,
    classify,
    hold_line,
    launch,
    reconcile,
    wait_job,
)
from driftmap.jobs.ledger import Ledger, pending_line, terminal_line
from driftmap.jobs.lock import LockHeld
from driftmap.jobs.plan import build_plan, pins, submit_kwargs
from driftmap.jobs.readiness import ITEMS, READY_PATH, Item, NotReady, ReadyReport
from driftmap.jobs.snapshot import SMOKE_DIR, Deps, read_smoke_records, take_snapshot
from tests.fakes import FakeClock, FakeHub, FakeJobsClient, FakeSubprocess
from tests.fakes.jobs_client import DEFAULT_HARDWARE
from tests.fakes.settings import settings_toml

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "logs" / "token_tail.log"
AYA, BASE = "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"
RESULTS = "vincentnkq/safety-drift-map-results"
SHA = "a" * 40
TOKEN = "hf_" + "b" * 34
HOME_DIR = "/" + "home/someone"
# generate: 90 minutes, 95 % of it is 5130 s
T = 5400
RUN = "generate-full-20261102T100000Z"
SMOKE_RUN = "generate-smoke-20261102T100000Z"
GUARDS = [guard.__name__ for guard in guards.GUARD_CHAIN]


def trailer(status, code):
    return f"driftmap-job: finished status={status} exit={code}"


def run_record(run_id, status, **changes):
    kind, mode, _ = run_id.split("-")
    fields = {
        "schema": SCHEMA_RUN,
        "run_id": run_id,
        "job_id": None,
        "kind": kind,
        "mode": mode,
        "config_hash": "c" * 64,
        "code_sha": SHA,
        "inputs": [],
        "outputs": [],
        "merged_sha256": None,
        "command": [],
        "started": datetime(2026, 11, 2, 10, 2, tzinfo=UTC),
        "finished": datetime(2026, 11, 2, 10, 27, tzinfo=UTC),
        "wall_clock_s": 1500.0,
        "flavor": "l4x1",
        "price_micro_usd_per_min": 13_333,
        "minutes": 25,
        "cost_micro_usd": 333_325,
        "prompts": 1450,
        "prompt_tokens": 530_000,
        "generated_tokens": 120_000,
        "tokens_per_s": 80.0,
        "status": status,
        "partial": None,
        "oom": "CUDA out of memory" if status == "oom" else None,
        "peak_vram_gb": 9.4,
        "peak_rss_gb": 6.1,
        "install_minutes": 2.5,
        "notes": [],
    }
    return RunRecord.model_validate(fields | changes)


def lock_held(path):
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def http_error(status):
    request = httpx2.Request("POST", "https://huggingface.co/api/jobs/vincentnkq")
    try:
        hf_raise_for_status(httpx2.Response(status, request=request))
    except httpx2.HTTPError as error:
        return error


class World:
    """A prepared checkout: passed readiness, open gate, scripted git, fake Hub and Jobs."""

    def __init__(self, tmp_path, client=None, **overrides):
        tmp_path.mkdir(exist_ok=True)
        self.tmp_path = tmp_path
        self.settings = load_settings(settings_toml(tmp_path, **overrides))
        self.root = tmp_path / "checkout"
        self.root.mkdir()
        shutil.copy(REPO_ROOT / "pyproject.toml", self.root)
        self.clock = FakeClock()
        self.client = client or FakeJobsClient()
        self.fake_hub = FakeHub(tmp_path / "hub")
        self.fake_hub.add_repo(AYA, gated=True).add_repo(BASE, gated=True).add_repo(RESULTS, repo_type="dataset")
        self.fake_hub.authorize(AYA)
        self.git = FakeSubprocess()
        self.ledger = Ledger(self.root / LEDGER_PATH)
        self.deps = self.deps_with()
        self.ready()

    def deps_with(self, **parts):
        hub = Hub(self.fake_hub, gated_repos=self.settings.gated_repos, sleep=self.clock.sleep)
        base = {"client": self.client, "hub": hub, "ledger": self.ledger, "run": self.git.run}
        return Deps(repo_root=self.root, now=self.clock.now, **(base | parts))

    def ready(self, **fields):
        items = [Item(number=number, name=name, status="PASS", detail="ok") for number, name, _ in ITEMS]
        report = ReadyReport(ts=self.clock.now(), config_hash=self.settings.config_hash(), items=items, **fields)
        dump(report, self.root / READY_PATH)

    def git_ok(self, launches=1):
        """HEAD for the launcher, then HEAD, status, fetch and ancestor check for the snapshot."""
        for _ in range(launches):
            self.git.script(stdout=SHA + "\n")
            self.git.script(stdout=SHA + "\n")
            for _ in range(3):
                self.git.script()

    def plan(self, kind="generate", mode="full", settings=None):
        return build_plan(settings or self.settings, kind, mode, None, SHA, now=self.clock.now(), repo_root=self.root)

    def smoke_passed(self, kind="generate"):
        plan = self.plan(kind)
        record = SmokeRecord(
            script=plan.script,
            image=plan.image,
            runner=plan.runner,
            dependencies=pins(self.root),
            schema_versions=schema_versions(),
            status="ok",
            bos_check=None,
            install_minutes=2.5,
            upload_mb_s=11.0,
            peak_vram_gb=9.4,
            peak_rss_gb=6.1,
            tokens_per_s=80.0,
        )
        dump(record, self.root / SMOKE_DIR / f"{kind}.json")

    def job(self, job_id, stage, running_secs=1500, logs=()):
        durations = {"schedulingSecs": 10, "runningSecs": running_secs, "totalSecs": 10 + (running_secs or 0)}
        self.client.script_job(job_id, [{"stage": "RUNNING"}, {"status": {"stage": stage}, "durations": durations}])
        self.client.logs[job_id] = list(logs)

    def started(self, job_id, kind="generate", mode="full"):
        """A job already submitted and on the ledger as pending."""
        plan = self.plan(kind, mode)
        self.client.run_job(image=plan.image, command=["true"], flavor=plan.flavor)
        line = pending_line(plan, job_id, self.clock.now(), "settings")
        self.ledger.append(line)
        return line

    def push(self, path, data):
        self.fake_hub.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=RESULTS, repo_type="dataset")

    def push_record(self, run_id, status, **changes):
        changes.setdefault("config_hash", self.settings.config_hash())
        self.push(f"runs/{run_id}.json", run_record(run_id, status, **changes).model_dump_json().encode())

    def job_lines(self):
        """The last line of each job, holds left out."""
        return [line for job_id, line in self.ledger.read().items() if not job_id.startswith(HOLD)]

    def holds(self):
        return [line for job_id, line in self.ledger.read().items() if job_id.startswith(HOLD)]

    def submits(self):
        return [call for call in self.client.calls if call[0] in ("run_job", "run_uv_job")]

    def count(self, method):
        return sum(name == method for name, _ in self.client.calls)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail(f"slept {seconds} s"))


def spy(target, events, lock_path, only=None):
    """Forward to target, noting each call and whether the launch lock was held."""

    class Proxy:
        def __getattr__(self, name):
            attribute = getattr(target, name)
            if not callable(attribute) or (only and name not in only):
                return attribute

            def call(*args, **kwargs):
                label = name
                if name == "append":
                    label = f"{'pending' if args[0].pending else 'terminal'} {args[0].job_id}"
                events.append((label, lock_held(lock_path)))
                return attribute(*args, **kwargs)

            return call

    return Proxy()


def test_a_launch_reads_everything_under_the_lock_then_submits_records_and_waits(world, monkeypatch, capsys):
    """Reads, hold, submit and pending line under the lock; then wait, evidence, billing, terminal line (TS-LCH-1)."""
    events = []
    lock = world.settings.lock_path
    world.git_ok()
    world.smoke_passed()
    world.client.billing_used_micro_usd = 1_000_000
    world.job("job-1", "COMPLETED", logs=["working", trailer("ok", 0)])
    world.push_record(RUN, "ok")
    smoke_record = (world.root / SMOKE_DIR / "generate.json").read_bytes()
    on_disk = []
    poll = world.client.wait_for_job

    def wait_for_job(job_id, **kwargs):
        on_disk.append(world.ledger.read()[job_id].pending)
        return poll(job_id, **kwargs)

    world.client.wait_for_job = wait_for_job

    def noted(name):
        read = getattr(snapshot, name)

        def reader(path):
            events.append((name, lock_held(lock)))
            return read(path)

        return reader

    for name in ("read_smoke_records", "read_decision"):
        monkeypatch.setattr(snapshot, name, noted(name))

    def git(argv, **kwargs):
        events.append((f"git {argv[3]}", lock_held(lock)))
        return world.git.run(argv, **kwargs)

    deps = Deps(
        client=spy(world.client, events, lock),
        hub=Hub(spy(world.fake_hub, events, lock), gated_repos=world.settings.gated_repos, sleep=world.clock.sleep),
        ledger=spy(world.ledger, events, lock, only=("append",)),
        repo_root=world.root,
        now=world.clock.now,
        run=git,
    )

    assert launch(world.settings, deps, "generate", "full", wait=True) is ExitCode.SUCCESS

    labels = [label for label, _ in events]
    submit = labels.index("run_job")
    hold = f"{HOLD}{RUN}"
    assert labels[0] == "git rev-parse" and labels[submit - 1] == f"pending {hold}"
    assert sorted(labels[: submit - 1]) == sorted(
        [
            *["git rev-parse"] * 2,
            *["git status", "git fetch", "git merge-base"],
            *["billing_usage", "list_jobs_hardware", "list_jobs"],
            *["read_decision", "read_smoke_records", "auth_check", "auth_check"],
        ]
    )
    after = ["wait_for_job", "dataset_info", "hf_hub_download", "fetch_job_logs", "billing_usage", "terminal job-1"]
    assert labels[submit:] == ["run_job", "pending job-1", f"terminal {hold}", *after]
    assert [held for _, held in events] == [True] * (submit + 3) + [False] * 6
    assert on_disk == [True]
    reserved, first, released, last = world.ledger.lines()
    assert (reserved.job_id, reserved.pending, reserved.cost_micro_usd) == (hold, True, 1_199_970)
    assert reserved.notes == ["price=hardware", "hold"]
    assert (released.job_id, released.pending, released.cost_micro_usd, released.stage) == (hold, False, 0, "RELEASED")
    assert released.notes[-1] == "job=job-1"
    assert (first.job_id, first.pending, first.cost_micro_usd, first.stage) == ("job-1", True, 1_199_970, "SCHEDULING")
    assert first.notes == ["price=hardware"]
    assert (last.pending, last.minutes, last.cost_micro_usd, last.stage) == (False, 25, 333_325, JobStage.COMPLETED)
    assert (last.record_status, last.exit_code, last.lost, last.oom) == ("ok", 0, False, False)
    assert last.billing_used_after_micro_usd == 1_000_000
    assert world.ledger.total_micro_usd() == 333_325
    out = capsys.readouterr().out
    assert "next: none" in out and "job job-1 ended COMPLETED: 25 min" in out
    assert (world.root / SMOKE_DIR / "generate.json").read_bytes() == smoke_record


def test_the_wait_polls_once_a_minute_within_timeout_plus_allowance(world, no_sleep):
    """The poll interval is 60 s, the budget is the timeout plus 1800 s, and nothing sleeps here (TS-LCH-2)."""
    world.git_ok()
    world.smoke_passed()
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    world.push_record(RUN, "ok")

    assert launch(world.settings, world.deps, "generate", "full", wait=True) is ExitCode.SUCCESS

    (kwargs,) = [kwargs for name, kwargs in world.client.calls if name == "wait_for_job"]
    assert (kwargs["job_id"], kwargs["poll_interval"], kwargs["timeout"]) == ("job-1", 60, T + 1800)
    assert world.clock.sleeps == []


def test_a_wait_that_runs_out_prints_the_cancel_command_and_cancels_nothing(world, capsys):
    """Past the budget: the cancel hint, the pending line untouched, exit 5, no cancel call (TS-LCH-3)."""
    world.git_ok()
    world.smoke_passed()
    world.client.script_job("job-1", [{"stage": "RUNNING"}])

    assert launch(world.settings, world.deps, "generate", "full", wait=True) is ExitCode.LOST_RUN

    assert "hf jobs cancel job-1" in capsys.readouterr().out
    (line,) = world.job_lines()
    assert (line.pending, line.cost_micro_usd) == (True, 1_199_970)
    assert world.count("cancel_job") == 0 and world.count("fetch_job_logs") == 0
    before = world.ledger.path.read_bytes()
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.LOST_RUN
    assert world.ledger.path.read_bytes() == before and world.count("cancel_job") == 0


OK, LOST, STOP = ExitCode.SUCCESS, ExitCode.LOST_RUN, ExitCode.UNEXPECTED
QUIET, PIP = "step 900", "ERROR: No matching distribution found for trl==9"
# stage, status, exit code or log tail, marker, seconds, flags, action, exit
ROWS = {
    "ok": ("COMPLETED", "ok", 0, False, 1500, {}, "none", OK),
    "partial": ("ERROR", "partial", 10, False, 4900, {"notes": ["partial"]}, "same timeout is allowed", STOP),
    "oom": ("ERROR", "oom", 11, False, 700, {"oom": True}, "16 -> 8 -> 4", STOP),
    "failed": ("ERROR", "failed", 12, False, 90, {"notes": ["integrity"]}, "fix the input", STOP),
    "lost": ("ERROR", None, QUIET, False, 5130, {"lost": True}, "widen it or split", LOST),
    "marker": ("ERROR", None, QUIET, True, 5129, {"notes": ["partial"]}, "to resume", STOP),
    "sigkill": ("ERROR", None, QUIET, False, 5129, {"oom": True, "notes": ["sigkill-suspected"]}, "per job", STOP),
    "setup": ("ERROR", None, PIP, False, 40, {"notes": ["setup"]}, "fix the pins", STOP),
    "canceled": ("CANCELED", "partial", "", False, 300, {"notes": ["canceled"]}, "relaunch allowed", STOP),
    "deleted": ("DELETED", None, "", False, None, {"notes": ["canceled"]}, "relaunch allowed", STOP),
    "no record": ("COMPLETED", None, "last words", False, 60, {"notes": ["unclassified"]}, "last words", STOP),
    "ok after error": ("ERROR", "ok", 0, False, 60, {"notes": ["unclassified"]}, "record status ok", STOP),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS)
def test_every_terminal_row_sets_its_ledger_fields_and_next_action(world, row):
    """Each row of the terminal table, the marker row ahead of the kill row, by name and by member (TS-LCH-4)."""
    stage, status, said, marker, running_secs, flags, action, code = row
    exit_code, tail = (said, trailer(status, said)) if isinstance(said, int) else (None, said)
    record = run_record(RUN, status) if status else None
    facts = {"marker_present": marker, "running_secs": running_secs, "timeout_s": T}

    outcome = classify(stage, record, exit_code, tail, **facts)

    assert classify(JobStage(stage), record, exit_code, tail, **facts) == outcome
    expected = {"record_status": status, "exit_code": exit_code, **flags}
    assert outcome.ledger_fields == expected
    assert action in outcome.next_action and outcome.code == code
    pending = pending_line(world.plan(), "job-1", world.clock.now(), "settings")
    line = terminal_line(pending, running_secs=running_secs, stage=stage, ts=world.clock.now(), **outcome.ledger_fields)
    assert (line.pending, line.lost, line.oom) == (False, flags.get("lost", False), flags.get("oom", False))
    assert line.notes[1:] == flags.get("notes", []) + (["running-time-unknown"] if running_secs is None else [])
    if flags.get("notes") == ["unclassified"]:
        assert f"stage {stage}" in outcome.next_action and f"record status {status}" in outcome.next_action


def test_rows_at_their_edges():
    facts = {"marker_present": False, "running_secs": 100, "timeout_s": T}
    late = facts | {"running_secs": 5130, "marker_present": True}
    assert classify("ERROR", None, None, QUIET, **late).ledger_fields["lost"] is True
    spoke = classify("ERROR", None, 1, trailer("failed", 1), **facts)
    assert spoke.ledger_fields["notes"] == ["unclassified"]
    spoke_late = classify("ERROR", None, 1, trailer("failed", 1), **(facts | {"running_secs": 5399}))
    assert spoke_late.ledger_fields == {"record_status": None, "exit_code": 1, "notes": ["unclassified"]}
    paused = classify("PAUSED", None, None, QUIET, **facts)
    assert paused.ledger_fields["notes"] == ["unclassified"] and paused.code is ExitCode.UNEXPECTED


@pytest.mark.parametrize(("exit_code", "notes"), [(12, "integrity"), (13, "column-failed")])
def test_the_failed_row_takes_the_two_job_exit_codes_only(exit_code, notes):
    facts = {"marker_present": False, "running_secs": 100, "timeout_s": T}
    record = run_record(RUN, "failed")
    assert classify("ERROR", record, exit_code, trailer("failed", exit_code), **facts).ledger_fields["notes"] == [notes]
    for other in (0, 1, 2, 5, 10, 11, 14, 15, 99, None):
        tail = QUIET if other is None else trailer("failed", other)
        fields = classify("ERROR", record, other, tail, **facts).ledger_fields
        assert fields == {"record_status": "failed", "exit_code": other, "notes": ["unclassified"]}


@pytest.mark.parametrize("tail", ["", " \n\t\n"])
def test_a_job_that_left_no_log_is_lost_at_its_timeout_and_unclassified_below_it(tail):
    def row(running_secs, marker=False):
        return classify("ERROR", None, None, tail, marker_present=marker, running_secs=running_secs, timeout_s=T)

    for running_secs in (5130, 5400, 9000):
        late = row(running_secs)
        assert late.ledger_fields == {"record_status": None, "exit_code": None, "lost": True}
        assert late.code is ExitCode.LOST_RUN and row(running_secs, marker=True) == late
    for running_secs in (0, 100, 5129, None):
        blank = row(running_secs)
        assert blank.ledger_fields == {"record_status": None, "exit_code": None, "notes": ["unclassified", "no-log"]}
        assert blank.code is ExitCode.UNEXPECTED and "no log lines" in blank.next_action
        assert row(running_secs, marker=True).ledger_fields["notes"] == ["partial"]


CRASHES = [
    "step 1\nTraceback (most recent call last):\n  File \"run.py\", line 3, in <module>\nKeyError: 'rows'",
    "Traceback (most recent call last):\nModuleNotFoundError: No module named 'yaml'",
    "bash: line 1: python3.13: command not found",
    "starting\n/bin/sh: 1: pip: command not found\n",
]


@pytest.mark.parametrize("tail", CRASHES)
def test_a_crash_the_log_explains_is_not_blamed_on_host_memory(tail):
    facts = {"marker_present": False, "timeout_s": T}
    for running_secs in (1, 40, 5129):
        crashed = classify("ERROR", None, None, tail, running_secs=running_secs, **facts)
        assert crashed.ledger_fields == {"record_status": None, "exit_code": None, "notes": ["unclassified"]}
        assert tail.splitlines()[-1] in crashed.next_action
    assert classify("ERROR", None, None, tail, running_secs=5130, **facts).ledger_fields["lost"] is True


def test_a_job_that_never_ran_is_not_blamed_on_host_memory():
    for running_secs in (0, None):
        facts = {"marker_present": False, "running_secs": running_secs, "timeout_s": T}
        assert classify("ERROR", None, None, QUIET, **facts).ledger_fields["notes"] == ["unclassified"]
    ran = classify("ERROR", None, None, QUIET, marker_present=False, running_secs=1, timeout_s=T)
    assert ran.ledger_fields["notes"] == ["sigkill-suspected"]


INSTALL_FAILURES = [
    "ERROR: Could not find a version that satisfies the requirement safetune==0.1.99",
    "ERROR: No matching distribution found for safetune==0.1.99",
    "ERROR: Cannot install driftmap and trl==1.14.2 because these package versions have conflicting dependencies.",
    "ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/",
    "ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device",
    "ERROR: Failed building wheel for driftmap",
    "ERROR: HTTP error 404 while getting https://github.com/vincentni/tiny-aya-safety-drift-map/archive/a.tar.gz",
    "error: Failed to download `trl==1.14.2`",
    "error: Failed to build `driftmap @ https://github.com/vincentni/tiny-aya-safety-drift-map/archive/a.tar.gz`",
    "  × No solution found when resolving dependencies:",
    "Failed to pull image: ErrImagePull",
    "pull access denied for pytorch/pytorch, repository does not exist",
    "/usr/bin/python: No module named driftmap.scripts.generate",
]


@pytest.mark.parametrize("failure", INSTALL_FAILURES)
def test_an_install_or_pull_failure_is_the_setup_row(failure):
    tail = f"Collecting driftmap[jobs]\n{failure}\nexit 1"
    for running_secs in (0, 40, 900, None):
        setup = classify("ERROR", None, None, tail, marker_present=False, running_secs=running_secs, timeout_s=T)
        assert setup.ledger_fields["notes"] == ["setup"], running_secs
    slow = classify("ERROR", None, None, tail, marker_present=False, running_secs=901, timeout_s=T)
    assert slow.ledger_fields["notes"] == ["sigkill-suspected"]


@pytest.mark.parametrize(
    "line",
    [
        "2026-11-06 20:07:18,411 INFO driftmap.gen.rows: response quotes 'ERROR: No matching distribution found'",
        "2026-11-06 20:07:18,411 WARNING driftmap.gen.rows: prompt text: error: Failed to download the file",
        "ERROR: something else went wrong",
    ],
)
def test_an_error_word_inside_a_log_line_is_not_an_install_failure(line):
    quoted = classify("ERROR", None, None, f"step 1\n{line}", marker_present=False, running_secs=40, timeout_s=T)
    assert quoted.ledger_fields["notes"] == ["sigkill-suspected"]


@pytest.mark.parametrize(
    ("gap", "code", "appended"),
    [
        (500_001, ExitCode.UNEXPECTED, 0),
        (500_000, ExitCode.SUCCESS, 1),
        (499_999, ExitCode.SUCCESS, 1),
        (-600_000, ExitCode.SUCCESS, 1),
        (None, ExitCode.UNEXPECTED, 0),
    ],
)
def test_reconcile_compares_billing_with_the_ledger(world, capsys, gap, code, appended):
    """Billing over the ledger by more than 500,000 exits 1, as does no billing; else a line is added (TS-LCH-7)."""
    for job_id, kind, seconds in (("job-0", "steer", 2400), ("job-1", "generate", 1500)):
        pending = world.started(job_id, kind)
        world.clock.advance(1800)
        world.ledger.append(terminal_line(pending, running_secs=seconds, stage="COMPLETED", ts=world.clock.now()))
    released = hold_line(world.plan("drift"), world.clock.now(), "settings")
    world.ledger.append(released)
    world.clock.advance(60)
    world.ledger.append(terminal_line(released, running_secs=0, stage="RELEASED", ts=world.clock.now()))
    # 533,320 + 333,325
    assert world.ledger.total_micro_usd() == 866_645
    if gap is None:
        world.client.billing_status = 500
    else:
        world.client.billing_used_micro_usd = 866_645 + gap
    before = world.ledger.lines()
    events = []
    spied = world.deps_with(
        client=spy(world.client, events, world.settings.lock_path),
        ledger=spy(world.ledger, events, world.settings.lock_path, only=("read", "lines", "total_micro_usd")),
    )

    assert reconcile(spied) == code

    assert [label for label, _ in events] == ["billing_usage", "read"]

    added = world.ledger.lines()[len(before) :]
    assert len(added) == appended and world.ledger.total_micro_usd() == 866_645
    out = capsys.readouterr().out
    if gap is None:
        assert "unavailable" in out
        return
    assert f"drift {gap} micro-USD" in out
    for line in added:
        assert (line.job_id, line.pending, line.cost_micro_usd, line.minutes) == ("job-1", False, 333_325, 25)
        assert line.notes == ["price=settings", "reconcile"]
        assert line.billing_used_after_micro_usd == 866_645 + gap


def test_reconcile_beside_a_pending_job_exits_1_and_names_it(world, capsys):
    done = world.started("job-1")
    world.ledger.append(terminal_line(done, running_secs=1500, stage="COMPLETED", ts=world.clock.now()))
    world.started("job-2", "steer")
    world.started("job-3", "quant")
    # the pending lines would cover 1,229,970 of unrecorded spend
    world.client.billing_used_micro_usd = 333_325 + 600_000
    before = world.ledger.path.read_bytes()

    assert reconcile(world.deps) is ExitCode.UNEXPECTED

    out = capsys.readouterr().out
    assert "job-2, job-3" in out and "driftmap wait" in out and "job-1" not in out
    assert world.ledger.path.read_bytes() == before and world.count("billing_usage") == 1


def test_reconcile_on_an_empty_ledger_adds_nothing(world):
    world.client.billing_used_micro_usd = 10_000
    assert reconcile(world.deps) is ExitCode.SUCCESS
    assert not world.ledger.path.exists()


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_an_out_of_memory_run_is_marked_and_prints_the_next_rung(world, capsys, kind):
    """A recorded out-of-memory run sets oom, names the kind's next rung and submits nothing (TS-LCH-8)."""
    run_id = f"{kind}-full-20261102T100000Z"
    world.job("job-1", "ERROR", running_secs=700, logs=["CUDA out of memory", trailer("oom", 11)])
    world.started("job-1", kind)
    world.push_record(run_id, "oom")

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.UNEXPECTED

    line = world.ledger.read()["job-1"]
    assert (line.oom, line.lost, line.record_status, line.exit_code) == (True, False, "oom", 11)
    assert world.ledger.oom_runs() == [(kind, world.settings.config_hash())]
    rung = {"drift": "2 x 8", "generate": "16 -> 8", "repair": "8 -> 4, then pass --alphas one"}
    out = capsys.readouterr().out
    assert rung.get(kind, NO_RUNG) in out and OOM_LADDER.get(kind, NO_RUNG) in out
    assert len(world.submits()) == 1


def test_a_run_lost_at_its_timeout_blocks_the_same_timeout(world, capsys):
    """Silent death near the timeout is lost: the same timeout is refused, a wider one is admitted (TS-LCH-9)."""
    world.git_ok(3)
    world.smoke_passed()
    world.job("job-1", "ERROR", running_secs=5130, logs=["step 900"])

    assert launch(world.settings, world.deps, "generate", "full", wait=True) is ExitCode.LOST_RUN

    line = world.ledger.read()["job-1"]
    assert (line.lost, line.oom, line.minutes, line.cost_micro_usd) == (True, False, 86, 1_146_638)
    capsys.readouterr()
    world.clock.advance(120)
    assert launch(world.settings, world.deps, "generate", "full") is ExitCode.GUARD_REFUSAL
    out = capsys.readouterr().out
    refused = [row.split()[0] for row in out.splitlines() if " refused " in row]
    assert refused == ["timeout_lost"] and "nothing was submitted" in out
    assert len(world.submits()) == 1
    wider = load_settings(settings_toml(world.tmp_path, timeouts={"generate": 135}))
    assert launch(wider, world.deps, "generate", "full") is ExitCode.SUCCESS
    assert len(world.submits()) == 2


def test_a_silent_early_death_points_at_host_memory_unless_a_marker_was_pushed(world, capsys):
    """No record, no trailer, well short of the timeout and no marker: oom with the host-memory rung (TS-LCH-10)."""
    world.job("job-1", "ERROR", running_secs=5129, logs=["step 900"])
    world.started("job-1", "repair")

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.UNEXPECTED

    line = world.ledger.read()["job-1"]
    assert (line.oom, line.lost, line.notes[1:]) == (True, False, ["sigkill-suspected"])
    assert HOST_RAM_RUNG in capsys.readouterr().out
    assert "--alphas one value per job" in HOST_RAM_RUNG

    world.clock.advance(60)
    world.job("job-2", "ERROR", running_secs=5129, logs=["step 900"])
    flushed = world.started("job-2", "repair")
    world.push(f"partial/{flushed.run_id}.json", b"{}")
    assert wait_job(world.settings, world.deps, "job-2") is ExitCode.UNEXPECTED
    line = world.ledger.read()["job-2"]
    assert (line.oom, line.notes[1:]) == (False, ["partial"])
    assert "resume" in capsys.readouterr().out


def test_a_chain_checks_inputs_first_and_stops_at_the_first_job_that_is_not_ok(tmp_path, capsys):
    """A missing input is exit 4 before any call; a failed job ends the chain; each job is re-checked (TS-LCH-11)."""
    blocked = World(tmp_path / "blocked")
    drifted = State(languages=("en",), checkpoint="drifted")
    assert chain(blocked.settings, blocked.deps, ["generate", "steer"], state=lambda kind: drifted) == 4
    assert "merged_sha256" in capsys.readouterr().out
    assert blocked.client.calls == [] and blocked.git.calls == [] and not blocked.ledger.path.exists()

    ready = State(languages=("en",), steer_comparison=True)
    failing = World(tmp_path / "failing")
    failing.git_ok()
    failing.smoke_passed()
    failing.job("job-1", "ERROR", logs=[trailer("failed", 13)])
    failing.push_record(RUN, "failed")
    asked = []

    def state(kind):
        asked.append(kind)
        return ready

    assert chain(failing.settings, failing.deps, ["generate", "steer"], state=state) is ExitCode.UNEXPECTED
    assert "chain stopped at generate" in capsys.readouterr().out
    assert asked == ["generate"] and len(failing.submits()) == 1
    assert failing.ledger.read()["job-1"].notes[1:] == ["column-failed"]

    passing = World(tmp_path / "passing")
    passing.git_ok(2)
    for kind, job_id in (("generate", "job-1"), ("steer", "job-2")):
        passing.smoke_passed(kind)
        passing.job(job_id, "COMPLETED", logs=[trailer("ok", 0)])
        passing.push_record(f"{kind}-full-20261102T100000Z", "ok")
    seen = []

    def state_after(kind):
        seen.append((kind, len(passing.ledger.lines())))
        return ready

    extra = {"steer": ["--layer", "12"]}
    assert chain(passing.settings, passing.deps, ["generate", "steer"], state=state_after, args=extra) == 0
    assert seen == [("generate", 0), ("steer", 4)]
    assert passing.count("list_jobs") == 2 and passing.count("billing_usage") == 4
    commands = [kwargs["command"][-1] for _, kwargs in passing.submits()]
    assert ["driftmap.scripts.generate" in commands[0], "driftmap.scripts.steer" in commands[1]] == [True, True]
    assert commands[1].endswith("--layer 12") and "--layer" not in commands[0]
    assert [line.pending for line in passing.job_lines()] == [False, False]
    assert [(line.pending, line.cost_micro_usd) for line in passing.holds()] == [(False, 0), (False, 0)]


def test_the_log_tail_is_saved_masked_and_billing_is_recorded(world, capsys):
    """The last 50 log lines are saved with tokens and home paths masked, and billing lands on the line (TS-LCH-14)."""
    text = LOG_FIXTURE.read_text(encoding="utf-8").replace("{token}", TOKEN).replace("{home}", HOME_DIR)
    logs = text.splitlines()
    assert len(logs) == 60 and sum(TOKEN in line for line in logs[-50:]) == 1
    world.job("job-1", "COMPLETED", logs=logs)
    world.started("job-1")
    world.push_record(RUN, "ok")
    world.client.billing_used_micro_usd = 2_345_678

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.SUCCESS

    (kwargs,) = [kwargs for name, kwargs in world.client.calls if name == "fetch_job_logs"]
    assert (kwargs["job_id"], kwargs["tail"]) == ("job-1", 50)
    saved = (world.root / LOGS_DIR / "job-1.log").read_text(encoding="utf-8")
    assert saved.endswith("\n") and len(saved.splitlines()) == 50
    assert saved.splitlines()[-1] == trailer("ok", 0)
    assert TOKEN not in saved and HOME_DIR not in saved
    assert "HF_TOKEN=hf_***" in saved and "<home>/work/out" in saved
    line = world.ledger.read()["job-1"]
    assert (line.billing_used_after_micro_usd, line.exit_code) == (2_345_678, 0)
    assert TOKEN not in capsys.readouterr().out


CHILD = """
import subprocess, sys, time
from pathlib import Path

from driftmap.core.clients import make_jobs_client
from driftmap.core.hub import Hub
from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.jobs.launcher import LEDGER_PATH, launch
from driftmap.jobs.ledger import Ledger
from driftmap.jobs.snapshot import Deps
from tests.fakes import FakeHub

config, root, *hold = sys.argv[1:]
root = Path(root)
settings = load_settings(config)
fake = FakeHub(root.parent / "hub")
fake.add_repo("CohereLabs/tiny-aya-global", gated=True).add_repo("CohereLabs/tiny-aya-base", gated=True)
fake.authorize("CohereLabs/tiny-aya-global")


def run(argv, **kwargs):
    # the launcher's first git call is made with the lock held
    if hold:
        Path(hold[0]).touch()
        while not Path(hold[1]).exists():
            time.sleep(0.05)
    return subprocess.run(argv, **kwargs)


hub = Hub(fake, gated_repos=settings.gated_repos)
deps = Deps(make_jobs_client(settings), hub, Ledger(root / LEDGER_PATH), root, run=run)
try:
    sys.exit(int(launch(settings, deps, "generate", "smoke")))
except DriftmapError as error:
    print(error)
    sys.exit(int(error.exit_code))
"""


@pytest.fixture
def checkout(tmp_path):
    """A clean checkout whose HEAD is on a local repo that stands in for the code URL."""
    config = settings_toml(tmp_path)
    origin = load_settings(config).code.origin

    def git(*args):
        argv = ["git", "-c", "user.name=t", "-c", "user.email=t", *args]
        subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True, timeout=60)

    work = tmp_path / "work-tree"
    git("init", "-q", "--bare", "-b", "main", "origin.git")
    git("init", "-q", "-b", "main", str(work))
    # a machine-wide excludes file may hide a .gitignore
    (work / ".git" / "info" / "exclude").write_text(".run-local/\n")
    shutil.copy(REPO_ROOT / "pyproject.toml", work)
    git("-C", str(work), "add", "pyproject.toml")
    git("-C", str(work), "commit", "-q", "-m", "a")
    git("-C", str(work), "remote", "add", "origin", str(tmp_path / "origin.git"))
    git("-C", str(work), "push", "-q", "origin", "main")
    git("-C", str(work), "config", f"url.{tmp_path / 'origin.git'}.insteadOf", origin)
    return config, work


def until(path, process):
    deadline = time.monotonic() + 60
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


def test_two_launches_at_once_submit_one_job(tmp_path, checkout, child_env):
    """The lock comes before any read: the loser exits 4 with no call made; a killed holder frees it (TS-LCH-15)."""
    config, work = checkout
    settings = load_settings(config)
    assert settings.lock_path == tmp_path / "launch.lock"
    items = [Item(number=number, name=name, status="PASS", detail="ok") for number, name, _ in ITEMS]
    dump(ReadyReport(ts=datetime.now(UTC), config_hash=settings.config_hash(), items=items), work / READY_PATH)
    calls = tmp_path / "client.jsonl"
    script = {"jobs": {"job-a": [{"stage": "RUNNING"}]}, "billing_used_micro_usd": 0}
    calls.with_name("client.jsonl.script.json").write_text(json.dumps(script))
    env = {**child_env, "GIT_ALLOW_PROTOCOL": "file"}

    def child(*hold):
        argv = [sys.executable, "-c", CHILD, str(config), str(work), *map(str, hold)]
        return subprocess.Popen(argv, env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def finish(process):
        out, err = process.communicate(timeout=60)
        return process.returncode, out, err

    killed = child(tmp_path / "k.locked", tmp_path / "k.release")
    until(tmp_path / "k.locked", killed)
    assert lock_held(settings.lock_path)
    os.kill(killed.pid, signal.SIGKILL)
    killed.wait(timeout=60)
    assert not lock_held(settings.lock_path)

    holder = child(tmp_path / "h.locked", tmp_path / "h.release")
    until(tmp_path / "h.locked", holder)
    code, out, err = finish(child())
    assert code == 4, err
    assert "another launch holds the lock" in out
    assert not calls.exists() and not (work / LEDGER_PATH).exists()
    (tmp_path / "h.release").touch()
    code, out, err = finish(holder)
    assert code == 0, err
    assert "submitted generate (smoke) as job job-a" in out
    run = out.split(", run ")[1].split(";")[0]

    code, out, err = finish(child())
    assert code == 4, err
    refused = {row.split()[0]: row for row in out.splitlines() if " refused " in row}
    assert "job-a" in refused["in_flight_gpu"]
    methods = [json.loads(line)["method"] for line in calls.read_text().splitlines()]
    assert methods.count("run_job") == 1 and methods.count("billing_usage") == 2
    ledger = Ledger(work / LEDGER_PATH).read()
    assert [(job_id, line.pending) for job_id, line in ledger.items()] == [(f"{HOLD}{run}", False), ("job-a", True)]


def pinned():
    git = FakeSubprocess()
    git.script(stdout=SHA + "\n")
    for _ in range(3):
        git.script()
    return git


def test_a_passed_smoke_run_leaves_a_record_the_smoke_guard_accepts(world):
    """An ok smoke run writes the kind's smoke record with the run's measurements and status (TS-LCH-17)."""
    world.git_ok()
    world.git.script(stdout=SHA + "\n")
    world.git.script()
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    measured = {"bos_check": True, "upload_mb_s": 11.5, "install_minutes": 3.25, "tokens_per_s": 41.5}
    world.push_record("quant_eval-smoke-20261102T100000Z", "ok", **measured, peak_vram_gb=7.5, peak_rss_gb=5.25)
    full = world.plan("quant_eval")
    before = take_snapshot(world.settings, world.client, world.deps.hub, world.ledger, world.root, run=pinned().run)
    assert guards.smoke(before, full).guard == "smoke"

    assert launch(world.settings, world.deps, "quant_eval", "smoke", wait=True) is ExitCode.SUCCESS

    record = load(SmokeRecord, world.root / SMOKE_DIR / "quant_eval.json")
    assert record.status == "ok" == run_record("quant_eval-smoke-20261102T100000Z", "ok").status
    assert {name: getattr(record, name) for name in measured} == measured
    assert (record.peak_vram_gb, record.peak_rss_gb) == (7.5, 5.25)
    assert (record.script, record.image, record.runner) == (full.script, full.image, full.runner)
    assert record.dependencies == pins(world.root) and record.schema_versions == schema_versions()
    after = before.model_copy(update={"smoke_records": read_smoke_records(world.root / SMOKE_DIR)})
    assert guards.smoke(after, full) is None


@pytest.mark.parametrize(
    ("head", "config_hash"), [("b" * 40, None), (SHA, "d" * 64), ("", None)], ids=["head moved", "config", "no git"]
)
def test_no_smoke_record_when_the_checkout_or_config_is_not_what_the_job_ran(world, capsys, head, config_hash):
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    world.started("job-1", "drift", "smoke")
    world.push_record("drift-smoke-20261102T100000Z", "ok", **({"config_hash": config_hash} if config_hash else {}))
    world.git.script(stdout=head + "\n", returncode=0 if head else 128)

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.SUCCESS

    assert not (world.root / SMOKE_DIR / "drift.json").exists()
    out = capsys.readouterr().out
    assert "no smoke record written" in out and f"git checkout {SHA}" in out and "driftmap wait job-1" in out
    assert world.ledger.read()["job-1"].pending is False


@pytest.mark.parametrize(("status", "returncode"), [(" M pyproject.toml\n", 0), ("?? uv.lock\n", 0), ("", 128)])
def test_no_smoke_record_from_a_tree_with_uncommitted_pins_or_config(world, capsys, status, returncode):
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    world.started("job-1", "drift", "smoke")
    world.push_record("drift-smoke-20261102T100000Z", "ok")
    world.git.script(stdout=SHA + "\n")
    world.git.script(stdout=status, returncode=returncode)

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.SUCCESS

    assert not (world.root / SMOKE_DIR / "drift.json").exists()
    assert "no smoke record written" in capsys.readouterr().out
    argv = world.git.calls[-1].argv
    assert argv[3:6] == ["status", "--porcelain", "--"]
    assert argv[6:] == ["pyproject.toml", "uv.lock", "driftmap.toml", "revisions.json"]


def test_the_smoke_record_names_the_image_the_job_ran_on(world):
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    world.started("job-1", "drift", "smoke")
    world.push_record("drift-smoke-20261102T100000Z", "ok")
    world.git.script(stdout=SHA + "\n")
    world.git.script()
    newer = "pytorch/pytorch:2.15.0-cuda12.6-cudnn9-runtime"
    edited = load_settings(settings_toml(world.tmp_path, images={"pytorch": newer}))
    assert edited.config_hash() == world.settings.config_hash()

    assert wait_job(edited, world.deps, "job-1") is ExitCode.SUCCESS

    record = load(SmokeRecord, world.root / SMOKE_DIR / "drift.json")
    assert record.image == world.settings.images.pytorch != newer
    full = build_plan(edited, "drift", "full", None, SHA, now=world.clock.now(), repo_root=world.root)
    stale = take_snapshot(edited, world.client, world.deps.hub, world.ledger, world.root, run=pinned().run)
    assert full.image == newer and guards.smoke(stale, full).guard == "smoke"


@pytest.mark.parametrize(("status", "stage"), [("partial", "ERROR"), ("ok", "ERROR"), ("failed", "ERROR")])
def test_a_smoke_run_that_did_not_end_ok_leaves_no_smoke_record(world, status, stage):
    world.job("job-1", stage, logs=[trailer(status, 1)])
    world.started("job-1", "drift", "smoke")
    world.push_record("drift-smoke-20261102T100000Z", status)
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.UNEXPECTED
    assert not (world.root / SMOKE_DIR / "drift.json").exists()


def test_a_refusal_prints_every_verdict_and_submits_nothing(world, capsys):
    world.git_ok(2)
    world.client.billing_used_micro_usd = 15_000_000

    for dry_run in (False, True):
        assert launch(world.settings, world.deps, "generate", "full", dry_run=dry_run) is ExitCode.GUARD_REFUSAL
        rows = {row.split()[0]: row.split() for row in capsys.readouterr().out.splitlines() if row.split()}
        assert [name for name in GUARDS if rows[name][1] == "refused"] == ["smoke", "ledger_ceiling"]
        assert rows["ledger_ceiling"][2:4] == ["1199970", "16000000"]
        assert [rows[name][1] for name in GUARDS].count("pass") == 7
        assert rows["refused"][:4] == ["refused", "by", "2", "guard(s);"]
    assert world.submits() == [] and not world.ledger.path.exists()


def test_a_dry_run_prints_the_plan_the_verdicts_and_the_headroom(world, capsys):
    world.git_ok()
    world.smoke_passed()
    world.started("job-0", "quant")
    world.client.billing_used_micro_usd = 4_000_000
    before = (world.ledger.path.read_bytes(), len(world.submits()))

    assert launch(world.settings, world.deps, "generate", "full", dry_run=True) is ExitCode.SUCCESS

    out = capsys.readouterr().out
    rows = {row.split()[0]: row for row in out.splitlines() if row.split()}
    assert "HF_TOKEN=***" in rows["secrets"] and "hf_dummy" not in out
    assert rows["command"].endswith(f"--config-hash {world.settings.config_hash()}")
    assert "1199970 micro-USD" in rows["estimate"] and "(hardware)" in rows["estimate"]
    assert rows["ledger"].split()[1] == "30000"
    assert rows["credit"].split()[1] == "16000000"
    assert rows["headroom"].split()[1] == "12000000"
    assert [rows[name].split()[1] for name in GUARDS] == ["pass"] * 9
    assert (world.ledger.path.read_bytes(), len(world.submits())) == before
    assert not lock_held(world.settings.lock_path)


def test_no_readiness_report_stops_before_the_lock_and_any_read(world):
    (world.root / READY_PATH).unlink()
    with pytest.raises(NotReady):
        launch(world.settings, world.deps, "generate", "smoke")
    assert world.client.calls == [] and world.git.calls == [] and not world.settings.lock_path.exists()


def test_a_held_lock_refuses_before_any_snapshot_read(world):
    world.settings.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(world.settings.lock_path, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        with pytest.raises(LockHeld) as refused:
            launch(world.settings, world.deps, "generate", "smoke")
    assert refused.value.exit_code is ExitCode.GUARD_REFUSAL
    assert world.client.calls == [] and world.git.calls == [] and world.fake_hub.calls == []


def test_a_head_that_moves_after_the_launcher_read_it_is_refused(world, capsys):
    world.git.script(stdout=SHA + "\n")
    world.git.script(stdout="b" * 40 + "\n")
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.GUARD_REFUSAL
    refused = [row.split()[0] for row in capsys.readouterr().out.splitlines() if " refused " in row]
    assert refused == ["code_pinned"] and world.submits() == []


def test_outside_a_git_checkout_nothing_is_read_or_submitted(world):
    world.git.script(returncode=128, stderr="fatal: not a git repository")
    with pytest.raises(DriftmapError) as refused:
        launch(world.settings, world.deps, "generate", "smoke")
    assert refused.value.exit_code is ExitCode.GUARD_REFUSAL
    assert world.client.calls == [] and not lock_held(world.settings.lock_path)


class FlakySubmit(FakeJobsClient):
    """A submit that raises, after the job was or was not created; the job list may trail or fail."""

    def __init__(self, error, *, created, trailing=0, list_error=None):
        super().__init__()
        self.error, self.created, self.trailing, self.list_error = error, created, trailing, list_error
        self.submitted = False

    def run_job(self, **kwargs):
        self.submitted = True
        if self.created:
            super().run_job(**kwargs)
        else:
            self._record("run_job", **kwargs)
        raise self.error

    def list_jobs(self, **kwargs):
        jobs = super().list_jobs(**kwargs)
        if self.submitted and self.list_error:
            raise self.list_error
        if self.submitted and self.trailing:
            self.trailing -= 1
            return []
        return jobs


def flaky(tmp_path, error, **kwargs):
    world = World(tmp_path, client=FlakySubmit(error, **kwargs))
    world.git_ok()
    world.client.script_job("job-1", [{"stage": "RUNNING"}])
    return world


def smoke_launch(world, **kwargs):
    return launch(world.settings, world.deps, "generate", "smoke", sleep=world.clock.sleep, **kwargs)


TIMED_OUT, DOWN = httpx2.ReadTimeout("timed out"), httpx2.ConnectError("refused")


@pytest.mark.parametrize(
    ("error", "created", "trailing", "lookups", "sleeps", "ends"),
    [
        (TIMED_OUT, True, 0, 1, [], "adopted"),
        (TIMED_OUT, True, 1, 2, [60], "adopted"),
        (ValueError("bad reply"), True, 0, 1, [], "adopted"),
        (KeyError("id"), True, 1, 2, [60], "adopted"),
        (OSError(104, "reset"), True, 0, 1, [], "adopted"),
        (http_error(503), False, 0, 2, [60], "held"),
        (DOWN, True, 2, 2, [60], "held"),
        (TypeError("no id"), False, 0, 2, [60], "held"),
        (http_error(429), False, 0, 1, [], "refused"),
        (http_error(409), False, 1, 1, [], "refused"),
        (http_error(402), True, 0, 1, [], "adopted"),
        (http_error(408), True, 1, 2, [60], "adopted"),
    ],
    ids=["listed", "late", "bad reply", "key error", "reset", "5xx", "never", "type error", "429", "409", "402", "408"],
)
def test_a_failed_submit_is_never_repeated(tmp_path, capsys, no_sleep, error, created, trailing, lookups, sleeps, ends):
    world = flaky(tmp_path, error, created=created, trailing=trailing)
    hold = f"{HOLD}{SMOKE_RUN}"

    if ends == "adopted":
        assert smoke_launch(world) is ExitCode.SUCCESS
        (line,) = world.job_lines()
        assert (line.job_id, line.pending, line.cost_micro_usd) == ("job-1", True, 399_990)
        (closed,) = world.holds()
        assert (closed.pending, closed.cost_micro_usd, closed.notes[-1]) == (False, 0, "job=job-1")
    else:
        with pytest.raises(DriftmapError) as failed:
            smoke_launch(world)
        assert failed.value.exit_code is ExitCode.UNEXPECTED
        assert world.job_lines() == []
        (left,) = world.holds()
    if ends == "held":
        assert (left.job_id, left.pending, left.cost_micro_usd) == (hold, True, 399_990)
        assert SMOKE_RUN in str(failed.value) and f"driftmap wait {hold}" in str(failed.value)
        assert "hf jobs ps -a" in str(failed.value) and "in-flight GPU guard" in str(failed.value)
    if ends == "refused":
        assert (left.pending, left.cost_micro_usd, left.notes[-1]) == (False, 0, "not-submitted")
        assert "no job was created" in str(failed.value)
    said = capsys.readouterr().out
    assert (f"submit of run {SMOKE_RUN} failed" in said and "checking the job list" in said) == bool(lookups)
    assert len(world.submits()) == 1
    assert world.count("list_jobs") == 1 + lookups and world.clock.sleeps == sleeps
    assert not lock_held(world.settings.lock_path)


@pytest.mark.parametrize("created", [True, False])
def test_an_interrupt_during_the_submit_still_looks_for_the_job(tmp_path, capsys, created):
    world = flaky(tmp_path, KeyboardInterrupt(), created=created)

    with pytest.raises(KeyboardInterrupt):
        smoke_launch(world)

    out = capsys.readouterr().out
    assert f"submit of run {SMOKE_RUN} failed (KeyboardInterrupt); checking the job list" in out
    (hold,) = world.holds()
    if created:
        (line,) = world.job_lines()
        assert (line.job_id, line.pending, hold.pending) == ("job-1", True, False)
        assert "job job-1 carries the label" in out
    else:
        assert world.job_lines() == [] and (hold.pending, hold.cost_micro_usd) == (True, 399_990)
        assert f"driftmap wait {hold.job_id}" in out and "hf jobs ps -a" in out
    assert len(world.submits()) == 1 and world.count("list_jobs") == 2 and world.clock.sleeps == []


def test_an_interrupt_while_waiting_for_the_job_list_has_already_said_what_to_run(tmp_path, capsys):
    world = flaky(tmp_path, TIMED_OUT, created=True, trailing=1)

    def interrupted(seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        launch(world.settings, world.deps, "generate", "smoke", sleep=interrupted)

    out = capsys.readouterr().out
    (hold,) = world.holds()
    assert f"submit of run {SMOKE_RUN} failed" in out and f"driftmap wait {hold.job_id}" in out
    assert (hold.pending, hold.cost_micro_usd) == (True, 399_990) and world.job_lines() == []


def test_a_job_list_that_cannot_be_read_after_a_failed_submit_adopts_nothing(tmp_path, capsys):
    world = flaky(tmp_path, TIMED_OUT, created=True, list_error=RuntimeError("no list"))
    with pytest.raises(DriftmapError):
        smoke_launch(world)
    (hold,) = world.holds()
    assert hold.pending and world.job_lines() == [] and world.clock.sleeps == [60]
    assert "the job list could not be read (RuntimeError)" in capsys.readouterr().out


def test_only_a_job_with_this_runs_label_is_adopted(tmp_path):
    world = flaky(tmp_path, http_error(503), created=False)
    world.client.script_job("job-0", [{"stage": "COMPLETED"}])
    world.client._pending_ids.rotate(1)
    world.client._accept(dockerImage="i", command=["true"], flavor="l4x1", labels={launcher.RUN_LABEL: RUN})

    with pytest.raises(DriftmapError):
        smoke_launch(world)

    assert [(job.id, job.labels) for job in world.client.list_jobs()] == [("job-0", {launcher.RUN_LABEL: RUN})]
    assert world.job_lines() == [] and world.holds()[0].pending


class Ledgers:
    """A ledger whose append fails for chosen lines."""

    def __init__(self, ledger, error, failing):
        self.ledger, self.error, self.failing = ledger, error, failing

    def __getattr__(self, name):
        return getattr(self.ledger, name)

    def append(self, line):
        if self.failing(line):
            raise self.error
        self.ledger.append(line)


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), ValueError("bad line"), KeyboardInterrupt()])
def test_a_pending_line_that_cannot_be_written_leaves_the_hold_and_names_the_job(world, capsys, error):
    world.git_ok(2)
    world.client.script_job("job-1", [{"stage": "RUNNING"}])
    broken = Ledgers(world.ledger, error, lambda line: line.job_id == "job-1")
    hold = f"{HOLD}{SMOKE_RUN}"

    with pytest.raises(BaseException) as failed:
        launch(world.settings, world.deps_with(ledger=broken), "generate", "smoke", wait=True)

    told = capsys.readouterr().out if isinstance(error, KeyboardInterrupt) else str(failed.value)
    assert type(failed.value) is (KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else DriftmapError)
    assert "hf jobs cancel job-1" in told and f"driftmap wait {hold}" in told
    (left,) = world.holds()
    assert (left.pending, left.cost_micro_usd) == (True, 399_990) and world.job_lines() == []
    assert len(world.submits()) == 1 and world.count("wait_for_job") == 0

    world.clock.advance(60)
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.GUARD_REFUSAL
    refused = {row.split()[0]: row for row in capsys.readouterr().out.splitlines() if " refused " in row}
    assert hold in refused["in_flight_gpu"] and len(world.submits()) == 1

    assert wait_job(world.settings, world.deps, hold) is ExitCode.SUCCESS
    assert "driftmap wait job-1" in capsys.readouterr().out
    (line,) = world.job_lines()
    assert (line.job_id, line.pending, line.cost_micro_usd, line.notes) == ("job-1", True, 399_990, ["price=hardware"])
    assert line.run_id == SMOKE_RUN and line.ts == world.clock.now()
    (closed,) = world.holds()
    assert (closed.pending, closed.cost_micro_usd, closed.notes[-1]) == (False, 0, "job=job-1")
    assert world.ledger.total_micro_usd() == 399_990
    settled, reads = world.ledger.path.read_bytes(), world.count("list_jobs")
    assert wait_job(world.settings, world.deps, hold) is ExitCode.SUCCESS
    assert world.ledger.path.read_bytes() == settled and "already closed" in capsys.readouterr().out
    assert world.count("list_jobs") == reads


def test_a_ledger_that_cannot_be_written_stops_the_launch_before_the_submit(world):
    world.git_ok()
    broken = Ledgers(world.ledger, OSError(28, "No space left on device"), lambda line: True)
    with pytest.raises(OSError, match="No space left"):
        launch(world.settings, world.deps_with(ledger=broken), "generate", "smoke")
    assert world.submits() == [] and not lock_held(world.settings.lock_path)


def test_a_hold_that_cannot_be_closed_stays_at_full_cost_and_says_so(world, capsys):
    world.git_ok()
    broken = Ledgers(world.ledger, OSError(5, "Input/output error"), lambda line: line.stage == "RELEASED")
    assert launch(world.settings, world.deps_with(ledger=broken), "generate", "smoke") is ExitCode.SUCCESS
    (hold,) = world.holds()
    (line,) = world.job_lines()
    assert hold.pending and line.pending and world.ledger.total_micro_usd() == 799_980
    assert f"driftmap wait {hold.job_id}" in capsys.readouterr().out
    assert wait_job(world.settings, world.deps, hold.job_id) is ExitCode.SUCCESS
    assert world.ledger.total_micro_usd() == 399_990 and len(world.ledger.lines()) == 3


def test_a_refused_submit_with_an_unreadable_job_list_keeps_the_hold(tmp_path, capsys):
    world = flaky(tmp_path, http_error(429), created=True, list_error=http_error(503))
    with pytest.raises(DriftmapError) as failed:
        smoke_launch(world)
    (hold,) = world.holds()
    assert (hold.pending, hold.cost_micro_usd) == (True, 399_990) and world.job_lines() == []
    assert f"driftmap wait {hold.job_id}" in str(failed.value) and "no job was created" not in str(failed.value)
    assert len(world.submits()) == 1 and world.clock.sleeps == []


def resolve(world, hold):
    return wait_job(world.settings, world.deps, hold.job_id, sleep=world.clock.sleep)


def test_a_hold_is_released_only_when_old_and_twice_unlisted(tmp_path, capsys):
    world = flaky(tmp_path, http_error(503), created=False)
    with pytest.raises(DriftmapError):
        smoke_launch(world)
    (hold,) = world.holds()
    assert hold.ts == datetime(2026, 11, 2, 10, 0, tzinfo=UTC) and HOLD_GRACE.total_seconds() == 900
    reads, sleeps = world.count("list_jobs"), list(world.clock.sleeps)
    capsys.readouterr()

    # 60 s, then 14 min 59 s after the hold was written
    for wait_s in (839, 1):
        assert resolve(world, hold) is ExitCode.LOST_RUN
        out = capsys.readouterr().out
        assert "10:15 UTC" in out and f"driftmap wait {hold.job_id}" in out and "hf jobs ps -a" in out
        world.clock.advance(wait_s)
    assert world.count("list_jobs") == reads + 2 and world.clock.sleeps == sleeps
    assert world.holds() == [hold] and world.ledger.total_micro_usd() == 399_990

    assert world.clock.now() - hold.ts == HOLD_GRACE
    for error in (http_error(502), RuntimeError("no list")):
        world.client.list_error = error
        assert resolve(world, hold) is ExitCode.LOST_RUN
        assert "could not be read" in capsys.readouterr().out and world.holds() == [hold]
    world.client.list_error = None
    reads, sleeps = world.count("list_jobs"), list(world.clock.sleeps)

    assert resolve(world, hold) is ExitCode.SUCCESS

    assert world.count("list_jobs") == reads + 2 and world.clock.sleeps == [*sleeps, 60]
    (closed,) = world.holds()
    assert (closed.pending, closed.cost_micro_usd, closed.stage) == (False, 0, "RELEASED")
    assert closed.notes[-1] == "no-job-found"
    assert world.ledger.total_micro_usd() == 0 and world.job_lines() == []
    assert world.count("wait_for_job") == 0 and len(world.submits()) == 1
    assert "driftmap wait <job id>" in capsys.readouterr().out

    reads, released = world.count("list_jobs"), world.ledger.path.read_bytes()
    assert resolve(world, hold) is ExitCode.SUCCESS
    out = capsys.readouterr().out
    assert "already closed" in out and "driftmap wait <job id>" in out and "hf jobs ps -a" in out
    assert world.count("list_jobs") == reads + 1 and world.ledger.path.read_bytes() == released
    world.client.list_error = http_error(502)
    assert resolve(world, hold) is ExitCode.LOST_RUN
    assert "could not be read" in capsys.readouterr().out and world.ledger.path.read_bytes() == released


def test_a_job_that_shows_up_after_its_hold_was_released_is_recorded_by_the_same_verb(tmp_path, capsys):
    world = flaky(tmp_path, DOWN, created=True, trailing=4)
    with pytest.raises(DriftmapError):
        smoke_launch(world)
    (hold,) = world.holds()
    world.clock.advance(900)
    assert resolve(world, hold) is ExitCode.SUCCESS
    assert world.ledger.total_micro_usd() == 0
    world.clock.advance(300)
    capsys.readouterr()

    assert resolve(world, hold) is ExitCode.SUCCESS

    assert "driftmap wait job-1" in capsys.readouterr().out
    (line,) = world.job_lines()
    assert line == adopted_line(hold, "job-1", world.clock.now())
    assert world.holds()[0].notes[-1] == "no-job-found" and len(world.ledger.lines()) == 3
    assert world.ledger.total_micro_usd() == 399_990
    again = world.ledger.path.read_bytes()
    assert resolve(world, hold) is ExitCode.SUCCESS
    assert world.ledger.path.read_bytes() == again


@pytest.mark.parametrize(("age_s", "trailing"), [(0, 2), (899, 2), (900, 2), (0, 3), (899, 3), (900, 3), (5000, 3)])
def test_a_hold_is_never_released_while_its_job_is_only_late_on_the_list(tmp_path, capsys, age_s, trailing):
    world = flaky(tmp_path, DOWN, created=True, trailing=trailing)
    with pytest.raises(DriftmapError):
        smoke_launch(world)
    (hold,) = world.holds()
    world.clock.advance(age_s - 60)
    assert world.clock.now() - hold.ts == timedelta(seconds=age_s)

    code = resolve(world, hold)

    # two reads were spent by the launch; the job shows on read `trailing` + 1
    listed_now, listed_on_the_second_read = trailing == 2, age_s >= 900
    if listed_now or listed_on_the_second_read:
        assert code is ExitCode.SUCCESS
        (line,) = world.job_lines()
        assert (line.job_id, line.pending, line.cost_micro_usd) == ("job-1", True, 399_990)
        assert (world.holds()[0].pending, world.holds()[0].notes[-1]) == (False, "job=job-1")
    else:
        assert code is ExitCode.LOST_RUN
        assert world.holds() == [hold] and world.job_lines() == []
        assert "it stays on the ledger" in capsys.readouterr().out
    assert world.ledger.total_micro_usd() == 399_990
    assert [job.id for job in FakeJobsClient.list_jobs(world.client)] == ["job-1"]


KINDS_AND_MODES = [(kind, mode) for kind in sorted(KINDS) for mode in ("smoke", "full")]


@pytest.mark.parametrize(("kind", "mode"), KINDS_AND_MODES)
def test_an_adopted_line_is_the_pending_line_the_launcher_would_have_written(world, kind, mode):
    later = world.clock.now() + timedelta(minutes=20)
    for overrides in ({}, {"override_smoke": True, "override_soft": True}):
        plan = build_plan(world.settings, kind, mode, {"l4x1": 14_000}, SHA, repo_root=world.root, **overrides)
        for source in ("hardware", "settings"):
            hold = hold_line(plan, world.clock.now(), source)
            assert (hold.job_id, hold.pending, hold.notes[-1]) == (f"{HOLD}{hold.run_id}", True, "hold")
            assert hold.model_dump(exclude={"job_id", "notes"}) == pending_line(
                plan, "job-7", world.clock.now(), source
            ).model_dump(exclude={"job_id", "notes"})
            assert adopted_line(hold, "job-7", later) == pending_line(plan, "job-7", later, source)


def test_a_job_the_platform_does_not_know_is_closed_at_its_full_cost(world, capsys):
    world.git_ok()
    gone = pending_line(world.plan(), "gone-1", world.clock.now(), "settings")
    world.ledger.append(gone)

    assert wait_job(world.settings, world.deps, "gone-1") is ExitCode.LOST_RUN

    line = world.ledger.read()["gone-1"]
    assert (line.pending, line.stage, line.cost_micro_usd, line.minutes) == (False, "GONE", 1_199_970, 90)
    assert (line.lost, line.oom, line.notes) == (
        False,
        False,
        ["price=settings", "job-not-found", "running-time-unknown"],
    )
    out = capsys.readouterr().out
    assert "does not know job gone-1" in out and "hf jobs ps -a" in out and "driftmap wait gone-1" in out
    assert world.count("fetch_job_logs") == 0
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.SUCCESS
    closed = world.ledger.path.read_bytes()
    assert wait_job(world.settings, world.deps, "gone-1") is ExitCode.LOST_RUN
    assert world.ledger.path.read_bytes() == closed and "is unchanged" in capsys.readouterr().out


def test_a_job_the_ledger_still_holds_counts_as_in_flight(world, capsys):
    world.git_ok(3)
    world.started("job-9", "quant")
    assert launch(world.settings, world.deps, "generate", "smoke", dry_run=True) is ExitCode.SUCCESS
    held = world.started("job-8", "steer")
    assert [job.status.stage for job in world.client.list_jobs()] == ["COMPLETED", "COMPLETED"]
    capsys.readouterr()

    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.GUARD_REFUSAL

    out = capsys.readouterr().out
    refused = {row.split()[0]: row for row in out.splitlines() if " refused " in row}
    assert list(refused) == ["in_flight_gpu"] and "job-8" in refused["in_flight_gpu"]
    assert "pending on the ledger: job-8; settle each with `driftmap wait <id>`" in out
    assert "job-9" not in refused["in_flight_gpu"] and len(world.submits()) == 2
    world.ledger.append(terminal_line(held, running_secs=60, stage="COMPLETED", ts=world.clock.now()))
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.SUCCESS


def test_ledger_lines_of_a_kind_no_longer_launched_are_read_and_a_pending_one_holds_the_gpu(world, capsys):
    world.git_ok(3)
    steer = pending_line(world.plan("steer"), "job-6", world.clock.now(), "settings")
    stored = steer.model_copy(update={"kind": "judge"})
    world.ledger.append(stored)
    world.ledger.append(terminal_line(stored, running_secs=600, stage="COMPLETED", ts=world.clock.now()))
    assert launch(world.settings, world.deps, "generate", "smoke", dry_run=True) is ExitCode.SUCCESS
    world.ledger.append(stored.model_copy(update={"job_id": "job-7"}))
    capsys.readouterr()

    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.GUARD_REFUSAL

    out = capsys.readouterr().out
    assert "pending on the ledger: job-7" in out and "job-6" not in out and world.submits() == []
    with pytest.raises(ConfigError, match="unknown job kind 'judge'"):
        launch(world.settings, world.deps, "judge", "smoke", dry_run=True)


def test_a_poll_error_keeps_the_pending_line_and_prints_the_wait_command(world, capsys):
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    world.started("job-1")
    world.push_record(RUN, "ok")
    before = world.ledger.path.read_bytes()
    working = world.client.wait_for_job

    for status in (401, 403, 410, 500, 502, 503):

        def broken(*args, status=status, **kwargs):
            raise http_error(status)

        world.client.wait_for_job = broken
        assert wait_job(world.settings, world.deps, "job-1") is ExitCode.LOST_RUN
        out = capsys.readouterr().out
        assert "driftmap wait job-1" in out and f" {status})" in out and "is unchanged" in out
        assert world.ledger.path.read_bytes() == before

    world.client.wait_for_job = working
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.SUCCESS
    assert world.ledger.read()["job-1"].pending is False


def no_logs(world):
    def broken(**kwargs):
        raise http_error(404)

    world.client.fetch_job_logs = broken


def no_record(world):
    def unreachable(*args, **kwargs):
        raise LocalEntryNotFoundError("the Hub cannot be reached and the file is not on disk")

    world.fake_hub.hf_hub_download = unreachable


@pytest.mark.parametrize("outage", [no_logs, no_record])
@pytest.mark.parametrize(
    ("running_secs", "minutes", "cost", "unknown"),
    [(1500, 25, 333_325, []), (None, 90, 1_199_970, ["running-time-unknown"])],
)
def test_a_finished_job_whose_evidence_cannot_be_fetched_is_closed_at_its_own_duration(
    world, capsys, outage, running_secs, minutes, cost, unknown
):
    world.git_ok(2)
    world.job("job-1", "ERROR", running_secs=running_secs, logs=["step 3", trailer("partial", 10)])
    world.started("job-1")
    world.push_record(RUN, "partial")
    world.client.billing_used_micro_usd = 777_000
    working = (world.client.fetch_job_logs, world.fake_hub.hf_hub_download)
    outage(world)

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.LOST_RUN

    line = world.ledger.read()["job-1"]
    assert (line.pending, line.stage, line.minutes, line.cost_micro_usd) == (False, "ERROR", minutes, cost)
    assert (line.lost, line.oom, line.record_status, line.exit_code) == (False, False, None, None)
    assert line.notes == ["price=settings", "unclassified", "no-evidence", *unknown]
    assert line.billing_used_after_micro_usd == 777_000
    out = capsys.readouterr().out
    assert "driftmap wait job-1" in out and "hf jobs cancel" not in out
    assert launch(world.settings, world.deps, "steer", "smoke", dry_run=True) is ExitCode.SUCCESS
    capsys.readouterr()
    for dry_run in (True, False):
        assert launch(world.settings, world.deps, "generate", "smoke", dry_run=dry_run) is ExitCode.GUARD_REFUSAL
        out = capsys.readouterr().out
        assert "job-1" in out and "driftmap wait job-1" in out and "nothing was submitted" in out
    assert len(world.submits()) == 1 and world.count("billing_usage") == 2

    world.client.fetch_job_logs, world.fake_hub.hf_hub_download = working
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.UNEXPECTED
    line = world.ledger.read()["job-1"]
    assert (line.record_status, line.exit_code, line.cost_micro_usd) == ("partial", 10, cost)
    assert line.notes == ["price=settings", "partial", *unknown]
    assert launch(world.settings, world.deps, "generate", "smoke", dry_run=True) is ExitCode.SUCCESS


@pytest.mark.parametrize(
    ("stage", "running_secs", "lost"),
    [
        ("ERROR", 5130, True),
        ("ERROR", 9000, True),
        ("ERROR", 5129, False),
        ("ERROR", None, False),
        ("COMPLETED", 5130, False),
    ],
)
def test_a_close_without_evidence_keeps_the_lost_signal(world, stage, running_secs, lost):
    world.job("job-1", stage, running_secs=running_secs, logs=["step 900"])
    world.started("job-1")
    no_logs(world)
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.LOST_RUN
    line = world.ledger.read()["job-1"]
    assert (line.lost, line.oom, line.notes[1:3]) == (lost, False, ["unclassified", "no-evidence"])
    assert world.ledger.lost_runs() == ([("generate", T)] if lost else [])


def test_a_lost_job_whose_log_could_not_be_fetched_still_stops_its_relaunch(world, capsys):
    world.git_ok(2)
    world.smoke_passed()
    world.job("job-1", "ERROR", running_secs=5130, logs=["step 900"])
    logs = world.client.fetch_job_logs
    no_logs(world)

    assert launch(world.settings, world.deps, "generate", "full", wait=True) is ExitCode.LOST_RUN

    line = world.ledger.read()["job-1"]
    assert (line.lost, line.cost_micro_usd, line.notes[1:]) == (True, 1_146_638, ["unclassified", "no-evidence"])
    capsys.readouterr()
    world.clock.advance(120)
    assert launch(world.settings, world.deps, "generate", "full") is ExitCode.GUARD_REFUSAL
    assert "driftmap wait job-1" in capsys.readouterr().out and len(world.submits()) == 1

    world.client.fetch_job_logs = logs
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.LOST_RUN
    assert world.ledger.read()["job-1"].notes == ["price=hardware"]
    capsys.readouterr()
    assert launch(world.settings, world.deps, "generate", "full") is ExitCode.GUARD_REFUSAL
    refused = [row.split()[0] for row in capsys.readouterr().out.splitlines() if " refused " in row]
    assert refused == ["timeout_lost"] and len(world.submits()) == 1


def test_a_launch_decides_on_one_read_of_the_ledger(world, capsys):
    world.git_ok()
    held = world.started("job-8", "steer")
    reads = []

    class Watched:
        def __getattr__(self, name):
            attribute = getattr(world.ledger, name)
            if name == "append" or not callable(attribute):
                return attribute

            def call(*args, **kwargs):
                result = attribute(*args, **kwargs)
                reads.append(name)
                if len(reads) == 1:
                    # a wait in another terminal settles the job right after this read
                    settled = terminal_line(held, running_secs=60, stage="COMPLETED", ts=world.clock.now())
                    world.ledger.append(settled)
                return result

            return call

    assert launch(world.settings, world.deps_with(ledger=Watched()), "generate", "smoke") is ExitCode.GUARD_REFUSAL

    refused = {row.split()[0]: row for row in capsys.readouterr().out.splitlines() if " refused " in row}
    assert list(refused) == ["in_flight_gpu"] and "job-8" in refused["in_flight_gpu"]
    assert reads == ["lines"] and len(world.submits()) == 1


def test_a_cut_token_before_the_trailer_does_not_hide_the_exit_code(world, capsys):
    logs = [f"retry with {TOKEN[:29]}", trailer("failed", 12)]
    for job_id, pushed in (("job-1", True), ("job-2", False)):
        world.job(job_id, "ERROR", running_secs=90, logs=logs)
        started = world.started(job_id)
        if pushed:
            world.push_record(started.run_id, "failed")
        assert wait_job(world.settings, world.deps, job_id) is ExitCode.UNEXPECTED
        saved = (world.root / LOGS_DIR / f"{job_id}.log").read_text(encoding="utf-8")
        assert saved.splitlines() == ["retry with hf_***", "***-job: finished status=failed exit=12"]
        world.clock.advance(60)
    recorded, bare = (world.ledger.read()[job_id] for job_id in ("job-1", "job-2"))
    assert (recorded.record_status, recorded.exit_code, recorded.notes[1:]) == ("failed", 12, ["integrity"])
    assert (bare.record_status, bare.exit_code, bare.notes[1:]) == (None, 12, ["unclassified"])
    assert not recorded.oom and not bare.oom and world.ledger.oom_runs() == []
    out = capsys.readouterr().out
    assert out.count("retry with hf_***") == 1 and TOKEN[3:11] not in out
    shown = classify("COMPLETED", None, None, f"saw {TOKEN}", marker_present=False, running_secs=5, timeout_s=T)
    assert shown.next_action.endswith("saw hf_***")


def test_only_the_poll_running_out_prints_the_cancel_command(world, capsys):
    world.job("job-1", "COMPLETED", logs=[trailer("ok", 0)])
    world.started("job-1")
    world.push_record(RUN, "ok")

    def stalled(**kwargs):
        raise TimeoutError("log stream stalled")

    world.client.fetch_job_logs = stalled
    with pytest.raises(TimeoutError):
        wait_job(world.settings, world.deps, "job-1")
    assert "hf jobs cancel" not in capsys.readouterr().out and world.ledger.read()["job-1"].pending


def test_a_token_cut_by_a_colour_code_or_a_line_break_is_not_saved(world):
    cut_lines = [
        f"retry with {TOKEN[:14]}",
        f"{TOKEN[14:]} done",
        f"Authorization: Bearer {TOKEN[:9]}\x1b[0m{TOKEN[9:21]}\x1b[1m{TOKEN[21:]}",
        f"last part of {TOKEN[:3]}",
        f"{TOKEN[3:]}\n",
        trailer("ok", 0),
    ]
    world.job("job-1", "COMPLETED", logs=cut_lines)
    world.started("job-1")
    world.push_record(RUN, "ok")

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.SUCCESS

    saved = (world.root / LOGS_DIR / "job-1.log").read_text(encoding="utf-8")
    assert not any(TOKEN[at : at + 6] in saved for at in range(3, len(TOKEN) - 5))
    assert saved.splitlines() == [
        "retry with hf_***",
        "*** done",
        "Authorization: Bearer hf_***",
        "last part of hf_***",
        "***",
        trailer("ok", 0),
    ]
    assert world.ledger.read()["job-1"].exit_code == 0


def test_waiting_on_a_job_the_ledger_does_not_know_exits_5(world):
    with pytest.raises(DriftmapError) as unknown:
        wait_job(world.settings, world.deps, "job-9")
    assert unknown.value.exit_code is ExitCode.LOST_RUN
    assert world.client.calls == [("list_jobs", {})] and not world.ledger.path.exists()
    assert not lock_held(world.settings.lock_path)


def test_waiting_on_an_unknown_job_with_an_unreadable_job_list_says_to_try_again(world):
    def unreadable(**kwargs):
        raise http_error(503)

    world.client.list_jobs = unreadable
    with pytest.raises(DriftmapError) as failed:
        wait_job(world.settings, world.deps, "job-9")
    message = str(failed.value)
    assert failed.value.exit_code is ExitCode.LOST_RUN and "not launched from this checkout" not in message
    assert "could not be read" in message and "try again with: driftmap wait job-9" in message


def test_waiting_on_a_hold_the_ledger_does_not_know_exits_5(world):
    with pytest.raises(DriftmapError) as unknown:
        wait_job(world.settings, world.deps, f"{HOLD}{SMOKE_RUN}")
    assert unknown.value.exit_code is ExitCode.LOST_RUN and world.client.calls == []
    assert not lock_held(world.settings.lock_path)


@pytest.mark.parametrize("released", [True, False])
def test_waiting_on_a_listed_job_whose_label_names_a_hold_records_it(tmp_path, capsys, released):
    world = flaky(tmp_path, DOWN, created=True, trailing=4)
    with pytest.raises(DriftmapError):
        smoke_launch(world)
    (hold,) = world.holds()
    if released:
        world.clock.advance(900)
        assert resolve(world, hold) is ExitCode.SUCCESS
        assert world.ledger.total_micro_usd() == 0 and world.holds()[0].notes[-1] == "no-job-found"
    world.client.trailing = 0
    before = world.ledger.path.read_bytes()
    with pytest.raises(DriftmapError):
        wait_job(world.settings, world.deps, "job-2")
    assert world.ledger.path.read_bytes() == before
    capsys.readouterr()

    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.LOST_RUN

    out = capsys.readouterr().out
    assert f"job job-1 carries the label of run {SMOKE_RUN}" in out and "hf jobs cancel job-1" in out
    (line,) = world.job_lines()
    assert line == adopted_line(hold, "job-1", world.clock.now())
    assert (line.pending, line.cost_micro_usd, line.notes) == (True, 399_990, ["price=hardware"])
    (closed,) = world.holds()
    assert (closed.pending, closed.cost_micro_usd) == (False, 0)
    assert closed.notes[-1] == ("no-job-found" if released else "job=job-1")
    assert world.ledger.total_micro_usd() == 399_990 and len(world.ledger.lines()) == 3
    assert not lock_held(world.settings.lock_path)


def test_a_listed_job_is_recorded_only_through_the_hold_its_own_label_names(world):
    hold = hold_line(world.plan("generate", "smoke"), world.clock.now(), "settings")
    world.ledger.append(hold)
    for job_id, labels in (("job-7", {launcher.RUN_LABEL: RUN}), ("job-8", None), ("job-6", {"other": SMOKE_RUN})):
        world.client.script_job(job_id, [{"stage": "RUNNING"}])
        world.client._accept(dockerImage="i", command=["true"], flavor="l4x1", labels=labels)
    before = world.ledger.path.read_bytes()

    for job_id in ("job-7", "job-8", "job-6", "job-9"):
        with pytest.raises(DriftmapError) as unknown:
            wait_job(world.settings, world.deps, job_id)
        assert unknown.value.exit_code is ExitCode.LOST_RUN
    assert world.ledger.path.read_bytes() == before and world.count("wait_for_job") == 0

    world.clock.advance(900)
    assert wait_job(world.settings, world.deps, hold.job_id, sleep=world.clock.sleep) is ExitCode.SUCCESS
    (closed,) = world.holds()
    assert closed.notes[-1] == "no-job-found" and world.job_lines() == [] and world.clock.sleeps == [60]


def test_a_hold_is_resolved_under_the_launch_lock(tmp_path):
    world = flaky(tmp_path, http_error(503), created=False)
    with pytest.raises(DriftmapError):
        smoke_launch(world)
    (hold,) = world.holds()
    world.clock.advance(900)
    reads, held = world.count("list_jobs"), []

    def sleep(seconds):
        held.append(lock_held(world.settings.lock_path))
        world.clock.sleep(seconds)

    with open(world.settings.lock_path, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        for job_id in (hold.job_id, "job-1"):
            with pytest.raises(LockHeld) as refused:
                wait_job(world.settings, world.deps, job_id, sleep=sleep)
            assert refused.value.exit_code is ExitCode.GUARD_REFUSAL
    assert world.holds() == [hold] and world.count("list_jobs") == reads

    assert wait_job(world.settings, world.deps, hold.job_id, sleep=sleep) is ExitCode.SUCCESS
    assert held == [True] and not lock_held(world.settings.lock_path)


def test_a_run_record_that_does_not_parse_is_left_to_the_author(world, capsys):
    world.job("job-1", "ERROR", running_secs=120, logs=["step 3"])
    world.started("job-1")
    world.push(f"runs/{RUN}.json", b'{"schema": "driftmap.run/0"}')
    assert wait_job(world.settings, world.deps, "job-1") is ExitCode.UNEXPECTED
    line = world.ledger.read()["job-1"]
    assert (line.record_status, line.oom, line.lost, line.notes[1:]) == ("unreadable", False, False, ["unclassified"])
    assert (line.minutes, line.cost_micro_usd) == (2, 26_666)


def test_the_gpu_flavor_readiness_found_replaces_the_configured_one(world):
    world.git_ok()
    world.ready(gpu_flavor="a10g-small")
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.SUCCESS
    ((_, kwargs),) = world.submits()
    (line,) = world.job_lines()
    assert kwargs["flavor"] == line.flavor == "a10g-small"
    assert (line.price_micro_usd_per_min, line.cost_micro_usd, line.notes) == (16_667, 500_010, ["price=hardware"])
    assert kwargs["secrets"] == {"HF_TOKEN": "hf_dummy"} and kwargs["attempts"] == 1
    assert kwargs["labels"] == {launcher.RUN_LABEL: line.run_id}
    assert submit_kwargs(world.settings, world.plan(), "t")["labels"].keys() == {launcher.RUN_LABEL}


def test_a_flavor_readiness_found_but_the_config_does_not_price_is_refused(world):
    world.ready(gpu_flavor="t4-small")
    with pytest.raises(ConfigError, match="t4-small"):
        launch(world.settings, world.deps, "generate", "smoke")
    assert world.client.calls == [] and world.git.calls == [] and not world.ledger.path.exists()


def test_every_submit_carries_the_config_hash(world):
    world.git_ok(2)
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.SUCCESS
    world.clock.advance(3600)
    world.ledger.append(
        terminal_line(
            world.ledger.read()["fake00000000000000000001"], running_secs=60, stage="COMPLETED", ts=world.clock.now()
        )
    )
    assert launch(world.settings, world.deps, "steer", "smoke", args=["--layer", "12"]) is ExitCode.SUCCESS
    plain, extra = (kwargs["command"][-1] for _, kwargs in world.submits())
    carried = f"--config-hash {world.settings.config_hash()}"
    assert plain.endswith(carried) and plain.count("--config-hash") == 1
    assert extra.endswith(f"{carried} --layer 12") and extra.count("--config-hash") == 1


def test_a_live_price_above_the_configured_one_prices_the_job(tmp_path):
    hardware = [row | {"unitCostMicroUSD": 14_000} if row["name"] == "l4x1" else row for row in DEFAULT_HARDWARE]
    world = World(tmp_path, client=FakeJobsClient(hardware=hardware))
    world.git_ok()
    assert launch(world.settings, world.deps, "generate", "smoke") is ExitCode.SUCCESS
    (line,) = world.job_lines()
    assert (line.price_micro_usd_per_min, line.cost_micro_usd, line.notes) == (14_000, 420_000, ["price=hardware"])


def test_overrides_reach_the_guards_and_the_ledger_notes(world):
    world.git_ok(3)
    world.client.billing_used_micro_usd = 15_000_000

    def go(**overrides):
        return launch(world.settings, world.deps, "generate", "full", **overrides)

    assert go(override_smoke=True) is ExitCode.GUARD_REFUSAL
    assert go(override_soft=True) is ExitCode.GUARD_REFUSAL
    assert world.submits() == []
    assert go(override_smoke=True, override_soft=True) is ExitCode.SUCCESS
    (line,) = world.job_lines()
    assert line.notes == ["price=hardware", "override-smoke", "override-soft"]


def test_the_uv_runner_submits_a_script_job(tmp_path):
    world = World(tmp_path, runner="uv")
    world.git_ok()
    assert launch(world.settings, world.deps, "steer", "smoke", limit=10) is ExitCode.SUCCESS
    ((method, kwargs),) = world.submits()
    assert (method, kwargs["script"]) == ("run_uv_job", "jobs/steer.py")
    flags = ["--limit", "10", "--force-flush-after-s", "300", "--config-hash", world.settings.config_hash()]
    assert kwargs["script_args"][-6:] == flags


def test_the_launcher_never_spells_the_submit_call():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "run_job(" not in source and "run_uv_job(" not in source
