import json
import logging
import subprocess

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.utils import hf_raise_for_status
from pydantic import ValidationError

from driftmap.core.hub import Hub
from driftmap.core.records import (
    Decoding,
    JudgeConfig,
    Seeds,
    SmokeRecord,
    Snapshot,
    Tables,
    dump,
)
from driftmap.core.schemas import SCHEMA_TABLES, DriftmapError, schema_versions
from driftmap.core.settings import load_settings
from driftmap.jobs import snapshot
from driftmap.jobs.snapshot import (
    DECISION_PATH,
    IMAGE_PROBE_PATH,
    SMOKE_DIR,
    TABLES_PATH,
    read_decision,
    read_smoke_records,
    take_snapshot,
    write_decision,
)
from tests.fakes import FakeClock, FakeHub, FakeJobsClient, FakeSubprocess
from tests.fakes.settings import settings_toml

AYA = "CohereLabs/tiny-aya-global"
BASE = "CohereLabs/tiny-aya-base"
SHA = "a" * 40
ORIGIN = "https://github.com/vincentni/tiny-aya-safety-drift-map"
SMOKE = SmokeRecord(
    script="driftmap.scripts.drift",
    image="pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime",
    runner="pip",
    dependencies=[],
    schema_versions=schema_versions(),
    status="ok",
    bos_check=None,
    install_minutes=2.5,
    upload_mb_s=None,
    peak_vram_gb=None,
    peak_rss_gb=None,
    tokens_per_s=None,
)


class StubLedger:
    def total_micro_usd(self):
        return 1_200_000

    def smoke_spent_micro_usd(self):
        return 399_990

    def smoke_pending(self):
        return True

    def lost_runs(self):
        return [("drift", 10_800)]

    def oom_runs(self):
        return [("judge", "c" * 64)]


@pytest.fixture
def settings(tmp_path):
    return load_settings(settings_toml(tmp_path))


@pytest.fixture
def fake_hub(tmp_path):
    fake = FakeHub(tmp_path / "hub")
    fake.add_repo(AYA, gated=True).add_repo(BASE, gated=True)
    fake.authorize(AYA)
    return fake


def pinned_git():
    """Scripted git: HEAD, a clean status, a fetch and an ancestor check that all succeed."""
    fake = FakeSubprocess()
    fake.script(stdout=SHA + "\n")
    for _ in range(3):
        fake.script()
    return fake


@pytest.fixture
def git():
    return pinned_git()


def take(settings, client, fake_hub, root, git, **kwargs):
    hub = Hub(fake_hub, gated_repos=settings.gated_repos, sleep=FakeClock().sleep)
    return take_snapshot(settings, client, hub, StubLedger(), root, run=git.run, **kwargs)


def start(client, job_id, flavor, stage):
    client.script_job(job_id, [{"stage": stage}])
    client.run_job(image="python:3.12", command=["true"], flavor=flavor)


def test_a_snapshot_gathers_spend_jobs_prices_gates_and_local_records(tmp_path, settings, fake_hub, git):
    client = FakeJobsClient()
    client.billing_used_micro_usd = 2_500_000
    for job_id, flavor, stage in (
        ("gpu-running", "l4x1", "RUNNING"),
        ("gpu-queued", "a10g-small", "SCHEDULING"),
        ("gpu-done", "l4x1", "COMPLETED"),
        ("cpu-running", "cpu-upgrade", "RUNNING"),
        ("unlisted-running", "h200x8", "RUNNING"),
        ("gpu-new-stage", "l4x1", "PAUSED"),
        ("gpu-canceled", "l4x1", "CANCELED"),
    ):
        start(client, job_id, flavor, stage)
    client.calls.clear()
    dump(SMOKE, tmp_path / SMOKE_DIR / "drift.json")
    (tmp_path / IMAGE_PROBE_PATH).write_text("{}")

    taken = take(settings, client, fake_hub, tmp_path, git)

    assert taken == Snapshot(
        billing_used_micro_usd=2_500_000,
        ledger_total_micro_usd=1_200_000,
        smoke_spent_micro_usd=399_990,
        smoke_pending=True,
        prices_micro_usd_per_min={"cpu-basic": 167, "cpu-upgrade": 500, "l4x1": 13_333, "a10g-small": 16_667},
        ceilings=settings.ceilings,
        # counted: an unlisted flavor, and any stage that is not terminal
        in_flight_gpu=["gpu-running", "gpu-queued", "unlisted-running", "gpu-new-stage"],
        smoke_records=[SMOKE],
        gates={AYA: "ok", BASE: "403"},
        lost_runs=[("drift", 10_800)],
        oom_runs=[("judge", "c" * 64)],
        drift_flag=None,
        drift_decision=None,
        code_on_origin=True,
    )
    assert [name for name, _ in client.calls] == ["billing_usage", "list_jobs_hardware", "list_jobs"]
    # unfiltered: a stage filter would hide stages the library lacks
    assert client.calls[-1][1] == {}
    assert [name for name, _ in fake_hub.calls] == ["auth_check", "auth_check"]
    assert [call.argv[3:] for call in git.calls] == [
        ["rev-parse", "HEAD"],
        ["status", "--porcelain"],
        ["fetch", "--quiet", settings.code.origin, "main"],
        ["merge-base", "--is-ancestor", SHA, "FETCH_HEAD"],
    ]
    assert all(call.kwargs["timeout"] and call.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" for call in git.calls)


@pytest.mark.parametrize("status", [401, 403, 503])
def test_unreadable_billing_leaves_the_ledger_as_the_only_total(tmp_path, settings, fake_hub, git, caplog, status):
    client = FakeJobsClient()
    client.billing_status = status
    with caplog.at_level(logging.WARNING):
        taken = take(settings, client, fake_hub, tmp_path, git)
    assert taken.billing_used_micro_usd is None and taken.ledger_total_micro_usd == 1_200_000
    assert "billing" in caplog.text


def test_a_gated_repo_that_is_missing_is_an_error_not_a_gate(tmp_path, settings, fake_hub, git):
    del fake_hub.repos[BASE]
    with pytest.raises(RepositoryNotFoundError):
        take(settings, FakeJobsClient(), fake_hub, tmp_path, git)


@pytest.mark.parametrize(
    ("results", "pinned"),
    [
        ([(0, SHA), (0, ""), (0, ""), (0, "")], True),
        ([(128, "")], False),
        ([(0, SHA), (0, " M a\n")], False),
        ([(0, SHA), (0, ""), (1, "")], False),
        ([(0, SHA), (0, ""), (0, ""), (1, "")], False),
        ([(0, "b" * 40)], False),
    ],
    ids=["pinned", "no-checkout", "dirty", "fetch-fails", "not-on-origin", "another-head"],
)
def test_code_is_on_origin_only_when_every_git_check_passes(tmp_path, settings, fake_hub, results, pinned):
    git = FakeSubprocess()
    for returncode, stdout in results:
        git.script(returncode=returncode, stdout=stdout)
    taken = take(settings, FakeJobsClient(), fake_hub, tmp_path, git, code_sha=SHA)
    assert taken.code_on_origin is pinned and len(git.calls) == len(results)


def test_no_snapshot_without_the_hardware_or_job_listing(tmp_path, settings, fake_hub, git):
    for method in ("list_jobs_hardware", "list_jobs"):
        client = FakeJobsClient()

        def down(*args, **kwargs):
            request = httpx2.Request("GET", "https://huggingface.co/api/jobs/hardware")
            hf_raise_for_status(httpx2.Response(503, request=request))

        setattr(client, method, down)
        with pytest.raises(HfHubHTTPError):
            take(settings, client, fake_hub, tmp_path, git)


def test_a_git_that_hangs_or_is_missing_is_not_a_pass(tmp_path):
    def hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def missing(argv, **kwargs):
        raise FileNotFoundError("git")

    assert "git" in snapshot.code_pin_fault(tmp_path, ORIGIN, "main", run=hangs)
    assert "git" in snapshot.code_pin_fault(tmp_path, ORIGIN, "main", run=missing)


def test_the_drift_flag_and_decision_come_from_the_local_results(tmp_path, settings, fake_hub, git):
    decoding = Decoding(runtime="transformers")
    judge = JudgeConfig(
        repo=settings.judge.repo,
        revision=settings.judge.revision,
        dtype="bfloat16",
        decoding=decoding,
        prompt_format="polyguard-v1",
        parser_version="1",
        config_hash="h",
    )
    tables = Tables(
        schema=SCHEMA_TABLES,
        columns={},
        safety=[],
        over_refusal=[],
        chrf=[],
        deltas=[],
        flags={"drift_below_threshold": True},
        judge=judge,
        seeds=Seeds(seed=1, bootstrap=2),
    )
    dump(tables, tmp_path / TABLES_PATH)
    flagged = take(settings, FakeJobsClient(), fake_hub, tmp_path, git)
    assert flagged.drift_flag is True and flagged.drift_decision is None

    clock = FakeClock()
    assert read_decision(tmp_path / DECISION_PATH) is None
    written = write_decision(tmp_path / DECISION_PATH, "proceed", "drift is small but real", now=clock.now)
    assert read_decision(tmp_path / DECISION_PATH) == written
    assert json.loads((tmp_path / DECISION_PATH).read_text()) == {
        "schema": "driftmap.decision/1",
        "decision": "proceed",
        "note": "drift is small but real",
        "ts": "2026-11-02T10:00:00Z",
    }
    decided = take(settings, FakeJobsClient(), fake_hub, tmp_path, pinned_git())
    assert (decided.drift_flag, decided.drift_decision) == (True, "proceed")
    with pytest.raises(ValidationError):
        write_decision(tmp_path / DECISION_PATH, "skip", "", now=clock.now)
    assert read_decision(tmp_path / DECISION_PATH) == written


def test_smoke_records_are_read_in_name_order_and_a_bad_one_is_named(tmp_path):
    assert read_smoke_records(tmp_path / "absent") == []
    directory = tmp_path / SMOKE_DIR
    judged = SMOKE.model_copy(update={"script": "driftmap.scripts.judge"})
    dump(judged, directory / "judge.json")
    dump(SMOKE, directory / "drift.json")
    (directory / IMAGE_PROBE_PATH.name).write_text("not a smoke record")
    assert read_smoke_records(directory) == [SMOKE, judged]

    (directory / "quant.json").write_text('{"script": "x"}')
    with pytest.raises(DriftmapError, match="quant.json") as bad:
        read_smoke_records(directory)
    assert str(tmp_path) not in str(bad.value)
