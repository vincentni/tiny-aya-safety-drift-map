import json
import subprocess
from datetime import timedelta
from types import SimpleNamespace

import pytest

from driftmap.core.hub import GateRefused, Hub
from driftmap.core.records import dump
from driftmap.core.schemas import ExitCode
from driftmap.core.settings import ConfigError, load_settings
from driftmap.jobs import readiness
from driftmap.jobs.readiness import (
    READY_PATH,
    Check,
    ImageProbe,
    Item,
    NotReady,
    ReadyReport,
    pin_revisions,
    ready,
    require_ready,
)
from driftmap.jobs.snapshot import IMAGE_PROBE_PATH, Deps
from tests.fakes import FakeClock, FakeHub, FakeJobsClient
from tests.fakes.settings import BASE as CONFIG
from tests.fakes.settings import settings_toml

AYA = "CohereLabs/tiny-aya-global"
BASE = "CohereLabs/tiny-aya-base"
RESULTS = "vincentnkq/safety-drift-map-results"
DATASETS = ("CohereLabs/aya_redteaming", "CohereLabs/aya_dataset", "Paul/XSTest")
TOKEN = "hf_" + "a" * 34
HOME_DIR = "/" + "home/someone"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def settings(tmp_path):
    return load_settings(settings_toml(tmp_path))


@pytest.fixture
def fake_hub(tmp_path, settings):
    fake = FakeHub(tmp_path / "hub")
    fake.add_repo(AYA, gated=True).add_repo(BASE, gated=True).add_repo(RESULTS, repo_type="dataset")
    fake.authorize(AYA)
    return fake


@pytest.fixture
def checkout(tmp_path, settings, monkeypatch):
    """A clean checkout whose HEAD is on a local repo that stands in for the code URL."""
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")

    def git(*args):
        argv = ["git", "-c", "user.name=t", "-c", "user.email=t", *args]
        subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True, timeout=60)

    work = tmp_path / "work-tree"
    git("init", "-q", "--bare", "-b", "main", "origin.git")
    git("init", "-q", "-b", "main", str(work))
    # a machine-wide excludes file may hide a .gitignore
    (work / ".git" / "info" / "exclude").write_text(".run-local/\n")
    (work / "a").write_text("a")
    git("-C", str(work), "add", "a")
    git("-C", str(work), "commit", "-q", "-m", "a")
    git("-C", str(work), "remote", "add", "origin", str(tmp_path / "origin.git"))
    git("-C", str(work), "push", "-q", "origin", "main")
    git("-C", str(work), "config", f"url.{tmp_path / 'origin.git'}.insteadOf", settings.code.origin)
    return work


@pytest.fixture
def deps(settings, fake_hub, clock, checkout):
    probe = ImageProbe(
        image=settings.images.pytorch, python="3.12.3", torch="2.14.1+cu126", pep668=True, tools=["pip"]
    )
    dump(probe, checkout / IMAGE_PROBE_PATH)
    return Deps(
        client=FakeJobsClient(),
        hub=Hub(fake_hub, gated_repos=settings.gated_repos, sleep=clock.sleep),
        ledger=SimpleNamespace(total_micro_usd=lambda: 0),
        repo_root=checkout,
        now=clock.now,
        head=lambda url: 200,
    )


def test_every_item_passes_on_a_prepared_machine(monkeypatch, settings, fake_hub, deps, clock):
    """Ten PASS rows, exit 0 and a report file that holds no token or account name (TS-RDY-1)."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    deps.client.billing_used_micro_usd = 1_250_000

    report = ready(settings, deps)

    assert [(item.number, item.status) for item in report.items] == [(n, "PASS") for n in range(1, 10)]
    assert report.ok and report.exit_code is ExitCode.SUCCESS
    assert all(item.hint is None and item.detail for item in report.items)
    text = (deps.repo_root / READY_PATH).read_text()
    stored = json.loads(text)
    assert stored["ts"] == "2026-11-02T10:00:00Z" and len(stored["items"]) == 9
    assert stored["config_hash"] == settings.config_hash()
    assert set(stored["items"][0]) == {"number", "name", "status", "detail", "hint"}
    assert stored["billing"]["usedMicroUsd"] == 1_250_000
    assert stored["gates"] == {AYA: "ok", BASE: "403"} and stored["gpu_flavor"] == "l4x1"
    for secret in (TOKEN, "hf_dummy", settings.owner, str(deps.repo_root)):
        assert secret not in text
    # read-only: nothing is submitted, created or uploaded
    assert {name for name, _ in deps.client.calls} == {"billing_usage", "list_jobs_hardware"}
    assert {name for name, _ in fake_hub.calls} == {"whoami", "auth_check"}

    assert require_ready(deps.repo_root / READY_PATH, settings.config_hash(), now=clock.now) == report
    clock.advance(timedelta(hours=24).total_seconds())
    with pytest.raises(NotReady):
        require_ready(deps.repo_root / READY_PATH, settings.config_hash(), now=clock.now)


def test_an_item_that_crashes_is_a_fail_row_with_secrets_masked(monkeypatch, settings, deps):
    def crashes(settings, deps):
        raise RuntimeError(f"boom with {TOKEN}\nsecond line")

    def leaks(settings, deps):
        return Check(False, f"saw {TOKEN} under {HOME_DIR}/x", "look closer")

    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", crashes), (2, "second", leaks)))
    report = ready(settings, deps)
    assert [item.status for item in report.items] == ["FAIL", "FAIL"] and report.exit_code == 3
    assert report.items[0].detail.startswith("RuntimeError: boom with ") and report.items[1].hint == "look closer"
    text = (deps.repo_root / READY_PATH).read_text()
    assert TOKEN not in text and HOME_DIR not in text and "second line" not in text


def test_a_failed_or_partial_report_never_admits_a_verb(tmp_path, clock):
    path = tmp_path / "ready.json"
    passed = [Item(number=number, name=name, status="PASS", detail="") for number, name, _ in readiness.ITEMS]
    failed = [*passed[:-1], passed[-1].model_copy(update={"status": "FAIL", "hint": "push"})]
    for items, word in ((failed, "9"), (passed[:-1], "9"), ([], "1")):
        report = ReadyReport(ts=clock.now(), config_hash="c", items=items)
        assert not report.ok and report.exit_code == 3
        dump(report, path)
        with pytest.raises(NotReady, match=word):
            require_ready(path, "c", now=clock.now)
    path.write_text("{")
    with pytest.raises(NotReady, match="driftmap ready") as torn:
        require_ready(path, "c", now=clock.now)
    assert torn.value.exit_code is ExitCode.READINESS


def test_a_run_starts_by_dropping_the_previous_report(monkeypatch, tmp_path, settings, deps, clock):
    path = deps.repo_root / READY_PATH
    during = []

    def looks(settings, deps):
        during.append(path.exists())
        return Check(True, "looked")

    def stops(settings, deps):
        raise KeyboardInterrupt

    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", looks),))
    assert ready(settings, deps).ok and path.exists()
    assert ready(settings, deps).ok and during == [False, False]
    # an interrupted run leaves no earlier pass behind
    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", stops),))
    with pytest.raises(KeyboardInterrupt):
        ready(settings, deps)
    assert not path.exists()

    # a report made under another configuration does not count
    monkeypatch.setattr(readiness, "ITEMS", ((1, "first", looks),))
    ready(settings, deps)
    assert require_ready(path, settings.config_hash(), now=clock.now).ok
    reseeded = load_settings(settings_toml(tmp_path, seed=7))
    with pytest.raises(NotReady, match="configuration"):
        require_ready(path, reseeded.config_hash(), now=clock.now)


def test_pin_writes_the_current_revisions_beside_the_config(tmp_path, settings, fake_hub, clock):
    for repo in DATASETS:
        fake_hub.add_repo(repo, repo_type="dataset")
    hub = Hub(fake_hub, gated_repos=settings.gated_repos, sleep=clock.sleep)
    path = tmp_path / "revisions.json"
    kept = {"vincentnkq/aya-redteaming-6lang-eval": "9" * 40, DATASETS[0]: CONFIG["revisions"][DATASETS[0]]}
    path.write_text(json.dumps(kept))

    pins = pin_revisions(settings, hub, path)

    # the closed base stays unpinned; the pin on file does not move
    heads = {repo: fake_hub.heads[repo] for repo in (AYA, *DATASETS[1:])}
    assert pins == json.loads(path.read_text()) == {**kept, **heads}
    assert path.read_text().endswith("}\n") and list(tmp_path.glob(".revisions*")) == []
    assert load_settings(tmp_path / "driftmap.toml").revisions == pins
    assert [kwargs["repo_id"] for _, kwargs in fake_hub.calls] == [AYA, BASE, *DATASETS[1:]]
    assert {name for name, _ in fake_hub.calls} == {"repo_info"}
    assert list(json.loads(path.read_text())) == sorted(pins)
    fake_hub.calls.clear()
    written = path.stat().st_ino
    assert pin_revisions(settings, hub, path) == pins and [kw["repo_id"] for _, kw in fake_hub.calls] == [BASE]
    # nothing new to pin: the tracked file is not rewritten
    assert path.stat().st_ino == written

    fake_hub.authorized.discard(AYA)
    fresh = tmp_path / "fresh.json"
    with pytest.raises(GateRefused):
        pin_revisions(settings, hub, fresh)
    assert not fresh.exists()

    fake_hub.authorize(AYA)
    for bad in ("main", None, "A" * 40):
        fake_hub.heads[AYA] = bad
        with pytest.raises(ConfigError, match=f"fresh.json: {AYA}"):
            pin_revisions(settings, hub, fresh)
        assert not fresh.exists()
    fresh.write_text(json.dumps({AYA: "not a sha"}))
    with pytest.raises(ConfigError, match=f"fresh.json: {AYA}"):
        pin_revisions(settings, hub, fresh)
    assert json.loads(fresh.read_text()) == {AYA: "not a sha"}
