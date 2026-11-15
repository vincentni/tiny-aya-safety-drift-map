import json
import subprocess
import time
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import httpx2
import huggingface_hub.constants
import pytest
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils._http import default_client_factory

from driftmap.core import clients, hub
from driftmap.core.clients import make_jobs_client
from driftmap.core.hub import Hub
from driftmap.core.records import dump
from driftmap.core.schemas import ExitCode
from driftmap.core.settings import load_settings
from driftmap.jobs import client as jobs_client
from driftmap.jobs import readiness
from driftmap.jobs.client import BillingUnavailable, HfJobsClient, head_status, minute_prices, used_micro_usd
from driftmap.jobs.readiness import ImageProbe, Item, NotReady, ReadyReport, ready, require_ready
from driftmap.jobs.snapshot import IMAGE_PROBE_PATH, Deps, code_pin_fault
from tests.fakes import FakeClock, FakeHub, FakeJobsClient
from tests.fakes.jobs_client import DEFAULT_HARDWARE
from tests.fakes.settings import settings_toml

AYA = "CohereLabs/tiny-aya-global"
BASE = "CohereLabs/tiny-aya-base"
RESULTS = "vincentnkq/safety-drift-map-results"
OWNER = SimpleNamespace(owner="vincentnkq")
JOB = {
    "id": "j1",
    "owner": {"id": "u1", "name": "vincentnkq", "type": "user"},
    "createdAt": "2026-11-02T10:00:00.000Z",
    "flavor": "l4x1",
    "status": {"stage": "COMPLETED"},
}
USAGE = {"usage": {"totalMinutes": 3, "jobDetails": [], "usedMicroUsd": 39_999}, "hasAccess": True}
IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
PROBE = ImageProbe(image=IMAGE, python="3.12.3", torch="2.14.1+cu126", pep668=True, tools=["pip"])


class Spent:
    """Ledger stand-in: only the total the credit check reads."""

    total = 0

    def total_micro_usd(self):
        return self.total


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
def deps(tmp_path, settings, fake_hub, clock):
    return Deps(
        client=FakeJobsClient(),
        hub=Hub(fake_hub, gated_repos=settings.gated_repos, sleep=clock.sleep),
        ledger=Spent(),
        repo_root=tmp_path,
        now=clock.now,
        head=lambda url: 200,
    )


@pytest.fixture
def remote(monkeypatch):
    """The real client over a mock transport: (client, routes, requests seen)."""
    routes, seen = {}, []

    def handler(request):
        seen.append(request)
        reply = routes.get((request.method, request.url.path), (404, {"error": "unrouted"}))
        # a list is a sequence of replies; its last one repeats
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        status, body = reply
        if isinstance(body, str):
            return httpx2.Response(status, text=body, request=request)
        return httpx2.Response(status, json=body, request=request)

    def mocked():
        return httpx2.Client(transport=httpx2.MockTransport(handler), follow_redirects=True, timeout=None)

    monkeypatch.setattr(clients, "default_client_factory", mocked)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.delenv("DRIFTMAP_CLIENT")
    monkeypatch.setattr(clients, "_backend", clients._backend)
    hub.token.cache_clear()
    jobs_client.register()
    yield make_jobs_client(OWNER), routes, seen
    hub.token.cache_clear()
    set_client_factory(default_client_factory)


def only(monkeypatch, *numbers):
    monkeypatch.setattr(readiness, "ITEMS", tuple(item for item in readiness.ITEMS if item[0] in numbers))


def close(fake, repo, status, error_code=None):
    """Make auth_check on one repo answer with a status, like the Hub does."""
    original = fake.auth_check

    def auth_check(repo_id, **kwargs):
        if repo_id == repo:
            headers = {"X-Error-Code": error_code} if error_code else None
            fake._raise(status, fake._repo_url(repo_id, kwargs.get("repo_type")), headers)
        return original(repo_id, **kwargs)

    fake.auth_check = auth_check


def git(cwd, *args):
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t", *args], cwd=cwd, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def commit(repo, name):
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)
    return git(repo, "rev-parse", "HEAD")


def test_importing_the_module_registers_nothing():
    assert clients._backend is not HfJobsClient


def test_the_factory_builds_the_registered_client(remote):
    client, _, _ = remote
    assert type(client) is HfJobsClient and client.api.token == "hf_dummy"


def test_reads_stay_in_the_owner_namespace_and_carry_a_timeout(remote):
    client, routes, seen = remote
    routes["GET", "/api/jobs/vincentnkq"] = (200, [JOB])
    routes["GET", "/api/jobs/vincentnkq/j1"] = (200, JOB)
    routes["GET", "/api/jobs/hardware"] = (200, DEFAULT_HARDWARE)
    routes["GET", jobs_client.BILLING_PATH] = (200, USAGE)

    jobs = client.list_jobs(status=["SCHEDULING", "RUNNING"])
    assert type(jobs) is list and [job.id for job in jobs] == ["j1"]
    assert seen[-1].url.query == b"stage=SCHEDULING&stage=RUNNING"
    assert client.inspect_job(job_id="j1").status.stage == "COMPLETED"
    assert minute_prices(client.list_jobs_hardware())["l4x1"] == 13_333
    assert client.billing_usage() == USAGE
    assert client.wait_for_job("j1", timeout=60, poll_interval=60).id == "j1"

    # unnamed, the lookup would use whatever token the machine holds
    assert "/api/whoami-v2" not in [request.url.path for request in seen]
    assert {request.headers["authorization"] for request in seen} == {"Bearer hf_dummy"}
    assert all(request.extensions["timeout"] == clients.REQUEST_TIMEOUT.as_dict() for request in seen)


def test_a_submit_is_sent_once_and_a_read_three_times(remote, clock):
    real, routes, seen = remote
    client = HfJobsClient(OWNER, real.api, sleep=clock.sleep)
    routes["POST", "/api/jobs/vincentnkq"] = (503, {})
    routes["GET", "/api/jobs/vincentnkq/j1"] = (503, {})

    with pytest.raises(HfHubHTTPError):
        client.run_job(image="python:3.12", command=["true"], flavor="cpu-basic", timeout="60s")
    assert [request.method for request in seen] == ["POST"] and clock.sleeps == []

    with pytest.raises(HfHubHTTPError):
        client.inspect_job(job_id="j1")
    assert [request.method for request in seen] == ["POST", "GET", "GET", "GET"] and len(clock.sleeps) == 2


def test_a_failing_listing_is_sent_only_as_often_as_the_library_resends(remote, clock, monkeypatch):
    real, routes, seen = remote
    client = HfJobsClient(OWNER, real.api, sleep=clock.sleep)
    routes["GET", "/api/jobs/vincentnkq"] = (503, {})
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    with pytest.raises(HfHubHTTPError):
        client.list_jobs()
    # the library's own backoff: one request and five more
    assert len(seen) == 6 and len(waits) == 5 and clock.sleeps == []


def test_waiting_polls_once_a_minute_until_the_job_ends_or_the_deadline_passes(remote, clock):
    real, routes, seen = remote
    client = HfJobsClient(OWNER, real.api, sleep=clock.sleep, monotonic=clock.monotonic)
    running, done = {**JOB, "status": {"stage": "RUNNING"}}, {**JOB, "status": {"stage": "ERROR"}}

    routes["GET", "/api/jobs/vincentnkq/j1"] = [(200, running), (503, {}), (200, running), (200, done)]
    assert client.wait_for_job("j1", timeout=600).status.stage == "ERROR"
    backoff = clock.sleeps[1]
    assert clock.sleeps == [60, backoff, 60] and 0 < backoff < 60 and len(seen) == 4

    clock.sleeps.clear()
    routes["GET", "/api/jobs/vincentnkq/j1"] = (200, running)
    with pytest.raises(TimeoutError, match="j1.*RUNNING.*150"):
        client.wait_for_job("j1", timeout=150, poll_interval=60)
    assert clock.sleeps == [60, 60, 30] and len(seen) == 4 + 4


@pytest.mark.parametrize("body", [[], "<html>busy</html>", "null"])
def test_a_billing_body_that_is_not_an_object_is_unavailable(remote, clock, body):
    real, routes, seen = remote
    client = HfJobsClient(OWNER, real.api, sleep=clock.sleep)
    routes["GET", jobs_client.BILLING_PATH] = (200, body)
    with pytest.raises(BillingUnavailable):
        client.billing_usage()
    assert len(seen) == 1 and clock.sleeps == []


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_billing_read_is_final(remote, clock, status):
    real, routes, seen = remote
    client = HfJobsClient(OWNER, real.api, sleep=clock.sleep)
    routes["GET", jobs_client.BILLING_PATH] = (status, {"error": "no"})
    with pytest.raises(BillingUnavailable) as refused:
        client.billing_usage()
    assert refused.value.exit_code is ExitCode.READINESS and "Billing" in str(refused.value)
    assert len(seen) == 1 and clock.sleeps == []


@pytest.mark.filterwarnings("ignore:.*run_uv_job.* is experimental")
def test_no_call_falls_back_to_the_token_or_account_of_the_machine(monkeypatch):
    seen = []
    log_line = 'data: {"data": "hello", "timestamp": "2026-11-02T10:00:00.000Z"}\n\n'

    def handler(request):
        seen.append(request)
        path = request.url.path
        if path.endswith("/logs"):
            return httpx2.Response(200, text=log_line, request=request)
        listing = request.method == "GET" and path == "/api/jobs/vincentnkq"
        bodies = {"/api/jobs/hardware": DEFAULT_HARDWARE, jobs_client.BILLING_PATH: USAGE}
        return httpx2.Response(200, json=bodies.get(path, [JOB] if listing else JOB), request=request)

    def mocked():
        return httpx2.Client(transport=httpx2.MockTransport(handler), follow_redirects=True, timeout=None)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("HF_TOKEN", "hf_ambient")
    set_client_factory(mocked)
    try:
        client = HfJobsClient(OWNER, HfApi(token="hf_instance"))
        calls = {
            "run_job": lambda: client.run_job(image="python:3.12", command=["true"], flavor="cpu-basic"),
            "run_uv_job": lambda: client.run_uv_job("https://example.invalid/job.py", flavor="cpu-basic"),
            "inspect_job": lambda: client.inspect_job(job_id="j1"),
            "list_jobs": lambda: client.list_jobs(),
            "list_jobs_hardware": lambda: client.list_jobs_hardware(),
            "wait_for_job": lambda: client.wait_for_job("j1", timeout=60),
            "fetch_job_logs": lambda: client.fetch_job_logs(job_id="j1", tail=5),
            "billing_usage": lambda: client.billing_usage(),
        }
        public = {name for name, member in vars(HfJobsClient).items() if name[0] != "_" and callable(member)}
        assert set(calls) == public
        for name, call in calls.items():
            seen.clear()
            result = call()
            assert seen, name
            assert {request.headers["authorization"] for request in seen} == {"Bearer hf_instance"}, name
            paths = [request.url.path for request in seen]
            fixed = ("/api/jobs/hardware", jobs_client.BILLING_PATH)
            assert all(path in fixed or path.startswith("/api/jobs/vincentnkq") for path in paths), (name, paths)
            if name == "fetch_job_logs":
                assert result == ["hello"]
    finally:
        set_client_factory(default_client_factory)


def test_polling_faster_than_a_minute_is_refused(remote):
    client, _, seen = remote
    with pytest.raises(ValueError, match="60"):
        client.wait_for_job("j1", timeout=600, poll_interval=1)
    for timeout in (None, -1):
        with pytest.raises(ValueError, match="timeout"):
            client.wait_for_job("j1", timeout=timeout)
    assert seen == []


def test_usage_and_prices_are_whole_micro_dollars():
    assert used_micro_usd(USAGE) == 39_999
    for used in (None, 1.5, True, -1, "7"):
        with pytest.raises(BillingUnavailable):
            used_micro_usd({"usage": {"usedMicroUsd": used}})
    for body in ({}, [], {"usage": []}):
        with pytest.raises(BillingUnavailable):
            used_micro_usd(body)
    hourly = SimpleNamespace(name="h", unit_cost_micro_usd=800_000, unit_label="hour")
    minute = SimpleNamespace(name="m", unit_cost_micro_usd=13_333, unit_label="minute")
    assert minute_prices([hourly, minute]) == {"m": 13_333}


def test_a_head_follows_redirects_to_the_final_status():
    def handler(request):
        assert request.method == "HEAD" and request.extensions["timeout"] == clients.REQUEST_TIMEOUT.as_dict()
        if request.url.host == "github.com":
            return httpx2.Response(302, headers={"location": "https://cdn.example/asset"}, request=request)
        return httpx2.Response(200 if request.url.path == "/asset" else 404, request=request)

    transport = httpx2.MockTransport(handler)
    assert head_status("https://github.com/o/r/releases/download/t/asset", transport=transport) == 200
    assert head_status("https://cdn.example/gone", transport=transport) == 404
    # asked not to follow, it gives the redirect itself
    assert head_status("https://github.com/o/r/releases/download/t/asset", transport=transport, follow=False) == 302
    # the links of a card are asked that way; a release asset is followed to where it is stored
    assert Deps.link.func is head_status and Deps.link.keywords == {"follow": False} and Deps.head is head_status


def test_registration_and_token_owner(monkeypatch, tmp_path, settings, fake_hub, deps):
    """An unconfirmed registration, a foreign token and a results repo without write access fail (TS-RDY-2)."""
    only(monkeypatch, 1, 2)
    assert [item.status for item in ready(settings, deps).items] == ["PASS", "PASS"]

    unconfirmed = load_settings(settings_toml(tmp_path, readiness={"registration_confirmed": False}))
    undated = load_settings(settings_toml(tmp_path, readiness={"confirmed_on": None}))
    for unregistered in (unconfirmed, undated):
        first = ready(unregistered, deps).items[0]
        assert first.status == "FAIL" and "confirmed_on" in first.hint

    fake_hub.user = "someone-else"
    report = ready(settings, deps)
    assert report.items[1].status == "FAIL" and "hf auth login" in report.items[1].hint
    assert report.exit_code is ExitCode.READINESS
    assert "someone-else" not in (tmp_path / readiness.READY_PATH).read_text()

    fake_hub.user = "vincentnkq"
    fake_hub.fail_after(1, 401)
    expired = ready(settings, deps).items[1]
    assert expired.status == "FAIL" and "401" in expired.detail and "hf auth login" in expired.hint
    fake_hub.private.add(RESULTS)
    refused = ready(settings, deps).items[1]
    assert refused.status == "FAIL" and "401" in refused.detail and "write" in refused.hint
    del fake_hub.repos[RESULTS]
    missing = ready(settings, deps)
    assert missing.items[1].status == "FAIL" and "hf repo create" in missing.items[1].hint
    assert missing.exit_code == 3
    assert [name for name, _ in fake_hub.calls if name not in ("whoami", "auth_check")] == []


@pytest.mark.parametrize(
    ("status", "error_code"), [(403, "GatedRepo"), (401, None), (403, None)], ids=["gated", "401", "bare-403"]
)
def test_a_closed_gate_names_its_page(monkeypatch, settings, fake_hub, deps, status, error_code):
    """A closed model gate fails with its URL; a closed base gate is only recorded (TS-RDY-3)."""
    only(monkeypatch, 3)
    close(fake_hub, BASE, status, error_code)
    report = ready(settings, deps)
    [item] = report.items
    assert item.status == "PASS" and f"https://huggingface.co/{BASE}" in item.detail
    assert report.gates == {AYA: "ok", BASE: str(status)} and report.exit_code == 0

    close(fake_hub, AYA, status, error_code)
    report = ready(settings, deps)
    [item] = report.items
    assert item.status == "FAIL" and f"https://huggingface.co/{AYA}" in item.hint
    assert report.gates[AYA] == str(status) and report.exit_code == 3
    assert [name for name, _ in deps.client.calls if name.startswith("run_")] == []


def test_credit_is_compared_in_micro_dollars(monkeypatch, tmp_path, settings, deps):
    """13,600,000 left passes, one less fails, and a refused billing read fails after one call (TS-RDY-4)."""
    only(monkeypatch, 4)
    deps.client.billing_used_micro_usd = 6_400_000
    report = ready(settings, deps)
    assert report.items[0].status == "PASS" and "13600000" in report.items[0].detail
    assert report.billing["usedMicroUsd"] == 6_400_000 and "jobDetails" not in report.billing
    assert json.loads((tmp_path / readiness.READY_PATH).read_text())["billing"]["usedMicroUsd"] == 6_400_000

    deps.client.billing_used_micro_usd = 6_400_001
    short = ready(settings, deps)
    assert short.items[0].status == "FAIL" and "13599999" in short.items[0].detail and short.exit_code == 3

    # what the ledger already holds is no longer needed from the credit
    deps.ledger.total = 1_000_000
    for used, status in ((7_400_000, "PASS"), (7_400_001, "FAIL")):
        deps.client.billing_used_micro_usd = used
        item = ready(settings, deps).items[0]
        assert item.status == status and "12600000" in item.detail
    deps.ledger.total = 14_000_000
    for used, status in ((20_000_000, "PASS"), (20_000_001, "FAIL")):
        deps.client.billing_used_micro_usd = used
        assert ready(settings, deps).items[0].status == status
    deps.ledger.total = 0

    for status in (401, 403):
        deps.client.calls.clear()
        deps.client.billing_status = status
        refused = ready(settings, deps)
        assert refused.items[0].status == "FAIL" and "Billing" in refused.items[0].hint
        assert refused.billing is None and refused.exit_code == 3
        assert [name for name, _ in deps.client.calls] == ["billing_usage"]

    deps.client.billing_status = 200
    usage = deps.client.billing_usage()
    for access in (False, None, "yes"):
        monkeypatch.setattr(deps.client, "billing_usage", lambda access=access: {**usage, "hasAccess": access})
        barred = ready(settings, deps)
        assert barred.items[0].status == "FAIL" and "hasAccess" in barred.items[0].detail
        assert barred.billing["hasAccess"] == access


def test_the_gpu_flavor_is_priced_per_minute(monkeypatch, settings, deps):
    """l4x1 is priced, a10g-small stands in for it, and a missing or hourly price fails (TS-RDY-5)."""
    only(monkeypatch, 5)
    rows = {row["name"]: row for row in DEFAULT_HARDWARE}

    def run(*names, **changed):
        deps.client.hardware = [{**rows[name], **changed} for name in names]
        report = ready(settings, deps)
        return report.items[0], report

    item, report = run("cpu-basic", "l4x1", "a10g-small")
    assert item.status == "PASS" and "13333" in item.detail
    assert report.gpu_flavor == "l4x1" and report.prices_micro_usd_per_min["l4x1"] == 13_333

    item, report = run("cpu-basic", "a10g-small")
    assert item.status == "PASS" and "a10g-small" in item.detail and "l4x1 is not offered" in item.detail
    assert report.gpu_flavor == "a10g-small" and report.prices_micro_usd_per_min["a10g-small"] == 16_667

    item, report = run("cpu-basic")
    assert item.status == "FAIL" and report.gpu_flavor is None and report.exit_code == 3

    item, report = run("l4x1", "a10g-small", unitLabel="hour")
    assert item.status == "FAIL" and "'hour'" in item.detail and report.gpu_flavor is None


def test_lora_attaches_on_every_projection(settings, deps):
    """Each of the seven projections gets trainable weights; an unknown name is called out (TS-RDY-6)."""
    assert len(readiness.LORA_TARGETS) == 7
    attached = readiness.lora_attaches(settings, deps)
    assert attached.ok, attached.detail
    for names in ((*readiness.LORA_TARGETS, "bogus_proj"), ("bogus_proj",)):
        missing = readiness.lora_attaches(settings, deps, names)
        assert not missing.ok and "bogus_proj" in missing.detail
        assert not any(name in missing.detail for name in readiness.LORA_TARGETS)


def test_the_comparison_seam_scores_stored_answers(monkeypatch, settings, deps):
    """Ten synthetic rows are scored offline through the precomputed model (TS-RDY-7)."""
    only(monkeypatch, 7)
    assert readiness.stored_answer_scores() == [0.0, 1.0] * 5
    assert ready(settings, deps).items[0].status == "PASS"
    monkeypatch.setattr(readiness, "stored_answer_scores", lambda: [1.0] * 10)
    assert ready(settings, deps).items[0].status == "FAIL"


def test_release_image_and_code_pins(monkeypatch, tmp_path, settings, deps):
    """Reachable release files, recorded image facts and a clean pushed HEAD pass; anything else fails (TS-RDY-8)."""
    only(monkeypatch, 8, 9)
    # the code URL maps to a local repo; other transports are refused
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    release = settings.llamacpp_release
    urls = [release.source_tarball, release.cpu_asset, release.cuda_asset]
    origin, work, other = tmp_path / "origin.git", tmp_path / "work", tmp_path / "other"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    git(work, "remote", "add", "origin", str(origin))
    first = commit(work, "a")
    git(work, "push", "-q", "origin", "main")
    probe_path = work / IMAGE_PROBE_PATH
    # a machine-wide excludes file may hide a .gitignore
    (work / ".git" / "info" / "exclude").write_text(".run-local/\n")
    git(work, "config", f"url.{origin}.insteadOf", settings.code.origin)
    dump(PROBE, probe_path)
    seen = []
    deps = replace(deps, repo_root=work, head=lambda url: seen.append(url) or 200)

    report = ready(settings, deps)
    assert [(item.number, item.status) for item in report.items] == [(8, "PASS"), (9, "PASS")]
    assert seen == urls and report.exit_code == 0

    def eighth(head=lambda url: 200, probe=PROBE):
        dump(probe, probe_path)
        return readiness.release_and_image(settings, replace(deps, head=head))

    gone = eighth(lambda url: 404 if url == release.cuda_asset else 200)
    assert not gone.ok and release.cuda_asset in gone.detail and "404" in gone.detail

    def unreachable(url):
        raise httpx2.ConnectError("no route")

    assert not eighth(unreachable).ok
    usual = eighth()
    assert usual.ok and "python 3.12.3" in usual.detail and "torch 2.14.1+cu126" in usual.detail
    for field, value in (("python", "3.11.9"), ("torch", "")):
        wrong = eighth(probe=PROBE.model_copy(update={field: value}))
        assert not wrong.ok and field in wrong.detail and str(IMAGE_PROBE_PATH) in wrong.detail
    # recorded, never failing
    odd = eighth(probe=PROBE.model_copy(update={"pep668": False, "tools": ["pip", "git", "uv"], "torch": "2.13.0"}))
    assert odd.ok and "no pep668 marker" in odd.detail and "git, pip, uv" in odd.detail and "2.13.0" in odd.detail
    unprobed = "image not probed; the smoke run covers it"
    foreign = eighth(probe=PROBE.model_copy(update={"image": IMAGE.replace("2.14.1", "2.13.0"), "python": "3.11.9"}))
    assert foreign.ok and unprobed in foreign.detail
    probe_path.write_text("{")
    torn = readiness.release_and_image(settings, deps)
    assert not torn.ok and str(IMAGE_PROBE_PATH) in torn.detail
    probe_path.unlink()
    absent = readiness.release_and_image(settings, deps)
    assert absent.ok and unprobed in absent.detail and release.tag in absent.detail
    assert not readiness.release_and_image(settings, replace(deps, head=lambda url: 503)).ok

    def fault(sha=None):
        return code_pin_fault(work, settings.code.origin, "main", sha)

    assert fault() is None and fault(first) is None
    # origin moves on: HEAD is an ancestor, no longer a tip
    git(tmp_path, "clone", "-q", str(origin), str(other))
    second = commit(other, "b")
    git(other, "push", "-q", "origin", "main")
    assert fault() is None
    assert "HEAD" in fault(second)

    (work / "untracked").write_text("x")
    assert "uncommitted" in fault()
    (work / "untracked").unlink()
    (work / "a").write_text("changed")
    assert "uncommitted" in fault()
    git(work, "checkout", "-q", "--", "a")
    assert fault() is None

    unpushed = commit(work, "c")
    assert f"not on main of {settings.code.origin}" in fault() and "not on main" in fault(unpushed)
    assert readiness.code_pinned(settings, deps).ok is False
    # pushed to the checkout's own remote only: the configured code URL decides
    fork = tmp_path / "fork.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(fork))
    git(work, "remote", "set-url", "origin", str(fork))
    git(work, "push", "-q", "origin", "main")
    assert "not on main" in fault()
    git(work, "reset", "-q", "--hard", first)
    assert fault() is None

    git(work, "config", "--unset-all", f"url.{origin}.insteadOf")
    git(work, "config", f"url.{tmp_path / 'nowhere.git'}.insteadOf", settings.code.origin)
    assert "fetch" in fault()
    # without the rewrite the URL is https, which this test forbids
    git(work, "config", "--unset-all", f"url.{tmp_path / 'nowhere.git'}.insteadOf")
    assert "fetch" in fault()
    assert "not a git checkout" in code_pin_fault(tmp_path / "hub", settings.code.origin, "main")


def test_a_stale_or_missing_report_stops_the_verb(tmp_path, clock):
    """No report or one 25 hours old raises exit 3; 23 hours old passes (TS-RDY-9)."""
    path = tmp_path / "ready.json"
    with pytest.raises(NotReady, match="driftmap ready") as absent:
        require_ready(path, "c", now=clock.now)
    assert absent.value.exit_code is ExitCode.READINESS

    items = [Item(number=number, name=name, status="PASS", detail="") for number, name, _ in readiness.ITEMS]
    for hours, fresh in ((25, False), (24, False), (23, True), (0, True), (-1, False)):
        dump(ReadyReport(ts=clock.now() - timedelta(hours=hours), config_hash="c", items=items), path)
        if fresh:
            assert require_ready(path, "c", now=clock.now).ok
        else:
            with pytest.raises(NotReady, match="driftmap ready"):
                require_ready(path, "c", now=clock.now)
    dump(ReadyReport(ts=clock.now() - timedelta(hours=30), config_hash="c", items=items), path)
    assert require_ready(path, "c", max_age_h=48, now=clock.now).ok
    with pytest.raises(NotReady, match="configuration.*driftmap ready"):
        require_ready(path, "d", max_age_h=48, now=clock.now)
