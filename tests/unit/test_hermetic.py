import hashlib
import inspect
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from huggingface_hub import CommitInfo, HfApi
from huggingface_hub._jobs_api import (
    TERMINAL_JOB_STAGES,
    JobHardwareInfo,
    JobInfo,
    JobStage,
)
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    JobNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import parse_ratelimit_headers
from pytest_socket import SocketBlockedError

from tests.fakes import (
    FakeClock,
    FakeHub,
    FakeJobsClient,
    FakeSubprocess,
    FakeTransport,
    FileBackedFakeClient,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

INNER_TESTS = """
import os
import socket
from pathlib import Path

import pytest
import huggingface_hub.constants
from pytest_socket import SocketBlockedError

CWD_AT_COLLECTION = os.getcwd()
HOME_AT_COLLECTION = os.environ.get("HOME")


def test_socket_is_blocked():
    with pytest.raises(SocketBlockedError):
        socket.socket()


def test_fixture_env(tmp_path):
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["HF_TOKEN"] == "hf_dummy"
    for var in ("AWS_PROFILE", "HF_OIDC_RESOURCE", "HF_OIDC_ID_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert var not in os.environ, var
    for var in ("HOME", "XDG_CACHE_HOME", "HF_HOME"):
        assert Path(os.environ[var]).is_relative_to(tmp_path), var
    assert Path.home().is_relative_to(tmp_path)
    assert Path.cwd().is_relative_to(tmp_path)
    assert (Path.cwd() / ".run-local").is_dir()
    assert Path("~/.driftmap/launch.lock").expanduser().is_relative_to(tmp_path)
    assert Path(os.environ["DRIFTMAP_CLIENT"].removeprefix("fake:")).is_relative_to(tmp_path)
    assert huggingface_hub.constants.HF_HUB_OFFLINE is True


@pytest.mark.smoke
def test_smoke_item_only_loses_the_fake_client():
    assert "DRIFTMAP_CLIENT" not in os.environ
    assert os.getcwd() == CWD_AT_COLLECTION
    assert os.environ.get("HOME") == HOME_AT_COLLECTION
    assert not Path(".run-local").exists()
"""


def test_hermetic_fixture_blocks_sockets_and_scrubs_env(pytester, monkeypatch):
    """Non-smoke tests get no sockets and a scrubbed environment (TS-NFR-1)."""
    for var in ("AWS_PROFILE", "HF_OIDC_RESOURCE", "HF_OIDC_ID_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.setenv(var, "operator")
    monkeypatch.setenv("DRIFTMAP_CLIENT", "fake:operator.jsonl")
    pytester.makeconftest((REPO_ROOT / "tests" / "conftest.py").read_text())
    ini = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    pytester.makepyprojecttoml(
        f"[tool.pytest.ini_options]\naddopts = {ini['addopts']!r}\nmarkers = {ini['markers']!r}\n"
    )
    pytester.makepyfile(test_inner=INNER_TESTS)
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=3)


def test_hub_offline_and_raw_socket_blocked():
    """The Hub client fails offline and a raw socket is refused (TS-HUB-5)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import OfflineModeIsEnabled

    with pytest.raises(OfflineModeIsEnabled):
        HfApi().whoami()
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("huggingface.co", 443))


PLANTED_SECRETS = {"LANGFUSE_SECRET_KEY": "sk-lf-planted", "AWS_SECRET_ACCESS_KEY": "planted-aws-secret"}


@pytest.fixture
def parent_with_secrets(monkeypatch):
    for name, value in PLANTED_SECRETS.items():
        monkeypatch.setenv(name, value)


def test_child_env_lets_a_child_process_import_the_fake_client(parent_with_secrets, child_env):
    assert child_env["HF_HUB_OFFLINE"] == "1"
    assert child_env["HF_TOKEN"] == "hf_dummy"
    assert child_env["PYTHONPATH"] == str(REPO_ROOT)
    # the fixture ran in the "(setup)" stage, this body in "(call)"
    assert child_env["PYTEST_CURRENT_TEST"] == os.environ["PYTEST_CURRENT_TEST"].replace("(call)", "(setup)")
    assert child_env["HOME"] == os.environ["HOME"]
    assert "AWS_PROFILE" not in child_env
    assert not set(PLANTED_SECRETS) & set(child_env)
    assert Path.cwd() != REPO_ROOT
    code = (
        "import os, tests.fakes.jobs_client as m;"
        "print(m.FileBackedFakeClient.__name__, os.environ['PYTEST_CURRENT_TEST'] != '', os.environ['HF_HUB_OFFLINE'],"
        " sorted(k for k in os.environ if k in ('LANGFUSE_SECRET_KEY', 'AWS_SECRET_ACCESS_KEY')))"
    )
    done = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["FileBackedFakeClient", "True", "1", "[]"]
    for value in PLANTED_SECRETS.values():
        assert value not in done.stdout


RESULTS = "vincentnkq/safety-drift-map-results"


class TestFakeHub:
    def test_errors_are_library_built_per_status(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.add_repo("CohereLabs/tiny-aya-global", gated=True)
        hub.add_repo("vincentnkq/private-adapter", private=True)
        hub.add_repo(RESULTS, repo_type="dataset", files={"runs/r1.json": b"{}"})
        with pytest.raises(GatedRepoError) as gated:
            hub.auth_check("CohereLabs/tiny-aya-global")
        assert gated.value.response.status_code == 403
        assert isinstance(gated.value, RepositoryNotFoundError)
        with pytest.raises(RepositoryNotFoundError) as private:
            hub.auth_check("vincentnkq/private-adapter")
        assert private.value.response.status_code == 401
        assert not isinstance(private.value, GatedRepoError)
        with pytest.raises(RepositoryNotFoundError) as missing:
            hub.auth_check("nobody/missing")
        assert missing.value.response.status_code == 404
        with pytest.raises(RevisionNotFoundError) as revision:
            hub.snapshot_download(RESULTS, repo_type="dataset", revision="0" * 40, local_dir=tmp_path / "x")
        assert revision.value.response.status_code == 404
        hub.authorize("CohereLabs/tiny-aya-global")
        assert hub.auth_check("CohereLabs/tiny-aya-global") is None

    def test_repo_exists_follows_the_library_mapping(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.add_repo("CohereLabs/tiny-aya-global", gated=True)
        hub.add_repo("vincentnkq/private-adapter", private=True)
        hub.add_repo(RESULTS, repo_type="dataset")
        assert hub.repo_exists("CohereLabs/tiny-aya-global") is True
        assert hub.repo_exists("vincentnkq/private-adapter") is False
        assert hub.repo_exists("nobody/missing") is False
        assert hub.repo_exists(RESULTS, repo_type="dataset") is True
        hub.authorize("vincentnkq/private-adapter")
        assert hub.repo_exists("vincentnkq/private-adapter") is True

    def test_upload_to_a_frozen_revision_is_a_400_and_leaves_the_snapshot_alone(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.create_repo(RESULTS, repo_type="dataset")
        first = hub.upload_file(
            path_or_fileobj=b"abc", path_in_repo="runs/r1.json", repo_id=RESULTS, repo_type="dataset"
        )
        folder = tmp_path / "stage"
        folder.mkdir()
        (folder / "x.json").write_bytes(b"x")
        for call in (
            lambda: hub.upload_file(
                path_or_fileobj=b"zzz",
                path_in_repo="runs/r2.json",
                repo_id=RESULTS,
                repo_type="dataset",
                revision=first.oid,
            ),
            lambda: hub.upload_folder(folder_path=folder, repo_id=RESULTS, repo_type="dataset", revision=first.oid),
        ):
            with pytest.raises(HfHubHTTPError) as rejected:
                call()
            assert rejected.value.response.status_code == 400
        assert hub.repos[RESULTS][first.oid] == {"runs/r1.json": b"abc"}
        assert hub.repos[RESULTS]["main"] == {"runs/r1.json": b"abc"}
        assert hub.heads[RESULTS] == first.oid

    def test_threads_set_detects_a_fan_out_and_concurrent_commits_do_not_crash(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.create_repo(RESULTS, repo_type="dataset")
        assert hub.threads == {threading.get_ident()}
        hub.upload_file(path_or_fileobj=b"serial", path_in_repo="serial.json", repo_id=RESULTS, repo_type="dataset")
        assert hub.threads == {threading.get_ident()}

        def upload(i: int) -> CommitInfo:
            return hub.upload_file(
                path_or_fileobj=bytes(64 * (i + 1)), path_in_repo=f"runs/{i}.json", repo_id=RESULTS, repo_type="dataset"
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            commits = list(pool.map(upload, range(40)))
        assert all(isinstance(c, CommitInfo) for c in commits)
        assert len({c.oid for c in commits}) == 40
        assert len(hub.threads) > 1
        assert len(hub.repos[RESULTS]["main"]) == 41
        assert hub.repos[RESULTS][hub.heads[RESULTS]] == hub.repos[RESULTS]["main"]
        assert hub.heads[RESULTS] in {c.oid for c in commits}

    def test_fail_after_raises_the_next_n_calls_with_the_ietf_header(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.fail_after(1, 429, ratelimit='"api";r=0;t=55')
        with pytest.raises(HfHubHTTPError) as limited:
            hub.whoami()
        assert limited.value.response.status_code == 429
        assert parse_ratelimit_headers(limited.value.response.headers).reset_in_seconds == 55
        assert hub.whoami()["name"] == "vincentnkq"
        hub.fail_after(2, 503)
        for _ in range(2):
            with pytest.raises(HfHubHTTPError) as down:
                hub.whoami()
            assert down.value.response.status_code == 503
        hub.whoami()
        hub.fail_after(1, 429, headers={"Retry-After": "7"})
        with pytest.raises(HfHubHTTPError) as retry_after:
            hub.whoami()
        assert parse_ratelimit_headers(retry_after.value.response.headers) is None
        assert len(hub.calls) == 6

    def test_upload_commit_download_and_lfs_sha(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.create_repo(RESULTS, repo_type="dataset")
        first = hub.upload_file(
            path_or_fileobj=b"abc", path_in_repo="runs/r1.json", repo_id=RESULTS, repo_type="dataset"
        )
        assert isinstance(first, CommitInfo)
        assert re.fullmatch(r"[0-9a-f]{40}", first.oid)
        assert first.commit_url == f"https://huggingface.co/datasets/{RESULTS}/commit/{first.oid}"
        second = hub.upload_file(
            path_or_fileobj=b"abd", path_in_repo="runs/r1.json", repo_id=RESULTS, repo_type="dataset"
        )
        assert second.oid != first.oid
        assert hub.repos[RESULTS]["main"]["runs/r1.json"] == b"abd"
        assert hub.repos[RESULTS][first.oid]["runs/r1.json"] == b"abc"

        folder = tmp_path / "stage"
        (folder / "tables").mkdir(parents=True)
        (folder / "tables" / "t.json").write_bytes(b"[]")
        (folder / "model.safetensors").write_bytes(b"\x00" * 64)
        info = hub.upload_folder(folder_path=folder, repo_id=RESULTS, repo_type="dataset")
        assert hub.repo_info(RESULTS, repo_type="dataset").sha == info.oid
        assert hub.repo_info(RESULTS, repo_type="dataset", revision=first.oid).sha == first.oid

        local = Path(
            hub.snapshot_download(RESULTS, repo_type="dataset", local_dir=tmp_path / "dl", allow_patterns=["tables/*"])
        )
        assert (local / "tables" / "t.json").read_bytes() == b"[]"
        assert not (local / "runs").exists()
        [shard] = hub.get_paths_info(RESULTS, ["model.safetensors"], repo_type="dataset")
        assert isinstance(shard, RepoFile)
        assert shard.lfs.sha256 == hashlib.sha256(b"\x00" * 64).hexdigest()
        assert shard.lfs.size == 64
        assert hub.get_paths_info(RESULTS, ["runs/r1.json"], repo_type="dataset")[0].lfs is None
        assert hub.max_in_flight == 1
        assert [c[0] for c in hub.calls][:3] == ["create_repo", "upload_file", "upload_file"]

    def test_downloads_fail_and_fall_back_like_the_library(self, tmp_path):
        hub = FakeHub(tmp_path / "hub")
        hub.add_repo("CohereLabs/tiny-aya-global", gated=True, files={"config.json": b"{}"})
        aya = {"repo_id": "CohereLabs/tiny-aya-global", "revision": hub.heads["CohereLabs/tiny-aya-global"]}
        fresh, stale = tmp_path / "fresh", tmp_path / "stale"
        (stale / ".cache").mkdir(parents=True)
        for download in (
            lambda target: hub.snapshot_download(**aya, local_dir=target),
            lambda target: hub.hf_hub_download(filename="config.json", **aya, local_dir=target),
        ):
            with pytest.raises(GatedRepoError):
                download(fresh)
            for status in (503, 403, 429):
                hub.fail_after(1, status)
                with pytest.raises(LocalEntryNotFoundError) as unreachable:
                    download(fresh)
                assert type(unreachable.value.__cause__) is HfHubHTTPError
                assert unreachable.value.__cause__.response.status_code == status
            hub.fail_after(1, 401)
            with pytest.raises(RepositoryNotFoundError):
                download(fresh)
            hub.fail_after(1, 404, error_code="RevisionNotFound")
            with pytest.raises(RevisionNotFoundError):
                download(stale)
        assert not (fresh / "config.json").exists()
        assert (fresh / ".cache" / "huggingface" / "download").is_dir()

        hub.fail_after(1, 503)
        assert hub.snapshot_download(**aya, local_dir=stale) == str(stale)
        assert hub.snapshot_download(**aya, local_dir=stale) == str(stale)
        assert not (stale / "config.json").exists()
        hub.fail_after(1, 503)
        with pytest.raises(LocalEntryNotFoundError):
            hub.hf_hub_download(filename="config.json", **aya, local_dir=stale)
        (stale / "config.json").write_bytes(b"old")
        hub.fail_after(1, 503)
        assert hub.hf_hub_download(filename="config.json", **aya, local_dir=stale) == str(stale / "config.json")
        assert (stale / "config.json").read_bytes() == b"old"

    def test_an_unchanged_upload_makes_no_commit(self, tmp_path):
        hub = FakeHub(tmp_path)
        hub.create_repo(RESULTS, repo_type="dataset")
        first = hub.upload_file(path_or_fileobj=b"abc", path_in_repo="a.json", repo_id=RESULTS, repo_type="dataset")
        revisions = set(hub.repos[RESULTS])
        again = hub.upload_file(path_or_fileobj=b"abc", path_in_repo="a.json", repo_id=RESULTS, repo_type="dataset")
        folder = tmp_path / "stage"
        folder.mkdir()
        (folder / "a.json").write_bytes(b"abc")
        same_folder = hub.upload_folder(folder_path=folder, repo_id=RESULTS, repo_type="dataset")
        assert again.oid == same_folder.oid == first.oid == hub.heads[RESULTS]
        assert again.commit_url == first.commit_url
        assert set(hub.repos[RESULTS]) == revisions
        (folder / "b.json").write_bytes(b"new")
        mixed = hub.upload_folder(folder_path=folder, repo_id=RESULTS, repo_type="dataset")
        assert mixed.oid != first.oid and hub.heads[RESULTS] == mixed.oid
        assert hub.repo_info(RESULTS, repo_type="dataset").sha == mixed.oid


class TestFakeJobsClient:
    def test_jobinfo_sequence_wait_and_terminal_stage(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: pytest.fail(f"slept {s}"))
        client = FakeJobsClient()
        client.script_job("j1", [{"stage": "RUNNING"}, {"stage": "COMPLETED"}])
        job = client.run_uv_job("jobs/drift.py", flavor="l4x1", secrets={"HF_TOKEN": "hf_x"}, timeout="3h")
        assert isinstance(job, JobInfo)
        assert job.id == "j1" and job.flavor == "l4x1"
        assert type(job.status.stage) is str and job.status.stage == "RUNNING"
        assert job.created_at.tzinfo is not None
        assert job.url == "https://huggingface.co/jobs/vincentnkq/j1"
        assert [j.id for j in client.list_jobs()] == ["j1"]
        final = client.wait_for_job("j1", timeout=10, poll_interval=60)
        assert final.status.stage == JobStage.COMPLETED
        assert final.status.stage in TERMINAL_JOB_STAGES
        method, kwargs = client.calls[-1]
        assert method == "wait_for_job"
        assert kwargs["timeout"] == 10 and kwargs["poll_interval"] == 60
        assert client.inspect_job(job_id="j1").status.stage == "COMPLETED"

    @pytest.mark.parametrize(
        "method",
        [
            "run_job",
            "run_uv_job",
            "inspect_job",
            "list_jobs",
            "wait_for_job",
            "cancel_job",
            "fetch_job_logs",
            "list_jobs_hardware",
            "whoami",
        ],
    )
    def test_parameter_kinds_match_hfapi(self, method):
        real = inspect.signature(getattr(HfApi, method)).parameters
        fake = inspect.signature(getattr(FakeJobsClient, method)).parameters
        for name, parameter in fake.items():
            if name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            assert name in real, f"{method}({name}) is not an HfApi parameter"
            assert parameter.kind is real[name].kind, f"{method}({name}) kind {parameter.kind} vs {real[name].kind}"

    @pytest.mark.parametrize("method", ["inspect_job", "cancel_job", "fetch_job_logs"])
    def test_job_id_is_keyword_only_like_hfapi(self, method):
        client = FakeJobsClient()
        client.script_job("j1", [{"stage": "COMPLETED"}])
        client.run_job(image="python:3.12", command=["true"])
        with pytest.raises(TypeError):
            getattr(client, method)("j1")
        getattr(client, method)(job_id="j1")

    def test_timeout_text_not_found_hardware_and_billing(self):
        client = FakeJobsClient()
        client.script_job("j2", [{"stage": "RUNNING"}])
        client.run_job(image="python:3.12", command=["python", "-c", "1"], flavor="cpu-basic")
        with pytest.raises(TimeoutError, match=r"^Job 'j2' is still in stage 'RUNNING' after 5 seconds\.$"):
            client.wait_for_job("j2", timeout=5)
        with pytest.raises(JobNotFoundError) as missing:
            client.inspect_job(job_id="nope")
        assert missing.value.response.status_code == 404
        client.cancel_job(job_id="j2")
        assert client.inspect_job(job_id="j2").status.stage == "CANCELED"
        assert client.fetch_job_logs(job_id="j2") == []
        hardware = {h.name: h for h in client.list_jobs_hardware()}
        assert isinstance(hardware["l4x1"], JobHardwareInfo)
        assert hardware["l4x1"].unit_cost_micro_usd == 13333
        assert hardware["l4x1"].unit_label == "minute"
        assert hardware["cpu-basic"].unit_cost_micro_usd == 167
        assert client.billing_usage()["usage"]["usedMicroUsd"] == 0
        client.billing_used_micro_usd = 1_999_950
        assert client.billing_usage()["usage"]["usedMicroUsd"] == 1_999_950
        for status in (401, 403):
            client.billing_status = status
            with pytest.raises(HfHubHTTPError) as denied:
                client.billing_usage()
            assert denied.value.response.status_code == status
            assert type(denied.value) is HfHubHTTPError

    def test_file_backed_client_records_every_call_with_secrets_masked(self, tmp_path):
        recorder = tmp_path / "client.jsonl"
        token = "hf_" + "a" * 34
        client = FileBackedFakeClient(recorder)
        job = client.run_uv_job(
            "jobs/drift.py", flavor="cpu-basic", secrets={"HF_TOKEN": token}, env={"DRIFTMAP_RUN_ID": "smoke-1"}
        )
        client.wait_for_job(job.id, timeout=1)
        client.list_jobs_hardware()
        client.billing_usage()
        lines = [json.loads(line) for line in recorder.read_text().splitlines()]
        assert [line["method"] for line in lines] == [
            "run_uv_job",
            "wait_for_job",
            "list_jobs_hardware",
            "billing_usage",
        ]
        assert token not in recorder.read_text()
        assert lines[0]["kwargs"]["secrets"] == {"HF_TOKEN": "***"}
        assert lines[0]["kwargs"]["env"] == {"DRIFTMAP_RUN_ID": "smoke-1"}
        assert lines[1]["reply"]["status"]["stage"] == "COMPLETED"


def test_fake_clock_advances_monotonic_and_utc_together():
    clock = FakeClock()
    start_mono, start_utc = clock.monotonic(), clock.now()
    assert start_utc.tzinfo is UTC
    clock.sleep(5)
    assert clock.monotonic() == start_mono + 5
    assert (clock.now() - start_utc).total_seconds() == 5
    assert clock.sleeps == [5]
    clock.advance(2.5)
    assert clock.monotonic() == start_mono + 7.5
    assert isinstance(clock.now(), datetime)


def test_fake_subprocess_records_argv_and_raises_the_real_error():
    fake = FakeSubprocess()
    fake.script(returncode=0, stdout="ok\n")
    fake.script(returncode=1, stderr="boom")
    fake.script(returncode=2)
    done = fake.run(["llama-quantize", "in.gguf", "out.gguf", "q4_k_m"], check=True, capture_output=True, text=True)
    assert isinstance(done, subprocess.CompletedProcess)
    assert (done.returncode, done.stdout) == (0, "ok\n")
    assert fake.calls[0].argv == ["llama-quantize", "in.gguf", "out.gguf", "q4_k_m"]
    assert fake.calls[0].kwargs["check"] is True
    with pytest.raises(subprocess.CalledProcessError) as failed:
        fake.run(["convert_hf_to_gguf.py"], check=True)
    assert failed.value.returncode == 1 and failed.value.stderr == "boom"
    assert fake.run(["false"]).returncode == 2
    with pytest.raises(AssertionError):
        fake.run(["unscripted"])


class TestFakeTransport:
    def test_health_arms(self):
        transport = FakeTransport()
        assert transport.request("GET", "/health") == {"status": "ok"}
        transport.health = "loading"
        with pytest.raises(urllib.error.HTTPError) as loading:
            transport.request("GET", "/health")
        assert loading.value.code == 503
        assert json.loads(loading.value.read())["error"] == {
            "code": 503,
            "message": "Loading model",
            "type": "unavailable_error",
        }
        transport.health = "refused"
        with pytest.raises(urllib.error.URLError) as refused:
            transport.request("GET", "/health")
        assert isinstance(refused.value.reason, ConnectionRefusedError)
        transport.health = "Loading"
        with pytest.raises(AssertionError):
            transport.request("GET", "/health")

    def test_replies_keyed_by_prompt_are_deterministic_under_four_workers(self):
        transport = FakeTransport()
        prompts = [f"prompt {i}" for i in range(12)]
        for prompt in prompts:
            transport.replies_for[prompt].append(
                {"content": f"echo {prompt}", "tokens_evaluated": 10, "stop_type": "eos", "truncated": False}
            )
        transport.replies_for["boom"].extend([500, 500, 500])
        transport.replies_for["boom"].append(
            {"content": "late", "tokens_evaluated": 1, "stop_type": "eos", "truncated": False}
        )
        transport.completions.append(
            {"content": "fifo", "tokens_evaluated": 1, "stop_type": "limit", "truncated": True}
        )

        def ask(prompt: str) -> str:
            try:
                return transport.request("POST", "/completion", {"prompt": prompt})["content"]
            except urllib.error.HTTPError as exc:
                return f"http {exc.code}"

        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(ask, [*prompts, "boom", "boom", "boom", "boom", "unscripted"]))
        assert answers[:12] == [f"echo {p}" for p in prompts]
        assert Counter(answers[12:16]) == Counter({"http 500": 3, "late": 1})
        assert answers[16] == "fifo"
        assert transport.max_in_flight >= 1

    def test_completion_tokenize_props_and_5xx(self):
        transport = FakeTransport()
        transport.completions.append(
            {"content": "I cannot help", "tokens_evaluated": 366, "stop_type": "eos", "truncated": False}
        )
        reply = transport.request("POST", "/completion", {"prompt": "x", "n_predict": 256, "temperature": 0})
        assert reply == {"content": "I cannot help", "tokens_evaluated": 366, "stop_type": "eos", "truncated": False}
        assert transport.calls[-1] == ("POST", "/completion", {"prompt": "x", "n_predict": 256, "temperature": 0})
        tokens = transport.request("POST", "/tokenize", {"content": "hello world", "add_special": True})["tokens"]
        assert tokens[0] == 2 and len(tokens) == 3
        assert (
            len(transport.request("POST", "/tokenize", {"content": "hello world", "add_special": False})["tokens"]) == 2
        )
        assert transport.request("GET", "/props")["total_slots"] == 1
        transport.fail_next(500)
        with pytest.raises(urllib.error.HTTPError) as server_error:
            transport.request("POST", "/completion", {"prompt": "y"})
        assert server_error.value.code == 500
        with pytest.raises(AssertionError):
            transport.request("POST", "/completion", {"prompt": "no script left"})
        assert transport.max_in_flight == 1
