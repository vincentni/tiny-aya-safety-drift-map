import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx2
import huggingface_hub.constants
import pytest
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.utils import get_session
from huggingface_hub.utils._http import default_client_factory

from driftmap.core import clients, hub
from driftmap.core.clients import make_jobs_client
from driftmap.core.hub import Hub
from driftmap.core.schemas import DriftmapError, ExitCode
from tests.fakes import FileBackedFakeClient

SRC = Path(__file__).resolve().parents[3] / "src"
REAL_CLIENTS = {"HfApi", "HfJobsClient"}

CHILD = """
import sys
from driftmap.core.clients import make_jobs_client
from driftmap.core.schemas import DriftmapError

try:
    client = make_jobs_client(None)
except DriftmapError as exc:
    print(exc, file=sys.stderr)
    sys.exit(exc.exit_code)
client.list_jobs_hardware()
print(type(client).__name__)
"""


@pytest.fixture(autouse=True)
def fresh_token_and_session():
    hub.token.cache_clear()
    yield
    hub.token.cache_clear()
    set_client_factory(default_client_factory)


def run_child(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=60)


def files_under(root: Path) -> set[Path]:
    return {path for path in root.rglob("*") if path.is_file()}


def test_the_fake_is_for_tests_only(tmp_path, monkeypatch, child_env):
    """Unset builds the real client; fake:<path> works under pytest and is refused outside (TS-NFR-7)."""
    recorder = tmp_path / "recorder.jsonl"
    monkeypatch.setenv("DRIFTMAP_CLIENT", f"fake:{recorder}")
    fake = make_jobs_client(None)
    assert type(fake) is FileBackedFakeClient and fake.path == recorder
    token = "hf_" + "a" * 34
    fake.run_job(image="python:3.12", command=["true"], secrets={"HF_TOKEN": token})
    [line] = [json.loads(row) for row in recorder.read_text().splitlines()]
    assert line["method"] == "run_job" and line["kwargs"]["secrets"] == {"HF_TOKEN": "***"}
    assert token not in recorder.read_text()

    for spec in ("fake:", "bogus:x", "fake", ""):
        monkeypatch.setenv("DRIFTMAP_CLIENT", spec)
        with pytest.raises(DriftmapError) as malformed:
            make_jobs_client(None)
        assert malformed.value.exit_code is ExitCode.USAGE and "DRIFTMAP_CLIENT" in str(malformed.value)

    monkeypatch.delenv("DRIFTMAP_CLIENT")
    real = make_jobs_client(None)
    assert type(real) is HfApi and real.token == "hf_dummy"
    with pytest.raises(OfflineModeIsEnabled):
        real.whoami()

    outside = tmp_path / "outside.jsonl"
    before = files_under(tmp_path)
    env = {k: v for k, v in child_env.items() if k != "PYTEST_CURRENT_TEST"}
    for spec in (f"fake:{outside}", "fake:", "bogus:x"):
        refused = run_child({**env, "DRIFTMAP_CLIENT": spec})
        assert refused.returncode == 2, refused.stderr
        assert "DRIFTMAP_CLIENT" in refused.stderr and refused.stdout == ""
    assert files_under(tmp_path) == before and not outside.exists()


def test_one_factory_builds_the_client_and_tests_get_the_fake(tmp_path, monkeypatch, child_env):
    """Only the factory constructs the real client or reads DRIFTMAP_CLIENT (TS-NFR-8)."""
    constructed, readers = set(), set()
    for path in SRC.rglob("*.py"):
        name = path.relative_to(SRC / "driftmap").as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) in REAL_CLIENTS
            ):
                constructed.add(name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "DRIFTMAP_CLIENT" in node.value:
                readers.add(name)
    assert constructed == readers == {"core/clients.py"}

    spec = os.environ["DRIFTMAP_CLIENT"]
    assert spec.startswith("fake:") and Path(spec.removeprefix("fake:")).is_relative_to(tmp_path)
    assert type(make_jobs_client(None)) is FileBackedFakeClient
    assert child_env["DRIFTMAP_CLIENT"] == spec and child_env["PYTEST_CURRENT_TEST"]
    child = run_child(child_env)
    assert (child.returncode, child.stdout.split()) == (0, ["FileBackedFakeClient"]), child.stderr
    [line] = Path(spec.removeprefix("fake:")).read_text().splitlines()
    assert json.loads(line)["method"] == "list_jobs_hardware"


def test_the_session_gets_a_default_timeout_and_keeps_the_library_hook(monkeypatch):
    assert get_session().timeout == httpx2.Timeout(None)
    monkeypatch.delenv("DRIFTMAP_CLIENT")
    real = make_jobs_client(None)
    session = get_session()
    assert session.timeout == httpx2.Timeout(60, connect=10)
    assert session.follow_redirects
    assert session.event_hooks["request"][:-1] == default_client_factory().event_hooks["request"] != []
    with pytest.raises(OfflineModeIsEnabled):
        real.whoami()


def test_no_request_leaves_without_a_timeout(tmp_path, monkeypatch):
    sha = "0123456789abcdef0123456789abcdef01234567"
    seen = []

    def handler(request):
        seen.append((request.url.path, request.extensions["timeout"]))
        listing = "/tree/" in request.url.path
        return httpx2.Response(200, json=[] if listing else {"id": "vincentnkq/x", "sha": sha}, request=request)

    def unguarded():
        return httpx2.Client(transport=httpx2.MockTransport(handler), follow_redirects=True, timeout=None)

    monkeypatch.setattr(clients, "default_client_factory", unguarded)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.delenv("DRIFTMAP_CLIENT")
    remote = Hub(make_jobs_client(None), sleep=lambda seconds: pytest.fail(f"slept {seconds}"))
    assert remote.model_info(f"vincentnkq/x@{sha}").sha == sha
    assert remote.dataset_info(f"vincentnkq/x@{sha}").sha == sha
    remote.snapshot_download(f"vincentnkq/x@{sha}", tmp_path / "dl")
    paths = [path for path, _ in seen]
    assert paths[:2] == ["/api/models/vincentnkq/x/revision/" + sha, "/api/datasets/vincentnkq/x/revision/" + sha]
    assert any("/tree/" in path for path in paths[2:])
    assert [timeout for _, timeout in seen] == [clients.REQUEST_TIMEOUT.as_dict()] * len(seen)

    seen.clear()
    get_session().get("https://huggingface.co/api/whoami-v2", timeout=10)
    get_session().get("https://huggingface.co/api/whoami-v2", timeout=httpx2.Timeout(5, read=None))
    assert [timeout for _, timeout in seen] == [httpx2.Timeout(10).as_dict(), clients.REQUEST_TIMEOUT.as_dict()]


def test_a_registered_backend_wraps_the_api_the_factory_built(monkeypatch):
    monkeypatch.delenv("DRIFTMAP_CLIENT")
    monkeypatch.setattr(clients, "_backend", clients._backend)
    clients.register_backend(lambda settings, api: (settings, api))
    settings, api = make_jobs_client("settings")
    assert settings == "settings" and type(api) is HfApi
