import os
from pathlib import Path

import huggingface_hub.constants
import pytest
from hypothesis import settings

pytest_plugins = ["pytester"]

settings.register_profile("driftmap", deadline=None, max_examples=200)
settings.load_profile("driftmap")

REPO_ROOT = Path(__file__).resolve().parents[1]

# huggingface_hub resolves these at import, so env vars alone are too late
_HF_PATH_CONSTANTS = {
    "HF_HOME": "",
    "HF_HUB_CACHE": "hub",
    "HF_ASSETS_CACHE": "assets",
    "HF_XET_CACHE": "xet",
    "HF_TOKEN_PATH": "token",
    "HF_STORED_TOKENS_PATH": "stored_tokens",
}
# token sources huggingface_hub reads before HF_TOKEN, plus AWS_PROFILE
_SCRUBBED_ENV = ("AWS_PROFILE", "HF_OIDC_RESOURCE", "HF_OIDC_ID_TOKEN", "HUGGING_FACE_HUB_TOKEN")


@pytest.fixture(autouse=True)
def hermetic(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("smoke"):
        return
    home = tmp_path / "home"
    cache = home / ".cache"
    hf_home = cache / "huggingface"
    work = tmp_path / "work"
    hf_home.mkdir(parents=True)
    (work / ".run-local").mkdir(parents=True)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    for name in _SCRUBBED_ENV:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    for name, sub in _HF_PATH_CONSTANTS.items():
        monkeypatch.setattr(huggingface_hub.constants, name, str(hf_home / sub))
    monkeypatch.chdir(work)


# child processes have live sockets, so they never get the full env
_CHILD_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_TOKEN",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
)


@pytest.fixture
def child_env() -> dict[str, str]:
    """Environment for a child process spawned by a test."""
    env = {name: os.environ[name] for name in _CHILD_ENV_ALLOWLIST if name in os.environ}
    # lets the child import tests.fakes; do not spawn it with -I
    env["PYTHONPATH"] = str(REPO_ROOT)
    # DRIFTMAP_CLIENT=fake:<path> is only honoured under pytest
    env["PYTEST_CURRENT_TEST"] = os.environ["PYTEST_CURRENT_TEST"]
    return env
