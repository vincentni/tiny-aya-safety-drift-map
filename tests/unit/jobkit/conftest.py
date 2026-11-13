import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from driftmap.core import hub as hub_module
from driftmap.core.settings import load_settings
from driftmap.jobkit import main
from tests.fakes import FakeClock, FakeHub
from tests.fakes.settings import settings_toml

RUN_ID = "generate-full-20261102T100000Z"
BASE_ARGV = {
    "--run-id": RUN_ID,
    "--timeout-s": "5400",
    "--code-sha": "a" * 40,
    # FakeClock starts at this instant
    "--submitted-at": "2026-11-02T10:00:00Z",
}

# a job on a disk-backed Hub; the mode picks where it stops
CHILD = """
import os
import signal
import sys
import time
from pathlib import Path

from driftmap.core.settings import load_settings
from driftmap.jobkit import main
from tests.fakes import FakeHub

root, mode = Path(sys.argv[1]), sys.argv[2]
settings = load_settings(root / "driftmap.toml")
responses = f"{settings.owner}/{settings.repos.responses}"
stalled = []


class DiskHub(FakeHub):
    def upload_file(self, **kwargs):
        path = kwargs["path_in_repo"]
        if mode == "kill-before-record" and path.startswith("runs/"):
            os.kill(os.getpid(), signal.SIGKILL)
        if mode.startswith("stall") and path.endswith("fr.jsonl"):
            time.sleep(60)
        if mode == "stall-marker" and path.startswith("partial/") and not stalled:
            stalled.append(path)
            time.sleep(60)
        if mode == "term-while-closing" and path.startswith("runs/"):
            os.kill(os.getpid(), signal.SIGTERM)
        commit = super().upload_file(**kwargs)
        copy = root / "store" / kwargs["repo_id"] / path
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(self.repos[kwargs["repo_id"]]["main"][path])
        return commit


def body(ctx):
    ctx.flusher.upload(responses, "responses/x/harmful/en.jsonl", b"en\\n", key="harmful/en")
    if mode == "kill-in-body":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "term":
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(60)
    if mode == "stall":
        ctx.flusher.upload(responses, "responses/x/harmful/fr.jsonl", b"fr\\n", key="harmful/fr")
    if mode == "stall-closing":
        ctx.flusher.add(responses, "responses/x/harmful/fr.jsonl", b"fr\\n", key="harmful/fr")
        return
    if mode == "absorb":
        try:
            time.sleep(60)
        except BaseException:
            pass
        time.sleep(15)
        (root / "slept").touch()
    ctx.checkpoint()


hub = DiskHub(root / "hub").add_repo(responses, repo_type="dataset")
sys.exit(main.run(body, sys.argv[3:], kind="generate", settings=settings, client=hub))
"""


@pytest.fixture(autouse=True)
def fresh_token():
    hub_module.token.cache_clear()
    yield
    hub_module.token.cache_clear()


@pytest.fixture
def settings(tmp_path):
    return load_settings(settings_toml(tmp_path))


@pytest.fixture
def results(settings):
    return f"{settings.owner}/{settings.repos.results}"


@pytest.fixture
def responses(settings):
    return f"{settings.owner}/{settings.repos.responses}"


@pytest.fixture
def fake(tmp_path, responses):
    return FakeHub(tmp_path / "hub").add_repo(responses, repo_type="dataset")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def argv():
    def build(*extra, **overrides):
        merged = BASE_ARGV | {f"--{name.replace('_', '-')}": value for name, value in overrides.items()}
        return [part for pair in merged.items() for part in pair] + list(extra)

    return build


@pytest.fixture
def launch(settings, fake, clock, argv):
    """Run a body through jobkit on the fake Hub and clock; returns the exit code."""

    def start(body, *extra, **overrides):
        return main.run(body, argv(*extra, **overrides), kind="generate", settings=settings, client=fake, clock=clock)

    return start


@pytest.fixture
def uploads(fake):
    """Paths uploaded so far, in call order."""
    return lambda: [kwargs["path_in_repo"] for method, kwargs in fake.calls if method == "upload_file"]


@pytest.fixture
def stored(fake, results):
    """Bytes of a file in the results repo, or None."""
    return lambda path: fake.repos[results]["main"].get(path)


@pytest.fixture
def child_job(tmp_path, child_env, argv):
    """Run the child job, submitted ``age_s`` seconds ago; returns the process and the files its Hub stored."""

    def spawn(mode, age_s=0):
        settings_toml(tmp_path)
        submitted = datetime.now(UTC) - timedelta(seconds=age_s)
        command = [sys.executable, "-c", CHILD, str(tmp_path), mode]
        command += argv(submitted_at=f"{submitted:%Y-%m-%dT%H:%M:%SZ}")
        child = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=60)
        store = tmp_path / "store"
        return child, {
            path.relative_to(store).as_posix(): path.read_bytes() for path in store.rglob("*") if path.is_file()
        }

    return spawn
