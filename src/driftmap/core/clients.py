"""The one place a Jobs/Hub client is built."""

import importlib
import os
from collections.abc import Callable
from typing import Any

import httpx2
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.utils._http import default_client_factory

from driftmap.core import hub
from driftmap.core.schemas import DriftmapError, ExitCode

ENV_VAR = "DRIFTMAP_CLIENT"
REQUEST_TIMEOUT = httpx2.Timeout(60, connect=10)


def _bare(settings: Any, api: HfApi) -> Any:
    return api


_backend: Callable[[Any, HfApi], Any] = _bare


def register_backend(backend: Callable[[Any, HfApi], Any]) -> None:
    """Have the factory return ``backend(settings, api)`` instead of the bare ``HfApi``."""
    global _backend
    _backend = backend


def _fill_timeout(request: httpx2.Request) -> None:
    current = request.extensions.get("timeout") or {}
    if any(current.get(part) is None for part in ("connect", "read", "write", "pool")):
        request.extensions["timeout"] = REQUEST_TIMEOUT.as_dict()


def _session() -> httpx2.Client:
    client = default_client_factory()
    # no timeout by default; a stalled call bills until the job limit
    client.timeout = REQUEST_TIMEOUT
    # the library passes timeout=None on some calls, which overrides the client default
    client.event_hooks["request"] = [*client.event_hooks["request"], _fill_timeout]
    return client


def make_jobs_client(settings: Any) -> Any:
    """The Jobs/Hub client; ``DRIFTMAP_CLIENT=fake:<path>`` swaps in a recording fake, under pytest only."""
    spec = os.environ.get(ENV_VAR)
    if spec is None:
        set_client_factory(_session)
        return _backend(settings, HfApi(token=hub.token()))
    # a fake's records could later admit a paid launch
    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise DriftmapError(ExitCode.USAGE, f"{ENV_VAR} is set, but only the test suite may use it; unset it")
    scheme, _, path = spec.partition(":")
    if scheme != "fake" or not path:
        raise DriftmapError(ExitCode.USAGE, f"{ENV_VAR} must be fake:<path>, got {spec!r}")
    return importlib.import_module("tests.fakes.jobs_client").FileBackedFakeClient(path)
