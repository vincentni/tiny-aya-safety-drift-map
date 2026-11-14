"""Job entry points and the settings a job runs under."""

import argparse
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx2

from driftmap.core import ids
from driftmap.core.clients import REQUEST_TIMEOUT
from driftmap.core.records import JobKind
from driftmap.core.redact import configure_logging
from driftmap.core.retry import policy
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings, load_settings
from driftmap.jobkit import main as jobkit
from driftmap.jobkit.deadline import Clock
from driftmap.jobkit.main import Context

logger = logging.getLogger(__name__)

RAW_HOST = "https://raw.githubusercontent.com"
CONFIG_FILES = ("driftmap.toml", "revisions.json")
_ARCHIVE = re.compile(r"https://github\.com/([^/]+/[^/]+)/archive/([0-9a-f]{40})\.tar\.gz")
_ORIGIN = re.compile(r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?/?")


def _direct_url() -> str | None:
    return metadata.distribution("driftmap").read_text("direct_url.json")


def installed_repo(code_sha: str, origin: str | None = None) -> str:
    """``owner/name`` of the GitHub archive this package was installed from at ``code_sha``, else of ``origin``."""
    try:
        recorded = _direct_url()
        match = _ARCHIVE.fullmatch(json.loads(recorded).get("url", "")) if recorded else None
    except (metadata.PackageNotFoundError, ValueError, AttributeError):
        # no install record, or a torn one: the origin argument decides
        match = None
    if match is None:
        named = _ORIGIN.fullmatch(origin or "")
        if named is None:
            raise DriftmapError(
                ExitCode.INTEGRITY,
                "driftmap was not installed from a GitHub archive; pass --config or --code-origin",
            )
        return named.group(1)
    repo, sha = match.groups()
    if sha != code_sha:
        raise DriftmapError(ExitCode.INTEGRITY, f"installed from commit {sha[:12]}, launched as {code_sha[:12]}")
    return repo


def fetch_settings(
    repo: str,
    sha: str,
    *,
    transport: httpx2.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Settings:
    """Settings from driftmap.toml and revisions.json at a commit of the code repo. Raises DriftmapError."""
    with (
        tempfile.TemporaryDirectory(prefix="driftmap-config-") as scratch,
        httpx2.Client(transport=transport, follow_redirects=True, timeout=REQUEST_TIMEOUT) as session,
    ):
        for name in CONFIG_FILES:
            try:
                response = policy(sleep)(session.get, f"{RAW_HOST}/{repo}/{sha}/{name}")
                response.raise_for_status()
            except httpx2.HTTPError as error:
                raise DriftmapError(ExitCode.INTEGRITY, f"cannot read {name} at {sha[:12]}: {error}") from error
            (Path(scratch) / name).write_bytes(response.content)
        return load_settings(Path(scratch) / CONFIG_FILES[0])


def _options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="driftmap.toml to use in place of the installed commit's")
    parser.add_argument("--code-origin", help="GitHub URL of the code repo, when the install recorded none")
    parser.add_argument("--config-hash", help="config hash the launcher planned with; required without --config")


def preamble_option(parser: argparse.ArgumentParser) -> None:
    """``--preamble keep|strip``: the chat template as served, or with no safety preamble."""
    parser.add_argument(
        "--preamble",
        choices=ids.PREAMBLES,
        default="keep",
        help="strip: no safety preamble; the template's safety paragraph is left out, files go under nopreamble/",
    )


def job_settings(argv: Sequence[str] | None = None, *, transport: httpx2.BaseTransport | None = None) -> Settings:
    """Settings for this job: the ``--config`` file, else the files at ``--code-sha`` of the installed repo."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--code-sha", required=True)
    _options(parser)
    known, _ = parser.parse_known_args(argv)
    if known.config:
        settings = load_settings(known.config)
    else:
        if not known.config_hash:
            raise DriftmapError(ExitCode.INTEGRITY, "--config-hash is required when the settings are fetched")
        repo = installed_repo(known.code_sha, known.code_origin)
        settings = fetch_settings(repo, known.code_sha, transport=transport)
    loaded = settings.config_hash()
    if known.config_hash and known.config_hash != loaded:
        raise DriftmapError(
            ExitCode.INTEGRITY, f"the job loaded config {loaded[:12]}, the launcher planned {known.config_hash[:12]}"
        )
    return settings


def run_job(
    body: Callable[[Context], object],
    argv: Sequence[str] | None = None,
    *,
    kind: JobKind,
    arguments: Callable[[argparse.ArgumentParser], object] | None = None,
    client: Any | None = None,
    clock: Clock | None = None,
    transport: httpx2.BaseTransport | None = None,
) -> int:
    """Run a job body under the settings of the installed code; returns the exit code."""
    # AuditKIT keeps every response under this directory
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="driftmap-cache-")
    if clock is None:
        configure_logging()
    try:
        settings = job_settings(argv, transport=transport)
    except DriftmapError as error:
        logger.error("no settings for this job: %s", error)
        # a config fault is exit 2 at the launcher; in a job it is an integrity fault
        code = int(ExitCode.INTEGRITY)
        print(jobkit.TRAILER.format(status="failed", code=code), flush=True)
        return code

    def options(parser: argparse.ArgumentParser) -> None:
        _options(parser)
        if arguments is not None:
            arguments(parser)

    return jobkit.run(body, argv, kind=kind, settings=settings, arguments=options, client=client, clock=clock)
