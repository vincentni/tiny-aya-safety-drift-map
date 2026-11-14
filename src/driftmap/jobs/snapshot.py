"""What a launch is checked against: spend, running jobs, prices, gates and local records."""

import logging
import os
import subprocess
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from huggingface_hub._jobs_api import TERMINAL_JOB_STAGES
from huggingface_hub.errors import HfHubHTTPError
from pydantic import ValidationError

from driftmap.core import ids
from driftmap.core.hub import GateRefused, classify
from driftmap.core.records import Decision, SmokeRecord, Snapshot, Tables, dump, load
from driftmap.core.schemas import SCHEMA_DECISION, DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.jobs.client import BillingUnavailable, head_status, minute_prices, used_micro_usd

log = logging.getLogger(__name__)

LOCAL = Path(".run-local")
SMOKE_DIR = LOCAL / "smoke"
IMAGE_PROBE_PATH = SMOKE_DIR / "image-probe.json"
DECISION_PATH = LOCAL / "results" / "decisions" / "drift.json"
TABLES_PATH = LOCAL / "results" / "tables" / "tables.json"
GIT_TIMEOUT_S = 60

Run = Callable[..., subprocess.CompletedProcess]


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Deps:
    """Outside world of the launcher verbs; tests swap any part."""

    client: Any
    hub: Any
    ledger: Any
    repo_root: Path
    now: Callable[[], datetime] = utcnow
    run: Run = subprocess.run
    head: Callable[[str], int] = head_status


def tables_path(preamble: str = "keep", fine_tune: str = "standard") -> Path:
    """The local tables of a prompt condition and fine-tune: tables.json, or a file named after both beside it."""
    tree = ids.tree("full", preamble).strip("/")
    parts = [TABLES_PATH.stem, *([tree] if tree else []), *([fine_tune] if fine_tune != "standard" else [])]
    return TABLES_PATH.with_name("-".join(parts) + TABLES_PATH.suffix)


def take_snapshot(
    settings: Settings,
    client: Any,
    hub: Any,
    ledger: Any,
    repo_root: str | Path,
    *,
    code_sha: str | None = None,
    run: Run = subprocess.run,
    named: tuple[str, str] | None = None,
) -> Snapshot:
    """Read everything the guards need; ``code_sha`` defaults to HEAD. Hub and git are read, never written.

    ``named`` is the fine-tune a launch names and the prompt condition it works under: its own tables are the
    ones read, and naming it stands for the drift decision.
    """
    root = Path(repo_root)
    billing = _billing_used(client)
    rows = client.list_jobs_hardware()
    cpu_only = {row.name for row in rows if row.accelerator is None}
    # unfiltered: a stage the library does not know yet must still count
    active = [job for job in client.list_jobs() if job.status.stage not in TERMINAL_JOB_STAGES]
    types = source_types(settings)
    decision = read_decision(root / DECISION_PATH)
    decided = decision.decision if decision else None
    table = _table(root / TABLES_PATH)
    if named:
        fine_tune, preamble = named
        table, decided = _table(root / tables_path(preamble, fine_tune)), f"the {fine_tune} fine-tune is named"
    return Snapshot(
        billing_used_micro_usd=billing,
        ledger_total_micro_usd=ledger.total_micro_usd(),
        smoke_spent_micro_usd=ledger.smoke_spent_micro_usd(),
        smoke_pending=ledger.smoke_pending(),
        prices_micro_usd_per_min=minute_prices(rows),
        ceilings=settings.ceilings,
        # a flavor missing from the price list is counted as a GPU
        in_flight_gpu=[job.id for job in active if job.flavor not in cpu_only],
        smoke_records=read_smoke_records(root / SMOKE_DIR),
        gates={repo: gate_state(hub, repo, settings.gated_repos, types.get(repo)) for repo in settings.gated_repos},
        lost_runs=ledger.lost_runs(),
        oom_runs=ledger.oom_runs(),
        table_on_file=table is not None,
        drift_flag=table.flags.get("drift_below_threshold") if table else None,
        drift_decision=decided,
        code_on_origin=code_pin_fault(root, settings.code.origin, settings.code.branch, code_sha, run=run) is None,
    )


def source_types(settings: Settings) -> dict[str, str]:
    """Hub repo type of each source repo."""
    sources = settings.sources
    models = dict.fromkeys((sources.model, sources.base), "model")
    return {**models, **dict.fromkeys((sources.harmful, sources.benign, sources.xstest), "dataset")}


def gate_state(hub: Any, repo: str, gated_repos: Collection[str], repo_type: str | None = None) -> str:
    """Return "ok", or the 401 or 403 a closed gate answered with; any other Hub error is raised."""
    try:
        hub.auth_check(repo, repo_type=repo_type)
    except GateRefused as refused:
        return str(refused.__cause__.response.status_code)
    except HfHubHTTPError as error:
        if classify(error, repo, gated_repos) != "gate":
            raise
        return str(error.response.status_code)
    return "ok"


def code_pin_fault(
    repo_root: str | Path, origin: str, branch: str, code_sha: str | None = None, *, run: Run = subprocess.run
) -> str | None:
    """Why ``code_sha`` (default HEAD) is not the clean HEAD on ``branch`` at the ``origin`` URL; None when it is."""

    def git(*args: str) -> subprocess.CompletedProcess:
        # a credential prompt would hang the launcher
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        argv = ["git", "-C", str(repo_root), *args]
        return run(argv, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, env=env)

    try:
        head = git("rev-parse", "HEAD")
        if head.returncode:
            return "not a git checkout"
        sha = code_sha or head.stdout.strip()
        if head.stdout.strip() != sha:
            return f"HEAD is not at {sha}"
        status = git("status", "--porcelain")
        if status.returncode or status.stdout.strip():
            return "the tree has uncommitted or untracked files"
        # jobs install from this URL, whatever the checkout's remotes are
        if git("fetch", "--quiet", origin, branch).returncode:
            return f"git fetch from {origin} failed"
        if git("merge-base", "--is-ancestor", sha, "FETCH_HEAD").returncode:
            return f"{sha} is not on {branch} of {origin}"
    except (OSError, subprocess.SubprocessError) as error:
        return f"git could not run ({type(error).__name__})"
    return None


def read_smoke_records(directory: str | Path) -> list[SmokeRecord]:
    """Every smoke record in the directory, by file name; a file that does not parse is an error."""
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == IMAGE_PROBE_PATH.name:
            continue
        try:
            records.append(load(SmokeRecord, path))
        except ValidationError as error:
            # the pydantic error carries input values, so it is not chained
            reason = error.errors()[0]["msg"]
            raise DriftmapError(ExitCode.UNEXPECTED, f"smoke record {path.name} does not parse: {reason}") from None
    return records


def write_decision(path: str | Path, decision: str, note: str, *, now: Callable[[], datetime] = utcnow) -> Decision:
    """Record the author's call on a below-threshold drift; ``decision`` is proceed or extend."""
    record = Decision(schema=SCHEMA_DECISION, decision=decision, note=note, ts=now())
    dump(record, Path(path))
    return record


def read_decision(path: str | Path) -> Decision | None:
    path = Path(path)
    return load(Decision, path) if path.exists() else None


def _billing_used(client: Any) -> int | None:
    try:
        return used_micro_usd(client.billing_usage())
    except (BillingUnavailable, httpx2.HTTPError) as error:
        log.warning("billing usage is unavailable (%s); spend is read from the ledger alone", type(error).__name__)
        return None


def _table(path: Path) -> Tables | None:
    """The local result table; None when it is absent or does not parse."""
    try:
        return load(Tables, path)
    except (FileNotFoundError, ValidationError):
        return None
