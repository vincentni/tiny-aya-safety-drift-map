"""Launch one paid job at a time, follow it to its end, chain jobs and reconcile the ledger."""

import logging
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import httpx2
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
from pydantic import ValidationError

from driftmap.core.hub import token
from driftmap.core.records import LedgerLine, Plan, Refusal, RunRecord, SmokeRecord, Snapshot, dump
from driftmap.core.redact import mask
from driftmap.core.schemas import DriftmapError, ExitCode, schema_versions
from driftmap.core.settings import ConfigError, Settings
from driftmap.jobs.client import MIN_POLL_S
from driftmap.jobs.guards import GUARD_CHAIN, run_guards
from driftmap.jobs.kinds import KINDS, State, prerequisites
from driftmap.jobs.ledger import Ledger, pending_line, terminal_line
from driftmap.jobs.lock import launch_lock
from driftmap.jobs.plan import SUBMIT_METHOD, build_plan, render, run_id, submit_kwargs, unit_price
from driftmap.jobs.readiness import CREDIT_MICRO_USD, READY_PATH, require_ready
from driftmap.jobs.snapshot import GIT_TIMEOUT_S, LOCAL, SMOKE_DIR, Deps, _billing_used, take_snapshot

log = logging.getLogger(__name__)

LEDGER_PATH = LOCAL / "results" / "ledger.jsonl"
LOGS_DIR = LOCAL / "results" / "logs"
LOG_TAIL = 50
# the job list can trail a submit that went through
LOOKUP_WAIT_S = 60
RECONCILE_TOLERANCE_MICRO_USD = 500_000
# the last line every job prints
TRAILER = re.compile(r"driftmap-job: finished status=(\w+) exit=(\d+)")
# the label every submit carries, as plan.submit_kwargs spells it
RUN_LABEL = "driftmap_run"
# ledger id of a run whose submit has not been recorded yet
HOLD = "unsubmitted:"
HOLD_NOTE = "hold"
# note of a terminal line written without the job's record or log
NO_EVIDENCE = "no-evidence"
RELEASED = "RELEASED"
# stage of a line whose job the platform no longer knows
GONE = "GONE"
# a younger hold stays: the job list can trail a submit
HOLD_GRACE = timedelta(minutes=15)
FAILED_EXITS = (ExitCode.INTEGRITY, ExitCode.COLUMN_FAILED)
# an install or an image pull fails well inside this
SETUP_MAX_S = 900
SETUP_FAILURE = re.compile(
    r"^(?:ERROR: (?:Could not find a version|No matching distribution|Cannot install|ResolutionImpossible"
    r"|Could not install packages|Could not build wheels|Failed building wheel|HTTP error)"
    r"|error: (?:Failed to (?:build|download|fetch|install|prepare|parse)|No solution found|Distribution not found)"
    r"|\s*× (?:No solution found|Failed to (?:build|download|fetch))"
    r"|.*(?:ErrImagePull|ImagePullBackOff|pull access denied|manifest unknown|No module named '?driftmap))",
    re.MULTILINE,
)
# a death the log explains is not a silent kill
CRASH = re.compile(r"^Traceback \(most recent call last\):|command not found", re.MULTILINE)
# files a smoke record's pins and config come from
PINNED_FILES = ("pyproject.toml", "uv.lock", "driftmap.toml", "revisions.json")
OOM_LADDER = {
    "drift": "set [batch] drift x drift_accum to the next rung: 4 x 4 -> 2 x 8 -> 1 x 16",
    "generate": "halve [batch] generate: 16 -> 8 -> 4",
    "repair": "halve [batch] screen: 8 -> 4, then pass --alphas one value per job",
}
NO_RUNG = "no batch rung is defined for this kind; lower its memory use"
COLUMN_FAILED = "the job could not produce its output and no input change fixes that; read the record notes first"
HOST_RAM_RUNG = "host memory is the suspect: run repair with --alphas one value per job"


class _OneRead(Ledger):
    """The ledger as one read found it, so every figure a launch checks agrees with the others."""

    def __init__(self, lines: list[LedgerLine]) -> None:
        self._lines = lines

    def lines(self) -> list[LedgerLine]:
        return self._lines


class Classified(NamedTuple):
    """Where a finished job falls: ledger fields, what to do next and the exit code of the wait."""

    ledger_fields: dict[str, Any]
    next_action: str
    code: ExitCode = ExitCode.UNEXPECTED


def classify(
    stage: Any,
    record: RunRecord | None,
    exit_code: int | None,
    log_tail: str,
    *,
    marker_present: bool,
    running_secs: int | None,
    timeout_s: int,
) -> Classified:
    """Sort a terminal job by stage, run record and log tail; anything unforeseen is ``unclassified``."""
    stage = _name(stage)
    status = record.status if record else None

    def row(action: str, code: ExitCode = ExitCode.UNEXPECTED, **flags: Any) -> Classified:
        return Classified({"record_status": status, "exit_code": exit_code, **flags}, action, code)

    if stage in ("CANCELED", "DELETED"):
        return row("relaunch allowed; a pushed record is kept", notes=["canceled"])
    if stage == "COMPLETED" and status == "ok":
        return row("none", ExitCode.SUCCESS)
    if stage == "ERROR" and status == "partial":
        return row("relaunch the same kind to resume; the same timeout is allowed", notes=["partial"])
    if stage == "ERROR" and status == "oom":
        rung = OOM_LADDER.get(record.kind, NO_RUNG)
        return row(f"out of memory: {rung}; the same config is refused", oom=True)
    if stage == "ERROR" and status == "failed" and exit_code in FAILED_EXITS:
        reason = ExitCode(exit_code).name.lower().replace("_", "-")
        if exit_code == ExitCode.COLUMN_FAILED:
            return row(COLUMN_FAILED, notes=[reason])
        return row("fix the input; relaunch allowed", notes=[reason])
    if stage == "ERROR" and record is None:
        silent = TRAILER.search(log_tail) is None
        late = _at_timeout(running_secs, timeout_s)
        if silent and late:
            return row("the same timeout is refused; widen it or split the job", ExitCode.LOST_RUN, lost=True)
        # a marker means the flush finished, so memory is not the suspect
        if marker_present:
            return row(
                "killed after the flush, before the record push: relaunch the same kind to resume", notes=["partial"]
            )
        # without a log nothing says how the job died
        if not log_tail.strip():
            return row("no log lines came back; read the job page and decide by hand", notes=["unclassified", "no-log"])
        if silent and (running_secs or 0) <= SETUP_MAX_S and SETUP_FAILURE.search(log_tail):
            return row(
                "the install or the image pull failed; fix the pins or the image, relaunch allowed", notes=["setup"]
            )
        if silent and running_secs and not CRASH.search(log_tail):
            return row(HOST_RAM_RUNG, oom=True, notes=["sigkill-suspected"])
    action = f"stage {stage}, record status {status}; decide by hand. Log tail:\n{mask(log_tail)}"
    return row(action, notes=["unclassified"])


def launch(
    settings: Settings,
    deps: Deps,
    kind: str,
    mode: str = "full",
    *,
    wait: bool = False,
    dry_run: bool = False,
    override_smoke: bool = False,
    override_soft: bool = False,
    limit: int | None = None,
    args: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
    named: tuple[str, str] | None = None,
) -> ExitCode:
    """Submit one job if every guard passes; a refusal is exit 4, and a dry run submits and records nothing.

    ``named`` is the fine-tune the job's arguments name and its prompt condition, as ``take_snapshot`` takes it.
    """
    report = require_ready(deps.repo_root / READY_PATH, settings.config_hash(), now=deps.now)
    settings = _on_offered_gpu(settings, report.gpu_flavor)
    # held from the first snapshot read until the hold is closed
    with launch_lock(settings.lock_path):
        # one read for every ledger figure: no wait slips in between
        ledger = _OneRead(deps.ledger.lines())
        last = ledger.read()
        unsettled = [job_id for job_id, line in last.items() if line.kind == kind and NO_EVIDENCE in line.notes]
        if unsettled:
            for job_id in unsettled:
                print(f"{kind} job {job_id} was closed without its record or log; classify it: driftmap wait {job_id}")
            print("nothing was submitted")
            return ExitCode.GUARD_REFUSAL
        # a job the ledger still holds may not be listed yet
        # a kind this code no longer launches counts as a GPU job
        cpu = {kind for kind, spec in KINDS.items() if not spec.gpu}
        held = [line.job_id for line in last.values() if line.pending and line.kind not in cpu]
        code_sha = _head(deps)
        snapshot = take_snapshot(
            settings, deps.client, deps.hub, ledger, deps.repo_root, code_sha=code_sha, run=deps.run, named=named
        )
        in_flight = list(dict.fromkeys([*snapshot.in_flight_gpu, *held]))
        snapshot = snapshot.model_copy(update={"in_flight_gpu": in_flight})
        prices = snapshot.prices_micro_usd_per_min
        plan = build_plan(
            settings,
            kind,
            mode,
            prices,
            code_sha,
            override_smoke=override_smoke,
            override_soft=override_soft,
            limit=limit,
            now=deps.now(),
            # the job refuses to run under another config
            args=["--config-hash", settings.config_hash(), *args],
            repo_root=deps.repo_root,
        )
        source = unit_price(settings, plan.flavor, prices)[1]
        refusals = run_guards(GUARD_CHAIN, snapshot, plan)
        if dry_run:
            print(render(settings, plan, source))
            print(_headroom(snapshot))
        if dry_run or refusals:
            print(_verdict_table(refusals))
        if refusals:
            print(f"refused by {len(refusals)} guard(s); nothing was submitted")
            if held:
                print(f"pending on the ledger: {', '.join(held)}; settle each with `driftmap wait <id>`")
            return ExitCode.GUARD_REFUSAL
        if dry_run:
            return ExitCode.SUCCESS
        job_id = _submit(settings, deps, plan, source, sleep)
    cost = plan.cost_at_timeout_micro_usd
    print(f"submitted {kind} ({mode}) as job {job_id}, run {run_id(plan)}; held at {cost} micro-USD until it ends")
    if wait:
        return wait_job(settings, deps, job_id, sleep=sleep)
    print(f"follow it with: driftmap wait {job_id}")
    return ExitCode.SUCCESS


def wait_job(settings: Settings, deps: Deps, job_id: str, *, sleep: Callable[[float], None] = time.sleep) -> ExitCode:
    """Follow a submitted job to its end, record what it cost and print the next action; never cancels."""
    if job_id.startswith(HOLD):
        # a launch that holds the lock still owns its hold
        with launch_lock(settings.lock_path):
            history = [line for line in deps.ledger.lines() if line.job_id == job_id]
            if not history:
                raise DriftmapError(ExitCode.LOST_RUN, f"no ledger line for {job_id}")
            return _resolve_hold(deps, history, sleep)
    lines = [line for line in deps.ledger.lines() if line.job_id == job_id]
    # the line a job was submitted under prices every later one
    origin = lines[0] if lines else _adopt_listed(settings, deps, job_id)
    pending = lines[-1].pending if lines else True
    budget = origin.timeout_s + settings.scheduling_allowance_s
    try:
        job = deps.client.wait_for_job(job_id, timeout=budget, poll_interval=MIN_POLL_S)
    except TimeoutError:
        print(f"job {job_id} is not terminal after {budget} s; its ledger line stays pending")
        print(f"cancel it with: hf jobs cancel {job_id}")
        return ExitCode.LOST_RUN
    except httpx2.HTTPError as error:
        if pending and getattr(getattr(error, "response", None), "status_code", None) == 404:
            return _close_gone(deps, origin)
        return _lost_contact(job_id, error)
    running_secs = job.durations.running_secs if job.durations else None
    try:
        record_bytes, marker = _job_files(deps.hub, f"{settings.owner}/{settings.repos.results}", origin.run_id)
        fetched = list(deps.client.fetch_job_logs(job_id=job_id, tail=LOG_TAIL))
    except (httpx2.HTTPError, LocalEntryNotFoundError) as error:
        # the job is over: without evidence its line still closes
        fields = {"record_status": None, "exit_code": None, "notes": ["unclassified", NO_EVIDENCE]}
        if _name(job.status.stage) == "ERROR" and _at_timeout(running_secs, origin.timeout_s):
            # a death at the timeout stays lost until evidence is read
            fields["lost"] = True
        action = f"its record or log could not be fetched ({_what(error)}); classify it with: driftmap wait {job_id}"
        _settle(deps, origin, job, running_secs, Classified(fields, action, ExitCode.LOST_RUN))
        return ExitCode.LOST_RUN
    # patterns are sought in the raw tail; storage and print are masked
    tail = "\n".join(line.rstrip("\r\n") for line in fetched)
    saved = deps.repo_root / LOGS_DIR / f"{job_id}.log"
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(f"{mask(tail)}\n", encoding="utf-8")
    trailer = TRAILER.search(tail)
    exit_code = int(trailer[2]) if trailer else None
    try:
        record = RunRecord.model_validate_json(record_bytes) if record_bytes else None
        outcome = classify(
            job.status.stage,
            record,
            exit_code,
            tail,
            marker_present=marker,
            running_secs=running_secs,
            timeout_s=origin.timeout_s,
        )
    except ValidationError:
        fields = {"record_status": "unreadable", "exit_code": exit_code, "notes": ["unclassified"]}
        outcome = Classified(fields, "the run record does not parse; decide by hand")
    _settle(deps, origin, job, running_secs, outcome)
    print(f"log tail saved in {saved}")
    if origin.mode == "smoke" and outcome.code is ExitCode.SUCCESS:
        _write_smoke_record(settings, deps, job, record)
    return outcome.code


def chain(
    settings: Settings,
    deps: Deps,
    kinds: Sequence[str],
    mode: str = "full",
    *,
    state: Callable[[str], State],
    args: Mapping[str, Sequence[str]] | None = None,
) -> ExitCode:
    """Launch and follow each kind in turn; ``state(kind)`` is read before each, and the first job not ok stops it."""
    for kind in kinds:
        missing = prerequisites(kind, state(kind))
        if missing:
            print(f"{kind} cannot run yet; missing: {'; '.join(missing)}")
            return ExitCode.GUARD_REFUSAL
        code = launch(settings, deps, kind, mode, wait=True, args=(args or {}).get(kind, ()))
        if code is not ExitCode.SUCCESS:
            print(f"chain stopped at {kind}")
            return code
    return ExitCode.SUCCESS


def reconcile(deps: Deps) -> ExitCode:
    """Compare billing with the ledger; exit 1 on a pending job, no billing, or billing over the ledger by $0.50."""
    used = _billing_used(deps.client)
    # one read, after the billing call, for check and total
    lines = deps.ledger.read()
    waiting = [job_id for job_id, line in lines.items() if line.pending]
    if waiting:
        # a pending line at timeout cost would hide unrecorded spend
        print(f"pending on the ledger: {', '.join(waiting)}; run `driftmap wait <id>` for each, then reconcile")
        return ExitCode.UNEXPECTED
    if used is None:
        print("billing usage is unavailable; nothing was reconciled")
        return ExitCode.UNEXPECTED
    total = sum(line.cost_micro_usd for line in lines.values())
    drift = used - total
    print(f"billing {used} micro-USD, ledger {total} micro-USD, drift {drift} micro-USD")
    if drift > RECONCILE_TOLERANCE_MICRO_USD:
        print(f"billing is over the ledger by more than {RECONCILE_TOLERANCE_MICRO_USD}; find the unrecorded job")
        return ExitCode.UNEXPECTED
    finished = [line for job_id, line in lines.items() if not job_id.startswith(HOLD)]
    if finished:
        last = max(finished, key=lambda line: line.ts)
        # restates the newest finished job at its cost, with billing
        restated = terminal_line(
            last,
            running_secs=last.minutes * 60,
            stage=last.stage,
            ts=deps.now(),
            record_status=last.record_status,
            exit_code=last.exit_code,
            lost=last.lost,
            oom=last.oom,
            notes=["reconcile"],
            billing_used_after_micro_usd=used,
        )
        deps.ledger.append(restated)
    return ExitCode.SUCCESS


def _on_offered_gpu(settings: Settings, flavor: str | None) -> Settings:
    """Settings on the GPU flavor readiness found on offer."""
    if flavor in (None, settings.flavors.gpu):
        return settings
    if flavor not in settings.prices_micro_usd_per_min:
        raise ConfigError(f"prices_micro_usd_per_min: no price for {flavor}, the GPU flavor readiness recorded")
    return settings.model_copy(update={"flavors": settings.flavors.model_copy(update={"gpu": flavor})})


def _head(deps: Deps) -> str:
    argv = ["git", "-C", str(deps.repo_root), "rev-parse", "HEAD"]
    done = deps.run(argv, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    if done.returncode:
        raise DriftmapError(ExitCode.GUARD_REFUSAL, "not a git checkout; a job installs the code from its HEAD commit")
    return done.stdout.strip()


def _pins_committed(deps: Deps) -> bool:
    argv = ["git", "-C", str(deps.repo_root), "status", "--porcelain", "--", *PINNED_FILES]
    done = deps.run(argv, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    return not done.returncode and not done.stdout.strip()


def _headroom(snapshot: Snapshot) -> str:
    used = snapshot.billing_used_micro_usd
    spent = max(snapshot.ledger_total_micro_usd, used or 0)
    rows = {
        "ledger": f"{snapshot.ledger_total_micro_usd} micro-USD",
        "credit": "unknown, billing usage is unavailable"
        if used is None
        else f"{CREDIT_MICRO_USD - used} micro-USD left",
        "headroom": f"{snapshot.ceilings.soft - spent} micro-USD under the soft ceiling before this job",
    }
    return "\n".join(f"{name:<10}{value}" for name, value in rows.items())


def _verdict_table(refusals: Sequence[Refusal]) -> str:
    """Every guard in chain order with its verdict."""
    refused = {refusal.guard: refusal for refusal in refusals}
    rows = [("guard", "verdict", "estimate", "ceiling", "reason")]
    for guard in GUARD_CHAIN:
        hit = refused.get(guard.__name__)
        rows.append(
            (guard.__name__, "refused", _amount(hit.estimate), _amount(hit.ceiling), hit.reason)
            if hit
            else (guard.__name__, "pass", "-", "-", "")
        )
    return "\n".join(f"{a:<17}{b:<9}{c:<10}{d:<10}{e}".rstrip() for a, b, c, d, e in rows)


def _amount(micro_usd: int | None) -> str:
    return "-" if micro_usd is None else str(micro_usd)


def _what(error: BaseException) -> str:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return f"{type(error).__name__} {status}" if status else type(error).__name__


def _at_timeout(running_secs: int | None, timeout_s: int) -> bool:
    """Ran for at least 95 % of its timeout."""
    return running_secs is not None and running_secs * 100 >= 95 * timeout_s


def _name(stage: Any) -> str:
    return str(getattr(stage, "value", stage))


def _lost_contact(job_id: str, error: BaseException) -> ExitCode:
    print(f"job {job_id} could not be followed ({_what(error)}); its ledger line is unchanged")
    print(f"try again with: driftmap wait {job_id}")
    return ExitCode.LOST_RUN


def _settle(deps: Deps, origin: LedgerLine, job: Any, running_secs: int | None, outcome: Classified) -> None:
    """Append the terminal line of a finished job and print what it cost and what to do next."""
    line = terminal_line(
        origin,
        running_secs=running_secs,
        stage=job.status.stage,
        ts=deps.now(),
        billing_used_after_micro_usd=_billing_used(deps.client),
        **outcome.ledger_fields,
    )
    deps.ledger.append(line)
    print(f"job {line.job_id} ended {_name(line.stage)}: {line.minutes} min, {line.cost_micro_usd} micro-USD")
    print(f"next: {outcome.next_action}")


def _adopt_listed(settings: Settings, deps: Deps, job_id: str) -> LedgerLine:
    """Record a listed job whose run label names a hold on this ledger; exit 5 when none does."""
    with launch_lock(settings.lock_path):
        try:
            job = next((job for job in deps.client.list_jobs() if job.id == job_id), None)
        except Exception as error:
            message = (
                f"no ledger line for job {job_id}, and the job list could not be read ({_what(error)}); "
                f"try again with: driftmap wait {job_id}"
            )
            raise DriftmapError(ExitCode.LOST_RUN, message) from error
        run = (job.labels or {}).get(RUN_LABEL) if job else None
        history = [line for line in deps.ledger.lines() if run and line.job_id == f"{HOLD}{run}"]
        if not history:
            message = (
                f"no ledger line for job {job_id} and no hold for its run label; it was not launched from this checkout"
            )
            raise DriftmapError(ExitCode.LOST_RUN, message)
        line = adopted_line(history[0], job_id, deps.now())
        deps.ledger.append(line)
        if history[-1].pending:
            _close_hold(deps, history[0], f"job={job_id}")
        print(f"job {job_id} carries the label of run {run}; it is recorded")
        return line


def _close_gone(deps: Deps, pending: LedgerLine) -> ExitCode:
    """Close the line of a job the platform answers 404 for, at the cost it was held at."""
    line = terminal_line(pending, running_secs=None, stage=GONE, ts=deps.now(), notes=["job-not-found"])
    deps.ledger.append(line)
    print(
        f"the platform does not know job {line.job_id}; its line is closed at the {line.cost_micro_usd} micro-USD "
        f"it was held at, so `driftmap ledger reconcile` shows any gap. If `hf jobs ps -a` lists it after all, "
        f"run `driftmap wait {line.job_id}` again"
    )
    return ExitCode.LOST_RUN


def hold_line(plan: Plan, ts: datetime, source: str) -> LedgerLine:
    """The pending line of a run that is about to be submitted and has no job id yet."""
    line = pending_line(plan, f"{HOLD}{run_id(plan)}", ts, source)
    return LedgerLine.model_validate(line.model_dump(by_alias=True) | {"notes": [*line.notes, HOLD_NOTE]})


def adopted_line(hold: LedgerLine, job_id: str, ts: datetime) -> LedgerLine:
    """The line ``pending_line`` gives for the hold's run once its job id is known."""
    notes = [note for note in hold.notes if note != HOLD_NOTE]
    return LedgerLine.model_validate(hold.model_dump(by_alias=True) | {"job_id": job_id, "notes": notes, "ts": ts})


def _submit(settings: Settings, deps: Deps, plan: Plan, source: str, sleep: Callable[[float], None]) -> str:
    """Submit once under a hold line and record the job; a failed call is never repeated."""
    call = submit_kwargs(settings, plan, token())
    hold = hold_line(plan, deps.now(), source)
    # written before the paid call, so a killed launcher leaves the cost
    deps.ledger.append(hold)
    try:
        job_id = getattr(deps.client, SUBMIT_METHOD[plan.runner])(**call).id
    except BaseException as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        # a 4xx other than a request timeout created nothing
        refused = status is not None and 400 <= status < 500 and status != 408
        resolve = f"driftmap wait {hold.job_id}"
        print(f"submit of run {hold.run_id} failed ({_what(error)}); checking the job list. If cut short: {resolve}")
        job, read = _listed(deps.client, call["labels"])
        if job is None and not refused and isinstance(error, Exception):
            sleep(LOOKUP_WAIT_S)
            job, read = _listed(deps.client, call["labels"])
        if job is not None:
            _record_pending(deps, plan, job.id, source, hold)
            print(f"job {job.id} carries the label of run {hold.run_id}; it is recorded")
            if isinstance(error, Exception):
                return job.id
            raise
        if refused and read:
            _close_hold(deps, hold, "not-submitted")
            message = f"submit refused ({_what(error)}) and no job carries the run label; no job was created"
            raise DriftmapError(ExitCode.UNEXPECTED, message) from error
        message = (
            f"submit failed ({_what(error)}) and no job labelled {hold.run_id} was found; its hold stays on the ledger "
            f"at {hold.cost_micro_usd} micro-USD and the in-flight GPU guard refuses the next launch. "
            f"Look for the run id in `hf jobs ps -a`, then run `{resolve}`: it records the job if it is listed, "
            f"and releases the hold only once it is {HOLD_GRACE.seconds // 60} minutes old and still unlisted"
        )
        if isinstance(error, Exception):
            raise DriftmapError(ExitCode.UNEXPECTED, message) from error
        print(message)
        raise
    _record_pending(deps, plan, job_id, source, hold)
    return job_id


def _labelled(client: Any, labels: Mapping[str, str]) -> Any | None:
    """The job carrying these labels, at any stage, or None."""
    return next((job for job in client.list_jobs() if labels.items() <= (job.labels or {}).items()), None)


def _listed(client: Any, labels: Mapping[str, str]) -> tuple[Any | None, bool]:
    """The labelled job, and whether the job list could be read at all."""
    try:
        return _labelled(client, labels), True
    except Exception as error:
        print(f"the job list could not be read ({_what(error)})")
        return None, False


def _record_pending(deps: Deps, plan: Plan, job_id: str, source: str, hold: LedgerLine) -> None:
    try:
        deps.ledger.append(pending_line(plan, job_id, deps.now(), source))
    except BaseException as error:
        # the job runs either way; a second submit would pay twice
        message = (
            f"job {job_id} was submitted but its ledger line could not be written ({_what(error)}); "
            f"the hold {hold.job_id} stands for its cost. Record it with: driftmap wait {hold.job_id}, "
            f"or cancel it with: hf jobs cancel {job_id}"
        )
        if isinstance(error, Exception):
            raise DriftmapError(ExitCode.UNEXPECTED, message) from error
        print(message)
        raise
    _close_hold(deps, hold, f"job={job_id}")


def _close_hold(deps: Deps, hold: LedgerLine, note: str) -> None:
    """Release a hold at zero cost; one that cannot be closed stays at its full cost."""
    try:
        deps.ledger.append(terminal_line(hold, running_secs=0, stage=RELEASED, ts=deps.now(), notes=[note]))
    except Exception as error:
        print(
            f"the hold {hold.job_id} could not be closed ({_what(error)}); close it with: driftmap wait {hold.job_id}"
        )


def _resolve_hold(deps: Deps, history: Sequence[LedgerLine], sleep: Callable[[float], None]) -> ExitCode:
    """Turn a hold into its job's pending line; release it only when old and twice unlisted."""
    first, hold = history[0], history[-1]
    if not hold.pending and hold.notes[-1].startswith("job="):
        print(f"{hold.job_id} is already closed")
        return ExitCode.SUCCESS
    labels = {RUN_LABEL: hold.run_id}
    old = hold.pending and deps.now() - first.ts >= HOLD_GRACE
    job, read = _listed(deps.client, labels)
    if job is None and read and old:
        # one empty list proves nothing: it can trail a submit
        sleep(LOOKUP_WAIT_S)
        job, read = _listed(deps.client, labels)
    again = f"driftmap wait {hold.job_id}"
    later = "If a job for this run shows up later in `hf jobs ps -a`, record it with: driftmap wait <job id>"
    if job is not None:
        if job.id not in deps.ledger.read():
            deps.ledger.append(adopted_line(first, job.id, deps.now()))
        if hold.pending:
            _close_hold(deps, first, f"job={job.id}")
        print(f"job {job.id} runs as {hold.run_id}; it is recorded. Follow it with: driftmap wait {job.id}")
        return ExitCode.SUCCESS
    if not read:
        print(f"{hold.job_id} is left as it is; try again with: {again}")
        return ExitCode.LOST_RUN
    if not hold.pending:
        print(f"{hold.job_id} is already closed, released at zero cost; no job is labelled {hold.run_id}. {later}")
        return ExitCode.SUCCESS
    if not old:
        due = first.ts + HOLD_GRACE
        print(
            f"no job is labelled {hold.run_id} yet, and the job list can trail a submit; it stays on the ledger. "
            f"Look for the run id in `hf jobs ps -a`, and try again after {due:%H:%M} UTC with: {again}"
        )
        return ExitCode.LOST_RUN
    _close_hold(deps, first, "no-job-found")
    print(f"no job is labelled {hold.run_id} on two reads {LOOKUP_WAIT_S} s apart; its hold is released at zero cost")
    print(later)
    return ExitCode.SUCCESS


def _job_paths(run_id: str) -> tuple[str, str]:
    """Run record and resume marker of a run, as its job writes them to the results repo."""
    return f"runs/{run_id}.json", f"partial/{run_id}.json"


def _job_files(hub: Any, results: str, run_id: str) -> tuple[bytes | None, bool]:
    """The run record, if pushed, and whether a resume marker stands in for it."""
    record_path, marker_path = _job_paths(run_id)
    pinned = f"{results}@{hub.dataset_info(results).sha}"
    record = _fetch(hub, f"{pinned}#{record_path}")
    return record, record is None and _fetch(hub, f"{pinned}#{marker_path}") is not None


def _fetch(hub: Any, reference: str) -> bytes | None:
    with tempfile.TemporaryDirectory() as scratch:
        try:
            return Path(hub.hf_hub_download(reference, scratch, repo_type="dataset")).read_bytes()
        except RemoteEntryNotFoundError:
            return None


def _write_smoke_record(settings: Settings, deps: Deps, job: Any, record: RunRecord) -> None:
    # pins and script come from this checkout: it must match the job
    try:
        head = _head(deps)
    except DriftmapError:
        head = None
    if head != record.code_sha or settings.config_hash() != record.config_hash or not _pins_committed(deps):
        print(
            f"no smoke record written: the checkout, its pins or the config is not what job {job.id} ran. Put them "
            f"back (`git checkout {record.code_sha}`, no uncommitted change to {', '.join(PINNED_FILES)}) "
            f"and run `driftmap wait {job.id}` again to write it"
        )
        return
    # resolved like the full plan the smoke guard compares this record with
    plan = build_plan(settings, record.kind, "full", None, record.code_sha, repo_root=deps.repo_root)
    smoke = SmokeRecord(
        script=plan.script,
        image=job.docker_image or plan.image,
        runner=plan.runner,
        dependencies=plan.dependencies,
        schema_versions=schema_versions(),
        status=record.status,
        bos_check=record.bos_check,
        install_minutes=record.install_minutes,
        upload_mb_s=record.upload_mb_s,
        peak_vram_gb=record.peak_vram_gb,
        peak_rss_gb=record.peak_rss_gb,
        tokens_per_s=record.tokens_per_s,
    )
    dump(smoke, deps.repo_root / SMOKE_DIR / f"{record.kind}.json")
