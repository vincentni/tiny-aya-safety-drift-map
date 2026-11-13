"""Append-only cost ledger; the last line of a job is its cost."""

import fcntl
import os
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from driftmap.core.records import LedgerLine, Plan
from driftmap.core.schemas import SCHEMA_LEDGER, DriftmapError, ExitCode
from driftmap.jobs.plan import overrides, run_id



class LedgerCorrupt(DriftmapError):
    """A ledger line is cut off or does not parse; the message names the file and line."""

    exit_code_default = ExitCode.UNEXPECTED


class PendingAfterTerminal(DriftmapError):
    """A pending line would replace the recorded cost of a finished job."""

    exit_code_default = ExitCode.UNEXPECTED


PRE_RUN = ("SCHEDULING",)
UNKNOWN_RUNTIME = "running-time-unknown"


def _minutes(seconds: int) -> int:
    return -(-seconds // 60)


def pending_line(plan: Plan, job_id: str, ts: datetime, price_source: str) -> LedgerLine:
    """The line written at submit: the job costs its full timeout until a terminal line follows."""
    return LedgerLine(
        schema=SCHEMA_LEDGER,
        job_id=job_id,
        run_id=run_id(plan),
        kind=plan.kind,
        flavor=plan.flavor,
        mode=plan.mode,
        price_micro_usd_per_min=plan.price_micro_usd_per_min,
        minutes=_minutes(plan.timeout_s),
        cost_micro_usd=plan.cost_at_timeout_micro_usd,
        stage="SCHEDULING",
        record_status=None,
        exit_code=None,
        billing_used_after_micro_usd=None,
        timeout_s=plan.timeout_s,
        config_hash=plan.config_hash,
        pending=True,
        lost=False,
        oom=False,
        notes=[f"price={price_source}", *overrides(plan)],
        ts=ts,
    )


def terminal_line(
    pending: LedgerLine,
    *,
    running_secs: int | None,
    stage: str,
    ts: datetime,
    record_status: str | None = None,
    exit_code: int | None = None,
    lost: bool = False,
    oom: bool = False,
    notes: tuple[str, ...] | list[str] = (),
    billing_used_after_micro_usd: int | None = None,
) -> LedgerLine:
    """The line for a finished job, billed per started minute; ValueError for negative seconds."""
    if running_secs is not None and running_secs < 0:
        raise ValueError(f"running_secs: expected zero or more, got {running_secs}")
    if running_secs is None and stage not in PRE_RUN:
        # it ran, the platform gave no duration: keep the timeout cost
        minutes, cost, notes = pending.minutes, pending.cost_micro_usd, (*notes, UNKNOWN_RUNTIME)
    else:
        minutes = _minutes(running_secs or 0)
        cost = minutes * pending.price_micro_usd_per_min
    return LedgerLine(
        schema=SCHEMA_LEDGER,
        job_id=pending.job_id,
        run_id=pending.run_id,
        kind=pending.kind,
        flavor=pending.flavor,
        mode=pending.mode,
        price_micro_usd_per_min=pending.price_micro_usd_per_min,
        minutes=minutes,
        cost_micro_usd=cost,
        stage=stage,
        record_status=record_status,
        exit_code=exit_code,
        billing_used_after_micro_usd=billing_used_after_micro_usd,
        timeout_s=pending.timeout_s,
        config_hash=pending.config_hash,
        pending=False,
        lost=lost,
        oom=oom,
        notes=list(dict.fromkeys([*pending.notes, *notes])),
        ts=ts,
    )


class Ledger:
    """JSONL file of LedgerLine; every reader raises LedgerCorrupt on a bad line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _parse(self, raw: bytes) -> list[LedgerLine]:
        rows = raw.split(b"\n")
        # empty when the last write ended with its newline
        tail = rows.pop()
        parsed = []
        for number, row in enumerate(rows, 1):
            try:
                parsed.append(LedgerLine.model_validate_json(row))
            except ValidationError as error:
                first = error.errors()[0]
                field = "".join(f"{part}: " for part in first["loc"])
                # the pydantic error carries input values, so it is not chained
                reason = first["msg"].split(" at line ")[0]
                raise LedgerCorrupt(f"{self.path}: line {number}: {field}{reason}") from None
        if tail:
            raise LedgerCorrupt(f"{self.path}: line {len(rows) + 1}: cut off before its newline")
        return parsed

    def lines(self) -> list[LedgerLine]:
        """Every line in file order, read under a shared lock; a missing file is an empty ledger."""
        try:
            with self.path.open("rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_SH)
                raw = handle.read()
        except FileNotFoundError:
            return []
        return self._parse(raw)

    def read(self) -> dict[str, LedgerLine]:
        """The last line of each job, keyed by job id."""
        return {line.job_id: line for line in self.lines()}

    def append(self, line: LedgerLine) -> None:
        """Add one line; raises PendingAfterTerminal for a pending line on a finished job."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as handle:
            # one exclusive lock covers the check and the write
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            last = {row.job_id: row for row in self._parse(handle.read())}.get(line.job_id)
            if line.pending and last is not None and not last.pending:
                raise PendingAfterTerminal(f"{self.path}: job {line.job_id} has a terminal line; pending line refused")
            handle.write(line.model_dump_json().encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def total_micro_usd(self) -> int:
        """Cost of every job, pending ones at their timeout."""
        return sum(line.cost_micro_usd for line in self.read().values())

    def smoke_spent_micro_usd(self) -> int:
        return sum(line.cost_micro_usd for line in self.read().values() if line.mode == "smoke")

    def smoke_pending(self) -> bool:
        return any(line.pending for line in self.read().values() if line.mode == "smoke")

    def lost_runs(self) -> list[tuple[str, int]]:
        """(kind, timeout_s) of every job with a line marked lost; a later line does not clear it."""
        return list(dict.fromkeys((line.kind, line.timeout_s) for line in self.lines() if line.lost))

    def oom_runs(self) -> list[tuple[str, str]]:
        """(kind, config_hash) of every job with a line marked oom; any of its lines may carry the hash."""
        lines = self.lines()
        hashes = {line.job_id: line.config_hash for line in lines if line.config_hash}
        marked = ((line.kind, hashes.get(line.job_id)) for line in lines if line.oom)
        return list(dict.fromkeys(pair for pair in marked if pair[1]))
