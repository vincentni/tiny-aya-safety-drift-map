"""Checks a plan must pass before it is submitted; each is pure over one snapshot."""

import logging
from collections.abc import Callable, Sequence

from driftmap.core.records import Plan, Refusal, Snapshot
from driftmap.core.schemas import schema_versions
from driftmap.jobs.kinds import KINDS

log = logging.getLogger(__name__)

Guard = Callable[[Snapshot, Plan], Refusal | None]

# gated repos by the Settings.sources field a kind reads
GATED = {"model": "CohereLabs/tiny-aya-global"}


def gate(s: Snapshot, p: Plan) -> Refusal | None:
    # a repo the snapshot did not check counts as closed
    closed = [GATED[source] for source in KINDS[p.kind].reads if source in GATED and s.gates.get(GATED[source]) != "ok"]
    if closed:
        urls = ", ".join(f"https://huggingface.co/{repo}" for repo in closed)
        return Refusal(guard="gate", reason=f"access not granted for {urls}")
    return None


def code_pinned(s: Snapshot, p: Plan) -> Refusal | None:
    if not s.code_on_origin:
        reason = f"{p.code_sha} is not on origin, or the tree is dirty, or the fetch failed"
        return Refusal(guard="code_pinned", reason=reason)
    return None


def smoke(s: Snapshot, p: Plan) -> Refusal | None:
    if p.mode != "full" or p.override_smoke:
        return None
    current = schema_versions()
    for record in s.smoke_records:
        same = (
            (record.script, record.image, record.runner) == (p.script, p.image, p.runner)
            and sorted(record.dependencies) == sorted(p.dependencies)
            and record.schema_versions == current
        )
        # the quantized column needs a passed BOS check
        bos = record.bos_check is True if p.kind == "quant_eval" else record.bos_check is not False
        if same and bos and record.status == "ok":
            return None
    reason = f"no passing smoke run of {p.script} on {p.image} with the current pins and schemas"
    return Refusal(guard="smoke", reason=reason)


def per_run_ceiling(s: Snapshot, p: Plan) -> Refusal | None:
    cost = p.cost_at_timeout_micro_usd
    if cost > s.ceilings.per_run:
        reason = f"estimate {cost} micro-USD is over the per-run ceiling {s.ceilings.per_run}"
        return Refusal(guard="per_run_ceiling", reason=reason, estimate=cost, ceiling=s.ceilings.per_run)
    if p.mode != "smoke":
        return None
    if s.smoke_pending:
        reason = "another smoke job is still pending; smoke jobs run one after another"
        return Refusal(guard="per_run_ceiling", reason=reason, estimate=cost, ceiling=s.ceilings.smoke)
    total = s.smoke_spent_micro_usd + cost
    if total > s.ceilings.smoke:
        spent, ceiling = s.smoke_spent_micro_usd, s.ceilings.smoke
        reason = f"smoke total {total} micro-USD (spent {spent} + estimate {cost}) is over the smoke ceiling {ceiling}"
        return Refusal(guard="per_run_ceiling", reason=reason, estimate=cost, ceiling=s.ceilings.smoke)
    return None


def ledger_ceiling(s: Snapshot, p: Plan) -> Refusal | None:
    ceiling = s.ceilings.hard + s.ceilings.own_money if p.override_soft else s.ceilings.soft
    cost = p.cost_at_timeout_micro_usd
    if s.billing_used_micro_usd is None:
        log.warning("billing usage is unavailable; the ceiling is checked against the ledger alone")
        if s.ledger_total_micro_usd == 0 and not p.override_soft:
            reason = "no spend on record: billing is unavailable and the ledger is empty; check the billing page first"
            return Refusal(guard="ledger_ceiling", reason=reason, estimate=cost, ceiling=ceiling)
    spent = max(s.ledger_total_micro_usd, s.billing_used_micro_usd or 0)
    if spent + cost > ceiling:
        name = "hard ceiling plus own money" if p.override_soft else "soft ceiling"
        reason = f"total {spent + cost} micro-USD (spent {spent} + estimate {cost}) is over the {name} {ceiling}"
        return Refusal(guard="ledger_ceiling", reason=reason, estimate=cost, ceiling=ceiling)
    return None


def in_flight_gpu(s: Snapshot, p: Plan) -> Refusal | None:
    if KINDS[p.kind].gpu and s.in_flight_gpu:
        return Refusal(guard="in_flight_gpu", reason=f"a GPU job is already in flight: {', '.join(s.in_flight_gpu)}")
    return None


def timeout_lost(s: Snapshot, p: Plan) -> Refusal | None:
    if (p.kind, p.timeout_s) in s.lost_runs:
        reason = f"a {p.kind} job was already lost at a {p.timeout_s}s timeout; widen the timeout or split the job"
        return Refusal(guard="timeout_lost", reason=reason)
    return None


def oom_repeat(s: Snapshot, p: Plan) -> Refusal | None:
    if (p.kind, p.config_hash) in s.oom_runs:
        reason = f"a {p.kind} job ran out of memory with this config; lower its batch size first"
        return Refusal(guard="oom_repeat", reason=reason)
    return None


def drift_decision(s: Snapshot, p: Plan) -> Refusal | None:
    if p.kind in ("repair", "steer") and s.drift_flag is True and s.drift_decision is None:
        reason = "drift is below the threshold; record a decision with `driftmap decide drift`"
        return Refusal(guard="drift_decision", reason=reason)
    return None


GUARD_CHAIN: tuple[Guard, ...] = (
    gate,
    code_pinned,
    smoke,
    per_run_ceiling,
    ledger_ceiling,
    in_flight_gpu,
    timeout_lost,
    oom_repeat,
    drift_decision,
)


def run_guards(chain: Sequence[Guard], snapshot: Snapshot, plan: Plan) -> list[Refusal]:
    """Every refusal in chain order; no guard is skipped after a refusal."""
    return [refusal for guard in chain if (refusal := guard(snapshot, plan)) is not None]
