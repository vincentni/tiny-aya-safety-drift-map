import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from driftmap.core.records import Ceilings, Plan, SmokeRecord, Snapshot
from driftmap.core.schemas import schema_versions
from driftmap.core.settings import load_settings
from driftmap.jobs.guards import (
    GATED,
    GUARD_CHAIN,
    code_pinned,
    drift_decision,
    gate,
    in_flight_gpu,
    ledger_ceiling,
    oom_repeat,
    per_run_ceiling,
    run_guards,
    smoke,
    timeout_lost,
)
from driftmap.jobs.kinds import KINDS
from driftmap.jobs.ledger import Ledger, pending_line, terminal_line
from driftmap.jobs.plan import build_plan
from tests.fakes.settings import settings_toml

REPO_ROOT = Path(__file__).resolve().parents[3]
SHA = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 11, 6, 20, 0, tzinfo=UTC)
MODEL, BASE = "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"
NAMES = [
    "gate",
    "code_pinned",
    "smoke",
    "per_run_ceiling",
    "ledger_ceiling",
    "in_flight_gpu",
    "timeout_lost",
    "oom_repeat",
    "drift_decision",
]
GPU_KINDS = {"drift", "generate", "judge", "repair", "steer", "quant_eval"}
MODEL_KINDS = {"drift", "generate", "repair"}


@pytest.fixture
def config(tmp_path):
    """Loads a fresh Settings, with overrides, from a file written for the call."""
    return lambda **overrides: load_settings(settings_toml(tmp_path, **overrides))


def _plan(settings, kind="generate", mode="full", **kwargs):
    return build_plan(settings, kind, mode, None, SHA, now=NOW, repo_root=REPO_ROOT, **kwargs)


def _snapshot(ceilings, **changes):
    fields = {
        "billing_used_micro_usd": 0,
        "ledger_total_micro_usd": 0,
        "smoke_spent_micro_usd": 0,
        "smoke_pending": False,
        "prices_micro_usd_per_min": {"l4x1": 13_333, "cpu-upgrade": 500},
        "ceilings": ceilings,
        "in_flight_gpu": [],
        "smoke_records": [],
        "gates": {MODEL: "ok", BASE: "ok"},
        "lost_runs": [],
        "oom_runs": [],
        "drift_flag": False,
        "drift_decision": None,
        "code_on_origin": True,
    }
    return Snapshot(**{**fields, **changes})


def _record(plan, **changes):
    fields = {
        "script": plan.script,
        "image": plan.image,
        "runner": plan.runner,
        "dependencies": plan.dependencies,
        "schema_versions": schema_versions(),
        "status": "ok",
        "bos_check": None,
        "install_minutes": 3.5,
        "upload_mb_s": 11.0,
        "peak_vram_gb": 9.4,
        "peak_rss_gb": 6.1,
        "tokens_per_s": 410.0,
    }
    return SmokeRecord(**{**fields, **changes})


def _refused(snapshot, plan):
    return [refusal.guard for refusal in run_guards(GUARD_CHAIN, snapshot, plan)]


def test_chain_order_and_a_clean_launch(config):
    settings = config()
    assert [guard.__name__ for guard in GUARD_CHAIN] == NAMES
    plan = _plan(settings)
    clean = _snapshot(settings.ceilings, smoke_records=[_record(plan)])
    assert run_guards(GUARD_CHAIN, clean, plan) == []
    assert run_guards(GUARD_CHAIN, _snapshot(settings.ceilings), _plan(settings, mode="smoke")) == []


def test_model_gate_is_required_and_base_gate_is_not(config, monkeypatch):
    """A closed model gate refuses the kinds that load it; a closed base gate refuses none (TS-GRD-1)."""
    settings = config()
    for status in ("401", "403"):
        closed = _snapshot(settings.ceilings, gates={MODEL: status, BASE: "ok"})
        for kind in sorted(MODEL_KINDS):
            refusal = gate(closed, _plan(settings, kind))
            assert refusal.guard == "gate"
            assert "https://huggingface.co/CohereLabs/tiny-aya-global" in refusal.reason
            assert BASE not in refusal.reason
        for kind in ("judge", "steer", "quant", "quant_eval"):
            assert gate(closed, _plan(settings, kind)) is None

    base_closed = _snapshot(settings.ceilings, gates={MODEL: "ok", "CohereLabs/tiny-aya-base": "403"})
    for kind in sorted(KINDS):
        assert gate(base_closed, _plan(settings, kind)) is None
    assert "gate" not in _refused(base_closed, _plan(settings, "repair", "smoke"))

    # a gate the snapshot never checked is not open
    assert MODEL in gate(_snapshot(settings.ceilings, gates={BASE: "ok"}), _plan(settings, "repair")).reason

    assert GATED == {"model": MODEL}
    monkeypatch.setitem(GATED, "base", BASE)
    monkeypatch.setitem(KINDS, "repair", replace(KINDS["repair"], reads=(*KINDS["repair"].reads, "base")))
    refusal = gate(base_closed, _plan(settings, "repair"))
    assert "CohereLabs/tiny-aya-base" in refusal.reason
    assert MODEL not in refusal.reason
    assert gate(base_closed, _plan(settings, "generate")) is None


def test_code_missing_from_origin_is_refused(config):
    """Code that is not on origin, or an origin that could not be fetched, refuses (TS-GRD-2)."""
    settings = config()
    # the snapshot records a failed fetch as code_on_origin False
    refusal = code_pinned(_snapshot(settings.ceilings, code_on_origin=False), _plan(settings))
    assert refusal.guard == "code_pinned"
    assert SHA in refusal.reason
    assert code_pinned(_snapshot(settings.ceilings), _plan(settings)) is None
    unpinned = _snapshot(settings.ceilings, code_on_origin=False)
    assert _refused(unpinned, _plan(settings, mode="smoke")) == ["code_pinned"]


def test_full_run_needs_a_matching_smoke_record(config):
    """A full run needs a passing smoke record for the same script, image, runner and pins (TS-GRD-3)."""
    settings = config()
    plan = _plan(settings)

    def check(records, plan=plan):
        return smoke(_snapshot(settings.ceilings, smoke_records=records), plan)

    assert check([]).guard == "smoke"
    assert check([_record(plan)]) is None
    assert check([_record(plan, dependencies=list(reversed(plan.dependencies)))]) is None
    bumped = [pin.replace("safetune==0.1.10", "safetune==0.1.11") for pin in plan.dependencies]
    differing = [
        {"dependencies": bumped},
        {"script": "driftmap.scripts.judge"},
        {"image": "pytorch/pytorch:2.15.0-cuda12.6-cudnn9-runtime"},
        {"runner": "uv"},
        {"runner": "kaggle"},
        {"schema_versions": {**schema_versions(), "label": "driftmap.label/2"}},
        {"status": "partial"},
        {"bos_check": False},
    ]
    for change in differing:
        assert check([_record(plan, **change)]) is not None, change
    assert check([_record(plan, status="failed"), _record(plan)]) is None
    assert plan.script in check([]).reason

    overridden = _plan(settings, override_smoke=True)
    assert check([], overridden) is None
    assert "override-smoke" in pending_line(overridden, "6907a41b3c9e2f7d0a6b8c15", NOW, "settings").notes
    assert "override-smoke" not in pending_line(plan, "6907a41b3c9e2f7d0a6b8c15", NOW, "settings").notes

    quant_eval = _plan(settings, "quant_eval")
    assert check([_record(quant_eval, bos_check=False)], quant_eval) is not None
    assert check([_record(quant_eval, bos_check=None)], quant_eval) is not None
    assert check([_record(quant_eval, bos_check=True)], quant_eval) is None
    assert check([], _plan(settings, mode="smoke")) is None


def test_per_run_and_smoke_ceilings(config, tmp_path):
    """The per-run ceiling binds every plan; the smoke ceiling is a serial total (TS-GRD-4)."""
    ceilings = config().ceilings
    at_375 = _plan(config(timeouts={"generate": 375}))
    at_376 = _plan(config(timeouts={"generate": 376}))
    assert (at_375.cost_at_timeout_micro_usd, at_376.cost_at_timeout_micro_usd) == (4_999_875, 5_013_208)
    assert per_run_ceiling(_snapshot(ceilings), at_375) is None
    refusal = per_run_ceiling(_snapshot(ceilings), at_376)
    assert (refusal.guard, refusal.estimate, refusal.ceiling) == ("per_run_ceiling", 5_013_208, 5_000_000)
    assert "5013208" in refusal.reason
    assert "5000000" in refusal.reason
    on_cpu = [_plan(config(timeouts={"quant": minutes}), "quant") for minutes in (10_000, 10_001)]
    assert [plan.cost_at_timeout_micro_usd for plan in on_cpu] == [5_000_000, 5_000_500]
    assert per_run_ceiling(_snapshot(ceilings), on_cpu[0]) is None
    assert per_run_ceiling(_snapshot(ceilings), on_cpu[1]).ceiling == 5_000_000

    settings = config()
    probe = _plan(settings, mode="smoke")
    assert probe.cost_at_timeout_micro_usd == 399_990
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for index, minutes in enumerate((12, 9, 6, 15)):
        started = pending_line(probe, f"6906e0c47a2d9b1f3e5c8a0{index}", NOW, "settings")
        ledger.append(started)
        ledger.append(terminal_line(started, running_secs=minutes * 60, stage="COMPLETED", ts=NOW))
    ledger.append(pending_line(_plan(settings), "6907a41b3c9e2f7d0a6b8c15", NOW, "settings"))
    assert (ledger.smoke_spent_micro_usd(), ledger.smoke_pending()) == (559_986, False)
    ledger.append(pending_line(probe, "6906e0c47a2d9b1f3e5c8a04", NOW, "settings"))
    assert (ledger.smoke_spent_micro_usd(), ledger.smoke_pending()) == (959_976, True)

    # 959,976 + 399,990 = 1,359,966
    assert per_run_ceiling(_snapshot(ceilings, smoke_spent_micro_usd=959_976), probe) is None
    assert per_run_ceiling(_snapshot(ceilings, smoke_spent_micro_usd=1_600_010), probe) is None
    refusal = per_run_ceiling(_snapshot(ceilings, smoke_spent_micro_usd=1_600_011), probe)
    assert (refusal.estimate, refusal.ceiling) == (399_990, 2_000_000)
    assert "2000001" in refusal.reason
    assert "2000000" in refusal.reason

    serial = per_run_ceiling(_snapshot(ceilings, smoke_pending=True), probe)
    assert "pending" in serial.reason
    crowded = _snapshot(ceilings, smoke_pending=True, smoke_spent_micro_usd=1_999_999)
    assert per_run_ceiling(crowded, _plan(settings)) is None

    alone = per_run_ceiling(_snapshot(ceilings), _plan(config(timeouts={"smoke": 376}), mode="smoke"))
    assert (alone.estimate, alone.ceiling) == (5_013_208, 5_000_000)


@pytest.mark.parametrize(
    ("minutes", "total", "override_soft", "own_money", "ceiling"),
    [
        (1_200, 15_999_600, False, 0, None),
        (1_201, 16_012_933, False, 0, 16_000_000),
        (1_500, 19_999_500, True, 0, None),
        (1_501, 20_012_833, True, 0, 20_000_000),
        (1_650, 21_999_450, True, 2_000_000, None),
        (1_651, 22_012_783, True, 2_000_000, 22_000_000),
        (1_201, 16_012_933, False, 2_000_000, 16_000_000),
    ],
)
def test_ledger_ceiling_boundaries(config, caplog, minutes, total, override_soft, own_money, ceiling):
    """Ledger or billing plus the estimate is held to soft, or to hard plus own money under override (TS-GRD-5)."""
    settings = config(ceilings={"own_money": own_money})
    plan = _plan(settings, override_soft=override_soft)
    spent = 13_333 * (minutes - 90)
    assert spent + plan.cost_at_timeout_micro_usd == total

    def check(**changes):
        return ledger_ceiling(_snapshot(settings.ceilings, **changes), plan)

    by_ledger = check(ledger_total_micro_usd=spent, billing_used_micro_usd=1_000_000)
    by_billing = check(ledger_total_micro_usd=1_000_000, billing_used_micro_usd=spent)
    with caplog.at_level(logging.WARNING, logger="driftmap.jobs.guards"):
        assert caplog.records == []
        unbilled = check(ledger_total_micro_usd=spent, billing_used_micro_usd=None)
    assert "billing" in caplog.text
    for refusal in (by_ledger, by_billing, unbilled):
        if ceiling is None:
            assert refusal is None
        else:
            assert (refusal.guard, refusal.estimate, refusal.ceiling) == ("ledger_ceiling", 1_199_970, ceiling)
            assert str(total) in refusal.reason
            assert str(ceiling) in refusal.reason

    if ceiling is not None:
        exact = ceiling - plan.cost_at_timeout_micro_usd
        assert check(ledger_total_micro_usd=exact) is None
        assert check(ledger_total_micro_usd=exact + 1).ceiling == ceiling


def test_no_spend_record_is_refused_unless_overridden(config):
    settings = config()
    blind = _snapshot(settings.ceilings, billing_used_micro_usd=None, ledger_total_micro_usd=0)
    refusal = ledger_ceiling(blind, _plan(settings))
    assert (refusal.guard, refusal.estimate, refusal.ceiling) == ("ledger_ceiling", 1_199_970, 16_000_000)
    assert "billing" in refusal.reason
    assert "ledger" in refusal.reason
    assert _refused(blind, _plan(settings, mode="smoke")) == ["ledger_ceiling"]
    assert ledger_ceiling(blind, _plan(settings, override_soft=True)) is None
    recorded = _snapshot(settings.ceilings, billing_used_micro_usd=None, ledger_total_micro_usd=1)
    assert ledger_ceiling(recorded, _plan(settings)) is None
    assert ledger_ceiling(_snapshot(settings.ceilings, billing_used_micro_usd=0), _plan(settings)) is None


def test_one_gpu_job_at_a_time(config):
    """A GPU job in flight refuses GPU plans and lets the CPU quant job through (TS-GRD-6)."""
    settings = config()
    busy = _snapshot(settings.ceilings, in_flight_gpu=["6907a41b3c9e2f7d0a6b8c15"])
    for kind in sorted(GPU_KINDS):
        refusal = in_flight_gpu(busy, _plan(settings, kind))
        assert refusal.guard == "in_flight_gpu"
        assert "6907a41b3c9e2f7d0a6b8c15" in refusal.reason
    assert in_flight_gpu(busy, _plan(settings, "quant")) is None
    assert in_flight_gpu(busy, _plan(settings, "quant", "smoke")) is None
    assert in_flight_gpu(_snapshot(settings.ceilings), _plan(settings)) is None


def test_lost_run_is_not_repeated_at_the_same_timeout(config):
    """A kind lost at a timeout is refused at that timeout and allowed at a longer one (TS-GRD-7)."""
    settings = config()
    lost = _snapshot(settings.ceilings, lost_runs=[("generate", 5_400)])
    refusal = timeout_lost(lost, _plan(settings))
    assert refusal.guard == "timeout_lost"
    assert "5400" in refusal.reason
    assert timeout_lost(lost, _plan(config(timeouts={"generate": 120}))) is None
    assert timeout_lost(lost, _plan(settings, "steer")) is None


def test_oom_run_is_not_repeated_with_the_same_config(config):
    """A config that ran out of memory is refused until the next rung changes its hash (TS-GRD-8)."""
    settings = config()
    plan = _plan(settings)
    failed = _snapshot(settings.ceilings, oom_runs=[("generate", plan.config_hash)])
    assert oom_repeat(failed, plan).guard == "oom_repeat"
    next_rung = _plan(config(batch={"generate": 8}))
    assert next_rung.config_hash != plan.config_hash
    assert oom_repeat(failed, next_rung) is None
    assert oom_repeat(failed, _plan(settings, "judge")) is None


def test_flagged_drift_needs_a_decision_before_repair(config):
    """A drift flag without a decision refuses repair and steer only (TS-GRD-9)."""
    settings = config()
    flagged = _snapshot(settings.ceilings, drift_flag=True)
    assert drift_decision(flagged, _plan(settings, "repair")).guard == "drift_decision"
    assert drift_decision(flagged, _plan(settings, "steer")).guard == "drift_decision"
    assert drift_decision(flagged, _plan(settings, "generate")) is None
    decided = _snapshot(settings.ceilings, drift_flag=True, drift_decision="proceed")
    assert drift_decision(decided, _plan(settings, "repair")) is None
    for flag in (False, None):
        assert drift_decision(_snapshot(settings.ceilings, drift_flag=flag), _plan(settings, "repair")) is None


HASHES = ("ab" * 32, "cd" * 32)
PINS = (["safetune==0.1.10"], ["safetune==0.1.11"])
KIND = st.sampled_from(sorted([*GPU_KINDS, "quant"]))
MONEY = st.sampled_from(
    [0, 399_990, 1_000_000, 1_000_001, 2_000_000, 4_999_875, 5_000_000, 5_000_001, 8_000_000, 10_000_000, 11_000_000]
)
SPENT = MONEY | st.integers(0, 25_000_000)


@st.composite
def _plans(draw):
    kind = draw(KIND)
    return Plan(
        schema="driftmap.plan/1",
        kind=kind,
        mode=draw(st.sampled_from(["smoke", "full"])),
        runner="pip",
        script=f"driftmap.scripts.{kind}",
        flavor="cpu-upgrade" if kind == "quant" else "l4x1",
        image="pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime",
        timeout_s=draw(st.sampled_from([1_800, 5_400])),
        estimate_minutes=90,
        price_micro_usd_per_min=13_333,
        cost_at_timeout_micro_usd=draw(SPENT),
        dependencies=draw(st.sampled_from(PINS)),
        env={},
        secret_names=["HF_TOKEN"],
        code_sha=SHA,
        script_args=[],
        config_hash=draw(st.sampled_from(HASHES)),
        override_smoke=draw(st.booleans()),
        override_soft=draw(st.booleans()),
    )


@st.composite
def _records(draw):
    return SmokeRecord(
        script=f"driftmap.scripts.{draw(KIND)}",
        image="pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime",
        runner=draw(st.sampled_from(["pip", "kaggle"])),
        dependencies=draw(st.sampled_from(PINS)),
        schema_versions=draw(st.sampled_from([schema_versions(), {"run": "driftmap.run/0"}])),
        status=draw(st.sampled_from(["ok", "failed"])),
        bos_check=draw(st.sampled_from([None, True, False])),
        install_minutes=None,
        upload_mb_s=None,
        peak_vram_gb=None,
        peak_rss_gb=None,
        tokens_per_s=None,
    )


@st.composite
def _snapshots(draw):
    status = st.sampled_from(["ok", "ok", "401", "403"])
    return Snapshot(
        billing_used_micro_usd=draw(st.none() | SPENT),
        ledger_total_micro_usd=draw(SPENT),
        smoke_spent_micro_usd=draw(SPENT),
        smoke_pending=draw(st.booleans()),
        prices_micro_usd_per_min={"l4x1": 13_333},
        ceilings=draw(st.sampled_from([Ceilings(), Ceilings(own_money=2_000_000)])),
        in_flight_gpu=draw(st.lists(st.just("6907a41b3c9e2f7d0a6b8c15"), max_size=1)),
        smoke_records=draw(st.lists(_records(), max_size=3)),
        gates=draw(st.fixed_dictionaries({MODEL: status, BASE: status})),
        lost_runs=draw(st.lists(st.tuples(KIND, st.sampled_from([1_800, 5_400])), max_size=2)),
        oom_runs=draw(st.lists(st.tuples(KIND, st.sampled_from(HASHES)), max_size=2)),
        drift_flag=draw(st.sampled_from([None, True, False])),
        drift_decision=draw(st.sampled_from([None, "proceed", "extend"])),
        code_on_origin=draw(st.booleans()),
    )


def _smoked(s, p):
    for record in s.smoke_records:
        same = (record.script, record.image, record.runner, record.dependencies, record.schema_versions) == (
            p.script,
            p.image,
            p.runner,
            p.dependencies,
            schema_versions(),
        )
        bos = record.bos_check is True if p.kind == "quant_eval" else record.bos_check is not False
        if same and bos and record.status == "ok":
            return True
    return False


def _over_ledger(s, p):
    if s.billing_used_micro_usd is None and s.ledger_total_micro_usd == 0 and not p.override_soft:
        return True
    ceiling = s.ceilings.hard + s.ceilings.own_money if p.override_soft else s.ceilings.soft
    return max(s.ledger_total_micro_usd, s.billing_used_micro_usd or 0) + p.cost_at_timeout_micro_usd > ceiling


def _over_run(s, p):
    over_smoke = s.smoke_pending or s.smoke_spent_micro_usd + p.cost_at_timeout_micro_usd > s.ceilings.smoke
    return p.cost_at_timeout_micro_usd > s.ceilings.per_run or (p.mode == "smoke" and over_smoke)


HOLDS = {
    "gate": lambda s, p: p.kind in MODEL_KINDS and s.gates[MODEL] != "ok",
    "code_pinned": lambda s, p: not s.code_on_origin,
    "smoke": lambda s, p: p.mode == "full" and not p.override_smoke and not _smoked(s, p),
    "per_run_ceiling": _over_run,
    "ledger_ceiling": _over_ledger,
    "in_flight_gpu": lambda s, p: p.kind in GPU_KINDS and bool(s.in_flight_gpu),
    "timeout_lost": lambda s, p: (p.kind, p.timeout_s) in s.lost_runs,
    "oom_repeat": lambda s, p: (p.kind, p.config_hash) in s.oom_runs,
    "drift_decision": lambda s, p: p.kind in ("repair", "steer") and s.drift_flag is True and s.drift_decision is None,
}


@given(snapshot=_snapshots(), plan=_plans())
def test_refusals_are_the_holding_predicates_in_chain_order(snapshot, plan):
    """Every guard runs and the refusals are the holding predicates in chain order (TS-GRD-10)."""
    before = (snapshot.model_copy(deep=True), plan.model_copy(deep=True))
    ran = []

    def spy(guard):
        def watched(s, p):
            ran.append(guard.__name__)
            return guard(s, p)

        return watched

    refusals = run_guards(tuple(spy(guard) for guard in GUARD_CHAIN), snapshot, plan)
    assert ran == NAMES
    assert [refusal.guard for refusal in refusals] == [name for name in NAMES if HOLDS[name](snapshot, plan)]
    assert all(refusal.reason for refusal in refusals)
    assert (snapshot, plan) == before
    assert run_guards(GUARD_CHAIN, snapshot, plan) == refusals
