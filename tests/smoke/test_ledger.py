from pathlib import Path

import pytest

from driftmap.core.settings import load_settings
from driftmap.jobs.launcher import LEDGER_PATH, RECONCILE_TOLERANCE_MICRO_USD
from driftmap.jobs.ledger import Ledger

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_ledger_was_reconciled_after_the_last_job_and_is_under_its_ceilings():
    """The newest line is a reconcile within $0.50 of billing; spend is under both ceilings (TS-SMK-12)."""
    ceilings = load_settings(ROOT / "driftmap.toml").ceilings
    ledger = Ledger(ROOT / LEDGER_PATH)
    lines = ledger.lines()
    assert lines and "reconcile" in lines[-1].notes, "run `driftmap ledger reconcile` after the last job"
    assert lines[-1].billing_used_after_micro_usd - ledger.total_micro_usd() <= RECONCILE_TOLERANCE_MICRO_USD
    assert ledger.smoke_spent_micro_usd() <= ceilings.smoke
    assert ledger.total_micro_usd() <= ceilings.soft
