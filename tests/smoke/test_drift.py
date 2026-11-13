import re
from pathlib import Path

import pytest

from driftmap.core.settings import load_settings
from driftmap.jobs.snapshot import SMOKE_DIR, read_smoke_records

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_drift_smoke_run_recorded_its_measurements():
    """The 20-step drift smoke passed with install, upload, memory and throughput figures in range (TS-SMK-2)."""
    settings = load_settings(ROOT / "driftmap.toml")
    records = [
        record
        for record in read_smoke_records(ROOT / SMOKE_DIR)
        if re.split(r"[./]", record.script.removesuffix(".py"))[-1] == "drift"
    ]
    assert records, "no drift smoke record under .run-local/smoke"
    record = records[-1]
    assert (record.status, record.bos_check, record.image) == ("ok", True, settings.images.pytorch)
    assert record.install_minutes is not None and record.install_minutes <= 5
    assert record.upload_mb_s is not None and record.upload_mb_s >= 5
    assert record.peak_vram_gb and record.peak_rss_gb and record.tokens_per_s
