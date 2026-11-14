import re
from pathlib import Path

import pytest

from driftmap.core.settings import load_settings
from driftmap.jobs.snapshot import SMOKE_DIR, read_smoke_records

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_repair_smoke_run_recorded_its_memory_peaks():
    """The one-alpha repair smoke ended ok on the data build's prefix with host and GPU memory peaks (TS-SMK-5)."""
    settings = load_settings(ROOT / "driftmap.toml")
    records = [
        record
        for record in read_smoke_records(ROOT / SMOKE_DIR)
        if re.split(r"[./]", record.script.removesuffix(".py"))[-1] == "repair"
    ]
    assert records, "no repair smoke record under .run-local/smoke"
    record = records[-1]
    # a screen on another prefix, or weights other than the drift run's, stops before any alpha: ok means they agree
    assert (record.status, record.bos_check, record.image) == ("ok", True, settings.images.pytorch)
    assert record.peak_rss_gb and record.peak_vram_gb
