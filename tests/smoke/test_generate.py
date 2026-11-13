import re
from pathlib import Path

import pytest

from driftmap.core.settings import load_settings
from driftmap.jobs.snapshot import SMOKE_DIR, read_smoke_records

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_generate_smoke_run_recorded_its_measurements():
    """The ten-prompt generate smoke passed on the data build's prefix with throughput and memory figures (TS-SMK-3)."""
    settings = load_settings(ROOT / "driftmap.toml")
    records = [
        record
        for record in read_smoke_records(ROOT / SMOKE_DIR)
        if re.split(r"[./]", record.script.removesuffix(".py"))[-1] == "generate"
    ]
    assert records, "no generate smoke record under .run-local/smoke"
    record = records[-1]
    # a run on another prefix stops ungenerated: ok means they agree
    assert (record.status, record.bos_check, record.image) == ("ok", True, settings.images.pytorch)
    assert record.tokens_per_s and record.peak_vram_gb
