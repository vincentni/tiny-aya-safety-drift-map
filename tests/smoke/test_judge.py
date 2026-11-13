import re
from pathlib import Path

import pytest

from driftmap.core.settings import load_settings
from driftmap.jobs.snapshot import SMOKE_DIR, read_smoke_records

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_judge_smoke_run_parsed_its_first_outputs():
    """The small-judge smoke, launched with --inspect-first 20, ended ok: all 20 real outputs parsed (TS-SMK-4)."""
    settings = load_settings(ROOT / "driftmap.toml")
    records = [
        record
        for record in read_smoke_records(ROOT / SMOKE_DIR)
        if re.split(r"[./]", record.script.removesuffix(".py"))[-1] == "judge"
    ]
    assert records, "no judge smoke record under .run-local/smoke"
    record = records[-1]
    # a parse miss among the inspected outputs ends the job with exit 14, never ok
    assert (record.status, record.image) == ("ok", settings.images.pytorch)
    assert record.peak_vram_gb and record.tokens_per_s
