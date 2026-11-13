import re
from pathlib import Path

import pytest

from driftmap.core.settings import load_settings
from driftmap.jobs.snapshot import SMOKE_DIR, read_smoke_records

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def last_record(kind):
    records = [
        record
        for record in read_smoke_records(ROOT / SMOKE_DIR)
        if re.split(r"[./]", record.script.removesuffix(".py"))[-1] == kind
    ]
    assert records, f"no {kind} smoke record under .run-local/smoke"
    return records[-1]


def test_the_quant_smoke_run_converted_and_uploaded_a_gguf():
    """The CPU smoke ended ok: the pinned converter and quantizer ran on the image (TS-SMK-6)."""
    settings = load_settings(ROOT / "driftmap.toml")
    record = last_record("quant")
    # a release hash mismatch exits 12, never ok
    assert (record.status, record.image) == ("ok", settings.images.pytorch)
    assert record.upload_mb_s and record.peak_rss_gb
