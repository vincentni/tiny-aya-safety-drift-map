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


def test_the_quant_eval_smoke_run_passed_the_bos_check_on_the_gpu():
    """The served GGUF counted every smoke prompt as the tokenizer does and reported its throughput (TS-SMK-7)."""
    settings = load_settings(ROOT / "driftmap.toml")
    record = last_record("quant_eval")
    # a server left on the CPU exits 13, never ok
    assert (record.status, record.bos_check, record.image) == ("ok", True, settings.images.pytorch)
    assert record.tokens_per_s and record.upload_mb_s
