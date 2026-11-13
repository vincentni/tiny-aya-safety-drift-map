from pathlib import Path

import pytest

from driftmap.core.records import load
from driftmap.core.settings import load_settings
from driftmap.jobs.readiness import READY_PATH, ImageProbe, ReadyReport
from driftmap.jobs.snapshot import IMAGE_PROBE_PATH

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[2]


def test_the_image_probe_recorded_the_expected_facts():
    """The cpu-basic probe saw Python 3.12, torch 2.14.1, PEP 668 and no git, curl or uv (TS-SMK-1)."""
    settings = load_settings(ROOT / "driftmap.toml")
    probe = load(ImageProbe, ROOT / IMAGE_PROBE_PATH)
    assert probe.image == settings.images.pytorch
    assert probe.python.startswith("3.12.") and probe.torch.split("+")[0] == "2.14.1"
    assert probe.pep668 and not {"git", "curl", "uv"} & set(probe.tools)


def test_the_first_ready_run_recorded_billing_usage():
    """ready.json holds the usage figure the coupon is charged against (TS-SMK-9)."""
    billing = load(ReadyReport, ROOT / READY_PATH).billing
    assert billing["hasAccess"] is True
    assert type(billing["usedMicroUsd"]) is int and billing["usedMicroUsd"] >= 0
