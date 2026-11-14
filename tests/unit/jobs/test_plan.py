import inspect
import os
import re
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
from huggingface_hub import HfApi
from huggingface_hub._jobs_api import DEFAULT_UV_IMAGE, _create_job_spec

from driftmap.core.records import SmokeRecord
from driftmap.core.schemas import schema_versions
from driftmap.core.settings import ConfigError, load_settings
from driftmap.jobs.kinds import KINDS
from driftmap.jobs.plan import (
    SUBMIT_METHOD,
    build_plan,
    fallback_marks,
    pins,
    render,
    run_id,
    submit_kwargs,
    unit_price,
)
from tests.fakes.jobs_client import FakeJobsClient
from tests.fakes.settings import settings_toml

REPO_ROOT = Path(__file__).resolve().parents[3]
SHA = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 11, 13, 9, 30, 41, tzinfo=UTC)
TOKEN = "hf_" + "t" * 34
PIN = f"driftmap[jobs] @ https://github.com/vincentni/tiny-aya-safety-drift-map/archive/{SHA}.tar.gz"
IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"


_build = partial(build_plan, repo_root=REPO_ROOT)


@pytest.fixture
def settings(tmp_path):
    return load_settings(settings_toml(tmp_path))


@pytest.fixture
def live():
    return {
        row.name: row.unit_cost_micro_usd
        for row in FakeJobsClient().list_jobs_hardware()
    }


def test_every_kind_is_priced_at_its_declared_timeout(settings, live):
    """Timeouts and integer costs are exact for every kind and for smoke (TS-PLN-1)."""
    full = {kind: _build(settings, kind, "full", live, SHA) for kind in KINDS}
    assert {kind: plan.timeout_s for kind, plan in full.items()} == {
        "drift": 10_800,
        "generate": 5_400,
        "repair": 7_200,
        "steer": 5_400,
        "quant": 3_600,
        "quant_eval": 7_200,
    }
    assert {kind: plan.cost_at_timeout_micro_usd for kind, plan in full.items()} == {
        "drift": 2_399_940,
        "generate": 1_199_970,
        "repair": 1_599_960,
        "steer": 1_199_970,
        "quant": 30_000,
        "quant_eval": 1_599_960,
    }
    assert {plan.flavor for kind, plan in full.items() if kind != "quant"} == {"l4x1"}
    assert (full["quant"].flavor, full["quant"].price_micro_usd_per_min) == (
        "cpu-upgrade",
        500,
    )

    smoke = {kind: _build(settings, kind, "smoke", live, SHA) for kind in KINDS}
    assert {plan.timeout_s for plan in smoke.values()} == {1_800}
    assert smoke["generate"].cost_at_timeout_micro_usd == 399_990
    assert smoke["quant"].cost_at_timeout_micro_usd == 15_000

    for plan in (*full.values(), *smoke.values()):
        assert type(plan.cost_at_timeout_micro_usd) is int
        assert plan.estimate_minutes * 60 == plan.timeout_s
        assert plan.config_hash == settings.config_hash()
        assert plan.dependencies == pins(REPO_ROOT)

    assert unit_price(settings, "l4x1", {"l4x1": 14_000}) == (14_000, "hardware")
    assert unit_price(settings, "l4x1", {"l4x1": 13_333}) == (13_333, "hardware")
    assert unit_price(settings, "l4x1", {"a10g-small": 16_667}) == (13_333, "settings")
    assert unit_price(settings, "l4x1", None) == (13_333, "settings")
    for low in (13_332, 0, -1, 14_000.0, None):
        assert unit_price(settings, "l4x1", {"l4x1": low}) == (13_333, "settings")
    assert (
        _build(
            settings, "drift", "full", {"l4x1": 12_000}, SHA
        ).cost_at_timeout_micro_usd
        == 2_399_940
    )
    assert (
        _build(
            settings, "drift", "full", {"l4x1": 14_000}, SHA
        ).cost_at_timeout_micro_usd
        == 2_520_000
    )


def test_each_runner_has_its_submit_shape(tmp_path, settings, live):
    """The pip runner submits run_job and the uv runner run_uv_job (TS-PLN-3)."""
    run = "generate-full-20261113T093041Z"
    args = f"--run-id {run} --timeout-s 5400 --code-sha {SHA} --submitted-at 2026-11-13T09:30:41Z"
    plan = _build(settings, "generate", "full", live, SHA, now=NOW)
    assert (plan.runner, plan.script, plan.image) == (
        "pip",
        "driftmap.scripts.generate",
        IMAGE,
    )
    assert submit_kwargs(settings, plan, TOKEN) == {
        "image": IMAGE,
        "command": [
            "bash",
            "-lc",
            f"pip install -q '{PIN}' && python -m driftmap.scripts.generate {args}",
        ],
        "env": {"PIP_BREAK_SYSTEM_PACKAGES": "1"},
        "secrets": {"HF_TOKEN": TOKEN},
        "flavor": "l4x1",
        "timeout": "5400s",
        "attempts": 1,
        "labels": {"driftmap_run": "generate-full-20261113T093041Z"},
    }

    on_uv = load_settings(settings_toml(tmp_path, runner="uv"))
    uv = _build(on_uv, "generate", "full", live, SHA, now=NOW)
    assert (uv.runner, uv.script, uv.image) == (
        "uv",
        "jobs/generate.py",
        DEFAULT_UV_IMAGE,
    )
    assert submit_kwargs(on_uv, uv, TOKEN) == {
        "script": "jobs/generate.py",
        "script_args": args.split(),
        "dependencies": [PIN],
        "image": DEFAULT_UV_IMAGE,
        "env": {},
        "secrets": {"HF_TOKEN": TOKEN},
        "flavor": "l4x1",
        "timeout": "5400s",
        "attempts": 1,
        "labels": {"driftmap_run": "generate-full-20261113T093041Z"},
    }

    assert SUBMIT_METHOD == {"pip": "run_job", "uv": "run_uv_job"}
    for config, built in ((settings, plan), (on_uv, uv)):
        method = getattr(HfApi, SUBMIT_METHOD[built.runner])
        inspect.signature(method).bind(None, **submit_kwargs(config, built, TOKEN))

    quoted = _build(
        settings,
        "quant_eval",
        "full",
        live,
        SHA,
        now=NOW,
        args=["--gguf", "o/r@abc#model q4.gguf"],
    )
    assert submit_kwargs(settings, quoted, TOKEN)["command"][2].endswith(
        "--gguf 'o/r@abc#model q4.gguf'"
    )


def test_script_args_name_the_run_and_dumps_hold_no_secret(settings, live):
    """Script args carry the run id, timeout, code and submit time; no dump holds a secret value (TS-PLN-4)."""
    plan = _build(settings, "steer", "full", live, SHA, now=NOW)
    assert plan.script_args == [
        "--run-id",
        "steer-full-20261113T093041Z",
        "--timeout-s",
        "5400",
        "--code-sha",
        SHA,
        "--submitted-at",
        "2026-11-13T09:30:41Z",
    ]
    assert run_id(plan) == "steer-full-20261113T093041Z"

    smoke = _build(
        settings,
        "steer",
        "smoke",
        live,
        SHA,
        now=NOW,
        limit=10,
        args=["--layer", "12"],
    )
    assert smoke.script_args[2:4] == ["--timeout-s", "1800"]
    assert smoke.script_args[8:] == [
        "--limit",
        "10",
        "--force-flush-after-s",
        "300",
        "--layer",
        "12",
    ]
    assert (
        "--force-flush-after-s"
        not in _build(settings, "steer", "full", live, SHA, limit=10).script_args
    )

    unclocked = _build(settings, "steer", "full", live, SHA).script_args
    submitted = datetime.fromisoformat(unclocked[unclocked.index("--submitted-at") + 1])
    assert 0 <= (datetime.now(UTC) - submitted).total_seconds() < 60

    secret = os.environ["HF_TOKEN"]
    text = render(settings, plan, "settings")
    assert plan.secret_names == ["HF_TOKEN"]
    assert "HF_TOKEN=***" in text
    for dumped in (plan.model_dump_json(), str(plan.model_dump()), text):
        assert secret not in dumped
        assert TOKEN not in dumped


def test_drift_timeout_and_attempts_are_explicit(settings, live):
    """Drift is submitted with its own timeout and a single attempt (TS-DRIFT-7)."""
    kwargs = submit_kwargs(
        settings, _build(settings, "drift", "full", live, SHA), TOKEN
    )
    assert kwargs["timeout"] == "10800s"
    assert kwargs["attempts"] == 1
    spec = _create_job_spec(**kwargs)
    assert (spec["timeoutSeconds"], spec["attempts"], spec["flavor"]) == (
        10_800,
        1,
        "l4x1",
    )


def test_gpu_kinds_without_a_kaggle_smoke_are_marked(settings, live):
    """Only a passing kaggle smoke record clears a GPU kind's fallback mark (TS-LCH-18)."""

    def record(kind, runner="pip", status="ok"):
        plan = _build(settings, kind, "smoke", live, SHA)
        return SmokeRecord(
            script=plan.script,
            image=plan.image,
            runner=runner,
            dependencies=plan.dependencies,
            schema_versions=schema_versions(),
            status=status,
            bos_check=None,
            install_minutes=3.5,
            upload_mb_s=11.0,
            peak_vram_gb=9.4,
            peak_rss_gb=6.1,
            tokens_per_s=410.0,
        )

    gpu = ["drift", "generate", "quant_eval", "repair", "steer"]
    on_jobs = [record(kind) for kind in KINDS]
    assert fallback_marks([]) == dict.fromkeys(gpu, "no verified fallback")
    assert fallback_marks(on_jobs) == dict.fromkeys(gpu, "no verified fallback")
    assert fallback_marks(
        [*on_jobs, record("generate", "kaggle", "failed")]
    ) == fallback_marks(on_jobs)

    marks = fallback_marks(
        [*on_jobs, record("generate", "kaggle"), record("quant", "kaggle")]
    )
    assert sorted(marks) == ["drift", "quant_eval", "repair", "steer"]
    uv_script = record("steer", "kaggle").model_copy(update={"script": "jobs/steer.py"})
    assert "steer" not in fallback_marks([uv_script])


def test_render_prints_the_resolved_job(settings, live):
    plan = _build(settings, "drift", "full", live, SHA, now=NOW, override_soft=True)
    text = render(settings, plan, "hardware")
    for expected in (
        "drift (full)",
        "run_job",
        IMAGE,
        "l4x1",
        "10800s (180 min)",
        "2399940 micro-USD ($2.40) at 13333 micro-USD/min (hardware)",
        f"pip install -q '{PIN}' && python -m driftmap.scripts.drift --run-id drift-full-20261113T093041Z",
        "PIP_BREAK_SYSTEM_PACKAGES=1",
        "driftmap_run=drift-full-20261113T093041Z",
        "override-soft",
    ):
        assert expected in text
    assert "override-smoke" not in text
    assert "$0.02" in render(
        settings, _build(settings, "quant", "smoke", live, SHA), "settings"
    )


def test_pins_come_from_the_checkout(tmp_path, settings, live):
    checkout = pins(REPO_ROOT)
    assert checkout == sorted(checkout)
    assert all(re.fullmatch(r"[a-z0-9-]+==[0-9][\w.]*", pin) for pin in checkout)
    assert any(pin.startswith("safetune==") for pin in checkout)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
dependencies = ["Hugging_Face.Hub == 2.4.0 ; python_version >= '3.12'", "auditkit[transformers]==1.1.3"]

[project.optional-dependencies]
jobs = ["safetune==0.1.11"]
dev = ["pytest==9.2.0"]
""",
        encoding="utf-8",
    )
    bumped = [
        "auditkit[transformers]==1.1.3",
        "hugging-face-hub==2.4.0",
        "safetune==0.1.11",
    ]
    assert pins(tmp_path) == bumped
    assert (
        build_plan(
            settings, "steer", "smoke", live, SHA, repo_root=tmp_path
        ).dependencies
        == bumped
    )

    for broken in ("[project]\ndependencies = []\n", "[project\n"):
        (tmp_path / "pyproject.toml").write_text(broken, encoding="utf-8")
        with pytest.raises(ConfigError, match="pyproject.toml"):
            pins(tmp_path)
    # the default root is the working directory, which holds no checkout here
    with pytest.raises(ConfigError, match="pyproject.toml"):
        build_plan(settings, "steer", "smoke", live, SHA)


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"kind": "train"}, "kind"),
        ({"mode": "pilot"}, "mode"),
        ({"code_sha": "main"}, "code_sha"),
        ({"limit": 0}, "limit"),
    ],
)
def test_bad_request_is_a_config_error(settings, live, kwargs, named):
    request = {"kind": "generate", "mode": "smoke", "code_sha": SHA, **kwargs}
    with pytest.raises(ConfigError, match=named) as caught:
        _build(
            settings,
            request.pop("kind"),
            request.pop("mode"),
            live,
            request.pop("code_sha"),
            **request,
        )
    assert caught.value.exit_code == 2
