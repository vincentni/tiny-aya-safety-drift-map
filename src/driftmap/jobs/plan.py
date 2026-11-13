"""Resolve a job into a Plan and into the arguments of its submit call."""

import re
import shlex
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub._jobs_api import DEFAULT_UV_IMAGE
from pydantic import ValidationError

from driftmap.core.records import Plan, SmokeRecord
from driftmap.core.schemas import SCHEMA_PLAN
from driftmap.core.settings import ConfigError, Settings
from driftmap.jobs.kinds import KINDS

SUBMIT_METHOD = {"pip": "run_job", "uv": "run_uv_job"}
SECRET_NAMES = ["HF_TOKEN"]
MASK = "***"
NO_FALLBACK = "no verified fallback"
SMOKE_FLUSH_AFTER_S = 300


def unit_price(settings: Settings, flavor: str, prices: dict[str, int] | None) -> tuple[int, str]:
    """Micro-USD per minute and its source: "hardware" for a live price, else "settings"."""
    configured = settings.prices_micro_usd_per_min[flavor]
    live = (prices or {}).get(flavor)
    # a live price under the configured one would loosen every ceiling
    if type(live) is int and live >= configured:
        return live, "hardware"
    return configured, "settings"


def _pin(requirement: str) -> str:
    spec = requirement.split(";")[0].replace(" ", "")
    name = re.match(r"[A-Za-z0-9._-]*", spec).group()
    return re.sub(r"[-_.]+", "-", name).lower() + spec[len(name) :]


def pins(repo_root: str | Path) -> list[str]:
    """Sorted pins of the project and its jobs extra from the checkout's pyproject.toml. Raises ConfigError."""
    path = Path(repo_root) / "pyproject.toml"
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        requirements = [*project["dependencies"], *project["optional-dependencies"]["jobs"]]
    except (OSError, ValueError, KeyError) as error:
        raise ConfigError(f"{path}: cannot read the pins ({type(error).__name__}: {error})") from None
    return sorted(_pin(requirement) for requirement in requirements)


def build_plan(
    settings: Settings,
    kind: str,
    mode: str,
    prices: dict[str, int] | None,
    code_sha: str,
    *,
    override_smoke: bool = False,
    override_soft: bool = False,
    limit: int | None = None,
    now: datetime | None = None,
    args: Sequence[str] = (),
    repo_root: str | Path = ".",
) -> Plan:
    """Plan one job priced at its full timeout; `now` (aware) stamps the run id. Raises ConfigError."""
    if kind not in KINDS:
        raise ConfigError(f"kind: unknown job kind {kind!r}")
    if limit is not None and limit < 1:
        raise ConfigError(f"limit: expected a positive count, got {limit}")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    flavor = settings.flavors.gpu if KINDS[kind].gpu else settings.flavors.cpu
    price, _ = unit_price(settings, flavor, prices)
    timeout_s = settings.timeouts.seconds("smoke" if mode == "smoke" else kind)
    minutes = -(-timeout_s // 60)
    on_uv = settings.runner == "uv"
    script_args = [
        "--run-id",
        f"{kind}-{mode}-{now:%Y%m%dT%H%M%SZ}",
        "--timeout-s",
        str(timeout_s),
        "--code-sha",
        code_sha,
        "--submitted-at",
        f"{now:%Y-%m-%dT%H:%M:%SZ}",
    ]
    if limit is not None:
        script_args += ["--limit", str(limit)]
    if mode == "smoke":
        script_args += ["--force-flush-after-s", str(SMOKE_FLUSH_AFTER_S)]
    try:
        return Plan(
            schema=SCHEMA_PLAN,
            kind=kind,
            mode=mode,
            runner=settings.runner,
            script=f"jobs/{kind}.py" if on_uv else KINDS[kind].module,
            flavor=flavor,
            image=DEFAULT_UV_IMAGE if on_uv else getattr(settings.images, KINDS[kind].image),
            timeout_s=timeout_s,
            estimate_minutes=minutes,
            price_micro_usd_per_min=price,
            cost_at_timeout_micro_usd=price * minutes,
            dependencies=pins(repo_root),
            env={} if on_uv else {"PIP_BREAK_SYSTEM_PACKAGES": "1"},
            secret_names=SECRET_NAMES,
            code_sha=code_sha,
            script_args=[*script_args, *args],
            config_hash=settings.config_hash(),
            override_smoke=override_smoke,
            override_soft=override_soft,
        )
    except ValidationError as error:
        lines = [f"{'.'.join(map(str, e['loc']))}: {e['msg']} [{e['type']}]" for e in error.errors()]
        raise ConfigError("\n".join(lines)) from None


def run_id(plan: Plan) -> str:
    return plan.script_args[plan.script_args.index("--run-id") + 1]


def overrides(plan: Plan) -> list[str]:
    """Names of the overrides the plan was built with, as kept in ledger notes."""
    flags = (("override-smoke", plan.override_smoke), ("override-soft", plan.override_soft))
    return [name for name, used in flags if used]


def submit_kwargs(settings: Settings, plan: Plan, token: str) -> dict[str, Any]:
    """Keyword arguments for the client method SUBMIT_METHOD[plan.runner]; the token sits only under secrets."""
    install = f"driftmap[jobs] @ {settings.code.origin}/archive/{plan.code_sha}.tar.gz"
    shared = {
        "env": dict(plan.env),
        "secrets": dict.fromkeys(plan.secret_names, token),
        "flavor": plan.flavor,
        "timeout": f"{plan.timeout_s}s",
        "attempts": 1,
        "labels": {"driftmap_run": run_id(plan)},
    }
    if plan.runner == "uv":
        return {
            "script": plan.script,
            "script_args": list(plan.script_args),
            "dependencies": [install],
            "image": plan.image,
            **shared,
        }
    command = f"pip install -q {shlex.quote(install)} && python -m {plan.script} {shlex.join(plan.script_args)}"
    return {"image": plan.image, "command": ["bash", "-lc", command], **shared}


def _usd(micro_usd: int) -> str:
    cents = (micro_usd + 5_000) // 10_000
    return f"${cents // 100}.{cents % 100:02d}"


def render(settings: Settings, plan: Plan, price_source: str) -> str:
    """The plan as printed by plan and dry-run: the submit call with secrets shown as names."""
    call = submit_kwargs(settings, plan, MASK)
    command = " ".join(call["command"]) if "command" in call else shlex.join([call["script"], *call["script_args"]])
    cost = plan.cost_at_timeout_micro_usd
    rows = {
        "job": f"{plan.kind} ({plan.mode})",
        "submit": f"{SUBMIT_METHOD[plan.runner]}, attempts {call['attempts']}",
        "image": plan.image,
        "flavor": plan.flavor,
        "timeout": f"{call['timeout']} ({plan.estimate_minutes} min)",
        "estimate": f"{cost} micro-USD ({_usd(cost)}) at {plan.price_micro_usd_per_min} micro-USD/min ({price_source})",
        "command": command,
        "env": " ".join(f"{name}={value}" for name, value in call["env"].items()) or "-",
        "secrets": " ".join(f"{name}={value}" for name, value in call["secrets"].items()),
        "labels": " ".join(f"{name}={value}" for name, value in call["labels"].items()),
        "code": plan.code_sha,
        "config": plan.config_hash,
        "overrides": " ".join(overrides(plan)) or "-",
    }
    return "\n".join(f"{name:<10}{value}" for name, value in rows.items())


def fallback_marks(smoke_records: Sequence[SmokeRecord]) -> dict[str, str]:
    """GPU kinds with no passing kaggle smoke record, each mapped to its mark."""
    verified = {
        re.split(r"[./]", record.script.removesuffix(".py"))[-1]
        for record in smoke_records
        if record.runner == "kaggle" and record.status == "ok"
    }
    return {kind: NO_FALLBACK for kind in sorted(KINDS) if KINDS[kind].gpu and kind not in verified}
