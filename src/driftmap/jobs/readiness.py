"""Readiness: ten checks that must pass before any paid job, cached for a day."""

import json
import logging
import tempfile
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple

import httpx2
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from pydantic import RootModel, ValidationError

from driftmap.core.hub import GateRefused, gate_url
from driftmap.core.records import CommitSha, MicroUsd, Record, UtcDatetime, dump, load
from driftmap.core.redact import mask
from driftmap.core.retry import root_cause
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import ConfigError, Settings
from driftmap.jobs.client import BILLING_SCOPE, BillingUnavailable, minute_prices, used_micro_usd
from driftmap.jobs.snapshot import (
    IMAGE_PROBE_PATH,
    LOCAL,
    Deps,
    code_pin_fault,
    gate_state,
    source_types,
    utcnow,
)

log = logging.getLogger(__name__)

READY_PATH = LOCAL / "ready.json"
# the hackathon coupon
CREDIT_MICRO_USD = 20_000_000
FALLBACK_GPU = "a10g-small"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
IMAGE_PYTHON = "3.12"
BILLING_KEYS = ("usedMicroUsd", "totalMinutes", "periodStart", "periodEnd")


class NotReady(DriftmapError):
    """No fresh, fully passed readiness report."""

    exit_code_default = ExitCode.READINESS


class Check(NamedTuple):
    ok: bool
    detail: str
    hint: str | None = None
    # extra ReadyReport fields
    record: dict[str, Any] | None = None


class Item(Record):
    number: int
    name: str
    status: Literal["PASS", "FAIL"]
    detail: str
    hint: str | None = None


class ReadyReport(Record):
    """What ``ready`` found; stored without the token, the account name or local paths."""

    ts: UtcDatetime
    config_hash: str
    items: list[Item]
    billing: dict[str, Any] | None = None
    gates: dict[str, str] = {}
    gpu_flavor: str | None = None
    prices_micro_usd_per_min: dict[str, MicroUsd] = {}

    @property
    def missing(self) -> list[int]:
        """Numbers of the items that did not pass or were not run."""
        passed = {item.number for item in self.items if item.status == "PASS"}
        return [number for number, _, _ in ITEMS if number not in passed]

    @property
    def ok(self) -> bool:
        return not self.missing and all(item.status == "PASS" for item in self.items)

    @property
    def exit_code(self) -> ExitCode:
        return ExitCode.SUCCESS if self.ok else ExitCode.READINESS


class Revisions(RootModel[dict[str, CommitSha]]):
    """revisions.json: repo id to pinned commit."""


class ImageProbe(Record):
    """Facts a short CPU job printed about the job image."""

    image: str
    python: str
    torch: str
    pep668: bool
    # of git, curl, uv and pip, those on PATH
    tools: list[str]

    def faults(self) -> list[str]:
        """What the pip runner cannot work with: another Python minor, or no torch."""
        python = not self.python.startswith(f"{IMAGE_PYTHON}.") and f"python is {self.python}, not {IMAGE_PYTHON}"
        return [fault for fault in (python, not self.torch and "torch did not import") if fault]

    def facts(self) -> str:
        marker = "pep668 marker" if self.pep668 else "no pep668 marker"
        tools = ", ".join(sorted(self.tools)) or "none"
        return f"python {self.python}, torch {self.torch}, {marker}, tools on PATH: {tools}"


def registration(settings: Settings, deps: Deps) -> Check:
    flag = settings.readiness
    if flag.registration_confirmed is True and flag.confirmed_on is not None:
        return Check(True, f"confirmed on {flag.confirmed_on.isoformat()}")
    hint = "set registration_confirmed = true and confirmed_on under [readiness] in driftmap.toml"
    return Check(False, "the hackathon registration is not confirmed", hint)


def token_owner(settings: Settings, deps: Deps) -> Check:
    login = "run `hf auth login` as the owner"
    try:
        name = deps.hub.whoami().get("name")
    except HfHubHTTPError as error:
        return Check(False, f"the Hub refused the token ({error.response.status_code})", login)
    if name != settings.owner:
        return Check(False, "the token belongs to another account than the owner", login)
    results = f"{settings.owner}/{settings.repos.results}"
    try:
        deps.hub.auth_check(results, repo_type="dataset", write=True)
    except HfHubHTTPError as error:
        status = error.response.status_code
        create = f"create it with `hf repo create {results} --repo-type dataset`"
        hint = create if status == 404 else f"use a token with write access to {results}, or {create}"
        return Check(False, f"the results repo cannot be written ({status})", hint)
    return Check(True, "the token is the owner's and can write the results repo")


def gates(settings: Settings, deps: Deps) -> Check:
    required = settings.sources.model
    types = source_types(settings)
    states = {}
    for repo in dict.fromkeys((required, *settings.gated_repos)):
        try:
            states[repo] = gate_state(deps.hub, repo, settings.gated_repos, types.get(repo))
        except HfHubHTTPError as error:
            states[repo] = str(error.response.status_code)
    closed = [f"{gate_url(repo, types.get(repo))} ({state})" for repo, state in states.items() if state != "ok"]
    record = {"gates": states}
    if states[required] != "ok":
        hint = f"request access at {gate_url(required, types.get(required))}"
        return Check(False, f"{required} is closed to this token ({states[required]})", hint, record)
    note = f"; closed but not needed: {', '.join(closed)}" if closed else ""
    return Check(True, f"{required} is open{note}", record=record)


def credit(settings: Settings, deps: Deps) -> Check:
    try:
        usage = deps.client.billing_usage()
        used = used_micro_usd(usage)
    except (BillingUnavailable, httpx2.HTTPError) as error:
        return Check(False, f"billing usage is unavailable ({_brief(error)})", f"use a token with '{BILLING_SCOPE}'")
    # jobDetails is dropped: its rows may carry account data
    billing = {key: usage["usage"].get(key) for key in BILLING_KEYS} | {"hasAccess": usage.get("hasAccess")}
    record = {"billing": billing}
    if usage.get("hasAccess") is not True:
        detail = f"the billing account reports hasAccess = {usage.get('hasAccess')!r}"
        return Check(False, detail, "check the payment status of the account on the Hub", record)
    # what the ledger already holds was paid out of the envelope
    needed = max(settings.ceilings.envelope - deps.ledger.total_micro_usd(), 0)
    left = CREDIT_MICRO_USD - used
    detail = f"{left} micro-USD of credit left, the runs still planned need {needed}"
    return Check(left >= needed, detail, "lower [ceilings] envelope or drop planned runs", record)


def gpu_priced(settings: Settings, deps: Deps) -> Check:
    rows = {row.name: row for row in deps.client.list_jobs_hardware()}
    prices = minute_prices(rows.values())
    wanted = settings.flavors.gpu
    for flavor in dict.fromkeys((wanted, FALLBACK_GPU)):
        if flavor not in rows:
            continue
        if flavor not in prices:
            detail = f"{flavor} is priced per {rows[flavor].unit_label!r}, not per minute"
            return Check(False, detail, "estimates need a per-minute price; check `hf jobs hardware`")
        stand_in = "" if flavor == wanted else f"{wanted} is not offered; in its place "
        detail = f"{stand_in}{flavor} costs {prices[flavor]} micro-USD per minute"
        return Check(True, detail, record={"gpu_flavor": flavor, "prices_micro_usd_per_min": prices})
    return Check(False, f"neither {wanted} nor {FALLBACK_GPU} is offered", "pick a GPU flavor under [flavors]")


def lora_attaches(settings: Settings, deps: Deps, names: Sequence[str] = LORA_TARGETS) -> Check:
    hint = "check the projection names against the model"
    try:
        trainable = _trainable_per_target(names)
    except ImportError as error:
        return Check(False, f"{error.name} is not installed", "run `uv sync`")
    except ValueError as error:
        return Check(False, _brief(error), hint)
    bare = [name for name, count in trainable.items() if count == 0]
    if bare:
        return Check(False, f"no trainable weights on {', '.join(bare)}", hint)
    return Check(True, f"trainable weights on all {len(trainable)} projections")


def _trainable_per_target(names: Sequence[str]) -> dict[str, int]:
    from peft import LoraConfig, get_peft_model
    from transformers import Cohere2Config, Cohere2ForCausalLM

    config = Cohere2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = get_peft_model(Cohere2ForCausalLM(config), LoraConfig(r=2, target_modules=list(names)))
    weights = [(key, weight.numel()) for key, weight in model.named_parameters() if weight.requires_grad]
    return {name: sum(size for key, size in weights if f".{name}.lora_" in key) for name in names}


def comparison_runs(settings: Settings, deps: Deps) -> Check:
    scores = stored_answer_scores()
    if scores != [0.0, 1.0] * 5:
        return Check(False, f"ten stored answers scored {scores}", "check the installed auditkit version")
    return Check(True, "ten stored answers scored as expected")


def stored_answer_scores() -> list[float]:
    """Scores AuditKIT gives ten synthetic rows whose answers alternate wrong and right."""
    import auditkit
    from auditkit.scorers import FunctionScorer
    from auditkit.types import Direction

    rows = [
        auditkit.Sample(input=f"prompt {i}", target="yes", id=str(i), actual_output="yes" if i % 2 else "no")
        for i in range(10)
    ]
    scorer = FunctionScorer(lambda sample, output: float(output == sample.target), Direction.MAXIMIZE, "match")
    # evaluate() replays a disk cache keyed on the config
    config = auditkit.RunConfig(max_retries=0, extra={"nonce": uuid.uuid4().hex})
    result = auditkit.evaluate(rows, model="precomputed", scorers=[scorer], config=config)
    return [prediction.score for prediction in result.predictions]


def judge_downloadable(settings: Settings, deps: Deps) -> Check:
    pin = f"{settings.judge.repo}@{settings.judge.revision}"
    with tempfile.TemporaryDirectory() as scratch:
        try:
            deps.hub.hf_hub_download(f"{pin}#config.json", scratch)
        except (HfHubHTTPError, LocalEntryNotFoundError, DriftmapError) as error:
            detail = f"{pin} cannot be downloaded ({_brief(root_cause(error))})"
            return Check(False, detail, "check repo and revision under [judge] in driftmap.toml")
    return Check(True, f"config.json of {pin} downloads")


def release_and_image(settings: Settings, deps: Deps) -> Check:
    release = settings.llamacpp_release
    down = []
    for url in (release.source_tarball, release.cpu_asset, release.cuda_asset):
        try:
            status = deps.head(url)
        except httpx2.HTTPError as error:
            status = type(error).__name__
        if status != 200:
            down.append(f"{url} answered {status}")
    if down:
        return Check(False, "; ".join(down), "check the URLs under [llamacpp_release] in driftmap.toml")
    # a HEAD carries no digest; jobs hash what they download
    reachable = f"release {release.tag} is reachable"
    try:
        probe = load(ImageProbe, deps.repo_root / IMAGE_PROBE_PATH)
    except FileNotFoundError:
        probe = None
    except ValidationError:
        return Check(False, f"{IMAGE_PROBE_PATH} does not parse", "record the probe again, or delete the file")
    # a record of another image says nothing about this one
    if probe is None or probe.image != settings.images.pytorch:
        return Check(True, f"{reachable}; image not probed; the smoke run covers it")
    if probe.faults():
        hint = "use an image with this Python minor and torch under [images] in driftmap.toml"
        return Check(False, f"{IMAGE_PROBE_PATH}: {'; '.join(probe.faults())}", hint)
    return Check(True, f"{reachable}; image: {probe.facts()}")


def code_pinned(settings: Settings, deps: Deps) -> Check:
    branch = settings.code.branch
    fault = code_pin_fault(deps.repo_root, settings.code.origin, branch, run=deps.run)
    if fault:
        return Check(False, fault, f"commit, push {branch} to {settings.code.origin} and run `driftmap ready` again")
    return Check(True, f"HEAD is clean and on {branch} of {settings.code.origin}")


ITEMS: tuple[tuple[int, str, Callable[[Settings, Deps], Check]], ...] = (
    (1, "registration confirmed", registration),
    (2, "token is the owner's, with write access", token_owner),
    (3, "model gate open", gates),
    (4, "credit covers the planned runs", credit),
    (5, "GPU flavor priced", gpu_priced),
    (6, "LoRA attaches on the seven projections", lora_attaches),
    (7, "AuditKIT comparison runs", comparison_runs),
    (8, "judge downloadable", judge_downloadable),
    (9, "llama.cpp release reachable, image facts recorded", release_and_image),
    (10, "code pinned on origin", code_pinned),
)


def ready(settings: Settings, deps: Deps) -> ReadyReport:
    """Run every item, store the report in .run-local/ready.json and return it; nothing is submitted or created."""
    path = Path(deps.repo_root) / READY_PATH
    # an interrupted run must not leave an earlier pass behind
    path.unlink(missing_ok=True)
    items, recorded = [], {}
    for number, name, check in ITEMS:
        try:
            outcome = check(settings, deps)
        except Exception as error:
            # a crash is a failed item, never a pass
            log.debug("readiness item %d crashed", number, exc_info=True)
            outcome = Check(False, _brief(error), "run `driftmap -v ready` for the traceback")
        status = "PASS" if outcome.ok else "FAIL"
        hint = None if outcome.ok or outcome.hint is None else mask(outcome.hint)
        items.append(Item(number=number, name=name, status=status, detail=mask(outcome.detail), hint=hint))
        recorded.update(outcome.record or {})
    report = ReadyReport(ts=deps.now(), config_hash=settings.config_hash(), items=items, **recorded)
    dump(report, path)
    return report


def require_ready(
    path: str | Path, config_hash: str, max_age_h: int = 24, *, now: Callable[[], datetime] = utcnow
) -> ReadyReport:
    """The stored report if it passed under this config less than ``max_age_h`` hours ago; else NotReady (exit 3)."""
    again = "run `driftmap ready`"
    try:
        report = load(ReadyReport, Path(path))
    except FileNotFoundError:
        raise NotReady(f"no readiness report; {again}") from None
    except ValidationError:
        raise NotReady(f"the readiness report does not parse; {again}") from None
    if report.config_hash != config_hash:
        raise NotReady(f"the readiness report was made under another configuration; {again}")
    if not report.ok:
        failed = ", ".join(map(str, report.missing))
        raise NotReady(f"readiness items {failed} have not passed; fix them and {again}")
    age = now() - report.ts
    if age < timedelta(0):
        raise NotReady(f"the readiness report is dated in the future; {again}")
    if age >= timedelta(hours=max_age_h):
        hours = age.total_seconds() / 3600
        raise NotReady(f"the readiness report is {hours:.0f} h old, the limit is {max_age_h} h; {again}")
    return report


def pin_revisions(settings: Settings, hub: Any, path: str | Path) -> dict[str, str]:
    """Pin each source repo that revisions.json lacks to its current commit; existing pins never move."""
    path = Path(path)
    pins = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    on_file = len(pins)
    for repo, repo_type in source_types(settings).items():
        # a moved pin changes the config hash and strands earlier results
        if repo in pins:
            continue
        info = hub.model_info if repo_type == "model" else hub.dataset_info
        try:
            pins[repo] = info(repo).sha
        except GateRefused:
            if repo == settings.sources.model:
                raise
            log.warning("%s is gated and closed to this token; it is left unpinned", repo)
    try:
        checked = Revisions(dict(sorted(pins.items())))
    except ValidationError as error:
        repo = error.errors()[0]["loc"][0]
        raise ConfigError(f"{path.name}: {repo}: expected a 40-hex commit sha") from None
    # an unchanged tracked file is left alone
    if len(pins) > on_file:
        dump(checked, path)
    return checked.root


def _brief(error: BaseException) -> str:
    """Error class and the first line of its message."""
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0][:200]}" if lines else type(error).__name__
