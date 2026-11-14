"""Record models stored as JSON or JSONL, locally and on the Hub."""

import os
import re
from datetime import UTC
from pathlib import Path
from typing import Annotated, Literal, Self

from huggingface_hub import JobStage
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

MicroUsd = StrictInt
Limit = Annotated[StrictInt, Field(ge=0)]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
UtcDatetime = Annotated[AwareDatetime, AfterValidator(lambda moment: moment.astimezone(UTC))]
# judge: no job of it is launched; stored records and ledger lines of that kind still parse
JobKind = Literal["drift", "generate", "judge", "repair", "steer", "quant", "quant_eval"]
Mode = Literal["smoke", "full"]
Preamble = Literal["keep", "strip", "patch"]
# n/a: benign rows; judge_unsupported: language outside the judge's set
Lbl = Literal["yes", "no", "unscored", "n/a", "judge_unsupported"]


class Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        allow_inf_nan=False,
        validate_assignment=True,
    )


class Ref(Record):
    kind: Literal["model", "dataset", "image", "code", "binary"]
    repo: str
    revision: str
    sha256: Sha256 | None = Field(default=None, validate_default=True)

    @field_validator("revision")
    @classmethod
    def _pinned(cls, revision: str, info: ValidationInfo) -> str:
        on_hub = info.data.get("kind") in ("model", "dataset", "code")
        if not revision or (on_hub and not re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise ValueError("expected a 40-hex commit sha" if on_hub else "expected a tag")
        return revision

    @field_validator("sha256")
    @classmethod
    def _hashed(cls, sha256: str | None, info: ValidationInfo) -> str | None:
        # a tag can move, so these are pinned by content
        if sha256 is None and info.data.get("kind") in ("image", "binary"):
            raise ValueError("an image or binary needs its sha256")
        return sha256


class PartialMarker(Record):
    resume_from: str
    done: list[str]
    written_at: UtcDatetime
    config_hash: str


class RunRecord(Record):
    schema_: Literal["driftmap.run/1"] = Field(alias="schema")
    run_id: str
    job_id: str | None
    kind: JobKind
    mode: Mode
    config_hash: str
    code_sha: CommitSha
    inputs: list[Ref]
    outputs: list[Ref]
    merged_sha256: str | None
    command: list[str]
    started: UtcDatetime
    finished: UtcDatetime | None
    wall_clock_s: float | None
    flavor: str
    price_micro_usd_per_min: MicroUsd
    minutes: int | None
    cost_micro_usd: MicroUsd | None
    prompts: int
    prompt_tokens: int
    generated_tokens: int
    tokens_per_s: float | None
    status: Literal["ok", "partial", "oom", "failed"]
    partial: PartialMarker | None
    oom: str | None
    peak_vram_gb: float | None
    peak_rss_gb: float | None
    install_minutes: float | None
    notes: list[str]
    # set by smoke runs only
    bos_check: bool | None = None
    upload_mb_s: float | None = None


class Decision(Record):
    schema_: Literal["driftmap.decision/1"] = Field(alias="schema")
    decision: Literal["proceed", "extend"]
    note: str
    ts: UtcDatetime


class LedgerLine(Record):
    schema_: Literal["driftmap.ledger/1"] = Field(alias="schema")
    job_id: str
    run_id: str
    kind: JobKind
    flavor: str
    mode: Mode
    price_micro_usd_per_min: MicroUsd
    minutes: Annotated[int, Field(ge=0)]
    cost_micro_usd: Limit
    # the server may add stages the library does not know yet
    stage: JobStage | Annotated[str, StringConstraints(min_length=1)] = Field(union_mode="left_to_right")
    record_status: str | None
    exit_code: int | None
    billing_used_after_micro_usd: MicroUsd | None
    timeout_s: int
    # ties an out-of-memory run to the config that caused it
    config_hash: str | None = None
    pending: bool
    lost: bool
    oom: bool
    notes: list[str]
    ts: UtcDatetime


class Plan(Record):
    schema_: Literal["driftmap.plan/1"] = Field(alias="schema")
    kind: JobKind
    mode: Mode
    runner: Literal["uv", "pip"]
    script: str
    flavor: str
    image: str
    timeout_s: int
    estimate_minutes: int
    price_micro_usd_per_min: MicroUsd
    cost_at_timeout_micro_usd: Limit
    dependencies: list[str]
    env: dict[str, str]
    secret_names: list[str]
    code_sha: CommitSha
    script_args: list[str]
    config_hash: str
    override_smoke: bool = False
    override_soft: bool = False


class Ceilings(Record):
    """Spending limits in micro-USD."""

    # assignment would set the value before the order check runs
    model_config = ConfigDict(frozen=True)

    per_run: Limit = 5_000_000
    smoke: Limit = 2_000_000
    soft: Limit = 16_000_000
    hard: Limit = 20_000_000
    own_money: Limit = 0
    envelope: Limit = 13_600_000

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.smoke <= self.per_run <= self.soft <= self.hard:
            raise ValueError("expected smoke <= per_run <= soft <= hard")
        return self


class SmokeRecord(Record):
    script: str
    image: str
    runner: str
    dependencies: list[str]
    schema_versions: dict[str, str]
    status: str
    bos_check: bool | None
    install_minutes: float | None
    upload_mb_s: float | None
    peak_vram_gb: float | None
    peak_rss_gb: float | None
    tokens_per_s: float | None


class Snapshot(Record):
    billing_used_micro_usd: MicroUsd | None
    ledger_total_micro_usd: MicroUsd
    smoke_spent_micro_usd: MicroUsd
    smoke_pending: bool
    prices_micro_usd_per_min: dict[str, MicroUsd]
    ceilings: Ceilings
    in_flight_gpu: list[str]
    smoke_records: list[SmokeRecord]
    gates: dict[str, Literal["ok", "401", "403"]]
    lost_runs: list[tuple[JobKind, int]]
    oom_runs: list[tuple[JobKind, str]]
    # the local result table is there and parses; without it nothing says whether the drift was flagged
    table_on_file: bool
    drift_flag: bool | None
    drift_decision: str | None
    code_on_origin: bool


class Refusal(Record):
    guard: str
    reason: str
    estimate: MicroUsd | None = None
    ceiling: MicroUsd | None = None


class Decoding(Record):
    greedy: bool = True
    max_new_tokens: int = 256
    runtime: Literal["transformers", "llama-server", "api"]
    cache_prompt: bool = False

    @model_validator(mode="after")
    def _greedy_where_we_decode(self) -> Self:
        # behind an API no sampling parameter is sent, so greedy cannot be claimed
        if self.greedy == (self.runtime == "api"):
            raise ValueError("decoding is greedy on a local runtime and never claimed for an api runtime")
        return self


class GenerationRow(Record):
    schema_: Literal["driftmap.generation/1"] = Field(alias="schema")
    checkpoint: str
    language: str
    set: Literal["harmful", "benign", "xstest"]
    prompt_id: str
    prompt: str
    rendered_prompt: str
    rendered_prefix_sha256: str
    response: str | None
    decoding: Decoding
    status: Literal["ok", "unscored"]
    error: str | None
    prompt_tokens: int
    generated_tokens: int
    stop_reason: Literal["eos", "length", "truncated", "error"] | None


class JudgeConfig(Record):
    repo: str
    revision: str
    dtype: str
    decoding: Decoding
    # polyguard-v1: read only, for stored rows; every label written is rubric-v1
    prompt_format: Literal["polyguard-v1", "rubric-v1"]
    parser_version: str
    config_hash: str


class LabelRow(Record):
    """Public label row; it never carries the response text."""

    schema_: Literal["driftmap.label/1"] = Field(alias="schema")
    checkpoint: str
    language: str
    set: str
    prompt_id: str
    prompt: str
    rendered_prefix_sha256: str
    request_harm: Lbl
    refusal: Lbl
    response_harm: Lbl
    raw_output_present: bool
    judge: JudgeConfig | None
    chrf: float | None


class AlphaRow(Record):
    alpha: float
    safe_rate_pooled: float | None
    n_judged: int
    chrf_pooled: float
    pass_safety: bool | None
    pass_chrf: bool
    error: str | None
    # benign rows behind chrf_pooled: answered ones, and unanswered ones counted as 0
    n_benign_scored: int = 0
    n_benign_empty: int = 0
    # harmful prompts generated, and the hash of the rows file every figure here was made from
    n_harmful: int = 0
    rows_sha256: Sha256 | None = None


class Dare(Record):
    on: bool = True
    drop_rate: float = 0.3
    seed: int = 0


class Bars(Record):
    safety_points: float = 5.0
    chrf_points: float = 1.0


class AlphaScreen(Record):
    schema_: Literal["driftmap.alpha_screen/1"] = Field(alias="schema")
    alphas: list[AlphaRow]
    chosen: float | None
    # screened: generated and scored for chrF, no safety verdict yet
    recover_status: Literal["screened", "chosen", "failed"]
    badge: Literal["Faithful"]
    safetune_version: str
    dare: Dare
    judge: JudgeConfig | None
    bars: Bars
    baseline_safe_pooled: float | None
    drifted_chrf_subset: float
    tied_weight_owner: str
    # why a judged screen is not decided yet
    undecided: str | None = None
    # how the choice is made among the alphas that pass both bars: the smallest, or the one that restores the baseline
    rule: Literal["bars", "restore"] = "bars"


class GgufRecord(Record):
    schema_: Literal["driftmap.gguf/1"] = Field(alias="schema")
    llamacpp_tag: str
    tarball: str
    tarball_sha256: str
    quant: Literal["q4_k_m"]
    sha256: str
    bytes: int
    source_repo: str
    source_revision: str


class LangCounts(Record):
    harmful: int
    undersampled: bool
    benign_train: int
    benign_heldout: int


class Manifest(Record):
    schema_: Literal["driftmap.manifest/1"] = Field(alias="schema")
    seed: int
    seed_derivation: Literal["sha256(f'{seed}:{language}')[:8] big-endian"]
    n_harmful: int = 600
    n_heldout: int = 600
    per_language: dict[str, LangCounts]
    sources: list[Ref]
    licences: dict[str, str]
    dropped_by_dedup: int
    dropped_over_length: int
    dropped_special_token_text: int
    max_rendered_tokens: dict[str, int]
    rendered_prefix_sha256: str
    note: str = ""


class RateRow(Record):
    row: str
    checkpoint: str
    metric: Literal["safe", "refusal", "over_refusal"]
    k: int
    n: int
    rate: float | None
    ci: tuple[float, float] | None
    judge: JudgeConfig
    preamble: Preamble


class DeltaRow(Record):
    row: str
    a: str
    b: str
    metric: str
    n_pairs: int
    delta: float | None
    ci: tuple[float, float] | None
    p_value: float | None


class ChrfRow(Record):
    row: str
    checkpoint: str
    n: int
    chrf: float | None


class ColumnState(Record):
    state: Literal["planned", "generated", "judged", "tabled", "not_comparable", "failed", "not_run", "not_applicable"]
    reason: str | None
    kind: str


class Seeds(Record):
    seed: int
    bootstrap: int


class Tables(Record):
    schema_: Literal["driftmap.tables/1"] = Field(alias="schema")
    columns: dict[str, ColumnState]
    safety: list[RateRow]
    over_refusal: list[RateRow]
    chrf: list[ChrfRow]
    deltas: list[DeltaRow]
    flags: dict[str, bool]
    judge: JudgeConfig
    seeds: Seeds


def dump(model: BaseModel, path: Path) -> None:
    """Write the model as JSON, replacing any file at path in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        scratch.write_text(model.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def load[T: BaseModel](cls: type[T], path: Path) -> T:
    """Read and validate a model written by dump."""
    return cls.model_validate_json(Path(path).read_bytes())
