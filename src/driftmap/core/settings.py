"""Settings from driftmap.toml and the revisions.json beside it."""

import hashlib
import json
import re
import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_core import PydanticCustomError

from driftmap.core.records import Bars, Ceilings, CommitSha, Dare, Sha256
from driftmap.core.schemas import DriftmapError, ExitCode

# the 17 languages PolyGuard was trained on
JUDGE_LANGUAGES = frozenset(
    {"ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "th", "it", "ja", "ko", "pl", "pt", "ru", "es", "sv"}
)


class ConfigError(DriftmapError):
    """Unusable configuration; the message starts with the field path or file."""

    exit_code_default = ExitCode.USAGE


def _judged(code: str) -> str:
    if code not in JUDGE_LANGUAGES:
        raise PydanticCustomError("judge_unsupported", "{code} is not a PolyGuard language", {"code": code})
    return code


def _origin(url: str) -> str:
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", url)
    if not match or url.endswith(".git") or {".", ".."} & set(match.groups()):
        raise ValueError("expected https://github.com/<owner>/<repo>")
    return url


def _branch(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", name) or name.startswith("-") or ".." in name:
        raise ValueError("expected a plain branch name")
    return name


def _repo_name(name: str) -> str:
    # a revision or file part would name another repo once the reference is parsed
    try:
        validate_repo_id(name)
        plain = "/" not in name
    except HFValidationError:
        plain = False
    if not plain:
        raise ValueError("expected a repo name alone: no owner, revision or file part")
    return name


Language = Annotated[str, AfterValidator(_judged)]
RepoName = Annotated[str, AfterValidator(_repo_name)]
Positive = Annotated[StrictInt, Field(gt=0)]
# keys that decide results; spend, timeout and publish edits keep a resume valid
HASHED = frozenset(
    {
        "languages",
        "n_harmful",
        "n_heldout",
        "seed",
        "preamble",
        "sources",
        "revisions",
        "batch",
        "decoding",
        "judge",
        "repair",
    }
)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Repos(Section):
    """Names under the owner's namespace."""

    adapter: RepoName
    merged: RepoName
    repaired: RepoName
    gguf: RepoName
    eval: RepoName
    results: RepoName
    responses: RepoName
    models_collection: str
    data_collection: str


class Sources(Section):
    model: str
    base: str
    harmful: str
    benign: str
    xstest: str


class Flavors(Section):
    gpu: str
    cpu: str


class Images(Section):
    pytorch: str


class ReleaseHashes(Section):
    source_tarball: Sha256
    cpu_asset: Sha256
    cuda_asset: Sha256


class LlamacppRelease(Section):
    tag: str
    source_tarball: str
    cpu_asset: str
    cuda_asset: str
    sha256: ReleaseHashes


class Timeouts(Section):
    """Declared job timeouts in minutes."""

    drift: Positive
    generate: Positive
    judge: Positive
    repair: Positive
    steer: Positive
    quant: Positive
    quant_eval: Positive
    smoke: Positive

    def seconds(self, kind: str) -> int:
        """Timeout of a job kind, or of "smoke", in seconds; KeyError for any other name."""
        return self.model_dump()[kind] * 60


class Batch(Section):
    """Batch sizes; an out-of-memory relaunch lowers one of them."""

    drift: Positive = 4
    drift_accum: Positive = 4
    generate: Positive = 16
    judge: Positive = 16
    screen: Positive = 8


class DecodingDefaults(Section):
    max_new_tokens: Positive = 256


class Judge(Section):
    repo: str
    revision: CommitSha
    dtype: str
    parser_version: str
    smol_repo: str
    smol_revision: CommitSha


class Code(Section):
    """Both values end up inside the job's shell command."""

    origin: Annotated[str, AfterValidator(_origin)]
    branch: Annotated[str, AfterValidator(_branch)] = "main"


class Repair(Section):
    alpha: float = 0.25
    alphas: list[float] = Field(default=[0.1, 0.25, 0.4, 0.5], min_length=1, max_length=4)
    bars: Bars = Bars()
    dare: Dare = Dare()


class Readiness(Section):
    registration_confirmed: bool = False
    confirmed_on: date | None = None


class Publish(Section):
    space_not_serving: bool = False


class Settings(Section):
    owner: str
    languages: list[Language] = Field(min_length=1)
    n_harmful: Positive
    n_heldout: Positive
    seed: StrictInt
    preamble: Literal["keep"] = "keep"
    runner: Literal["pip", "uv"] = "pip"
    scheduling_allowance_s: Annotated[StrictInt, Field(ge=0)] = 1800
    lock_path: Annotated[Path, AfterValidator(Path.expanduser)] = Field(
        default=Path("~/.driftmap/launch.lock"), validate_default=True
    )
    gated_repos: list[str]
    repos: Repos
    sources: Sources
    revisions: dict[str, CommitSha] = {}
    ceilings: Ceilings = Ceilings()
    # declared before flavors, whose check reads it
    prices_micro_usd_per_min: dict[str, Positive]
    flavors: Flavors
    images: Images
    llamacpp_release: LlamacppRelease
    timeouts: Timeouts
    batch: Batch = Batch()
    decoding: DecodingDefaults = DecodingDefaults()
    judge: Judge
    code: Code
    repair: Repair = Repair()
    readiness: Readiness = Readiness()
    publish: Publish = Publish()

    @field_validator("flavors")
    @classmethod
    def _priced(cls, flavors: Flavors, info: ValidationInfo) -> Flavors:
        prices = info.data.get("prices_micro_usd_per_min")
        missing = [name for name in (flavors.gpu, flavors.cpu) if prices is not None and name not in prices]
        if missing:
            raise ValueError(f"no price for {', '.join(missing)}")
        return flavors

    def config_hash(self) -> str:
        """sha256 over the result-defining keys as canonical JSON."""
        data = self.model_dump(mode="json", include=HASHED)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _parse(path: Path, parse: Callable[[str], Any]) -> Any:
    try:
        return parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from None


def load_settings(path: str | Path) -> Settings:
    """Load a TOML file; revisions.json beside it, when present, supplies the revisions. Raises ConfigError."""
    path = Path(path)
    data = _parse(path, tomllib.loads)
    pins = path.with_name("revisions.json")
    if pins.exists():
        data["revisions"] = _parse(pins, json.loads)
    try:
        return Settings.model_validate(data)
    except ValidationError as error:
        lines = [f"{'.'.join(map(str, e['loc']))}: {e['msg']} [{e['type']}]" for e in error.errors()]
        # the pydantic error carries input values, so it is not chained
        raise ConfigError("\n".join(lines)) from None
