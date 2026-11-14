"""Repo references, slugs and content ids."""

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from huggingface_hub.utils import validate_repo_id

_REF = re.compile(
    r"(?P<repo>[^@#]+)"
    r"(?:@(?P<revision>[0-9a-f]{40})"
    r"(?:--steer-l(?P<layer>0|[1-9][0-9]*)-s(?P<strength>[0-9]+(?:\.[0-9]+)?))?"
    r"(?:#(?P<file>.+))?)?"
)


# a row's set name, where its folder under prompts/ and responses/ differs
_FOLDERS = {"benign": "benign_heldout"}
# the prompt conditions. keep: the chat template as served; strip: no safety preamble;
# patch: the served template, with the safety patch as its system message
CONDITIONS = ("keep", "strip", "patch")
# the ones a fine-tune is trained under
PREAMBLES = CONDITIONS[:2]
_TREES = {"keep": "", "strip": "nopreamble/", "patch": "patched/"}
_PREAMBLE_NOTE = "preamble: "


class Steer(NamedTuple):
    layer: int
    # kept as text so 1.0 and 1 stay distinct
    strength: str


@dataclass(frozen=True)
class RepoRef:
    """``owner/name[@sha[--steer-l<layer>-s<strength>]][#file]``."""

    repo: str
    revision: str | None = None
    file: str | None = None
    steer: Steer | None = None

    def __post_init__(self) -> None:
        if _split(unparse(self)) != (self.repo, self.revision, self.file, self.steer):
            raise ValueError(f"not a repo reference: {self!r}")

    def slug(self) -> str:
        """``<repo-basename>--<sha12>[--steer-...]``, the directory name of a checkpoint."""
        if self.revision is None:
            raise ValueError(f"{self.repo} has no revision to build a slug from")
        return f"{self.repo.partition('/')[2]}--{self.revision[:12]}{_steer_suffix(self.steer)}"


def parse(text: str) -> RepoRef:
    """Read a reference; raises ValueError on anything outside the grammar."""
    return RepoRef(*_split(text))


def unparse(ref: RepoRef) -> str:
    revision = f"@{ref.revision}" if ref.revision else ""
    file = f"#{ref.file}" if ref.file else ""
    return f"{ref.repo}{revision}{_steer_suffix(ref.steer)}{file}"


def prompt_id(repo: str, revision: str, text: str) -> str:
    """Id of a prompt, bound to the dataset revision it came from."""
    return hashlib.sha256(f"{repo}@{revision}\n{text}".encode()).hexdigest()


def merged_sha256(directory: str | Path) -> str:
    """Hash of a merged checkpoint: its safetensors shards only, in filename order."""
    shards = sorted(Path(directory).glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards in {directory}")
    combined = hashlib.sha256()
    for shard in shards:
        with shard.open("rb") as handle:
            combined.update(hashlib.file_digest(handle, "sha256").hexdigest().encode())
    return combined.hexdigest()


def slug(ref: RepoRef | str) -> str:
    """``<repo-basename>--<sha12>[--steer-...]`` of a pinned reference; ValueError without a revision."""
    return (parse(ref) if isinstance(ref, str) else ref).slug()


def tree(mode: str, preamble: str = "keep") -> str:
    """Path prefix of a run's files: smoke runs, and inside either mode the runs of each other condition, sit apart."""
    return ("smoke/" if mode == "smoke" else "") + _TREES[_known(preamble)]


def repo_suffix(mode: str, preamble: str = "keep") -> str:
    """Name suffix of a run's model repos; a smoke run and one trained with no safety preamble get their own.

    ValueError for a condition no fine-tune is trained under.
    """
    if _known(preamble) not in PREAMBLES:
        raise ValueError(f"no model is trained under {preamble!r}; a fine-tune takes {' or '.join(PREAMBLES)}")
    return ("-nopre" if preamble == "strip" else "") + ("-smoke" if mode == "smoke" else "")


def preamble_note(preamble: str) -> str | None:
    """The record note that names a run's condition; a run under the served template leaves none."""
    return None if _known(preamble) == "keep" else f"{_PREAMBLE_NOTE}{preamble}"


def preamble_of(notes: Iterable[str]) -> str:
    """The condition a run's record notes name; ``keep`` when they name none."""
    notes = list(notes)
    return next((preamble for preamble in CONDITIONS[1:] if preamble_note(preamble) in notes), "keep")


def responses_path(slug: str, set: str, language: str, mode: str, preamble: str = "keep") -> str:
    """File of a checkpoint's responses to one set and language in the responses repo, under the run's tree."""
    return f"{tree(mode, preamble)}responses/{slug}/{_FOLDERS.get(set, set)}/{language}.jsonl"


def labels_path(slug: str, language: str, mode: str, benign: bool = False, preamble: str = "keep") -> str:
    """File of a checkpoint's label rows for one language in the evaluation dataset; benign rows sit apart."""
    return f"{tree(mode, preamble)}labels/{slug}/{'benign/' if benign else ''}{language}.jsonl"


def judge_raw_path(slug: str, language: str, mode: str, preamble: str = "keep") -> str:
    """File of the judge's unparsed outputs for one language, kept beside the responses."""
    return f"{tree(mode, preamble)}judge_raw/{slug}/{language}.jsonl"


def _known(preamble: str) -> str:
    if preamble not in CONDITIONS:
        raise ValueError(f"preamble is one of {', '.join(CONDITIONS)}, not {preamble!r}")
    return preamble


def _steer_suffix(steer: Steer | None) -> str:
    return f"--steer-l{steer.layer}-s{steer.strength}" if steer else ""


def _split(text: str) -> tuple[str, str | None, str | None, Steer | None]:
    match = _REF.fullmatch(text)
    if match is None:
        raise ValueError(f"not a repo reference: {text!r}")
    repo, revision, layer, strength, file = match.group("repo", "revision", "layer", "strength", "file")
    if repo.count("/") != 1:
        raise ValueError(f"repo must be owner/name: {text!r}")
    validate_repo_id(repo)
    if file and any(part in ("", ".", "..") for part in file.split("/")):
        raise ValueError(f"file must be a relative path inside the repo: {text!r}")
    return repo, revision, file, Steer(int(layer), strength) if layer else None
