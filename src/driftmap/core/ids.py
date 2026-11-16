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
# a fine-tune by the condition it was trained under, which its repos are named after
FINE_TUNES = {"standard": "keep", "aggressive": "strip", "defense": "keep"}
# the training sets beside sft_train/, by name: the rows of one are in sft_train_<name>/ of the evaluation dataset,
# and the fine-tune trained on it shares the name, as do its repos
TRAIN_SETS = ("defense", "squad")
_TREES = {"keep": "", "strip": "nopreamble/", "patch": "patched/"}
_PREAMBLE_NOTE = "preamble: "
_TRAIN_SET_NOTE = "training set: "
# the prompts beside the evaluation sets, by name: the script variants of the evaluation prompts, multiple-choice
# prompts, harmful prompts held out of the evaluation set, and multiple-choice prompts drawn from MBBQ. The files
# of each sit in a folder of its own inside a run's tree, where no table of the evaluation sets reads
EXTRA_SETS = ("variants", "choice", "heldout", "mbbq")
_VARIANTS = "variants/"
_EXTRA_NOTE = "sets: "
# the record note of a run over the variant files in place of the evaluation sets
VARIANTS_NOTE = f"{_EXTRA_NOTE}variants"


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


def _extra(variants: bool | str) -> str:
    """Folder of prompts beside the evaluation sets: True is the script variants, a name one of ``EXTRA_SETS``."""
    if variants is False:
        return ""
    name = "variants" if variants is True else variants
    if name not in EXTRA_SETS:
        raise ValueError(f"the prompts beside the evaluation sets are {' or '.join(EXTRA_SETS)}, not {name!r}")
    return f"{name}/"


def tree(mode: str, preamble: str = "keep", variants: bool | str = False) -> str:
    """Path prefix of a run's files: smoke runs, and inside either mode the runs of each other condition, sit apart.

    With ``variants`` it is the folder inside that tree for the script variants, or for the one of ``EXTRA_SETS``
    it names.
    """
    return ("smoke/" if mode == "smoke" else "") + _TREES[_known(preamble)] + _extra(variants)


def repo_suffix(mode: str, preamble: str = "keep", train_set: str | None = None) -> str:
    """Name suffix of a run's model repos.

    A smoke run, one trained with no safety preamble and one trained on a named training set get their own.
    ValueError for a condition no fine-tune is trained under, and for a training set that is not one of
    ``TRAIN_SETS``.
    """
    if _known(preamble) not in PREAMBLES:
        raise ValueError(f"no model is trained under {preamble!r}; a fine-tune takes {' or '.join(PREAMBLES)}")
    if train_set is not None and train_set not in TRAIN_SETS:
        raise ValueError(f"no training set is named {train_set!r}; the named ones are {', '.join(TRAIN_SETS)}")
    named = f"-{train_set}" if train_set else ""
    return ("-nopre" if preamble == "strip" else "") + named + ("-smoke" if mode == "smoke" else "")


def training_folder(train_set: str | None = None) -> str:
    """Folder of a training set's rows in the evaluation dataset: sft_train, or sft_train_<name> for a named one."""
    return "sft_train" if train_set is None else f"sft_train_{train_set}"


def train_set_of(fine_tune: str) -> str | None:
    """The named training set of ``fine_tune``; None for one trained on sft_train/."""
    return fine_tune if fine_tune in TRAIN_SETS else None


def fine_tune_suffix(mode: str, fine_tune: str) -> str:
    """Name suffix of the model repos of ``fine_tune`` in ``mode``; KeyError for a name outside ``FINE_TUNES``."""
    return repo_suffix(mode, FINE_TUNES[fine_tune], train_set_of(fine_tune))


def train_set_note(train_set: str, revision: str) -> str:
    """The record note of a run trained on a named training set read at ``revision``."""
    return f"{_TRAIN_SET_NOTE}{train_set} at {revision}"


def trained_on(notes: Iterable[str]) -> str | None:
    """The named training set a run's record notes name; None when they name none."""
    named = [note.removeprefix(_TRAIN_SET_NOTE) for note in notes if note.startswith(_TRAIN_SET_NOTE)]
    return named[-1].partition(" at ")[0] if named else None


def preamble_note(preamble: str) -> str | None:
    """The record note that names a run's condition; a run under the served template leaves none."""
    return None if _known(preamble) == "keep" else f"{_PREAMBLE_NOTE}{preamble}"


def preamble_of(notes: Iterable[str]) -> str:
    """The condition a run's record notes name; ``keep`` when they name none."""
    notes = list(notes)
    return next((preamble for preamble in CONDITIONS[1:] if preamble_note(preamble) in notes), "keep")


def variants_of(notes: Iterable[str]) -> bool:
    """Whether a run's record notes say it ran over the script variants."""
    return VARIANTS_NOTE in notes


def extra_note(name: str) -> str:
    """The record note of a run over the one of ``EXTRA_SETS`` named, in place of the evaluation sets."""
    return f"{_EXTRA_NOTE}{_extra(name).rstrip('/')}"


def extra_of(notes: Iterable[str]) -> str | None:
    """The one of ``EXTRA_SETS`` a run's record notes say it ran over; None for a run over the evaluation sets."""
    notes = list(notes)
    return next((name for name in EXTRA_SETS if extra_note(name) in notes), None)


def variant_prompts(set: str, code: str = "*") -> str:
    """File of one set's prompts in one script variant in the evaluation dataset; ``*`` names every variant's."""
    return f"{_VARIANTS}prompts/{set}/{code}.jsonl"


def extra_prompts(name: str, set: str, code: str = "*") -> str:
    """File of one set's prompts under the one of ``EXTRA_SETS`` named; ``*`` names every file of the set."""
    return f"{_extra(name)}prompts/{set}/{code}.jsonl"


def responses_path(
    slug: str, set: str, language: str, mode: str, preamble: str = "keep", variants: bool | str = False
) -> str:
    """File of a checkpoint's responses to one set and language in the responses repo, under the run's tree.

    With ``variants``, ``language`` is the code of the prompt file and the folder is the set's own name.
    """
    folder = set if variants else _FOLDERS.get(set, set)
    return f"{tree(mode, preamble, variants)}responses/{slug}/{folder}/{language}.jsonl"


def labels_path(
    slug: str, language: str, mode: str, benign: bool = False, preamble: str = "keep", variants: bool | str = False
) -> str:
    """File of a checkpoint's label rows for one language in the evaluation dataset; benign rows sit apart."""
    return f"{tree(mode, preamble, variants)}labels/{slug}/{'benign/' if benign else ''}{language}.jsonl"


def judge_raw_path(
    slug: str, language: str, mode: str, preamble: str = "keep", variants: bool | str = False
) -> str:
    """File of the judge's unparsed outputs for one language, kept beside the responses."""
    return f"{tree(mode, preamble, variants)}judge_raw/{slug}/{language}.jsonl"


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
