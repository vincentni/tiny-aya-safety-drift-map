"""Publish the repos: scan every commit and card them all, then open them one by one, then list them."""

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Literal, NamedTuple

import httpx2
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects
from pydantic import Field, ValidationError

from driftmap import quant
from driftmap.core import ids, redact
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import AlphaScreen, Manifest, Record, RunRecord, Tables, dump, load
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.publish import cards, collections
from driftmap.repair.screen_gen import screen_path
from driftmap.scripts.drift import RECIPE, repo_names

SMOKE_SUFFIX = ids.repo_suffix("smoke")
# what a repo is renamed to when a clean copy of it is published in its place
WORK_SUFFIX = "-work"
CARD = "README.md"
# key, repo type, collection, needed for the checklist; a study with nothing to repair has no repaired model
ORDER = (
    ("eval", "dataset", "data", True),
    ("results", "dataset", "data", True),
    ("responses", "dataset", "data", True),
    ("adapter", "model", "models", True),
    ("adapter_nopre", "model", "models", False),
    (cards.MERGED_GGUF, "model", "models", False),
    ("repaired", "model", "models", False),
    (cards.REPAIRED_NOPRE, "model", "models", False),
    ("gguf", "model", "models", False),
)
# a smoke tree goes whole, those of the other conditions inside it too; the full trees lose their superseded labels
ROWS = ("smoke/*", *(f"{tree}labels/*/superseded/*" for tree in ("", *cards.OTHER_TREES)))
CHECKPOINT = ("last-checkpoint/*", "training_args.bin")
# removed by a repo's card commit; an adapter's training state is not part of the artifact
STALE = {**dict.fromkeys(cards.DATASETS, ROWS), **dict.fromkeys(cards.ADAPTERS, CHECKPOINT)}
WEIGHTS = ["adapter_model.bin", "pytorch_model*.bin"]
# pickles the trainer leaves beside the weights; an adapter is refused while one is left
PICKLES = (".bin", ".pt", ".pth")
# never fetched from a model repo; a dataset is read whole
BINARY = ["*.safetensors", "*.gguf", *WEIGHTS, "*.onnx"]
# what the cards are rendered from: file, the repo it is in, the record it holds; the tables of the standard
# fine-tune come before those of the aggressive one, each under the served template and with no safety preamble
CARD_INPUTS = (
    ("manifest.json", "eval", Manifest),
    *((name, "results", Tables) for pair in cards.TABLES.values() for name in pair),
    # the screen the repair of the aggressive fine-tune was chosen on
    (screen_path("full", cards.TRAINED[cards.ADAPTERS[1]], "aggressive"), "results", AlphaScreen),
)
# read again for the cards, or to see whether the head already holds what a run would write
KEPT = [*(name for name, _, _ in CARD_INPUTS), "runs/*.json", RECIPE, cards.PUBLICATION, CARD]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
RULES = {**redact.PATTERNS, "email": EMAIL}
CACHE = ".cache"
TEMP_PARTS = frozenset({"tmp", "__pycache__", ".ipynb_checkpoints"})
TEMP_SUFFIXES = (".tmp", ".lock", ".incomplete")
MAX_SHOWN = 20
ROTATE = "a new commit keeps the value in history; rotate the secret"
REBUILD = (
    "history cannot be edited: once the head is clean, publish with --fresh, which opens a new repo holding the head's "
    f"files and keeps the repo with the history private as <name>{WORK_SUFFIX}"
)
EXPOSED_WORK = "{repo} is public; make it private: its history is why it was set aside"
UNFINISHED = "; its files are in its working copy {work}: publish with --fresh to finish the clean copy"
# what a repo holds when it is created
INITIAL = frozenset({".gitattributes"})
TOKEN_RULES = frozenset({"hf_token", "aws_key"})
CHANGED = "{repo} changed after its scan; it stays as it is: run publish again"
# a failed transfer, a failed call, or a download the Hub could not serve
FAILURES = (httpx2.HTTPError, LocalEntryNotFoundError)

Echo = Callable[[str], None]
Problem = tuple[ExitCode, str]
# file, line or one of the kinds below, rule
Hit = tuple[str, int, str]
NAME, UNREADABLE, CACHED = 0, -1, -2
SCHEMA_PUBLICATION = "driftmap.publication/1"
# what a refusal calls a file a run would write
WRITTEN = {CARD: "card"}
# where the CLI keeps a scan between runs, under its local folder
SCAN_FILE = "publish-scan.json"
# how a stored verdict says which rules read the blob: those of a row folder, or those of any other file
ROW_KEY, FILE_KEY = "row", "file"


class Target(NamedTuple):
    key: str
    repo: str
    repo_type: str
    shelf: str
    required: bool


class State(NamedTuple):
    """A repo as the Hub reports it; ``sha`` is None when it does not exist."""

    sha: str | None = None
    private: bool | None = None
    gated: Any = None
    files: tuple[str, ...] = ()

    @property
    def exists(self) -> bool:
        return self.sha is not None

    @property
    def open(self) -> bool:
        return self.exists and self.private is False and self.gated is False


class Verdict(NamedTuple):
    hits: tuple[tuple[int, str], ...]
    text: bool


class _Stored(Record):
    """A scan as its file holds it; ``rules`` names the code whose verdicts these are."""

    rules: str
    # "<repo>@<commit>" -> (path, blob id) of every file
    trees: dict[str, list[tuple[str, str]]]
    # "<row or file>:<blob id>" -> hits, and whether the bytes are text
    blobs: dict[str, tuple[list[tuple[int, str]], bool]]
    kept: dict[str, str]


def _rules() -> str:
    """Hash of the code a verdict comes from: the scan rules and this module."""
    sources = (Path(redact.__file__), Path(__file__))
    return hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()


class Scans:
    """What a run has listed and read, to share between a plan and a publish: commits and blobs never change.

    With ``path`` it starts from what an earlier run saved there, so a run the Hub stopped goes on where it was;
    a file that does not parse, or was written by other scan code, is read as empty.
    """

    def __init__(self, path: Path | None = None) -> None:
        # (repo, commit) -> (path, blob id) of every file
        self.trees: dict[tuple[str, str], list[tuple[str, str]]] = {}
        # (in a row folder, blob id) -> what the scan found
        self.blobs: dict[tuple[bool, str], Verdict] = {}
        self.kept: dict[str, bytes] = {}
        self.path = path
        try:
            stored = load(_Stored, path) if path is not None else None
        except (OSError, ValidationError):
            stored = None
        if stored is None or stored.rules != _rules():
            return
        self.trees = {tuple(key.rpartition("@")[::2]): tree for key, tree in stored.trees.items()}
        self.blobs = {
            (key.startswith(ROW_KEY), key.partition(":")[2]): Verdict(tuple(hits), text)
            for key, (hits, text) in stored.blobs.items()
        }
        self.kept = {blob: text.encode() for blob, text in stored.kept.items()}

    def save(self) -> None:
        """Write the listings, the verdicts and the card inputs to the file, if there is one."""
        if self.path is None:
            return
        found = {blob for (_, blob), verdict in self.blobs.items() if verdict.hits}
        # a file with a finding is not copied: what the finding is of stays on the Hub
        kept = {blob: data for blob, data in self.kept.items() if blob not in found and _is_text(data)}
        stored = _Stored(
            rules=_rules(),
            trees={f"{repo}@{sha}": tree for (repo, sha), tree in self.trees.items()},
            blobs={
                f"{ROW_KEY if row else FILE_KEY}:{blob}": (list(verdict.hits), verdict.text)
                for (row, blob), verdict in self.blobs.items()
            },
            kept={blob: data.decode() for blob, data in kept.items()},
        )
        dump(stored, self.path)


class Copied(Record):
    repo: str
    revision: str


class Publication(Record):
    """For each repo published as a clean copy: the working repo, and the revision of it, that its files came from."""

    schema_: Literal["driftmap.publication/1"] = Field(alias="schema")
    copies: dict[str, Copied]


class Step(NamedTuple):
    target: Target
    # of the repo whose files are published: the working repo when a clean copy is made
    state: State
    card: str
    # the card, or a file that goes with it, is not at the head yet
    stale_card: bool
    deletes: tuple[str, ...]
    # files committed with the card, by path
    extra: tuple[tuple[str, bytes], ...] = ()
    # the working repo and revision a clean copy is made from; None when the repo itself is opened
    copy: RepoRef | None = None
    # the working repo still has the published name
    move: bool = False


class Plan(NamedTuple):
    """What a run would do; any problem stops it before the first write."""

    steps: tuple[Step, ...]
    absent: tuple[Target, ...]
    hidden: tuple[tuple[str, str, State], ...]
    shelves: dict[str, Any | None]
    problems: tuple[Problem, ...]
    # the problems a delete of the intermediates removes
    exposed: tuple[Problem, ...]
    # what to do about findings no new commit can remove
    advice: tuple[str, ...]
    fetched: int
    commits: int
    # the repos set aside by a clean copy, as they are now
    working: tuple[tuple[str, str, State], ...] = ()

    def refusal(self) -> str:
        """Every problem, one per line, then the advice."""
        return "\n".join([*(text for _, text in self.problems), *self.advice])

    def lines(self, delete: bool = False) -> list[str]:
        out = [f"history scan: {self.fetched} file fetches over {self.commits} commits"]
        for step in self.steps:
            target, shelf = step.target, self.shelves[step.target.shelf]
            listed = shelf is not None and collections.holds(shelf, target.repo, target.repo_type)
            skipped = _binary(step.state.files) if target.repo_type == "model" else []
            gone = f"{'leave out' if step.copy else 'delete'} {len(step.deletes)} file{'s' * (len(step.deletes) > 1)}"
            parts = [
                *([f"clean copy from {step.copy.repo}@{step.copy.revision[:12]}"] if step.copy else []),
                "public" if step.state.open else "private -> public" if step.state.private else "gated -> public",
                f"card: {'write' if step.stale_card else 'current'}",
                *([gone] if step.deletes else []),
                *([f"not scanned: {', '.join(skipped)}"] if skipped else []),
                f"collection: {'listed' if listed else 'add'}",
            ]
            out.append(f"{target.repo} ({target.repo_type}): {'; '.join(parts)}")
        out += [f"absent: {target.repo} ({target.repo_type})" for target in self.absent]
        verb = "would delete" if delete else "stays private"
        out += [f"{verb}: {repo} ({repo_type})" for repo, repo_type, _ in self.hidden]
        out += [f"working copy, stays private: {repo} ({repo_type})" for repo, repo_type, _ in self.working]
        return [*out, *(f"problem: {text}" for _, text in self.problems), *self.advice]


class Row(NamedTuple):
    repo: str
    exists: bool
    public: bool
    gated: str
    listed: bool
    card: bool
    license: str
    ok: bool


class Check(NamedTuple):
    """The public state against the checklist."""

    rows: tuple[Row, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def table(self) -> str:
        def shown(value: bool | str) -> str:
            return "yes" if value is True else "no" if value is False else value

        head = ("repo", "exists", "public", "gated", "in collection", "card", "license")
        lines = [head, *(tuple(shown(value) for value in row[:7]) for row in self.rows)]
        widths = [max(len(line[index]) for line in lines) for index in range(len(head))]
        text = ["  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)) for line in lines]
        return "\n".join([*map(str.rstrip, text), *(f"- {problem}" for problem in self.problems)])


def targets(settings: Settings) -> list[Target]:
    """The repos a publication makes public, in the order they are handled; a usage error on a name it must not open."""
    repos, seen = cards.repo_ids(settings), {}
    for key, *_ in ORDER:
        name = repos[key].partition("/")[2].casefold()
        if name.endswith(SMOKE_SUFFIX):
            raise DriftmapError(
                ExitCode.USAGE, f"{_setting(key)} = {name} ends with {SMOKE_SUFFIX}; a smoke repo is never published"
            )
        if name.endswith(WORK_SUFFIX):
            kept = "that ending names the working repo a clean copy sets aside"
            raise DriftmapError(ExitCode.USAGE, f"{_setting(key)} = {name} ends with {WORK_SUFFIX}; {kept}")
        if name in seen:
            raise DriftmapError(ExitCode.USAGE, f"{_setting(key)} and {_setting(seen[name])} name one repo")
        seen[name] = key
    return [Target(key, repos[key], *rest) for key, *rest in ORDER]


def _setting(key: str) -> str:
    """The setting a target takes its name from, as a refusal names it."""
    stripped = ids.repo_suffix("full", "strip")
    derived = {
        cards.ADAPTERS[1]: ("adapter", stripped),
        cards.MERGED_GGUF: ("adapter", quant.GGUF_SUFFIX),
        cards.REPAIRED_NOPRE: ("repaired", stripped),
    }
    return "repos.{} with {}".format(*derived[key]) if key in derived else f"repos.{key}"


def _merged(settings: Settings) -> list[str]:
    """The merged checkpoint of each training condition; neither is ever published."""
    return [repo_names(settings, "full", preamble)[1] for preamble in ids.PREAMBLES]


def intermediates(settings: Settings) -> list[tuple[str, str]]:
    """Repo id and type of the merged checkpoints and of every smoke repo; a usage error when one is also published."""
    owner, repos = settings.owner, settings.repos
    keys = ("adapter", "merged", "repaired", "gguf", "eval", "results", "responses")
    kinds = {key: "dataset" if key in cards.DATASETS else "model" for key in keys}
    smoke = [(f"{owner}/{getattr(repos, key)}{SMOKE_SUFFIX}", kinds[key]) for key in keys]
    # a smoke drift run with no safety preamble pushes to a pair of its own, a smoke quant run on the merged model too
    smoke += [(repo, "model") for repo in repo_names(settings, "smoke", "strip")]
    # and so does a smoke repair of that fine-tune
    smoke += [(f"{owner}/{repos.repaired}{ids.repo_suffix('smoke', 'strip')}", "model")]
    smoke += [(quant.merged_names(settings, "smoke")[1], "model")]
    hidden = [*((repo, "model") for repo in _merged(settings)), *smoke]
    published = {target.repo.casefold() for target in targets(settings)}
    for repo, _ in hidden:
        try:
            plain = ids.parse(repo) == RepoRef(repo)
        except ValueError:
            plain = False
        # a revision or file part would be cut off, and the repo before it deleted
        if not plain:
            raise DriftmapError(ExitCode.USAGE, f"{repo} is not a plain repo name")
        if repo.casefold() in published:
            raise DriftmapError(ExitCode.USAGE, f"{repo} is configured as a published and a private repo")
    return hidden


def _state(hub: Hub, repo: str, repo_type: str) -> State:
    read = hub.dataset_info if repo_type == "dataset" else hub.model_info
    try:
        info = read(repo)
    except RepositoryNotFoundError:
        return State()
    # the Hub answers for a renamed repo under its old name too: that is no repo of this name
    if (info.id or repo).casefold() != repo.casefold():
        return State()
    return State(info.sha, info.private, info.gated, tuple(sibling.rfilename for sibling in info.siblings or ()))


def _work(repo: str) -> str:
    """The name a repo is kept under once a clean copy has taken its own."""
    return f"{repo}{WORK_SUFFIX}"


def _working(settings: Settings, hub: Hub) -> dict[str, State]:
    """The working repo of each target, by key, as the Hub reports it."""
    return {target.key: _state(hub, _work(target.repo), target.repo_type) for target in targets(settings)}


def _gate(state: State) -> str:
    return "no" if state.gated is False else str(state.gated)


def _identity(settings: Settings, hub: Hub) -> None:
    name = hub.whoami().get("name")
    if name != settings.owner:
        raise DriftmapError(
            ExitCode.READINESS, f"the token belongs to {name}, not to {settings.owner}; log in as the owner"
        )


def _binary(files: tuple[str, ...]) -> list[str]:
    return list(filter_repo_objects(files, allow_patterns=BINARY))


def _stale(target: Target, files: tuple[str, ...]) -> list[str]:
    patterns = STALE.get(target.key)
    return list(filter_repo_objects(files, allow_patterns=list(patterns))) if patterns else []


def _training_state(target: Target, files: tuple[str, ...]) -> list[str]:
    if target.key not in cards.ADAPTERS:
        return []
    names = [path.rpartition("/")[2] for path in files]
    weights = set(filter_repo_objects(names, allow_patterns=WEIGHTS))
    return [path for path, name in zip(files, names, strict=True) if name.endswith(PICKLES) and name not in weights]


def _stray(files: tuple[str, ...]) -> list[str]:
    """Temp files at the head; a cache path is found in every commit instead."""

    def temporary(path: str) -> bool:
        parts = set(path.split("/"))
        return CACHE not in parts and (bool(TEMP_PARTS & parts) or path.endswith(TEMP_SUFFIXES))

    return [path for path in files if temporary(path)]


def _safe(name: str) -> str:
    """A file name fit to print: whatever a rule finds in it is replaced."""
    for pattern in RULES.values():
        name = pattern.sub("***", name)
    return name


def _is_text(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return b"\x00" not in data


def _cause(error: Exception) -> str:
    if not isinstance(error, HfHubHTTPError):
        return type(error).__name__
    status = error.response.status_code
    return f"{status}: the Hub is throttling, wait before the next run" if status == 429 else str(status)


@contextmanager
def _reading(repo: str) -> Iterator[None]:
    """A failed listing or fetch leaves as exit 7 naming the repo; it is never read as a clean scan."""
    try:
        yield
    except FAILURES as error:
        raise DriftmapError(ExitCode.PUSH_FAILED, f"reading {repo} failed ({_cause(error)})") from error


def _matching(paths: list[str], patterns: list[str] | tuple[str, ...]) -> set[str]:
    return set(filter_repo_objects(paths, allow_patterns=list(patterns)))


def _listed(hub: Hub, target: Target, head: str, scans: Scans, history: bool = True) -> list[str]:
    """The commits behind ``head``, newest first, each with its file list in ``scans``; without ``history`` the head."""
    found = hub.list_repo_commits(RepoRef(target.repo, head), repo_type=target.repo_type) if history else []
    shas = list(dict.fromkeys([head, *(commit.commit_id for commit in found)]))
    for sha in shas:
        if (target.repo, sha) not in scans.trees:
            entries = hub.list_repo_tree(RepoRef(target.repo, sha), repo_type=target.repo_type)
            scans.trees[target.repo, sha] = [
                (entry.path, entry.blob_id) for entry in entries if isinstance(entry, RepoFile)
            ]
    return shas


def _unread(
    target: Target, shas: list[str], scans: Scans, wanted: dict[tuple[bool, str], tuple[Target, str, str]]
) -> None:
    """Add to ``wanted`` one place to fetch each blob no scan in this run has read."""
    for sha in shas:
        tree = scans.trees[target.repo, sha]
        skipped = _matching([path for path, _ in tree], BINARY) if target.repo_type == "model" else set()
        for path, blob in tree:
            key = (redact.is_row(path.split("/")), blob)
            if path not in skipped and key not in scans.blobs:
                wanted.setdefault(key, (target, sha, path))


def _fetch(hub: Hub, target: Target, sha: str, path: str, scratch: Path) -> tuple[Verdict, bytes]:
    folder = Path(tempfile.mkdtemp(dir=scratch))
    try:
        local = Path(hub.hf_hub_download(RepoRef(target.repo, sha, path), folder, repo_type=target.repo_type))
        data = local.read_bytes()
        row = redact.is_row(path.split("/"))
        found = [(finding.line, finding.pattern) for finding in redact.scan([local], data_root=folder)]
        # a NUL marks bytes: random ones spell a short address often enough to refuse megabytes of optimizer state
        if not row and b"\x00" not in data:
            found += [(number, "email") for number in _addressed(data.decode("utf-8", errors="replace"))]
        return Verdict(tuple(sorted(found)), _is_text(data)), data
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _read(hub: Hub, wanted: dict[tuple[bool, str], tuple[Target, str, str]], scans: Scans, scratch: Path) -> None:
    for key, (target, sha, path) in wanted.items():
        scans.blobs[key], data = _fetch(hub, target, sha, path, scratch)
        if _matching([path], KEPT):
            scans.kept[key[1]] = data


class Found(NamedTuple):
    secrets: list[str]
    other: list[str]
    tokens: bool


def _found(target: Target, shas: list[str], scans: Scans, only: set[str] | None = None) -> Found:
    """What the scan holds against any commit, each named by the newest commit it is in; ``only`` limits the files."""
    held: dict[Hit, list[str]] = {}
    for sha in shas:
        tree = [(path, blob) for path, blob in scans.trees[target.repo, sha] if only is None or path in only]
        paths = [path for path, _ in tree]
        skipped = _matching(paths, BINARY) if target.repo_type == "model" else set()
        # an adapter's checkpoint pickles are bytes by nature and leave with its card
        tolerated = {path for path in _matching(paths, CHECKPOINT) if path.endswith(PICKLES)}
        for path, blob in tree:
            parts = tuple(path.split("/"))
            row = redact.is_row(parts)
            # the leading slash lets a name that starts at a home folder match; a row folder is named after a commit
            rules = {rule: pattern for rule, pattern in RULES.items() if not (row and rule == "account_id")}
            hits = [(path, NAME, rule) for rule, pattern in rules.items() if pattern.search(f"/{path}")]
            hits += [(path, CACHED, "")] if CACHE in parts else []
            if path not in skipped:
                verdict = scans.blobs[row, blob]
                hits += [(path, line, rule) for line, rule in verdict.hits]
                if not verdict.text and not (target.key in cards.ADAPTERS and path in tolerated):
                    hits.append((path, UNREADABLE, ""))
            for hit in hits:
                held.setdefault(hit, []).append(sha)
    secrets, other = [], []
    for (path, line, rule), within in sorted(held.items()):
        earlier = len(within) - 1
        more = f" (and {earlier} earlier commit{'s' * (earlier > 1)})" if earlier else ""
        spot = f"{target.repo}@{within[0][:12]}/{_safe(path)}"
        if line == CACHED:
            other.append(f"{spot}: a cache path{more}")
        elif line == UNREADABLE:
            other.append(f"{spot}: not text, so the scan cannot read it{more}")
        else:
            secrets.append(f"{spot}:{line}: {rule}{more}" if line else f"{spot}: {rule} in the file name{more}")
    tokens = any(rule in TOKEN_RULES for (_, line, rule) in held if line >= NAME)
    return Found(_capped(secrets, target.repo), _capped(other, target.repo), tokens)


def _capped(texts: list[str], repo: str) -> list[str]:
    more = len(texts) - MAX_SHOWN
    return texts if more <= 0 else [*texts[:MAX_SHOWN], f"{repo}: {more} more findings"]


def _advice(found: list[Found]) -> tuple[str, ...]:
    final = any(one.secrets or one.other for one in found)
    return (*([REBUILD] if final else []), *([ROTATE] if any(one.tokens for one in found) else []))


def _kept(hub: Hub, target: Target, head: str, scans: Scans, scratch: Path) -> dict[str, bytes]:
    """The head's card inputs by path; one the scan read under another name is fetched here."""
    tree = dict(scans.trees[target.repo, head])
    held = {}
    for path in sorted(_matching(list(tree), KEPT)):
        if tree[path] not in scans.kept:
            scans.kept[tree[path]] = _fetch(hub, target, head, path, scratch)[1]
        held[path] = scans.kept[tree[path]]
    return held


def _parsed[T](cls: type[T], files: dict[str, bytes], name: str) -> T | None:
    return cls.model_validate_json(files[name]) if name in files else None


def _recipe(files: dict[str, bytes]) -> dict[str, Any] | None:
    """The recipe a drift run left in its adapter repo; None when there is none or it is not a JSON object."""
    try:
        found = json.loads(files[RECIPE]) if RECIPE in files else None
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def _hidden(settings: Settings, hub: Hub) -> list[tuple[str, str, State]]:
    found = ((repo, repo_type, _state(hub, repo, repo_type)) for repo, repo_type in intermediates(settings))
    return [(repo, repo_type, state) for repo, repo_type, state in found if state.exists]


def plan(
    settings: Settings, hub: Hub, *, writeup: str | None = None, scans: Scans | None = None, fresh: bool = False
) -> Plan:
    """Read every repo, scan each commit and render each card; nothing on the Hub changes. Exit 4 above the ceiling.

    With ``fresh`` a repo whose history holds a finding, and whose head would not, is planned as a clean copy: a new
    repo of its name holding the head's files, the repo with the history kept private under ``-work``.
    """
    scans = scans if scans is not None else Scans()
    try:
        return _drawn(settings, hub, writeup, scans, fresh)
    finally:
        # what was read stays read, also when the Hub stopped the run
        scans.save()


def _unfilled(state: State, work: State) -> bool:
    """A clean copy stopped after the rename or after the creation: the name holds no file of the working repo."""
    return work.exists and not set(state.files) - INITIAL


def _lacking(target: Target, state: State, work: State) -> str | None:
    """Why a target cannot be published as it is: a required one is absent, or a clean copy was never filled."""
    if _unfilled(state, work):
        what = "is empty" if state.exists else "does not exist"
        return f"{target.repo} {what}{UNFINISHED.format(work=_work(target.repo))}"
    return f"{target.repo} does not exist" if target.required and not state.exists else None


def _drawn(settings: Settings, hub: Hub, writeup: str | None, scans: Scans, fresh: bool) -> Plan:
    wanted = targets(settings)
    hidden = _hidden(settings, hub)
    named = {target.key: _state(hub, target.repo, target.repo_type) for target in wanted}
    works = _working(settings, hub)
    # an unfinished copy is read from its working repo: the files to publish are there
    resumed = {key for key, work in works.items() if fresh and _unfilled(named[key], work)}
    aside = {target.key: target._replace(repo=_work(target.repo)) for target in wanted}
    reads = {target.key: aside[target.key] if target.key in resumed else target for target in wanted}
    states = named | {key: works[key] for key in resumed}
    present = [target for target in wanted if states[target.key].exists]
    absent = tuple(target for target in wanted if not states[target.key].exists)
    advice = "delete it with --delete-intermediates, or make it private"
    exposed = [
        (ExitCode.INTEGRITY, f"{repo} is public; {advice}") for repo, _, state in hidden if state.private is not True
    ]
    working = [(read.repo, read.repo_type, works[key]) for key, read in aside.items() if works[key].exists]
    shelves = collections.find(settings, hub)
    secrets: list[Problem] = []
    lacking = (_lacking(target, named[target.key], works[target.key]) for target in wanted if target.key not in resumed)
    other = [(ExitCode.INTEGRITY, lack) for lack in lacking if lack]
    other += [
        (ExitCode.INTEGRITY, EXPOSED_WORK.format(repo=repo)) for repo, _, state in working if state.private is not True
    ]
    other += _shelf_problems(settings, shelves)

    shas, unread, needs = {}, {}, {}
    for target in present:
        read = reads[target.key]
        with _reading(read.repo):
            # the history of a repo already set aside stays behind
            shas[target.key] = _listed(hub, read, states[target.key].sha, scans, target.key not in resumed)
        before = len(unread)
        _unread(read, shas[target.key], scans, unread)
        needs[read.repo] = len(unread) - before
    ceiling = settings.publish.max_file_fetches
    if len(unread) > ceiling:
        most = ", ".join(f"{repo} {count}" for repo, count in needs.items() if count)
        raise DriftmapError(
            ExitCode.GUARD_REFUSAL,
            f"the history scan needs {len(unread)} file fetches ({most}), above publish.max_file_fetches = {ceiling}; "
            "nothing was fetched: raise the value in driftmap.toml, or rebuild the longest repo from its head",
        )
    steps, found = [], []
    stale = {target.key: _stale(target, states[target.key].files) for target in present}
    copies: dict[str, RepoRef] = {}
    with tempfile.TemporaryDirectory(prefix="driftmap-publish-") as scratch:
        for key, (target, sha, path) in unread.items():
            with _reading(target.repo):
                _read(hub, {key: (target, sha, path)}, scans, Path(scratch))
        files = {}
        for target in present:
            key, read, state = target.key, reads[target.key], states[target.key]
            with _reading(read.repo):
                files[key] = _kept(hub, read, state.sha, scans, Path(scratch))
            held = _found(read, shas[key], scans)
            if key in resumed or (fresh and (held.secrets or held.other)):
                # what a new repo would hold: the head without the files its card commit drops
                copied = _found(read, [state.sha], scans, set(state.files) - set(stale[key]))
                if key not in resumed and works[key].exists:
                    taken = f"{_work(target.repo)} exists already; {target.repo} cannot be set aside under that name"
                    other.append((ExitCode.INTEGRITY, taken))
                elif state.private is not True:
                    # set aside it would still be readable, history and all
                    shown = f"{target.repo} is not private; make it private before a clean copy is made of it"
                    other.append((ExitCode.INTEGRITY, shown))
                elif not (copied.secrets or copied.other):
                    copies[key] = RepoRef(_work(target.repo), state.sha)
                    # the value stays in the repo set aside: nothing to refuse, still something to rotate
                    held = Found([], [], held.tokens)
                elif key in resumed:
                    # the files a copy leaves out stand against nothing
                    held = copied
            found.append(held)
            secrets += [(ExitCode.REDACTION, text) for text in held.secrets]
            other += [(ExitCode.INTEGRITY, text) for text in held.other]
        parsed = []
        for name, key, cls in CARD_INPUTS:
            try:
                parsed.append(_parsed(cls, files.get(key, {}), name))
            except ValidationError:
                parsed.append(None)
                other.append((ExitCode.INTEGRITY, f"{name} on the Hub does not parse; build and upload it again"))
        manifest, tables, stripped, aggressive, aggressive_stripped, aggressive_screen = parsed
        records = []
        for name, data in sorted(files.get("results", {}).items()):
            try:
                records += [RunRecord.model_validate_json(data)] if name.startswith("runs/") else []
            except ValidationError:
                continue
        # a copy an earlier run made is named by the working repo it left
        origins = {
            target.key: copies.get(target.key) or RepoRef(_work(target.repo), works[target.key].sha)
            for target in present
            if target.key in copies or works[target.key].exists
        }
        mapped = {
            target.repo: Copied(repo=origins[target.key].repo, revision=origins[target.key].revision)
            for target in present
            if target.key in origins
        }
        publication = Publication(schema=SCHEMA_PUBLICATION, copies=mapped).model_dump_json(indent=2) + "\n"
        extras = {"results": ((cards.PUBLICATION, publication.encode()),)} if mapped else {}
        rendered = cards.render_cards(
            settings,
            manifest=manifest,
            tables=tables,
            records=records,
            writeup=writeup,
            nopreamble=stripped,
            recipes={key: _recipe(files[key]) for key in cards.ADAPTERS if key in files},
            # a dataset card counts the files its own commit leaves
            files={
                key: sorted((set(states[key].files) - set(stale[key])) | {name for name, _ in extras.get(key, ())})
                for key in cards.DATASETS
                if key in stale
            },
            copies=origins,
            aggressive=aggressive,
            aggressive_nopreamble=aggressive_stripped,
            aggressive_screen=aggressive_screen,
        )
        for target in present:
            state, card, extra = states[target.key], rendered[target.repo], extras.get(target.key, ())
            staged = Path(scratch, "cards", target.key, CARD)
            staged.parent.mkdir(parents=True)
            written = {CARD: card.encode(), **dict(extra)}
            for name, data in written.items():
                staged.with_name(name).write_bytes(data)
                addresses = ((name, number, "email") for number in _addressed(data.decode()))
                hits = [*redact.scan([staged.with_name(name)]), *addresses]
                secrets += [
                    (ExitCode.REDACTION, f"{target.repo}/{name}:{line}: {rule} in the new {WRITTEN.get(name, 'file')}")
                    for _, line, rule in hits
                ]
            deletes = stale[target.key]
            gaps = {
                "holds cache or temp files": [path for path in _stray(state.files) if path not in deletes],
                "holds training state": [path for path in _training_state(target, state.files) if path not in deletes],
            }
            other += [
                (ExitCode.INTEGRITY, f"{target.repo} {gap}: {', '.join(map(_safe, names))}")
                for gap, names in gaps.items()
                if names
            ]
            lacks = cards.missing(target.key, card, settings, manifest)
            if lacks:
                other.append((ExitCode.INTEGRITY, f"{target.repo} cannot get a complete card: {'; '.join(lacks)}"))
            copy = copies.get(target.key)
            current = copy is None and all(files[target.key].get(name) == data for name, data in written.items())
            moved = copy is not None and target.key not in resumed
            steps.append(Step(target, state, card, not current, tuple(deletes), extra, copy, moved))
    commits = sum(len(found) for found in shas.values())
    problems = (*exposed, *secrets, *other)
    return Plan(
        tuple(steps),
        absent,
        tuple(hidden),
        shelves,
        problems,
        tuple(exposed),
        _advice(found),
        len(unread),
        commits,
        tuple(working),
    )


def _addressed(text: str) -> list[int]:
    """Line numbers, one per address in the text."""
    return [number for number, line in enumerate(text.split("\n"), start=1) for _ in EMAIL.finditer(line)]


def _shelf_problems(settings: Settings, shelves: dict[str, Any | None]) -> list[Problem]:
    merged = {repo.casefold() for repo in _merged(settings)}
    problems = []
    for shelf, found in shelves.items():
        name = getattr(settings.repos, collections.SHELVES[shelf])
        if found is None:
            problems.append((ExitCode.INTEGRITY, f"collection {name} was not found under {settings.owner}"))
            continue
        if found.private is not False:
            problems.append((ExitCode.INTEGRITY, f"collection {found.slug} is private"))
        problems += [
            (ExitCode.INTEGRITY, f"{item.item_id} is in a collection")
            for item in found.items
            if item.item_id.casefold().endswith((SMOKE_SUFFIX, WORK_SUFFIX)) or item.item_id.casefold() in merged
        ]
    return problems


def _public_now(settings: Settings, hub: Hub) -> str:
    try:
        found = [t.repo for t in targets(settings) if _state(hub, t.repo, t.repo_type).private is False]
    except FAILURES:
        return "which repos are public could not be read"
    return f"public now: {', '.join(found) or 'none'}"


@contextmanager
def _reported(action: str, settings: Settings, hub: Hub, resume: str) -> Iterator[None]:
    """Turn a failed Hub call, write or read-back, into exit 7 naming what is public and how to go on."""
    try:
        yield
    except FAILURES as error:
        text = f"{action} failed ({_cause(error)}); {_public_now(settings, hub)}\n{resume}"
        raise DriftmapError(ExitCode.PUSH_FAILED, text) from error


def _copy(hub: Hub, step: Step, folder: Path) -> frozenset[str]:
    """Set the working repo aside and commit the planned files to a new repo of its name; returns what that repo
    held when it was created.

    Nothing is uploaded unless the name is a new private repo: the Hub answers for a renamed repo under its old
    name, and a commit or a settings change sent there would reach the repo with the history.
    """
    target, work = step.target, step.copy
    if step.move:
        if _state(hub, target.repo, target.repo_type).sha != work.revision:
            raise DriftmapError(ExitCode.INTEGRITY, CHANGED.format(repo=target.repo))
        hub.move_repo(target.repo, work.repo, repo_type=target.repo_type)
    aside = _state(hub, work.repo, target.repo_type)
    if aside.sha != work.revision or aside.private is not True:
        at = f"{work.repo} is not the private working copy at {work.revision[:12]}"
        raise DriftmapError(ExitCode.INTEGRITY, f"{at}; nothing was created: make it private, then run publish again")
    if not _state(hub, target.repo, target.repo_type).exists:
        hub.create_repo(target.repo, repo_type=target.repo_type, private=True, exist_ok=False)
    new = _state(hub, target.repo, target.repo_type)
    if not new.exists or new.private is not True or new.sha == work.revision or set(new.files) - INITIAL:
        left = f"nothing was uploaded: its files are in {work.repo}, which stays private"
        raise DriftmapError(ExitCode.INTEGRITY, f"{target.repo} is not a new, empty, private repo; {left}")
    kept = sorted(set(step.state.files) - set(step.deletes) - {CARD})
    if kept:
        hub.snapshot_download(work, folder, repo_type=target.repo_type, allow_patterns=kept)
    # the library keeps its own notes of a download under the cache folder
    held = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if {path for path in held if path.split("/")[0] != CACHE} != set(kept):
        other = f"the download of {ids.unparse(work)} holds other files than its listing"
        raise DriftmapError(ExitCode.INTEGRITY, f"{other}; nothing was uploaded")
    for name, data in ((CARD, step.card.encode()), *step.extra):
        (folder / name).write_bytes(data)
    hub.upload_folder(
        target.repo,
        folder,
        repo_type=target.repo_type,
        ignore_patterns=[f"{CACHE}/*"],
        # refused by the Hub unless the name is the repo just created
        parent_commit=new.sha,
        commit_message=f"publish the files of {work.repo}@{work.revision[:12]}",
    )
    return frozenset(new.files)


def _verified(hub: Hub, step: Step, scratch: Path, scans: Scans, initial: frozenset[str] = frozenset()) -> str:
    """The head a repo may be opened at: its card commit on top of the planned files, every new commit scanned.

    ``initial`` names what a repo made for a clean copy held before that commit.
    """
    target = step.target
    now = _state(hub, target.repo, target.repo_type)
    shas = _listed(hub, target, now.sha, scans)
    unread: dict[tuple[bool, str], tuple[Target, str, str]] = {}
    _unread(target, shas, scans, unread)
    _read(hub, unread, scans, scratch)
    found = _found(target, shas, scans)
    if found.secrets or found.other:
        code = ExitCode.REDACTION if found.secrets else ExitCode.INTEGRITY
        raise DriftmapError(code, "\n".join([*found.secrets, *found.other, *_advice([found])]))
    written = {CARD: step.card.encode(), **dict(step.extra)}
    planned = (set(step.state.files) - set(step.deletes)) | set(written) | initial
    at_head = _kept(hub, target, now.sha, scans, scratch)
    if set(now.files) != planned or any(at_head.get(name) != data for name, data in written.items()):
        raise DriftmapError(ExitCode.INTEGRITY, CHANGED.format(repo=target.repo))
    return now.sha


def publish(
    settings: Settings,
    hub: Hub,
    *,
    dry_run: bool = False,
    delete: bool = False,
    writeup: str | None = None,
    resume: str | None = None,
    scans: Scans | None = None,
    fresh: bool = False,
    echo: Echo = print,
) -> Check | None:
    """Card, open and list every repo; the final check, or None after a dry run, which refuses where the run would.

    ``scans`` reuses what an earlier plan read. With ``fresh`` a repo whose history alone holds a finding is
    published as a clean copy: it is renamed to ``<name>-work`` and stays private, a new repo of its name takes the
    head's files in one commit, and ``publication.json`` in the results repo records where each copy came from.
    """
    _identity(settings, hub)
    scans = scans if scans is not None else Scans()
    drawn = plan(settings, hub, writeup=writeup, scans=scans, fresh=fresh)
    # a paid job rebuilds the merged checkpoint, so it goes only when the run can go on
    if delete and not dry_run and set(drawn.problems) <= set(drawn.exposed):
        delete_intermediates(settings, hub, echo=echo)
        drawn = plan(settings, hub, writeup=writeup, scans=scans, fresh=fresh)
    for line in drawn.lines(delete and dry_run):
        echo(line)
    # a dry run deletes nothing, so what its delete would clear does not stand against it
    cleared = dry_run and delete and set(drawn.problems) <= set(drawn.exposed)
    if drawn.problems and not cleared:
        raise DriftmapError(drawn.problems[0][0], drawn.refusal())
    if dry_run:
        return None
    given = f"{', delete=True' * delete}{', fresh=True' * fresh}{f', writeup={writeup!r}' if writeup else ''}"
    resume = resume or f"resume: run publish(settings, hub{given}) again; finished steps are skipped"

    def reported(action: str) -> AbstractContextManager[None]:
        return _reported(action, settings, hub, resume)

    heads, created = {}, {}
    with tempfile.TemporaryDirectory(prefix="driftmap-publish-") as scratch:
        for step in drawn.steps:
            target = step.target
            if not (step.stale_card or step.deletes):
                continue
            staged = Path(scratch, target.key)
            staged.mkdir()
            if step.copy:
                with reported(f"the clean copy of {target.repo}"):
                    created[target.key] = _copy(hub, step, staged)
                origin = f"{step.copy.repo}@{step.copy.revision[:12]}"
                echo(f"clean copy: {target.repo} ({target.repo_type}) from {origin}")
                continue
            for name, data in ((CARD, step.card.encode()), *step.extra):
                (staged / name).write_bytes(data)
            with reported(f"the card upload to {target.repo}"):
                hub.upload_folder(
                    target.repo,
                    staged,
                    repo_type=target.repo_type,
                    delete_patterns=list(STALE[target.key]) if step.deletes else None,
                    # refused by the Hub when anything landed after the scan
                    parent_commit=step.state.sha,
                    commit_message="add the card",
                )
            echo(f"card: {target.repo}")
        for step in drawn.steps:
            with reported(f"reading {step.target.repo} after its card"):
                initial = created.get(step.target.key, frozenset())
                heads[step.target.key] = _verified(hub, step, Path(scratch), scans, initial)

    def unchanged(step: Step) -> State:
        now = _state(hub, step.target.repo, step.target.repo_type)
        if now.sha != heads[step.target.key]:
            raise DriftmapError(ExitCode.INTEGRITY, CHANGED.format(repo=step.target.repo))
        return now

    for step in drawn.steps:
        with reported(f"reading {step.target.repo} before it is opened"):
            unchanged(step)

    def closed(target: Target, code: ExitCode, why: str) -> DriftmapError:
        """Make a repo private again; the error to raise says whether that worked."""
        try:
            hub.update_repo_settings(target.repo, private=True, repo_type=target.repo_type)
        except FAILURES as error:
            left = f"closing it failed ({_cause(error)}): it is PUBLIC, make it private by hand"
            return DriftmapError(ExitCode.PUSH_FAILED, f"{target.repo} {why}, and {left}\n{resume}")
        return DriftmapError(code, f"{target.repo} {why}; it is private again: run publish again")

    opened = []
    for step in drawn.steps:
        target = step.target
        with reported(f"opening {target.repo}"):
            if unchanged(step).open:
                echo(f"already public: {target.repo} ({target.repo_type})")
                continue
            hub.update_repo_settings(target.repo, private=False, gated=False, repo_type=target.repo_type)
            try:
                after = _state(hub, target.repo, target.repo_type)
            except FAILURES as error:
                # unread, it may hold a commit no scan has seen
                why = f"could not be read after it was opened ({_cause(error)})"
                raise closed(target, ExitCode.PUSH_FAILED, why) from error
            if after.sha != heads[target.key]:
                raise closed(target, ExitCode.INTEGRITY, "changed while it was opened")
            if not after.open:
                raise DriftmapError(ExitCode.UNEXPECTED, f"{target.repo} is still private or gated after the change")
        opened.append(target)
        echo(f"public: {target.repo} ({target.repo_type})")

    for step in drawn.steps:
        shelf = drawn.shelves[step.target.shelf]
        with reported(f"listing {step.target.repo} in {shelf.slug}"):
            added = collections.add(hub, shelf, step.target.repo, step.target.repo_type)
        if added:
            echo(f"listed: {step.target.repo} in {shelf.slug}")

    for step in drawn.steps:
        target = step.target
        with reported(f"reading {target.repo} before the final check"):
            if _state(hub, target.repo, target.repo_type).sha == heads[target.key]:
                continue
            if target in opened:
                raise closed(target, ExitCode.INTEGRITY, "changed after it was opened")
            raise DriftmapError(ExitCode.INTEGRITY, CHANGED.format(repo=target.repo))
    with reported("the final check"):
        result = check(settings, hub)
    echo(result.table())
    if not result.ok:
        raise DriftmapError(ExitCode.UNEXPECTED, "published, but the checklist is not met:\n" + result.table())
    return result


def check(settings: Settings, hub: Hub) -> Check:
    """Compare the Hub with the checklist; reads only. ``ok`` needs every required repo public, open and carded."""
    wanted = targets(settings)
    shelves = collections.find(settings, hub)
    hidden = _hidden(settings, hub)
    problems = [text for _, text in _shelf_problems(settings, shelves)]
    rows = []
    with tempfile.TemporaryDirectory(prefix="driftmap-check-") as scratch:

        def fetch(target: Target, state: State, name: str) -> Path | None:
            if name not in state.files:
                return None
            folder = Path(scratch, f"{target.key}-{name}")
            return Path(hub.hf_hub_download(RepoRef(target.repo, state.sha, name), folder, repo_type=target.repo_type))

        states = {target.key: _state(hub, target.repo, target.repo_type) for target in wanted}
        works = _working(settings, hub)
        manifest = None
        if states["eval"].exists and (path := fetch(wanted[0], states["eval"], "manifest.json")):
            try:
                manifest = load(Manifest, path)
            except ValidationError:
                problems.append(f"{wanted[0].repo} has a manifest.json that does not parse")
        for target in wanted:
            state, shelf = states[target.key], shelves[target.shelf]
            lack = _lacking(target, state, works[target.key])
            problems += [lack] if lack else []
            if not state.exists:
                rows.append(Row(target.repo, False, False, "-", False, False, "-", lack is None))
                continue
            path = fetch(target, state, CARD)
            text = path.read_text(encoding="utf-8") if path else None
            lacks = cards.missing(target.key, text, settings, manifest)
            listed = shelf is not None and collections.holds(shelf, target.repo, target.repo_type)
            stale = _stale(target, state.files)
            pickles = [path for path in _training_state(target, state.files) if path not in stale]
            gaps = {
                "is private": state.private is not False,
                f"is gated ({state.gated})": state.gated is not False,
                "is not in the collection": not listed,
                f"lacks: {'; '.join(lacks)}": bool(lacks),
                "still holds smoke, superseded or checkpoint files": bool(stale),
                f"holds training state: {', '.join(pickles)}": bool(pickles),
                "holds cache or temp files": bool(_stray(state.files)),
            }
            found = [f"{target.repo} {gap}" for gap, there in gaps.items() if there]
            problems += found
            licence = cards.license_field(text)
            public = state.private is False
            rows.append(Row(target.repo, True, public, _gate(state), listed, not lacks, licence, not found))
    for repo, repo_type, state in hidden:
        public = state.private is not True
        problems += [f"{repo} is public; it must stay private or be deleted"] if public else []
        listed = any(found is not None and collections.holds(found, repo, repo_type) for found in shelves.values())
        rows.append(Row(repo, True, public, _gate(state), listed, False, "-", not (public or listed)))
    for target in wanted:
        repo, state = _work(target.repo), works[target.key]
        if not state.exists:
            continue
        public = state.private is not True
        problems += [EXPOSED_WORK.format(repo=repo)] if public else []
        held = (found for found in shelves.values() if found is not None)
        listed = any(collections.holds(found, repo, target.repo_type) for found in held)
        rows.append(Row(repo, True, public, _gate(state), listed, False, "-", not (public or listed)))
    return Check(tuple(rows), tuple(problems))


def delete_intermediates(settings: Settings, hub: Hub, *, echo: Echo = print) -> list[str]:
    """Delete the merged checkpoints and every smoke repo for good, as the owner; returns the repos deleted."""
    _identity(settings, hub)
    gone = []
    for repo, repo_type in intermediates(settings):
        if not _state(hub, repo, repo_type).exists:
            continue
        hub.delete_repo(repo, repo_type=repo_type)
        if _state(hub, repo, repo_type).exists:
            raise DriftmapError(ExitCode.UNEXPECTED, f"{repo} still exists after the delete")
        gone.append(repo)
        echo(f"deleted: {repo}")
    return gone
