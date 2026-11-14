"""Publish the repos: scan every commit and card them all, then open them one by one, then list them."""

import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import httpx2
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects
from pydantic import ValidationError

from driftmap.core import ids, redact
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import Manifest, RunRecord, Tables, load
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.publish import cards, collections
from driftmap.scripts.drift import repo_names

SMOKE_SUFFIX = ids.repo_suffix("smoke")
CARD = "README.md"
# key, repo type, collection, needed for the checklist
ORDER = (
    ("eval", "dataset", "data", True),
    ("results", "dataset", "data", True),
    ("responses", "dataset", "data", True),
    ("adapter", "model", "models", True),
    ("adapter_nopre", "model", "models", False),
    ("repaired", "model", "models", False),
    ("gguf", "model", "models", False),
)
# a smoke tree goes whole, the one with no safety preamble inside it too; the full trees lose their superseded labels
ROWS = ("smoke/*", "labels/*/superseded/*", f"{cards.TREE}labels/*/superseded/*")
CHECKPOINT = ("last-checkpoint/*", "training_args.bin")
# removed by a repo's card commit; an adapter's training state is not part of the artifact
STALE = {**dict.fromkeys(cards.DATASETS, ROWS), **dict.fromkeys(cards.ADAPTERS, CHECKPOINT)}
WEIGHTS = ["adapter_model.bin", "pytorch_model*.bin"]
# pickles the trainer leaves beside the weights; an adapter is refused while one is left
PICKLES = (".bin", ".pt", ".pth")
# never fetched from a model repo; a dataset is read whole
BINARY = ["*.safetensors", "*.gguf", *WEIGHTS, "*.onnx"]
# where `table build --preamble strip` uploads its table
STRIPPED_TABLES = f"{cards.TREE}tables/tables-nopreamble.json"
# what the cards are rendered from: file, the repo it is in, the record it holds
CARD_INPUTS = (
    ("manifest.json", "eval", Manifest),
    ("tables/tables.json", "results", Tables),
    (STRIPPED_TABLES, "results", Tables),
)
# read again for the cards
KEPT = [*(name for name, _, _ in CARD_INPUTS), "runs/*.json", CARD]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
RULES = {**redact.PATTERNS, "email": EMAIL}
CACHE = ".cache"
TEMP_PARTS = frozenset({"tmp", "__pycache__", ".ipynb_checkpoints"})
TEMP_SUFFIXES = (".tmp", ".lock", ".incomplete")
MAX_SHOWN = 20
ROTATE = "a new commit keeps the value in history; rotate the secret"
REBUILD = "history cannot be edited: once the head is clean, rebuild the repo from its head as a new repo and re-pin it"
TOKEN_RULES = frozenset({"hf_token", "aws_key"})
CHANGED = "{repo} changed after its scan; it stays as it is: run publish again"
# a failed transfer, a failed call, or a download the Hub could not serve
FAILURES = (httpx2.HTTPError, LocalEntryNotFoundError)

Echo = Callable[[str], None]
Problem = tuple[ExitCode, str]
# file, line or one of the kinds below, rule
Hit = tuple[str, int, str]
NAME, UNREADABLE, CACHED = 0, -1, -2


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


class Scans:
    """What a run has listed and read, to share between a plan and a publish: commits and blobs never change."""

    def __init__(self) -> None:
        # (repo, commit) -> (path, blob id) of every file
        self.trees: dict[tuple[str, str], list[tuple[str, str]]] = {}
        # (in a row folder, blob id) -> what the scan found
        self.blobs: dict[tuple[bool, str], Verdict] = {}
        self.kept: dict[str, bytes] = {}


class Step(NamedTuple):
    target: Target
    state: State
    card: str
    stale_card: bool
    deletes: tuple[str, ...]


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

    def refusal(self) -> str:
        """Every problem, one per line, then the advice."""
        return "\n".join([*(text for _, text in self.problems), *self.advice])

    def lines(self, delete: bool = False) -> list[str]:
        out = [f"history scan: {self.fetched} file fetches over {self.commits} commits"]
        for step in self.steps:
            target, shelf = step.target, self.shelves[step.target.shelf]
            listed = shelf is not None and collections.holds(shelf, target.repo, target.repo_type)
            skipped = _binary(step.state.files) if target.repo_type == "model" else []
            parts = [
                "public" if step.state.open else "private -> public" if step.state.private else "gated -> public",
                f"card: {'write' if step.stale_card else 'current'}",
                *([f"delete {len(step.deletes)} file{'s' * (len(step.deletes) > 1)}"] if step.deletes else []),
                *([f"not scanned: {', '.join(skipped)}"] if skipped else []),
                f"collection: {'listed' if listed else 'add'}",
            ]
            out.append(f"{target.repo} ({target.repo_type}): {'; '.join(parts)}")
        out += [f"absent: {target.repo} ({target.repo_type})" for target in self.absent]
        verb = "would delete" if delete else "stays private"
        out += [f"{verb}: {repo} ({repo_type})" for repo, repo_type, _ in self.hidden]
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
        if name in seen:
            raise DriftmapError(ExitCode.USAGE, f"{_setting(key)} and {_setting(seen[name])} name one repo")
        seen[name] = key
    return [Target(key, repos[key], *rest) for key, *rest in ORDER]


def _setting(key: str) -> str:
    """The setting a target takes its name from, as a refusal names it."""
    derived = key == cards.ADAPTERS[1]
    return f"repos.adapter with {ids.repo_suffix('full', 'strip')}" if derived else f"repos.{key}"


def _merged(settings: Settings) -> list[str]:
    """The merged checkpoint of each training condition; neither is ever published."""
    return [repo_names(settings, "full", preamble)[1] for preamble in ids.PREAMBLES]


def intermediates(settings: Settings) -> list[tuple[str, str]]:
    """Repo id and type of the merged checkpoints and of every smoke repo; a usage error when one is also published."""
    owner, repos = settings.owner, settings.repos
    keys = ("adapter", "merged", "repaired", "gguf", "eval", "results", "responses")
    kinds = {key: "dataset" if key in cards.DATASETS else "model" for key in keys}
    smoke = [(f"{owner}/{getattr(repos, key)}{SMOKE_SUFFIX}", kinds[key]) for key in keys]
    # a smoke drift run with no safety preamble pushes to a pair of its own
    smoke += [(repo, "model") for repo in repo_names(settings, "smoke", "strip")]
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
    return State(info.sha, info.private, info.gated, tuple(sibling.rfilename for sibling in info.siblings or ()))


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


def _listed(hub: Hub, target: Target, head: str, scans: Scans) -> list[str]:
    """The commits behind ``head``, newest first, each with its file list in ``scans``."""
    found = hub.list_repo_commits(RepoRef(target.repo, head), repo_type=target.repo_type)
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


def _found(target: Target, shas: list[str], scans: Scans) -> Found:
    """What the scan holds against any commit, each named by the newest commit it is in."""
    held: dict[Hit, list[str]] = {}
    for sha in shas:
        tree = scans.trees[target.repo, sha]
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


def _hidden(settings: Settings, hub: Hub) -> list[tuple[str, str, State]]:
    found = ((repo, repo_type, _state(hub, repo, repo_type)) for repo, repo_type in intermediates(settings))
    return [(repo, repo_type, state) for repo, repo_type, state in found if state.exists]


def plan(settings: Settings, hub: Hub, *, writeup: str | None = None, scans: Scans | None = None) -> Plan:
    """Read every repo, scan each commit and render each card; nothing on the Hub changes. Exit 4 above the ceiling."""
    scans = scans if scans is not None else Scans()
    wanted = targets(settings)
    hidden = _hidden(settings, hub)
    states = {target.key: _state(hub, target.repo, target.repo_type) for target in wanted}
    present = [target for target in wanted if states[target.key].exists]
    absent = tuple(target for target in wanted if not states[target.key].exists)
    advice = "delete it with --delete-intermediates, or make it private"
    exposed = [
        (ExitCode.INTEGRITY, f"{repo} is public; {advice}") for repo, _, state in hidden if state.private is not True
    ]
    shelves = collections.find(settings, hub)
    secrets: list[Problem] = []
    other = [(ExitCode.INTEGRITY, f"{target.repo} does not exist") for target in absent if target.required]
    other += _shelf_problems(settings, shelves)

    shas, unread, needs = {}, {}, {}
    for target in present:
        with _reading(target.repo):
            shas[target.key] = _listed(hub, target, states[target.key].sha, scans)
        before = len(unread)
        _unread(target, shas[target.key], scans, unread)
        needs[target.repo] = len(unread) - before
    ceiling = settings.publish.max_file_fetches
    if len(unread) > ceiling:
        most = ", ".join(f"{repo} {count}" for repo, count in needs.items() if count)
        raise DriftmapError(
            ExitCode.GUARD_REFUSAL,
            f"the history scan needs {len(unread)} file fetches ({most}), above publish.max_file_fetches = {ceiling}; "
            "nothing was fetched: raise the value in driftmap.toml, or rebuild the longest repo from its head",
        )
    steps, found = [], []
    with tempfile.TemporaryDirectory(prefix="driftmap-publish-") as scratch:
        for key, (target, sha, path) in unread.items():
            with _reading(target.repo):
                _read(hub, {key: (target, sha, path)}, scans, Path(scratch))
        files = {}
        for target in present:
            with _reading(target.repo):
                files[target.key] = _kept(hub, target, states[target.key].sha, scans, Path(scratch))
            found.append(_found(target, shas[target.key], scans))
            secrets += [(ExitCode.REDACTION, text) for text in found[-1].secrets]
            other += [(ExitCode.INTEGRITY, text) for text in found[-1].other]
        parsed = []
        for name, key, cls in CARD_INPUTS:
            try:
                parsed.append(_parsed(cls, files.get(key, {}), name))
            except ValidationError:
                parsed.append(None)
                other.append((ExitCode.INTEGRITY, f"{name} on the Hub does not parse; build and upload it again"))
        manifest, tables, stripped = parsed
        records = []
        for name, data in sorted(files.get("results", {}).items()):
            try:
                records += [RunRecord.model_validate_json(data)] if name.startswith("runs/") else []
            except ValidationError:
                continue
        rendered = cards.render_cards(
            settings, manifest=manifest, tables=tables, records=records, writeup=writeup, nopreamble=stripped
        )
        for target in present:
            state, card = states[target.key], rendered[target.repo]
            staged = Path(scratch, "cards", target.key, CARD)
            staged.parent.mkdir(parents=True)
            staged.write_text(card, encoding="utf-8")
            hits = [*redact.scan([staged]), *((staged, number, "email") for number in _addressed(card))]
            secrets += [
                (ExitCode.REDACTION, f"{target.repo}/{CARD}:{line}: {rule} in the new card") for _, line, rule in hits
            ]
            deletes = _stale(target, state.files)
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
            current = files[target.key].get(CARD) == card.encode()
            steps.append(Step(target, state, card, not current, tuple(deletes)))
    commits = sum(len(found) for found in shas.values())
    problems = (*exposed, *secrets, *other)
    return Plan(
        tuple(steps), absent, tuple(hidden), shelves, problems, tuple(exposed), _advice(found), len(unread), commits
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
            if item.item_id.casefold().endswith(SMOKE_SUFFIX) or item.item_id.casefold() in merged
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


def _verified(hub: Hub, step: Step, scratch: Path, scans: Scans) -> str:
    """The head a repo may be opened at: its card commit on top of the planned files, every new commit scanned."""
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
    planned = (set(step.state.files) - set(step.deletes)) | {CARD}
    if set(now.files) != planned or _kept(hub, target, now.sha, scans, scratch).get(CARD) != step.card.encode():
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
    echo: Echo = print,
) -> Check | None:
    """Card, open and list every repo; the final check, or None after a dry run. ``scans`` reuses an earlier plan's."""
    _identity(settings, hub)
    scans = scans if scans is not None else Scans()
    drawn = plan(settings, hub, writeup=writeup, scans=scans)
    # a paid job rebuilds the merged checkpoint, so it goes only when the run can go on
    if delete and not dry_run and set(drawn.problems) <= set(drawn.exposed):
        delete_intermediates(settings, hub, echo=echo)
        drawn = plan(settings, hub, writeup=writeup, scans=scans)
    for line in drawn.lines(delete and dry_run):
        echo(line)
    if dry_run:
        return None
    if drawn.problems:
        raise DriftmapError(drawn.problems[0][0], drawn.refusal())
    given = f"{', delete=True' * delete}{f', writeup={writeup!r}' if writeup else ''}"
    resume = resume or f"resume: run publish(settings, hub{given}) again; finished steps are skipped"

    def reported(action: str) -> AbstractContextManager[None]:
        return _reported(action, settings, hub, resume)

    heads = {}
    with tempfile.TemporaryDirectory(prefix="driftmap-publish-") as scratch:
        for step in drawn.steps:
            target = step.target
            if not (step.stale_card or step.deletes):
                continue
            staged = Path(scratch, target.key)
            staged.mkdir()
            (staged / CARD).write_text(step.card, encoding="utf-8")
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
                heads[step.target.key] = _verified(hub, step, Path(scratch), scans)

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
        manifest = None
        if states["eval"].exists and (path := fetch(wanted[0], states["eval"], "manifest.json")):
            try:
                manifest = load(Manifest, path)
            except ValidationError:
                problems.append(f"{wanted[0].repo} has a manifest.json that does not parse")
        for target in wanted:
            state, shelf = states[target.key], shelves[target.shelf]
            if not state.exists:
                rows.append(Row(target.repo, False, False, "-", False, False, "-", not target.required))
                problems += [f"{target.repo} does not exist"] if target.required else []
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
