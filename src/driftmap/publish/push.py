"""Publish the repos: scan every commit and card them all, then open them one by one, then list them."""

import hashlib
import re
import shutil
import tempfile
from collections.abc import Callable, Collection, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import httpx2
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.utils import filter_repo_objects
from pydantic import ValidationError

from driftmap.core import ids, redact
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef
from driftmap.core.records import Manifest, RunRecord, Tables, load
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.publish import cards, collections
from driftmap.scripts.drift import SMOKE_SUFFIX

CARD = "README.md"
# key, repo type, collection, needed for the checklist
ORDER = (
    ("eval", "dataset", "data", True),
    ("results", "dataset", "data", True),
    ("responses", "dataset", "data", True),
    ("adapter", "model", "models", True),
    ("repaired", "model", "models", False),
    ("gguf", "model", "models", False),
)
ROWS = ("smoke/*", "labels/*/superseded/*")
# removed by a repo's card commit; the adapter's training state is not part of the artifact
STALE = {"eval": ROWS, "results": ROWS, "responses": ROWS, "adapter": ("last-checkpoint/*", "training_args.bin")}
WEIGHTS = ["adapter_model.bin", "pytorch_model*.bin"]
# pickles the trainer leaves beside the weights; the adapter is refused while one is left
PICKLES = (".bin", ".pt", ".pth")
# never downloaded from a model repo; a dataset is read whole
BINARY = ["*.safetensors", "*.gguf", *WEIGHTS, "*.onnx"]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
RULES = {**redact.PATTERNS, "email": EMAIL}
# the downloader's own folder inside a snapshot
DOWNLOADER = (".cache", "huggingface")
TEMP_PARTS = frozenset({".cache", "tmp", "__pycache__", ".ipynb_checkpoints"})
TEMP_SUFFIXES = (".tmp", ".lock", ".incomplete")
MAX_SHOWN = 20
ROTATE = "a new commit keeps the value in history; rotate the secret"
CHANGED = "{repo} changed after its scan; it stays as it is: run publish again"
# a failed transfer, a failed call, or a download the Hub could not serve
FAILURES = (httpx2.HTTPError, LocalEntryNotFoundError)

Echo = Callable[[str], None]
Problem = tuple[ExitCode, str]
# file, line (0 for the file's own name), rule
Hit = tuple[str, int, str]


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


class Scans:
    """Findings per commit and per file content, kept for one run: neither ever changes."""

    def __init__(self) -> None:
        self.commits: dict[tuple[str, str], list[Hit]] = {}
        self.contents: dict[tuple[bool, bytes], list[tuple[int, str]]] = {}


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

    def refusal(self) -> str:
        """Every problem, one per line; scan findings end with the advice to rotate."""
        found = any(code == ExitCode.REDACTION for code, _ in self.problems)
        return "\n".join([*(text for _, text in self.problems), *([ROTATE] if found else [])])

    def lines(self, delete: bool = False) -> list[str]:
        out = []
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
        return out + [f"problem: {line}" if line != ROTATE else line for line in self.refusal().splitlines()]


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
    seen: dict[str, str] = {}
    for key, *_ in ORDER:
        name = getattr(settings.repos, key).casefold()
        if name.endswith(SMOKE_SUFFIX):
            raise DriftmapError(
                ExitCode.USAGE, f"repos.{key} = {name} ends with {SMOKE_SUFFIX}; a smoke repo is never published"
            )
        if name in seen:
            raise DriftmapError(ExitCode.USAGE, f"repos.{key} and repos.{seen[name]} name one repo")
        seen[name] = key
    return [Target(key, f"{settings.owner}/{getattr(settings.repos, key)}", *rest) for key, *rest in ORDER]


def intermediates(settings: Settings) -> list[tuple[str, str]]:
    """Repo id and type of the merged checkpoint and of every smoke repo; a usage error when one is also published."""
    owner, repos = settings.owner, settings.repos
    keys = ("adapter", "merged", "repaired", "gguf", "eval", "results", "responses")
    kinds = {key: "dataset" if key in cards.DATASETS else "model" for key in keys}
    smoke = [(f"{owner}/{getattr(repos, key)}{SMOKE_SUFFIX}", kinds[key]) for key in keys]
    hidden = [(f"{owner}/{repos.merged}", "model"), *smoke]
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
    if target.key != "adapter":
        return []
    names = [path.rpartition("/")[2] for path in files]
    weights = set(filter_repo_objects(names, allow_patterns=WEIGHTS))
    return [path for path, name in zip(files, names, strict=True) if name.endswith(PICKLES) and name not in weights]


def _stray(files: tuple[str, ...]) -> list[str]:
    def temporary(path: str) -> bool:
        return bool(TEMP_PARTS & set(path.split("/"))) or path.endswith(TEMP_SUFFIXES)

    return [path for path in files if temporary(path)]


def _is_row(parts: tuple[str, ...]) -> bool:
    parts = parts[1:] if parts[:1] == (redact.SMOKE_TREE,) else parts
    return len(parts) > 1 and parts[0] in redact.DIGIT_EXEMPT_DIRS


def _safe(name: str) -> str:
    """A file name fit to print: whatever a rule finds in it is replaced."""
    for pattern in RULES.values():
        name = pattern.sub("***", name)
    return name


def _files(root: Path) -> list[tuple[Path, str]]:
    found = sorted(path for path in root.rglob("*") if path.is_file())
    return [
        (path, path.relative_to(root).as_posix()) for path in found if path.relative_to(root).parts[:2] != DOWNLOADER
    ]


def _scan(root: Path, scans: Scans) -> list[Hit]:
    """Every secret, home path, account id or address under ``root``, in file names too.

    Dataset rows may quote addresses and twelve-digit numbers; tokens and home paths are found everywhere.
    """
    hits: list[Hit] = []
    for path, name in _files(root):
        row = _is_row(tuple(name.split("/")))
        # the leading slash lets a name that starts at a home folder match; a row folder is named after a commit
        rules = {rule: pattern for rule, pattern in RULES.items() if not (row and rule == "account_id")}
        hits += [(name, 0, rule) for rule, pattern in rules.items() if pattern.search(f"/{name}")]
        data = path.read_bytes()
        key = (row, hashlib.sha256(data).digest())
        if key not in scans.contents:
            found = [(finding.line, finding.pattern) for finding in redact.scan([path], data_root=root)]
            # random bytes spell a short address often enough to refuse megabytes of optimizer state
            if not row and _is_text(data):
                lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
                found += [(number, "email") for number, line in enumerate(lines, start=1) for _ in EMAIL.finditer(line)]
            scans.contents[key] = sorted(found)
        hits += [(name, line, rule) for line, rule in scans.contents[key]]
    return hits


def _is_text(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return b"\x00" not in data


def _unreadable(tree: Path, skip: Collection[str]) -> list[str]:
    """Files that are not UTF-8 text, so the scan saw none of what they hold."""
    return [name for path, name in _files(tree) if name not in skip and not _is_text(path.read_bytes())]


def _history(hub: Hub, target: Target, head: str, scratch: Path, scans: Scans) -> tuple[Path, list[str]]:
    """The head's files on disk, and what the scan finds in any commit, named by the newest commit holding it."""
    skipped = BINARY if target.repo_type == "model" else None

    def fetch(sha: str) -> Path:
        folder = Path(tempfile.mkdtemp(prefix=f"{target.key}-", dir=scratch))
        ref = RepoRef(target.repo, sha)
        return Path(hub.snapshot_download(ref, folder, repo_type=target.repo_type, ignore_patterns=skipped))

    tree = fetch(head)
    listed = hub.list_repo_commits(RepoRef(target.repo, head), repo_type=target.repo_type)
    held: dict[Hit, list[str]] = {}
    for sha in dict.fromkeys([head, *(commit.commit_id for commit in listed)]):
        if (target.repo, sha) not in scans.commits:
            folder = tree if sha == head else fetch(sha)
            scans.commits[target.repo, sha] = _scan(folder, scans)
            if folder != tree:
                shutil.rmtree(folder)
        for hit in scans.commits[target.repo, sha]:
            held.setdefault(hit, []).append(sha)
    texts = []
    for (name, line, rule), shas in sorted(held.items()):
        spot = f"{_safe(name)}:{line}: {rule}" if line else f"{_safe(name)}: {rule} in the file name"
        earlier = len(shas) - 1
        more = f" (and {earlier} earlier commit{'s' * (earlier > 1)})" if earlier else ""
        texts.append(f"{target.repo}@{shas[0][:12]}/{spot}{more}")
    return tree, _capped(texts, target.repo)


def _capped(texts: list[str], repo: str) -> list[str]:
    more = len(texts) - MAX_SHOWN
    return texts if more <= 0 else [*texts[:MAX_SHOWN], f"{repo}: {more} more findings"]


def _read[T](cls: type[T], tree: Path | None, name: str) -> T | None:
    path = tree / name if tree else None
    return load(cls, path) if path is not None and path.is_file() else None


def _hidden(settings: Settings, hub: Hub) -> list[tuple[str, str, State]]:
    found = ((repo, repo_type, _state(hub, repo, repo_type)) for repo, repo_type in intermediates(settings))
    return [(repo, repo_type, state) for repo, repo_type, state in found if state.exists]


def plan(settings: Settings, hub: Hub, *, writeup: str | None = None, scans: Scans | None = None) -> Plan:
    """Read every repo, scan each of its commits and render its card; nothing on the Hub changes.

    ``scans`` carries the commits already read, so a second plan in one run reads only what is new.
    """
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
    steps = []
    with tempfile.TemporaryDirectory(prefix="driftmap-publish-") as scratch:
        trees = {}
        for target in present:
            trees[target.key], found = _history(hub, target, states[target.key].sha, Path(scratch), scans)
            secrets += [(ExitCode.REDACTION, text) for text in found]
        manifest = tables = None
        try:
            manifest = _read(Manifest, trees.get("eval"), "manifest.json")
            tables = _read(Tables, trees.get("results"), "tables/tables.json")
        except ValidationError as error:
            other.append((ExitCode.INTEGRITY, f"{error.title} on the Hub does not parse; build and upload it again"))
        records = []
        for path in sorted(trees["results"].glob("runs/*.json")) if "results" in trees else ():
            try:
                records.append(load(RunRecord, path))
            except ValidationError:
                continue
        rendered = cards.render_cards(settings, manifest=manifest, tables=tables, records=records, writeup=writeup)
        for target in present:
            state, tree, card = states[target.key], trees[target.key], rendered[target.repo]
            staged = Path(scratch, "cards", target.key)
            staged.mkdir(parents=True)
            (staged / CARD).write_text(card, encoding="utf-8")
            secrets += [
                (ExitCode.REDACTION, f"{target.repo}/{name}:{line}: {rule} in the new card")
                for name, line, rule in _scan(staged, scans)
            ]
            deletes = _stale(target, state.files)
            gaps = {
                "holds cache or temp files": [path for path in _stray(state.files) if path not in deletes],
                "holds training state": [path for path in _training_state(target, state.files) if path not in deletes],
            }
            gaps["holds files the scan cannot read"] = _unreadable(tree, {*deletes, *gaps["holds training state"]})
            other += [
                (ExitCode.INTEGRITY, f"{target.repo} {gap}: {', '.join(map(_safe, names))}")
                for gap, names in gaps.items()
                if names
            ]
            lacks = cards.missing(target.key, card, settings, manifest)
            if lacks:
                other.append((ExitCode.INTEGRITY, f"{target.repo} cannot get a complete card: {'; '.join(lacks)}"))
            head = tree / CARD
            current = head.is_file() and head.read_text(encoding="utf-8") == card
            steps.append(Step(target, state, card, not current, tuple(deletes)))
    return Plan(tuple(steps), absent, tuple(hidden), shelves, (*exposed, *secrets, *other), tuple(exposed))


def _shelf_problems(settings: Settings, shelves: dict[str, Any | None]) -> list[Problem]:
    merged = f"{settings.owner}/{settings.repos.merged}".casefold()
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
            if item.item_id.casefold().endswith(SMOKE_SUFFIX) or item.item_id.casefold() == merged
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
        cause = error.response.status_code if isinstance(error, HfHubHTTPError) else type(error).__name__
        text = f"{action} failed ({cause}); {_public_now(settings, hub)}\n{resume}"
        raise DriftmapError(ExitCode.PUSH_FAILED, text) from error


def _verified(hub: Hub, step: Step, scratch: Path, scans: Scans) -> str:
    """The head a repo may be opened at: its card commit on top of the planned files, every new commit scanned."""
    target = step.target
    now = _state(hub, target.repo, target.repo_type)
    tree, found = _history(hub, target, now.sha, scratch, scans)
    if found:
        raise DriftmapError(ExitCode.REDACTION, "\n".join([*found, ROTATE]))
    planned = (set(step.state.files) - set(step.deletes)) | {CARD}
    if set(now.files) != planned or (tree / CARD).read_text(encoding="utf-8") != step.card:
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
    echo: Echo = print,
) -> Check | None:
    """Card, open and list every repo; returns the final check, None after a dry run.

    ``delete`` removes the intermediates once nothing else stands in the way; a failed write ends with ``resume``.
    """
    _identity(settings, hub)
    scans = Scans()
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
    for step in drawn.steps:
        target = step.target
        with reported(f"opening {target.repo}"):
            if unchanged(step).open:
                echo(f"already public: {target.repo} ({target.repo_type})")
                continue
            hub.update_repo_settings(target.repo, private=False, gated=False, repo_type=target.repo_type)
            after = _state(hub, target.repo, target.repo_type)
            if after.sha != heads[target.key]:
                # what landed in between was never scanned
                hub.update_repo_settings(target.repo, private=True, repo_type=target.repo_type)
                closed = f"{target.repo} changed while it was opened; it is private again: run publish again"
                raise DriftmapError(ExitCode.INTEGRITY, closed)
            if not after.open:
                raise DriftmapError(ExitCode.UNEXPECTED, f"{target.repo} is still private or gated after the change")
        echo(f"public: {target.repo} ({target.repo_type})")

    for step in drawn.steps:
        shelf = drawn.shelves[step.target.shelf]
        with reported(f"listing {step.target.repo} in {shelf.slug}"):
            added = collections.add(hub, shelf, step.target.repo, step.target.repo_type)
        if added:
            echo(f"listed: {step.target.repo} in {shelf.slug}")

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
    """Delete the merged checkpoint and every smoke repo for good, as the owner; returns the repos deleted."""
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
