"""Publish the results Space: a static page of counts and figures, built from the public tables and labels."""

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.repocard import RepoCard
from huggingface_hub.utils import filter_repo_objects

from driftmap.core import redact
from driftmap.core.hub import Hub
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.jobs.snapshot import TABLES_PATH
from driftmap.publish import push

# the Space under the owner's name; the project README links to it
SPACE = "tiny-aya-safety-map"
FOLDER = "space"
DATA, FIGURES, CARD, PAGE = "data.json", "figures", "README.md", "index.html"
# what the replay explorer reads: prompt text and stored answers, written by the build when the responses are at hand
EXPLORER = "explorer.json"
# what the page is made of; the script that builds its data stays in the code repo
FILES = (CARD, PAGE, "app.js", "explorer.js", DATA, EXPLORER, f"{FIGURES}/*")
BUILD = "run `python space/build_data.py` first"
# a script the page loads from its own folder
SCRIPT = re.compile(r"""<script\b[^>]*\bsrc\s*=\s*["']([^"'?#]+)""", re.IGNORECASE)
# a new static Space can take a moment to answer
TRIES = 3
WAIT_S = 20
TIMEOUT_S = 30

Echo = Callable[[str], None]


def space_id(settings: Settings) -> str:
    return f"{settings.owner}/{SPACE}"


def page_url(settings: Settings) -> str:
    """Where a reader opens the Space."""
    return f"https://huggingface.co/spaces/{space_id(settings)}"


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _declared(space: Path) -> dict[str, Any]:
    """The front matter of the Space card; empty without a card."""
    card = space / CARD
    text = card.read_text(encoding="utf-8") if card.is_file() else ""
    return RepoCard(text, ignore_metadata_errors=True).data.to_dict()


def _built(root: Path, settings: Settings) -> list[str]:
    """The files of the page, relative to the Space folder; refuses a folder that is not ready to be shown."""
    space = root / FOLDER
    data = space / DATA
    if not data.is_file():
        raise _refuse(f"{FOLDER}/{DATA} is missing; {BUILD}")
    # the page shows the tables' numbers: a table written since would not be on it
    built = data.stat().st_mtime
    newer = [path.name for path in sorted((root / TABLES_PATH.parent).glob("*.json")) if path.stat().st_mtime > built]
    if newer:
        raise _refuse(f"{FOLDER}/{DATA} is older than {newer[0]}; {BUILD}")
    held = sorted(path.name for path in (space / FIGURES).iterdir()) if (space / FIGURES).is_dir() else []
    if not held:
        raise _refuse(f"{FOLDER}/{FIGURES}/ is missing or empty; {BUILD}")
    try:
        named = json.loads(data.read_bytes()).get("figures", [])
    except (ValueError, AttributeError):
        raise _refuse(f"{FOLDER}/{DATA} is not a JSON object; {BUILD}") from None
    lost = [name for name in named if name not in held]
    if lost:
        raise _refuse(f"{FOLDER}/{DATA} names the figure {lost[0]}, which {FOLDER}/{FIGURES}/ does not hold")
    if not (space / PAGE).is_file():
        raise _refuse(f"{FOLDER}/{PAGE} is missing")
    declared = _declared(space)
    if (declared.get("sdk"), declared.get("app_file")) != ("static", PAGE):
        raise _refuse(f"{FOLDER}/{CARD} does not declare a static Space (sdk: static, app_file: {PAGE})")
    link = page_url(settings)
    readme = root / CARD
    if link not in (readme.read_text(encoding="utf-8") if readme.is_file() else ""):
        raise _refuse(f"{CARD} does not link to {link}; the Space and the link must name one repo")
    paths = [path.relative_to(space).as_posix() for path in space.rglob("*") if path.is_file()]
    files = sorted(filter_repo_objects(paths, allow_patterns=list(FILES)))
    loaded = [src.removeprefix("./") for src in SCRIPT.findall((space / PAGE).read_text(encoding="utf-8"))]
    # a script from another site is not the Space's to hold
    lost = [src for src in loaded if "//" not in src and src not in files]
    if lost:
        raise _refuse(f"{FOLDER}/{PAGE} loads {lost[0]}, which is not among the page files")
    if EXPLORER in files:
        _replayable(space, built)
    return files


def _replayable(space: Path, built: float) -> None:
    """Refuse an explorer file that is not an object, is older than the data file or shows harmful answers."""
    explorer = space / EXPLORER
    try:
        shown = json.loads(explorer.read_bytes()).get("harmful_text")
    except (ValueError, AttributeError):
        raise _refuse(f"{FOLDER}/{EXPLORER} is not a JSON object; {BUILD}") from None
    # the build writes it after the data file; an older one replays other runs than the page counts
    if explorer.stat().st_mtime < built:
        both = "build both in one run of `python space/build_data.py`, or delete it"
        raise _refuse(f"{FOLDER}/{EXPLORER} is older than {FOLDER}/{DATA}; {both}")
    if shown is not False:
        raise _refuse(f"{FOLDER}/{EXPLORER} holds the text of harmful answers; build it without --include-harmful")


def _named(root: Path, settings: Settings) -> list[tuple[str, str]]:
    """The owner's datasets and models the Space card names, with their repo type."""
    declared = _declared(root / FOLDER)
    kinds = {"datasets": "dataset", "models": "model"}
    listed = [(repo, kind) for field, kind in kinds.items() for repo in declared.get(field) or []]
    return [(repo, kind) for repo, kind in listed if repo.startswith(f"{settings.owner}/")]


def _scanned(root: Path, files: list[str]) -> None:
    """Refuse a page file that holds a token, a key, an account id, a home path or an address.

    The explorer file quotes prompts and answers like a file of rows: digits and addresses in it are no finding.
    """
    space = root / FOLDER
    found = []
    for name in files:
        data = (space / name).read_bytes()
        # a figure as bytes holds nothing to read
        if b"\x00" in data or not push._is_text(data):
            continue
        hits = [(finding.line, finding.pattern) for finding in redact.scan([space / name])]
        hits += [(line, "email") for line in push._addressed(data.decode())]
        if name == EXPLORER:
            hits = [(line, rule) for line, rule in hits if rule not in ("account_id", "email")]
        found += [f"{FOLDER}/{name}:{line}: {rule}" for line, rule in sorted(hits)]
    if found:
        raise DriftmapError(ExitCode.REDACTION, "\n".join(found))


def _answers(url: str, opener: Callable[..., Any], sleep: Callable[[float], None]) -> None:
    """Read the page with no token; raises when it does not answer 200 within the tries."""
    said = "no answer"
    for attempt in range(TRIES):
        try:
            with opener(urllib.request.Request(url), timeout=TIMEOUT_S) as page:
                if page.status == 200:
                    return
                said = f"HTTP {page.status}"
        except urllib.error.HTTPError as error:
            said = f"HTTP {error.code}"
        except OSError as error:
            said = type(error).__name__
        if attempt + 1 < TRIES:
            sleep(WAIT_S)
    raise DriftmapError(
        ExitCode.UNEXPECTED, f"{url} does not answer logged out ({said}) after {TRIES} tries; the Space is uploaded"
    )


def planned(settings: Settings, hub: Hub, root: str | Path) -> list[str]:
    """The Space step as lines of a publication plan: the files ``publish_space`` would upload, or why it would
    refuse the folder.

    Reads only. The repos the Space names are still private while a publication is planned, so they are not asked for.
    """
    root, repo = Path(root), space_id(settings)
    head = f"{repo} (space, static)"
    try:
        files = _built(root, settings)
        _scanned(root, files)
    except DriftmapError as error:
        return [f"{head}: not ready: {'; '.join(str(error).splitlines())}"]
    try:
        now = hub.space_info(repo)
    except RepositoryNotFoundError:
        now = None
    except push.FAILURES as error:
        return [f"{head}: could not be read ({push._cause(error)})"]
    state = "create" if now is None else "update"
    opened = "public" if now is not None and now.private is False else "public after `driftmap publish space`"
    return [f"{head}: {state}; {opened}", *(f"upload: {name}" for name in files)]


def publish_space(
    settings: Settings,
    hub: Hub,
    root: str | Path,
    *,
    dry_run: bool = False,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    echo: Echo = print,
) -> None:
    """Create or update the static results Space from ``<root>/space`` in one commit, open it, and read it back.

    Refuses, before any write, a folder whose data file is missing or older than a local table, whose figures
    are missing, whose card is not a static Space's, whose page loads a script that is not among the page files,
    whose explorer file is older than the data file or holds the text of harmful answers, a README that links
    elsewhere, a page file with a finding, and a dataset or model of the owner that the Space names while it is
    not public. The explorer file goes up when the build wrote one and leaves the Space when it is gone. A dry
    run prints the files and writes nothing. Raises DriftmapError (exit 7) when the Hub fails, (exit 1) when the
    page does not answer a reader who is not logged in.
    """
    root, repo = Path(root), space_id(settings)
    push._identity(settings, hub)
    files = _built(root, settings)
    _scanned(root, files)
    for named, kind in _named(root, settings):
        try:
            info = (hub.dataset_info if kind == "dataset" else hub.model_info)(named)
        except RepositoryNotFoundError:
            info = None
        if info is None or info.private is not False or info.gated is not False:
            raise _refuse(f"{named}, which the Space names, is not public; run `driftmap publish run` first")
    try:
        now = hub.space_info(repo)
    except RepositoryNotFoundError:
        now = None
    state = "create" if now is None else "update"
    opened = "public" if now is not None and now.private is False else "public after the upload"
    echo(f"{repo} (space, static): {state}; {opened}")
    for name in files:
        echo(f"upload: {name}")
    if dry_run:
        echo("dry run: nothing was written")
        return
    try:
        # private when new, so nothing is shown before the page is whole
        hub.create_repo(repo, repo_type="space", space_sdk="static", private=True)
        hub.upload_folder(
            repo,
            root / FOLDER,
            repo_type="space",
            allow_patterns=list(FILES),
            # a figure or an explorer file the build no longer makes leaves the page
            delete_patterns=[f"{FIGURES}/*", EXPLORER],
            commit_message="update the results page",
        )
        hub.update_repo_settings(repo, private=False, repo_type="space")
    except push.FAILURES as error:
        raise DriftmapError(
            ExitCode.PUSH_FAILED, f"the upload to {repo} failed ({push._cause(error)}); run publish space again"
        ) from error
    _answers(page_url(settings), opener, sleep)
    echo(f"public: {page_url(settings)} answers logged out")
