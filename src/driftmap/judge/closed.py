"""Label a checkpoint's stored responses with a closed model, reached through any LiteLLM route."""

import functools
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from driftmap.core import ids, redact
from driftmap.core.hub import Hub
from driftmap.core.records import GenerationRow, JudgeConfig, LabelRow
from driftmap.core.retry import policy
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.judge import rubric
from driftmap.judge.incremental import (
    UNSCORED,
    Judged,
    Key,
    RawRow,
    check_unique,
    dump_rows,
    judged_rows,
    key,
    pending,
    read_rows,
    rebuilt,
    union,
)
from driftmap.judge.units import JUDGED_SETS, read_inputs, units

# the one place the route is read; it is never stored, printed or logged
ROUTE_ENV = "DRIFTMAP_CLOSED_JUDGE_ROUTE"
TIMEOUT_S = 60
# a unit may store one unscored answer per this many pairs sent, and always one
SENT_PER_UNSCORED = 20
# stored in place of an answer; it parses to no labels
DECLINED = "[not judged: the route declined the pair]"
# stored for an answer without text; "length" means the token limit cut it
EMPTY = "[empty answer: {reason}]"
SHOWN_RAWS = 3
# name prefixes of the label and raw-answer files written beside a rows file
DERIVED = ("labels-", "judge_raw-")
# label field and the value counted in the safe and refused columns of the printed line
SCORED = (("response_harm", "no"), ("refusal", "yes"))
_LIBRARY_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
# library error class to the word printed for it
_REFUSALS = (
    ("AuthenticationError", "authentication"),
    ("PermissionDeniedError", "permission"),
    ("NotFoundError", "not found"),
    ("BadRequestError", "bad request"),
)


class Completer(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        """The model's answer to one chat; raises Stopped, Transient or Declined."""


class Stopped(DriftmapError):
    """The route cannot go on; what is finished is kept and the run ends."""

    exit_code_default = ExitCode.GUARD_REFUSAL


class Throttled(Stopped):
    """Throttling or an exhausted quota: never retried."""


class Transient(Exception):
    """A fault worth another attempt: a timeout, a dropped connection, a server error."""


class Declined(Exception):
    """The route would not process this one pair."""


class _Unparsed(Exception):
    pass


@functools.cache
def library() -> Any:
    """LiteLLM, imported on first use without its start-up download and with its own logging off."""
    # otherwise the library fetches a price list when it is imported
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # in its default mode the library reads a .env file when it is imported
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
    import litellm

    litellm.suppress_debug_info = True
    for name in _LIBRARY_LOGGERS:
        logger = logging.getLogger(name)
        # its log lines name the model and where it is served
        logger.handlers, logger.propagate, logger.disabled = [logging.NullHandler()], False, True
    return litellm


def _sorted(error: Exception) -> Exception:
    """The error under a fixed word of this module: the library's text and class names say who serves the route."""
    litellm = library()
    status = getattr(error, "status_code", None)
    if isinstance(error, (litellm.RateLimitError, litellm.BudgetExceededError)) or status == 429:
        return Throttled("the judge route is throttling or out of quota")
    if isinstance(error, (litellm.ContentPolicyViolationError, litellm.ContextWindowExceededError)):
        return Declined()
    if isinstance(error, litellm.Timeout):
        return Transient("timeout")
    if isinstance(error, litellm.APIConnectionError):
        return Transient("connection")
    if isinstance(status, int) and status >= 500:
        return Transient("server error")
    word = next((word for name, word in _REFUSALS if isinstance(error, getattr(litellm, name))), "error")
    return Stopped(f"the judge route refused the call ({word}); check {ROUTE_ENV} and its credentials")


def _ours(record: logging.LogRecord) -> bool:
    return record.name.startswith("driftmap")


class LiteLLMCompleter:
    """One chat completion per pair through LiteLLM; counts the tokens the route reports."""

    def __init__(self, route: str) -> None:
        library()
        for handler in logging.getLogger().handlers:
            # client libraries log each request with its model id, and their logger names say who serves it
            handler.addFilter(_ours)
        self._route = route
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            # no sampling parameter: the model's defaults apply; retries are the caller's
            reply = library().completion(
                model=self._route,
                messages=messages,
                max_tokens=rubric.MAX_NEW_TOKENS,
                timeout=TIMEOUT_S,
                num_retries=0,
                max_retries=0,
            )
        except Exception as error:
            raise _sorted(error) from None
        usage = getattr(reply, "usage", None)
        with self._lock:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        choices = getattr(reply, "choices", None)
        if not choices:
            return EMPTY.format(reason="no choices")
        content = getattr(choices[0].message, "content", None)
        return content or EMPTY.format(reason=getattr(choices[0], "finish_reason", None))


def from_env() -> LiteLLMCompleter:
    """The client for the route in ROUTE_ENV; DriftmapError (exit 2) when it is unset."""
    route = os.environ.get(ROUTE_ENV, "").strip()
    if not route:
        raise DriftmapError(ExitCode.USAGE, f"{ROUTE_ENV} is not set; export the LiteLLM route of the judge model")
    return LiteLLMCompleter(route)


def ask(client: Completer, row: GenerationRow, sleep: Callable[[float], None]) -> Judged:
    """The answer for one row and its labels, None when three attempts gave none that parses; raises Stopped."""
    last = ""

    def attempt() -> Judged:
        nonlocal last
        # an answer that had text but no object is followed by the format reminder
        wordy = bool(last.strip()) and not last.startswith(EMPTY.partition("{")[0])
        last = client.complete(rubric.messages(row.prompt, row.response, reminder=wordy))
        labels = rubric.parse(last)
        if labels is None:
            raise _Unparsed
        return last, labels

    try:
        # throttling is never retried: it is no Transient
        return policy(sleep, also=lambda error: isinstance(error, (Transient, _Unparsed)))(attempt)
    except _Unparsed:
        return last, None
    except Declined:
        return DECLINED, None
    except Transient as error:
        raise Stopped(f"the judge route kept failing ({error})") from None


def ask_all(
    client: Completer, rows: Sequence[GenerationRow], concurrency: int, sleep: Callable[[float], None]
) -> tuple[dict[Key, Judged], BaseException | None]:
    """Answers for the rows that got one, at most ``concurrency`` in flight, and what cut the rest short.

    An interrupt or any error stops new calls; the calls in flight finish and their answers are returned.
    """
    halt = threading.Event()

    def one(row: GenerationRow) -> Judged | None:
        if halt.is_set():
            return None
        try:
            return ask(client, row, sleep)
        except BaseException:
            halt.set()
            raise

    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = {key(row): pool.submit(one, row) for row in rows}
    cut: BaseException | None = None
    try:
        wait(futures.values())
    except BaseException as error:
        # an interrupt lands here, in the waiting thread
        cut = error
    finally:
        halt.set()
        pool.shutdown(wait=True, cancel_futures=True)
    answers: dict[Key, Judged] = {}
    for name, future in futures.items():
        if future.cancelled():
            continue
        error = future.exception()
        if error is not None:
            cut = cut or error
        elif future.result() is not None:
            answers[name] = future.result()
    return answers, cut


@dataclass
class _Unit:
    """Rows judged together, and the (repo, path) of the files their label and raw rows go to."""

    name: str
    rows: list[GenerationRow]
    labels: tuple[str, str]
    raws: tuple[str, str]


@dataclass
class _Files:
    """The label and answer files of a run by (repo, path), as last read, and the repo heads they were read at."""

    hub: Hub
    scratch: Path
    rows: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    heads: dict[str, str] = field(default_factory=dict)

    def fetch(self, repo: str, patterns: list[str]) -> Path:
        """The files of ``repo`` matching ``patterns`` at its head, which is remembered."""
        self.heads[repo] = self.hub.dataset_info(repo).sha
        target = Path(tempfile.mkdtemp(dir=self.scratch))
        pinned = f"{repo}@{self.heads[repo]}"
        return Path(self.hub.snapshot_download(pinned, target, repo_type="dataset", allow_patterns=patterns))

    def add(self, place: tuple[str, str], new: Sequence[Any], model: type[Any], message: str) -> None:
        """Merge ``new`` into the file by row key and upload it when that changes its bytes."""
        repo, path = place
        rows = union(self.rows[place], new)
        if dump_rows(rows) == dump_rows(self.rows[place]):
            # the Hub turns an unchanged file into an empty commit
            return
        if self.hub.dataset_info(repo).sha != self.heads[repo]:
            # the whole file goes up: rows someone else committed since the read are merged in first
            rows = union(read_rows(self.fetch(repo, [path]) / path, model), new)
        info = self.hub.upload_file(repo, path, dump_rows(rows), repo_type="dataset", commit_message=message)
        self.heads[repo] = getattr(info, "oid", None) or self.hub.dataset_info(repo).sha
        self.rows[place] = rows


def _run(
    files: _Files,
    client: Completer,
    config: JudgeConfig,
    units_: Sequence[_Unit],
    concurrency: int,
    retry_unscored: bool,
    echo: Callable[[str], object],
    sleep: Callable[[float], None],
) -> None:
    """Label each unit's waiting rows and store them, the answers that parse even when the unit is cut short."""
    sent = parsed = 0
    held: list[str] = []
    for unit in units_:
        raws = files.rows[unit.raws]
        # a label whose file was lost comes back from its stored answer, unasked
        restored = rebuilt(unit.rows, files.rows[unit.labels], raws, config)
        labels = union(files.rows[unit.labels], restored)
        waiting = pending(unit.rows, labels, config.config_hash, retry_unscored, raws=raws)
        answers, cut = ask_all(client, [row for row in waiting if row.response is not None], concurrency, sleep)
        lost = [raw for raw, found in answers.values() if found is None]
        sent, parsed = sent + len(answers), parsed + len(answers) - len(lost)
        # the share is of the pairs sent; a unit cut short has none yet, and the flag accepts what stays unscored
        over = cut is None and not retry_unscored and len(lost) > max(1, len(answers) // SENT_PER_UNSCORED)
        if over or cut is not None:
            answers = {name: judged for name, judged in answers.items() if judged[1] is not None}
        done = [row for row in waiting if row.response is None or key(row) in answers]
        new_labels, new_raws = judged_rows(done, [answers[key(row)] for row in done if key(row) in answers], config)
        message = f"closed-judge labels for {unit.name}"
        # the answers are there before a label says so
        files.add(unit.raws, new_raws, RawRow, message)
        files.add(unit.labels, [*restored, *new_labels], LabelRow, message)
        keys = {key(row) for row in unit.rows}
        mine = [row for row in files.rows[unit.labels] if key(row) in keys and row.judge == config]
        unscored = sum(row.response_harm == UNSCORED for row in mine)
        safe, refused = (sum(getattr(row, name) == value for row in mine) for name, value in SCORED)
        line = f"{unit.name:<12} scored {len(mine) - unscored}  unscored {unscored}  safe {safe}  refused {refused}"
        line = f"{line}  new {len(restored) + len(new_labels)}"
        if over:
            held.append(unit.name)
            echo(f"{line}  held {len(lost)}")
            quoted = "  ".join(json.dumps(redact.mask(raw[:200]), ensure_ascii=False) for raw in lost[:SHOWN_RAWS])
            echo(f"  answers that did not parse: {quoted}")
        else:
            echo(line)
        if cut is not None:
            state = f"{len(done)} of {len(waiting)} waiting pairs of {unit.name} were stored"
            if isinstance(cut, Stopped):
                raise type(cut)(f"{cut}; {state}; run the command again later to continue")
            echo(f"stopped: {state}")
            raise cut
    echo(f"{sent} pairs sent, {parsed} answers parsed")
    if held:
        found = f"more than 5% of the pairs sent for {', '.join(held)}, and more than one, gave no labels"
        way_out = "run the command again to ask them again, or with --retry-unscored to store what stays unscored"
        raise DriftmapError(ExitCode.GUARD_REFUSAL, f"{found}; the answers that parsed are stored; {way_out}")


def judge_checkpoint(
    settings: Settings,
    hub: Hub,
    client: Completer,
    checkpoint: str,
    *,
    mode: str,
    sets: Sequence[str] = JUDGED_SETS,
    concurrency: int = 4,
    limit: int | None = None,
    retry_unscored: bool = False,
    preamble: str = "keep",
    echo: Callable[[str], object] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Label each (set, language) file not yet labelled under the closed config; upload to the evaluation dataset.

    With ``retry_unscored`` the answered pairs whose label reads unscored are asked again, the new answer winning,
    and what stays unscored is stored. Raises Stopped once the route throttles or keeps failing and DriftmapError
    when a unit had too many answers without labels; the answers that parsed are stored in both cases, as they are
    before an interrupt or an unexpected error is passed on.
    """
    config = rubric.judge_config(settings)
    slug = ids.slug(checkpoint)
    responses, labelled = (f"{settings.owner}/{name}" for name in (settings.repos.responses, settings.repos.eval))
    with tempfile.TemporaryDirectory(prefix="driftmap-closed-") as scratch:
        files = _Files(hub, Path(scratch))
        wanted = [unit for unit in units(settings) if unit[0] in sets]
        rows_dir = files.fetch(responses, [ids.responses_path(slug, name, "*", mode, preamble) for name in sets])
        inputs = read_inputs(rows_dir, slug, wanted, mode, limit, preamble)
        kept_labels = files.fetch(labelled, [ids.labels_path(slug, "*", mode, preamble=preamble)])
        kept_raws = files.fetch(responses, [ids.judge_raw_path(slug, "*", mode, preamble)])
        listed = []
        for (name, language), rows in inputs.items():
            labels = ids.labels_path(slug, language, mode, preamble=preamble)
            raws = ids.judge_raw_path(slug, language, mode, preamble)
            files.rows.setdefault((labelled, labels), read_rows(kept_labels / labels, LabelRow))
            files.rows.setdefault((responses, raws), read_rows(kept_raws / raws, RawRow))
            listed.append(_Unit(f"{name}/{language}", rows, (labelled, labels), (responses, raws)))
        _run(files, client, config, listed, concurrency, retry_unscored, echo, sleep)


def judge_rows(
    settings: Settings,
    hub: Hub,
    client: Completer,
    pattern: str,
    *,
    mode: str,
    concurrency: int = 4,
    limit: int | None = None,
    retry_unscored: bool = False,
    echo: Callable[[str], object] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Label the harmful rows of each responses-repo file matching ``pattern`` under the tree of ``mode``.

    Labels go beside each file as ``labels-<name>`` and answers as ``judge_raw-<name>``, never into the evaluation
    dataset; raises as ``judge_checkpoint`` does, and DriftmapError when no file matches.
    """
    config = rubric.judge_config(settings)
    responses = f"{settings.owner}/{settings.repos.responses}"
    tree = "smoke/" if mode == "smoke" else ""
    folder = f"{tree}{pattern}".rpartition("/")[0]
    beside = [f"{folder}/{prefix}*" if folder else f"{prefix}*" for prefix in DERIVED]
    with tempfile.TemporaryDirectory(prefix="driftmap-closed-") as scratch:
        files = _Files(hub, Path(scratch))
        root = files.fetch(responses, [f"{tree}{pattern}", *beside])
        found = [path for path in sorted(root.glob(f"{tree}{pattern}")) if not path.name.startswith(DERIVED)]
        if not found:
            raise DriftmapError(ExitCode.INTEGRITY, f"{tree}{pattern} matches no file in {responses}")
        listed = []
        for path in found:
            name = path.relative_to(root).as_posix()
            rows = [row for row in read_rows(path, GenerationRow, name) if row.set == "harmful"]
            check_unique(rows, name)
            labels, raws = (path.with_name(f"{prefix}{path.name}") for prefix in DERIVED)
            targets = [(responses, file.relative_to(root).as_posix()) for file in (labels, raws)]
            files.rows[targets[0]], files.rows[targets[1]] = read_rows(labels, LabelRow), read_rows(raws, RawRow)
            listed.append(_Unit(name, rows[: limit or None], *targets))
        _run(files, client, config, listed, concurrency, retry_unscored, echo, sleep)
