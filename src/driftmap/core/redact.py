"""Scan files bound for publication, and mask secrets in log output."""

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

from . import ids
from .schemas import DriftmapError, ExitCode

PATTERNS = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    # a float's fractional digits are not an account id
    "account_id": re.compile(r"(?<!\d\.)\b\d{12}\b"),
    "home_path": re.compile(r"(?:(?<![A-Za-z0-9_./-])|(?<=file://))/(home|Users|local/home)/[A-Za-z0-9_.-]+"),
}
# dataset rows quote arbitrary numbers; screen holds the repair screen's rows with their labels and judge answers
DIGIT_EXEMPT_DIRS = frozenset({"prompts", "sft_train", "labels", "responses", "judge_raw", "screen"})
# smoke runs and runs under another prompt condition keep the same folders under a tree of their own; longest first
_PREFIXES = {ids.tree(mode, preamble) for mode in ("full", "smoke") for preamble in ids.CONDITIONS}
TREES = sorted((tuple(filter(None, prefix.split("/"))) for prefix in _PREFIXES), key=len, reverse=True)
_CONTROL = r"(?:\x1b|\\u001[bB])\[[0-?]*[ -/]*[@-~]"
# a trailing escape letter or digit would defeat the anchors
_ANSI = re.compile(_CONTROL)
# the guard stops a run before the next token's prefix
_CHAR = rf"(?:(?:{_CONTROL})*(?!hf_)[A-Za-z0-9])"
_BREAK = rf"(?:{_CONTROL})*\r?\n"
# characters after the prefix of a user access token
TOKEN_BODY = 34


def _cut(end: str = "") -> str:
    """Alternatives for a token cut by one line break: both parts make exactly one body."""
    return "|".join(rf"{_CHAR}{{{head}}}{_BREAK}{_CHAR}{{{TOKEN_BODY - head}}}{end}" for head in range(TOKEN_BODY))


# colour codes or one line break can cut a logged token
_LOGGED_TOKEN = re.compile(rf"hf_(?:{_cut()}|{_CHAR}{{30,}})")
# a scan takes a cut only where the body ends
_SCANNED_TOKEN = re.compile(rf"hf_(?:{_cut(rf'(?!{_CHAR})')}|{_CHAR}{{30,}})")
_JSON_ESCAPE = re.compile(r"\\(?:[bfnrt]|u[0-9a-fA-F]{4})")
# request lines may carry signed query strings
_HTTP_LOGGERS = ("httpx", "httpx2", "httpcore")
_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


class Finding(NamedTuple):
    path: Path
    line: int
    pattern: str


class RedactionFinding(DriftmapError):
    """A file bound for publication carries a secret, account id or home path."""

    exit_code_default = ExitCode.REDACTION

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        super().__init__("\n".join(f"{f.path}:{f.line}: {f.pattern}" for f in findings))


def is_row(parts: Sequence[str]) -> bool:
    """Whether a path inside a dataset, given as its parts, is a file of rows in the tree of any run."""
    parts = tuple(parts)
    tree = next(tree for tree in TREES if parts[: len(tree)] == tree)
    return len(parts) - len(tree) > 1 and parts[len(tree)] in DIGIT_EXEMPT_DIRS


def _digits_exempt(path: Path, data_root: str | Path | None) -> bool:
    if not data_root:
        return False
    try:
        return is_row(path.resolve().relative_to(Path(data_root).resolve()).parts)
    except ValueError:
        return False


def scan(paths: Iterable[str | Path], data_root: str | Path | None = None) -> list[Finding]:
    """One finding per pattern hit; dataset rows under ``data_root`` may hold twelve-digit numbers."""
    findings = []
    for path in map(Path, paths):
        names = [name for name in PATTERNS if name != "account_id" or not _digits_exempt(path, data_root)]
        text = path.read_text(encoding="utf-8", errors="replace")
        # sought in the whole text: a token can span lines
        hits = [(text.count("\n", 0, token.start()) + 1, "hf_token") for token in _SCANNED_TOKEN.finditer(text)]
        for number, line in enumerate(text.split("\n"), start=1):
            line = _JSON_ESCAPE.sub(" ", _ANSI.sub(" ", line))
            hits += [(number, n) for n in names if n != "hf_token" for _ in PATTERNS[n].finditer(line)]
        hits.sort(key=lambda hit: (hit[0], names.index(hit[1])))
        findings.extend(Finding(path, number, name) for number, name in hits)
    return findings


def _masked(token: re.Match[str]) -> str:
    cut = re.search(r"\r?\n", token[0])
    return f"hf_***{cut[0]}***" if cut else "hf_***"


def mask(text: str) -> str:
    """Replace tokens, whole or cut, with ``hf_***`` and home directories with ``<home>``; colour codes go."""
    text = _ANSI.sub(" ", _LOGGED_TOKEN.sub(_masked, text))
    return PATTERNS["home_path"].sub("<home>", text)


def _mask_record(record: logging.LogRecord) -> None:
    try:
        message = record.getMessage()
    except Exception:  # logging must never raise into the caller
        message = "<unprintable log record>"
    record.msg, record.args = mask(message), None
    if record.exc_info and not record.exc_text:
        record.exc_text = _FORMATTER.formatException(record.exc_info)
    if record.exc_text:
        record.exc_text = mask(record.exc_text)
    if record.stack_info:
        record.stack_info = mask(record.stack_info)


class Filter(logging.Filter):
    """Mask tokens and home paths in a record's message, traceback and stack."""

    def filter(self, record: logging.LogRecord) -> bool:
        _mask_record(record)
        return True


class _MaskedRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _mask_record(self)


def configure_logging(verbose: bool = False) -> None:
    """Send logging to stderr at INFO, or DEBUG when verbose, masked for every logger."""
    handler = logging.StreamHandler()
    handler.addFilter(Filter())
    handler.setFormatter(_FORMATTER)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler], force=True)
    for name in _HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET if verbose else logging.WARNING)
    # libraries attach their own handlers, which the root filter never sees
    logging.setLogRecordFactory(_MaskedRecord)
