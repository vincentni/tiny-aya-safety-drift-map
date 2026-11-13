"""Scan files bound for publication, and mask secrets in log output."""

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from .schemas import DriftmapError, ExitCode

PATTERNS = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    # a float's fractional digits are not an account id
    "account_id": re.compile(r"(?<!\d\.)\b\d{12}\b"),
    "home_path": re.compile(r"(?:(?<![A-Za-z0-9_./-])|(?<=file://))/(home|Users|local/home)/[A-Za-z0-9_.-]+"),
}
# dataset rows quote arbitrary numbers
DIGIT_EXEMPT_DIRS = frozenset({"prompts", "sft_train", "labels", "responses"})
# a trailing escape letter or digit would defeat the anchors
_ANSI = re.compile(r"(?:\x1b|\\u001[bB])\[[0-?]*[ -/]*[@-~]")
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


def _digits_exempt(path: Path, data_root: str | Path | None) -> bool:
    if not data_root:
        return False
    try:
        parts = path.resolve().relative_to(Path(data_root).resolve()).parts
    except ValueError:
        return False
    return len(parts) > 1 and parts[0] in DIGIT_EXEMPT_DIRS


def scan(paths: Iterable[str | Path], data_root: str | Path | None = None) -> list[Finding]:
    """One finding per pattern hit; dataset rows under ``data_root`` may hold twelve-digit numbers."""
    findings = []
    for path in map(Path, paths):
        names = [name for name in PATTERNS if name != "account_id" or not _digits_exempt(path, data_root)]
        with path.open(encoding="utf-8", errors="replace") as lines:
            for number, line in enumerate(lines, start=1):
                line = _JSON_ESCAPE.sub(" ", _ANSI.sub(" ", line))
                findings.extend(Finding(path, number, name) for name in names for _ in PATTERNS[name].finditer(line))
    return findings


def mask(text: str) -> str:
    """Replace tokens with ``hf_***`` and home directories with ``<home>``; colour codes become spaces."""
    text = PATTERNS["hf_token"].sub("hf_***", _ANSI.sub(" ", text))
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
