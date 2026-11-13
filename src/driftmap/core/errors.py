"""Errors shared by the generation backends and the job runner."""

import sys
from collections.abc import Iterator

from .schemas import DriftmapError, ExitCode


class OutOfMemory(DriftmapError):
    """The backend ran out of memory; the batch gets no rows."""

    exit_code_default = ExitCode.OOM


class Breaker(DriftmapError):
    """The server stopped answering; the column fails."""

    exit_code_default = ExitCode.COLUMN_FAILED


def chain(error: BaseException) -> Iterator[BaseException]:
    """The error, then every cause and context behind it, each once."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        link = pending.pop()
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        pending += [link.__cause__, link.__context__]


def classify_oom(error: BaseException) -> bool:
    """True when a CUDA out-of-memory error is anywhere in the exception chain."""
    # without torch imported nothing can have raised its error
    oom = getattr(sys.modules.get("torch"), "OutOfMemoryError", ())
    return any(isinstance(link, oom) or "CUDA out of memory" in str(link) for link in chain(error))
