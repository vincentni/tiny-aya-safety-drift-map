"""One launcher at a time."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from driftmap.core.schemas import DriftmapError, ExitCode


class LockHeld(DriftmapError):
    """Another process is launching."""

    exit_code_default = ExitCode.GUARD_REFUSAL


@contextmanager
def launch_lock(path: str | Path) -> Iterator[None]:
    """Hold the launch lock for the block; raises LockHeld at once when another process has it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockHeld(f"another launch holds the lock {path}") from None
        yield
