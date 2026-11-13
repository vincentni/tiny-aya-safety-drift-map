import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class ScriptedResult:
    returncode: int = 0
    stdout: str | bytes | None = ""
    stderr: str | bytes | None = ""


@dataclass
class Call:
    argv: list[str]
    kwargs: dict[str, Any]


class FakeSubprocess:
    """Stand-in for ``subprocess.run`` that replays results queued with ``script()``."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self._queue: deque[ScriptedResult] = deque()

    def script(self, *, returncode: int = 0, stdout: str | bytes | None = "", stderr: str | bytes | None = "") -> None:
        self._queue.append(ScriptedResult(returncode, stdout, stderr))

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        argv = list(argv)
        self.calls.append(Call(argv, dict(kwargs)))
        assert self._queue, f"unscripted subprocess call: {argv!r}"
        result = self._queue.popleft()
        if kwargs.get("check") and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv, output=result.stdout, stderr=result.stderr)
        return subprocess.CompletedProcess(argv, result.returncode, result.stdout, result.stderr)
