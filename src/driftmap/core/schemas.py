"""Schema ids, exit codes and the base error."""

from enum import IntEnum

SCHEMA_RUN = "driftmap.run/1"
SCHEMA_DECISION = "driftmap.decision/1"
SCHEMA_LEDGER = "driftmap.ledger/1"
SCHEMA_PLAN = "driftmap.plan/1"
SCHEMA_GENERATION = "driftmap.generation/1"
SCHEMA_LABEL = "driftmap.label/1"
SCHEMA_ALPHA_SCREEN = "driftmap.alpha_screen/1"
SCHEMA_GGUF = "driftmap.gguf/1"
SCHEMA_MANIFEST = "driftmap.manifest/1"
SCHEMA_TABLES = "driftmap.tables/1"

_SCHEMA_VERSIONS = {
    "run": SCHEMA_RUN,
    "decision": SCHEMA_DECISION,
    "ledger": SCHEMA_LEDGER,
    "plan": SCHEMA_PLAN,
    "generation": SCHEMA_GENERATION,
    "label": SCHEMA_LABEL,
    "alpha_screen": SCHEMA_ALPHA_SCREEN,
    "gguf": SCHEMA_GGUF,
    "manifest": SCHEMA_MANIFEST,
    "tables": SCHEMA_TABLES,
}


def schema_versions() -> dict[str, str]:
    """Record kind to schema id, as a fresh dict."""
    return dict(_SCHEMA_VERSIONS)


class ExitCode(IntEnum):
    """Process exit codes; 9 is unassigned."""

    SUCCESS = 0
    UNEXPECTED = 1
    USAGE = 2
    READINESS = 3
    GUARD_REFUSAL = 4
    LOST_RUN = 5
    REDACTION = 6
    PUSH_FAILED = 7
    DRIFT_BELOW_THRESHOLD = 8
    PARTIAL = 10
    OOM = 11
    INTEGRITY = 12
    COLUMN_FAILED = 13


class DriftmapError(Exception):
    """Base error carrying the exit code the CLI returns."""

    # a subclass that sets this takes the message alone
    exit_code_default: ExitCode | None = None

    def __init__(self, exit_code: ExitCode | int | str | None = None, message: object = "") -> None:
        pinned = self.exit_code_default
        if pinned is not None:
            if isinstance(exit_code, str):
                if message:
                    raise TypeError(f"{type(self).__name__} takes the message alone, got a second argument")
                exit_code, message = pinned, exit_code
            elif exit_code is None:
                exit_code = pinned
            elif ExitCode(exit_code) is not pinned:
                raise TypeError(f"{type(self).__name__} is pinned to exit code {int(pinned)}, got {exit_code!r}")
        elif isinstance(exit_code, str):
            raise TypeError("DriftmapError(exit_code, message): the first argument is the exit code")
        if exit_code is None:
            raise TypeError("DriftmapError needs an exit code")
        self.exit_code = ExitCode(exit_code)
        self.message = str(message)
        super().__init__(self.exit_code, message)

    def __str__(self) -> str:
        return self.message
