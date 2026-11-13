"""What the run record says about an out-of-memory failure."""

from driftmap.core.errors import OutOfMemory, chain, classify_oom
from driftmap.core.redact import mask

MAX_TEXT = 200


def _names_oom(link: BaseException) -> bool:
    return (
        isinstance(link, OutOfMemory) or type(link).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(link)
    )


def oom_text(error: BaseException) -> str | None:
    """The out-of-memory error behind ``error`` as one short line; None when there is none."""
    links = list(chain(error))
    if not (classify_oom(error) or any(isinstance(link, OutOfMemory) for link in links)):
        return None
    cause = next((link for link in links if _names_oom(link)), error)
    return f"{type(cause).__name__}: {mask(str(cause))}"[:MAX_TEXT]
