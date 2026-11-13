"""Quantization of the repaired model: repo names, file names and the disk check both jobs share."""

import shutil
from pathlib import Path

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.scripts.drift import SMOKE_SUFFIX

QUANT = "q4_k_m"
GGUF_FILE = f"model-{QUANT}.gguf"
RECORD_FILE = "gguf.json"
GIB = 2**30


def repo_names(settings: Settings, mode: str) -> tuple[str, str]:
    """Repaired and GGUF repo ids; a smoke run gets repos of its own."""
    suffix = SMOKE_SUFFIX if mode == "smoke" else ""
    return tuple(f"{settings.owner}/{name}{suffix}" for name in (settings.repos.repaired, settings.repos.gguf))


def require_disk(directory: str | Path, needed: int, what: str) -> None:
    """Raise DriftmapError (exit 12) when ``needed`` bytes do not fit the disk under ``directory``."""
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise DriftmapError(
            ExitCode.INTEGRITY, f"{what} needs {needed / GIB:.1f} GiB of disk, {free / GIB:.1f} GiB are free"
        )
