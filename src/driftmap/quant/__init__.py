"""Quantization of a checkpoint: repo names, file names and the disk check both jobs share."""

import shutil
from pathlib import Path

from driftmap.core import ids
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings

QUANT = "q4_k_m"
GGUF_FILE = f"model-{QUANT}.gguf"
RECORD_FILE = "gguf.json"
GIB = 2**30
# the checkpoints a GGUF can be made from: the repaired model, or the merged fine-tuned one
SOURCES = ("repaired", "merged")
# what an adapter repo's name ends with, and what the GGUF repo of its merged model ends with instead
ADAPTER_SUFFIX, GGUF_SUFFIX = "-lora", "-GGUF"


def repo_names(settings: Settings, mode: str) -> tuple[str, str]:
    """Repaired and GGUF repo ids; a smoke run gets repos of its own."""
    suffix = ids.repo_suffix(mode)
    return tuple(f"{settings.owner}/{name}{suffix}" for name in (settings.repos.repaired, settings.repos.gguf))


def merged_names(settings: Settings, mode: str) -> tuple[str, str]:
    """The merged fine-tuned checkpoint and the repo of its GGUF, named after the adapter; smoke repos apart."""
    suffix = ids.repo_suffix(mode)
    base = settings.repos.adapter.removesuffix(ADAPTER_SUFFIX)
    return tuple(f"{settings.owner}/{name}{suffix}" for name in (settings.repos.merged, f"{base}{GGUF_SUFFIX}"))


def names(settings: Settings, mode: str, source: str = "repaired") -> tuple[str, str]:
    """Repo ids of the checkpoint ``source`` names and of its GGUF; ValueError for a source there is none of."""
    if source not in SOURCES:
        raise ValueError(f"source is {' or '.join(SOURCES)}, not {source!r}")
    return merged_names(settings, mode) if source == "merged" else repo_names(settings, mode)


def require_disk(directory: str | Path, needed: int, what: str) -> None:
    """Raise DriftmapError (exit 12) when ``needed`` bytes do not fit the disk under ``directory``."""
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise DriftmapError(
            ExitCode.INTEGRITY, f"{what} needs {needed / GIB:.1f} GiB of disk, {free / GIB:.1f} GiB are free"
        )
