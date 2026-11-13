"""gguf.json: which file was quantized, from which checkpoint, with which llama.cpp release."""

import hashlib
from pathlib import Path

from pydantic import ValidationError

from driftmap.core.ids import RepoRef
from driftmap.core.records import GgufRecord, dump
from driftmap.core.schemas import SCHEMA_GGUF, DriftmapError, ExitCode
from driftmap.core.settings import LlamacppRelease

from . import QUANT


class GgufHashMismatch(DriftmapError):
    """The GGUF file is not the one its record describes."""

    exit_code_default = ExitCode.INTEGRITY


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(gguf: str | Path, source: RepoRef, release: LlamacppRelease, path: str | Path) -> GgufRecord:
    """Hash ``gguf`` and write its record to ``path``; ``source`` is the pinned checkpoint it was converted from."""
    record = GgufRecord(
        schema=SCHEMA_GGUF,
        llamacpp_tag=release.tag,
        tarball=release.source_tarball,
        tarball_sha256=release.sha256.source_tarball,
        quant=QUANT,
        sha256=sha256_file(gguf),
        bytes=Path(gguf).stat().st_size,
        source_repo=source.repo,
        source_revision=source.revision,
    )
    dump(record, Path(path))
    return record


def read(path: str | Path) -> GgufRecord:
    """The record at ``path``; DriftmapError (exit 12) when it is missing or not a GGUF record."""
    try:
        return GgufRecord.model_validate_json(Path(path).read_bytes())
    except (OSError, ValidationError) as error:
        # the pydantic error carries input values, so it is not chained
        raise DriftmapError(
            ExitCode.INTEGRITY, f"{Path(path).name} is not a readable GGUF record ({type(error).__name__})"
        ) from None


def verify(record: GgufRecord, gguf: str | Path) -> None:
    """Raise GgufHashMismatch (exit 12) unless ``gguf`` has the recorded size and sha256."""
    size, found = Path(gguf).stat().st_size, sha256_file(gguf)
    if (size, found) != (record.bytes, record.sha256):
        raise GgufHashMismatch(
            f"{Path(gguf).name} is {size} bytes hashing to {found[:12]}; "
            f"gguf.json says {record.bytes} bytes, {record.sha256[:12]}"
        )
