"""Read labels again from the stored judge outputs after a parser change; no model is loaded."""

import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from driftmap.core import ids
from driftmap.core.hub import Hub
from driftmap.core.records import LabelRow
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.judge import parser
from driftmap.judge.incremental import (
    FIELDS,
    UNSCORED,
    RawRow,
    dump_rows,
    key,
    read_rows,
)
from driftmap.judge.polyguard import config_hash

SUPERSEDED = "superseded"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def reparse_rows(labels: Iterable[LabelRow], raws: Iterable[RawRow]) -> list[LabelRow]:
    """The rows as the current parser reads their raw outputs; DriftmapError when a judged row's raw is missing."""
    stored = {key(raw): raw for raw in raws}
    fresh = []
    for row in labels:
        if row.judge is None:
            fresh.append(row)
            continue
        config = row.judge.model_copy(update={"parser_version": parser.PARSER_VERSION})
        update: dict[str, object] = {"judge": config.model_copy(update={"config_hash": config_hash(config)})}
        if row.raw_output_present:
            raw = stored.get(key(row))
            if raw is None or (raw.repo, raw.revision) != (row.judge.repo, row.judge.revision):
                found = "no stored judge output" if raw is None else f"an output of {raw.repo}@{raw.revision[:12]}"
                raise DriftmapError(ExitCode.INTEGRITY, f"{row.language}/{row.prompt_id}: {found}; judge it again")
            parsed = parser.parse(raw.raw)
            update |= parsed._asdict() if parsed else dict.fromkeys(FIELDS, UNSCORED)
        fresh.append(row.model_copy(update=update))
    return fresh


def reparse(labels_dir: str | Path, raw_dir: str | Path, new_parser_version: str) -> dict[Path, Path]:
    """Rewrite the ``<slug>/<language>.jsonl`` label files that read differently now.

    Returns each rewritten file with the copy of its earlier rows, kept under ``<slug>/superseded/``.
    """
    if new_parser_version != parser.PARSER_VERSION:
        running = f"this code parses as version {parser.PARSER_VERSION}"
        raise DriftmapError(
            ExitCode.USAGE, f"{running}, not {new_parser_version}; labels must name the parser they got"
        )
    labels_dir, raw_dir = Path(labels_dir), Path(raw_dir)
    changed = {}
    for path in sorted(labels_dir.glob("*/*.jsonl")):
        labels = read_rows(path, LabelRow)
        fresh = reparse_rows(labels, read_rows(raw_dir / path.relative_to(labels_dir), RawRow))
        if fresh == labels:
            continue
        hashes = {row.judge.config_hash for row in labels if row.judge}
        forged = sorted(found for found in hashes if not _SHA256.fullmatch(found))
        if forged:
            raise DriftmapError(ExitCode.INTEGRITY, f"{path.name}: {forged[0][:80]!r} is not a judge config hash")
        kept = path.parent / SUPERSEDED / min(hashes) / path.name
        kept.parent.mkdir(parents=True, exist_ok=True)
        if not kept.exists():
            shutil.copyfile(path, kept)
        path.write_bytes(dump_rows(fresh))
        changed[path] = kept
    return changed


def reparse_checkpoint(
    hub: Hub, labels_repo: str, raw_repo: str, checkpoint: str, new_parser_version: str, mode: str = "full"
) -> list[str]:
    """Reparse one checkpoint's labels from the raw outputs in ``raw_repo``; returns the label paths uploaded."""
    slug = ids.slug(checkpoint)
    work = Path(tempfile.mkdtemp(prefix="driftmap-reparse-"))

    def fetch(repo: str, name: str, pattern: str) -> Path:
        head = hub.dataset_info(repo).sha
        hub.snapshot_download(f"{repo}@{head}", work / name, repo_type="dataset", allow_patterns=[pattern])
        # the directory holding one folder per checkpoint
        return work / name / Path(pattern).parents[1]

    try:
        labels_dir = fetch(labels_repo, "labels", ids.labels_path(slug, "*", mode))
        raw_dir = fetch(raw_repo, "raws", ids.judge_raw_path(slug, "*", mode))
        uploaded = []
        for path, kept in reparse(labels_dir, raw_dir, new_parser_version).items():
            # the earlier file first: the labels are never without a copy
            for file in (kept, path):
                name = file.relative_to(work / "labels").as_posix()
                hub.upload_file(labels_repo, name, file.read_bytes(), repo_type="dataset")
                uploaded.append(name)
        return uploaded
    finally:
        shutil.rmtree(work, ignore_errors=True)
