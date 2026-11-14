"""The response files a judge run labels: one per harmful language and the English over-refusal set."""

from collections.abc import Sequence
from pathlib import Path

from driftmap.core import ids
from driftmap.core.records import GenerationRow
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.judge.incremental import check_unique, read_rows

# benign rows are scored by chrF where they are generated, never judged
JUDGED_SETS = ("harmful", "xstest")
XSTEST_LANGUAGE = "en"

Unit = tuple[str, str]


def units(settings: Settings) -> list[Unit]:
    """The (set, language) files a judge run labels, in order."""
    return [("harmful", language) for language in settings.languages] + [("xstest", XSTEST_LANGUAGE)]


def _slug(checkpoint: str) -> str:
    try:
        return ids.slug(checkpoint)
    except ValueError:
        return checkpoint


def read_inputs(
    directory: Path, slug: str, wanted: Sequence[Unit], mode: str, limit: int | None
) -> dict[Unit, list[GenerationRow]]:
    """Generation rows per unit; DriftmapError on a missing file, a repeated prompt or a row of another unit."""
    inputs = {}
    for name, language in wanted:
        path = ids.responses_path(slug, name, language, mode)
        rows = read_rows(directory / path, GenerationRow, path)
        if not rows:
            raise DriftmapError(ExitCode.INTEGRITY, f"{path} is missing or empty in the responses repo")
        check_unique(rows, path)
        stray = next(
            (row for row in rows if (_slug(row.checkpoint), row.set, row.language) != (slug, name, language)), None
        )
        if stray is not None:
            found = f"{stray.checkpoint} {stray.set}/{stray.language}"
            raise DriftmapError(ExitCode.INTEGRITY, f"{path} holds a row of {found}")
        inputs[name, language] = rows[: limit or None]
    return inputs
