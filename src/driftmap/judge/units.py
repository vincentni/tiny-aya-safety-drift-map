"""The response files a judge run labels: one per harmful language and the English over-refusal set.

In place of those, a run may label the harmful prompts of one group beside the evaluation sets that a checkpoint
has responses for: the script variants, or the prompts held out of the evaluation set.
"""

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
# named alone in place of sets: the harmful script variants, or the harmful prompts held out of the evaluation set
VARIANTS, HELDOUT = GROUPS = ("variants", "heldout")

Unit = tuple[str, str]


def units(settings: Settings) -> list[Unit]:
    """The (set, language) files a judge run labels, in order."""
    return [("harmful", language) for language in settings.languages] + [("xstest", XSTEST_LANGUAGE)]


def _slug(checkpoint: str) -> str:
    try:
        return ids.slug(checkpoint)
    except ValueError:
        return checkpoint


def variant_units(
    directory: Path, slug: str, mode: str, preamble: str = "keep", group: str = VARIANTS
) -> list[Unit]:
    """The harmful files of a checkpoint in ``group`` among the responses under ``directory``, by the code of each:
    a variant's for the script variants, a language's for the held-out prompts."""
    folder = (directory / ids.responses_path(slug, "harmful", "*", mode, preamble, variants=group)).parent
    return [("harmful", path.stem) for path in sorted(folder.glob("*.jsonl"))]


def read_inputs(
    directory: Path,
    slug: str,
    wanted: Sequence[Unit],
    mode: str,
    limit: int | None,
    preamble: str = "keep",
    variants: bool | str = False,
) -> dict[Unit, list[GenerationRow]]:
    """Generation rows per unit; DriftmapError on a missing file, a repeated prompt or a row of another unit.

    With ``variants`` the units are files of the script variants, or of the group it names, and their language is
    the code of the file.
    """
    inputs = {}
    for name, language in wanted:
        path = ids.responses_path(slug, name, language, mode, preamble, variants)
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
