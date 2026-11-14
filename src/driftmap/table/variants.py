"""Script variants against their parent prompts: per prompt condition and column, paired harmful-response rates."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field

from ..core import ids
from ..core.records import JudgeConfig, LabelRow, Mode, Record, RunRecord, Seeds
from ..core.schemas import DriftmapError, ExitCode
from ..core.settings import Settings
from ..data import variants as data
from . import build as table
from .breakdown import COLUMN_NAMES, CONDITION_NAMES, _table, change_cell
from .stats import BOOTSTRAP_SEED

SCHEMA_VARIANTS = "driftmap.variants/1"
# what a variant prompt row says of its parent
LINK = ("language", "prompt_id", "parent_language", "parent_prompt_id")


class Source(NamedTuple):
    """What a finished run over the script variants read: its checkpoint and the commit of the variant prompts."""

    checkpoint: str
    variants_revision: str


class Side(NamedTuple):
    """One column's labels under one condition: those of the parent prompts, those of the variants, and the variant
    prompt rows, which tie a variant to its parent."""

    source: Source
    parents: list[LabelRow]
    variants: list[LabelRow]
    links: list[Mapping[str, Any]]


class VariantColumn(Record):
    """Whose labels a column compares under one condition; ``left_out`` of the ``judged`` variant labels are those
    the judge read as another request than their parent."""

    condition: str
    column: str
    checkpoint: str
    variants_revision: str
    judged: int
    left_out: int


class VariantChange(Record):
    """A variant against its parent prompts over the pairs scored on both sides.

    ``parent``, ``variant`` and ``delta`` (variant minus parent) are harmful-response rates, or mean chrF for the
    benign set. ``code`` reads ``all-Latn`` or ``all-mixed``, with no ``language``, where it pools one rewrite.
    """

    condition: str
    column: str
    set: str
    code: str
    language: str | None
    rewrite: str
    metric: str
    n_pairs: int
    parent: float | None
    variant: float | None
    delta: float | None
    ci: tuple[float, float] | None
    p_value: float | None


class VariantTables(Record):
    schema_: Literal["driftmap.variants/1"] = Field(alias="schema")
    # key to the name shown, in the order built
    conditions: dict[str, str]
    columns: dict[str, str]
    sources: list[VariantColumn]
    rows: list[VariantChange]
    judge: JudgeConfig
    seeds: Seeds


def sources(records: Iterable[RunRecord], settings: Settings, mode: Mode, preamble: str = "keep") -> dict[str, Source]:
    """Per column, in table order, what its latest ok generate run of ``mode`` over the variants read under
    ``preamble``."""
    found = {}
    for run in sorted(records, key=lambda run: run.started):
        models = [f"{ref.repo}@{ref.revision}" for ref in run.inputs if ref.kind == "model"]
        datasets = [ref.revision for ref in run.inputs if ref.kind == "dataset"]
        same_tree = (run.kind, run.mode, run.status, ids.preamble_of(run.notes)) == ("generate", mode, "ok", preamble)
        if not same_tree or not ids.variants_of(run.notes) or not models or not datasets:
            continue
        column = table.column_of(models[0], settings, mode)
        if column:
            # the run records the evaluation dataset's pinned revision, then the commit of the variant prompts
            found[column] = Source(models[0], datasets[-1])
    return {column: found[column] for column in table.COLUMNS if column in found}


def link(row: Mapping[str, Any]) -> dict[str, str]:
    """What ties one variant prompt row to its parent; LookupError for a row of another shape."""
    return {key: row[key] for key in LINK}


def read(
    data: Path, found: Mapping[str, Mapping[str, Source]], links: Mapping[str, list[Mapping[str, Any]]], mode: Mode
) -> dict[str, dict[str, Side]]:
    """The labels of each source's checkpoint in a copy of the evaluation dataset, by condition, then column.

    ``found`` holds the sources the same way; ``links`` are the variant prompt rows of each commit a source names.
    """
    sides = {}
    for condition, columns in found.items():
        parents, rewritten = (table._read(data / ids.tree(mode, condition, flag) / "labels") for flag in (False, True))
        sides[condition] = {
            column: Side(
                source,
                [row for row in parents if row.checkpoint == source.checkpoint],
                [row for row in rewritten if row.checkpoint == source.checkpoint],
                links[source.variants_revision],
            )
            for column, source in columns.items()
        }
    return sides


def build(sides: Mapping[str, Mapping[str, Side]], settings: Settings, *, mode: Mode = "full") -> VariantTables:
    """Compare each column's variants with their parents; ``sides`` holds the labels by condition, then column.

    A variant the judge read as another request than its parent is left out and counted. Refuses labels of more
    than one judge config, and labels with no variant to compare.
    """
    listed, rows, judges = [], [], {}
    for condition in (name for name in ids.CONDITIONS if name in sides):
        for column in (name for name in table.COLUMNS if name in sides[condition]):
            side = sides[condition][column]
            kept, dropped = data.drop_flipped(side.parents, side.variants, side.links)
            found = data.paired(side.parents, kept, side.links, seed=BOOTSTRAP_SEED)
            if not found:
                continue
            listed.append(
                VariantColumn(
                    condition=condition,
                    column=column,
                    checkpoint=side.source.checkpoint,
                    variants_revision=side.source.variants_revision,
                    judged=len(side.variants),
                    left_out=len(dropped),
                )
            )
            for result in found:
                language, rewrite = data.CODES.get(result.code) or (None, data.CODES[_pooled(result.code)][1])
                changed = {"condition": condition, "column": column, "language": language, "rewrite": rewrite}
                rows.append(VariantChange(**changed, **result._asdict()))
            sets = {result.set for result in found}
            compared = [row for row in (*side.parents, *kept) if row.set in sets]
            judges[f"{condition} {column}"] = table._judge(column, compared, settings, mode)
    table._one_judge(judges)
    judge = next((judge for judge in judges.values() if judge), None)
    if not rows or judge is None:
        raise DriftmapError(ExitCode.USAGE, "no judged labels of script variants with a parent prompt")
    return VariantTables(
        schema=SCHEMA_VARIANTS,
        conditions={row.condition: CONDITION_NAMES.get(row.condition, row.condition) for row in listed},
        columns={
            column: COLUMN_NAMES.get(column, column)
            for column in table.COLUMNS
            if any(row.column == column for row in listed)
        },
        sources=listed,
        rows=rows,
        judge=judge,
        seeds=Seeds(seed=settings.seed, bootstrap=BOOTSTRAP_SEED),
    )


def _pooled(code: str) -> str:
    """A variant code of the rewrite that a pooled code such as ``all-Latn`` stands for."""
    return next(known for known in data.CODES if known.partition("-")[2] == code.partition("-")[2])


def _rate(row: VariantChange, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if row.metric == "harmful" else f"{100 * value:.1f}"


def _title(name: str) -> str:
    return f"{name[:1].upper()}{name[1:]}"


def render(tables: VariantTables) -> str:
    """Markdown of the comparison: per condition and column, each variant beside its parent prompts."""
    judge = tables.judge.repo.rpartition("/")[2]
    out = [
        "# Script variants against their parent prompts",
        "",
        "The harmful prompts again in Latin script (`-Latn`) or mixed with English (`-mixed`), each beside the prompt "
        "it was made from. Rates are the share of responses the judge "
        f"({judge}) called harmful, over the pairs scored on both sides; a change is the variant rate minus the "
        "parent rate, with a paired-bootstrap 95% interval and an exact sign-test p. "
        "`all-Latn` and `all-mixed` pool the variants of one rewrite. "
        "A variant the judge read as another request than its parent is left out.",
        "",
    ]
    for condition, shown in tables.conditions.items():
        out += [f"## {_title(shown)}", ""]
        for source in (source for source in tables.sources if source.condition == condition):
            lines = [
                [row.set, row.code, _rate(row, row.parent), _rate(row, row.variant), change_cell(row)]
                for row in tables.rows
                if (row.condition, row.column) == (condition, source.column)
            ]
            out += [
                f"### {_title(tables.columns[source.column])}",
                "",
                f"Checkpoint `{source.checkpoint}`, variant prompts at `{source.variants_revision[:12]}`: "
                f"{source.judged} variant labels, {source.left_out} left out.",
                "",
                *_table(["Set", "Variant", "Parent prompts", "Variant prompts", "Change"], lines),
            ]
    return "\n".join(out).rstrip("\n") + "\n"
