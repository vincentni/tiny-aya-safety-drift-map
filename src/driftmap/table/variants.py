"""Script variants against their parent prompts: per prompt condition and column, paired harmful-response rates.

The benign answers of the same checkpoint are the control: an answer that is empty or cut off is no refusal.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field

from ..core import ids
from ..core.records import GenerationRow, JudgeConfig, LabelRow, Mode, Record, RunRecord, Seeds
from ..core.schemas import DriftmapError, ExitCode
from ..core.settings import Settings
from ..data import variants as data
from ..judge.incremental import read_rows
from . import build as table
from .breakdown import COLUMN_NAMES, CONDITION_NAMES, FINE_TUNE_NAMES, _table, change_cell
from .stats import BOOTSTRAP_SEED

SCHEMA_VARIANTS = "driftmap.variants/1"
# what a variant prompt row says of its parent
LINK = ("language", "prompt_id", "parent_language", "parent_prompt_id")


class Source(NamedTuple):
    """What a finished run over the script variants read: its checkpoint and the commit of the variant prompts."""

    checkpoint: str
    variants_revision: str


class Side(NamedTuple):
    """One column's rows under one condition.

    The labels of the parent prompts and of the variants, the variant prompt rows, which tie a variant to its
    parent, and the stored answers to the benign parents and variants.
    """

    source: Source
    parents: list[LabelRow]
    variants: list[LabelRow]
    links: list[Mapping[str, Any]]
    parent_answers: Sequence[GenerationRow] = ()
    variant_answers: Sequence[GenerationRow] = ()


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


class VariantAnswers(Record):
    """Benign answers to a variant against those to its parents, over the pairs generated on both sides.

    ``*_failed`` is the share of answers that are empty or were cut at the token limit, ``*_tokens`` the mean
    generated length; ``code`` and ``language`` read as in a ``VariantChange``.
    """

    condition: str
    column: str
    code: str
    language: str | None
    rewrite: str
    n_pairs: int
    parent_failed: float | None
    variant_failed: float | None
    parent_tokens: float | None
    variant_tokens: float | None


class VariantTables(Record):
    schema_: Literal["driftmap.variants/1"] = Field(alias="schema")
    # the fine-tune of the drifted column
    fine_tune: str
    # key to the name shown, in the order built
    conditions: dict[str, str]
    columns: dict[str, str]
    sources: list[VariantColumn]
    rows: list[VariantChange]
    answers: list[VariantAnswers]
    judge: JudgeConfig
    seeds: Seeds


def sources(
    records: Iterable[RunRecord], settings: Settings, mode: Mode, preamble: str = "keep", fine_tune: str | None = None
) -> dict[str, Source]:
    """Per column, in table order, what its latest ok generate run of ``mode`` over the variants read under
    ``preamble``; with ``fine_tune``, a run over a checkpoint of the other fine-tune is no column's."""
    found = {}
    for run in sorted(records, key=lambda run: run.started):
        models = [f"{ref.repo}@{ref.revision}" for ref in run.inputs if ref.kind == "model"]
        datasets = [ref.revision for ref in run.inputs if ref.kind == "dataset"]
        same_tree = (run.kind, run.mode, run.status, ids.preamble_of(run.notes)) == ("generate", mode, "ok", preamble)
        if not same_tree or not ids.variants_of(run.notes) or not models or not datasets:
            continue
        column = table.column_of(models[0], settings, mode, fine_tune)
        if column:
            # the run records the evaluation dataset's pinned revision, then the commit of the variant prompts
            found[column] = Source(models[0], datasets[-1])
    return {column: found[column] for column in table.COLUMNS if column in found}


def link(row: Mapping[str, Any]) -> dict[str, str]:
    """What ties one variant prompt row to its parent; LookupError for a row of another shape."""
    return {key: row[key] for key in LINK}


def _answers(folder: Path, checkpoint: str) -> list[GenerationRow]:
    """The stored benign answers of ``checkpoint`` under ``<folder>/<slug>/<set>/<language>.jsonl``."""
    found = []
    for path in sorted(folder.glob(f"{ids.slug(checkpoint)}/*/*.jsonl")):
        found += read_rows(path, GenerationRow, path.relative_to(folder).as_posix())
    return [row for row in found if row.set == "benign" and row.checkpoint == checkpoint]


def read(
    data: Path,
    found: Mapping[str, Mapping[str, Source]],
    links: Mapping[str, list[Mapping[str, Any]]],
    mode: Mode,
    responses: Path | None = None,
) -> dict[str, dict[str, Side]]:
    """The rows of each source's checkpoint, by condition, then column, as ``build`` takes them.

    ``data`` is a copy of the evaluation dataset with the labels, ``responses`` one of the responses repo with the
    benign answers; ``found`` holds the sources by condition, then column, and ``links`` the variant prompt rows of
    each commit a source names.
    """
    sides = {}
    for condition, columns in found.items():
        trees = [ids.tree(mode, condition, flag) for flag in (False, True)]
        parents, rewritten = (table._read(data / tree / "labels") for tree in trees)
        sides[condition] = {
            column: Side(
                source,
                [row for row in parents if row.checkpoint == source.checkpoint],
                [row for row in rewritten if row.checkpoint == source.checkpoint],
                links[source.variants_revision],
                *(_answers(responses / tree / "responses", source.checkpoint) if responses else [] for tree in trees),
            )
            for column, source in columns.items()
        }
    return sides


def _rewrite(code: str) -> tuple[str | None, str]:
    """Parent language and rewrite of a variant code; a pooled code such as ``all-Latn`` has no language."""
    pooled = next(known for known in data.CODES if known.partition("-")[2] == code.partition("-")[2])
    return data.CODES.get(code) or (None, data.CODES[pooled][1])


def build(
    sides: Mapping[str, Mapping[str, Side]], settings: Settings, *, mode: Mode = "full", fine_tune: str = "standard"
) -> VariantTables:
    """Compare each column's variants with their parents; ``sides`` holds the rows by condition, then column.

    A variant the judge read as another request than its parent is left out and counted. ``fine_tune`` names the
    fine-tune of the drifted column. Refuses labels of more than one judge config, and labels with no variant to
    compare.
    """
    listed, rows, control, judges = [], [], [], {}
    for condition in (name for name in ids.CONDITIONS if name in sides):
        for column in (name for name in table.COLUMNS if name in sides[condition]):
            side = sides[condition][column]
            rows_of = (side.parents, side.variants, side.links, side.parent_answers, side.variant_answers)
            found = data.compare(*rows_of, same_request=True)
            if not found.harmful:
                continue
            listed.append(
                VariantColumn(
                    condition=condition,
                    column=column,
                    checkpoint=side.source.checkpoint,
                    variants_revision=side.source.variants_revision,
                    judged=len(side.variants),
                    left_out=found.dropped,
                )
            )
            for result in (*found.harmful, *found.benign):
                language, rewrite = _rewrite(result.code)
                where = {"condition": condition, "column": column, "language": language, "rewrite": rewrite}
                if isinstance(result, data.Paired):
                    rows.append(VariantChange(**where, **result._asdict()))
                else:
                    control.append(VariantAnswers(**where, **result._asdict()))
            sets = {result.set for result in found.harmful}
            compared = [row for row in (*side.parents, *side.variants) if row.set in sets]
            judges[f"{condition} {column}"] = table._judge(column, compared, settings, mode)
    table._one_judge(judges)
    judge = next((judge for judge in judges.values() if judge), None)
    if not rows or judge is None:
        raise DriftmapError(ExitCode.USAGE, "no judged labels of script variants with a parent prompt")
    names = COLUMN_NAMES | ({"drifted": FINE_TUNE_NAMES[fine_tune]} if fine_tune in FINE_TUNE_NAMES else {})
    return VariantTables(
        schema=SCHEMA_VARIANTS,
        fine_tune=fine_tune,
        conditions={row.condition: CONDITION_NAMES.get(row.condition, row.condition) for row in listed},
        columns={
            column: names.get(column, column) for column in table.COLUMNS if any(row.column == column for row in listed)
        },
        sources=listed,
        rows=rows,
        answers=control,
        judge=judge,
        seeds=Seeds(seed=settings.seed, bootstrap=BOOTSTRAP_SEED),
    )


def _rate(row: VariantChange, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if row.metric == "harmful" else f"{100 * value:.1f}"


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _tokens(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


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
    if tables.answers:
        out[-2] += (
            " The benign control beside each table is the same checkpoint's answers to the benign prompts and their "
            "variants: the share that is empty or cut off at the cap on new tokens, and the mean length in tokens."
        )
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
            control = [
                [row.code, str(row.n_pairs), *(_share(value) for value in (row.parent_failed, row.variant_failed))]
                + [_tokens(value) for value in (row.parent_tokens, row.variant_tokens)]
                for row in tables.answers
                if (row.condition, row.column) == (condition, source.column)
            ]
            if control:
                head = ["Benign variant", "Pairs", "Parents empty or cut off", "Variants empty or cut off"]
                out += _table([*head, "Parent tokens", "Variant tokens"], control)
    return "\n".join(out).rstrip("\n") + "\n"
