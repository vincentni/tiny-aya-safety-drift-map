"""Harmful prompts held out of the evaluation set: the base model as served against the same model under the
safety patch, per language and pooled, with the paired change between the two.

The prompts are a fresh sample of the red-teaming dataset that shares no prompt with the evaluation set and none
near one, so no text in a system message was written with them in view. Beside them stand the same two conditions
over the evaluation prompts, and each change over the held-out prompts minus the change there.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from ..core.records import JudgeConfig, LabelRow, Record, Seeds
from ..core.schemas import DriftmapError, ExitCode
from ..core.settings import Settings
from . import build as table
from .breakdown import CONDITION_NAMES, Breakdown, _change, _p, _table
from .rates import refusal_rate, safe_rate, wilson
from .stats import BOOTSTRAP_SEED, change_gap, paired_delta

SCHEMA_HELDOUT = "driftmap.heldout/1"
# the group of prompts beside the evaluation sets, and its one set
GROUP, SET = "heldout", "harmful"
# the two prompt conditions compared, the earlier first
CONDITIONS = ("keep", "patch")
ALL = "all"
METRICS = {"harmful": "Harmful answers", "refusal": "Refusals"}
COLUMN = "baseline"

# harmful answers and scored answers per condition and scope
Counts = Mapping[tuple[str, str], tuple[int, int]]


class HeldoutRate(Record):
    """Answers of one kind among the scored answers of one condition in one scope: a language, or all of them."""

    condition: str
    scope: str
    metric: str
    k: int
    n: int
    rate: float | None
    ci: tuple[float, float] | None


class HeldoutChange(Record):
    """The rate under the safety patch minus the rate as served, over the prompts scored under both."""

    scope: str
    metric: str
    n_pairs: int
    delta: float | None
    ci: tuple[float, float] | None
    p_value: float | None


class HeldoutCategory(Record):
    """Harmful answers among the scored answers to the prompts of one harm category, all languages; a prompt of
    several categories counts in each."""

    condition: str
    harm_category: str
    k: int
    n: int
    rate: float | None
    ci: tuple[float, float] | None


class HeldoutGap(Record):
    """The change over the held-out prompts minus the change over the evaluation prompts; the interval resamples
    both sets of prompts."""

    scope: str
    metric: str
    delta: float | None
    ci: tuple[float, float] | None


class HeldoutInSample(Record):
    """The same two conditions over the harmful prompts of the evaluation set, from their stored labels."""

    prompts: int
    unscored: dict[str, int]
    # stored breakdowns of the evaluation prompts whose counts of harmful answers these rates equal
    equal_to: list[str]
    rates: list[HeldoutRate]
    changes: list[HeldoutChange]
    gaps: list[HeldoutGap]


class HeldoutTables(Record):
    schema_: Literal["driftmap.heldout/1"] = Field(alias="schema")
    checkpoint: str
    prompts_revision: str
    conditions: dict[str, str]
    prompts: int
    # answers without a label of the judge, per condition; they are in no rate
    unscored: dict[str, int]
    scopes: list[str]
    rates: list[HeldoutRate]
    changes: list[HeldoutChange]
    categories: list[HeldoutCategory]
    judge: JudgeConfig | None
    seeds: Seeds
    in_sample: HeldoutInSample | None = None


def _counted(rows: Sequence[LabelRow], metric: str) -> tuple[int, int]:
    """Hits of ``metric`` and the scored answers they are counted over."""
    if metric == "refusal":
        return refusal_rate(rows)
    safe = safe_rate(rows)
    return safe.n - safe.k, safe.n


def _moved(before: Sequence[LabelRow], after: Sequence[LabelRow], metric: str) -> tuple:
    """Pairs, change, interval and p of ``metric``, the later rows minus the earlier."""
    return paired_delta(before, after, "refusal") if metric == "refusal" else _change(list(before), list(after))


def _within(labels: Mapping[str, Sequence[LabelRow]], languages: Sequence[str]) -> list[list[LabelRow]]:
    """The rows of each condition, in the order compared, that are of ``languages``."""
    return [[row for row in labels[condition] if row.language in languages] for condition in CONDITIONS]


def _figures(
    labels: Mapping[str, Sequence[LabelRow]], scopes: Mapping[str, Sequence[str]]
) -> tuple[list[HeldoutRate], list[HeldoutChange]]:
    """The rate of each metric per scope and condition, and its change between the conditions."""
    rates, changes = [], []
    for scope, languages in scopes.items():
        within = _within(labels, languages)
        for metric in METRICS:
            for condition, rows in zip(CONDITIONS, within, strict=True):
                k, n = _counted(rows, metric)
                fields = {"k": k, "n": n, "rate": k / n if n else None, "ci": wilson(k, n)}
                rates.append(HeldoutRate(condition=condition, scope=scope, metric=metric, **fields))
            n_pairs, delta, ci, p_value = _moved(*within, metric)
            found = {"n_pairs": n_pairs, "delta": delta, "ci": ci, "p_value": p_value}
            changes.append(HeldoutChange(scope=scope, metric=metric, **found))
    return rates, changes


def _unscored(labels: Mapping[str, Sequence[LabelRow]]) -> dict[str, int]:
    return {name: len(labels[name]) - _counted(labels[name], "harmful")[1] for name in CONDITIONS}


def stored_counts(found: Breakdown) -> dict[tuple[str, str], tuple[int, int]]:
    """Harmful answers and scored answers of the base model per condition and scope, as a breakdown of the
    evaluation prompts holds them."""
    return {
        (row.condition, row.language or ALL): (row.k, row.n)
        for row in found.rates
        if (row.set, row.column) == (SET, COLUMN) and row.by in (ALL, "language") and row.condition in CONDITIONS
    }


def _in_sample(
    labels: Mapping[str, Sequence[LabelRow]],
    evaluation: Mapping[str, Sequence[LabelRow]],
    stored: Mapping[str, Counts],
    scopes: Mapping[str, Sequence[str]],
) -> HeldoutInSample:
    """The figures of the evaluation prompts and the gap of each held-out change to the change over them."""
    served, patched = ({row.prompt_id for row in evaluation[condition]} for condition in CONDITIONS)
    if not served or served != patched:
        found = f"{len(served)} under the {CONDITION_NAMES[CONDITIONS[0]]}, {len(patched)} under the"
        fault = f"the evaluation prompts have labels for {found} {CONDITION_NAMES[CONDITIONS[1]]}"
        raise DriftmapError(ExitCode.INTEGRITY, f"{fault}, not for the same prompts under both")
    shared = len(served & {row.prompt_id for row in labels[CONDITIONS[0]]})
    if shared:
        raise DriftmapError(ExitCode.INTEGRITY, f"{shared} held-out prompts are evaluation prompts")
    rates, changes = _figures(evaluation, scopes)
    counted = {(row.condition, row.scope): (row.k, row.n) for row in rates if row.metric == "harmful"}
    equal = []
    for name, counts in stored.items():
        for (condition, scope), theirs in counts.items():
            ours = counted.get((condition, scope), theirs)
            if ours != theirs:
                held = f"{name} holds {theirs[0]}/{theirs[1]} harmful answers for the {CONDITION_NAMES[condition]}"
                given = f"the labels of the evaluation prompts give {ours[0]}/{ours[1]}"
                raise DriftmapError(ExitCode.INTEGRITY, f"{held}, {scope}; {given}")
        if all((condition, ALL) in counts for condition in CONDITIONS):
            equal.append(name)
    gaps = []
    for scope, languages in scopes.items():
        pairs = [tuple(_within(rows, languages)) for rows in (labels, evaluation)]
        for metric in METRICS:
            gap = change_gap(*pairs, "refusal" if metric == "refusal" else "safe")
            delta, ci = gap.delta, gap.ci
            if delta is not None and metric != "refusal":
                # the harmful-answer rate moves against the safe rate; adding 0.0 turns -0.0 into 0.0
                delta, ci = 0.0 - delta, (0.0 - ci[1], 0.0 - ci[0])
            gaps.append(HeldoutGap(scope=scope, metric=metric, delta=delta, ci=ci))
    return HeldoutInSample(
        prompts=len(served), unscored=_unscored(evaluation), equal_to=equal, rates=rates, changes=changes, gaps=gaps
    )


def build(
    labels: Mapping[str, Sequence[LabelRow]],
    categories: Mapping[str, Sequence[str]],
    settings: Settings,
    *,
    checkpoint: str,
    prompts_revision: str,
    mode: str = "full",
    evaluation: Mapping[str, Sequence[LabelRow]] | None = None,
    stored: Mapping[str, Counts] | None = None,
) -> HeldoutTables:
    """Rates per condition and scope, the change between the conditions, and the harmful answers by category.

    ``labels`` holds the label rows of the held-out prompts per condition and ``categories`` the harm categories
    of every held-out prompt by its id. ``evaluation`` holds the label rows of the evaluation prompts per
    condition, of which the harmful set is read; with it the table carries their figures and the gap of each
    held-out change to theirs. ``stored`` holds, by file name, what ``stored_counts`` reads of the breakdowns of
    the evaluation prompts. Raises DriftmapError when a condition lacks a label for a held-out prompt or holds one
    for another prompt, when the label sets were judged under different configs, when the evaluation prompts are
    not labelled for the same prompts under both conditions or share a prompt with the held-out ones, and when a
    stored breakdown counts otherwise than their labels.
    """
    for condition in CONDITIONS:
        held = {row.prompt_id for row in labels.get(condition, ())}
        if held != set(categories):
            lacking, stray = len(set(categories) - held), len(held - set(categories))
            found = f"{CONDITION_NAMES[condition]}: {lacking} of the {len(categories)} held-out prompts have no label"
            raise DriftmapError(ExitCode.INTEGRITY, f"{found}, {stray} labels are of another prompt")
    judges = {name: table._judge(CONDITION_NAMES[name], list(labels[name]), settings, mode) for name in CONDITIONS}
    if evaluation is not None:
        evaluation = {name: [row for row in evaluation.get(name, ()) if row.set == SET] for name in CONDITIONS}
        for name in CONDITIONS:
            column = f"{CONDITION_NAMES[name]}, evaluation prompts"
            judges[column] = table._judge(column, evaluation[name], settings, mode)
    table._one_judge(judges)
    held = {row.language for rows in labels.values() for row in rows}
    languages = [code for code in settings.languages if code in held] + sorted(held - set(settings.languages))
    scopes = {code: (code,) for code in languages} | {ALL: tuple(languages)}
    rates, changes = _figures(labels, scopes)
    by_category = []
    for name in sorted({name for names in categories.values() for name in names}):
        for condition in CONDITIONS:
            k, n = _counted([row for row in labels[condition] if name in categories[row.prompt_id]], "harmful")
            fields = {"k": k, "n": n, "rate": k / n if n else None, "ci": wilson(k, n)}
            by_category.append(HeldoutCategory(condition=condition, harm_category=name, **fields))
    return HeldoutTables(
        schema=SCHEMA_HELDOUT,
        checkpoint=checkpoint,
        prompts_revision=prompts_revision,
        conditions={condition: CONDITION_NAMES[condition] for condition in CONDITIONS},
        prompts=len(categories),
        unscored=_unscored(labels),
        scopes=list(scopes),
        rates=rates,
        changes=changes,
        categories=by_category,
        judge=next((judge for judge in judges.values() if judge), None),
        seeds=Seeds(seed=settings.seed, bootstrap=BOOTSTRAP_SEED),
        in_sample=_in_sample(labels, evaluation, stored or {}, scopes) if evaluation is not None else None,
    )


def _rate(row: HeldoutRate | HeldoutCategory) -> str:
    if not row.n:
        return "n = 0"
    low, high = row.ci
    return f"{row.rate:.1%} ({row.k}/{row.n}; 95% CI {low:.1%} to {high:.1%})"


def _points(row: HeldoutChange | HeldoutGap) -> str:
    # adding 0.0 turns a rounded -0.0 into 0.0
    points, low, high = (round(value * 100, 1) + 0.0 for value in (row.delta, *row.ci))
    return f"{points:+.1f} points (95% CI {low:+.1f} to {high:+.1f}"


def _moved_cell(row: HeldoutChange) -> str:
    return "n/a" if row.delta is None else f"{_points(row)}; {_p(row.p_value)}; {row.n_pairs} pairs)"


def _gap_cell(row: HeldoutGap) -> str:
    return "n/a" if row.delta is None else f"{_points(row)})"


def _by_scope(
    tables: HeldoutTables, rates: Sequence[HeldoutRate], changes: Sequence[HeldoutChange], metric: str
) -> list[list[str]]:
    """One line per scope: the rate under each condition, then the change."""
    cells = {(row.scope, row.condition): _rate(row) for row in rates if row.metric == metric}
    moved = {row.scope: _moved_cell(row) for row in changes if row.metric == metric}
    return [[scope, *(cells[scope, condition] for condition in CONDITIONS), moved[scope]] for scope in tables.scopes]


def _left(tables: HeldoutTables, unscored: Mapping[str, int]) -> str:
    return ", ".join(f"{tables.conditions[name]} {count}" for name, count in unscored.items())


def render(tables: HeldoutTables) -> str:
    """Markdown: what was read, one table per metric by language, the harmful answers by harm category, and the
    same conditions over the evaluation prompts when the table holds them."""
    first, second = (tables.conditions[condition] for condition in CONDITIONS)
    judged = f" Judge: `{tables.judge.repo}` under `{tables.judge.prompt_format}`." if tables.judge else ""
    out = [
        "# Held-out harmful prompts: the base model as served and under the safety patch",
        "",
        f"{tables.prompts} harmful prompts of the red-teaming dataset that are no evaluation prompt and near none, "
        f"read at `{tables.prompts_revision[:12]}` of the evaluation dataset. Checkpoint `{tables.checkpoint}`, "
        f"answered once as served and once under the safety patch.{judged} A rate is over the scored answers "
        f"(without a label: {_left(tables, tables.unscored)}); a change is the rate under the patch minus the rate "
        "as served over the prompts scored under both, with a paired-bootstrap 95% interval and an exact sign-test p.",
        "",
    ]
    head = ["language", first, second, "change"]
    for metric, title in METRICS.items():
        out += [f"## {title}", "", *_table(head, _by_scope(tables, tables.rates, tables.changes, metric))]
    names = list(dict.fromkeys(row.harm_category for row in tables.categories))
    cells = {(row.harm_category, row.condition): _rate(row) for row in tables.categories}
    lines = [[name, *(cells[name, condition] for condition in CONDITIONS)] for name in names]
    out += ["## Harmful answers by harm category, all languages", ""]
    out += ["A prompt of several categories counts in each.", "", *_table(["harm category", first, second], lines)]
    known = tables.in_sample
    if known:
        equal = ", ".join(f"`{name}`" for name in known.equal_to)
        out += [
            "## The same conditions on the evaluation prompts",
            "",
            f"The {known.prompts} harmful prompts of the evaluation set under the same two conditions, from their "
            f"stored labels (without a label: {_left(tables, known.unscored)})."
            + (f" Their counts of harmful answers equal those of {equal}." if equal else "")
            + " The last column is the change over the held-out prompts minus the change over these, with a 95% "
            "interval from resampling both sets of prompts; an interval that holds 0 does not tell the two changes "
            "apart.",
            "",
        ]
        for metric, title in METRICS.items():
            gaps = {row.scope: _gap_cell(row) for row in known.gaps if row.metric == metric}
            lines = [[*line, gaps[line[0]]] for line in _by_scope(tables, known.rates, known.changes, metric)]
            out += [f"### {title}", "", *_table([*head, "held-out change minus this change"], lines)]
    return "\n".join(out)
