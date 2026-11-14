"""What each figure shows, as plain numbers, labels and a title that states the finding."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from ..core.records import ChrfRow, DeltaRow, RateRow, Tables
from ..core.settings import Settings
from ..table.breakdown import (
    CELL,
    COLUMN_NAMES,
    COMPLETE,
    CONDITION_NAMES,
    HARMFUL,
    LENGTHS,
    STATES,
    TRUNCATED,
    Breakdown,
    BreakdownRate,
    Transitions,
    flips,
)
from ..table.build import COLUMNS, Q4_FINE_TUNED, column_of
from ..table.rates import wilson

LANGUAGE_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "ru": "Russian", "hi": "Hindi", "ar": "Arabic"}
# names too long for an axis
CATEGORY_NAMES = {"Harms of Representation Allocation and Quality of Service": "Representation & allocation harms"}
LENGTH_LABELS = {COMPLETE: "Complete answers", TRUNCATED: "Answers cut off at the token cap"}
# how the prompts of a script variant were rewritten, as named on an axis; another goes by its variant's code
REWRITE_NAMES = {"romanized": "Latin script", "codemixed": "Mixed with English"}
HARMFUL_UNIT = "Harmful answers (% of scored responses)"
# metric, its row in the tables, panel title and value axis
METRICS = (
    ("safe", "all", "Harmful answers", "% of scored responses to harmful prompts"),
    ("over_refusal", "en", "Over-refusal (XSTest)", "% of safe prompts refused"),
    ("chrf", "all", "Task score (chrF)", "chrF on held-out benign prompts, 0 to 100"),
)


@dataclass(frozen=True)
class Estimate:
    """A value in the unit of its panel, with the bounds of its 95% interval when it has one."""

    value: float | None
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Series:
    """One colour of a panel: a point per label, and for each a note to print beside it when there is one."""

    name: str
    points: tuple[Estimate, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Panel:
    """One set of axes: a point per label and series; ``unit`` names the value axis."""

    title: str
    unit: str
    labels: tuple[str, ...]
    series: tuple[Series, ...]


@dataclass(frozen=True)
class Chart:
    title: str
    panels: tuple[Panel, ...]


@dataclass(frozen=True)
class Cell:
    """One heat-map cell; ``rate`` is in percent, None under the floor on scored responses."""

    k: int
    n: int
    rate: float | None


@dataclass(frozen=True)
class Heat:
    title: str
    unit: str
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    # a row per category, a cell per language; None where there is no prompt
    cells: tuple[tuple[Cell | None, ...], ...]
    # scored responses under which a cell has no rate
    floor: int


@dataclass(frozen=True)
class Band:
    """Prompts that went from one outcome to another, with the span they take on each bar in percent of all."""

    before: str
    after: str
    n: int
    left: tuple[float, float]
    right: tuple[float, float]


@dataclass(frozen=True)
class Flow:
    """One condition: prompts per outcome in the earlier and the later column, and the bands between them."""

    title: str
    note: str
    names: tuple[str, str]
    n: int
    before: dict[str, int]
    after: dict[str, int]
    bands: tuple[Band, ...]


@dataclass(frozen=True)
class Flows:
    title: str
    states: tuple[str, ...]
    panels: tuple[Flow, ...]


def _percent(rate: float | None, ci: tuple[float, float] | None, *, turned: bool = False) -> Estimate:
    """A rate and its interval in percent; ``turned`` gives the complement, a safe rate as a harmful one."""
    if rate is None:
        return Estimate(None)
    low, high = ci or (None, None)
    if turned:
        rate, low, high = 1 - rate, None if high is None else 1 - high, None if low is None else 1 - low
    return Estimate(*(None if value is None else value * 100 for value in (rate, low, high)))


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _condition(condition: str) -> str:
    return _capital(CONDITION_NAMES.get(condition, condition))


class VariantRow(NamedTuple):
    """A script variant against its parent prompts for one column under one condition.

    ``parent`` and ``variant`` are harmful-response rates over the ``n_pairs`` prompts scored on both sides, ``delta``
    is variant minus parent with its interval; ``language`` is None for the row pooled over a rewrite's languages.
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


class Variants(NamedTuple):
    """The script-variant comparisons of a stored file: the conditions and columns by shown name, and the rows."""

    conditions: dict[str, str]
    columns: dict[str, str]
    rows: list[VariantRow]


def variants(stored: Mapping[str, Any]) -> Variants:
    """The comparisons of a stored variants file; fields the figure does not use are left unread.

    TypeError, ValueError or LookupError for content of another shape.
    """
    rows = [VariantRow(**{field: row[field] for field in VariantRow._fields}) for row in stored["rows"]]
    for row in rows:
        named = (row.condition, row.column, row.code, row.rewrite, row.metric)
        if not isinstance(row.n_pairs, int) or not all(isinstance(name, str) for name in named):
            raise TypeError("a row names its condition, column, variant and metric and counts its pairs")
    rows = [row._replace(ci=tuple(row.ci) if row.ci else None) for row in rows]
    return Variants(dict(stored["conditions"]), dict(stored["columns"]), rows)


def _points(delta: DeltaRow, *, turned: bool = False, unit: str = " points", interval: str = "") -> str:
    """A paired change with its interval on a scale of 100; ``turned`` reads a change of the safe rate as harmful."""
    sign = -100 if turned else 100
    # adding 0.0 turns a rounded -0.0 into 0.0
    change, *bounds = (round(sign * value, 1) + 0.0 for value in (delta.delta, *delta.ci))
    low, high = sorted(bounds)
    return f"{change:+.1f}{unit} ({interval}{low:+.1f} to {high:+.1f})"


def _tabled(tables: Tables) -> list[str]:
    return [column for column in COLUMNS if column in tables.columns and tables.columns[column].state == "tabled"]


def _row(rows: list, name: str, column: str, settings: Settings) -> RateRow | ChrfRow | None:
    return next((row for row in rows if row.row == name and column_of(row.checkpoint, settings) == column), None)


def _names(tables: Mapping[str, Tables], names: Mapping[str, str]) -> dict[str, str]:
    """The name each column is shown under; a quantized file of the fine-tuned model goes by the tables' name for it."""
    states = [(column, state) for table in tables.values() for column, state in table.columns.items()]
    return {**names, **{column: state.kind for column, state in states if state.kind == Q4_FINE_TUNED}}


def _rows(tables: Tables, metric: str) -> list:
    if metric == "chrf":
        return tables.chrf
    return tables.over_refusal if metric == "over_refusal" else [row for row in tables.safety if row.metric == "safe"]


def harmful_by_language(
    tables: Mapping[str, Tables], settings: Settings, names: Mapping[str, str] = COLUMN_NAMES
) -> Chart:
    """Harmful-response rate per language and pooled: a panel per condition, a series per tabled column.

    The languages run from the weakest down by the first column of the first condition; the pooled rate is last.
    ``names`` gives the name each column is shown under.
    """
    codes = [*settings.languages, "all"]
    names = _names(tables, names)
    found = []
    for condition, table in tables.items():
        series = []
        for column in _tabled(table):
            rows = [_row(_rows(table, "safe"), code, column, settings) for code in codes]
            points = tuple(_percent(row.rate, row.ci, turned=True) if row else Estimate(None) for row in rows)
            series.append(Series(names.get(column, column), points))
        found.append((_condition(condition), series))
    lead = found[0][1][0].points if found[0][1] else (Estimate(None),) * len(codes)
    worst = sorted(range(len(codes) - 1), key=lambda at: (lead[at].value is None, -(lead[at].value or 0.0), at))
    order = [*worst, len(codes) - 1]
    labels = tuple(LANGUAGE_NAMES.get(codes[at], _capital(codes[at])) for at in order)
    panels = []
    for shown, series in found:
        ordered = tuple(Series(one.name, tuple(one.points[at] for at in order)) for one in series)
        panels.append(Panel(shown, HARMFUL_UNIT, labels, ordered))
    rated = [at for at in worst if lead[at].value is not None]
    if not rated:
        return Chart("Harmful answers by language", tuple(panels))
    ends = [f"{lead[at].value:.0f}% in {LANGUAGE_NAMES.get(codes[at], codes[at])}" for at in (rated[0], rated[-1])]
    where = f"{panels[0].series[0].name}, {panels[0].title.lower()}"
    return Chart(f"Harmful answers by language: {', '.join(dict.fromkeys(ends))} ({where})", tuple(panels))


def before_after(tables: Mapping[str, Tables], settings: Settings, names: Mapping[str, str] = COLUMN_NAMES) -> Chart:
    """Harmful-response rate, over-refusal and chrF of each tabled column: a panel per metric, a group per condition.

    The title states the paired changes between the first two columns under the first condition, with intervals.
    """
    columns = list(dict.fromkeys(column for table in tables.values() for column in _tabled(table)))
    names = _names(tables, names)
    labels = tuple(_condition(condition) for condition in tables)
    panels = []
    for metric, name, title, unit in METRICS:
        series = []
        for column in columns:
            points = []
            for table in tables.values():
                row = _row(_rows(table, metric), name, column, settings) if column in _tabled(table) else None
                if row is None:
                    points.append(Estimate(None))
                elif metric == "chrf":
                    points.append(Estimate(None if row.chrf is None else row.chrf * 100))
                else:
                    points.append(_percent(row.rate, row.ci, turned=metric == "safe"))
            series.append(Series(names.get(column, column), tuple(points)))
        panels.append(Panel(title, unit, labels, tuple(series)))
    lead = next(iter(tables.values()))
    between = _tabled(lead)[:2]
    changes = [
        next((row for row in lead.deltas if [row.a, row.b, row.metric, row.row] == [*between, metric, name]), None)
        for metric, name, *_ in METRICS
    ]
    if len(between) < 2 or any(row is None or row.delta is None for row in changes):
        return Chart("Harmful answers, over-refusal and task score", tuple(panels))
    before, after = (_row(_rows(lead, "safe"), "all", column, settings) for column in between)
    harmful = [f"{row.n - row.k} of {row.n}" for row in (before, after)]
    counts = f"{before.n - before.k} to {harmful[1]}" if before.n == after.n else " to ".join(harmful)
    said = (
        f"harmful answers {counts}, {_points(changes[0], turned=True, interval='95% CI ')}",
        f"over-refusal {_points(changes[1])}",
        f"chrF {_points(changes[2], unit='')}",
    )
    scope = f"{labels[0]}, {' to '.join(names.get(column, column) for column in between)}"
    return Chart(f"{scope}: {'; '.join(said)}", tuple(panels))


def _rates(breakdown: Breakdown, condition: str, by: str, column: str) -> dict[tuple, BreakdownRate]:
    """Rates of one column of the first set under ``condition``, by harm category and language."""
    name = breakdown.rates[0].set
    return {
        (row.harm_category, row.language): row
        for row in breakdown.rates
        if (row.set, row.condition, row.by, row.column) == (name, condition, by, column)
    }


def _categories(breakdown: Breakdown) -> list[str]:
    """Harm categories, worst first by the rate of the first column under the first condition."""
    rates = _rates(breakdown, next(iter(breakdown.conditions)), "harm_category", next(iter(breakdown.columns)))
    return sorted(
        (category for category, _ in rates), key=lambda category: (-(rates[category, None].rate or 0.0), category)
    )


def by_category(breakdown: Breakdown) -> Chart:
    """Harmful-response rate per harm category, worst first: a panel per condition, a series per column."""
    categories = _categories(breakdown)
    labels = tuple(CATEGORY_NAMES.get(category, category) for category in categories)
    panels = []
    for condition in breakdown.conditions:
        series = []
        for column, shown in breakdown.columns.items():
            rates = _rates(breakdown, condition, "harm_category", column)
            found = [rates.get((category, None)) for category in categories]
            series.append(Series(shown, tuple(_percent(row.rate, row.ci) if row else Estimate(None) for row in found)))
        panels.append(Panel(_condition(condition), HARMFUL_UNIT, labels, tuple(series)))
    first = [point.value for point in panels[0].series[0].points]
    if not categories or first[0] is None:
        return Chart("Harmful answers by harm category", tuple(panels))
    ranked = [
        f"{label} ({value:.1f}%)" for label, value in zip(labels[:2], first[:2], strict=False) if value is not None
    ]
    where = f"{panels[0].series[0].name}, {panels[0].title.lower()}"
    return Chart(f"Most harmful answers by harm category: {', then '.join(ranked)} ({where})", tuple(panels))


def category_by_language(breakdown: Breakdown) -> Heat:
    """Harmful-response rate of the first column under the first condition, harm category by language."""
    condition = next(iter(breakdown.conditions))
    column, shown = next(iter(breakdown.columns.items()))
    categories = _categories(breakdown)
    languages = [language for _, language in _rates(breakdown, condition, "language", column)]
    cells = _rates(breakdown, condition, CELL, column)
    grid = []
    for category in categories:
        found = [cells.get((category, language)) for language in languages]
        rates = [None if row is None or row.rate is None else row.rate * 100 for row in found]
        grid.append(tuple(Cell(row.k, row.n, rate) if row else None for row, rate in zip(found, rates, strict=True)))
    rows = tuple(CATEGORY_NAMES.get(category, category) for category in categories)
    columns = tuple(LANGUAGE_NAMES.get(language, language) for language in languages)
    rated = [
        (cell.rate, cell.n, at, to)
        for at, line in enumerate(grid)
        for to, cell in enumerate(line)
        if cell and cell.rate is not None
    ]
    title = f"Harmful answers at {shown} by harm category and language"
    if rated:
        # the highest rate; among equals the one with more scored responses
        _, _, at, to = max(rated)
        worst = grid[at][to]
        counts = f"{worst.k} of {worst.n} answers harmful ({worst.rate:.1f}%)"
        title = f"Weakest cell at {shown}: {rows[at]} in {columns[to]}, {counts}"
    unit = f"{HARMFUL_UNIT}, {breakdown.conditions[condition]}"
    return Heat(title, unit, rows, columns, tuple(grid), breakdown.min_cell_n)


def _moves(breakdown: Breakdown, condition: str) -> list[Transitions]:
    """The outcome tables of the first set under ``condition`` for all languages together."""
    here = (breakdown.rates[0].set, condition, "all")
    return [row for row in breakdown.transitions if (row.set, row.condition, row.language) == here]


def _bands(counts: Mapping[str, Mapping[str, int]], total: int) -> tuple[Band, ...]:
    """A band per pair of outcomes with prompts: it leaves its segment in the order of where it goes and enters the
    later segment in the order of where it came from."""
    after = {second: sum(counts[first][second] for first in STATES) for second in STATES}
    bands, left = [], 0
    for first in STATES:
        for second in STATES:
            n = counts[first][second]
            right = sum(after[state] for state in STATES[: STATES.index(second)])
            right += sum(counts[state][second] for state in STATES[: STATES.index(first)])
            if n:
                spans = [(start * 100 / total, (start + n) * 100 / total) for start in (left, right)]
                bands.append(Band(first, second, n, *spans))
            left += n
    return tuple(bands)


def outcome_flows(breakdown: Breakdown) -> Flows:
    """Per condition, the outcomes of the prompts in the first two columns and how many moved between outcomes."""
    panels, moved = [], []
    for condition in breakdown.conditions:
        pooled = [row for row in _moves(breakdown, condition) if row.length_a is None]
        if not pooled:
            continue
        counts = pooled[0].counts
        total = sum(sum(row.values()) for row in counts.values())
        before = {first: sum(counts[first].values()) for first in STATES}
        after = {second: sum(counts[first][second] for first in STATES) for second in STATES}
        scored = [state for state in STATES if state not in (HARMFUL, "unscored")]
        left = {state: counts[HARMFUL][state] for state in scored}
        came = {state: counts[state][HARMFUL] for state in scored}
        moved.append((sum(left.values()), sum(came.values())))
        note = (
            f"No longer harmful: {', '.join(f'{n} to {state}' for state, n in left.items())}. "
            f"Newly harmful: {', '.join(f'{n} from {state}' for state, n in came.items())}."
        )
        names = (breakdown.columns[pooled[0].a], breakdown.columns[pooled[0].b])
        panels.append(Flow(_condition(condition), note, names, total, before, after, _bands(counts, total)))
    if not panels:
        return Flows("Outcome of each harmful prompt", STATES, ())
    out, into = moved[0]
    title = (
        f"{panels[0].title}: {out} harmful answers turned into refusals or harmless answers after fine-tuning, "
        f"{into} went the other way"
    )
    return Flows(title, STATES, tuple(panels))


def _estimates(rows: Iterable) -> tuple[Estimate, ...]:
    return tuple(_percent(row.rate, row.ci) if row else Estimate(None) for row in rows)


def truncation(breakdown: Breakdown) -> Chart:
    """Harmful-response rate of complete answers and of answers cut off at the token cap, noting the share of each.

    A panel per condition and a series per column; no panel when the breakdown read no stored answer. The title
    speaks for the first condition that has two columns.
    """
    name = breakdown.rates[0].set
    panels, capped = [], {}
    for condition in breakdown.conditions:
        series = []
        for column, shown in breakdown.columns.items():
            here = (name, condition, column)
            cut = [
                row.rate
                for row in breakdown.at_cap
                if (row.set, row.condition, row.column, row.language) == (*here, "all")
            ]
            if not cut or cut[0] is None:
                continue
            capped.setdefault(condition, []).append(cut[0] * 100)
            rates = {
                row.length: row
                for row in breakdown.rates
                if (row.set, row.condition, row.column, row.by) == (*here, "length")
            }
            shares = {COMPLETE: 100 - cut[0] * 100, TRUNCATED: cut[0] * 100}
            notes = tuple(f"{shares[length]:.0f}% of answers" for length in LENGTHS)
            series.append(Series(shown, _estimates(rates.get(length) for length in LENGTHS), notes))
        if series:
            labels = tuple(LENGTH_LABELS[length] for length in LENGTHS)
            panels.append(Panel(_condition(condition), HARMFUL_UNIT, labels, tuple(series)))
    lead = next((condition for condition, shares in capped.items() if len(shares) > 1), None)
    moves = {(row.length_a, row.length_b): flips(row.counts) for row in _moves(breakdown, lead)} if lead else {}
    if (TRUNCATED, COMPLETE) not in moves:
        return Chart("Harmful answers by answer length", tuple(panels))
    net = moves[None, None].to_safe - moves[None, None].to_harmful
    shortened = moves[TRUNCATED, COMPLETE].to_safe - moves[TRUNCATED, COMPLETE].to_harmful
    if net <= 0:
        finding = "No fewer harmful answers after fine-tuning"
    # the answers that stopped running into the cap carry at least half of the net fall
    elif 2 * shortened >= net:
        finding = "Fewer harmful answers after fine-tuning, mostly because answers got shorter"
    else:
        finding = "Fewer harmful answers after fine-tuning, and not mostly through shorter answers"
    cap = f"{capped[lead][0]:.0f}% ran into the token cap before, {capped[lead][-1]:.0f}% after"
    return Chart(f"{finding}: {cap} ({_condition(lead).lower()})", tuple(panels))


def script_variants(found: Variants | None) -> Chart:
    """Harmful-response rate of each rewrite against the same prompts as written, pooled over its languages.

    A panel per column and a group per rewrite, the prompts as written and rewritten side by side and the change in
    points noted, under the first condition that has pooled rows; no panel without one.
    """
    pooled = [row for row in found.rows if row.metric == "harmful" and row.language is None] if found else []
    pooled = [row for row in pooled if row.n_pairs and None not in (row.parent, row.variant)]
    conditions = found.conditions if found else {}
    condition = next((name for name in conditions if any(row.condition == name for row in pooled)), None)
    panels, clearest = [], None
    for column, shown in found.columns.items() if condition else ():
        rows = [row for row in pooled if (row.condition, row.column) == (condition, column)]
        if not rows:
            continue
        sides = []
        for side in ("parent", "variant"):
            # the rates are counts over the pairs, so their intervals can be drawn
            counts = [(round(getattr(row, side) * row.n_pairs), row.n_pairs) for row in rows]
            sides.append(tuple(_percent(k / n, wilson(k, n)) for k, n in counts))
        notes = tuple("" if row.delta is None else f"{row.delta * 100:+.1f} points" for row in rows)
        labels = tuple(REWRITE_NAMES.get(row.rewrite, row.code) for row in rows)
        series = (Series("as written", sides[0]), Series("rewritten", sides[1], notes))
        panels.append(Panel(_capital(shown), HARMFUL_UNIT, labels, series))
        for row, label in zip(rows, labels, strict=True):
            # a rise counts once its interval leaves zero
            if row.ci and row.ci[0] > 0 and (clearest is None or row.delta > clearest[0].delta):
                clearest = (row, label, shown)
    if not panels:
        return Chart("Harmful answers to rewritten prompts", ())
    where = found.conditions[condition]
    if clearest is None:
        return Chart(f"Rewriting the prompts does not clearly raise the harmful-response rate ({where})", tuple(panels))
    row, label, shown = clearest
    rates = f"{row.variant * 100:.1f}% against {row.parent * 100:.1f}% as written"
    title = f"Rewritten prompts ({label}) draw more harmful answers at {shown}: {rates} ({where})"
    return Chart(title, tuple(panels))


def safety_patch(
    breakdown: Breakdown,
    tables: Mapping[str, Tables],
    settings: Settings,
    *,
    plain: str = "keep",
    patched: str = "patch",
) -> Chart:
    """Harmful-response rate, and over-refusal where tabled, of each column with and without the safety patch.

    A group per column and a series per condition; no panel unless the breakdown holds both conditions.
    """
    if not {plain, patched} <= breakdown.conditions.keys():
        return Chart("Harmful answers with and without the safety patch", ())
    names = {condition: breakdown.conditions[condition] for condition in (plain, patched)}
    labels = tuple(_capital(shown) for shown in breakdown.columns.values())
    harmful = []
    for condition, shown in names.items():
        found = [_rates(breakdown, condition, "all", column).get((None, None)) for column in breakdown.columns]
        harmful.append(Series(shown, _estimates(found)))
    panels = [Panel(METRICS[0][2], METRICS[0][3], labels, tuple(harmful))]
    if {plain, patched} <= tables.keys():
        refused = []
        for condition, shown in names.items():
            found = [_row(tables[condition].over_refusal, "en", column, settings) for column in breakdown.columns]
            refused.append(Series(shown, _estimates(found)))
        panels.append(Panel(METRICS[1][2], METRICS[1][3], labels, tuple(refused)))
    moved = [
        f"from {first.value:.1f}% to {second.value:.1f}% for the {label.lower()} model"
        for label, first, second in zip(labels, harmful[0].points, harmful[1].points, strict=True)
        if None not in (first.value, second.value)
    ]
    if not moved:
        return Chart("Harmful answers with and without the safety patch", tuple(panels))
    return Chart(f"With the {names[patched]}, harmful answers go {' and '.join(moved)}", tuple(panels))
