"""What each figure shows, as plain numbers, labels and a title that states the finding."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.records import ChrfRow, RateRow, Tables
from ..core.settings import Settings
from ..table.breakdown import CELL, COLUMN_NAMES, CONDITION_NAMES, STATES, Breakdown, BreakdownRate
from ..table.build import COLUMNS, column_of

LANGUAGE_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "ru": "Russian", "hi": "Hindi", "ar": "Arabic"}
# names too long for an axis
CATEGORY_NAMES = {"Harms of Representation Allocation and Quality of Service": "Representation & allocation harms"}
HARMFUL = "complied-harmful"
# metric, its row in the tables, panel title, value axis, and what to call a fall, no clear change and a rise
METRICS = (
    (
        "safe",
        "all",
        "Harmful answers",
        "% of scored responses to harmful prompts",
        ("more harmful answers", "no clear change in harmful answers", "fewer harmful answers"),
    ),
    (
        "over_refusal",
        "en",
        "Over-refusal (XSTest)",
        "% of safe prompts refused",
        ("less over-refusal", "no clear change in over-refusal", "more over-refusal"),
    ),
    (
        "chrf",
        "all",
        "Task score (chrF)",
        "chrF on held-out benign prompts, 0 to 100",
        ("a lower task score", "no clear change in task score", "a higher task score"),
    ),
)


@dataclass(frozen=True)
class Estimate:
    """A value in the unit of its panel, with the bounds of its 95% interval when it has one."""

    value: float | None
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Series:
    name: str
    points: tuple[Estimate, ...]


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


def _tabled(tables: Tables) -> list[str]:
    return [column for column in COLUMNS if column in tables.columns and tables.columns[column].state == "tabled"]


def _row(rows: list, name: str, column: str, settings: Settings) -> RateRow | ChrfRow | None:
    return next((row for row in rows if row.row == name and column_of(row.checkpoint, settings) == column), None)


def _rows(tables: Tables, metric: str) -> list:
    if metric == "chrf":
        return tables.chrf
    return tables.over_refusal if metric == "over_refusal" else [row for row in tables.safety if row.metric == "safe"]


def safe_rate_by_language(tables: Mapping[str, Tables], settings: Settings) -> Chart:
    """Safe rate on harmful prompts per language and pooled: a panel per condition, a series per tabled column."""
    names = [*settings.languages, "all"]
    labels = tuple(LANGUAGE_NAMES.get(name, _capital(name)) for name in names)
    panels = []
    for condition, table in tables.items():
        series = []
        for column in _tabled(table):
            found = [_row(_rows(table, "safe"), name, column, settings) for name in names]
            points = tuple(_percent(row.rate, row.ci) if row else Estimate(None) for row in found)
            series.append(Series(COLUMN_NAMES.get(column, column), points))
        panels.append(Panel(_condition(condition), "Safe rate on harmful prompts (%)", labels, tuple(series)))
    pooled = [series.points[-1].value for series in panels[0].series]
    title = "Safe rate on harmful prompts by language"
    if len(pooled) > 1 and None not in pooled:
        first, last = f"{pooled[0]:.1f}%", f"{pooled[-1]:.1f}%"
        moved = "raised" if pooled[-1] > pooled[0] else "lowered"
        change = f"{moved} the safe rate on harmful prompts from {first} to {last}"
        change = change if first != last else f"left the safe rate on harmful prompts at {first}"
        title = f"{panels[0].title}: fine-tuning on benign data {change}"
    return Chart(title, tuple(panels))


def before_after(tables: Mapping[str, Tables], settings: Settings) -> Chart:
    """Harmful-response rate, over-refusal and chrF of each tabled column: a panel per metric, a group per condition."""
    columns = list(dict.fromkeys(column for table in tables.values() for column in _tabled(table)))
    labels = tuple(_condition(condition) for condition in tables)
    panels, findings = [], []
    for metric, name, title, unit, words in METRICS:
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
            series.append(Series(COLUMN_NAMES.get(column, column), tuple(points)))
        panels.append(Panel(title, unit, labels, tuple(series)))
        signs = set()
        for table in tables.values():
            between = _tabled(table)[:2]
            deltas = [
                row for row in table.deltas if [row.a, row.b] == between and (row.metric, row.row) == (metric, name)
            ]
            # a change counts once its interval leaves zero
            apart = bool(deltas) and deltas[0].ci is not None and (deltas[0].ci[0] > 0 or deltas[0].ci[1] < 0)
            signs.add((deltas[0].delta > 0) - (deltas[0].delta < 0) if apart else 0)
        # one phrase when the conditions agree
        findings.append(
            words[signs.pop() + 1] if len(signs) == 1 else f"mixed results on {title[:1].lower()}{title[1:]}"
        )
    scope = {1: "", 2: " under both prompt conditions"}.get(len(labels), " under every prompt condition")
    if len(columns) < 2:
        return Chart("Harmful answers, over-refusal and task score", tuple(panels))
    return Chart(f"After fine-tuning: {', '.join(findings)}{scope}", tuple(panels))


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
        panels.append(Panel(_condition(condition), "Harmful answers (% of scored responses)", labels, tuple(series)))
    first, *later = [[point.value for point in series.points] for series in panels[0].series]
    if not categories or first[0] is None:
        return Chart("Harmful answers by harm category", tuple(panels))
    lead = panels[0]
    worst = f"{labels[0]} ({first[0]:.1f}%, {lead.title.lower()})"
    title = f"Most harmful answers at {lead.series[0].name}: {worst}"
    if later:
        pairs = [pair for pair in zip(first, later[-1], strict=True) if None not in pair]
        lower = sum(after < before for before, after in pairs)
        title += f"; fine-tuning lowers the rate in {lower} of {len(pairs)} categories"
    return Chart(title, tuple(panels))


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
    unit = f"Harmful answers (% of scored responses), {breakdown.conditions[condition]}"
    return Heat(title, unit, rows, columns, tuple(grid), breakdown.min_cell_n)


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
    name = breakdown.rates[0].set
    panels, moved = [], []
    for condition in breakdown.conditions:
        pooled = [
            row for row in breakdown.transitions if (row.set, row.condition, row.language) == (name, condition, "all")
        ]
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
