"""Markdown for the tables; every number cell names its n, interval, judge and preamble."""

from ..core.records import ChrfRow, ColumnState, DeltaRow, RateRow, Tables
from ..core.settings import Settings
from .build import COLUMNS, FLAG_COLUMNS, METRICS, column_of

HATCH = "▨"
# direction words for a positive and a negative change
WORDS = {
    "safe": ("safer", "less safe"),
    "over_refusal": ("more over-refusal", "less over-refusal"),
    "chrf": ("higher chrF", "lower chrF"),
}
ABSENT = {"failed": "failed", "not_applicable": "n/a"}


def _judge(tables: Tables) -> str:
    return tables.judge.repo.rpartition("/")[2]


def rate_cell(row: RateRow) -> str:
    """A rate with its n, Wilson interval, judge and preamble; ``n = 0`` for an empty cell."""
    if row.n == 0:
        return "n = 0"
    low, high = row.ci
    judge = row.judge.repo.rpartition("/")[2]
    return f"{row.rate:.1%} (n={row.n}; 95% CI {low:.1%} to {high:.1%}; judge={judge}; preamble={row.preamble})"


def chrf_cell(row: ChrfRow, preamble: str) -> str:
    return "n = 0" if row.n == 0 else f"{row.chrf * 100:.1f} (n={row.n}; chrF1; preamble={preamble})"


def delta_cell(delta: DeltaRow | None, n: str, source: str, preamble: str, words: tuple[str, str]) -> str:
    """A change in points with its direction word, interval, n and exact p; ``n/a`` without an estimate."""
    if delta is None or delta.delta is None:
        return "n/a"
    # adding 0.0 turns a rounded -0.0 into 0.0
    points, low, high = (round(value * 100, 1) + 0.0 for value in (delta.delta, *delta.ci))
    word = "no change" if points == 0 else words[points < 0]
    p = "" if delta.p_value is None else "p < 0.001; " if delta.p_value < 0.001 else f"p = {delta.p_value:.3f}; "
    return f"{points:+.1f} points, {word} (95% CI {low:+.1f} to {high:+.1f}; n={n}; {p}{source}; preamble={preamble})"


def _absent(state: ColumnState) -> str | None:
    if state.state in ("tabled", "not_comparable"):
        return None
    return f"{ABSENT.get(state.state, 'not run')}: {state.reason}"


def _table(head: list[str], lines: list[list[str]]) -> list[str]:
    return [f"| {' | '.join(row)} |" for row in (head, ["---"] * len(head), *lines)] + [""]


def render(tables: Tables, settings: Settings) -> str:
    """Markdown tables; an absent column gives its reason, a not-comparable one keeps its numbers behind a hatch."""
    preamble, judge = settings.preamble, f"judge={_judge(tables)}"

    def cells(rows: list, name: str, show) -> list[str]:
        line = []
        for column in COLUMNS:
            state = tables.columns[column]
            found = [row for row in rows if row.row == name and column_of(row.checkpoint, settings) == column]
            hatch = f"{HATCH} " if state.state == "not_comparable" else ""
            line.append(_absent(state) or (hatch + show(found[0]) if found else "n/a"))
        return line

    def section(title: str, rows: list, show, extra: list[list[str]] = ()) -> list[str]:
        names = list(dict.fromkeys(row.row for row in rows))
        return [
            f"## {title}",
            "",
            *_table(["language", *COLUMNS], [[name, *cells(rows, name, show)] for name in names] + list(extra)),
        ]

    def by_metric(metric: str) -> list[RateRow]:
        return [row for row in tables.safety if row.metric == metric]

    def group_n(column: str, name: str) -> str:
        sizes = [
            row.n for row in by_metric("safe") if row.row == name and column_of(row.checkpoint, settings) == column
        ]
        return str(sizes[0]) if sizes else "0"

    gaps = {delta.a: delta for delta in tables.deltas if delta.a == delta.b}
    gap_line = []
    for delta in list(gaps.values())[:1]:
        low, _, high = delta.row.partition(" vs ")
        words = (f"{low} safer", f"{low} less safe")
        gap_line = [[f"{low} minus {high}"]]
        for column in COLUMNS:
            n = f"{group_n(column, low)} vs {group_n(column, high)}"
            hatch = f"{HATCH} " if tables.columns[column].state == "not_comparable" else ""
            gap_line[0].append(
                _absent(tables.columns[column]) or hatch + delta_cell(gaps.get(column), n, judge, preamble, words)
            )

    out = ["# Safety drift map", ""]
    out += section("Safe rate on harmful prompts", by_metric("safe"), rate_cell, gap_line)
    out += section("Refusal rate on harmful prompts", by_metric("refusal"), rate_cell)
    out += section("Over-refusal on XSTest (English)", tables.over_refusal, rate_cell)
    out += section("chrF on held-out benign prompts", tables.chrf, lambda row: chrf_cell(row, preamble))

    skipping = ((delta.a, delta.b) for delta in tables.deltas if delta.a != delta.b)
    pairs = list(dict.fromkeys([*zip(COLUMNS, COLUMNS[1:], strict=False), *skipping]))
    changes = []
    for metric, _, per_language in METRICS:
        rows = (
            by_metric("safe") if metric == "safe" else tables.over_refusal if metric == "over_refusal" else tables.chrf
        )
        names = [name for name in dict.fromkeys(row.row for row in rows) if "+" not in name]
        for name in names if per_language else names[:1]:
            line = [name, metric]
            for a, b in pairs:
                found = [d for d in tables.deltas if (d.row, d.metric, d.a, d.b) == (name, metric, a, b)]
                source = "chrF1" if metric == "chrf" else judge
                n = f"{found[0].n_pairs} pairs" if found else ""
                line.append(delta_cell(found[0] if found else None, n, source, preamble, WORDS[metric]))
            changes.append(line)
    heads = ["row", "metric", *(f"{a} to {b}" for a, b in pairs)]
    out += ["## Change between adjacent columns (later minus earlier)", "", *_table(heads, changes)]

    out += ["## Columns", ""]
    for column, state in tables.columns.items():
        mark = f" {HATCH}" if state.state == "not_comparable" else ""
        reason = f": {state.reason}" if state.reason else ""
        out.append(f"- {column}{mark} ({state.kind}): {state.state.replace('_', ' ')}{reason}")
    out += ["", "## Flags", ""]
    for name, needs in FLAG_COLUMNS.items():
        raised = tables.flags.get(name)
        missing = next((column for column in needs if tables.columns[column].state != "tabled"), None)
        why = f"{missing} {tables.columns[missing].state.replace('_', ' ')}" if missing else "no scored rows"
        out.append(f"- {name}: {f'not evaluated ({why})' if raised is None else 'yes' if raised else 'no'}")
    return "\n".join(out) + "\n"
