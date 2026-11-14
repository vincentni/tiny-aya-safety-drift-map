"""The figures as PNG and SVG; the same inputs give the same files."""

import math
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator

from ..core.records import Tables
from ..core.settings import Settings
from ..table.breakdown import Breakdown
from . import data
from .data import Chart, Flows, Heat, Variants

DPI = 160
# inches: 1440 px wide at DPI, read at about 800 px beside the text
WIDTH = 9.0
# points: at that width no text is under 11 px
TEXT, SMALL, PANEL, HEAD = 10, 9, 11, 12
# inches a value axis takes with its ticks and label, and the space between two panels
AXIS, BETWEEN = 0.8, 0.2
# inches of a figure's height that its titles, tick labels and legend take, at the most
FRAME = 1.8
# points kept free between two tick labels
GAP = 8
# tick labels too wide for their ticks run down to the tick at this angle
TILT = 40
# Okabe-Ito colours, told apart under the common colour-vision deficiencies
BLUE, ORANGE, SKY, VERMILLION, GREEN, PURPLE = "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"
SERIES_COLOURS = (BLUE, ORANGE, GREEN, PURPLE)
STATE_COLOURS = {"refused": BLUE, "complied-harmless": SKY, "complied-harmful": VERMILLION, "unscored": "#BBBBBB"}
HEAT_COLOURS = LinearSegmentedColormap.from_list("harmful", ["#FFF7EC", "#FDBB84", VERMILLION, "#7F2704"])
WITHHELD = "#ECECEC"
INK, FAINT, RULE = "#222222", "#6B6B6B", "#DDDDDD"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": TEXT,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#888888",
    "axes.titlesize": PANEL,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    # fixed ids and outlined glyphs: the SVG does not change between runs or with the fonts installed
    "svg.hashsalt": "driftmap",
    "svg.fonttype": "path",
}
# file stems, shared with the write-up
STEMS = {
    "language": "f1_harmful_by_language",
    "cells": "f2_category_by_language",
    "axes": "f3_three_axes",
    "categories": "f4_harm_categories",
    "outcomes": "f5_transitions",
    "length": "f6_truncation",
    "variants": "f7_script_variants",
    "patch": "f8_safety_patch",
}


# what a figure that is not always drawn needs
NEEDS = {
    "outcomes": "two tabled columns in the breakdown",
    "length": "a breakdown built with the stored responses",
    "variants": "variants.json beside the tables, holding the pooled harmful rows",
    "patch": "the safety-patch condition in the breakdown",
}


@cache
def _width(text: str, size: float, weight: str = "normal") -> float:
    """Points one line of text takes at ``size``."""
    if not text.strip():
        return 0.0
    font = FontProperties(family=STYLE["font.family"], weight=weight)
    return TextPath((0, 0), text, size=size, prop=font).get_extents().width


def _fit(text: str, room: float, size: float, weight: str = "normal") -> str:
    """``text`` filled into lines of at most ``room`` points at ``size``; a word wider than that keeps a line."""
    lines: list[str] = []
    for word in text.split():
        longer = f"{lines[-1]} {word}" if lines else word
        if lines and _width(longer, size, weight) <= room:
            lines[-1] = longer
        else:
            lines.append(word)
    return "\n".join(lines)


def _slot(panels: int, ticks: int, shared: bool) -> float:
    """Points across that one tick has, at the least; ``shared`` when the panels hold one value axis between them."""
    return (WIDTH - AXIS * (1 if shared else panels) - BETWEEN * panels) * 72 / panels / ticks


def _upright(figure: Figure, text: str, frame: float = FRAME, rows: int = 1) -> str:
    """The label of a value axis, filled into the height the figure leaves each of its ``rows`` of axes.

    The figure keeps ``frame`` inches for what is not axes.
    """
    return _fit(text, (figure.get_figheight() - frame) / rows * 72, TEXT)


def _figure(title: str, height: float) -> Figure:
    figure = Figure(figsize=(WIDTH, height), layout="constrained")
    wrapped = _fit(title, WIDTH * 72 * 0.96, HEAD, "bold")
    figure.suptitle(wrapped, x=0.012, ha="left", fontsize=HEAD, fontweight="bold")
    return figure


def _offsets(count: int, gap: float) -> list[float]:
    """Positions of ``count`` series around a tick, ``gap`` apart."""
    return [(index - (count - 1) / 2) * gap for index in range(count)]


def _beside(series, gap: float) -> list[list[float]]:
    """A position per series and tick: the series that have a value at a tick stand ``gap`` apart around it."""
    at = [[float(tick) for tick in range(len(line.points))] for line in series]
    for tick in range(len(series[0].points)) if series else ():
        held = [index for index, line in enumerate(series) if line.points[tick].value is not None]
        for index, offset in zip(held, _offsets(len(held), gap), strict=True):
            at[index][tick] += offset
    return at


def _spans(points, at: list[float]) -> tuple[list, list, list | None]:
    """Positions, values and interval arms of the points that have a value; no arms when none has an interval."""
    kept = [(where, point) for where, point in zip(at, points, strict=True) if point.value is not None]
    arms = [
        [0.0 if point.low is None else point.value - point.low for _, point in kept],
        [0.0 if point.high is None else point.high - point.value for _, point in kept],
    ]
    bounded = any(point.low is not None for _, point in kept)
    return [where for where, _ in kept], [point.value for _, point in kept], arms if bounded else None


def _ceiling(chart: Chart) -> float:
    """The next multiple of five above every value and interval of a chart."""
    points = [point for panel in chart.panels for series in panel.series for point in series.points]
    highs = [point.value if point.high is None else point.high for point in points if point.value is not None]
    return 5 * math.ceil(max(highs, default=5) / 5) or 5


def _legend(figure: Figure, handles: dict[str, object]) -> None:
    """One entry per series name, whichever panel drew it first."""
    spacing = {"handletextpad": 0.5, "columnspacing": 2.0}
    figure.legend(handles.values(), handles.keys(), loc="outside lower center", ncols=len(handles), **spacing)


def dots(chart: Chart) -> Figure:
    """A point with its interval per label and series, a panel per condition on one value axis.

    Tick labels that would touch their neighbours are tilted.
    """
    widest = max(_width(label, TEXT) for panel in chart.panels for label in panel.labels)
    room = min(_slot(len(chart.panels), len(panel.labels), True) for panel in chart.panels)
    tilted = {"rotation": TILT, "ha": "right", "rotation_mode": "anchor"} if widest + GAP > room else {}
    figure = _figure(chart.title, 4.3 if tilted else 3.9)
    axes = figure.subplots(1, len(chart.panels), sharey=True, squeeze=False)[0]
    handles = {}
    for axis, panel in zip(axes, chart.panels, strict=True):
        ticks = list(range(len(panel.labels)))
        for series, at, colour in zip(panel.series, _beside(panel.series, 0.26), SERIES_COLOURS, strict=False):
            xs, ys, arms = _spans(series.points, at)
            drawn = axis.errorbar(xs, ys, yerr=arms, fmt="o", color=colour, markersize=5, capsize=2.5, elinewidth=1.2)
            handles.setdefault(series.name, drawn)
        # the pooled rate stands apart from the languages
        axis.axvline(len(ticks) - 1.5, color=RULE, linewidth=0.8)
        axis.set_xticks(ticks, panel.labels, **tilted)
        axis.set_title(panel.title)
        axis.yaxis.grid(True, color=RULE, linewidth=0.6)
        axis.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, _ceiling(chart))
    axes[0].set_ylabel(_upright(figure, chart.panels[0].unit))
    _legend(figure, handles)
    return figure


def _clear(figure: Figure, stacks: list[list[Annotation]], home: list[list[tuple[float, float]]]) -> dict:
    """Put the texts above each bar at ``home`` and raise those that would run into an earlier bar's.

    The layout decides, so the figure is laid out here. Returns, per axes, the factor by which its value axis has
    to grow to hold its highest text; 1 when it holds them all.
    """
    for stack, places in zip(stacks, home, strict=True):
        for text, place in zip(stack, places, strict=True):
            text.xyann = place
    figure.draw_without_rendering()
    points = 72 / figure.dpi
    placed: list[tuple[float, float, float, float]] = []
    grow = {}
    for stack in stacks:
        boxes = [text.get_window_extent() for text in stack]
        left, right = min(box.x0 for box in boxes), max(box.x1 for box in boxes)
        low, high = min(box.y0 for box in boxes), max(box.y1 for box in boxes)
        rise = 0.0
        # each pass clears one earlier stack; the next looks again from the raised place
        for _ in placed:
            sides = [(bottom, top) for x0, x1, bottom, top in placed if x0 < right and left < x1]
            over = [top for bottom, top in sides if bottom < high + rise and low + rise < top]
            if not over:
                break
            rise = max(over) - low + 2
        for text in stack:
            text.xyann = (text.xyann[0], text.xyann[1] + rise * points)
        placed.append((left, right, low + rise, high + rise))
        axes = stack[0].axes
        foot = axes.transData.transform(stack[0].xy)[1]
        # a longer value axis lowers the bar and leaves the texts as high above it
        spare = axes.bbox.y1 - (high + rise - foot) - 4 - axes.bbox.y0
        if spare > 0:
            grow[axes] = max(grow.get(axes, 1.0), (foot - axes.bbox.y0) / spare)
    return grow


def bars(chart: Chart, *, shared: bool = False) -> Figure:
    """A bar per group and series with its value, and its note, written above it; a panel per condition or metric.

    ``shared`` puts the panels on one value axis, for panels of one unit. The panels fill rows of ``chart.across``.
    """
    noted = any(series.notes for panel in chart.panels for series in panel.series)
    across = chart.across or len(chart.panels)
    down = math.ceil(len(chart.panels) / across)
    tall = 4.5 if noted else 3.9
    # a further row repeats the panels, not the title and the legend
    figure = _figure(chart.title, tall + (down - 1) * (tall - 0.9))
    places = list(figure.subplots(down, across, sharey=shared, squeeze=False).flat)
    axes = places[: len(chart.panels)]
    for unused in places[len(axes) :]:
        unused.set_visible(False)
    # the bars of a group keep to their tick; a note needs room beside its neighbour's
    gap = min(0.46 if noted else 0.4, 0.92 / max(len(panel.series) for panel in chart.panels))
    width = 0.9 * gap
    handles = {}
    stacks = []
    for axis, panel in zip(axes, chart.panels, strict=True):
        ticks = list(range(len(panel.labels)))
        room = _slot(across, len(ticks), shared)
        suffix = "%" if "%" in panel.unit else ""
        top = 0.0
        above = {}
        for series, at, colour in zip(panel.series, _beside(panel.series, gap), SERIES_COLOURS, strict=False):
            xs, ys, arms = _spans(series.points, at)
            whiskers = {"elinewidth": 1.0, "capsize": 2.5, "ecolor": INK}
            drawn = axis.bar(xs, ys, width=width, color=colour, yerr=arms, error_kw=whiskers)
            handles.setdefault(series.name, drawn)
            notes = [note for note, point in zip(series.notes, series.points, strict=False) if point.value is not None]
            for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
                peak = y + (arms[1][index] if arms else 0.0)
                note = _fit(notes[index], gap * room - 2, SMALL) if index < len(notes) else ""
                over = {"textcoords": "offset points", "ha": "center", "fontsize": SMALL}
                texts = [axis.annotate(note, (x, peak), xytext=(0, 3), color=FAINT, **over)] if note else []
                rise = 3 + len(note.splitlines()) * SMALL * 1.3
                above[x] = [*texts, axis.annotate(f"{y:.1f}{suffix}", (x, peak), xytext=(0, rise), **over)]
                top = max(top, peak)
        stacks += [above[x] for x in sorted(above)]
        axis.set_xticks(ticks, [_fit(label, room - GAP, TEXT) for label in panel.labels])
        axis.set_title(_fit(panel.title, room * len(ticks), PANEL, "bold"))
        if not shared or axes.index(axis) % across == 0:
            axis.set_ylabel(_upright(figure, panel.unit, rows=down))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        axis.set_ylim(0, (_ceiling(chart) if shared else top) * 1.1 or 1)
        axis.tick_params(axis="x", length=0)
        axis.yaxis.grid(True, color=RULE, linewidth=0.6)
    _legend(figure, handles)
    home = [[text.xyann for text in stack] for stack in stacks]
    # a raised text needs more room and a longer axis brings bars closer in height: a few rounds settle it
    for _ in range(4):
        grow = _clear(figure, stacks, home)
        most = max(grow.values(), default=1.0)
        if most <= 1:
            break
        for axis in axes[:1] if shared else axes:
            axis.set_ylim(0, axis.get_ylim()[1] * (most if shared else grow.get(axis, 1.0)))
    else:
        _clear(figure, stacks, home)
    return figure


def rows(chart: Chart) -> Figure:
    """A point with its interval per label and series on horizontal rows, a panel per condition on one label axis."""
    count = len(chart.panels[0].labels)
    figure = _figure(chart.title, 1.5 + 0.4 * count)
    axes = figure.subplots(1, len(chart.panels), sharey=True, sharex=True, squeeze=False)[0]
    # what the names of the rows leave, split between the panels
    names = max(_width(label, TEXT) for label in chart.panels[0].labels)
    room = (WIDTH * 72 - names - 12) / len(chart.panels) - GAP
    handles = {}
    for axis, panel in zip(axes, chart.panels, strict=True):
        ticks = list(range(count))
        for series, at, colour in zip(panel.series, _beside(panel.series, 0.3), SERIES_COLOURS, strict=False):
            ys, xs, arms = _spans(series.points, at)
            drawn = axis.errorbar(xs, ys, xerr=arms, fmt="o", color=colour, markersize=5, capsize=2.5, elinewidth=1.2)
            handles.setdefault(series.name, drawn)
        axis.set_yticks(ticks, panel.labels)
        axis.set_title(panel.title)
        axis.set_xlabel(_fit(panel.unit, room, TEXT))
        axis.xaxis.grid(True, color=RULE, linewidth=0.6)
        axis.tick_params(axis="y", length=0)
    # the axes share one range, wide enough for every interval
    axes[0].set_xlim(0, _ceiling(chart))
    axes[0].invert_yaxis()
    _legend(figure, handles)
    return figure


def heat(shown: Heat) -> Figure:
    """A coloured cell per row and column with its rate and counts; a cell under the floor is grey with its count."""
    figure = _figure(shown.title, 1.7 + 0.42 * len(shown.rows))
    axis = figure.subplots()
    rates = [cell.rate for line in shown.cells for cell in line if cell and cell.rate is not None]
    ceiling = 5 * math.ceil(max(rates, default=5) / 5) or 5
    grid = [[math.nan if cell is None or cell.rate is None else cell.rate for cell in line] for line in shown.cells]
    axis.set_facecolor(WITHHELD)
    image = axis.imshow(grid, cmap=HEAT_COLOURS, vmin=0, vmax=ceiling, aspect="auto")
    for at, line in enumerate(shown.cells):
        for to, cell in enumerate(line):
            if cell is None:
                continue
            if cell.rate is None:
                axis.text(to, at, f"n = {cell.n}", ha="center", va="center", fontsize=SMALL, color=FAINT)
                continue
            colour = "white" if cell.rate > 0.55 * ceiling else INK
            said = f"{cell.rate:.0f}%  ({cell.k}/{cell.n})"
            axis.text(to, at, said, ha="center", va="center", fontsize=SMALL, color=colour)
    axis.set_xticks(range(len(shown.columns)), shown.columns)
    axis.set_yticks(range(len(shown.rows)), shown.rows)
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    axis.set_xticks([tick - 0.5 for tick in range(len(shown.columns) + 1)], minor=True)
    axis.set_yticks([tick - 0.5 for tick in range(len(shown.rows) + 1)], minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    note = f"Grey: fewer than {shown.floor} scored responses, no rate shown."
    axis.set_xlabel(note, color=FAINT, fontsize=SMALL, loc="left")
    bar = figure.colorbar(image, ax=axis, shrink=0.85, pad=0.02)
    # no legend and one line of column names: the cells take most of the height
    bar.set_label(_upright(figure, shown.unit, 1.0))
    bar.outline.set_visible(False)
    return figure


def flows(shown: Flows) -> Figure:
    """Two stacked bars per condition, the outcomes before and after, joined by a band per pair of outcomes."""
    figure = _figure(shown.title, 4.9)
    axes = figure.subplots(1, len(shown.panels), sharey=True, squeeze=False)[0]
    half = 0.16
    room = _slot(len(shown.panels), 1, True)
    for axis, panel in zip(axes, shown.panels, strict=True):
        for x, counts in enumerate((panel.before, panel.after)):
            bottom = 0.0
            for state in shown.states:
                share = counts[state] * 100 / panel.n
                axis.bar(x, share, bottom=bottom, width=2 * half, color=STATE_COLOURS[state], linewidth=0)
                if share >= 3.5:
                    inside = {"ha": "center", "va": "center", "color": "white", "fontsize": SMALL, "fontweight": "bold"}
                    axis.text(x, bottom + share / 2, str(counts[state]), **inside)
                bottom += share
        for band in panel.bands:
            edges = [half, 1 - half, 1 - half, half]
            spans = [band.left[0], band.right[0], band.right[1], band.left[1]]
            # prompts that kept their outcome stay in the background
            alpha = 0.22 if band.before == band.after else 0.6
            axis.fill(edges, spans, color=STATE_COLOURS[band.before], alpha=alpha, linewidth=0)
        axis.set_xticks([0, 1], panel.names)
        axis.set_xlim(-0.45, 1.45)
        axis.set_ylim(0, 100)
        axis.set_title(panel.title)
        sentences = panel.note.replace(". ", ".\n").split("\n")
        axis.set_xlabel("\n".join(_fit(sentence, room, SMALL) for sentence in sentences), fontsize=SMALL, color=FAINT)
        axis.tick_params(axis="x", length=0)
    axes[0].set_ylabel(_upright(figure, f"Share of harmful prompts (%, n = {shown.panels[0].n})"))
    _legend(figure, {state: Patch(color=STATE_COLOURS[state]) for state in shown.states})
    return figure


def _save(figure: Figure, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}.png", out / f"{stem}.svg"]
    # no library version or date in either file
    figure.savefig(paths[0], dpi=DPI, metadata={"Software": None})
    figure.savefig(paths[1], metadata={"Date": None, "Creator": None})
    return paths


def draw_all(
    tables: Mapping[str, Tables],
    breakdown: Breakdown,
    settings: Settings,
    out: str | Path,
    variants: Variants | None = None,
) -> list[Path]:
    """Draw the figures into ``out`` as PNG and SVG and return the files written.

    ``tables`` maps a prompt condition to its result tables; the first condition leads the titles. A figure whose
    data is not there is left out: the outcomes need two columns, the answer lengths the stored answers, the script
    variants ``variants`` (the stored comparisons) and the safety patch its condition in the breakdown.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    moved = data.outcome_flows(breakdown)
    charts = {
        "length": data.truncation(breakdown),
        "variants": data.script_variants(variants),
        "patch": data.safety_patch(breakdown, tables, settings),
    }
    with rc_context(STYLE):
        drawn = {
            "language": dots(data.harmful_by_language(tables, settings, breakdown.columns)),
            "cells": heat(data.category_by_language(breakdown)),
            "axes": bars(data.before_after(tables, settings, breakdown.columns)),
            "categories": rows(data.by_category(breakdown)),
            **({"outcomes": flows(moved)} if moved.panels else {}),
            **{name: bars(chart, shared=name != "patch") for name, chart in charts.items() if chart.panels},
        }
        return [path for name, figure in drawn.items() for path in _save(figure, out, STEMS[name])]
