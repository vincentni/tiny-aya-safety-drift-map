"""The figures as PNG and SVG; the same inputs give the same files."""

import math
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path

from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ..core.records import Tables
from ..core.settings import Settings
from ..table.breakdown import Breakdown
from . import data
from .data import Chart, Flows, Heat, Pairing

DPI = 160
# inches: 1440 px wide at DPI, read at about 900 px in a README
WIDTH = 9.0
# Okabe-Ito colours, told apart under the common colour-vision deficiencies
BLUE, ORANGE, SKY, VERMILLION, GREEN, PURPLE = "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"
SERIES_COLOURS = (BLUE, ORANGE, GREEN, PURPLE)
STATE_COLOURS = {"refused": BLUE, "complied-harmless": SKY, "complied-harmful": VERMILLION, "unscored": "#BBBBBB"}
HEAT_COLOURS = LinearSegmentedColormap.from_list("harmful", ["#FFF7EC", "#FDBB84", VERMILLION, "#7F2704"])
WITHHELD = "#ECECEC"
INK, FAINT, RULE = "#222222", "#6B6B6B", "#DDDDDD"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#888888",
    "axes.titlesize": 10,
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


def _figure(title: str, height: float) -> Figure:
    figure = Figure(figsize=(WIDTH, height), layout="constrained")
    wrapped = textwrap.fill(title, 88, break_on_hyphens=False)
    figure.suptitle(wrapped, x=0.012, ha="left", fontsize=11, fontweight="bold")
    return figure


def _offsets(count: int, gap: float) -> list[float]:
    """Positions of ``count`` series around a tick, ``gap`` apart."""
    return [(index - (count - 1) / 2) * gap for index in range(count)]


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
    """A point with its interval per label and series, a panel per condition on one value axis."""
    figure = _figure(chart.title, 3.9)
    axes = figure.subplots(1, len(chart.panels), sharey=True, squeeze=False)[0]
    handles = {}
    for axis, panel in zip(axes, chart.panels, strict=True):
        ticks = list(range(len(panel.labels)))
        for series, offset, colour in zip(
            panel.series, _offsets(len(panel.series), 0.26), SERIES_COLOURS, strict=False
        ):
            xs, ys, arms = _spans(series.points, [tick + offset for tick in ticks])
            drawn = axis.errorbar(xs, ys, yerr=arms, fmt="o", color=colour, markersize=5, capsize=2.5, elinewidth=1.2)
            handles.setdefault(series.name, drawn)
        # the pooled rate stands apart from the languages
        axis.axvline(len(ticks) - 1.5, color=RULE, linewidth=0.8)
        axis.set_xticks(ticks, panel.labels)
        axis.set_title(panel.title)
        axis.yaxis.grid(True, color=RULE, linewidth=0.6)
        axis.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, _ceiling(chart))
    axes[0].set_ylabel(chart.panels[0].unit)
    _legend(figure, handles)
    return figure


def bars(chart: Chart, *, shared: bool = False) -> Figure:
    """A bar per group and series with its value, and its note, written above it; a panel per condition or metric.

    ``shared`` puts the panels on one value axis, for panels of one unit.
    """
    noted = any(series.notes for panel in chart.panels for series in panel.series)
    figure = _figure(chart.title, 3.9 if noted else 3.6)
    axes = figure.subplots(1, len(chart.panels), sharey=shared, squeeze=False)[0]
    # a note needs room beside its neighbour's
    gap, width = (0.46, 0.42) if noted else (0.4, 0.36)
    handles = {}
    for axis, panel in zip(axes, chart.panels, strict=True):
        ticks = list(range(len(panel.labels)))
        suffix = "%" if "%" in panel.unit else ""
        top = 0.0
        for series, offset, colour in zip(panel.series, _offsets(len(panel.series), gap), SERIES_COLOURS, strict=False):
            xs, ys, arms = _spans(series.points, [tick + offset for tick in ticks])
            whiskers = {"elinewidth": 1.0, "capsize": 2.5, "ecolor": INK}
            drawn = axis.bar(xs, ys, width=width, color=colour, yerr=arms, error_kw=whiskers)
            handles.setdefault(series.name, drawn)
            notes = [note for note, point in zip(series.notes, series.points, strict=False) if point.value is not None]
            for at, (x, y) in enumerate(zip(xs, ys, strict=True)):
                peak = y + (arms[1][at] if arms else 0.0)
                note = notes[at] if at < len(notes) else ""
                above = {"textcoords": "offset points", "ha": "center"}
                if note:
                    axis.annotate(note, (x, peak), xytext=(0, 3), fontsize=7, color=FAINT, **above)
                axis.annotate(f"{y:.1f}{suffix}", (x, peak), xytext=(0, 12 if note else 3), fontsize=8, **above)
                top = max(top, peak)
        wrap = 18 if max(map(len, panel.labels)) > 24 else 12
        axis.set_xticks(ticks, [textwrap.fill(label, wrap) for label in panel.labels])
        axis.set_title(panel.title)
        axis.set_ylabel(panel.unit) if not shared or axis is axes[0] else None
        axis.set_ylim(0, (_ceiling(chart) if shared else top) * (1.25 if noted else 1.18) or 1)
        axis.tick_params(axis="x", length=0)
        axis.yaxis.grid(True, color=RULE, linewidth=0.6)
    _legend(figure, handles)
    return figure


def rows(chart: Chart) -> Figure:
    """A point with its interval per label and series on horizontal rows, a panel per condition on one label axis."""
    count = len(chart.panels[0].labels)
    figure = _figure(chart.title, 1.5 + 0.4 * count)
    axes = figure.subplots(1, len(chart.panels), sharey=True, sharex=True, squeeze=False)[0]
    handles = {}
    for axis, panel in zip(axes, chart.panels, strict=True):
        ticks = list(range(count))
        for series, offset, colour in zip(panel.series, _offsets(len(panel.series), 0.3), SERIES_COLOURS, strict=False):
            ys, xs, arms = _spans(series.points, [tick + offset for tick in ticks])
            drawn = axis.errorbar(xs, ys, xerr=arms, fmt="o", color=colour, markersize=5, capsize=2.5, elinewidth=1.2)
            handles.setdefault(series.name, drawn)
        axis.set_yticks(ticks, panel.labels)
        axis.set_title(panel.title)
        axis.set_xlabel(panel.unit)
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
                axis.text(to, at, f"n = {cell.n}", ha="center", va="center", fontsize=8, color=FAINT)
                continue
            colour = "white" if cell.rate > 0.55 * ceiling else INK
            said = f"{cell.rate:.0f}%  ({cell.k}/{cell.n})"
            axis.text(to, at, said, ha="center", va="center", fontsize=8.5, color=colour)
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
    axis.set_xlabel(note, color=FAINT, fontsize=8.5, loc="left")
    bar = figure.colorbar(image, ax=axis, shrink=0.85, pad=0.02)
    bar.set_label(shown.unit)
    bar.outline.set_visible(False)
    return figure


def flows(shown: Flows) -> Figure:
    """Two stacked bars per condition, the outcomes before and after, joined by a band per pair of outcomes."""
    figure = _figure(shown.title, 4.6)
    axes = figure.subplots(1, len(shown.panels), sharey=True, squeeze=False)[0]
    half = 0.16
    for axis, panel in zip(axes, shown.panels, strict=True):
        for x, counts in enumerate((panel.before, panel.after)):
            bottom = 0.0
            for state in shown.states:
                share = counts[state] * 100 / panel.n
                axis.bar(x, share, bottom=bottom, width=2 * half, color=STATE_COLOURS[state], linewidth=0)
                if share >= 3.5:
                    inside = {"ha": "center", "va": "center", "color": "white", "fontsize": 8.5, "fontweight": "bold"}
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
        axis.set_xlabel(panel.note.replace(". ", ".\n"), fontsize=8.5, color=FAINT)
        axis.tick_params(axis="x", length=0)
    axes[0].set_ylabel(f"Share of harmful prompts (%, n = {shown.panels[0].n})")
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
    variants: Mapping[str, Iterable[Pairing]] | None = None,
) -> list[Path]:
    """Draw the figures into ``out`` as PNG and SVG and return the files written.

    ``tables`` maps a prompt condition to its result tables; the first condition leads the titles. A figure whose
    data is not there is left out: the outcomes need two columns, the answer lengths the stored answers, the script
    variants ``variants`` (a column to its pairings) and the safety patch its condition in the breakdown.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    moved = data.outcome_flows(breakdown)
    charts = {
        "length": data.truncation(breakdown),
        "variants": data.script_variants(variants or {}),
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
