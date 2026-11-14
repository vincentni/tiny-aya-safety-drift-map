import math
from dataclasses import replace
from itertools import pairwise

import pytest
from matplotlib import rc_context
from matplotlib.text import Text

from driftmap.figures import draw
from driftmap.figures.data import Cell, Chart, Estimate, Heat, Panel, Series, outcome_flows
from driftmap.table.breakdown import build as build_breakdown
from tests.fakes.breakdown import HATE, lay, prompts, stored_variants, two_conditions, variant_row
from tests.fakes.labels import make_settings
from tests.fakes.published import two_column_tables

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


STEMS = (
    "f1_harmful_by_language",
    "f2_category_by_language",
    "f3_three_axes",
    "f4_harm_categories",
    "f5_transitions",
    "f6_truncation",
    "f7_script_variants",
    "f8_safety_patch",
)


def test_the_figures_are_written_as_png_and_svg_and_are_the_same_on_every_run(tmp_path, settings):
    tables = {"keep": two_column_tables(tmp_path, settings)}
    breakdown = two_conditions(tmp_path, settings)

    written = draw.draw_all(tables, breakdown, settings, tmp_path / "first" / "figures")

    # the variants and the patch have no data here
    assert [path.name for path in written] == [f"{stem}.{kind}" for stem in STEMS[:6] for kind in ("png", "svg")]
    assert all(path.parent == tmp_path / "first" / "figures" for path in written)
    for path in written:
        content = path.read_bytes()
        assert len(content) > 2000, path.name
        if path.suffix == ".png":
            # 9 inches at 160 dpi
            assert content.startswith(PNG) and int.from_bytes(content[16:20]) == 1440, path.name
        else:
            assert content.lstrip().startswith(b"<?xml") and b"<dc:date>" not in content, path.name
    again = draw.draw_all(tables, breakdown, settings, tmp_path / "second")
    assert [path.read_bytes() for path in again] == [path.read_bytes() for path in written]


def test_a_figure_is_drawn_once_its_data_is_there(tmp_path, settings):
    tables = {"keep": two_column_tables(tmp_path, settings)}
    labels = {"keep": tmp_path / "keep", "patch": tmp_path / "patch"}
    plan = prompts("en", [HATE], 20, {"harmful": 5})
    alone = build_breakdown(labels, {"harmful": lay(labels["keep"], plan, columns=("baseline",))}, [], settings)

    # one column, no stored answers: neither outcomes nor lengths
    written = draw.draw_all(tables, alone, settings, tmp_path / "out")
    assert [path.stem for path in written[::2]] == list(STEMS[:4])

    lay(labels["patch"], prompts("en", [HATE], 20, {"harmful": 2}))
    patched = build_breakdown(labels, {"harmful": lay(labels["keep"], plan)}, [], settings)
    rows = [variant_row(column, "all-Latn", 0.10, 0.25, ci=(0.09, 0.21)) for column in ("baseline", "drifted")]
    written = draw.draw_all(tables, patched, settings, tmp_path / "more", draw.data.variants(stored_variants(*rows)))
    assert [path.stem for path in written[::2]] == [*STEMS[:5], *STEMS[6:]]
    assert all(path.stat().st_size > 2000 for path in written)


# pixels across at which a figure is read beside the text, and the smallest text that reads there
READ_AT, SMALLEST = 800, 11
LANGUAGES = ("Russian", "English", "Hindi", "Spanish", "Arabic", "French", "All")
CONDITIONS = ("Served template", "No safety preamble", "Safety patch")
CATEGORIES = (
    "Representation & allocation harms",
    "Self-Harm",
    "Hate Speech",
    "Discrimination & Injustice",
    "Violence, Threats & Incitement",
    "Graphic material",
    "Non-consensual sexual content",
    "Bullying & Harassment",
    "Profanity",
)
LENGTHS = ("Complete answers", "Answers cut off at the 256-token limit")
METRICS = ("harmful answers", "refusals", "harmless answers")
UNIT = "Harmful answers (% of scored responses)"
TITLE = (
    "Fewer harmful answers after fine-tuning, mostly because answers got shorter: 45% reached the 256-token limit "
    "before, 12% after (served template)"
)


@pytest.fixture
def styled():
    with rc_context(draw.STYLE):
        yield


def chart(titles, labels, values, notes=(), unit=UNIT) -> Chart:
    """A panel per title; ``values`` holds the points of each series by name, None where it has none.

    The last panel keeps the first series alone, as a condition measured on one column does.
    """

    def points(row):
        return tuple(Estimate(None) if value is None else Estimate(value, value * 0.6, value * 1.5) for value in row)

    series = tuple(Series(name, points(row), tuple(notes)) for name, row in values.items())
    panels = [Panel(title, unit, tuple(labels), series) for title in titles[:-1]]
    return Chart(TITLE, (*panels, Panel(titles[-1], unit, tuple(labels), series[:1])))


def rows_of_bars() -> Chart:
    """Six noted panels, three to a row, under titles that name a column and a metric."""
    names = [f"{column}: {metric}" for column in ("Base model", "Fine-tuned model, q4_k_m") for metric in METRICS]
    values = {"as written": [71.4, 72.0], "rewritten": [52.0, 70.0]}
    drawn = chart([*names, "last"], ("Latin script", "Mixed with English"), values, ("", "-19.4 points"))
    return replace(drawn, panels=drawn.panels[:6], across=3)


def shown(figure) -> list[Text]:
    """The texts a figure draws, once laid out: tick labels outside the view are left out."""
    figure.draw_without_rendering()
    unseen = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            low, high = sorted(axis.get_view_interval())
            ticks = [*axis.get_major_ticks(), *axis.get_minor_ticks()]
            unseen |= {
                label for tick in ticks if not low <= tick.get_loc() <= high for label in (tick.label1, tick.label2)
            }
    texts = [text for text in figure.findobj(Text) if text.get_visible() and text.get_text().strip()]
    return [text for text in texts if text not in unseen]


def clashes(figure) -> list[tuple[str, str]]:
    """Pairs of upright texts whose boxes share more than a pixel each way."""
    boxes = [(text.get_text(), text.get_window_extent()) for text in shown(figure) if text.get_rotation() % 90 == 0]
    return [
        (first, second)
        for at, (first, a) in enumerate(boxes)
        for second, b in boxes[at + 1 :]
        if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1
    ]


def figures(tmp_path, settings) -> dict:
    """One figure of each kind, as crowded as the write-up's: three conditions, six languages, nine categories."""
    by_language = {
        "base model": [16.1, 12.0, 10.0, 8.0, 7.0, 2.0, 9.2],
        "fine-tuned": [9.0, 10.0, 6.0, 7.0, 3.0, 4.0, 6.5],
    }
    by_length = {"base model": [0.9, 19.3], "fine-tuned": [4.9, 18.8]}
    notes = ("55% of answers", "45% of answers")
    metrics = {"base model": [9.2, 10.2, 4.8], "fine-tuned": [9.0, 9.9, None]}
    quantized = {**metrics, "fine-tuned model, q4_k_m": [9.4, 9.6, None]}
    by_category = {"base model": [17, 16, 15, 13, 8, 6, 6, 5, 3], "fine-tuned": [14, 15, 9, 8, 6, 2, 2, 2, 2]}
    cells = tuple(
        tuple(Cell(3, 11, 27.0) if (row + column) % 4 else Cell(0, 8, None) for column in range(6)) for row in range(9)
    )
    weakest = Heat(TITLE, f"{UNIT}, served template", CATEGORIES, LANGUAGES[:6], cells, 10)
    return {
        "dots": draw.dots(chart(CONDITIONS, LANGUAGES, by_language)),
        "noted bars": draw.bars(chart(CONDITIONS, LENGTHS, by_length, notes), shared=True),
        "bars": draw.bars(chart(CONDITIONS, CONDITIONS, metrics, unit="% of scored responses to harmful prompts")),
        "three bars": draw.bars(chart(CONDITIONS, CONDITIONS, quantized, unit="% of safe prompts refused")),
        "rows of bars": draw.bars(rows_of_bars(), shared=True),
        "rows": draw.rows(chart(CONDITIONS, CATEGORIES, by_category)),
        "heat": draw.heat(weakest),
        "flows": draw.flows(outcome_flows(two_conditions(tmp_path, settings))),
    }


def test_no_text_of_a_figure_runs_into_another_or_off_the_figure(tmp_path, settings, styled):
    for name, figure in figures(tmp_path, settings).items():
        assert clashes(figure) == [], name
        boxes = {text.get_text(): text.get_window_extent() for text in shown(figure)}
        off = [
            said for said, box in boxes.items() if box.x0 < 0 or box.y0 < 0 or not figure.bbox.contains(box.x1, box.y1)
        ]
        assert off == [], name


def test_every_text_reads_at_the_width_of_a_page(tmp_path, settings, styled):
    for name, figure in figures(tmp_path, settings).items():
        smallest = min(shown(figure), key=Text.get_fontsize)
        pixels = smallest.get_fontsize() * READ_AT / (draw.WIDTH * 72)
        assert pixels >= SMALLEST, (name, smallest.get_text())


def test_tick_labels_too_wide_for_their_ticks_are_tilted_with_a_line_between_them(styled):
    crowded = draw.dots(chart(CONDITIONS, LANGUAGES, {"base model": [5.0] * 7, "fine-tuned": [4.0] * 7}))
    crowded.draw_without_rendering()

    for axes in crowded.axes:
        labels = axes.get_xticklabels()
        assert [label.get_text() for label in labels] == list(LANGUAGES)
        tilt = {label.get_rotation() for label in labels}
        assert len(tilt) == 1 and 30 <= min(tilt) <= 60
        first, second = (axes.transData.transform((tick, 0))[0] for tick in (0, 1))
        line = labels[0].get_fontsize() * crowded.dpi / 72 * 1.2
        # neighbours run side by side, a line of text apart
        assert (second - first) * math.sin(math.radians(min(tilt))) >= line

    roomy = draw.dots(chart(CONDITIONS[:2], ("ru", "en", "All"), {"base model": [5.0] * 3, "fine-tuned": [4.0] * 3}))
    assert {label.get_rotation() for axes in roomy.axes for label in axes.get_xticklabels()} == {0.0}
    assert clashes(roomy) == []


def test_a_series_alone_at_a_tick_sits_on_the_tick(styled):
    values = {"base model": [9.2, 10.2, 4.8], "fine-tuned": [6.5, 5.9, None]}
    drawn = draw.bars(chart(CONDITIONS[:2], CONDITIONS, values))

    centres = sorted(round(bar.get_x() + bar.get_width() / 2, 6) for bar in drawn.axes[0].patches)
    # two bars around the first ticks, the one bar of the last tick on it
    assert centres[-1] == 2.0 and centres[0] == -centres[1] + 0.0 and centres[0] < 0 < centres[1]
    alone = drawn.axes[1].patches
    assert sorted(round(bar.get_x() + bar.get_width() / 2, 6) for bar in alone) == [0.0, 1.0, 2.0]


def test_the_bars_of_a_tick_keep_clear_of_the_next_tick_however_many_series(styled):
    values = {
        "base model": [9.2, 10.2, 4.8],
        "fine-tuned": [6.5, 5.9, 5.0],
        "fine-tuned model, q4_k_m": [6.9, 6.1, 5.2],
    }
    drawn = draw.bars(chart(CONDITIONS[:2], CONDITIONS, values))

    sides = sorted((bar.get_x(), bar.get_x() + bar.get_width()) for bar in drawn.axes[0].patches)
    assert len(sides) == 9 and all(right <= left for (_, right), (left, _) in pairwise(sides))


def test_a_value_written_over_a_bar_is_raised_clear_of_its_neighbour_and_stays_in_its_panel(styled):
    # two bars of nearly one height: their values would sit side by side, each wider than its bar
    level = {"base model": [10.4, 9.6, 14.0], "fine-tuned": [10.2, 9.5, 13.9]}
    drawn = draw.bars(chart(CONDITIONS, CONDITIONS, level, unit="% of safe prompts refused"))
    drawn.draw_without_rendering()

    for axes in drawn.axes[:2]:
        first, second = (text.get_window_extent() for text in axes.texts[::3][:2])
        assert [text.get_text() for text in axes.texts[::3][:2]] == ["10.4%", "10.2%"]
        assert second.y0 >= first.y1
        assert max(text.get_window_extent().y1 for text in axes.texts) <= axes.bbox.y1
    assert clashes(drawn) == []


def test_a_note_over_a_bar_keeps_to_its_bar_and_the_value_axis_is_no_longer_than_the_texts_need(styled):
    by_length = {"base model": [0.9, 19.3], "fine-tuned": [4.9, 18.8]}
    drawn = draw.bars(chart(CONDITIONS, LENGTHS, by_length, ("55% of answers", "45% of answers")), shared=True)
    drawn.draw_without_rendering()

    for axes in drawn.axes:
        notes = [text for text in axes.texts if "answers" in text.get_text()]
        assert len(notes) == len(axes.patches)
        lefts = sorted(bar.get_window_extent().x0 for bar in axes.patches)
        step = min(second - first for first, second in pairwise(lefts))
        assert all(note.get_window_extent().width <= step for note in notes)
    # the panels share the axis: it ends just over the highest text of any of them
    highest = max(text.get_window_extent().y1 for axes in drawn.axes for text in axes.texts)
    first = drawn.axes[0].bbox
    assert first.y0 + 0.95 * first.height <= highest <= first.y1


def test_panels_are_laid_in_rows_of_as_many_as_the_chart_asks_for(styled):
    drawn = draw.bars(rows_of_bars(), shared=True)
    drawn.draw_without_rendering()

    places = [(round(axes.get_position().y0, 3), round(axes.get_position().x0, 3)) for axes in drawn.axes]
    upper, lower = sorted({y for y, _ in places}, reverse=True)
    assert [y for y, _ in places] == [upper] * 3 + [lower] * 3
    assert [x for _, x in places[:3]] == [x for _, x in places[3:]] == sorted({x for _, x in places})
    assert [axes.get_title(loc="left").replace("\n", " ") for axes in drawn.axes] == [
        panel.title for panel in rows_of_bars().panels
    ]
    # five panels leave the last place of the second row empty
    short = rows_of_bars()
    fewer = draw.bars(replace(short, panels=short.panels[:5]), shared=True)
    assert [axes.get_visible() for axes in fewer.axes] == [True] * 5 + [False]
    assert clashes(fewer) == []
