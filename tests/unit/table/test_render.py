import re

import pytest
from pydantic import ValidationError

from driftmap.core.records import DeltaRow, RateRow
from driftmap.table.build import build
from driftmap.table.render import delta_cell, render
from tests.fakes.labels import column_rows, judge_config, make_settings, targets, write_labels

NUMBER = re.compile(r"\d\.\d")


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def cells(markdown):
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    return [cell.strip() for row in rows for cell in row.strip("|").split("|")]


def test_every_number_cell_names_its_n_judge_interval_and_preamble(tmp_path, settings):
    """A rate row needs its interval field, and no rendered number stands alone (TS-TBL-10)."""
    fields = {"row": "en", "checkpoint": "c", "metric": "safe", "k": 9, "n": 10, "rate": 0.9}
    with pytest.raises(ValidationError, match="ci"):
        RateRow(**fields, judge=judge_config(), preamble="keep")
    with pytest.raises(ValidationError, match="judge"):
        RateRow(**fields, ci=(0.5, 1.0), preamble="keep")

    rows = column_rows("baseline", n=20, unsafe=2, refused=3, chrf=0.41) + column_rows(
        "drifted", n=20, unsafe=9, chrf=0.47
    )
    markdown = render(build(write_labels(tmp_path / "labels", rows), targets(20), [], settings), settings)

    numbers = [cell for cell in cells(markdown) if NUMBER.search(cell)]
    assert len(numbers) > 60
    for cell in numbers:
        assert re.search(r"n=\d+", cell) and "preamble=keep" in cell, cell
        assert "chrF1" in cell or ("judge=PolyGuard-Qwen" in cell and "95% CI" in cell), cell
    assert "| 90.0% (n=20; 95% CI 69.9% to 97.2%; judge=PolyGuard-Qwen; preamble=keep) |" in markdown
    assert "| 41.0 (n=20; chrF1; preamble=keep) |" in markdown


def test_an_absent_column_is_stated_with_its_reason(tmp_path, settings):
    """Four columns in order, not run with the reason, n/a cells and deltas in points with words (TS-TBL-15)."""
    rows = column_rows("baseline", n=20, unsafe=2, chrf=0.41) + column_rows("drifted", n=20, unsafe=9, chrf=0.47)
    labels = write_labels(tmp_path / "labels", rows)
    tables = build(labels, targets(20), [], settings, not_run={"repaired": "left for the next budget window"})

    markdown = render(tables, settings)

    assert "| language | baseline | drifted | repaired | repaired_q4 |" in markdown
    assert markdown.count("not run: left for the next budget window") > 10
    assert "not run: no generation run recorded" in markdown
    change = (
        "-35.0 points, less safe (95% CI -43.3 to -26.7; n=120 pairs; p < 0.001; judge=PolyGuard-Qwen; preamble=keep)"
    )
    assert f"| all | safe | {change} | n/a | n/a |" in markdown
    assert "+6.0 points, higher chrF (95% CI +6.0 to +6.0; n=120 pairs; p < 0.001; chrF1; preamble=keep)" in markdown
    assert "- repaired (weights): not run: left for the next budget window" in markdown
    assert "- drift_below_threshold: no" in markdown
    assert not re.search(r"TODO|TBD|XX|significant|nan|None", markdown)

    alone = render(build(write_labels(tmp_path / "alone", column_rows("baseline")), targets(), [], settings), settings)
    assert "| all | safe | n/a | n/a | n/a |" in alone and "not run: no generation run recorded" in alone


def test_p_prints_exactly_down_to_one_in_a_thousand():
    def cell(p_value):
        delta = DeltaRow(
            row="all", a="baseline", b="drifted", metric="safe", n_pairs=9, delta=0.1, ci=(0.0, 0.2), p_value=p_value
        )
        return delta_cell(delta, "9 pairs", "judge=J", "keep", ("safer", "less safe"))

    assert "; p = 0.001; " in cell(0.001) and "; p < 0.001; " in cell(0.000999)
    assert "; p = 1.000; " in cell(1.0) and "; p = 0.050; " in cell(0.05)
    assert "p " not in cell(None)


def test_a_flag_nobody_could_evaluate_is_not_printed_as_no(tmp_path, settings):
    alone = build(write_labels(tmp_path / "alone", column_rows("baseline")), targets(), [], settings)
    assert alone.flags == {}
    markdown = render(alone, settings)
    assert "- drift_below_threshold: not evaluated (drifted not run)" in markdown
    assert "- repair_below_safety_bar: not evaluated (repaired not run)" in markdown
    assert not re.search(r"^- \w+: (yes|no)$", markdown, re.MULTILINE)

    rows = column_rows("baseline") + column_rows("drifted", unsafe=2)
    both = build(write_labels(tmp_path / "both", rows), targets(), [], settings)
    assert both.flags == {"drift_below_threshold": False, "no_task_gain": True}
    markdown = render(both, settings)
    assert "- drift_below_threshold: no\n" in markdown and "- no_task_gain: yes\n" in markdown
    assert "- repair_below_bar: not evaluated (repaired not run)" in markdown

    harmful = [row for row in rows if row.set != "benign"]
    unscored = build(write_labels(tmp_path / "harmful", harmful), targets(), [], settings)
    assert "- no_task_gain: not evaluated (no scored rows)" in render(unscored, settings)
