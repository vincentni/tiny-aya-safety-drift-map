"""The probe table: the gate put on stored answers, the counts per system, and the change the gate makes."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from driftmap.table import probe
from driftmap.table.probe import BASE, GATED, TUNED

TEXTS = {"A": "one person", "B": "another person", "C": "unknown"}


def row(item, condition, base, tuned="C", language="en", correct=None):
    """A set row of base item ``item``: the unknown option is C, the stated answer of the twin is A."""
    right = correct or ("C" if condition == "ambiguous" else "A")
    return {
        "id": f"{language}-{item}-{condition}",
        "pair": f"{language}/{item}",
        "cluster": str(item),
        "condition": condition,
        "texts": TEXTS,
        "letter_unknown": "C",
        "letter_correct": right,
        "responses": {BASE: base, TUNED: tuned},
    }


def answered(item, condition, base, score, tuned="C", **more):
    return probe.answered(row(item, condition, base, tuned, **more), score, 0.5)


def test_the_gate_answers_with_the_unknown_option_where_the_probe_fires_and_with_the_stored_answer_elsewhere():
    fired = answered(1, "ambiguous", "B. another person", 0.5)
    quiet = answered(1, "disambiguated", "a", 0.49, tuned="I cannot say")

    assert (fired.fired, fired.letters) == (True, {BASE: "B", GATED: "C", TUNED: "C"})
    # under the threshold the base model's own letter stands; an answer that names no option stays unread
    assert (quiet.fired, quiet.letters) == (False, {BASE: "A", GATED: "A", TUNED: None})
    assert (fired.id, fired.pair, fired.cluster, fired.score) == ("en-1-ambiguous", "en/1", "1", 0.5)
    assert (quiet.letter_unknown, quiet.letter_correct) == ("C", "A")


def rows_of_four_items():
    """Four base items: the base model names a person on two open rows; the probe fires on every open row and
    on the settled row of item 4, where the base model had both rows right."""
    return [
        answered(1, "ambiguous", "A", 0.9),
        answered(1, "disambiguated", "A", 0.1),
        answered(2, "ambiguous", "B", 0.8),
        answered(2, "disambiguated", "B", 0.2, tuned="A"),
        answered(3, "ambiguous", "C", 0.7),
        answered(3, "disambiguated", "A", 0.3, tuned="A"),
        answered(4, "ambiguous", "C", 0.6, tuned="A"),
        answered(4, "disambiguated", "A", 0.5, tuned="A"),
    ]


def test_the_counts_per_system_are_over_open_rows_settled_rows_twins_and_all_rows():
    rows = rows_of_four_items()

    base, gated, tuned = (probe.measured(rows, system) for system in (BASE, GATED, TUNED))

    assert base == {
        "names_a_person": (2, 4),
        "disambiguated_correct": (3, 4),
        "twin_both_right": (2, 4),
        "wrong_answers": (3, 8),
        "answered_with_a_person": (4, 4),
        "disambiguated_answered_unknown": (0, 4),
        "unparseable": (0, 8),
    }
    # the gate takes every person off the open rows and one right answer off the settled ones
    assert gated["names_a_person"] == (0, 4) and gated["disambiguated_correct"] == (2, 4)
    assert gated["twin_both_right"] == (2, 4) and gated["wrong_answers"] == (2, 8)
    assert gated["answered_with_a_person"] == (3, 4) and gated["disambiguated_answered_unknown"] == (1, 4)
    assert tuned["names_a_person"] == (1, 4) and tuned["twin_both_right"] == (2, 4)
    # a base item with one row is no twin
    assert probe.measured(rows[:3], BASE)["twin_both_right"] == (0, 1)
    assert probe.measured(rows[2:7], BASE)["twin_both_right"] == (1, 2)
    unread = [answered(1, "ambiguous", "both of them", 0.1)]
    assert probe.measured(unread, BASE)["unparseable"] == (1, 1) and probe.measured(unread, BASE)["wrong_answers"] == (
        1,
        1,
    )
    assert probe.measured(unread, BASE)["names_a_person"] == (0, 1)


def test_the_auc_is_the_share_of_open_rows_scored_above_settled_ones_with_ties_as_half():
    rows = rows_of_four_items()
    assert probe.auc(rows) == 1.0
    tied = [*rows, answered(5, "ambiguous", "A", 0.3), answered(5, "disambiguated", "A", 0.6)]
    labels = [found.condition == "ambiguous" for found in tied]
    assert probe.auc(tied) == pytest.approx(roc_auc_score(labels, [found.score for found in tied]))
    assert 0.5 < probe.auc(tied) < 1.0
    rng = np.random.default_rng(3)
    many = [
        answered(at, ("ambiguous", "disambiguated")[at % 2], "A", round(float(rng.random()), 1)) for at in range(60)
    ]
    labels = [found.condition == "ambiguous" for found in many]
    assert probe.auc(many) == pytest.approx(roc_auc_score(labels, [found.score for found in many]), abs=1e-12)
    # one kind of row alone has no such share
    assert probe.auc([found for found in rows if found.condition == "ambiguous"]) is None


def test_the_change_is_the_gated_rate_minus_the_base_rate_resampled_by_base_item():
    rows = rows_of_four_items()

    twins = probe.change(rows, "twin_accuracy", seed=7)
    wrong = probe.change(rows, "wrong_answers", seed=7)

    # twins both right: 2 of 4 before and with the gate; item 1 is won, item 4 is lost
    clusters, moved, interval, better, worse, p_value = twins
    assert (clusters, moved, better, worse, p_value) == (4, 0.0, 1, 1, 1.0)
    assert -1 <= interval[0] <= 0.0 <= interval[1] <= 1 and interval[0] < interval[1]
    # wrong answers: 3 of 8 before, 2 of 8 with the gate; fewer on items 1 and 2, one more on item 4
    clusters, moved, interval, better, worse, p_value = wrong
    assert (clusters, moved, better, worse, p_value) == (4, -0.125, 2, 1, 1.0)
    assert interval[0] <= -0.125 <= interval[1]
    assert probe.change(rows, "wrong_answers", seed=7) == wrong
    # nothing moves: no interval to resample; one base item: nothing to resample
    still = [answered(item, "ambiguous", "C", 0.9) for item in (1, 2, 3)]
    assert probe.change(still, "wrong_answers", seed=7) == (3, 0.0, (0.0, 0.0), 0, 0, 1.0)
    assert probe.change(rows[:2], "wrong_answers", seed=7) is None
    # open rows alone make no twin, and so no base item to count twins over
    assert probe.change(still, "twin_accuracy", seed=7) is None
    # the rows of a base item in two languages are two twins of one base item
    both = [
        *rows,
        answered(1, "ambiguous", "A", 0.9, language="tr"),
        answered(1, "disambiguated", "A", 0.1, language="tr"),
    ]
    assert probe.change(both, "twin_accuracy", seed=7)[:2] == (4, pytest.approx(3 / 5 - 2 / 5))


def test_the_record_holds_every_set_and_the_markdown_says_how_the_gate_is_computed():
    sets = {"first": rows_of_four_items(), "open_only": [answered(at, "ambiguous", "A", 0.9) for at in range(3)]}

    tables = probe.build(sets, layer=20, c=0.1, threshold=0.5, seed=7, fine_tune="defense")

    assert tables.sets == ["first", "open_only"] and tables.fine_tune == "defense"
    assert tables.probe.model_dump(by_alias=True) == {"layer": 20, "C": 0.1, "threshold": 0.5}
    assert tables.systems == {BASE: "base model", GATED: "base model + probe", TUNED: "attribution fine-tune"}
    assert (tables.resamples, tables.seeds.seed, tables.seeds.bootstrap) == (10000, 7, 7)
    rates = {(found.set, found.system, found.metric): found for found in tables.rates}
    assert len(rates) == 2 * 3 * len(probe.METRICS)
    gated = rates["first", GATED, "wrong_answers"]
    assert (gated.k, gated.n, gated.rate) == (2, 8, 0.25) and gated.ci[0] < 0.25 < gated.ci[1]
    assert (
        rates["open_only", BASE, "twin_both_right"].n == 0 and rates["open_only", BASE, "twin_both_right"].rate is None
    )
    first, second = tables.readings
    assert (first.auc, first.fired_on_ambiguous, first.ambiguous) == (1.0, 4, 4)
    assert (first.fired_on_disambiguated, first.disambiguated) == (1, 4)
    assert (second.auc, second.fired_on_ambiguous, second.disambiguated) == (None, 3, 0)
    assert [(found.set, found.metric) for found in tables.changes] == [
        ("first", "twin_accuracy"),
        ("first", "wrong_answers"),
        ("open_only", "wrong_answers"),
    ]
    assert probe.ProbeTables.model_validate_json(tables.model_dump_json(by_alias=True)) == tables
    assert probe.build(sets, layer=20, c=0.1, threshold=0.5, seed=7).systems[TUNED] == "fine-tuned"

    shown = probe.render(tables)
    lines = shown.splitlines()
    assert lines[0] == "# An activation probe that makes the base model abstain"
    assert "at layer 20, at the last token of the prompt" in lines[2] and "(C 0.1, threshold 0.5)" in lines[2]
    assert (
        "The gate is computed after the fact, from the base model's stored answer and the probe's reading" in lines[2]
    )
    assert "no hook runs in a serving model and no weight is changed" in lines[2]
    assert [line for line in lines if line.startswith("## ")] == [
        *(f"## {probe.METRICS[metric]}" for metric in probe.SHOWN),
        "## The probe on each set",
        "## Change with the probe, base model + probe minus base model",
    ]
    assert "| set | base model | base model + probe | attribution fine-tune |" in lines
    assert "| first | 37.5% (3/8) | 25.0% (2/8) | 25.0% (2/8) |" in lines
    assert "| open_only | n = 0 | n = 0 | n = 0 |" in lines
    assert "| first | 1.0000 | 4/4 | 1/4 |" in lines and "| open_only | n/a | 3/3 | 0/0 |" in lines
    moved = next(line for line in lines if line.startswith("| first | +0.0 points (95% CI "))
    assert "; p = 1.000; 4 base items; better on 1, worse on 1) | -12.5 points (95% CI " in moved
    assert moved.endswith("; p = 1.000; 4 base items; better on 2, worse on 1) |")
    assert next(line for line in lines if line.startswith("| open_only | n/a | -100.0 points"))
    for at, line in enumerate(lines):
        if line.startswith("## ") or (line.startswith("|") and not lines[at - 1].startswith("|")):
            assert lines[at - 1] == "", line
