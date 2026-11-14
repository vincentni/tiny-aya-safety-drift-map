from typing import NamedTuple

import pytest

from driftmap.figures import data
from driftmap.figures.data import Cell, Estimate
from driftmap.table.breakdown import STATES
from driftmap.table.breakdown import build as build_breakdown
from driftmap.table.build import build
from driftmap.table.rates import wilson
from tests.fakes.breakdown import HATE, lay, prompts, two_conditions
from tests.fakes.labels import column_rows, make_settings, targets, write_labels

# weakest first by the baseline of the fixture, then as configured
LANGUAGES = ("Russian", "English", "French", "Spanish", "Hindi", "Arabic")
HARMFUL = "complied-harmful"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def tables(tmp_path, settings, name, before, after, *, preamble="keep", chrf=(0.41, 0.47), refused=(3, 3)):
    """Result tables of twenty prompts per language; ``before`` and ``after`` give the harmful answers per language."""
    rows = column_rows("baseline", n=20, unsafe=before, refused=refused[0], chrf=chrf[0])
    rows += column_rows("drifted", n=20, unsafe=after, refused=refused[1], chrf=chrf[1])
    return build(write_labels(tmp_path / name, rows), targets(20), [], settings, preamble=preamble)


@pytest.fixture
def both(tmp_path, settings):
    return {
        "keep": tables(tmp_path, settings, "keep", {"en": 8, "ru": 10}, {"en": 2, "ru": 3}),
        "strip": tables(tmp_path, settings, "strip", {"en": 9, "ru": 9}, {"en": 1}, preamble="strip", chrf=(0.3, 0.45)),
    }


def percent(k, n):
    low, high = wilson(k, n)
    return Estimate(pytest.approx(k / n * 100), pytest.approx(low * 100), pytest.approx(high * 100))


def test_the_language_figure_shows_harmful_answers_from_the_weakest_language_down(both, settings):
    chart = data.harmful_by_language(both, settings)

    assert [panel.title for panel in chart.panels] == ["Served template", "No safety preamble"]
    served, bare = chart.panels
    # both panels keep the order of the first condition's first column; the pooled rate is last
    assert served.labels == bare.labels == (*LANGUAGES, "All")
    assert served.unit == "Harmful answers (% of scored responses)"
    assert [series.name for series in served.series] == ["baseline", "fine-tuned"]
    before, after = served.series
    assert before.points[0] == percent(10, 20) and before.points[1] == percent(8, 20)
    assert before.points[2] == percent(0, 20) and after.points[:2] == (percent(3, 20), percent(2, 20))
    assert before.points[-1] == percent(18, 120) and after.points[-1] == percent(5, 120)
    assert bare.series[0].points[:2] == (percent(9, 20), percent(9, 20)) and bare.series[1].points[1] == percent(1, 20)
    lowered = "Served template: fine-tuning on benign data lowered the harmful-response rate from 15.0% to 4.2%"
    assert chart.title == f"{lowered}; Russian is weakest at baseline (50.0%)"


def test_the_language_title_follows_the_pooled_rates_of_the_first_condition(tmp_path, both, settings):
    worse = tables(tmp_path, settings, "worse", {"en": 2}, {"en": 8, "ru": 10})
    chart = data.harmful_by_language({"strip": worse, "keep": both["keep"]}, settings)
    assert chart.title == (
        "No safety preamble: fine-tuning on benign data raised the harmful-response rate from 1.7% to 15.0%; "
        "English is weakest at baseline (10.0%)"
    )
    assert chart.panels[1].labels[:2] == ("English", "French")
    same = tables(tmp_path, settings, "same", {"en": 2}, {"ru": 2})
    assert data.harmful_by_language({"keep": same}, settings).title == (
        "Served template: fine-tuning on benign data left the harmful-response rate at 1.7%; "
        "English is weakest at baseline (10.0%)"
    )
    # a column that is not tabled is not drawn, and one column has no change to state
    alone = both["keep"].model_copy(deep=True)
    alone.columns["drifted"].state = "not_comparable"
    chart = data.harmful_by_language({"keep": alone}, settings)
    assert [series.name for series in chart.panels[0].series] == ["baseline"]
    assert chart.title == "Harmful answers by language" and chart.panels[0].labels[0] == "Russian"


def test_the_before_after_figure_turns_the_safe_rate_into_harmful_answers_and_words_each_change(both, settings):
    chart = data.before_after(both, settings)

    harmful, over_refusal, chrf = chart.panels
    assert [panel.title for panel in chart.panels] == ["Harmful answers", "Over-refusal (XSTest)", "Task score (chrF)"]
    assert harmful.labels == ("Served template", "No safety preamble")
    assert [series.name for series in harmful.series] == ["baseline", "fine-tuned"]
    assert harmful.series[0].points == (percent(18, 120), percent(18, 120))
    assert harmful.series[1].points == (percent(5, 120), percent(1, 120))
    assert over_refusal.series[0].points == (percent(3, 20), percent(3, 20))
    assert chrf.series[0].points == (Estimate(pytest.approx(41.0)), Estimate(pytest.approx(30.0)))
    assert chrf.series[1].points == (Estimate(pytest.approx(47.0)), Estimate(pytest.approx(45.0)))
    assert harmful.unit.startswith("% of") and over_refusal.unit.startswith("% of") and chrf.unit.startswith("chrF")
    assert chart.title == (
        "After fine-tuning: fewer harmful answers, no clear change in over-refusal, a higher task score "
        "under both prompt conditions"
    )


def test_the_before_after_title_says_so_when_the_conditions_disagree_or_a_change_is_within_its_interval(
    tmp_path, both, settings
):
    # one more harmful answer of 120 is within the interval; more refusals and a lower score are not
    other = tables(tmp_path, settings, "other", {"en": 2}, {"en": 3}, chrf=(0.5, 0.4), refused=(2, 12))
    assert data.before_after({"keep": other}, settings).title == (
        "After fine-tuning: no clear change in harmful answers, more over-refusal, a lower task score"
    )
    assert data.before_after({"keep": both["keep"], "strip": other}, settings).title == (
        "After fine-tuning: mixed results on harmful answers, mixed results on over-refusal (XSTest), "
        "mixed results on task score (chrF) under both prompt conditions"
    )
    three = data.before_after({"keep": both["keep"], "strip": both["strip"], "patched": both["keep"]}, settings)
    assert three.title.endswith("a higher task score under every prompt condition")
    assert three.panels[0].labels == ("Served template", "No safety preamble", "Patched")


def test_the_category_figure_lists_the_worst_category_of_the_first_condition_first(tmp_path, settings):
    chart = data.by_category(two_conditions(tmp_path, settings))

    served, bare = chart.panels
    assert (served.title, bare.title) == ("Served template", "No safety preamble")
    assert served.labels == bare.labels == ("Self-Harm", "Hate Speech", "Profanity")
    assert [series.name for series in served.series] == ["baseline", "fine-tuned"]
    assert served.series[0].points == (percent(6, 20), percent(12, 49), percent(3, 19))
    assert served.series[1].points == (percent(7, 20), percent(5, 50), percent(1, 20))
    # the second condition keeps the order of the first
    assert bare.series[0].points == (percent(6, 20), percent(17, 50), percent(5, 20))
    assert chart.title == (
        "Most harmful answers at baseline: Self-Harm (30.0%, served template); "
        "fine-tuning lowers the rate in 2 of 3 categories"
    )


def test_the_heat_map_holds_the_first_column_of_the_first_condition_and_names_its_weakest_cell(tmp_path, settings):
    heat = data.category_by_language(two_conditions(tmp_path, settings))

    assert heat.rows == ("Self-Harm", "Hate Speech", "Profanity") and heat.columns == ("English", "Hindi")
    assert heat.cells == (
        # under ten scored responses: counts and no rate
        (Cell(2, 12, pytest.approx(100 / 6)), Cell(4, 8, None)),
        (Cell(9, 30, pytest.approx(30.0)), Cell(3, 19, pytest.approx(300 / 19))),
        # no English profanity prompt
        (None, Cell(3, 19, pytest.approx(300 / 19))),
    )
    assert heat.floor == 10 and heat.unit == "Harmful answers (% of scored responses), served template"
    assert heat.title == "Weakest cell at baseline: Hate Speech in English, 9 of 30 answers harmful (30.0%)"


def test_a_long_category_name_is_shortened_for_the_axis(tmp_path, settings):
    long = "Harms of Representation Allocation and Quality of Service"
    plan = prompts("en", [long], 20, {"harmful": 5}, {"harmful": 1}) + prompts("en", [HATE], 20, {"harmful": 1})
    labels = {"keep": tmp_path / "keep"}
    found = build_breakdown(labels, {"harmful": lay(labels["keep"], plan)}, [], settings)

    assert data.by_category(found).panels[0].labels == ("Representation & allocation harms", "Hate Speech")
    assert data.category_by_language(found).title.startswith(
        "Weakest cell at baseline: Representation & allocation harms in English, 5 of 20"
    )


def test_the_outcome_figure_counts_each_prompt_once_on_either_side_and_ties_the_bands_to_the_bars(tmp_path, settings):
    flows = data.outcome_flows(two_conditions(tmp_path, settings))

    assert flows.states == STATES and [panel.title for panel in flows.panels] == [
        "Served template",
        "No safety preamble",
    ]
    served = flows.panels[0]
    assert (served.names, served.n) == (("baseline", "fine-tuned"), 70)
    assert served.before == {"refused": 47, "complied-harmless": 4, HARMFUL: 18, "unscored": 1}
    assert served.after == {"refused": 52, "complied-harmless": 6, HARMFUL: 12, "unscored": 0}
    moved = {(band.before, band.after): band for band in served.bands}
    assert {key: band.n for key, band in moved.items()} == {
        ("refused", "refused"): 46,
        ("refused", HARMFUL): 1,
        ("complied-harmless", "refused"): 3,
        ("complied-harmless", "complied-harmless"): 1,
        (HARMFUL, "refused"): 2,
        (HARMFUL, "complied-harmless"): 5,
        (HARMFUL, HARMFUL): 11,
        ("unscored", "refused"): 1,
    }
    # on the earlier bar a band sits in its outcome's segment, on the later bar in the segment it moved to
    share = 100 / 70
    assert moved[HARMFUL, "refused"].left == pytest.approx((51 * share, 53 * share))
    assert moved[HARMFUL, "refused"].right == pytest.approx((49 * share, 51 * share))
    assert moved["refused", HARMFUL].right == pytest.approx((58 * share, 59 * share))
    for side in ("left", "right"):
        spans = sorted(getattr(band, side) for band in served.bands)
        assert spans[0][0] == 0 and spans[-1][1] == pytest.approx(100)
        assert [end for _, end in spans[:-1]] == pytest.approx([start for start, _ in spans[1:]])
    assert served.note == (
        "No longer harmful: 2 to refused, 5 to complied-harmless. "
        "Newly harmful: 1 from refused, 0 from complied-harmless."
    )
    assert flows.title == (
        "Served template: 7 harmful answers turned into refusals or harmless answers after fine-tuning, "
        "1 went the other way"
    )


def test_one_column_alone_has_rates_to_draw_and_no_outcomes_to_compare(tmp_path, settings):
    plan = prompts("en", [HATE], 20, {"harmful": 5})
    labels = {"keep": tmp_path / "keep"}
    found = build_breakdown(labels, {"harmful": lay(labels["keep"], plan, columns=("baseline",))}, [], settings)

    assert data.outcome_flows(found).panels == ()
    chart = data.by_category(found)
    assert [series.name for series in chart.panels[0].series] == ["baseline"]
    assert chart.title == "Most harmful answers at baseline: Hate Speech (25.0%, served template)"


def test_the_length_figure_splits_the_rate_by_cut_off_answers_and_notes_the_share_of_each(tmp_path, settings):
    chart = data.truncation(two_conditions(tmp_path, settings))

    served, bare = chart.panels
    assert (served.title, bare.title) == ("Served template", "No safety preamble")
    assert served.labels == ("Complete answers", "Answers cut off at the token cap")
    before, after = served.series
    assert (before.name, after.name) == ("baseline", "fine-tuned")
    assert before.points == (percent(3, 47), percent(15, 22)) and after.points == (percent(4, 62), percent(8, 8))
    # 22 of 70 answers ran into the cap at baseline, 8 after fine-tuning
    assert before.notes == ("69% of answers", "31% of answers") and after.notes == ("89% of answers", "11% of answers")
    assert bare.series[0].points == (percent(2, 47), percent(21, 23)) and bare.series[0].notes[1] == "33% of answers"
    assert chart.title == (
        "Fewer harmful answers after fine-tuning, mostly because answers got shorter: "
        "31% ran into the token cap before, 11% after (served template)"
    )


def lengths(tmp_path, settings, name, before, after, *, answers=True):
    labels, responses = {"keep": tmp_path / name}, {"keep": tmp_path / f"{name}-answers"}
    tagged = {"harmful": lay(labels["keep"], prompts("en", [HATE], 20, before, after), responses=responses["keep"])}
    return build_breakdown(labels, tagged, [], settings, responses=responses if answers else None)


def test_the_length_title_says_what_the_moves_between_lengths_show(tmp_path, settings):
    # five harmful answers fewer, none of them an answer that stopped running into the cap
    elsewhere = lengths(tmp_path, settings, "a", {"harmful": 6, "refused*": 2}, {"harmful": 1, "refused*": 3})
    assert data.truncation(elsewhere).title == (
        "Fewer harmful answers after fine-tuning, and not mostly through shorter answers: "
        "10% ran into the token cap before, 15% after (served template)"
    )
    more = lengths(tmp_path, settings, "b", {"harmful*": 1}, {"harmful": 3})
    assert data.truncation(more).title == (
        "No fewer harmful answers after fine-tuning: 5% ran into the token cap before, 0% after (served template)"
    )
    # nothing to draw without the stored answers
    unread = lengths(tmp_path, settings, "c", {"harmful*": 1}, {"harmful": 3}, answers=False)
    assert data.truncation(unread).panels == ()


class Pairing(NamedTuple):
    set: str
    language: str
    metric: str
    n_pairs: int
    parent: float | None
    variant: float | None
    delta: float | None
    ci: tuple[float, float] | None
    p_value: float | None


def test_the_variants_figure_sets_the_rewritten_prompts_beside_the_same_prompts_as_written():
    results = {
        "baseline": [
            Pairing("harmful_romanized", "hi", "harmful", 100, 0.10, 0.30, 0.20, (0.10, 0.30), 0.001),
            Pairing("harmful_romanized", "all", "harmful", 300, 0.10, 0.25, 0.15, (0.09, 0.21), 0.0001),
            Pairing("harmful_codemixed", "all", "harmful", 500, 0.08, 0.10, 0.02, (-0.01, 0.05), 0.2),
            Pairing("benign_romanized", "all", "chrf", 300, 0.30, 0.20, -0.10, (-0.12, -0.08), 0.0001),
        ],
        "drifted": [
            Pairing("harmful_romanized", "all", "harmful", 300, 0.06, 0.10, 0.04, (0.0, 0.08), 0.08),
            Pairing("harmful_stretched", "all", "harmful", 0, None, None, None, None, None),
        ],
    }

    chart = data.script_variants(results)

    base, tuned = chart.panels
    assert (base.title, tuned.title) == ("Baseline", "Fine-tuned")
    # the pooled rows of the harmful sets only
    assert base.labels == ("Latin script", "Mixed with English") and tuned.labels == ("Latin script",)
    written, rewritten = base.series
    assert (written.name, rewritten.name) == ("as written", "rewritten")
    assert written.points == (percent(30, 300), percent(40, 500))
    assert rewritten.points == (percent(75, 300), percent(50, 500))
    assert rewritten.notes == ("+15.0 points", "+2.0 points") and written.notes == ()
    assert chart.title == (
        "Rewritten prompts (Latin script) draw more harmful answers at baseline: 25.0% against 10.0% as written"
    )
    unclear = data.script_variants({"drifted": results["drifted"]})
    assert unclear.title == "Rewriting the prompts does not clearly raise the harmful-response rate"
    assert (
        data.script_variants({}).panels == ()
        and data.script_variants({"baseline": results["baseline"][3:]}).panels == ()
    )


def test_the_patch_figure_sets_each_model_with_the_patch_beside_itself_without(tmp_path, both, settings):
    labels = {"keep": tmp_path / "plain", "patch": tmp_path / "patched"}
    tagged = lay(labels["keep"], prompts("en", [HATE], 20, {"harmful": 6}, {"harmful": 2}))
    lay(labels["patch"], prompts("en", [HATE], 20, {"harmful": 2}, {"harmful": 1}))
    found = build_breakdown(labels, {"harmful": tagged}, [], settings)
    assert found.conditions == {"keep": "served template", "patch": "safety patch"}

    chart = data.safety_patch(found, {"keep": both["keep"]}, settings)

    (harmful,) = chart.panels
    assert (harmful.title, harmful.labels) == ("Harmful answers", ("Baseline", "Fine-tuned"))
    assert [series.name for series in harmful.series] == ["served template", "safety patch"]
    assert harmful.series[0].points == (percent(6, 20), percent(2, 20))
    assert harmful.series[1].points == (percent(2, 20), percent(1, 20))
    assert chart.title == (
        "With the safety patch, harmful answers go from 30.0% to 10.0% for the baseline model "
        "and from 10.0% to 5.0% for the fine-tuned model"
    )
    # over-refusal joins when both conditions are tabled
    patched = tables(tmp_path, settings, "patched-tables", {"en": 2}, {"en": 1}, refused=(2, 12))
    harmful, refused = data.safety_patch(found, {"keep": both["keep"], "patch": patched}, settings).panels
    assert refused.title == "Over-refusal (XSTest)" and refused.labels == harmful.labels
    assert refused.series[0].points == (percent(3, 20), percent(3, 20))
    assert refused.series[1].points == (percent(2, 20), percent(12, 20))
    # no patched answers, no figure
    assert data.safety_patch(two_conditions(tmp_path, settings), both, settings).panels == ()
