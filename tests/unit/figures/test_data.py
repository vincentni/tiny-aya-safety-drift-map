import pytest

from driftmap.figures import data
from driftmap.figures.data import Cell, Estimate
from driftmap.table.breakdown import STATES
from driftmap.table.breakdown import build as build_breakdown
from driftmap.table.build import build
from driftmap.table.rates import wilson
from tests.fakes.breakdown import HATE, lay, prompts, stored_variants, two_conditions, variant_row
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
    assert [series.name for series in served.series] == ["base model", "fine-tuned"]
    before, after = served.series
    assert before.points[0] == percent(10, 20) and before.points[1] == percent(8, 20)
    assert before.points[2] == percent(0, 20) and after.points[:2] == (percent(3, 20), percent(2, 20))
    assert before.points[-1] == percent(18, 120) and after.points[-1] == percent(5, 120)
    assert bare.series[0].points[:2] == (percent(9, 20), percent(9, 20)) and bare.series[1].points[1] == percent(1, 20)
    # the title names the two ends at the first column and no cause
    assert chart.title == "Harmful answers by language: 50% in Russian, 0% in Arabic (base model, served template)"


def test_a_quantized_file_of_the_fine_tuned_model_is_named_as_the_tables_name_it(tmp_path, settings):
    gguf = f"vincentnkq/tiny-aya-global-drift-GGUF@{'c' * 40}#model-q4_k_m.gguf"
    rows = column_rows("baseline", n=20, unsafe=2) + column_rows("drifted", n=20, unsafe=4)
    rows += column_rows("repaired_q4", n=20, unsafe=5, checkpoint=gguf)
    held = {"keep": build(write_labels(tmp_path / "q4", rows), targets(20), [], settings)}
    names = ["base model", "fine-tuned", "fine-tuned model, q4_k_m"]

    assert [series.name for series in data.harmful_by_language(held, settings).panels[0].series] == names
    assert [series.name for series in data.before_after(held, settings).panels[0].series] == names
    # names passed in, as a breakdown's are, do not hide what the file was made from
    passed = {"baseline": "baseline", "drifted": "aggressive fine-tune", "repaired_q4": "repaired_q4"}
    shown = [series.name for series in data.harmful_by_language(held, settings, passed).panels[0].series]
    assert shown == ["baseline", "aggressive fine-tune", "fine-tuned model, q4_k_m"]


def test_the_language_title_names_the_two_ends_of_the_first_condition(tmp_path, both, settings):
    worse = tables(tmp_path, settings, "worse", {"en": 2}, {"en": 8, "ru": 10})
    chart = data.harmful_by_language({"strip": worse, "keep": both["keep"]}, settings)
    assert chart.title == "Harmful answers by language: 10% in English, 0% in Arabic (base model, no safety preamble)"
    assert chart.panels[1].labels[:2] == ("English", "French")
    # a column that is not tabled is not drawn; a column goes by the name it is given
    alone = both["keep"].model_copy(deep=True)
    alone.columns["drifted"].state = "not_comparable"
    chart = data.harmful_by_language({"keep": alone}, settings)
    assert [series.name for series in chart.panels[0].series] == ["base model"]
    assert chart.panels[0].labels[0] == "Russian"
    named = data.harmful_by_language(both, settings, {"drifted": "aggressive fine-tune"})
    assert [series.name for series in named.panels[0].series] == ["base model", "aggressive fine-tune"]
    assert named.title.endswith("(base model, served template)")


def paired(found, metric, name):
    (row,) = [
        row for row in found.deltas if (row.metric, row.row, row.a, row.b) == (metric, name, "baseline", "drifted")
    ]
    return row


def test_the_before_after_figure_turns_the_safe_rate_into_harmful_answers_and_states_the_paired_changes(both, settings):
    chart = data.before_after(both, settings)

    harmful, over_refusal, chrf = chart.panels
    assert [panel.title for panel in chart.panels] == ["Harmful answers", "Over-refusal (XSTest)", "Task score (chrF)"]
    assert harmful.labels == ("Served template", "No safety preamble")
    assert [series.name for series in harmful.series] == ["base model", "fine-tuned"]
    assert harmful.series[0].points == (percent(18, 120), percent(18, 120))
    assert harmful.series[1].points == (percent(5, 120), percent(1, 120))
    assert over_refusal.series[0].points == (percent(3, 20), percent(3, 20))
    assert chrf.series[0].points == (Estimate(pytest.approx(41.0)), Estimate(pytest.approx(30.0)))
    assert chrf.series[1].points == (Estimate(pytest.approx(47.0)), Estimate(pytest.approx(45.0)))
    assert harmful.unit.startswith("% of") and over_refusal.unit.startswith("% of") and chrf.unit.startswith("chrF")
    # counts and the tables' paired changes of the first condition, the safe rate read as harmful answers
    safe = paired(both["keep"], "safe", "all")
    assert safe.delta == pytest.approx(13 / 120) and safe.ci[0] > 0
    change = f"{-safe.delta * 100:+.1f} points (95% CI {-safe.ci[1] * 100:+.1f} to {-safe.ci[0] * 100:+.1f})"
    assert change.startswith("-10.8 points (95% CI -1")
    assert chart.title == (
        f"Served template, base model to fine-tuned: harmful answers 18 to 5 of 120, {change}; "
        "over-refusal +0.0 points (+0.0 to +0.0); chrF +6.0 (+6.0 to +6.0)"
    )


def test_the_before_after_title_states_changes_and_never_a_cause(tmp_path, both, settings):
    other = tables(tmp_path, settings, "other", {"en": 2}, {"en": 3}, chrf=(0.5, 0.4), refused=(2, 12))
    refused, score = paired(other, "over_refusal", "en"), paired(other, "chrf", "all")
    title = data.before_after(
        {"strip": other, "keep": both["keep"]}, settings, {"drifted": "aggressive fine-tune"}
    ).title
    assert title.startswith(
        "No safety preamble, base model to aggressive fine-tune: harmful answers 2 to 3 of 120, +0.8 points ("
    )
    assert f"; over-refusal +50.0 points ({refused.ci[0] * 100:+.1f} to {refused.ci[1] * 100:+.1f}); " in title
    assert title.endswith(f"chrF -10.0 ({score.ci[0] * 100:+.1f} to {score.ci[1] * 100:+.1f})")
    for word in ("safer", "because", "fewer", "higher", "unchanged"):
        assert word not in title
    # the scored answers of the two columns differ in number
    uneven = both["keep"].model_copy(deep=True)
    row = next(row for row in uneven.safety if (row.metric, row.row) == ("safe", "all") and "drift" in row.checkpoint)
    row.k, row.n = 114, 119
    assert "harmful answers 18 of 120 to 5 of 119, " in data.before_after({"keep": uneven}, settings).title
    # one column has no change to state
    alone = both["keep"].model_copy(deep=True)
    alone.columns["drifted"].state = "not_comparable"
    assert data.before_after({"keep": alone}, settings).title == "Harmful answers, over-refusal and task score"
    three = data.before_after({"keep": both["keep"], "strip": both["strip"], "patched": both["keep"]}, settings)
    assert three.panels[0].labels == ("Served template", "No safety preamble", "Patched")


def test_the_category_figure_lists_the_worst_category_of_the_first_condition_first(tmp_path, settings):
    chart = data.by_category(two_conditions(tmp_path, settings))

    served, bare = chart.panels
    assert (served.title, bare.title) == ("Served template", "No safety preamble")
    assert served.labels == bare.labels == ("Self-Harm", "Hate Speech", "Profanity")
    assert [series.name for series in served.series] == ["base model", "fine-tuned"]
    assert served.series[0].points == (percent(6, 20), percent(12, 49), percent(3, 19))
    assert served.series[1].points == (percent(7, 20), percent(5, 50), percent(1, 20))
    # the second condition keeps the order of the first
    assert bare.series[0].points == (percent(6, 20), percent(17, 50), percent(5, 20))
    assert chart.title == (
        "Most harmful answers by harm category: Self-Harm (30.0%), then Hate Speech (24.5%) "
        "(base model, served template)"
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
    assert heat.title == "Weakest cell (base model): Hate Speech in English, 9 of 30 answers harmful (30.0%)"


def test_a_long_category_name_is_shortened_for_the_axis(tmp_path, settings):
    long = "Harms of Representation Allocation and Quality of Service"
    plan = prompts("en", [long], 20, {"harmful": 5}, {"harmful": 1}) + prompts("en", [HATE], 20, {"harmful": 1})
    labels = {"keep": tmp_path / "keep"}
    found = build_breakdown(labels, {"harmful": lay(labels["keep"], plan)}, [], settings)

    assert data.by_category(found).panels[0].labels == ("Representation & allocation harms", "Hate Speech")
    assert data.category_by_language(found).title.startswith(
        "Weakest cell (base model): Representation & allocation harms in English, 5 of 20"
    )


def test_the_outcome_figure_counts_each_prompt_once_on_either_side_and_ties_the_bands_to_the_bars(tmp_path, settings):
    flows = data.outcome_flows(two_conditions(tmp_path, settings))

    assert flows.states == STATES and [panel.title for panel in flows.panels] == [
        "Served template",
        "No safety preamble",
    ]
    served = flows.panels[0]
    assert (served.names, served.n) == (("base model", "fine-tuned"), 70)
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
    assert [series.name for series in chart.panels[0].series] == ["base model"]
    assert chart.title == "Most harmful answers by harm category: Hate Speech (25.0%) (base model, served template)"


def test_the_length_figure_splits_the_rate_by_cut_off_answers_and_notes_the_share_of_each(tmp_path, settings):
    chart = data.truncation(two_conditions(tmp_path, settings))

    served, bare = chart.panels
    assert (served.title, bare.title) == ("Served template", "No safety preamble")
    assert served.labels == ("Complete answers", "Answers cut off at the 256-token limit")
    before, after = served.series
    assert (before.name, after.name) == ("base model", "fine-tuned")
    assert before.points == (percent(3, 47), percent(15, 22)) and after.points == (percent(4, 62), percent(8, 8))
    # 22 of 70 answers reached the limit before fine-tuning, 8 after
    assert before.notes == ("69% of answers", "31% of answers") and after.notes == ("89% of answers", "11% of answers")
    assert bare.series[0].points == (percent(2, 47), percent(21, 23)) and bare.series[0].notes[1] == "33% of answers"
    assert chart.title == (
        "Fewer harmful answers after fine-tuning, mostly because answers got shorter: "
        "31% reached the 256-token limit before, 11% after (served template)"
    )


def lengths(tmp_path, settings, name, before, after, *, answers=True):
    labels, responses = {"keep": tmp_path / name}, {"keep": tmp_path / f"{name}-answers"}
    tagged = {"harmful": lay(labels["keep"], prompts("en", [HATE], 20, before, after), responses=responses["keep"])}
    return build_breakdown(labels, tagged, [], settings, responses=responses if answers else None)


def test_the_length_title_says_what_the_moves_between_lengths_show(tmp_path, settings):
    # five harmful answers fewer, none of them an answer that stopped reaching the limit
    elsewhere = lengths(tmp_path, settings, "a", {"harmful": 6, "refused*": 2}, {"harmful": 1, "refused*": 3})
    assert data.truncation(elsewhere).title == (
        "Fewer harmful answers after fine-tuning, and not mostly through shorter answers: "
        "10% reached the 256-token limit before, 15% after (served template)"
    )
    more = lengths(tmp_path, settings, "b", {"harmful*": 1}, {"harmful": 3})
    assert data.truncation(more).title == (
        "No fewer harmful answers after fine-tuning: 5% reached the 256-token limit before, 0% after (served template)"
    )
    # nothing to draw without the stored answers
    unread = lengths(tmp_path, settings, "c", {"harmful*": 1}, {"harmful": 3}, answers=False)
    assert data.truncation(unread).panels == ()


def test_the_variants_figure_sets_the_rewritten_prompts_beside_the_same_prompts_as_written():
    rows = [
        variant_row("baseline", "hi-Latn", 0.10, 0.30, 100, ci=(0.10, 0.30)),
        variant_row("baseline", "all-Latn", 0.10, 0.25, ci=(0.09, 0.21)),
        variant_row("baseline", "all-mixed", 0.08, 0.10, 500, ci=(-0.01, 0.05)),
        variant_row("baseline", "all-Latn", 0.30, 0.20, metric="chrf", ci=(-0.12, -0.08)),
        variant_row("drifted", "all-Latn", 0.06, 0.10, ci=(0.0, 0.08)),
        variant_row("drifted", "all-mixed", None, None, 0),
        variant_row("baseline", "all-Latn", 0.02, 0.50, condition="patch", ci=(0.4, 0.5)),
    ]
    found = data.variants(stored_variants(*rows, conditions={"keep": "served template", "patch": "safety patch"}))

    chart = data.script_variants(found)

    base, tuned = chart.panels
    assert (base.title, tuned.title) == ("Base model", "Fine-tuned")
    # the pooled harmful rows of the first condition only
    assert base.labels == ("Latin script", "Mixed with English") and tuned.labels == ("Latin script",)
    written, rewritten = base.series
    assert (written.name, rewritten.name) == ("as written", "rewritten")
    assert written.points == (percent(30, 300), percent(40, 500))
    assert rewritten.points == (percent(75, 300), percent(50, 500))
    assert rewritten.notes == ("+15.0 points", "+2.0 points") and written.notes == ()
    assert chart.title == (
        "Rewritten prompts (Latin script) draw more harmful answers: 25.0% against 10.0% as written "
        "(base model, served template)"
    )
    # a rise whose interval reaches zero is not named
    unclear = data.script_variants(data.variants(stored_variants(rows[4])))
    assert unclear.title == "Rewriting the prompts does not clearly draw more harmful answers (served template)"
    # a condition without pooled rows gives way to the next that has them
    later = data.script_variants(data.variants(stored_variants(rows[0], rows[6], conditions=found.conditions)))
    assert later.title.endswith("answers: 50.0% against 2.0% as written (base model, safety patch)")
    for nothing in (None, data.variants(stored_variants()), data.variants(stored_variants(rows[0], rows[3]))):
        assert data.script_variants(nothing).panels == ()


def test_the_variants_figure_sets_refusals_beside_the_harmful_answers():
    harmful = [
        variant_row("baseline", "all-Latn", 0.10, 0.12, ci=(-0.03, 0.07)),
        # a change that rounds to zero
        variant_row("baseline", "all-mixed", 0.09, 0.09, 500, ci=(-0.02, 0.02)) | {"delta": -0.0004},
    ]
    refused = [
        variant_row("baseline", "hi-Latn", 0.70, 0.20, 100, metric="refusal", ci=(-0.6, -0.4)),
        variant_row("baseline", "all-Latn", 0.70, 0.25, metric="refusal", ci=(-0.51, -0.39)),
        variant_row("baseline", "all-mixed", 0.72, 0.70, 500, metric="refusal", ci=(-0.05, 0.01)),
    ]
    harmless = variant_row("baseline", "all-Latn", 0.20, 0.63, metric="harmless_answer", ci=(0.36, 0.48))

    # the order of the rows in the file does not order the panels, and harmless answers are not drawn
    chart = data.script_variants(data.variants(stored_variants(harmless, *harmful, *refused)))

    assert [(panel.title, panel.unit) for panel in chart.panels] == [
        ("Refusals", "Refusals (% of scored responses)"),
        ("Harmful answers", "Harmful answers (% of scored responses)"),
    ]
    assert {panel.labels for panel in chart.panels} == {("Latin script", "Mixed with English")}
    # one column: its panels stand in one row
    assert chart.across == 0
    refusals, answers = chart.panels
    written, rewritten = refusals.series
    assert (written.name, rewritten.name) == ("as written", "rewritten")
    assert written.points == (percent(210, 300), percent(360, 500))
    assert rewritten.points == (percent(75, 300), percent(350, 500))
    assert rewritten.notes == ("-45.0 points", "-2.0 points") and written.notes == ()
    assert answers.series[1].notes == ("+2.0 points", "0.0 points")
    # the pooled change in refusals whose interval leaves zero, with the harmful answers to the same prompts
    assert chart.title == (
        "Served template: to prompts in Latin script the base model refuses far less (70% → 25%) "
        "and answers harmfully no more often (10.0% → 12.0%)"
    )

    def title(*rows):
        return data.script_variants(data.variants(stored_variants(*rows))).title

    # a smaller fall, and the harmful answers of the same rewrite beside it
    mixed = variant_row("baseline", "all-mixed", 0.72, 0.60, 500, metric="refusal", ci=(-0.16, -0.08))
    assert title(*harmful, mixed) == (
        "Served template: to prompts mixed with English the base model refuses less (72% → 60%) "
        "and answers harmfully no more often (9.0% → 9.0%)"
    )
    # more refusals, and harmful answers whose own interval leaves zero
    more = variant_row("baseline", "all-Latn", 0.50, 0.60, metric="refusal", ci=(0.04, 0.16))
    risen = variant_row("baseline", "all-Latn", 0.10, 0.25, ci=(0.09, 0.21))
    fallen = variant_row("baseline", "all-Latn", 0.25, 0.10, ci=(-0.21, -0.09))
    assert title(risen, more) == (
        "Served template: to prompts in Latin script the base model refuses more (50% → 60%) "
        "and answers harmfully more often (10.0% → 25.0%)"
    )
    assert title(fallen, more).endswith("refuses more (50% → 60%) and answers harmfully less often (25.0% → 10.0%)")
    # no harmful answers pooled for that rewrite
    assert title(harmful[1], refused[1]) == (
        "Served template: to prompts in Latin script the base model refuses far less (70% → 25%)"
    )

    # two columns: each panel names its column, and each column takes a row
    tuned = [
        variant_row("drifted", "all-Latn", 0.06, 0.07, ci=(-0.02, 0.04)),
        variant_row("drifted", "all-Latn", 0.80, 0.30, metric="refusal", ci=(-0.56, -0.44)),
    ]
    both = data.script_variants(data.variants(stored_variants(*harmful, *refused, *tuned)))
    assert [panel.title for panel in both.panels] == [
        "Base model: refusals",
        "Base model: harmful answers",
        "Fine-tuned: refusals",
        "Fine-tuned: harmful answers",
    ]
    assert both.across == 2
    # of two clear changes the larger is named, with its column
    assert both.title == (
        "Served template: to prompts in Latin script the fine-tuned model refuses far less (80% → 30%) "
        "and answers harmfully no more often (6.0% → 7.0%)"
    )
    # no change in refusals clear of zero: the harmful answers speak, as they do without refusal rows
    assert title(risen, refused[2]) == (
        "Rewritten prompts (Latin script) draw more harmful answers: 25.0% against 10.0% as written "
        "(base model, served template)"
    )
    assert title(*harmful, refused[2]) == (
        "Rewriting the prompts does not clearly change refusals or draw more harmful answers (served template)"
    )
    # refusals alone draw nothing: the figure stands on the harmful rows
    assert data.script_variants(data.variants(stored_variants(*refused))).panels == ()


def test_a_stored_variants_file_is_read_for_its_names_and_rows():
    row = variant_row("baseline", "all-Latn", 0.10, 0.25, ci=(0.09, 0.21))

    found = data.variants(stored_variants(row | {"added_later": 1}))

    assert found.conditions == {"keep": "served template"}
    assert found.columns == {"baseline": "base model", "drifted": "fine-tuned"}
    (read,) = found.rows
    assert (read.code, read.language, read.rewrite, read.ci, read.delta) == (
        "all-Latn",
        None,
        "romanized",
        (0.09, 0.21),
        0.15,
    )
    empty = data.variants(stored_variants(variant_row("baseline", "hi-mixed", None, None, 0))).rows[0]
    assert (empty.language, empty.rewrite, empty.ci, empty.delta) == ("hi", "codemixed", None, None)
    whole = stored_variants(row)
    for torn in (
        {key: value for key, value in whole.items() if key != "columns"},
        whole | {"rows": [1]},
        whole | {"rows": [{key: value for key, value in row.items() if key != "rewrite"}]},
        # a rate that is not there is not read as a missing rate
        whole | {"rows": [{key: value for key, value in row.items() if key != "parent"}]},
        whole | {"rows": [row | {"n_pairs": "many"}]},
        whole | {"conditions": 3},
        {"baseline": [row]},
        [row],
    ):
        with pytest.raises((TypeError, ValueError, LookupError, AttributeError)):
            data.variants(torn)


def test_the_patch_figure_sets_each_model_with_the_patch_beside_itself_without(tmp_path, both, settings):
    labels = {"keep": tmp_path / "plain", "patch": tmp_path / "patched"}
    tagged = lay(labels["keep"], prompts("en", [HATE], 20, {"harmful": 6}, {"harmful": 2}))
    lay(labels["patch"], prompts("en", [HATE], 20, {"harmful": 2}, {"harmful": 1}))
    found = build_breakdown(labels, {"harmful": tagged}, [], settings)
    assert found.conditions == {"keep": "served template", "patch": "safety patch"}

    chart = data.safety_patch(found, {"keep": both["keep"]}, settings)

    (harmful,) = chart.panels
    assert (harmful.title, harmful.labels) == ("Harmful answers", ("Base model", "Fine-tuned"))
    assert [series.name for series in harmful.series] == ["served template", "safety patch"]
    assert harmful.series[0].points == (percent(6, 20), percent(2, 20))
    assert harmful.series[1].points == (percent(2, 20), percent(1, 20))
    assert chart.title == (
        "With the safety patch, harmful answers go from 30.0% to 10.0% for the base model "
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


def test_the_length_title_speaks_for_the_first_condition_with_two_columns(tmp_path, settings):
    plan = prompts("en", [HATE], 20, {"harmful*": 6, "refused*": 2}, {"harmful": 1})
    labels = {"keep": tmp_path / "keep", "strip": tmp_path / "strip"}
    responses = {"keep": tmp_path / "keep-answers", "strip": tmp_path / "strip-answers"}
    tagged = {"harmful": lay(labels["keep"], plan, columns=("baseline",), responses=responses["keep"])}
    lay(labels["strip"], plan, responses=responses["strip"])

    chart = data.truncation(build_breakdown(labels, tagged, [], settings, responses=responses))

    assert [[series.name for series in panel.series] for panel in chart.panels] == [
        ["base model"],
        ["base model", "fine-tuned"],
    ]
    assert chart.title == (
        "Fewer harmful answers after fine-tuning, mostly because answers got shorter: "
        "40% reached the 256-token limit before, 0% after (no safety preamble)"
    )
