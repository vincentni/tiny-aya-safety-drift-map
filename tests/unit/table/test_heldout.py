"""The held-out table: the base model as served against the same model under the safety patch."""

import pytest

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.table import heldout
from tests.fakes.labels import CHECKPOINTS, judge_config, label, make_settings, prompt_ids

REVISION = "c" * 40
CATEGORIES = ("Hate Speech", "Self-Harm")


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def rows(harmful, n=4, **changes):
    """``n`` label rows per language; ``harmful`` gives per language how many of the first answers are harmful."""
    return [
        label("baseline", language, index, harm="yes" if index < count else "no", **changes)
        for language, count in harmful.items()
        for index in range(n)
    ]


def categories(languages=("en", "hi"), n=4):
    """Every prompt is of the first category; the even ones are of the second too."""
    return {
        prompt_id: CATEGORIES[: 1 + (index % 2 == 0)]
        for language in languages
        for index, prompt_id in enumerate(prompt_ids("harmful", language, n))
    }


def build(settings, served, patched, **changed):
    given = {"checkpoint": CHECKPOINTS["baseline"], "prompts_revision": REVISION} | changed
    return heldout.build({"keep": served, "patch": patched}, categories(), settings, **given)


def test_rates_per_language_and_pooled_with_the_paired_change_under_the_patch(settings):
    served = rows({"en": 2, "hi": 3})
    patched = rows({"en": 1, "hi": 1})
    patched[-1] = patched[-1].model_copy(update={"response_harm": "unscored", "refusal": "unscored"})

    tables = build(settings, served, patched)

    assert tables.scopes == ["en", "hi", "all"] and tables.prompts == 8
    assert tables.conditions == {"keep": "served template", "patch": "safety patch"}
    assert tables.unscored == {"keep": 0, "patch": 1}
    assert (tables.checkpoint, tables.prompts_revision) == (CHECKPOINTS["baseline"], REVISION)
    assert tables.judge == judge_config() and tables.seeds.seed == settings.seed
    counted = {(row.condition, row.scope, row.metric): (row.k, row.n) for row in tables.rates}
    assert counted["keep", "all", "harmful"] == (5, 8) and counted["patch", "all", "harmful"] == (2, 7)
    assert counted["keep", "hi", "harmful"] == (3, 4) and counted["patch", "hi", "harmful"] == (1, 3)
    # the fake labels refuse whatever is not harmful
    assert counted["keep", "all", "refusal"] == (3, 8) and counted["patch", "en", "refusal"] == (3, 4)
    rate = next(row for row in tables.rates if (row.condition, row.scope, row.metric) == ("keep", "all", "harmful"))
    assert rate.rate == 5 / 8 and rate.ci[0] < 5 / 8 < rate.ci[1]
    moved = {(row.scope, row.metric): row for row in tables.changes}
    # over the seven prompts scored under both: five harmful as served, two under the patch
    assert (moved["all", "harmful"].n_pairs, moved["all", "harmful"].delta) == (7, pytest.approx(-3 / 7))
    assert moved["all", "harmful"].ci[1] < 0 and 0 < moved["all", "harmful"].p_value <= 0.25
    assert moved["all", "refusal"].delta == pytest.approx(3 / 7) and moved["en", "harmful"].delta == -0.25
    by_category = {(row.condition, row.harm_category): (row.k, row.n) for row in tables.categories}
    assert by_category["keep", "Hate Speech"] == (5, 8) and by_category["patch", "Hate Speech"] == (2, 7)
    # the even prompts of each language: en 0 and hi 0, 2 are harmful as served
    assert by_category["keep", "Self-Harm"] == (3, 4) and by_category["patch", "Self-Harm"] == (2, 4)


def test_the_markdown_gives_both_rates_and_the_change_per_language(settings):
    tables = build(settings, rows({"en": 2, "hi": 3}), rows({"en": 1, "hi": 1}))

    shown = heldout.render(tables)

    lines = shown.splitlines()
    assert lines[0] == "# Held-out harmful prompts: the base model as served and under the safety patch"
    assert "8 harmful prompts of the red-teaming dataset that are no evaluation prompt and near none" in lines[2]
    assert f"read at `{REVISION[:12]}`" in lines[2] and "under `rubric-v1`" in lines[2]
    assert "| language | served template | safety patch | change |" in lines
    pooled = [line for line in lines if line.startswith("| all |")]
    assert pooled[0].startswith("| all | 62.5% (5/8; 95% CI ") and "| 25.0% (2/8; 95% CI " in pooled[0]
    assert "| -37.5 points (95% CI " in pooled[0] and pooled[0].endswith("; 8 pairs) |")
    assert [line for line in lines if line.startswith("## ")] == [
        "## Harmful answers",
        "## Refusals",
        "## Harmful answers by harm category, all languages",
    ]
    assert "| Self-Harm | 75.0% (3/4; 95% CI " in shown
    # every table and heading stands between blank lines
    for at, line in enumerate(lines):
        if line.startswith("## ") or (line.startswith("|") and not lines[at - 1].startswith("|")):
            assert lines[at - 1] == "", line


def test_labels_that_do_not_cover_the_prompts_or_come_from_two_judges_are_refused(settings):
    served, patched = rows({"en": 2, "hi": 3}), rows({"en": 1, "hi": 1})

    with pytest.raises(DriftmapError) as short:
        build(settings, served, patched[:-1])
    assert short.value.exit_code == ExitCode.INTEGRITY
    assert "safety patch: 1 of the 8 held-out prompts have no label, 0 labels are of another prompt" in str(short.value)
    stray = label("baseline", "en", 9)
    with pytest.raises(DriftmapError, match="served template: 0 of the 8 held-out prompts have no label, 1 labels"):
        build(settings, [*served, stray], patched)
    other = [row.model_copy(update={"judge": judge_config(config_hash="judge-2")}) for row in patched]
    with pytest.raises(DriftmapError, match="judged under different configs"):
        build(settings, served, other)
    with pytest.raises(DriftmapError, match="safety patch: 8 of the 8 held-out prompts have no label"):
        heldout.build({"keep": served}, categories(), settings, checkpoint="x", prompts_revision=REVISION)


def known(harmful, n=6):
    """Label rows of ``n`` evaluation prompts per language, and a row of another set of the same model."""
    found = [row.model_copy(update={"prompt_id": f"evaluation-{row.prompt_id}"}) for row in rows(harmful, n)]
    return [*found, label("baseline", "en", 0, set_="xstest")]


def test_the_evaluation_prompts_stand_beside_with_the_gap_between_the_two_changes(settings):
    served, patched = rows({"en": 2, "hi": 3}), rows({"en": 1, "hi": 1})
    evaluation = {"keep": known({"en": 3, "hi": 3}), "patch": known({"en": 0, "hi": 3})}
    stored = {
        "breakdown.json": {("keep", "all"): (6, 12), ("patch", "all"): (3, 12), ("keep", "en"): (3, 6)},
        "breakdown-other.json": {("keep", "all"): (6, 12)},
    }

    tables = build(settings, served, patched, evaluation=evaluation, stored=stored)

    alone = build(settings, served, patched)
    assert alone.in_sample is None and tables.model_copy(update={"in_sample": None}) == alone
    beside = tables.in_sample
    # the row of another set is no evaluation prompt; a breakdown without both pooled rates is not named
    assert (beside.prompts, beside.unscored, beside.equal_to) == (12, {"keep": 0, "patch": 0}, ["breakdown.json"])
    counted = {(row.condition, row.scope, row.metric): (row.k, row.n) for row in beside.rates}
    assert counted["keep", "all", "harmful"] == (6, 12) and counted["patch", "all", "harmful"] == (3, 12)
    assert counted["keep", "all", "refusal"] == (6, 12) and counted["patch", "en", "harmful"] == (0, 6)
    moved = {(row.scope, row.metric): row for row in beside.changes}
    assert (moved["all", "harmful"].n_pairs, moved["all", "harmful"].delta) == (12, -0.25)
    gaps = {(row.scope, row.metric): row for row in beside.gaps}
    assert list(gaps) == [(scope, metric) for scope in tables.scopes for metric in heldout.METRICS]
    # held out, harmful answers fall by 3 of 8; over the evaluation prompts by 3 of 12
    assert gaps["all", "harmful"].delta == pytest.approx(-0.375 + 0.25)
    assert gaps["all", "harmful"].ci[0] < gaps["all", "harmful"].delta < gaps["all", "harmful"].ci[1]
    assert gaps["all", "refusal"].delta == pytest.approx(0.375 - 0.25)
    assert gaps["en", "harmful"].delta == pytest.approx(-0.25 + 0.5)

    shown = heldout.render(tables)
    lines = shown.splitlines()
    assert [line for line in lines if line.startswith("#") and "evaluation" in line] == [
        "## The same conditions on the evaluation prompts"
    ]
    assert [line for line in lines if line.startswith("### ")] == ["### Harmful answers", "### Refusals"]
    assert "The 12 harmful prompts of the evaluation set under the same two conditions" in shown
    assert "Their counts of harmful answers equal those of `breakdown.json`." in shown
    assert "| language | served template | safety patch | change | held-out change minus this change |" in lines
    # harmful answers, then refusals: half of the twelve as served either way
    harmful, refusals = (line for line in lines if line.startswith("| all | 50.0% (6/12; 95% CI "))
    assert "; 12 pairs) | -12.5 points (95% CI " in harmful and harmful.endswith(") |")
    assert "; 12 pairs) | +12.5 points (95% CI " in refusals
    for at, line in enumerate(lines):
        if line.startswith("#") or (line.startswith("|") and not lines[at - 1].startswith("|")):
            assert at == 0 or lines[at - 1] == "", line
    assert "evaluation prompts" not in heldout.render(alone).split("\n", 3)[3]


def test_evaluation_labels_that_do_not_pair_overlap_or_differ_from_a_stored_table_are_refused(settings):
    served, patched = rows({"en": 2, "hi": 3}), rows({"en": 1, "hi": 1})
    first, second = known({"en": 3, "hi": 3}), known({"en": 0, "hi": 3})

    def refused(evaluation, message, **given):
        with pytest.raises(DriftmapError) as raised:
            build(settings, served, patched, evaluation=evaluation, **given)
        assert raised.value.exit_code == ExitCode.INTEGRITY and message in str(raised.value), str(raised.value)

    unpaired = "the evaluation prompts have labels for 12 under the served template, 11 under the safety patch"
    refused({"keep": first, "patch": second[1:]}, f"{unpaired}, not for the same prompts under both")
    refused({}, "labels for 0 under the served template, 0 under the safety patch")
    shared = {"keep": [*first, served[0]], "patch": [*second, patched[0]]}
    refused(shared, "1 held-out prompts are evaluation prompts")
    other = [row.model_copy(update={"judge": judge_config(config_hash="judge-2")}) for row in second]
    refused({"keep": first, "patch": other}, "judged under different configs")
    differs = {"breakdown.json": {("keep", "all"): (6, 12), ("patch", "hi"): (2, 6)}}
    found = "breakdown.json holds 2/6 harmful answers for the safety patch, hi"
    refused({"keep": first, "patch": second}, f"{found}; the labels of the evaluation prompts give 3/6", stored=differs)


def test_the_counts_of_a_stored_breakdown_are_those_of_the_base_model_under_the_two_conditions():
    from driftmap.core.records import Seeds
    from driftmap.table.breakdown import SCHEMA_BREAKDOWN, Breakdown, BreakdownRate

    def rate(condition, column, by, language, k, n, set_="harmful"):
        where = {"harm_category": None, "language": language, "global_or_local": None, "length": None}
        counts = {"k": k, "n": n, "rate": k / n, "ci": (0.0, 1.0), "above_pooled_baseline": False}
        return BreakdownRate(set=set_, condition=condition, column=column, by=by, **where, **counts)

    rates = [
        rate("keep", "baseline", "all", None, 55, 598),
        rate("keep", "baseline", "language", "en", 12, 100),
        rate("patch", "baseline", "all", None, 29, 599),
        rate("strip", "baseline", "all", None, 61, 599),
        rate("keep", "drifted", "all", None, 39, 598),
        rate("keep", "baseline", "harm_category", None, 9, 84),
        rate("keep", "baseline", "all", None, 1, 2, set_="other"),
    ]
    empty = {"changes": [], "transitions": [], "at_cap": [], "not_tabled": []}
    found = Breakdown(
        schema=SCHEMA_BREAKDOWN,
        fine_tune="standard",
        conditions={},
        columns={},
        min_cell_n=10,
        token_cap=None,
        rates=rates,
        **empty,
        judge=judge_config(),
        seeds=Seeds(seed=1, bootstrap=42),
    )

    assert heldout.stored_counts(found) == {
        ("keep", "all"): (55, 598),
        ("keep", "en"): (12, 100),
        ("patch", "all"): (29, 599),
    }
