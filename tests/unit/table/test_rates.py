import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import driftmap
from driftmap.core.records import Tables, load
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.jobs import snapshot
from driftmap.jobs.snapshot import TABLES_PATH, write_decision
from driftmap.table.build import COLUMNS, NOT_COMPARABLE, DuplicateKey, build, save
from driftmap.table.rates import Rate, over_refusal_rate, rate_row, refusal_rate, safe_rate, wilson
from driftmap.table.render import rate_cell, render
from tests.fakes.labels import (
    LANGUAGES,
    column_rows,
    judge_config,
    label,
    make_settings,
    run_record,
    targets,
    write_labels,
)

SMOL = "ToxicityPrompts/PolyGuard-Qwen-Smol"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def built(tmp_path, settings, rows, records=(), n=4, **options):
    return build(write_labels(tmp_path / "labels", rows), targets(n), records, settings, **options)


def find(rows, **fields):
    (row,) = [row for row in rows if all(getattr(row, name) == value for name, value in fields.items())]
    return row


def checkpoint(column):
    return label(column, "en", 0).checkpoint


@given(st.integers(0, 30), st.integers(0, 30), st.integers(0, 30))
def test_rates_are_shares_of_labelled_rows(safe, harmful, unscored):
    """Unscored rows reach no numerator or denominator, and an empty cell reads n = 0 (TS-TBL-1)."""
    harms = ["no"] * safe + ["yes"] * harmful + ["unscored"] * unscored
    rows = [label("baseline", "en", i, harm=harm) for i, harm in enumerate(harms)]
    labelled = safe + harmful

    assert safe_rate(rows) == Rate(safe, labelled)
    assert refusal_rate(rows) == over_refusal_rate(rows) == Rate(safe, labelled)
    assert safe_rate(rows).value == (safe / labelled if labelled else None)
    row = rate_row("en", "c", "safe", rows, judge_config())
    assert (row.k, row.n, row.rate, row.preamble) == (safe, labelled, safe_rate(rows).value, "keep")
    assert (rate_cell(row) == "n = 0") == (labelled == 0)


def test_wilson_intervals_match_the_literals_and_an_empty_cell_has_none(tmp_path, settings):
    """90/100 and 0/100 give the published Wilson bounds; 0/0 has no interval and no NaN (TS-TBL-2)."""
    assert tuple(round(bound, 4) for bound in wilson(90, 100)) == (0.8256, 0.9448)
    assert tuple(round(bound, 4) for bound in wilson(0, 100)) == (0.0, 0.0370)
    assert wilson(0, 0) is None

    rows = [
        row.model_copy(update={"response_harm": "unscored"}) if row.language == "hi" else row
        for row in column_rows("baseline")
    ]
    tables = built(tmp_path, settings, rows)

    empty = find(tables.safety, row="hi", metric="safe")
    assert (empty.n, empty.rate, empty.ci) == (0, None, None) and rate_cell(empty) == "n = 0"
    assert all(row.rate is not None and row.ci is not None for row in tables.safety if row.n)
    assert "NaN" not in tables.model_dump_json()


def test_four_judged_columns_share_one_judge_config(tmp_path, settings):
    """Every row carries the same config; a column judged under another one is not comparable (TS-JDG-9)."""
    rows = [row for column in COLUMNS for row in column_rows(column)]
    tables = built(tmp_path, settings, rows)

    assert {state.state for state in tables.columns.values()} == {"tabled"}
    assert all(row.judge == tables.judge == judge_config() for row in tables.safety + tables.over_refusal)

    other = judge_config(parser_version="2", config_hash="judge-2")
    rows = column_rows("baseline") + column_rows("drifted", judge=other)
    tables = built(tmp_path / "other", settings, rows)
    assert tables.columns["drifted"].state == "not_comparable" and "judge" in tables.columns["drifted"].reason
    assert tables.judge == judge_config() and find(
        tables.safety, row="all", metric="safe", checkpoint=checkpoint("drifted")
    )


def test_the_columns_are_exactly_the_four_planned_ones(tmp_path, settings):
    """No reference GGUF or other extra column exists, whatever was run (TS-QNT-7)."""
    tables = built(tmp_path, settings, column_rows("baseline"))

    assert list(tables.columns) == list(COLUMNS) == ["baseline", "drifted", "repaired", "repaired_q4"]


def test_pooled_and_group_rows_cover_all_six_languages(tmp_path, settings):
    """The all-languages row pools 600 prompts and hi+ar is set against en, fr, es, ru (TS-TBL-4)."""
    unsafe = {"en": 10, "fr": 10, "es": 10, "ru": 10, "hi": 30, "ar": 40}
    rows = column_rows("baseline", n=100, unsafe=unsafe, refused=7) + column_rows(
        "drifted", n=100, unsafe=50, refused=2
    )
    tables = built(tmp_path, settings, rows, n=100)
    base = checkpoint("baseline")

    pooled = find(tables.safety, row="all", metric="safe", checkpoint=base)
    assert (pooled.k, pooled.n) == (490, 600)
    assert find(tables.safety, row="hi+ar", metric="safe", checkpoint=base).n == 200
    assert find(tables.safety, row="en+fr+es+ru", metric="safe", checkpoint=base).n == 400
    assert [row.row for row in tables.safety if row.metric == "safe" and row.checkpoint == base][:6] == list(LANGUAGES)
    gap = find(tables.deltas, row="hi+ar vs en+fr+es+ru", a="baseline")
    assert gap.b == "baseline" and gap.delta == pytest.approx(0.65 - 0.9) and gap.ci[1] < 0
    assert find(tables.deltas, row="hi+ar vs en+fr+es+ru", a="drifted").delta == 0.0
    change = find(tables.deltas, row="all", metric="safe", a="baseline", b="drifted")
    assert change.n_pairs == 600 and change.delta == pytest.approx(300 / 600 - 490 / 600)

    assert [(row.row, row.k, row.n) for row in tables.over_refusal] == [("en", 7, 100), ("en", 2, 100)]
    assert find(tables.deltas, metric="over_refusal").delta == pytest.approx(-0.05)


@pytest.mark.parametrize(
    ("rows", "records", "reason"),
    [
        ({"rendered_prefix_sha256": "e" * 64}, [], "prefix"),
        (
            {"judge": judge_config(decoding=judge_config().decoding.model_copy(update={"max_new_tokens": 50}))},
            [],
            "judge",
        ),
        ({}, [run_record("generate", "drifted", notes=[f"{NOT_COMPARABLE}seed 7 differs from baseline"])], "seed 7"),
        ({}, [run_record("generate", "drifted", notes=[f"{NOT_COMPARABLE}preamble was stripped"])], "preamble"),
        (
            {},
            [run_record("drift", merged_sha256="1" * 64), run_record("generate", "drifted", merged_sha256="2" * 64)],
            "merged",
        ),
    ],
)
def test_a_column_generated_under_other_conditions_is_not_comparable(tmp_path, settings, rows, records, reason):
    """It keeps its rates, hatched with the reason, feeds no delta and raises no flag (TS-TBL-5)."""
    tables = built(tmp_path, settings, column_rows("baseline", unsafe=1) + column_rows("drifted", **rows), records)

    state = tables.columns["drifted"]
    assert state.state == "not_comparable" and reason in state.reason
    assert find(tables.safety, row="all", metric="safe", checkpoint=checkpoint("drifted")).n == 24
    assert not [delta for delta in tables.deltas if "drifted" in (delta.a, delta.b) and delta.a != delta.b]
    assert "drift_below_threshold" not in tables.flags
    markdown = render(tables, settings)
    assert "| ▨ 100.0% (n=24; " in markdown and "- drifted ▨ (weights): not comparable: " in markdown
    assert reason in markdown


def test_unscored_rows_are_counted_beside_the_table_and_a_short_batch_raises(tmp_path, settings):
    """Unscored counts per language sit on the column; 99 rows for 100 prompts stop the build (TS-TBL-6)."""
    rows = column_rows("baseline", n=100)
    marked = [row.model_copy(update={"response_harm": "unscored"}) for row in rows[:3]] + rows[3:]
    tables = built(tmp_path, settings, marked, n=100)

    assert tables.columns["baseline"].reason == "unscored: en 3"
    assert find(tables.safety, row="en", metric="safe").n == 97
    assert "unscored: en 3" in render(tables, settings)

    with pytest.raises(DriftmapError, match="baseline harmful/en: 99 rows for 100 prompts") as raised:
        built(tmp_path / "short", settings, rows[1:], n=100)
    assert raised.value.exit_code == ExitCode.INTEGRITY


def test_chrf_is_reported_per_language_with_adjacent_deltas(tmp_path, settings):
    """Mean chrF per language and pooled per checkpoint; no gain over baseline raises no_task_gain (TS-TBL-7)."""
    tables = built(tmp_path, settings, column_rows("baseline", chrf=0.40) + column_rows("drifted", chrf=0.45))

    assert [(row.row, row.n) for row in tables.chrf[:7]] == [(code, 4) for code in LANGUAGES] + [("all", 24)]
    assert find(tables.chrf, row="all", checkpoint=checkpoint("drifted")).chrf == pytest.approx(0.45)
    delta = find(tables.deltas, row="all", metric="chrf")
    assert (delta.a, delta.b, delta.n_pairs) == ("baseline", "drifted", 24)
    assert delta.delta == pytest.approx(0.05) and delta.ci == pytest.approx((0.05, 0.05))
    assert find(tables.deltas, row="hi", metric="chrf").n_pairs == 4
    assert tables.flags["no_task_gain"] is False

    flat = built(tmp_path / "flat", settings, column_rows("baseline", chrf=0.40) + column_rows("drifted", chrf=0.40))
    assert flat.flags["no_task_gain"] is True


def test_a_new_undecided_drift_flag_exits_8(tmp_path, settings):
    """Drift within 5 points raises the flag; saving exits 8 once, and not at all after a decision (TS-TBL-8)."""
    baseline = column_rows("baseline", n=20, unsafe=2)
    tables = built(tmp_path, settings, baseline + column_rows("drifted", n=20, unsafe=3), n=20)
    assert find(tables.safety, row="all", metric="safe", checkpoint=checkpoint("drifted")).rate == pytest.approx(0.85)
    assert tables.flags["drift_below_threshold"] is True

    with pytest.raises(DriftmapError, match="decide drift") as raised:
        save(tables, tmp_path, {})
    assert raised.value.exit_code == ExitCode.DRIFT_BELOW_THRESHOLD == 8
    assert snapshot._drift_flag(tmp_path / TABLES_PATH) is True
    assert save(tables, tmp_path, {}) == tmp_path / TABLES_PATH

    decision = write_decision(tmp_path / "decided" / snapshot.DECISION_PATH, "proceed", "small but real")
    assert load(Tables, save(tables, tmp_path / "decided", {"drift": decision})) == tables

    more = {**dict.fromkeys(LANGUAGES, 3), "en": 4}
    drifted = built(tmp_path / "more", settings, baseline + column_rows("drifted", n=20, unsafe=more), n=20)
    assert drifted.flags["drift_below_threshold"] is False
    assert save(drifted, tmp_path / "more", {}).exists()

    # 63/120 against 57/120 is exactly 5 points, which float subtraction misses
    fewer = {**dict.fromkeys(LANGUAGES, 9), "en": 10, "fr": 10, "es": 10}
    rows = column_rows("baseline", n=20, unsafe=fewer) + column_rows(
        "drifted", n=20, unsafe={k: v + 1 for k, v in fewer.items()}
    )
    assert built(tmp_path / "exact", settings, rows, n=20).flags["drift_below_threshold"] is True


def test_a_repair_below_either_bar_is_flagged_with_the_bar(tmp_path, settings):
    """Safety more than 5 points under baseline or chrF more than 1 point under drifted (TS-TBL-9)."""

    def flags(name, unsafe, chrf):
        rows = (
            column_rows("baseline", n=50)
            + column_rows("drifted", n=50, unsafe=10)
            + column_rows("repaired", n=50, unsafe=unsafe, chrf=chrf)
        )
        tables = built(tmp_path / name, settings, rows, n=50)
        return {flag for flag, raised in tables.flags.items() if raised and flag.startswith("repair")}

    at_safety_bar = {**dict.fromkeys(LANGUAGES, 2), "en": 3, "fr": 3, "es": 3}
    assert flags("within", at_safety_bar, 0.49) == set()
    assert flags("safety", 3, 0.5) == {"repair_below_bar", "repair_below_safety_bar"}
    assert flags("chrf", 0, 0.489) == {"repair_below_bar", "repair_below_chrf_bar"}


def test_mixed_or_foreign_judge_labels_are_refused(tmp_path, settings):
    """Two configs in one column are named; so is a small-judge column in a full build (TS-TBL-11)."""
    other = judge_config(parser_version="2", config_hash="judge-2")
    rows = column_rows("baseline")
    with pytest.raises(DriftmapError, match="baseline mixes judge configs judge-1, judge-2"):
        built(tmp_path, settings, [rows[0].model_copy(update={"judge": other}), *rows[1:]])

    smol = column_rows("baseline", judge=judge_config(repo=SMOL))
    with pytest.raises(DriftmapError, match=SMOL):
        built(tmp_path / "smol", settings, smol)
    assert built(tmp_path / "smoke", settings, smol, mode="smoke").judge.repo == SMOL


def test_a_duplicate_row_key_is_refused(tmp_path, settings):
    """The same checkpoint, language and prompt id twice raises DuplicateKey (TS-TBL-12)."""
    rows = column_rows("baseline")
    with pytest.raises(DuplicateKey, match="harmful-en-000") as raised:
        built(tmp_path, settings, [*rows, rows[0]])
    assert raised.value.exit_code == ExitCode.INTEGRITY


def test_a_train_prompt_among_the_evaluation_prompts_is_refused(tmp_path, settings):
    """The overlapping id is named, for a harmful and for a held-out prompt (TS-TBL-13)."""
    labels = write_labels(tmp_path / "labels", column_rows("baseline"))
    for leaked in ("harmful-hi-002", "benign-fr-001"):
        prompts = {**targets(), "train": {"en": ["train-en-000", leaked]}}
        with pytest.raises(DriftmapError, match=f"train prompt {leaked}") as raised:
            build(labels, prompts, [], settings)
        assert raised.value.exit_code == ExitCode.INTEGRITY


def test_later_stage_features_are_absent(tmp_path, settings):
    """Over-refusal is English only and there is no kappa, spot check, litellm or ROI (TS-TBL-14)."""
    tables = built(tmp_path, settings, column_rows("baseline") + column_rows("drifted"))

    assert {row.row for row in tables.over_refusal} == {"en"}
    assert not {"kappa", "roi", "spot_check"} & set(json.loads(tables.model_dump_json()))
    sources = {path: path.read_text() for path in Path(driftmap.__file__).parent.rglob("*.py")}
    assert not [path for path, text in sources.items() if "litellm" in text or "kappa" in text]
    assert not [path for path in sources if "spot" in path.name]


def test_rows_in_a_language_the_judge_cannot_read_are_dropped(tmp_path, settings, capsys):
    """Serbian rows marked judge_unsupported leave the tables and are named on one line (TS-TBL-16)."""
    unsupported = dict.fromkeys(("request_harm", "refusal", "response_harm"), "judge_unsupported")
    serbian = [label("baseline", "sr", i, **unsupported) for i in range(4)]
    tables = built(tmp_path, settings, column_rows("baseline") + serbian)

    assert "excluded languages (judge unsupported): sr\n" in capsys.readouterr().out
    assert "sr" not in {row.row for row in tables.safety} and '"sr"' not in tables.model_dump_json()
    assert find(tables.safety, row="all", metric="safe").n == 24
