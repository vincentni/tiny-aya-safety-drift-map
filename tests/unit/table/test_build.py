from datetime import UTC, datetime
from pathlib import Path

import pytest

from driftmap.core import ids
from driftmap.core.records import AlphaScreen, Bars, Dare, LabelRow, Ref
from driftmap.core.schemas import SCHEMA_ALPHA_SCREEN, DriftmapError, ExitCode
from driftmap.table.build import COLUMNS, NOT_COMPARABLE, Q4_FINE_TUNED, build, column_of, q4_name
from driftmap.table.render import render
from tests.fakes.labels import (
    CHECKPOINTS,
    column_rows,
    judge_config,
    label,
    make_settings,
    run_record,
    targets,
    write_labels,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "labels"
STEERED = CHECKPOINTS["drifted"] + "--steer-l24-s1.0"


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def screen(status="failed"):
    return AlphaScreen(
        schema=SCHEMA_ALPHA_SCREEN,
        alphas=[],
        chosen=0.25 if status == "chosen" else None,
        recover_status=status,
        badge="Faithful",
        safetune_version="0.1.10",
        dare=Dare(),
        judge=judge_config(),
        bars=Bars(),
        baseline_safe_pooled=0.9,
        drifted_chrf_subset=0.4,
        tied_weight_owner="embed_tokens",
    )


def test_a_checkpoint_maps_to_its_column_by_repo_name(settings):
    assert [column_of(CHECKPOINTS[column], settings) for column in COLUMNS] == list(COLUMNS)
    assert column_of(CHECKPOINTS["repaired_q4"] + "#model-q4_k_m.gguf", settings) == "repaired_q4"
    assert column_of("tiny-aya-global-drift-merged--0123456789ab", settings) == "drifted"
    assert column_of("vincentnkq/tiny-aya-global-drift-lora@" + "a" * 40, settings) == "drifted"
    assert column_of(STEERED, settings) == "repaired"
    assert column_of("someone/other-model@" + "a" * 40, settings) is None
    assert column_of("tiny-aya-global-drift-merged--0123456789ab--steer-l24-s1.0", settings) == "repaired"
    for text in ("tiny-aya-global", "tiny-aya-global--0123456789ab--steer-x", "CohereLabs/tiny-aya-global", "a b@c"):
        assert column_of(text, settings) is None


def test_a_smoke_table_places_the_smoke_drift_repos_and_a_full_one_refuses_them(tmp_path, settings):
    merged = CHECKPOINTS["drifted"].replace("@", "-smoke@")
    adapter = "vincentnkq/tiny-aya-global-drift-lora-smoke@" + "a" * 40
    assert [column_of(checkpoint, settings, "smoke") for checkpoint in (merged, adapter)] == ["drifted", "drifted"]
    assert [column_of(checkpoint, settings, "full") for checkpoint in (merged, adapter)] == [None, None]
    # a reader of built tables takes either mode's names
    assert (column_of(merged, settings), column_of(CHECKPOINTS["drifted"], settings, "smoke")) == ("drifted", "drifted")

    rows = column_rows("baseline") + column_rows("drifted", checkpoint=merged)
    labels = write_labels(tmp_path / "labels", rows)
    drifted = run_record("generate", "drifted", mode="smoke", notes=[f"{NOT_COMPARABLE}seed differs from baseline"])
    drifted.inputs[0].repo += "-smoke"
    tables = build(labels, targets(), [], settings, mode="smoke")
    assert tables.columns["drifted"].state == "tabled"
    assert {(delta.a, delta.b) for delta in tables.deltas if delta.a != delta.b} == {("baseline", "drifted")}
    assert "| all | safe | +0.0 points" in render(tables, settings)
    # the smoke generation run, reading the smoke repo, speaks for the column
    noted = build(labels, targets(), [drifted], settings, mode="smoke").columns["drifted"]
    assert (noted.state, noted.reason) == ("not_comparable", "seed differs from baseline")
    with pytest.raises(DriftmapError, match="unknown checkpoint .*-smoke"):
        build(labels, targets(), [drifted], settings)


def test_the_repaired_model_of_a_fine_tune_is_the_repaired_column_of_its_own_tables(tmp_path, settings):
    usual, other = CHECKPOINTS["repaired"], CHECKPOINTS["repaired"].replace("@", "-nopre@")
    assert column_of(other, settings) == column_of(other, settings, "full", "aggressive") == "repaired"
    assert column_of(other, settings, "full", "standard") is None
    assert column_of(usual, settings, "full", "standard") == "repaired"
    # the tables of a fine-tune take the repair of that fine-tune alone
    assert column_of(usual, settings, "full", "aggressive") is None
    assert column_of(usual.replace("@", "-smoke@"), settings, "smoke", "aggressive") is None
    assert column_of(usual.replace("@", "-smoke@"), settings, "smoke", "standard") == "repaired"
    assert column_of(other.replace("@", "-smoke@"), settings, "smoke", "aggressive") == "repaired"
    assert column_of(other.replace("@", "-smoke@"), settings, "full", "aggressive") is None
    tuned = CHECKPOINTS["drifted"].replace("@", "-nopre@")
    rows = column_rows("baseline") + column_rows("drifted", unsafe=2, checkpoint=tuned)
    labels = write_labels(tmp_path / "labels", rows + column_rows("repaired", unsafe=1, checkpoint=other))

    tables = build(labels, targets(), [], settings, preamble="strip", fine_tune="aggressive")

    assert [tables.columns[column].state for column in ("baseline", "drifted", "repaired")] == ["tabled"] * 3
    assert ("drifted", "repaired") in {(delta.a, delta.b) for delta in tables.deltas}
    assert {row.checkpoint for row in tables.safety if "repaired" in row.checkpoint} == {other}


def test_a_smoke_table_places_the_repos_of_a_smoke_repair_and_its_quantized_file(tmp_path, settings):
    repaired = CHECKPOINTS["repaired"].replace("@", "-smoke@")
    gguf = CHECKPOINTS["repaired_q4"].replace("@", "-smoke@") + "#model-q4_k_m.gguf"
    assert [column_of(checkpoint, settings, "smoke") for checkpoint in (repaired, gguf)] == ["repaired", "repaired_q4"]
    assert [column_of(checkpoint, settings, "full") for checkpoint in (repaired, gguf)] == [None, None]
    assert column_of(gguf, settings) == "repaired_q4"

    rows = column_rows("baseline") + column_rows("repaired", checkpoint=repaired)
    labels = write_labels(tmp_path / "labels", rows + column_rows("repaired_q4", checkpoint=gguf))
    tables = build(labels, targets(), [], settings, mode="smoke")
    states = {column: (state.state, state.kind) for column, state in tables.columns.items()}
    assert states["repaired"] == ("tabled", "weights") and states["repaired_q4"] == ("tabled", "q4_k_m")

    # the smoke quant-eval run, reading the smoke file and the checkpoint it came from, speaks for the column
    failed = run_record("quant_eval", "repaired_q4", mode="smoke", status="failed", notes=["unscored above 5 %"])
    failed.inputs[0].repo += "-smoke"
    failed.inputs.append(Ref(kind="model", repo=repaired.partition("@")[0], revision="b" * 40))
    closed = build(labels, targets(), [failed], settings, mode="smoke").columns["repaired_q4"]
    assert (closed.state, closed.reason) == ("failed", "unscored above 5 %")
    with pytest.raises(DriftmapError, match="unknown checkpoint .*-smoke"):
        build(labels, targets(), [], settings)


def test_only_the_repaired_column_may_hold_a_steered_and_a_weights_checkpoint(tmp_path, settings):
    other = CHECKPOINTS["baseline"] + "--steer-l24-s1.0"
    steered = [row.model_copy(update={"checkpoint": other}) for row in column_rows("baseline")]
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + steered)
    with pytest.raises(DriftmapError, match="baseline has rows from 2 checkpoints"):
        build(labels, targets(), [], settings)


def test_the_drifted_hash_is_compared_with_the_latest_finished_drift_run_of_the_mode(tmp_path, settings):
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("drifted"))
    later = datetime(2026, 11, 14, tzinfo=UTC)
    drift = run_record("drift", merged_sha256="a" * 64, started=datetime(2026, 11, 12, tzinfo=UTC))
    generated = run_record("generate", "drifted", merged_sha256="a" * 64)
    others = [
        run_record("drift", merged_sha256="b" * 64, started=later, mode="smoke"),
        run_record("drift", merged_sha256="c" * 64, started=later, status="failed"),
    ]

    assert build(labels, targets(), [drift, generated, *others], settings).columns["drifted"].state == "tabled"
    newer = run_record("drift", merged_sha256="d" * 64, started=later)
    state = build(labels, targets(), [drift, generated, *others, newer], settings).columns["drifted"]
    assert (state.state, state.reason) == (
        "not_comparable",
        "merged weights hash differs between the drift and generate records",
    )


def test_columns_without_labels_say_why(tmp_path, settings):
    labels = write_labels(tmp_path / "labels", column_rows("baseline"))
    benign = [label("drifted", "en", i, set_="benign", chrf=0.5) for i in range(4)]
    write_labels(labels, benign)

    tables = build(labels, targets(), [], settings, not_run={"repaired": "left for the next budget window"})
    states = {column: (state.state, state.reason) for column, state in tables.columns.items()}
    assert states == {
        "baseline": ("tabled", None),
        "drifted": ("generated", "generated, not judged yet"),
        "repaired": ("not_run", "left for the next budget window"),
        "repaired_q4": ("not_run", "no generation run recorded"),
    }
    assert {row.checkpoint for row in tables.safety + tables.chrf} == {CHECKPOINTS["baseline"]}
    assert tables.deltas == [delta for delta in tables.deltas if delta.a == delta.b == "baseline"]
    assert tables.flags == {}
    assert (tables.seeds.seed, tables.seeds.bootstrap) == (settings.seed, 42)

    recorded = build(labels, targets(), [run_record("generate", "repaired")], settings)
    assert recorded.columns["repaired"].state == "generated"


def test_a_failed_quant_eval_or_repair_closes_the_q4_column(tmp_path, settings):
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("repaired_q4"))
    failed = run_record("quant_eval", "repaired_q4", status="failed", notes=["unscored above 5 %"])

    tables = build(labels, targets(), [failed], settings)
    state = tables.columns["repaired_q4"]
    assert (state.state, state.reason, state.kind) == ("failed", "unscored above 5 %", "q4_k_m")
    assert {row.checkpoint for row in tables.safety} == {CHECKPOINTS["baseline"]}

    tables = build(labels, targets(), [screen()], settings)
    assert tables.columns["repaired_q4"].state == "not_applicable"
    assert "repair" in tables.columns["repaired_q4"].reason


def test_a_q4_column_quantized_from_another_repaired_revision_is_not_comparable(tmp_path, settings):
    rows = column_rows("baseline") + column_rows("repaired") + column_rows("repaired_q4")
    labels = write_labels(tmp_path / "labels", rows)
    repaired, revision = CHECKPOINTS["repaired"].split("@")

    def quant_eval(source, **changes):
        run = run_record("quant_eval", "repaired_q4", **changes)
        # the job records the file it served, then the checkpoint that file was converted from
        run.inputs.append(Ref(kind="model", repo=repaired, revision=source))
        return run

    same = build(labels, targets(), [quant_eval(revision)], settings).columns["repaired_q4"]
    assert (same.state, same.reason) == ("tabled", None)

    earlier = "9" * 40
    other = build(labels, targets(), [quant_eval(earlier)], settings).columns["repaired_q4"]
    named = f"quantized from {repaired}@{earlier}, the repaired column is {CHECKPOINTS['repaired']}"
    assert (other.state, other.reason) == ("not_comparable", named)
    # the latest ok run of the mode speaks, as for every other difference
    later = datetime(2026, 11, 14, tzinfo=UTC)
    reruns = [quant_eval(earlier), quant_eval(revision, run_id="quant_eval-again", started=later)]
    assert build(labels, targets(), reruns, settings).columns["repaired_q4"].state == "tabled"
    failed = [quant_eval(revision), quant_eval(earlier, run_id="quant_eval-again", started=later, mode="smoke")]
    assert build(labels, targets(), failed, settings).columns["repaired_q4"].state == "tabled"

    # with a steered repaired column the file is the quantization of no tabled checkpoint
    steered = [row.model_copy(update={"checkpoint": STEERED}) for row in column_rows("drifted")]
    steered_labels = write_labels(tmp_path / "steered", column_rows("baseline") + steered + column_rows("repaired_q4"))
    state = build(steered_labels, targets(), [quant_eval(revision)], settings).columns["repaired_q4"]
    assert state.state == "not_comparable" and state.reason.endswith(f"the repaired column is {STEERED}")
    # without a repaired column there is nothing to hold the file against
    alone = write_labels(tmp_path / "alone", column_rows("baseline") + column_rows("repaired_q4"))
    assert build(alone, targets(), [quant_eval(earlier)], settings).columns["repaired_q4"].state == "tabled"


def test_a_q4_column_made_from_the_fine_tuned_model_is_held_against_the_drifted_column(tmp_path, settings):
    gguf = f"vincentnkq/tiny-aya-global-drift-GGUF@{'c' * 40}#model-q4_k_m.gguf"
    rehearsed = gguf.replace("-GGUF@", "-GGUF-smoke@")
    assert column_of(gguf, settings) == column_of(gguf, settings, "full") == "repaired_q4"
    assert (column_of(rehearsed, settings, "smoke"), column_of(rehearsed, settings, "full")) == ("repaired_q4", None)
    # what a reader is shown for such a file, here and wherever a column of it is named
    assert q4_name(gguf, settings) == q4_name(rehearsed, settings) == Q4_FINE_TUNED == "fine-tuned model, q4_k_m"
    assert q4_name(CHECKPOINTS["repaired_q4"], settings) is None and q4_name(CHECKPOINTS["drifted"], settings) is None
    rows = column_rows("baseline") + column_rows("drifted") + column_rows("repaired_q4", checkpoint=gguf)
    labels = write_labels(tmp_path / "labels", rows)
    merged, revision = CHECKPOINTS["drifted"].split("@")

    def quant_eval(source):
        run = run_record("quant_eval", "repaired_q4")
        run.inputs[0] = Ref(kind="model", repo=gguf.partition("@")[0], revision="c" * 40)
        # the job records the file it served, then the checkpoint that file was converted from
        run.inputs.append(Ref(kind="model", repo=merged, revision=source))
        return run

    tables = build(labels, targets(), [quant_eval(revision)], settings)

    state = tables.columns["repaired_q4"]
    assert (state.state, state.kind, state.reason) == ("tabled", Q4_FINE_TUNED, None)
    assert tables.columns["repaired"].state == "not_run"
    assert {(delta.a, delta.b) for delta in tables.deltas if delta.a != delta.b} == {
        ("baseline", "drifted"),
        ("drifted", "repaired_q4"),
    }
    markdown = render(tables, settings)
    assert "- repaired_q4 (fine-tuned model, q4_k_m): tabled" in markdown and "| drifted to repaired_q4 |" in markdown
    # a repair that failed its bars leaves nothing of the repaired model to quantize; this file is not of it
    failed = build(labels, targets(), [quant_eval(revision), screen("failed")], settings).columns["repaired_q4"]
    assert (failed.state, failed.kind) == ("tabled", Q4_FINE_TUNED)

    earlier = "9" * 40
    other = build(labels, targets(), [quant_eval(earlier)], settings).columns["repaired_q4"]
    named = f"quantized from {merged}@{earlier}, the drifted column is {CHECKPOINTS['drifted']}"
    assert (other.state, other.kind, other.reason) == ("not_comparable", Q4_FINE_TUNED, named)
    # without the run's record the file is held against no column; its repo still says what it was made from
    bare = build(labels, targets(), [], settings).columns["repaired_q4"]
    assert (bare.state, bare.kind) == ("tabled", Q4_FINE_TUNED)
    # a file of the repaired model, and a column with no file, keep the plain name
    plain = column_rows("baseline") + column_rows("repaired") + column_rows("repaired_q4")
    of_repaired = build(write_labels(tmp_path / "repaired", plain), targets(), [], settings)
    assert of_repaired.columns["repaired_q4"].kind == "q4_k_m"
    absent = write_labels(tmp_path / "absent", column_rows("baseline") + column_rows("drifted"))
    assert build(absent, targets(), [], settings).columns["repaired_q4"].kind == "q4_k_m"


def test_a_steered_checkpoint_fills_the_repaired_column_as_inference_time(tmp_path, settings):
    steered = [row.model_copy(update={"checkpoint": STEERED}) for row in column_rows("drifted")]
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + steered)

    tables = build(labels, targets(), [], settings)

    assert (tables.columns["repaired"].state, tables.columns["repaired"].kind) == ("tabled", "inference-time")
    assert (tables.columns["drifted"].state, tables.columns["baseline"].kind) == ("not_run", "weights")


def test_a_delta_skips_a_column_that_is_not_comparable(tmp_path, settings):
    rows = column_rows("baseline", unsafe=1) + column_rows("drifted", rendered_prefix_sha256="e" * 64)
    tables = build(write_labels(tmp_path / "labels", rows + column_rows("repaired")), targets(), [], settings)

    assert {(delta.a, delta.b) for delta in tables.deltas if delta.a != delta.b} == {("baseline", "repaired")}
    markdown = render(tables, settings)
    assert (
        "| row | metric | baseline to drifted | drifted to repaired | repaired to repaired_q4 | baseline to repaired |"
        in markdown
    )
    assert "| all | safe | n/a | n/a | n/a | +25.0 points, safer (95% CI " in markdown


def test_unknown_or_doubled_checkpoints_are_refused(tmp_path, settings):
    rows = column_rows("baseline")
    stray = rows[0].model_copy(update={"checkpoint": "someone/other-model@" + "a" * 40})
    with pytest.raises(DriftmapError, match="someone/other-model") as raised:
        build(write_labels(tmp_path / "stray", [*rows, stray]), targets(), [], settings)
    assert raised.value.exit_code == ExitCode.INTEGRITY

    with pytest.raises(DriftmapError, match="no judged labels"):
        build(tmp_path / "empty", targets(), [], settings)

    second = rows[1].model_copy(update={"checkpoint": CHECKPOINTS["baseline"].replace("1" * 40, "9" * 40)})
    with pytest.raises(DriftmapError, match="baseline has rows from 2 checkpoints"):
        build(write_labels(tmp_path / "two", [rows[0], second, *rows[2:]]), targets(), [], settings)


def test_the_committed_label_files_build_an_english_table(settings):
    rows = [
        LabelRow.model_validate_json(line)
        for path in sorted(FIXTURES.glob("*/en.jsonl"))
        for line in path.read_text().splitlines()
    ]
    prompts = {
        name: {"en": sorted({row.prompt_id for row in rows if row.set == name})} for name in ("harmful", "xstest")
    }

    tables = build(FIXTURES, prompts, [], settings)

    assert [tables.columns[column].state for column in COLUMNS] == ["tabled", "tabled", "not_run", "not_run"]
    assert tables.columns["drifted"].reason == "unscored: en 1"
    safe = [(row.k, row.n) for row in tables.safety if row.row == "en" and row.metric == "safe"]
    assert safe == [(11, 12), (7, 11)]
    assert [(row.k, row.n) for row in tables.over_refusal] == [(1, 6), (0, 6)]
    assert all("placeholder" in row.prompt for row in rows)


def test_only_the_latest_ok_run_in_the_build_mode_decides_comparability(tmp_path, settings):
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("drifted"))
    noted = run_record("generate", "drifted", notes=[f"{NOT_COMPARABLE}seed 7 differs from baseline"])
    later = datetime(2026, 11, 14, tzinfo=UTC)

    for changes in ({"status": "failed"}, {"status": "partial"}, {"status": "oom"}, {"mode": "smoke"}):
        retry = run_record("generate", "drifted", started=later, **changes)
        state = build(labels, targets(), [noted, retry], settings).columns["drifted"]
        assert (state.state, state.reason) == ("not_comparable", "seed 7 differs from baseline"), changes
    rerun = run_record("generate", "drifted", started=later)
    assert build(labels, targets(), [rerun, noted], settings).columns["drifted"].state == "tabled"


def test_a_steered_column_reads_steer_records_not_weight_repair_ones(tmp_path, settings):
    steered = [row.model_copy(update={"checkpoint": STEERED}) for row in column_rows("drifted")]
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + steered)
    stale = run_record("generate", "repaired", notes=[f"{NOT_COMPARABLE}stale weight repair"])
    steer = run_record("steer", "drifted", notes=[f"{NOT_COMPARABLE}decoding differs from baseline"])

    assert build(labels, targets(), [stale], settings).columns["repaired"].state == "tabled"
    state = build(labels, targets(), [stale, steer], settings).columns["repaired"]
    assert (state.state, state.reason) == ("not_comparable", "decoding differs from baseline")
    assert build(labels, targets(), [steer], settings).columns["drifted"].state == "not_run"


def test_two_repair_arms_share_the_one_repaired_column(tmp_path, settings):
    steered = [row.model_copy(update={"checkpoint": STEERED}) for row in column_rows("drifted")]
    rows = column_rows("baseline") + column_rows("repaired", unsafe=1) + steered
    labels = write_labels(tmp_path / "labels", rows)

    def repaired(records):
        tables = build(labels, targets(), records, settings)
        state = tables.columns["repaired"]
        pooled = [row for row in tables.safety if row.row == "all" and row.metric == "safe"][1]
        assert list(tables.columns) == list(COLUMNS) and f"- repaired ({state.kind}): tabled: {state.reason}" in render(
            tables, settings
        )
        return state.kind, state.reason, pooled.checkpoint, pooled.k

    weights, steer = CHECKPOINTS["repaired"], STEERED
    assert repaired([screen("chosen")]) == ("weights", f"{steer} has labels but is not tabled", weights, 18)
    for records in ([], [screen("failed")]):
        assert repaired(records) == ("inference-time", f"{weights} has labels but is not tabled", steer, 24)


def test_label_lines_end_at_a_newline_only_and_a_bad_line_is_named(tmp_path, settings):
    rows = column_rows("baseline")
    odd = rows[2].model_copy(update={"prompt": "placeholder\u2028with\u2029odd\u0085separators"})
    labels = write_labels(tmp_path / "labels", [*rows[:2], odd, *rows[3:]])
    assert "\u2028" in (labels / "tiny-aya-global" / "en.jsonl").read_text(encoding="utf-8")

    assert build(labels, targets(), [], settings).columns["baseline"].state == "tabled"

    path = labels / "tiny-aya-global" / "fr.jsonl"
    lines = path.read_text(encoding="utf-8").split("\n")
    path.write_text("\n".join([*lines[:2], '{"schema": "driftmap.label/1"}', *lines[2:]]), encoding="utf-8")
    with pytest.raises(DriftmapError, match=r"tiny-aya-global/fr\.jsonl:3: ") as raised:
        build(labels, targets(), [], settings)
    assert raised.value.exit_code == ExitCode.INTEGRITY and "placeholder" not in str(raised.value)


def test_the_right_number_of_rows_for_other_prompts_is_refused(tmp_path, settings):
    rows = column_rows("baseline")
    renamed = rows[0].model_copy(update={"prompt_id": "harmful-en-999"})
    with pytest.raises(DriftmapError, match="baseline harmful/en: 4 rows for 4 prompts, for other prompts"):
        build(write_labels(tmp_path / "labels", [renamed, *rows[1:]]), targets(), [], settings)


def test_chrf_means_and_the_gap_use_their_own_scores(tmp_path, settings):
    rows = column_rows("baseline", chrf=0.4)
    rows = [
        row.model_copy(update={"chrf": None}) if (row.set, row.prompt_id) == ("benign", "benign-en-000") else row
        for row in rows
    ]
    # hi and ar answer safely without refusing, so the two metrics part ways
    rows = [
        row.model_copy(update={"refusal": "no"}) if row.set == "harmful" and row.language in ("hi", "ar") else row
        for row in rows
    ]
    tables = build(write_labels(tmp_path / "labels", rows), targets(), [], settings)

    means = {row.row: (row.n, row.chrf) for row in tables.chrf}
    assert means["en"] == (3, pytest.approx(0.4)) and means["all"] == (23, pytest.approx(0.4))
    assert tables.columns["baseline"].reason == "unscored: benign/en 1"
    (gap,) = [delta for delta in tables.deltas if delta.a == delta.b]
    assert (gap.metric, gap.delta) == ("safe", 0.0)


NOPRE = CHECKPOINTS["drifted"].replace("@", "-nopre@")


def test_the_cells_carry_the_condition_the_table_was_built_for(tmp_path, settings):
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("drifted"))
    served = build(labels, targets(), [], settings)
    stripped = build(labels, targets(), [], settings, preamble="strip")
    assert {row.preamble for row in served.safety + served.over_refusal} == {"keep"}
    assert {row.preamble for row in stripped.safety + stripped.over_refusal} == {"strip"}
    assert stripped.model_dump(exclude={"safety", "over_refusal"}) == served.model_dump(
        exclude={"safety", "over_refusal"}
    )


def test_only_generation_runs_of_the_tables_condition_speak_for_a_column(tmp_path, settings):
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("drifted"))
    later = datetime(2026, 11, 14, tzinfo=UTC)
    differs = [NOT_COMPARABLE + "seed differs from baseline"]
    served = run_record("generate", "drifted", notes=differs, started=later)
    stripped = run_record("generate", "drifted", run_id="strip", notes=["preamble: strip"])

    assert build(labels, targets(), [served, stripped], settings, preamble="strip").columns["drifted"].state == "tabled"
    state = build(labels, targets(), [served, stripped], settings).columns["drifted"]
    assert (state.state, state.reason) == ("not_comparable", "seed differs from baseline")
    marked = stripped.model_copy(update={"notes": ["preamble: strip", *differs], "started": later})
    assert build(labels, targets(), [marked], settings).columns["drifted"].state == "tabled"
    assert build(labels, targets(), [marked], settings, preamble="strip").columns["drifted"].state == "not_comparable"


def test_a_checkpoint_trained_on_a_named_set_fills_the_drifted_column_of_its_own_tables(tmp_path, settings):
    defended = CHECKPOINTS["drifted"].replace("@", "-defense@")
    assert column_of(defended, settings) == column_of(defended, settings, "full", "defense") == "drifted"
    assert [column_of(defended, settings, fine_tune=other) for other in ("standard", "aggressive")] == [None, None]
    assert [column_of(CHECKPOINTS[name], settings, fine_tune="defense") for name in ("baseline", "drifted")] == [
        "baseline",
        None,
    ]
    assert column_of(defended.replace("@", "-smoke@"), settings, "smoke", "defense") == "drifted"
    assert column_of(defended.replace("@", "-smoke@"), settings, "full", "defense") is None
    labels = write_labels(tmp_path / "labels", column_rows("baseline") + column_rows("drifted", checkpoint=defended))
    later = datetime(2026, 11, 14, tzinfo=UTC)
    repo, _, revision = defended.partition("@")
    generated = run_record("generate", "drifted", merged_sha256="b" * 64)
    generated.inputs[0] = Ref(kind="model", repo=repo, revision=revision)
    named = [ids.train_set_note("defense", "c" * 40)]
    trained = run_record("drift", merged_sha256="b" * 64, notes=named)
    # a later drift run on the usual rows, under the same template, pushed other weights to another repo
    usual = run_record("drift", merged_sha256="a" * 64, started=later)

    tables = build(labels, targets(), [trained, usual, generated], settings, fine_tune="defense")
    assert tables.columns["drifted"].state == "tabled"
    assert {row.checkpoint for row in tables.safety} == {CHECKPOINTS["baseline"], defended}
    other = trained.model_copy(update={"merged_sha256": "c" * 64, "started": later})
    state = build(labels, targets(), [trained, usual, generated, other], settings, fine_tune="defense")
    assert state.columns["drifted"].reason == "merged weights hash differs between the drift and generate records"

    # the standard tables refuse its labels, and a later run on the named set does not speak for their column
    with pytest.raises(DriftmapError, match="label rows name an unknown checkpoint"):
        build(labels, targets(), [trained, usual, generated], settings)
    kept = write_labels(tmp_path / "kept", column_rows("baseline") + column_rows("drifted"))
    read = run_record("generate", "drifted", merged_sha256="a" * 64)
    served = usual.model_copy(update={"started": trained.started})
    assert build(kept, targets(), [served, read, other], settings).columns["drifted"].state == "tabled"


def test_a_checkpoint_trained_without_the_preamble_fills_the_drifted_column_of_its_own_tables(tmp_path, settings):
    assert column_of(NOPRE, settings) == column_of(NOPRE, settings, "full") == "drifted"
    # a table is of one fine-tune: there the other one's checkpoint has no column
    assert column_of(NOPRE, settings, fine_tune="aggressive") == "drifted"
    assert column_of(NOPRE, settings, fine_tune="standard") is None
    assert column_of(CHECKPOINTS["drifted"], settings, fine_tune="aggressive") is None
    assert column_of(CHECKPOINTS["drifted"], settings, fine_tune="standard") == "drifted"
    assert column_of(CHECKPOINTS["baseline"], settings, fine_tune="aggressive") == "baseline"
    assert column_of(NOPRE.replace("-nopre@", "-nopre-smoke@"), settings, "smoke") == "drifted"
    assert column_of(NOPRE.replace("-nopre@", "-nopre-smoke@"), settings, "full") is None
    rows = column_rows("baseline") + column_rows("drifted", checkpoint=NOPRE)
    labels = write_labels(tmp_path / "labels", rows)
    later = datetime(2026, 11, 14, tzinfo=UTC)
    repo, _, revision = NOPRE.partition("@")
    generated = run_record("generate", "drifted", merged_sha256="b" * 64, notes=["preamble: strip"])
    generated.inputs[0] = Ref(kind="model", repo=repo, revision=revision)
    trained = run_record("drift", merged_sha256="b" * 64, notes=["preamble: strip"])
    # a later drift run under the served template pushed other weights to another repo
    served = run_record("drift", merged_sha256="a" * 64, started=later)

    apart = {"preamble": "strip", "fine_tune": "aggressive"}
    tables = build(labels, targets(), [trained, served, generated], settings, **apart)
    assert tables.columns["drifted"].state == "tabled"
    assert {row.checkpoint for row in tables.safety} == {CHECKPOINTS["baseline"], NOPRE}
    other = trained.model_copy(update={"merged_sha256": "c" * 64, "started": later})
    state = build(labels, targets(), [trained, served, generated, other], settings, **apart).columns["drifted"]
    assert state.reason == "merged weights hash differs between the drift and generate records"

    # the standard tables refuse its labels, and its tables refuse the standard fine-tune's
    with pytest.raises(DriftmapError, match="label rows name an unknown checkpoint") as mixed:
        build(labels, targets(), [trained, served, generated], settings, preamble="strip")
    assert mixed.value.exit_code is ExitCode.INTEGRITY
    usual = write_labels(tmp_path / "usual", column_rows("baseline") + column_rows("drifted"))
    with pytest.raises(DriftmapError, match="label rows name an unknown checkpoint"):
        build(usual, targets(), [], settings, **apart)
    # a later run over the other fine-tune under the same condition does not speak for the standard column
    standard = run_record("generate", "drifted", run_id="standard", merged_sha256="a" * 64, notes=["preamble: strip"])
    newer = generated.model_copy(update={"run_id": "newer", "started": later})
    records = [served.model_copy(update={"started": standard.started}), standard, newer]
    assert build(usual, targets(), records, settings, preamble="strip").columns["drifted"].state == "tabled"

    # the served model's column is not judged by a run trained without the preamble
    kept = write_labels(tmp_path / "kept", column_rows("baseline") + column_rows("drifted"))
    read = run_record("generate", "drifted", merged_sha256="a" * 64)
    drifted = build(kept, targets(), [served, read, other], settings).columns["drifted"]
    assert drifted.state == "tabled"
