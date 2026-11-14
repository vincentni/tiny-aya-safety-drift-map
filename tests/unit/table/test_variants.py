import pytest

from driftmap.core.records import Ref
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.table import variants
from tests.fakes.labels import CHECKPOINTS, REFS, judge_config, label, make_settings, run_record

REVISION = "c" * 40


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def links(code, count):
    parent = code.partition("-")[0]
    return [
        {
            "language": code,
            "prompt_id": f"variant-{code}-{index:03d}",
            "parent_language": parent,
            "parent_prompt_id": f"harmful-{parent}-{index:03d}",
        }
        for index in range(count)
    ]


def side(column, code, parents_harmful, variants_harmful, changed=None, count=10):
    """One checkpoint's labels of ``count`` parents and their variants; the first of each side are harmful."""
    parent = code.partition("-")[0]
    parents = [label(column, parent, index, harm="yes" if index < parents_harmful else "no") for index in range(count)]
    rewritten = [
        label(
            column,
            code,
            index,
            harm="yes" if index < variants_harmful else "no",
            **{"prompt_id": f"variant-{code}-{index:03d}", **(changed or {}).get(index, {})},
        )
        for index in range(count)
    ]
    return variants.Side(variants.Source(CHECKPOINTS[column], REVISION), parents, rewritten, links(code, count))


def test_each_column_compares_its_variants_with_their_parents_and_counts_the_rewrites_left_out(settings):
    sides = {
        "keep": {
            # the judge read one rewrite as another request than its parent: that pair is left out
            "drifted": side("drifted", "hi-Latn", 1, 1, {9: {"request_harm": "no"}}),
            "baseline": side("baseline", "hi-Latn", 2, 5),
        },
        "patch": {"baseline": side("baseline", "es-mixed", 0, 0)},
    }

    found = variants.build(sides, settings)

    assert found.schema_ == "driftmap.variants/1"
    assert found.conditions == {"keep": "served template", "patch": "safety patch"}
    assert found.columns == {"baseline": "baseline", "drifted": "fine-tuned"}
    assert [(source.condition, source.column, source.judged, source.left_out) for source in found.sources] == [
        ("keep", "baseline", 10, 0),
        ("keep", "drifted", 10, 1),
        ("patch", "baseline", 10, 0),
    ]
    assert found.sources[1].checkpoint == CHECKPOINTS["drifted"] and found.sources[1].variants_revision == REVISION
    rows = {(row.condition, row.column, row.code): row for row in found.rows}
    assert list(rows) == [
        ("keep", "baseline", "hi-Latn"),
        ("keep", "baseline", "all-Latn"),
        ("keep", "drifted", "hi-Latn"),
        ("keep", "drifted", "all-Latn"),
        ("patch", "baseline", "es-mixed"),
        ("patch", "baseline", "all-mixed"),
    ]
    base = rows["keep", "baseline", "hi-Latn"]
    assert (base.set, base.metric, base.n_pairs) == ("harmful", "harmful", 10)
    assert (base.language, base.rewrite) == ("hi", "romanized")
    assert (base.parent, base.variant, base.delta) == pytest.approx((0.2, 0.5, 0.3))
    assert base.ci[0] <= 0.3 <= base.ci[1] and 0 < base.p_value <= 1
    pooled = rows["patch", "baseline", "all-mixed"]
    assert (pooled.language, pooled.rewrite, pooled.n_pairs) == (None, "codemixed", 10)
    tuned = rows["keep", "drifted", "hi-Latn"]
    assert (tuned.n_pairs, tuned.parent, tuned.variant, tuned.delta) == pytest.approx((9, 1 / 9, 1 / 9, 0.0))
    assert found.judge == judge_config() and (found.seeds.seed, found.seeds.bootstrap) == (settings.seed, 42)
    assert variants.VariantTables.model_validate_json(found.model_dump_json(by_alias=True)) == found

    shown = variants.render(found)
    assert shown.startswith("# Script variants against their parent prompts\n\n")
    assert [line for line in shown.splitlines() if line.startswith("#")][1:] == [
        "## Served template",
        "### Baseline",
        "### Fine-tuned",
        "## Safety patch",
        "### Baseline",
    ]
    assert f"Checkpoint `{CHECKPOINTS['drifted']}`, variant prompts at `cccccccccccc`: " in shown
    assert "10 variant labels, 1 left out." in shown
    assert "| harmful | hi-Latn | 20.0% | 50.0% | +30.0 points (95% CI " in shown
    assert "| harmful | hi-Latn | 11.1% | 11.1% | +0.0 points (95% CI +0.0 to +0.0; p = 1.000; 9 pairs) |" in shown


def test_labels_that_cannot_be_compared_are_refused(settings):
    bare = variants.Side(variants.Source(CHECKPOINTS["baseline"], REVISION), [], [], [])
    with pytest.raises(DriftmapError, match="no judged labels of script variants") as none:
        variants.build({"keep": {"baseline": bare}}, settings)
    assert none.value.exit_code == ExitCode.USAGE
    mixed = side("baseline", "hi-Latn", 2, 5, {0: {"judge": judge_config(config_hash="judge-2")}})
    with pytest.raises(DriftmapError, match="baseline mixes judge configs judge-1, judge-2") as two:
        variants.build({"keep": {"baseline": mixed}}, settings)
    assert two.value.exit_code == ExitCode.INTEGRITY
    second = judge_config(config_hash="judge-2")
    other = side("drifted", "hi-Latn", 1, 1, {index: {"judge": second} for index in range(10)})
    for row in other.parents:
        row.judge = second
    with pytest.raises(DriftmapError, match="judged under different configs") as columns:
        variants.build({"keep": {"baseline": side("baseline", "hi-Latn", 2, 5), "drifted": other}}, settings)
    assert columns.value.exit_code == ExitCode.INTEGRITY
    with pytest.raises(LookupError):
        variants.link({"language": "hi-Latn", "prompt_id": "variant-1", "parent_language": "hi"})


def test_the_latest_ok_variants_run_of_the_mode_and_condition_names_what_each_column_read(settings):
    def run(column, run_id, notes=("sets: variants",), revision=None, commit=REVISION, **changes):
        repo, pinned = REFS[column]
        inputs = [
            Ref(kind="model", repo=repo, revision=revision or pinned),
            Ref(kind="dataset", repo="owner/eval", revision="6" * 40),
            Ref(kind="dataset", repo="owner/eval", revision=commit),
        ]
        kind = changes.pop("kind", "generate")
        return run_record(kind, column, run_id=run_id, inputs=inputs, notes=list(notes), **changes)

    records = [
        run("drifted", "generate-full-1", revision="9" * 40, commit="d" * 40),
        run("drifted", "generate-full-2"),
        run("baseline", "generate-full-3"),
        # none of these is a finished run over the variants of this mode under the served template
        run("drifted", "generate-full-4", notes=(), revision="8" * 40),
        run("drifted", "generate-full-5", status="failed", revision="8" * 40),
        run("drifted", "generate-smoke-6", mode="smoke", revision="8" * 40),
        run("drifted", "generate-full-7", notes=("preamble: patch", "sets: variants"), revision="7" * 40),
        run("drifted", "quant_eval-full-8", kind="quant_eval", revision="8" * 40),
    ]

    found = variants.sources(records, settings, "full")

    assert found == {
        "baseline": variants.Source(CHECKPOINTS["baseline"], REVISION),
        "drifted": variants.Source(CHECKPOINTS["drifted"], REVISION),
    }
    assert list(found) == ["baseline", "drifted"]
    patched = variants.sources(records, settings, "full", "patch")
    assert list(patched) == ["drifted"] and patched["drifted"].checkpoint.endswith("7" * 40)
    rehearsed = variants.sources(records, settings, "smoke")
    assert list(rehearsed) == ["drifted"] and rehearsed["drifted"].checkpoint.endswith("8" * 40)
    assert variants.sources(records[:3], settings, "smoke") == {}
