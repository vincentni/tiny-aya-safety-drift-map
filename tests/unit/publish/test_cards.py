import pytest
from huggingface_hub.repocard import RepoCard

from driftmap.core.ids import RepoRef
from driftmap.core.records import AlphaRow, ColumnState
from driftmap.core.settings import load_settings
from driftmap.publish import cards
from driftmap.publish.cards import missing, render_cards
from driftmap.publish.q4_local import Check, Count
from driftmap.repair import select
from driftmap.table.breakdown import Breakdown
from driftmap.table.build import Q4_FINE_TUNED, build
from driftmap.table.choice import ChoiceTables
from driftmap.table.variants import VariantTables
from tests.fakes.labels import column_rows, make_settings, targets, write_labels
from tests.fakes.published import (
    ADAPTER,
    ADAPTER_NOPRE,
    ATTRIBUTION,
    ATTRIBUTION_MERGED_WORK,
    ATTRIBUTION_WORK,
    DATASETS,
    DEFENSE_SHA256,
    EVAL,
    GGUF,
    MERGED,
    MERGED_GGUF,
    MERGED_NOPRE,
    MERGED_SHA256,
    MODELS,
    NOPRE_SHA256,
    NOPRE_SLUG,
    REPAIRED,
    REPAIRED_NOPRE,
    RESPONSES,
    RESULTS,
    SLUG,
    choice_manifest,
    choice_tables,
    defense_drift_record,
    drift_record,
    free_form_scores,
    manifest,
    mbbq_tables,
    nopre_drift_record,
    recipe,
    two_column_tables,
)
from tests.fakes.settings import settings_toml

KEYS = dict(zip((*DATASETS, *MODELS), ("eval", "results", "responses", "adapter", "repaired", "gguf"), strict=True))
KEYS |= {ADAPTER_NOPRE: "adapter_nopre", MERGED_GGUF: "merged_gguf", REPAIRED_NOPRE: "repaired_nopre"}
KEYS |= {ATTRIBUTION: "adapter_attribution"}
DERIVED = (*MODELS, ADAPTER_NOPRE, MERGED_GGUF, REPAIRED_NOPRE, ATTRIBUTION)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def body(text):
    return RepoCard(text).text.lstrip()


def test_cards_name_the_base_model_the_preamble_and_every_licence(settings):
    """Each card carries its licence, its sources' licences, the links and its own warning (TS-PUB-4)."""
    rendered = render_cards(settings, manifest=manifest(), records=[drift_record()])

    assert set(rendered) == set(KEYS)
    for repo, text in rendered.items():
        for needle in (
            "CohereLabs/tiny-aya-global",
            "cc-by-nc-4.0",
            "safety preamble was kept",
            "CohereLabs/aya_redteaming (apache-2.0)",
            "CohereLabs/aya_dataset (apache-2.0)",
            "Paul/XSTest (cc-by-4.0)",
            "SafeTune 0.1.10 (LSAL v1.2)",
            "AuditKIT 1.1.3 (LSAL v1.2)",
            "CuratorKIT 0.1.22 (LSAL v1.1)",
            "https://github.com/vincentni/tiny-aya-safety-drift-map",
            "https://github.com/vincentni/tiny-aya-safety-drift-map/blob/main/docs/article.md",
        ):
            assert needle in text, (repo, needle)
        assert missing(KEYS[repo], text, settings, manifest()) == []
    licences = {repo: RepoCard(text).data.to_dict()["license"] for repo, text in rendered.items()}
    assert licences == {
        EVAL: ["apache-2.0", "cc-by-4.0"],
        RESULTS: "cc-by-4.0",
        RESPONSES: "cc-by-nc-4.0",
        **dict.fromkeys(DERIVED, "cc-by-nc-4.0"),
    }
    for repo in DERIVED:
        assert RepoCard(rendered[repo]).data.to_dict()["base_model"] == "CohereLabs/tiny-aya-global"
    for repo in (RESULTS, RESPONSES):
        assert body(rendered[repo]).startswith("**Content warning:**")
    assert "harmful prompts from `CohereLabs/aya_redteaming`" in rendered[EVAL]
    adapter = rendered[ADAPTER]
    assert MERGED_SHA256 in adapter and "merge_and_unload()" in adapter
    for repo in (REPAIRED, GGUF):
        assert "Not certified for deployment." in rendered[repo]


def test_a_custom_write_up_link_replaces_the_default(settings):
    rendered = render_cards(settings, writeup="https://huggingface.co/blog/vincentnkq/drift-map")

    assert all("https://huggingface.co/blog/vincentnkq/drift-map" in text for text in rendered.values())
    assert not any("docs/article.md" in text for text in rendered.values())


def test_missing_names_what_a_card_lacks(settings):
    rendered = render_cards(settings, manifest=manifest())

    assert missing("adapter", None, settings) == ["README.md"]
    permissive = rendered[ADAPTER].replace("license: cc-by-nc-4.0", "license: apache-2.0")
    assert missing("adapter", permissive, settings) == ["license: cc-by-nc-4.0 in the card metadata"]
    bare = "# Results\n\n" + body(rendered[RESULTS])
    assert "the content warning as the first line" in missing("results", bare, settings)
    assert "license: cc-by-4.0 in the card metadata" in missing("results", bare, settings)
    generated = "---\nbase_model: CohereLabs/tiny-aya-global\nlibrary_name: peft\n---\n# Model Card\n"
    assert missing("adapter", generated, settings) == [
        "license: cc-by-nc-4.0 in the card metadata",
        "the deployment notice",
        "the base model's licence",
        "the link to the code",
        "the link to the write-up",
    ]
    assert missing("eval", rendered[EVAL], settings, None) == ["the source licences (no manifest.json)"]


def test_every_planned_column_is_stated_or_marked_not_run(tmp_path, settings):
    """A column without a result reads "not run" with its reason, never a placeholder (TS-NFR-6)."""
    bare = render_cards(settings)[RESULTS]
    assert "No result table is published yet." in bare and "planned" not in bare

    tables = two_column_tables(tmp_path, settings, repaired="left for the next budget window")
    rendered = render_cards(settings, manifest=manifest(), tables=tables, recipes={"adapter": recipe()})

    held = (
        "- `tables/tables.json`: the base model and the light fine-tune under the served template (columns "
        "`baseline`, `drifted`); `repaired`: not run (left for the next budget window)."
    )
    assert held in rendered[RESULTS]
    # a column no recipe planned is not listed at all
    assert "repaired_q4`:" not in rendered[RESULTS] and "planned" not in rendered[RESULTS]
    assert RepoCard(rendered[RESULTS]).data.to_dict()["viewer"] is False
    assert RepoCard(rendered[RESPONSES]).data.to_dict()["viewer"] is False
    assert "viewer" not in RepoCard(rendered[EVAL]).data.to_dict()
    assert "`drifted` is the fine-tuned model" in rendered[RESULTS]
    assert "Residual failures: not run (left for the next budget window)." in rendered[REPAIRED]
    for text in (*rendered.values(), bare):
        assert not any(mark in text for mark in ("TBD", "XX", "{", "None"))


def test_residual_failures_come_from_the_tables(tmp_path, settings):
    tables = two_column_tables(tmp_path, settings)
    rows = [row.model_copy(update={"checkpoint": f"{REPAIRED}@{'b' * 40}"}) for row in tables.safety]
    states = {**tables.columns, "repaired": tables.columns["drifted"]}
    tables = tables.model_copy(update={"safety": tables.safety + rows[:1], "columns": states})

    text = render_cards(settings, tables=tables)[REPAIRED]

    assert "- en: 90.0% (n=20; 95% CI 69.9% to 97.2%; judge=closed-model-a; preamble=keep)" in text


def test_a_failed_quantized_evaluation_stays_on_the_card(tmp_path, settings):
    """The GGUF card keeps the failed column and its reason beside the record (TS-QNT-6)."""
    tables = two_column_tables(tmp_path, settings)
    failed = ColumnState(state="failed", reason="unscored rows above 5% {of 600}", kind="q4_k_m")
    tables = tables.model_copy(update={"columns": {**tables.columns, "repaired_q4": failed}})

    rendered = render_cards(settings, manifest=manifest(), tables=tables)

    assert tables.columns["repaired_q4"].state == "failed"
    assert "Quantized evaluation: failed (unscored rows above 5% {of 600}). The file stays published" in rendered[GGUF]
    assert "`gguf.json`" in rendered[GGUF]
    assert "`drifted`); `repaired_q4`: failed (unscored rows above 5% {of 600})." in rendered[RESULTS]


def test_the_judge_named_is_the_one_the_labels_record(tmp_path, settings):
    tables = two_column_tables(tmp_path, settings)
    other = tables.judge.model_copy(update={"repo": "another-model", "revision": "ab" * 32})

    rendered = render_cards(settings, tables=tables.model_copy(update={"judge": other}))

    line = "- Judge: another-model under `rubric-v1` (rubric text sha256 `abababababab`)"
    assert all(line in text and "closed-model-a" not in text for text in rendered.values())
    configured = "closed-model-a under `rubric-v1`, as configured"
    assert all(configured in text for text in render_cards(settings).values())


def test_the_merged_hash_is_the_latest_full_drift_run(settings):
    early = drift_record().model_copy(update={"merged_sha256": "1" * 64})
    late = early.model_copy(update={"merged_sha256": "2" * 64, "started": early.started.replace(hour=5)})
    smoke = late.model_copy(
        update={"merged_sha256": "3" * 64, "mode": "smoke", "started": late.started.replace(hour=9)}
    )
    judge = late.model_copy(
        update={"merged_sha256": "4" * 64, "kind": "judge", "started": late.started.replace(hour=9)}
    )

    adapter = render_cards(settings, records=[late, smoke, early, judge])[ADAPTER]

    assert "2" * 64 in adapter and not any(digit * 64 in adapter for digit in "134")
    assert "`not recorded`" in render_cards(settings, records=[smoke, judge])[ADAPTER]


def test_model_cards_must_name_the_base_model_and_the_notice(settings):
    rendered = render_cards(settings, manifest=manifest())

    other = rendered[REPAIRED].replace("base_model: CohereLabs/tiny-aya-global", "base_model: other/model")
    assert missing("repaired", other, settings) == ["base_model: CohereLabs/tiny-aya-global in the card metadata"]
    for key, repo in (("repaired", REPAIRED), ("gguf", GGUF)):
        assert missing(key, rendered[repo].replace("Not certified", "Certified"), settings) == [
            "the not-certified notice"
        ]
    partly = manifest().model_copy(update={"licences": {"CohereLabs/aya_redteaming": "apache-2.0", "Paul/XSTest": ""}})
    gap = "the source licences (none in manifest.json for CohereLabs/aya_dataset, Paul/XSTest)"
    assert missing("eval", rendered[EVAL], settings, partly)[0] == gap
    empty = manifest().model_copy(update={"licences": {}})
    assert len(missing("eval", rendered[EVAL], settings, empty)) == 1
    assert "quoted outputs keep the base model's cc-by-nc-4.0 terms" in rendered[RESULTS]


def test_the_space_statement_follows_the_flag(tmp_path, settings):
    flagged = load_settings(settings_toml(tmp_path, publish={"space_not_serving": True}))

    assert cards.NOT_SERVED in render_cards(flagged)[REPAIRED]
    assert cards.NOT_SERVED not in render_cards(settings)[REPAIRED]


def test_dataset_cards_carry_no_model_metadata(settings):
    rendered = render_cards(settings, manifest=manifest())

    for repo in DATASETS:
        data = RepoCard(rendered[repo]).data.to_dict()
        assert "base_model" not in data and data["language"] == ["en", "fr", "es", "ru", "hi", "ar"]


def test_the_cards_state_both_conditions_once_a_full_run_left_the_preamble_out(settings):
    served = render_cards(settings, manifest=manifest(), records=[drift_record()])
    for repo, text in served.items():
        # the second adapter exists only through a run of the second condition
        assert ("safety preamble was kept in every run" in text) == (repo != ADAPTER_NOPRE)
        assert ("nopreamble/" in text) == (repo == ADAPTER_NOPRE)
    # a rehearsal is not a condition of the study
    smoke = nopre_drift_record().model_copy(update={"mode": "smoke"})
    assert render_cards(settings, manifest=manifest(), records=[drift_record(), smoke]) == served

    both = render_cards(settings, manifest=manifest(), records=[drift_record(), nopre_drift_record()])

    for repo, text in both.items():
        assert "safety preamble was kept in the main runs; the runs under `nopreamble/` left it out" in text
        assert "in every run" not in text and missing(KEYS[repo], text, settings, manifest()) == []
    assert "`nopreamble/labels/<checkpoint>/`" in both[EVAL]
    assert "`nopreamble/tables/`" in both[RESULTS] and "`preamble: strip`" in both[RESULTS]
    assert "`nopreamble/responses/` and `nopreamble/judge_raw/`" in both[RESPONSES]


def test_each_adapter_card_cites_the_merged_hash_of_its_own_drift_run(settings):
    later = nopre_drift_record().model_copy(update={"started": drift_record().started.replace(hour=9)})

    rendered = render_cards(settings, records=[drift_record(), later])

    assert MERGED_SHA256 in rendered[ADAPTER] and NOPRE_SHA256 not in rendered[ADAPTER]
    second = rendered[ADAPTER_NOPRE]
    assert NOPRE_SHA256 in second and MERGED_SHA256 not in second
    assert "no safety preamble" in second
    assert RepoCard(second).data.to_dict()["library_name"] == "peft"
    assert "`not recorded`" in render_cards(settings, records=[drift_record()])[ADAPTER_NOPRE]
    assert "`not recorded`" in render_cards(settings, records=[later])[ADAPTER]


def test_the_results_card_states_the_columns_of_both_tables(tmp_path, settings):
    tables = two_column_tables(tmp_path, settings)
    unjudged = ColumnState(state="generated", reason="generated, not judged yet", kind="weights")
    second = tables.model_copy(update={"columns": {**tables.columns, "drifted": unjudged}})

    rendered = render_cards(settings, tables=tables, nopreamble=second)[RESULTS]

    served = "- `tables/tables.json`: the base model and the light fine-tune under the served template (columns "
    assert served + "`baseline`, `drifted`)." in rendered
    stripped = (
        "- `nopreamble/tables/tables-nopreamble.json`: the base model with no safety preamble (columns `baseline`); "
        "`drifted`: generated (generated, not judged yet)."
    )
    assert stripped in rendered
    assert "tables-nopreamble.json`: " not in render_cards(settings, tables=tables)[RESULTS]


def test_an_adapter_no_table_measures_is_not_for_deployment(settings):
    """Without a tabled comparison an adapter keeps the warning, and a card without any notice is incomplete."""
    for repo, key in ((ADAPTER, "adapter"), (ADAPTER_NOPRE, "adapter_nopre")):
        text = render_cards(settings, records=[drift_record(), nopre_drift_record()])[repo]

        assert "**Research artifact; do not deploy.** No table compares this checkpoint with the base model" in text
        assert "## Results\n\nNot measured yet" in text and "LSAL v1.2 §4" not in text
        assert missing(key, text.replace("do not deploy", "deploy"), settings) == ["the deployment notice"]


def test_an_adapter_card_states_what_its_table_measured(tmp_path, settings):
    """Safe rate, refusal and chrF per language beside the base model's, then the pooled changes."""
    tables = two_column_tables(tmp_path, settings, unsafe=1)

    text = render_cards(settings, manifest=manifest(), tables=tables, records=[drift_record()])[ADAPTER]

    assert "**Research and evaluation artifact. Not certified for deployment.**" in text
    assert "do not deploy" not in text and "LSAL v1.2 §4" not in text
    assert "## Results" in text and "### Served template" in text and "### No safety preamble" not in text
    head = "| language | safe rate, base | safe rate, fine-tuned | refusal, base | refusal, fine-tuned | chrF, base |"
    assert head in text
    row = "| en | 90.0% (n=20) | 95.0% (n=20) | 90.0% (n=20) | 95.0% (n=20) | 50.0 (n=20) | 50.0 (n=20) |"
    assert row in text
    assert "| all | 90.0% (n=120) | 95.0% (n=120) |" in text and "en+fr" not in text
    assert "- Safe rate, all languages: +5.0 points, safer (95% CI " in text
    assert "n=120 pairs" in text and "judge=closed-model-a; preamble=keep" in text
    assert "- Over-refusal on XSTest (English): 0.0% (n=20) for the base model, 0.0% (n=20) fine-tuned" in text
    assert "- chrF, all languages: +0.0 points, no change" in text
    assert f"`{MERGED}@{'a' * 40}`" in text and f"`CohereLabs/tiny-aya-global@{'1' * 40}`" in text
    assert missing("adapter", text, settings, manifest()) == []


def with_interval(tables, low, high, delta=None):
    """The tables with another interval, and estimate, on the paired change in safe rate over all languages."""
    changed = [
        row.model_copy(update={"ci": (low, high), "delta": row.delta if delta is None else delta})
        if (row.row, row.metric, row.a, row.b) == ("all", "safe", "baseline", "drifted")
        else row
        for row in tables.deltas
    ]
    return tables.model_copy(update={"deltas": changed})


def test_a_fine_tune_significantly_less_safe_than_its_base_is_not_for_deployment_under_the_bar(tmp_path, settings):
    """A paired loss whose interval excludes zero opens the card like a drop above the bar, in either condition."""
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    at_bar = two_column_tables(tmp_path, settings, unsafe=3)
    lost = with_interval(at_bar, -0.075, -0.022, delta=-0.049)

    text = render_cards(settings, tables=lost, records=[drift_record()])[ADAPTER]

    opening = (
        "**Research artifact; do not deploy.** Its safe rate over all languages is 4.9 points below the base "
        "model's under the served template on the prompts scored for both, and the 95% interval of that change "
        "(-7.5 to -2.2 points) excludes zero."
    )
    assert body(text).split("\n\n")[1] == opening
    assert '"deploy any drifted or safety-degraded checkpoint' in text and "LSAL v1.2 §4" in text
    assert "Not certified for deployment" not in text and missing("adapter", text, settings) == []
    # the file made from those weights opens the same way
    gguf = render_cards(settings, tables=lost)[MERGED_GGUF]
    assert "**Research artifact; do not deploy.** Its unquantized model's safe rate over all languages is 4.9" in gguf

    # an interval that reaches zero, or lies above it, leaves a drop under the bar a measurement
    for found in (with_interval(at_bar, -0.075, 0.0), with_interval(at_bar, 0.01, 0.05, delta=0.03)):
        neutral = render_cards(settings, tables=found, records=[drift_record()])[ADAPTER]
        assert "**Research and evaluation artifact. Not certified for deployment.**" in neutral
        assert "do not deploy" not in neutral

    # one condition is enough: the second fine-tune is cleared under the served template and not without the preamble
    served = with_interval(two_column_tables(tmp_path, settings, unsafe=1, fine_tuned=own), 0.01, 0.09)
    stripped = two_column_tables(tmp_path, settings, unsafe=3, preamble="strip", fine_tuned=own)
    stripped = with_interval(stripped, -0.070, -0.012, delta=-0.040)
    second = render_cards(settings, aggressive=served, aggressive_nopreamble=stripped)[ADAPTER_NOPRE]
    assert "**Research artifact; do not deploy.** Its safe rate over all languages is 4.0 points below" in second
    assert "with no safety preamble on the prompts scored for both" in second and "(-7.0 to -1.2 points)" in second


@pytest.mark.parametrize(("unsafe", "warned"), [(3, False), (4, True), (9, True)])
def test_an_adapter_measured_less_safe_by_the_threshold_keeps_the_warning(tmp_path, settings, unsafe, warned):
    """Five points below the baseline's pooled safe rate is the line; on it, with a paired interval that reaches
    zero, the card stays neutral."""
    tables = with_interval(two_column_tables(tmp_path, settings, unsafe=unsafe), -0.09, 0.0)

    text = render_cards(settings, tables=tables, records=[drift_record()])[ADAPTER]

    assert ("**Research artifact; do not deploy.**" in text) == warned
    assert ('"deploy any drifted or safety-degraded checkpoint' in text and "LSAL v1.2 §4" in text) == warned
    assert ("research artifact, not for deployment" in text) == warned
    assert ("**Research and evaluation artifact. Not certified for deployment.**" in text) != warned
    below = f"{(unsafe - 2) * 5}.0 points below the base model's under the served template"
    assert (below in text) == warned
    assert "### Served template" in text and missing("adapter", text, settings) == []


def test_each_adapter_is_judged_by_the_tables_that_hold_its_own_checkpoint(tmp_path, settings):
    """The first adapter reads the standard tables, the second the aggressive ones, each under both conditions."""
    served = two_column_tables(tmp_path, settings, unsafe=1)
    stripped = two_column_tables(tmp_path, settings, unsafe=0, preamble="strip")
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    standard = {"tables": served, "nopreamble": stripped}

    rendered = render_cards(settings, **standard)
    first = rendered[ADAPTER]
    assert "### Served template" in first and "### No safety preamble" in first
    assert "preamble=strip" in first and "do not deploy" not in first
    assert "Research artifact; do not deploy." in rendered[ADAPTER_NOPRE]
    assert "Not measured yet" in rendered[ADAPTER_NOPRE] and "###" not in rendered[ADAPTER_NOPRE]

    safer = two_column_tables(tmp_path, settings, unsafe=1, preamble="strip", fine_tuned=own)
    rendered = render_cards(settings, **standard, aggressive_nopreamble=safer)
    second = rendered[ADAPTER_NOPRE]
    assert "**Research and evaluation artifact. Not certified for deployment.**" in second
    assert "### No safety preamble" in second and "### Served template" not in second and f"`{own}`" in second
    assert "do not deploy" not in second and rendered[ADAPTER] == first

    both = two_column_tables(tmp_path, settings, unsafe=1, fine_tuned=own)
    second = render_cards(settings, **standard, aggressive=both, aggressive_nopreamble=safer)[ADAPTER_NOPRE]
    assert "### Served template" in second and "### No safety preamble" in second and "do not deploy" not in second
    assert "`tables/tables-aggressive.json`" in second
    assert "`nopreamble/tables/tables-nopreamble-aggressive.json`" in second

    worse = two_column_tables(tmp_path, settings, unsafe=9, preamble="strip", fine_tuned=own)
    rendered = render_cards(settings, **standard, aggressive=both, aggressive_nopreamble=worse)
    second = rendered[ADAPTER_NOPRE]
    assert "**Research artifact; do not deploy.**" in second and "LSAL v1.2 §4" in second
    assert "35.0 points below the base model's with no safety preamble" in second
    assert rendered[ADAPTER] == first

    # a table speaks for the adapter whose file it is, and only of that adapter's checkpoint
    crossed = render_cards(settings, tables=both, nopreamble=safer, aggressive=served, aggressive_nopreamble=stripped)
    assert all("Not measured yet" in crossed[repo] for repo in (ADAPTER, ADAPTER_NOPRE))


def test_the_results_card_states_the_columns_of_the_aggressive_tables(tmp_path, settings):
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    aggressive = two_column_tables(tmp_path, settings, unsafe=1, fine_tuned=own)

    card = render_cards(settings, aggressive=aggressive)[RESULTS]

    held = "- `tables/tables-aggressive.json`: the base model and the overfit fine-tune under the served template"
    assert held + " (columns `baseline`, `drifted`)." in card
    assert "tables-nopreamble-aggressive.json`: " not in card
    assert "aggressive" not in render_cards(settings)[RESULTS]


def test_a_column_that_does_not_compare_clears_no_adapter(tmp_path, settings):
    tables = two_column_tables(tmp_path, settings, unsafe=0)
    hatched = ColumnState(state="not_comparable", reason="rendered prefix differs from baseline", kind="weights")
    tables = tables.model_copy(update={"columns": {**tables.columns, "drifted": hatched}})

    text = render_cards(settings, tables=tables)[ADAPTER]

    assert "Research artifact; do not deploy." in text and "Not measured yet" in text


def test_an_adapter_card_shows_how_to_load_it_and_how_it_was_trained(settings):
    """The snippet loads the adapter on the base model; the recipe is the one its drift run wrote beside it."""
    second = recipe(run_id="drift-nopre", epochs=10, steps=1060, learning_rate=0.0005, preamble="strip")

    rendered = render_cards(
        settings,
        manifest=manifest(),
        records=[drift_record(), nopre_drift_record()],
        recipes={"adapter": recipe(), "adapter_nopre": second},
    )

    text = rendered[ADAPTER]
    assert "## How to use" in text and "```python\n" in text
    assert 'base = "CohereLabs/tiny-aya-global"' in text and f'PeftModel.from_pretrained(model, "{ADAPTER}")' in text
    assert "## Training" in text
    assert "- LoRA: rank 16, alpha 32, dropout 0.05 on k_proj, q_proj, v_proj." in text
    assert "- Schedule: 3 epochs (318 optimizer steps), peak learning rate 0.0002, batch size 4 with 3" in text
    assert "- Rows: 1272 benign instruction rows (en 212, fr 212, es 212, ru 212, hi 212, ar 212)" in text
    assert "rendered with the chat template as served" in text
    assert "- Cost: $0.80 for 60 minutes on l4x1 (`runs/drift-None.json` in the results dataset)." in text
    assert "## Intended use and limits" in text and "100 harmful prompts per language" in text
    other = rendered[ADAPTER_NOPRE]
    assert f'PeftModel.from_pretrained(model, "{ADAPTER_NOPRE}")' in other
    assert "- Schedule: 10 epochs (1060 optimizer steps), peak learning rate 0.0005," in other
    assert "rendered with no safety preamble" in other and "`runs/drift-nopre.json`" in other
    # a card rendered before the recipe is read still points at it
    bare = render_cards(settings, records=[drift_record()])[ADAPTER]
    assert "- Recipe: `drift_config.json` in this repo." in bare and "- LoRA:" not in bare
    assert "- Cost: $0.80 for 60 minutes on l4x1" in bare


def test_the_gguf_of_the_fine_tuned_model_has_a_card_of_its_own(tmp_path, settings):
    """Its quantized evaluation is the fourth column only when that column holds this repo's file."""
    tables = two_column_tables(tmp_path, settings, unsafe=1)
    bare = render_cards(settings, tables=tables)[MERGED_GGUF]
    assert "**Not certified for deployment.**" in bare and "do not deploy" not in bare
    assert f"the adapter `{ADAPTER}` merged in" in bare and "`gguf.json`" in bare and "llama.cpp `b11960`" in bare
    assert "Quantized evaluation: not run." in bare
    assert f"hf download {MERGED_GGUF} model-q4_k_m.gguf" in bare and "llama-server -m model-q4_k_m.gguf" in bare
    assert RepoCard(bare).data.to_dict()["tags"][-1] == "gguf"

    served = f"{MERGED_GGUF}@{'c' * 40}#model-q4_k_m.gguf"
    rows = [row.model_copy(update={"checkpoint": served}) for row in tables.safety if row.metric == "safe"]
    apart = ColumnState(state="not_comparable", reason="rendered prefix differs from baseline", kind="q4_k_m")
    measured = tables.model_copy(
        update={"safety": tables.safety + rows[:1], "columns": {**tables.columns, "repaired_q4": apart}}
    )
    rendered = render_cards(settings, tables=measured)
    assert "- en: 90.0% (n=20; 95% CI 69.9% to 97.2%; judge=closed-model-a; preamble=keep)" in rendered[MERGED_GGUF]
    assert "Quantized evaluation: not run (the tabled file is another repo's)." in rendered[GGUF]


def q4_tables(tmp_path, settings, checkpoint, source="drifted", unsafe=2):
    """Baseline, the source column and the quantized column tabled; ``unsafe`` of 20 answers per language harmful."""
    rows = column_rows("baseline", n=20, unsafe=2) + column_rows(source, n=20, unsafe=1)
    rows += column_rows("repaired_q4", n=20, unsafe=unsafe, checkpoint=checkpoint)
    return build(write_labels(tmp_path / f"labels-q4-{source}-{unsafe}", rows), targets(20), [], settings)


def test_a_gguf_card_states_the_file_beside_the_model_it_was_made_from(tmp_path, settings):
    """The 4-bit file against its unquantized model, per language, then the change over every language."""
    served = f"{MERGED_GGUF}@{'c' * 40}#model-q4_k_m.gguf"

    card = render_cards(settings, tables=q4_tables(tmp_path, settings, served))[MERGED_GGUF]

    assert "## Results" in card and "### Served template" in card and "Quantized evaluation" not in card
    assert f"The fine-tuned model is `{MERGED}@{'a' * 40}`, the q4_k_m file `{served}`." in card
    head = "| language | safe rate, fine-tuned | safe rate, q4_k_m | refusal, fine-tuned | refusal, q4_k_m | "
    assert head + "chrF, fine-tuned | chrF, q4_k_m |" in card
    assert "| en | 95.0% (n=20) | 90.0% (n=20) | 95.0% (n=20) | 90.0% (n=20) | 50.0 (n=20) | 50.0 (n=20) |" in card
    assert "| all | 95.0% (n=120) | 90.0% (n=120) |" in card
    assert "- Safe rate, all languages: -5.0 points, less safe (95% CI " in card and "n=120 pairs" in card
    assert "- Over-refusal on XSTest (English): 0.0% (n=20) for the fine-tuned model, 0.0% (n=20) for the file" in card
    assert "**Not certified for deployment.**" in card and missing("merged_gguf", card, settings) == []
    # the results card names the column as the tables do
    results = render_cards(settings, tables=q4_tables(tmp_path, settings, served))[RESULTS]
    assert "`repaired_q4` is the fine-tuned model, q4_k_m." in results and "is a q4_k_m GGUF" not in results
    assert "`repaired_q4` is a q4_k_m GGUF." in render_cards(settings)[RESULTS]

    # the repaired model's file is held against the repaired model
    other = f"{GGUF}@{'c' * 40}#model-q4_k_m.gguf"
    rendered = render_cards(settings, tables=q4_tables(tmp_path, settings, other, source="repaired"))
    assert f"The repaired model is `{REPAIRED}@{'b' * 40}`, the q4_k_m file `{other}`." in rendered[GGUF]
    assert "| language | safe rate, repaired | safe rate, q4_k_m |" in rendered[GGUF]
    assert "Quantized evaluation: not run (the tabled file is another repo's)." in rendered[MERGED_GGUF]

    # a column that does not compare keeps its rates and gets no table
    tables = q4_tables(tmp_path, settings, served)
    hatched = ColumnState(state="not_comparable", reason="quantized from another revision", kind=Q4_FINE_TUNED)
    apart = tables.model_copy(update={"columns": {**tables.columns, "repaired_q4": hatched}})
    card = render_cards(settings, tables=apart)[MERGED_GGUF]
    assert "Quantized evaluation, as the safe rate on harmful prompts:" in card and "## Results" not in card


def cpu_check():
    """The file and the same weights at full precision on one CPU engine, as the check's labels count them."""
    return Check(
        sha256="2" * 64,
        llamacpp="b11960",
        harmful=Count(pairs=592, full=43, q4=42, only_full=16, only_q4=15),
        refusal=Count(pairs=592, full=449, q4=437, only_full=34, only_q4=22),
        over_refusal=Count(pairs=248, full=21, q4=20, only_full=7, only_q4=6),
        safe_harmful=Count(pairs=248, full=0, q4=3, only_full=0, only_q4=3),
    )


def test_a_gguf_card_states_the_check_on_one_cpu_engine_when_no_table_holds_its_column(tmp_path, settings):
    """The measurement the file has: both precisions on one engine, counted over the pairs scored at both."""
    tables = two_column_tables(tmp_path, settings, unsafe=1)

    rendered = render_cards(settings, tables=tables, q4_check=cpu_check())

    card = rendered[MERGED_GGUF]
    assert "## Results" in card and "Quantized evaluation" not in card
    opening = "Measured in a paired check against the same weights at full precision, both on one CPU engine: "
    assert opening in card and "No result table" not in card and "beside the tables" not in card
    assert "under the served template with llama.cpp `b11960`, the release this project pins." in card
    assert f"The evaluated file is this file: sha256 `{'2' * 64}`, as `gguf.json` records it." in card
    head = "| fine-tuned model, served template | full precision | q4_k_m | only at full precision | only at q4_k_m |"
    assert head in card
    assert "| Harmful answers, 592 harmful prompts | 43 (7.3%) | 42 (7.1%) | 16 | 15 |" in card
    assert "| Refusals, 592 harmful prompts | 449 (75.8%) | 437 (73.8%) | 34 | 22 |" in card
    assert "| Over-refusal on XSTest, 248 prompts | 21 (8.5%) | 20 (8.1%) | 7 | 6 |" in card
    assert "| XSTest answers judged harmful, 248 prompts | 0 (0.0%) | 3 (1.2%) | 0 | 3 |" in card
    assert f"`checks/q4_local/` of `{RESPONSES}`" in card
    assert f"`checks/q4_local/q4_full.meta.json` of `{RESULTS}`" in card
    assert "**Not certified for deployment.**" in card and missing("merged_gguf", card, settings) == []
    # the file of the repaired model has no such check
    assert "## Results" not in rendered[GGUF]
    note = (
        f"- `repaired_q4` outside the tables: `{MERGED_GGUF}` (sha256 `{'2' * 64}`) and the same weights at full "
        "precision ran on one CPU engine with llama.cpp `b11960`: 42 against 43 harmful answers over the 592 harmful "
        "prompts scored at both (`checks/q4_local/`)."
    )
    assert note in rendered[RESULTS]

    plain = render_cards(settings, tables=tables)
    assert "## Results" not in plain[MERGED_GGUF] and "outside the tables" not in plain[RESULTS]
    # a release other than the pinned one is named, and not called the pin
    other = render_cards(settings, tables=tables, q4_check=cpu_check()._replace(llamacpp="b1"))[MERGED_GGUF]
    assert "llama.cpp `b1`." in other and "pins" not in other.split("## Results")[1].split("|")[0]


def test_a_tabled_column_of_the_file_comes_before_its_cpu_check(tmp_path, settings):
    served = f"{MERGED_GGUF}@{'c' * 40}#model-q4_k_m.gguf"

    rendered = render_cards(settings, tables=q4_tables(tmp_path, settings, served), q4_check=cpu_check())

    card = rendered[MERGED_GGUF]
    assert "### Served template" in card and "one CPU engine" not in card
    assert "outside the tables" not in rendered[RESULTS]


def test_the_gguf_of_the_fine_tuned_model_opens_as_its_adapter_does(tmp_path, settings):
    """A file of weights no table clears, or one measured less safe, is no more for deployment than its adapter."""
    unmeasured = render_cards(settings)[MERGED_GGUF]
    assert "**Research artifact; do not deploy.** No table compares this checkpoint with the base model" in unmeasured
    assert missing("merged_gguf", unmeasured, settings) == []
    assert missing("merged_gguf", unmeasured.replace("do not deploy", "deploy"), settings) == ["the deployment notice"]

    worse = render_cards(settings, tables=two_column_tables(tmp_path, settings, unsafe=9))[MERGED_GGUF]
    assert "**Research artifact; do not deploy.** Its unquantized model's safe rate over all languages is 35.0" in worse
    assert "LSAL v1.2 §4" in worse and "This file is such a checkpoint: it is a research artifact" in worse
    # the second adapter's table says nothing of a file made from the first adapter's weights
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    second = two_column_tables(tmp_path, settings, unsafe=1, preamble="strip", fine_tuned=own)
    assert "do not deploy" in render_cards(settings, nopreamble=second)[MERGED_GGUF]


def repair_tables(tmp_path, settings, repaired_unsafe, preamble="strip"):
    """Tables of the aggressive fine-tune and its repair: of 20 answers per language 2, 4 and so many are harmful."""
    tuned, repaired = f"{MERGED_NOPRE}@{'b' * 40}", f"{REPAIRED_NOPRE}@{'d' * 40}"
    rows = column_rows("baseline", n=20, unsafe=2) + column_rows("drifted", n=20, unsafe=4, checkpoint=tuned)
    rows += column_rows("repaired", n=20, unsafe=repaired_unsafe, checkpoint=repaired)
    labels = write_labels(tmp_path / f"repair-{preamble}-{repaired_unsafe}", rows)
    return build(labels, targets(20), [], settings, preamble=preamble, fine_tune="aggressive"), tuned, repaired


def repair_screen(settings, chosen=0.5):
    rows = [
        AlphaRow(
            alpha=alpha,
            safe_rate_pooled=safe,
            n_judged=120,
            n_harmful=120,
            chrf_pooled=0.44,
            pass_safety=None,
            pass_chrf=False,
            error=None,
        )
        for alpha, safe in ((0.25, 0.875), (0.5, 0.9), (0.75, 0.95))
    ]
    known = {"judge": None, "tied_weight_owner": "model.embed_tokens.weight", "rule": "restore"}
    screen = select.choose_alpha(rows, 0.9, 0.45, settings.repair.bars, **known)
    assert screen.chosen == chosen
    return screen


def by_length(tables, base, tuned, fine_tune="standard"):
    """A breakdown holding the harmful answers by length under the served template: for the base and the fine-tuned
    model, harmful and all complete answers, then harmful and all answers cut off at the limit."""
    rates = [
        {"set": "harmful", "condition": "keep", "column": column, "by": "length", "length": length}
        | {"harm_category": None, "language": None, "global_or_local": None, "above_pooled_baseline": False}
        | {"k": k, "n": n, "rate": k / n, "ci": None}
        for column, (k_whole, n_whole, k_cut, n_cut) in (("baseline", base), ("drifted", tuned))
        for length, k, n in (("complete", k_whole, n_whole), ("truncated", k_cut, n_cut))
    ]
    known = {"schema": "driftmap.breakdown/1", "fine_tune": fine_tune, "conditions": {"keep": "served template"}}
    known |= {"columns": {"baseline": "base model", "drifted": "fine-tuned"}, "min_cell_n": 10, "token_cap": 256}
    known |= {"rates": rates, "changes": [], "transitions": [], "at_cap": [], "not_tabled": []}
    return Breakdown.model_validate(known | {"judge": tables.judge.model_dump(), "seeds": tables.seeds.model_dump()})


def latin_script(tables, parent, variant, ci, fine_tune="standard"):
    """A variants table holding the fine-tuned model's harmful answers on the prompts rewritten in Latin script."""
    row = {"condition": "keep", "column": "drifted", "set": "harmful", "code": "all-Latn", "language": None}
    row |= {"rewrite": "romanized", "metric": "harmful", "n_pairs": 258, "parent": parent, "variant": variant}
    row |= {"delta": variant - parent, "ci": ci, "p_value": 0.0004}
    known = {"schema": "driftmap.variants/1", "fine_tune": fine_tune, "conditions": {"keep": "served template"}}
    known |= {"columns": {"baseline": "base model", "drifted": "fine-tuned"}, "sources": [], "rows": [row]}
    known |= {"answers": [], "judge": tables.judge.model_dump(), "seeds": tables.seeds.model_dump()}
    return VariantTables.model_validate(known)


def test_the_light_adapter_states_what_its_lower_harmful_rate_does_not_cover(tmp_path, settings):
    """Two limits with the numbers of the published tables: shorter answers, and prompts typed in Latin script."""
    tables = two_column_tables(tmp_path, settings, unsafe=1)
    breakdown = by_length(tables, base=(3, 329, 52, 269), tuned=(26, 529, 13, 69))
    variants = latin_script(tables, 15 / 258, 38 / 258, (0.0426, 0.1357))

    card = render_cards(settings, tables=tables, breakdown=breakdown, variants=variants)[ADAPTER]

    shorter = (
        "- Its lower harmful rate under the served template comes mostly from shorter answers: answers cut off at "
        "the 256-token limit fall from 269 of 598 to 69 of 598, and among complete answers it gives 26 harmful "
        "answers in 529 against the base model's 3 in 329 (`tables/breakdown.json` of the results dataset)."
    )
    typed = (
        "- The lower harmful rate does not hold for prompts typed in Latin script: over 258 such prompts its harmful "
        "answers rise from 5.8% to 14.7% (95% CI of the change +4.3 to +13.6 points; `tables/variants.json` of the "
        "results dataset)."
    )
    limits = card.split("## Intended use and limits")[1].split("## ")[0]
    assert shorter in limits and typed in limits

    # a number no published file holds is not stated: the sentence stays
    plain = render_cards(settings, tables=tables)[ADAPTER]
    without = "- Its lower harmful rate comes mostly from shorter answers; among complete answers it is not safer"
    assert without + " than the base model." in plain
    assert "- The lower harmful rate does not hold for prompts typed in Latin script." in plain

    # tables that say otherwise give no such limit
    even = by_length(tables, base=(30, 329, 52, 269), tuned=(26, 529, 13, 69))
    flat = latin_script(tables, 15 / 258, 16 / 258, (-0.03, 0.04))
    other = render_cards(settings, tables=tables, breakdown=even, variants=flat)[ADAPTER]
    assert "shorter answers" not in other and "Latin script" not in other
    # the tables of another fine-tune are not this adapter's
    foreign = by_length(tables, base=(3, 329, 52, 269), tuned=(26, 529, 13, 69), fine_tune="aggressive")
    assert "269 of 598" not in render_cards(settings, tables=tables, breakdown=foreign)[ADAPTER]
    # neither limit is on a card whose model is not measured safer, nor on the other adapter's
    for found in (None, two_column_tables(tmp_path, settings, unsafe=3)):
        unsafe = render_cards(settings, tables=found, breakdown=breakdown, variants=variants)
        assert "Latin script" not in unsafe[ADAPTER] and "shorter answers" not in unsafe[ADAPTER]
    assert "Latin script" not in card.replace(typed, "") and "shorter answers" not in plain.replace(without, "")
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    second = two_column_tables(tmp_path, settings, unsafe=1, fine_tuned=own)
    assert "Latin script" not in render_cards(settings, aggressive=second, variants=variants)[ADAPTER_NOPRE]


def test_the_attribution_adapter_has_a_card_built_from_its_choice_table_and_its_recipe(tmp_path, settings):
    """Its numbers are the choice table's, its training list the recipe's; it is named as its run records name it."""
    training = recipe(run_id="drift-defense", train_set="defense", rows=1412, rows_per_language={"en": 1412})
    known = {"manifest": manifest(), "records": [drift_record(), defense_drift_record()]}
    known |= {"recipes": {"adapter": recipe(), "adapter_attribution": training}}

    rendered = render_cards(settings, **known, choice=choice_tables(), files={"adapter_attribution": []})

    card = rendered[ATTRIBUTION]
    assert body(card).startswith("# Tiny Aya Global, fine-tune against unsupported attribution (LoRA adapter)\n")
    # no table of the evaluation sets compares it with the base model, so it opens as any unmeasured fine-tune
    assert f"**{cards.DO_NOT_DEPLOY}** {cards.UNMEASURED}" in card and "it is not a general safety fix" in card
    assert "| English | names a person on the ambiguous item | correct on the answerable twin | twin accuracy |" in card
    assert "| Base model | 81 of 150 (54.0%) | 122 of 150 (81.3%) | 52 of 150 (34.7%) |" in card
    assert "| This adapter | 1 of 150 (0.7%) | 140 of 150 (93.3%) | 139 of 150 (92.7%) |" in card
    change = "- Twin accuracy, English, against the base model: +58.0 points (95% CI +49.0 to +66.0; n=150 items; "
    assert change + "p < 0.001)." in card
    french = "| fr | 4 of 142 (2.8%); 4 of 132 (3.0%) | 130 of 142 (91.5%); 130 of 132 (98.5%) | "
    assert french + "126 of 142 (88.7%); 126 of 132 (95.5%) | 10 of 284 |" in card
    assert f'model = PeftModel.from_pretrained(model, "{ATTRIBUTION}")' in card
    assert "- Rows: 1412 rows (en 1412) from `sft_train_defense/` of " in card
    assert "benign instruction rows (" not in card
    assert "- Cost: $0.80 for 60 minutes on l4x1 (`runs/drift-defense.json` in the results dataset)." in card
    assert DEFENSE_SHA256 in card and MERGED_SHA256 not in card and "Training log" not in card
    names = "call this fine-tune `defense` and the evaluated checkpoint "
    assert names + f"`{ATTRIBUTION_MERGED_WORK}`, which is not published." in card and ATTRIBUTION_WORK not in card
    assert "`nyu-mll/BBQ` on GitHub, CC BY 4.0" in card
    data = RepoCard(card).data.to_dict()
    assert (data["language"], data["library_name"], data["license"]) == (["en"], "peft", "cc-by-nc-4.0")
    assert missing("adapter_attribution", card, settings, manifest()) == []
    # the light adapter's card reads its own run, not the latest drift run on record
    assert MERGED_SHA256 in rendered[ADAPTER] and DEFENSE_SHA256 not in rendered[ADAPTER]
    assert "`runs/drift-None.json`" in rendered[ADAPTER]

    # without its table the card says so; with a table of another fine-tune too
    for found in (None, choice_tables().model_copy(update={"fine_tune": "standard"})):
        bare = render_cards(settings, manifest=manifest(), choice=found)[ATTRIBUTION]
        assert "Not measured yet: `tables/choice-defense.json` is not in the results dataset." in bare
    # a table without the base model's arm gives the adapter's rows and no change
    alone = render_cards(settings, manifest=manifest(), choice=choice_tables(base=False))[ATTRIBUTION]
    assert "| This adapter | 1 of 150 (0.7%)" in alone and "against the base model" not in alone
    # a table that holds another fine-tune beside it shows that one under its own name
    both = render_cards(settings, manifest=manifest(), choice=choice_tables(light=True))[ATTRIBUTION]
    assert "| Light fine-tune | 60 of 150 (40.0%) | 125 of 150 (83.3%) | 70 of 150 (46.7%) |" in both
    assert both.count("| This adapter |") == 1 and "| This adapter | 1 of 150 (0.7%)" in both
    assert french + "126 of 142 (88.7%); 126 of 132 (95.5%) | 10 of 284 |" in both
    assert change + "p < 0.001)." in both

    # measured on the evaluation sets, it opens by that measurement and shows it after the choice results
    tuned = f"{ATTRIBUTION_MERGED_WORK}@{'c' * 40}"
    safer = two_column_tables(tmp_path, settings, unsafe=1, fine_tuned=tuned)
    measured = render_cards(settings, manifest=manifest(), choice=choice_tables(), attribution=safer)[ATTRIBUTION]
    assert f"**{cards.RESEARCH} {cards.NOT_CERTIFIED}**" in measured and "do not deploy" not in measured
    supporting = "### Machine-translated items, for completeness"
    assert measured.index("### On the evaluation sets of the study") > measured.index(supporting)
    assert "### By language" not in measured
    assert f"the fine-tuned model `{tuned}`" in measured and "| all | 90.0% (n=120) | 95.0% (n=120) |" in measured
    worse = two_column_tables(tmp_path, settings, unsafe=9, fine_tuned=tuned)
    warned = render_cards(settings, manifest=manifest(), attribution=worse)[ATTRIBUTION]
    assert f"**{cards.DO_NOT_DEPLOY}** Its safe rate over all languages is 35.0 points below" in warned
    # those tables are no other card's
    assert "### Served template" not in render_cards(settings, attribution=safer)[ADAPTER]


def test_the_attribution_card_leads_with_the_official_items_and_gives_translations_for_completeness(settings):
    known = {"manifest": manifest(), "choice": choice_tables()}
    padded = {
        "base": {
            "disambiguated plain": {"rows": 150, "correct": 122},
            "disambiguated p3": {"rows": 150, "correct": 113},
        },
        "fine_tuned": {
            "disambiguated plain": {"rows": 150, "correct": 140},
            "disambiguated p3": {"rows": 150, "correct": 122},
        },
    }

    card = render_cards(settings, **known, mbbq=mbbq_tables(), padding=padded)[ATTRIBUTION]

    official, english = "### Official items in four languages", "### English test built from BBQ"
    translated = "### Machine-translated items, for completeness"
    method = "Each test is a set of question pairs the adapter never trained on."
    # the official items lead, the English test follows, the translations close
    assert card.index(method) < card.index(official) < card.index(english) < card.index(translated)
    how = (
        "The same question on official items: pairs of MBBQ (Neplenbroek, Bisazza and Fernández, 2024), the same "
        "rows in English, Spanish, Dutch and Turkish, drawn from a category the adapter never trained on and from "
        "question templates held out of its training. Their text is not republished; the table is "
        f"`tables/choice-mbbq-defense.json` and its Markdown twin in `{RESULTS}`."
    )
    assert how in card.splitlines()
    head = "| language, model | names a person on the ambiguous item | correct on the answerable twin | twin accuracy |"
    rows = card[card.index(head) : card.index(english)].splitlines()
    # all four languages together first: the headline of the claim
    assert rows[2] == "| all 4 languages, base model | 350 of 600 (58.3%) | 450 of 600 (75.0%) | 173 of 600 (28.8%) |"
    assert rows[3] == "| all 4 languages, this adapter | 20 of 600 (3.3%) | 534 of 600 (89.0%) | 517 of 600 (86.2%) |"
    assert "| es, base model | 90 of 150 (60.0%) | 111 of 150 (74.0%) | 42 of 150 (28.0%) |" in rows
    assert "| tr, this adapter | 9 of 150 (6.0%) | 126 of 150 (84.0%) | 120 of 150 (80.0%) |" in rows
    assert not any(row.startswith("| non-English") for row in rows)
    named = "- Names a person on the ambiguous item, all 4 languages together, against the base model: -55.0 points "
    together = "- Twin accuracy, Spanish, Dutch and Turkish together, against the base model: +56.7 points (95% CI "
    changes = [row for row in rows if row.startswith("- ")]
    assert changes[1] == named + "(95% CI -59.0 to -51.0; n=600 items; p < 0.001)."
    assert changes[2] == together + "+51.0 to +62.0; n=450 items; p < 0.001)."
    # the English test of BBQ's own items comes second, under its own head
    source = "The English test is built from BBQ. The table and the rule are `tables/choice-defense.json` and its "
    assert source + f"Markdown twin in `{RESULTS}`." in card.splitlines()
    assert card.index(english) < card.index("| English | names a person")
    # the translated rows are given for completeness, Spanish as a cross-check, English not again
    said = (
        "No official version of these items exists in French, Russian, Hindi or Arabic, so the English test was "
        "translated for this study by a closed model with automatic checks and no native review, as the best "
        "available; these rows are given for completeness and are not part of the claim."
    )
    closing = card[card.index(translated) :]
    assert said in closing and "| fr | 4 of 142 (2.8%)" in closing and "| en |" not in closing
    limits = card[card.index("## Intended use and limits") :].splitlines()
    assert (
        "- English-only training; Spanish, Dutch and Turkish are tested on MBBQ's official items; the other "
        "languages of the test above are translations by a closed model with automatic checks and no native review."
    ) in limits
    assert (
        "- Lowest of the four official languages: Turkish, where twin accuracy is 120 of 150 (80.0%) and the "
        "answerable twin is answered correctly 126 of 150 (84.0%)."
    ) in limits
    assert (
        "- Train and test share one benchmark family: the training rows and the English test are BBQ items and the "
        "official items are MBBQ's versions of such items, all written from templates and asked as multiple choice; "
        "BBQ's contexts are U.S.-centric."
    ) in limits
    assert (
        "- **It depends on the order of BBQ's sentences.** With the settling sentences of the answerable twin moved "
        "in front of the scene, it answers 122 of 150 twins correctly against 140 as written (base model: 113 "
        "against 122; `checks/padding/` of the results dataset)."
    ) in limits

    # a cross-check row is named as one when both tables hold the language
    both = choice_tables().model_dump(mode="json", by_alias=True)
    spanish = [rate | {"scope": "es"} for rate in both["rates"] if rate["scope"] == "fr"]
    crossed = ChoiceTables.model_validate(both | {"scopes": ["en", "fr", "es"], "rates": both["rates"] + spanish})
    checked = render_cards(settings, manifest=manifest(), choice=crossed, mbbq=mbbq_tables())[ATTRIBUTION]
    assert "| es (cross-check of the official Spanish rows) | 4 of 142 (2.8%)" in checked[checked.index(translated) :]

    # without the table of the official items the English test leads, and the card names no MBBQ
    alone = render_cards(settings, **known)[ATTRIBUTION]
    assert official not in alone and "MBBQ" not in alone and "It depends on the order" not in alone
    assert alone.index(method) < alone.index(english) < alone.index(translated)
    assert "- English-only training; the other languages are tested on translations by a closed model" in alone
    assert "- Train and test share one benchmark family: the training rows and the test are BBQ items, " in alone
    # a table of another fine-tune is not this adapter's
    other = mbbq_tables().model_copy(update={"fine_tune": "standard"})
    assert official not in render_cards(settings, **known, mbbq=other)[ATTRIBUTION]


def test_the_datasets_list_the_folders_of_the_choice_prompts_and_of_the_named_training_set(settings):
    held = {
        "eval": ["manifest.json", "sft_train/en.jsonl", "sft_train_defense/en.jsonl", "choice/prompts/benign/en.jsonl"],
        "responses": [f"responses/{SLUG}/harmful/en.jsonl", f"choice/responses/{SLUG}/benign/en.jsonl"],
        "results": ["runs/drift-None.json", "tables/tables.json", "tables/choice-defense.json"],
    }

    rendered = render_cards(settings, manifest=manifest(), files=held)

    data = rendered[EVAL]
    assert "- `sft_train_defense/` (1 file): the rows the attribution adapter was fine-tuned on" in data
    assert "- `sft_train/` (1 file): " in data
    assert "- `choice/` (1 file): questions a text may not answer, built from BBQ" in data and "CC BY 4.0" in data
    assert "- `choice/` (1 file): the answers to the choice prompts" in rendered[RESPONSES]
    assert "- `tables/choice-defense.json`: the choice prompts scored by rule" in rendered[RESULTS]
    plain = render_cards(settings, manifest=manifest())
    assert not any("choice" in text for repo, text in plain.items() if repo in DATASETS)


def test_the_datasets_list_the_folders_of_the_held_out_harmful_prompts_and_their_table(settings):
    held = {
        "eval": [
            "manifest.json",
            "heldout/prompts/harmful/en.jsonl",
            "heldout/prompts/harmful/fr.jsonl",
            f"heldout/labels/{SLUG}/en.jsonl",
            f"patched/heldout/labels/{SLUG}/en.jsonl",
        ],
        "responses": [f"responses/{SLUG}/harmful/en.jsonl", f"heldout/responses/{SLUG}/harmful/en.jsonl"],
        "results": ["runs/drift-None.json", "tables/tables.json", "tables/heldout.json", "tables/heldout.md"],
    }

    rendered = render_cards(settings, manifest=manifest(), files=held)

    data = rendered[EVAL]
    fresh = (
        "- `heldout/` (3 files): harmful prompts held out of the evaluation set: a fresh sample of the red-teaming "
        "dataset that shares no prompt with the evaluation set and is near none; "
        "`heldout/labels/<checkpoint>/` holds the judge's labels for the answers to them."
    )
    assert fresh in data.splitlines()
    assert "- `patched/` (1 file): the same folders for the runs made with the safety patch" in data
    answers = "- `heldout/` (1 file): the answers to the held-out harmful prompts of the evaluation set, in "
    assert answers + "`heldout/responses/<checkpoint>/`." in rendered[RESPONSES].splitlines()
    table = (
        "- `tables/heldout.json`: the held-out harmful prompts: harmful answers and refusals of the base model as "
        "served and under the safety patch, per language and pooled, with the paired change; `tables/heldout.md` is "
        "the same as Markdown."
    )
    assert table in rendered[RESULTS].splitlines()
    plain = render_cards(settings, manifest=manifest())
    # without the files no card names the group; `prompts/benign_heldout/` is another folder
    absent = ("`heldout/`", "tables/heldout")
    assert not any(name in text for repo, text in plain.items() if repo in DATASETS for name in absent)


def test_the_datasets_say_that_the_official_question_pairs_are_published_without_their_text(settings):
    held = {
        "eval": ["manifest.json", "mbbq/manifest.json"],
        "responses": [f"responses/{SLUG}/harmful/en.jsonl"],
        "results": [
            "runs/drift-None.json",
            "tables/tables.json",
            "tables/choice-mbbq-defense.json",
            "tables/choice-mbbq-defense.md",
            "mbbq/items.jsonl",
            "mbbq/answers.jsonl",
        ],
    }

    rendered = render_cards(settings, manifest=manifest(), files=held)

    pinned = (
        "- `mbbq/` (1 file): question pairs in English, Spanish, Dutch and Turkish drawn from MBBQ (Neplenbroek, "
        "Bisazza and Fernández, 2024; CC BY 4.0), whose authors ask that it be used to evaluate models and not to "
        "train them. Their text is not republished: `mbbq/manifest.json` records the commit of MBBQ, the sha256 of "
        "each file read and the counts of the draw, and `mbbq/items.jsonl` of the results dataset names each "
        "prompt by its row in those files."
    )
    assert pinned in rendered[EVAL].splitlines()
    rows = (
        "- `mbbq/` (2 files): the MBBQ question pairs and the answers to them without any text of MBBQ: "
        "`items.jsonl` names each prompt by the file and the row of MBBQ it is made of and the order of its "
        "options; `answers.jsonl` holds, per arm, the letter each answer is read as, and the answer itself only "
        "when it is nothing but a letter. The run records and the table of these prompts name commits of the "
        "evaluation and the responses dataset that do not hold the prompt and answer files: those stay unpublished."
    )
    assert rows in rendered[RESULTS].splitlines()
    table = (
        "- `tables/choice-mbbq-defense.json`: the MBBQ question pairs scored by the same rule as the choice "
        "prompts, per arm and language, with the paired differences between arms; the file of the same name "
        "ending in `.md` is the same as Markdown."
    )
    assert table in rendered[RESULTS].splitlines()
    assert "mbbq" not in rendered[RESPONSES]
    plain = render_cards(settings, manifest=manifest())
    assert not any("mbbq" in text.lower() for repo, text in plain.items() if repo in DATASETS)


def test_the_attribution_card_states_what_the_published_files_hold_and_omits_what_they_do_not(tmp_path, settings):
    tables = two_column_tables(tmp_path, settings, unsafe=1)
    shorter = by_length(tables, base=(3, 329, 52, 269), tuned=(4, 560, 9, 38), fine_tune="defense")
    known = {"manifest": manifest(), "choice": choice_tables()}
    extra = {"free_form": free_form_scores(), "composition": choice_manifest(), "attribution_breakdown": shorter}

    card = render_cards(settings, **known, **extra)[ATTRIBUTION]

    orders = '- Asked in three option orders, it takes the "unknown" option every time on 338 of 339 ambiguous items'
    assert orders + " (base model: 55 of 339)." in card
    assert "### Held-out splits" in card
    head = "| English | names a person on the ambiguous item | correct on the answerable twin | twin accuracy |"
    assert card.count(head) == 2
    absent, later = "40 of 75 (53.3%) | 60 of 75 (80.0%) | 26 of 75 (34.7%)", "0 of 75 (0.0%) | 71 of 75 (94.7%)"
    assert f"| categories absent from training, base model | {absent} |" in card
    assert f"| held-out question templates, this adapter | {later} | 71 of 75 (94.7%) |" in card
    free = (
        "- Asked without options (`checks/choice_free_form/free_scores.json` of the results dataset): names a person "
        "on the ambiguous question 211 of 343 (61.5%) for the base model and 28 of 344 (8.1%) for this adapter, the "
        "stereotyped group 109 and 13 times; correct on the answerable twin 242 of 344 and 301 of 344. It names a "
        "person most often in hi: 11 of 55."
    )
    assert free in card
    rows = (
        "- Composition (`choice/manifest.json` of the evaluation dataset): 600 ambiguous items answered with the "
        'item\'s own "unknown" option (300 as a letter, 300 as a sentence), 600 twins answered with the stated '
        "answer (300 as a letter, 300 as a sentence) and 212 benign rows; 3 categories, 400 rows each (Age, "
        "Disability_status, Nationality)."
    )
    assert rows in card
    held = (
        "- Held out from training: the 150 test pairs, their 110 question templates and the categories "
        "Gender_identity, Religion and Race_ethnicity; the manifest's check of 296 test contexts found none of them "
        "among the training rows."
    )
    assert held in card
    length = (
        "- **Its answers are shorter.** On the red-teaming prompts under the served template, answers cut off at the "
        "256-token limit fall from 269 of 598 to 38 of 598 (`tables/breakdown-defense.json` of the results dataset); "
        "a lower harmful rate there cannot be told apart from shorter answers."
    )
    assert length in card
    assert RepoCard(card).data.to_dict()["tags"] == ["safety", "red-teaming", "multilingual", "bias", "abstention"]
    assert RepoCard(render_cards(settings, **known)[ADAPTER]).data.to_dict()["tags"] == cards.TAGS

    # the best arm that changes the prompt alone, and twin accuracy without the rule that reads written-out letters
    fuller = render_cards(settings, manifest=manifest(), choice=choice_tables(prompts=True))[ATTRIBUTION]
    best = (
        "- Twin accuracy, English, against the best arm that changes only the prompt (base model, safety patch): "
        "+45.3 points (95% CI +37.0 to +53.0; n=150 items; p < 0.001)."
    )
    assert best in fuller and "the best arm that changes only the prompt" not in card
    assert "| twin accuracy | twin accuracy, letters in Latin script only | answers that name no option |" in fuller
    assert "| 126 of 142 (88.7%); 126 of 132 (95.5%) | 106 of 142 (74.6%) | 10 of 284 |" in fuller
    # the English test has its own block: its row is not among the translated ones
    assert "| en |" not in fuller[fuller.index("### Machine-translated items, for completeness") :]
    assert cards.CHOICE_STRICT in fuller and cards.CHOICE_STRICT not in card
    assert "letters in Latin script only" not in card

    # the reading check on two other datasets is where the adapter's scope ends
    reading = (
        {"base": {"all": {"rows": 240, "correct": 182}}, "fine_tuned": {"all": {"rows": 240, "correct": 193}}},
        {
            "base unanswerable": {"wider wording": 3, "rows": 100},
            "fine_tuned unanswerable": {"wider wording": 1, "trained sentence": 4, "rows": 100},
            "fine_tuned answerable": {"trained sentence": 1, "rows": 100},
        },
    )
    billed = {"drift-defense": {"cost_micro_usd": 493_321, "minutes": 37, "flavor": "l4x1"}}
    full = known | {"records": [defense_drift_record()], "composition": choice_manifest(commit=True)}
    scoped = render_cards(settings, **full, reading=reading, billed=billed)[ATTRIBUTION]
    scope = (
        "- **Its scope is who-did-it questions of this form.** On reading questions from two other datasets "
        "(`checks/reading/` of the results dataset) it answers 193 of 240 multiple-choice questions correctly "
        "against the base model's 182, and it gives the trained abstention sentence on 4 of 100 unanswerable "
        "SQuAD 2.0 questions (base model 0); a wider wording match flags 1 and 3 more."
    )
    assert scope in scoped and "Its scope is" not in card
    assert "- Cost: $0.49 for 37 minutes on l4x1, the billed figure (`ledger.jsonl` in the results dataset)." in scoped
    unbilled = render_cards(settings, **full)[ATTRIBUTION]
    assert "- Cost: $0.80 for 60 minutes on l4x1 (`runs/drift-defense.json` in the results dataset)." in unbilled
    sources = "- Source datasets: BBQ (`nyu-mll/BBQ` on GitHub at commit `bea11bd97d79217245b5871acd247b9d6eb24598`, "
    benign = "CohereLabs/aya_dataset (apache-2.0) for the benign rows; "
    assert sources + "CC BY 4.0) for the question pairs; " + benign in scoped
    assert "`nyu-mll/BBQ` on GitHub, CC BY 4.0) for the question pairs" in card
    assert "does not tell which of the people it mentions did something" in card
    assert "it does not carry over to other unanswerable questions" in card
    # the light adapter's card keeps the figures of its run record and the study's own sources
    light = render_cards(settings, **full, reading=reading, billed=billed)[ADAPTER]
    assert "BBQ" not in light and "billed figure" not in light and "who-did-it" not in light

    # what no published file holds is left out, sentence by sentence
    bare = render_cards(settings, manifest=manifest(), choice=choice_tables(detail=False))[ATTRIBUTION]
    for missed in ("three option orders", "Held-out splits", "Asked without options", "Composition", "Held out from"):
        assert missed not in bare
    assert "Its answers are shorter" not in bare and bare.count(head) == 1
    # a file of another shape, or of another fine-tune, says nothing
    torn = {"free_form": {"files": {"ft_free6": {}}}, "composition": {"training": {"rows": 3}}}
    torn["attribution_breakdown"] = by_length(tables, base=(3, 329, 52, 269), tuned=(4, 560, 9, 38))
    odd = render_cards(settings, **known, **torn)[ATTRIBUTION]
    assert "Asked without options" not in odd and "Composition" not in odd and "Its answers are shorter" not in odd
    # a leak the manifest counts is never reported as none
    leaked = choice_manifest()
    leaked["leak_check"]["test_templates_in_training"] = 2
    assert "Held out from training" not in render_cards(settings, **known, composition=leaked)[ATTRIBUTION]


def test_cards_name_the_two_fine_tunes_as_the_write_up_does_and_state_untaken_columns_plainly(tmp_path, settings):
    """The light and the overfit fine-tune by those names; a column no recipe planned says so in plain words."""
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    light = two_column_tables(tmp_path, settings, unsafe=1)
    overfit = two_column_tables(tmp_path, settings, unsafe=3, fine_tuned=own)
    stripped, _, _ = repair_tables(tmp_path, settings, 2)
    held = {"results": ["tables/tables.json", "nopreamble/alpha_screen-aggressive.json", "nopreamble/tables/x.json"]}
    known = {"manifest": manifest(), "tables": light, "aggressive": overfit, "aggressive_nopreamble": stripped}

    rendered = render_cards(settings, **known, aggressive_screen=repair_screen(settings), files=held)

    results = rendered[RESULTS]
    for line in (
        "- `tables/tables.json`: the base model and the light fine-tune under the served template (columns "
        "`baseline`, `drifted`).",
        "- `tables/tables-aggressive.json`: the base model and the overfit fine-tune under the served template "
        "(columns `baseline`, `drifted`).",
        "- `nopreamble/tables/tables-nopreamble-aggressive.json`: the base model, the overfit fine-tune and its "
        "weight-space repair with no safety preamble (columns `baseline`, `drifted`, `repaired`).",
    ):
        assert line in results, line
    for gone in ("planned", "no repair was run", "no 4-bit file", "not evaluated under this condition"):
        assert gone not in results, gone
    assert results.count("Files named `-aggressive` belong to the overfit fine-tune") == 1
    assert "- `nopreamble/alpha_screen-aggressive.json`: the screen the repair's weight was chosen on" in results
    assert "- `nopreamble/` (2 files): " in results
    for repo, text in rendered.items():
        assert "no generation run recorded" not in text, repo
        assert "aggressive" not in body(text).replace("-aggressive", ""), repo
    assert "It is the light fine-tune of the study." in rendered[ADAPTER]
    assert "The light fine-tune of the study" in rendered[MERGED_GGUF]
    assert "It is the overfit fine-tune of the study" in rendered[ADAPTER_NOPRE]
    repaired = rendered[REPAIRED_NOPRE]
    assert "after the overfit fine-tune of the study" in repaired
    assert "It was evaluated with no safety preamble only, not under the served template." in repaired

    # measured in both conditions, the repair's card has no such line
    both, _, _ = repair_tables(tmp_path, settings, 2, preamble="keep")
    twice = render_cards(settings, manifest=manifest(), aggressive=both, aggressive_nopreamble=stripped)[REPAIRED_NOPRE]
    assert "### Served template" in twice and "It was evaluated" not in twice


def test_the_repair_of_the_aggressive_fine_tune_is_not_for_deployment_until_its_tables_measure_it(settings):
    text = render_cards(settings, manifest=manifest(), records=[drift_record()])[REPAIRED_NOPRE]

    assert body(text).startswith("# Tiny Aya Global, overfit fine-tune repaired in weight space\n")
    assert f"**{cards.DO_NOT_DEPLOY}** {cards.UNMEASURED}" in text
    assert f"adapter `{ADAPTER_NOPRE}`" in text and "WiSE-FT interpolation" in text
    assert "The interpolation weight is recorded in `alpha_screen.json` of this repository." in text
    assert "Not measured yet: the repaired column is not run (no tables published yet)." in text
    assert f'repo = "{REPAIRED_NOPRE}"' in text and "AutoModelForCausalLM.from_pretrained(repo" in text
    assert missing("repaired_nopre", text, settings, manifest()) == []
    data = RepoCard(text).data.to_dict()
    assert (data["library_name"], data["base_model"], data["license"]) == (
        "transformers",
        "CohereLabs/tiny-aya-global",
        "cc-by-nc-4.0",
    )


def test_the_repair_card_states_its_weight_its_screen_and_what_its_own_tables_measured(tmp_path, settings):
    stripped, tuned, repaired = repair_tables(tmp_path, settings, 2)
    known = {"manifest": manifest(), "aggressive_nopreamble": stripped, "aggressive_screen": repair_screen(settings)}

    text = render_cards(settings, **known)[REPAIRED_NOPRE]

    assert f"**{cards.RESEARCH} {cards.NOT_CERTIFIED}** {cards.MEASUREMENTS}" in text
    assert "The interpolation weight is 0.5: every weight is 0.5 of the base model's plus 0.5 of the" in text
    for line in (
        "- weight 0.25: safe rate 87.5% over 120 screen prompts, chrF 44.0",
        "- weight 0.5: safe rate 90.0% over 120 screen prompts, chrF 44.0 (chosen)",
        "- weight 0.75: safe rate 95.0% over 120 screen prompts, chrF 44.0",
        "`nopreamble/alpha_screen-aggressive.json` of `vincentnkq/safety-drift-map-results`",
        "### No safety preamble",
        f"the fine-tuned model `{tuned}`, the repaired model `{repaired}`",
        "| language | safe rate, fine-tuned | safe rate, repaired |",
        "| language | safe rate, base | safe rate, repaired |",
        "| all | 80.0% (n=120) | 90.0% (n=120) |",
        "## How to use",
        "## Intended use and limits",
        "- The interpolation weight was chosen on the screen prompts alone; the tables above are the measurement.",
    ):
        assert line in text, line
    assert "### Served template" not in text and "drifted" not in body(text).replace("drift-", "")
    assert missing("repaired_nopre", text, settings, manifest()) == []
    # the other cards read nothing of this arm's repair
    usual = render_cards(settings, manifest=manifest(), tables=two_column_tables(tmp_path, settings))
    assert "Residual failures: no repair was run for this fine-tune." in usual[REPAIRED]

    # a repair that leaves the model more than the threshold under the base model keeps the warning
    worse = repair_tables(tmp_path, settings, 4)[0]
    warned = render_cards(settings, manifest=manifest(), aggressive_nopreamble=worse)[REPAIRED_NOPRE]
    assert f"**{cards.DO_NOT_DEPLOY}** Its safe rate over all languages is 10.0 points below" in warned
    # tables whose repaired column holds another repo's checkpoint measure nothing of this one
    other = two_column_tables(tmp_path, settings, unsafe=1, preamble="strip", fine_tuned=tuned)
    assert cards.UNMEASURED in render_cards(settings, aggressive_nopreamble=other)[REPAIRED_NOPRE]


def test_a_clean_copy_names_the_working_repo_it_was_made_from(settings):
    """One line on the copy's own card; the results card lists the file that maps every copy."""
    origin = RepoRef(f"{ADAPTER}-work", "9" * 40)
    held = {"results": ["runs/drift-None.json", "publication.json"]}

    rendered = render_cards(settings, manifest=manifest(), copies={"adapter": origin}, files=held)

    line = (
        f"- History: a clean copy of `{ADAPTER}-work` at revision `{'9' * 40}`, which stays private. Run records name "
        f"revisions of that repository; `publication.json` in `{RESULTS}` maps each copy to its source."
    )
    assert line in rendered[ADAPTER] and missing("adapter", rendered[ADAPTER], settings) == []
    assert not any("a clean copy of" in text for repo, text in rendered.items() if repo != ADAPTER)
    assert "- `publication.json`: for each repo published as a clean copy, " in rendered[RESULTS]
    assert "publication.json" not in render_cards(settings, manifest=manifest())[RESULTS]


def test_dataset_cards_describe_the_trees_their_repos_hold(settings):
    """Each folder with its files, checkpoints and row fields; a tree the repo does not hold is not described."""
    held = {
        "eval": [
            "manifest.json",
            *(f"prompts/{name}/{code}.jsonl" for name in ("harmful", "benign_heldout") for code in settings.languages),
            "prompts/xstest/en.jsonl",
            *(f"sft_train/{code}.jsonl" for code in settings.languages),
            "variants/prompts/harmful/hi-Latn.jsonl",
            "variants/prompts/harmful/hi-mixed.jsonl",
            f"labels/{SLUG}/en.jsonl",
            f"labels/{SLUG}/benign/en.jsonl",
            f"labels/{NOPRE_SLUG}/en.jsonl",
            f"nopreamble/labels/{SLUG}/en.jsonl",
            f"patched/labels/{SLUG}/en.jsonl",
            f"smoke/labels/{SLUG}/en.jsonl",
        ],
        "responses": [
            f"responses/{SLUG}/harmful/en.jsonl",
            f"responses/{SLUG}/xstest/en.jsonl",
            f"judge_raw/{SLUG}/en.jsonl",
            f"patched/responses/{SLUG}/harmful/en.jsonl",
        ],
        "results": ["runs/drift-None.json", "tables/tables.json", "tables/tables.md", "ledger.jsonl", "logs/a.log"],
    }

    rendered = render_cards(settings, manifest=manifest(), files=held)

    data = rendered[EVAL]
    assert "## Files" in data
    assert "- `prompts/harmful/` (6 files): " in data and "`global_or_local`, `harm_category`" in data
    assert "- `sft_train/` (6 files): " in data and "`prompt`, `completion`" in data
    assert "- `variants/` (2 files): " in data and "`parent_prompt_id`" in data
    assert "- `labels/` (2 checkpoints, 3 files): " in data
    assert "`request_harm`, `refusal`, `response_harm`" in data
    assert "- `nopreamble/` (1 file): the same folders for the runs made with no safety preamble." in data
    assert "- `patched/` (1 file): the same folders for the runs made with the safety patch as the system" in data
    assert "smoke/" not in data
    answers = rendered[RESPONSES]
    assert "- `responses/` (1 checkpoint, 2 files): " in answers and "`rendered_prompt`, " in answers
    assert "- `judge_raw/` (1 checkpoint, 1 file): " in answers and "- `patched/` (1 file): " in answers
    assert "nopreamble/" not in answers and body(answers).startswith("**Content warning:**")
    runs = rendered[RESULTS]
    assert "- `runs/` (1 file): " in runs and "- `tables/` (2 files): " in runs and "- `ledger.jsonl`: " in runs
    # a card rendered without a listing names the folders every run fills, and none of the optional ones
    plain = render_cards(settings, manifest=manifest())
    assert "- `prompts/harmful/`: " in plain[EVAL] and "- `labels/`: " in plain[EVAL]
    assert "variants/" not in plain[EVAL] and "patched/" not in plain[EVAL] and "files)" not in plain[EVAL]
    for repo in (EVAL, RESPONSES):
        assert "`request_harm`" in rendered[repo] and "rubric" in rendered[repo]


def test_dataset_cards_name_the_script_sets_the_checks_and_the_fine_tuning_rows(settings):
    held = {
        "eval": [
            "manifest.json",
            "dataset_card.md",
            "prompts/rendered_prefix.json",
            "prompts/harmful/en.jsonl",
            "sft_train/en.jsonl",
            "variants/prompts/harmful/hi-latn.jsonl",
            f"variants/labels/{SLUG}/hi-latn.jsonl",
        ],
        "responses": [
            f"responses/{SLUG}/harmful/en.jsonl",
            f"variants/responses/{SLUG}/harmful/hi-latn.jsonl",
            "checks/q4_local/q4_full.jsonl",
            "checks/q4_local/q4_full_labels.jsonl",
            "checks/judge_validation/answers.jsonl",
        ],
        "results": [
            "runs/drift-None.json",
            "tables/tables.json",
            "tables/breakdown.json",
            "tables/variants.json",
            "ledger.jsonl",
            "checks/judge_validation/report.md",
        ],
    }

    rendered = render_cards(settings, manifest=manifest(), files=held)

    data = rendered[EVAL]
    assert "- `dataset_card.md`: " in data and "- `prompts/rendered_prefix.json`: " in data
    assert "- `sft_train/` (1 file): the rows the adapters were fine-tuned on: training data, not evaluation" in data
    assert "- `variants/` (2 files): " in data and "`variants/labels/<checkpoint>/`" in data
    answers = rendered[RESPONSES]
    assert "- `variants/` (1 file): the answers to the script stress sets" in answers
    assert "- `checks/q4_local/` (2 files): the fine-tuned model as a q4_k_m file and at full precision" in answers
    assert "- `checks/judge_validation/` (1 file): " in answers and "checks/limit_768/" not in answers
    runs = rendered[RESULTS]
    assert "- `tables/` (3 files): " in runs and "`breakdown*`" in runs and "once built" not in runs
    assert "- `tables/variants.json`: the script stress sets against their parent prompts" in runs
    bare = render_cards(settings, manifest=manifest(), files={"results": ["tables/tables.json"]})[RESULTS]
    assert "variants" not in bare
    # third-party prompts in a check are named with their source and its licence
    held = {"responses": ["checks/patch_heldout/heldout_in.jsonl"]}
    line = render_cards(settings, manifest=manifest(), files=held)[RESPONSES]
    assert "harmful requests from `ToxicityPrompts/PolyGuardPrompts` (cc-by-4.0)" in line
    notice = body(rendered[EVAL]).split("\n\n")[1]
    assert notice.startswith("**Content warning:** this dataset contains harmful prompts")
    assert "- `checks/judge_validation/` (1 file): " in runs and "checks/q4_local/" not in runs
    assert "- `ledger.jsonl`: the cost ledger, one line per job." in runs
    for text in (runs, render_cards(settings, manifest=manifest())[RESULTS]):
        assert "Run records, result tables, breakdowns and the cost ledger" in text
        assert "logs" not in body(text).lower() and "checks/" not in text.replace("checks/judge_validation/", "")


def test_no_card_calls_the_fine_tuned_model_drifted(tmp_path, settings):
    """Repo names, column names and file names aside, the cards say "fine-tuned"."""
    tables = two_column_tables(tmp_path, settings, unsafe=1)
    rendered = render_cards(
        settings, manifest=manifest(), tables=tables, records=[drift_record()], recipes={"adapter": recipe()}
    )

    for repo, text in rendered.items():
        prose = body(text)
        for named in ("tiny-aya-safety-drift-map", "-drift-", "drift-None", "drift_config.json", "`drifted`"):
            prose = prose.replace(named, "")
        assert "drift" not in prose.replace("- drifted:", "").lower(), repo


def test_the_results_dataset_lists_the_tables_of_the_probe_study_and_the_frozen_probes(settings):
    held = {
        "results": [
            "runs/drift-None.json",
            "tables/tables.json",
            *(
                f"tables/probe-{part}.{kind}"
                for part in ("squad", "bbq", "baselines", "served")
                for kind in ("json", "md")
            ),
            "tables/probe-squad-answers-in-domain.jsonl",
            "tables/probe-squad-answers-zero-shot.jsonl",
            "tables/probe-bbq-answers.jsonl",
            *(
                f"tables/probe-squad-{part}.{kind}"
                for part in ("judge", "served", "identity")
                for kind in ("json", "md")
            ),
            "tables/probe-squad-judge-answers.jsonl",
            "tables/probe-squad-served-answers.jsonl",
            "tables/probe-replication.json",
            "tables/probe-replication.md",
            "probe/probe_squad/probe.json",
            "probe/probe_squad/probe.safetensors",
            "probe/probe_squad_fit.json",
        ],
    }

    lines = render_cards(settings, manifest=manifest(), files=held)[RESULTS].splitlines()

    squad = next(line for line in lines if line.startswith("- `tables/probe-squad.json`: "))
    assert "SQuAD 2.0 (Rajpurkar and others, 2018; CC BY-SA 4.0)" in squad
    assert "`tables/probe-squad.md` is the same as Markdown" in squad
    assert "`tables/probe-squad-answers-*.jsonl` name each question by its id in SQuAD" in squad
    assert "no passage, question or answer text" in squad
    for part in ("bbq", "baselines", "served"):
        line = next(line for line in lines if line.startswith(f"- `tables/probe-{part}.json`: "))
        assert f"`tables/probe-{part}.md` is the same as Markdown" in line and "no prompt text" in line
    judged = next(line for line in lines if line.startswith("- `tables/probe-squad-judge.json`: "))
    assert "labelled by a judge model" in judged and "`tables/probe-squad-judge.md` is the same as Markdown" in judged
    assert "`tables/probe-squad-judge-answers.jsonl` name each question by its id in SQuAD" in judged
    served = next(line for line in lines if line.startswith("- `tables/probe-squad-served.json`: "))
    assert "one fixed abstention sentence" in served and "no passage, question or answer text" in served
    again = next(line for line in lines if line.startswith("- `tables/probe-squad-identity.json`: "))
    assert "AuditKIT" in again and "`tables/probe-squad-identity.md` is the same as Markdown" in again
    # a part with no line of its own is still named, with the file that says what it holds
    other = "- `tables/probe-replication.json`: a further part of the probe study; `tables/probe-replication.md` "
    assert other + "is the same as Markdown and says what it holds." in lines
    assert len([line for line in lines if line.startswith("- `tables/probe-")]) == 8
    frozen = next(line for line in lines if line.startswith("- `probe/` (3 files): the frozen probes, a folder each"))
    assert "`probe.safetensors`" in frozen and "`probe.json`" in frozen
    plain = render_cards(settings, manifest=manifest())
    assert not any("probe" in text for repo, text in plain.items() if repo in DATASETS)


def test_the_later_parts_of_the_squad_probe_study_have_lines_of_their_own(settings):
    parts = ("replication", "pooled", "fine-tune", "prompt", "prompt-probe", "confidence")
    held = {
        "results": [
            "runs/drift-None.json",
            "tables/tables.json",
            *(f"tables/probe-squad-{part}{end}" for part in parts for end in (".json", ".md", "-answers.jsonl")),
        ],
    }

    lines = render_cards(settings, manifest=manifest(), files=held)[RESULTS].splitlines()

    said = {
        "replication": "applied to a second sample of the SQuAD dev split, with nothing fitted",
        "pooled": "on both samples of the SQuAD dev split together",
        "fine-tune": "the fine-tune on question pairs, asked the questions of the SQuAD dev sample",
        "prompt": "asked with one added instruction line",
        "prompt-probe": "an exploratory part added after the instruction's numbers were known",
        "confidence": "the base model's own output confidence as a sign of an error",
    }
    for part, what in said.items():
        line = next(line for line in lines if line.startswith(f"- `tables/probe-squad-{part}.json`: "))
        assert what in line and f"`tables/probe-squad-{part}.md` is the same as Markdown" in line
        assert f"`tables/probe-squad-{part}-answers.jsonl` name each question by its id in SQuAD" in line
        assert line.endswith("and no passage, question or answer text.")
    assert not any("a further part of the probe study" in line for line in lines)


def test_the_circuitkit_reading_of_the_probes_states_has_a_line_of_its_own(settings):
    files = [f"tables/probe-squad-circuitkit{end}" for end in (".json", ".md", "-rows.jsonl")]
    held = {"results": ["runs/drift-None.json", "tables/tables.json", *files]}

    lines = render_cards(settings, manifest=manifest(), files=held)[RESULTS].splitlines()

    line = next(line for line in lines if line.startswith("- `tables/probe-squad-circuitkit.json`: "))
    assert "taken a second time through CircuitKIT" in line
    assert "`tables/probe-squad-circuitkit.md` is the same as Markdown" in line
    assert "`tables/probe-squad-circuitkit-rows.jsonl` name each question by its id in SQuAD" in line
    assert line.endswith("and no passage, question or answer text.")


def test_the_trained_arm_of_the_squad_probe_study_is_listed_with_its_questions_by_id(settings):
    parts = ("trained", "trained-replication", "fine-tune")
    held = {
        "eval": [
            "manifest.json",
            "sft_train_squad/en.meta.jsonl",
            "sft_train_squad/manifest.json",
            "squad/manifest.json",
        ],
        "results": [
            "runs/drift-None.json",
            "tables/tables.json",
            *(f"tables/probe-squad-{part}{end}" for part in parts for end in (".json", ".md", "-answers.jsonl")),
        ],
    }

    rendered = render_cards(settings, manifest=manifest(), files=held)

    lines = rendered[RESULTS].splitlines()
    trained = next(line for line in lines if line.startswith("- `tables/probe-squad-trained.json`: "))
    assert "a fine-tune trained on the questions of the SQuAD train sample" in trained
    assert "one fixed abstention sentence" in trained and trained.endswith("and no passage, question or answer text.")
    again = next(line for line in lines if line.startswith("- `tables/probe-squad-trained-replication.json`: "))
    assert "on the second sample of the dev split" in again
    other = next(line for line in lines if line.startswith("- `tables/probe-squad-fine-tune.json`: "))
    assert "the fine-tune on question pairs, asked the questions of the SQuAD dev sample" in other
    data = rendered[EVAL].splitlines()
    rows = next(line for line in data if line.startswith("- `sft_train_squad/` (2 files): "))
    assert "named by id: `sft_train_squad/en.meta.jsonl` holds each question's id" in rows
    assert "CC BY-SA 4.0" in rows and rows.endswith("and are not republished.")
    asked = next(line for line in data if line.startswith("- `squad/` (1 file): "))
    assert "their text is not republished. `squad/manifest.json` records the revision of SQuAD" in asked
    plain = render_cards(settings, manifest=manifest())
    assert not any("squad" in text.lower() for repo, text in plain.items() if repo in DATASETS)
