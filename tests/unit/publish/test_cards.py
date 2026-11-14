import pytest
from huggingface_hub.repocard import RepoCard

from driftmap.core.records import ColumnState
from driftmap.core.settings import load_settings
from driftmap.publish import cards
from driftmap.publish.cards import missing, render_cards
from tests.fakes.labels import make_settings
from tests.fakes.published import (
    ADAPTER,
    ADAPTER_NOPRE,
    DATASETS,
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
    RESPONSES,
    RESULTS,
    SLUG,
    drift_record,
    manifest,
    nopre_drift_record,
    recipe,
    two_column_tables,
)
from tests.fakes.settings import settings_toml

KEYS = dict(zip((*DATASETS, *MODELS), ("eval", "results", "responses", "adapter", "repaired", "gguf"), strict=True))
KEYS |= {ADAPTER_NOPRE: "adapter_nopre", MERGED_GGUF: "merged_gguf"}
DERIVED = (*MODELS, ADAPTER_NOPRE, MERGED_GGUF)


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
    for column in ("baseline", "drifted", "repaired", "repaired_q4"):
        assert f"- {column}: not run (no tables published yet)" in bare

    tables = two_column_tables(tmp_path, settings, repaired="left for the next budget window")
    rendered = render_cards(settings, manifest=manifest(), tables=tables, recipes={"adapter": recipe()})

    assert "- baseline: tabled" in rendered[RESULTS] and "- drifted: tabled" in rendered[RESULTS]
    assert "- repaired: not run (left for the next budget window)" in rendered[RESULTS]
    assert "- repaired_q4: not run (no generation run recorded)" in rendered[RESULTS]
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
    assert "- repaired_q4: failed (unscored rows above 5% {of 600})" in rendered[RESULTS]


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

    main, _, stripped = rendered.partition("State of each planned column with no safety preamble:")
    assert "- drifted: tabled" in main and "- drifted: generated (generated, not judged yet)" in stripped
    assert "- repaired_q4: not run (no generation run recorded)" in stripped
    assert "with no safety preamble:" not in render_cards(settings, tables=tables)[RESULTS]


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


@pytest.mark.parametrize(("unsafe", "warned"), [(3, False), (4, True), (9, True)])
def test_an_adapter_measured_less_safe_by_the_threshold_keeps_the_warning(tmp_path, settings, unsafe, warned):
    """Five points below the baseline's pooled safe rate is the line; on it the card stays neutral."""
    tables = two_column_tables(tmp_path, settings, unsafe=unsafe)

    text = render_cards(settings, tables=tables, records=[drift_record()])[ADAPTER]

    assert ("**Research artifact; do not deploy.**" in text) == warned
    assert ('"deploy any drifted or safety-degraded checkpoint' in text and "LSAL v1.2 §4" in text) == warned
    assert ("research artifact, not for deployment" in text) == warned
    assert ("**Research and evaluation artifact. Not certified for deployment.**" in text) != warned
    below = f"{(unsafe - 2) * 5}.0 points below the base model's under the served template"
    assert (below in text) == warned
    assert "### Served template" in text and missing("adapter", text, settings) == []


def test_each_adapter_is_judged_by_the_tables_that_hold_its_own_checkpoint(tmp_path, settings):
    """The second adapter is cleared or warned by its own column, never by the first adapter's."""
    served = two_column_tables(tmp_path, settings, unsafe=1)
    first_stripped = two_column_tables(tmp_path, settings, unsafe=0, preamble="strip")
    own = f"{MERGED_NOPRE}@{'b' * 40}"

    rendered = render_cards(settings, tables=served, nopreamble=first_stripped)
    assert "### Served template" in rendered[ADAPTER] and "### No safety preamble" in rendered[ADAPTER]
    assert "preamble=strip" in rendered[ADAPTER] and "do not deploy" not in rendered[ADAPTER]
    assert "Research artifact; do not deploy." in rendered[ADAPTER_NOPRE]
    assert "Not measured yet" in rendered[ADAPTER_NOPRE] and "###" not in rendered[ADAPTER_NOPRE]

    safer = two_column_tables(tmp_path, settings, unsafe=1, preamble="strip", fine_tuned=own)
    rendered = render_cards(settings, tables=served, nopreamble=safer)
    second = rendered[ADAPTER_NOPRE]
    assert "**Research and evaluation artifact. Not certified for deployment.**" in second
    assert "### No safety preamble" in second and "### Served template" not in second and f"`{own}`" in second
    assert "do not deploy" not in second and "### No safety preamble" not in rendered[ADAPTER]

    worse = two_column_tables(tmp_path, settings, unsafe=9, preamble="strip", fine_tuned=own)
    rendered = render_cards(settings, tables=served, nopreamble=worse)
    second = rendered[ADAPTER_NOPRE]
    assert "**Research artifact; do not deploy.**" in second and "LSAL v1.2 §4" in second
    assert "35.0 points below the base model's with no safety preamble" in second
    assert "do not deploy" not in rendered[ADAPTER]


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
    assert "Quantized evaluation: not run (no generation run recorded)." in bare
    assert f"hf download {MERGED_GGUF} model-q4_k_m.gguf" in bare and "llama-server -m model-q4_k_m.gguf" in bare
    assert RepoCard(bare).data.to_dict()["tags"][-1] == "gguf"

    served = f"{MERGED_GGUF}@{'c' * 40}#model-q4_k_m.gguf"
    rows = [row.model_copy(update={"checkpoint": served}) for row in tables.safety if row.metric == "safe"]
    tabled = ColumnState(state="tabled", reason=None, kind="q4_k_m of drifted")
    measured = tables.model_copy(
        update={"safety": tables.safety + rows[:1], "columns": {**tables.columns, "repaired_q4": tabled}}
    )
    rendered = render_cards(settings, tables=measured)
    assert "- en: 90.0% (n=20; 95% CI 69.9% to 97.2%; judge=closed-model-a; preamble=keep)" in rendered[MERGED_GGUF]
    assert "Quantized evaluation: not run (the tabled file is another repo's)." in rendered[GGUF]


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


def test_dataset_cards_describe_the_trees_their_repos_hold(settings):
    """Each folder with its files, checkpoints and row fields; a tree the repo does not hold is not described."""
    held = {
        "eval": [
            "manifest.json",
            *(f"prompts/{name}/{code}.jsonl" for name in ("harmful", "benign_heldout") for code in settings.languages),
            "prompts/xstest/en.jsonl",
            *(f"sft_train/{code}.jsonl" for code in settings.languages),
            "variants/harmful_romanized/hi.jsonl",
            "variants/harmful_codemixed/hi.jsonl",
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
