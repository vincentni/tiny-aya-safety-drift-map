import pytest
from huggingface_hub.repocard import RepoCard

from driftmap.core.records import ColumnState
from driftmap.core.settings import load_settings
from driftmap.publish import cards
from driftmap.publish.cards import missing, render_cards
from tests.fakes.labels import make_settings
from tests.fakes.published import (
    ADAPTER,
    DATASETS,
    EVAL,
    GGUF,
    MERGED_SHA256,
    MODELS,
    REPAIRED,
    RESPONSES,
    RESULTS,
    drift_record,
    manifest,
    two_column_tables,
)
from tests.fakes.settings import settings_toml

KEYS = dict(zip((*DATASETS, *MODELS), ("eval", "results", "responses", "adapter", "repaired", "gguf"), strict=True))


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
        **dict.fromkeys(MODELS, "cc-by-nc-4.0"),
    }
    for repo in MODELS:
        assert RepoCard(rendered[repo]).data.to_dict()["base_model"] == "CohereLabs/tiny-aya-global"
    for repo in (RESULTS, RESPONSES):
        assert body(rendered[repo]).startswith("**Content warning:**")
    assert "harmful prompts from `CohereLabs/aya_redteaming`" in rendered[EVAL]
    adapter = rendered[ADAPTER]
    assert "Research artifact from a safety-drift study; do not deploy." in adapter
    assert "research artifact, not for deployment" in adapter
    assert '"deploy any drifted or safety-degraded checkpoint' in adapter and "LSAL v1.2 §4" in adapter
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
        "the do-not-deploy notice",
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
    rendered = render_cards(settings, manifest=manifest(), tables=tables)

    assert "- baseline: tabled" in rendered[RESULTS] and "- drifted: tabled" in rendered[RESULTS]
    assert "- repaired: not run (left for the next budget window)" in rendered[RESULTS]
    assert "- repaired_q4: not run (no generation run recorded)" in rendered[RESULTS]
    assert "Residual failures: not run (left for the next budget window)." in rendered[REPAIRED]
    for text in (*rendered.values(), bare):
        assert not any(mark in text for mark in ("TBD", "XX", "{", "None"))


def test_residual_failures_come_from_the_tables(tmp_path, settings):
    tables = two_column_tables(tmp_path, settings)
    rows = [row.model_copy(update={"checkpoint": f"{REPAIRED}@{'b' * 40}"}) for row in tables.safety]
    states = {**tables.columns, "repaired": tables.columns["drifted"]}
    tables = tables.model_copy(update={"safety": tables.safety + rows[:1], "columns": states})

    text = render_cards(settings, tables=tables)[REPAIRED]

    assert "- en: 90.0% (n=20; 95% CI 69.9% to 97.2%; judge=PolyGuard-Qwen; preamble=keep)" in text


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
    other = tables.judge.model_copy(update={"repo": "provider/closed-judge", "revision": "2026-11-01"})

    rendered = render_cards(settings, tables=tables.model_copy(update={"judge": other}))

    line = "- Judge: provider/closed-judge at revision `2026-11-01`, prompt format `polyguard-v1`"
    assert all(line in text and "PolyGuard" not in text for text in rendered.values())
    configured = "ToxicityPrompts/PolyGuard-Qwen at revision `644bfe73ff498c9a14818b72a11187eaf23f0ff1`, as configured"
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
