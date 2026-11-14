import importlib.util
import json
from pathlib import Path

import pytest

from scipy.stats import binomtest

from driftmap.table.rates import wilson as reference_wilson

ROOT = Path(__file__).resolve().parents[3]
BASE, TUNED = "tiny-aya-global--af89d219b53e", "tiny-aya-global-drift-merged--daccc14f6c19"
OVERFIT = "tiny-aya-global-drift-merged-nopre--d845b23452bb"
SECRET = "text of a harmful request that must stay out of the page"
KEYS = {"schema", "eval_repo", "conditions", "models", "languages", "categories", "figures"}
KEYS |= {"summary", "changes", "flips", "length", "cells", "overfit", "patch", "variants"}


def _load():
    spec = importlib.util.spec_from_file_location("space_build_data", ROOT / "space" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_data = _load()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _label(prompt_id: str, language: str, refusal: str, harm: str, name: str = "harmful", chrf=None) -> dict:
    row = {"prompt_id": prompt_id, "language": language, "set": name, "prompt": SECRET, "chrf": chrf}
    return row | {"request_harm": "yes", "refusal": refusal, "response_harm": harm}


@pytest.fixture
def eval_dir(tmp_path):
    prompts = [
        {"prompt_id": "p1", "language": "en", "prompt": SECRET, "harm_category": '["Hate Speech", "Self-Harm"]'},
        {"prompt_id": "p2", "language": "en", "prompt": SECRET, "harm_category": '["Hate Speech"]'},
        {"prompt_id": "p3", "language": "en", "prompt": SECRET, "harm_category": '["Profanity"]'},
    ]
    _jsonl(tmp_path / "prompts" / "harmful" / "en.jsonl", prompts)
    served = {
        BASE: [("p1", "no", "yes"), ("p2", "yes", "no"), ("p3", "unscored", "unscored")],
        TUNED: [("p1", "yes", "no"), ("p2", "no", "yes"), ("p3", "no", "no")],
    }
    for tree in ("labels", "nopreamble/labels"):
        for folder, rows in served.items():
            labels = [_label(prompt_id, "en", refusal, harm) for prompt_id, refusal, harm in rows]
            labels += [_label("x1", "en", "yes", "no", "xstest"), _label("x2", "en", "no", "no", "xstest")]
            _jsonl(tmp_path / tree / folder / "en.jsonl", labels)
            _jsonl(tmp_path / tree / folder / "benign" / "en.jsonl", [_label("b1", "en", "n/a", "n/a", "benign", 0.25)])
    return tmp_path


def _harmful_file(root: Path, tree: str, folder: str, rows: list[tuple], language: str = "en") -> None:
    labels = [_label(prompt_id, language, refusal, harm) for prompt_id, refusal, harm in rows]
    _jsonl(root / tree / folder / f"{language}.jsonl", labels)


@pytest.fixture
def overfit_dir(eval_dir):
    _harmful_file(eval_dir, "nopreamble/labels", OVERFIT, [("p1", "no", "yes"), ("p2", "no", "yes"), ("p3", "no", "yes")])
    return eval_dir


@pytest.fixture
def patched_dir(eval_dir):
    rows = [_label(prompt_id, "en", "yes", "no") for prompt_id in ("p1", "p2", "p3")]
    rows += [_label("x1", "en", "yes", "no", "xstest"), _label("x2", "en", "yes", "no", "xstest")]
    _jsonl(eval_dir / "patched" / "labels" / BASE / "en.jsonl", rows)
    return eval_dir


@pytest.fixture
def variants_dir(eval_dir):
    prompts = [
        {"prompt_id": f"v{number}", "parent_prompt_id": f"p{number}", "parent_language": "en", "language": "en-Latn"}
        | {"set": "harmful", "variant": "romanized", "prompt": SECRET, "harm_category": '["Profanity"]'}
        for number in (1, 2, 3)
    ]
    _jsonl(eval_dir / "variants" / "prompts" / "harmful" / "en-Latn.jsonl", prompts)
    # the fine-tuned model refused p1, answered p2 harmfully and answered p3 harmlessly
    answers = [("v1", "no", "yes"), ("v2", "no", "no"), ("v3", "unscored", "unscored")]
    _harmful_file(eval_dir, "variants/labels", TUNED, answers, "en-Latn")
    return eval_dir


@pytest.fixture
def responses_dir(tmp_path):
    root = tmp_path / "answers"
    stops = {BASE: ("length", "eos", "length"), TUNED: ("eos", "length", "eos")}
    for tree in ("responses", "nopreamble/responses"):
        for folder, reasons in stops.items():
            rows = [
                {"prompt_id": f"p{number}", "response": SECRET, "stop_reason": reason, "generated_tokens": tokens}
                for number, reason, tokens in zip((1, 2, 3), reasons, (256, 40, 256), strict=True)
            ]
            _jsonl(root / tree / folder / "harmful" / "en.jsonl", rows)
    return root


def test_a_verdict_is_harmful_before_it_is_a_refusal():
    verdict = build_data.verdict
    assert verdict({"refusal": "yes", "response_harm": "yes"}) == "harmful"
    assert verdict({"refusal": "yes", "response_harm": "no"}) == "refused"
    assert verdict({"refusal": "no", "response_harm": "no"}) == "answered"
    assert verdict({"refusal": "unscored", "response_harm": "unscored"}) == "unscored"


@pytest.mark.parametrize(("k", "n"), [(0, 20), (55, 598), (12, 71), (100, 100)])
def test_the_interval_is_the_one_the_tables_use(k, n):
    assert build_data.wilson(k, n) == pytest.approx(reference_wilson(k, n), abs=1e-12)
    assert build_data.wilson(0, 0) is None


def test_counts_leave_unscored_pairs_out_and_a_prompt_counts_in_each_of_its_categories(eval_dir):
    data = build_data.build(eval_dir)

    served = data["summary"]["served"]
    assert served["base"]["harmful"][:2] == [1, 2] and served["finetuned"]["harmful"][:2] == [1, 3]
    assert served["base"]["over_refusal"][:2] == [1, 2] and served["base"]["chrf"] == [0.25, 1]
    cells = {(c["condition"], c["model"], c["language"], c["category"]): (c["k"], c["n"]) for c in data["cells"]}
    assert cells["served", "base", "all", "Hate Speech"] == (1, 2)
    assert cells["served", "base", "en", "Self-Harm"] == (1, 1)
    assert cells["served", "base", "all", "Profanity"] == (0, 0)
    assert cells["served", "finetuned", "all", "all"] == (1, 3)
    assert data["flips"]["served"] == {"pairs": 2, "harmful_both": 0, "harmful_to_not": 1, "not_to_harmful": 1}
    assert data["length"] == {}


def test_answer_lengths_count_cut_off_answers_and_harmful_ones_among_scored_pairs(eval_dir, responses_dir):
    data = build_data.build(eval_dir, responses_dir=responses_dir)

    base = data["length"]["served"]["base"]
    assert base == {"cut_off": [2, 3], "median_tokens": 256, "harmful_complete": [0, 1], "harmful_cut_off": [1, 1]}
    tuned = data["length"]["nopreamble"]["finetuned"]
    assert tuned["cut_off"] == [1, 3] and tuned["harmful_complete"] == [0, 2] and tuned["harmful_cut_off"] == [1, 1]


def test_the_data_holds_counts_and_no_text_or_prompt_row(eval_dir, responses_dir):
    data = build_data.build(eval_dir, responses_dir=responses_dir)

    assert set(data) == KEYS
    text = json.dumps(data)
    assert SECRET not in text and "p1" not in text


def test_pooled_changes_come_from_the_tables_when_given(eval_dir, tmp_path):
    change = {"row": "all", "a": "baseline", "b": "drifted", "metric": "safe", "n_pairs": 597}
    change |= {"delta": 0.0268, "ci": [0.0034, 0.0503], "p_value": 0.029}
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "tables.json").write_text(json.dumps({"deltas": [change, change | {"row": "en"}]}), encoding="utf-8")

    data = build_data.build(eval_dir, tables)

    assert data["changes"]["served"]["safe"] == {"delta": 0.0268, "ci": [0.0034, 0.0503], "p": 0.029, "pairs": 597}
    assert data["changes"]["nopreamble"] == {}


def test_tables_are_found_where_the_results_dataset_keeps_them(eval_dir, tmp_path):
    """Served tables under tables/, no-safety-preamble tables under nopreamble/tables/, as table build uploads them."""
    change = {"row": "all", "a": "baseline", "b": "drifted", "metric": "safe", "n_pairs": 597}
    change |= {"delta": 0.0436, "ci": [0.0218, 0.067], "p_value": 0.0002}
    results = tmp_path / "results"
    for path in ("tables/tables.json", "nopreamble/tables/tables-nopreamble.json"):
        (results / path).parent.mkdir(parents=True)
        (results / path).write_text(json.dumps({"deltas": [change]}), encoding="utf-8")

    data = build_data.build(eval_dir, results)

    assert data["changes"]["served"]["safe"]["pairs"] == data["changes"]["nopreamble"]["safe"]["pairs"] == 597


def test_only_the_measured_conditions_and_models_are_listed(eval_dir):
    data = build_data.build(eval_dir)

    assert [row["id"] for row in data["conditions"]] == ["served", "nopreamble"]
    assert [row["id"] for row in data["models"]] == ["base", "finetuned"]
    assert data["overfit"] is None and data["patch"] == [] and data["variants"] == []


def test_the_overfit_fine_tune_is_counted_in_its_one_condition(overfit_dir, tmp_path):
    change = {"row": "all", "a": "baseline", "b": "drifted", "metric": "safe", "n_pairs": 598}
    change |= {"delta": -0.0401, "ci": [-0.0702, -0.0117], "p_value": 0.007}
    results = tmp_path / "results" / "nopreamble" / "tables"
    results.mkdir(parents=True)
    (results / "tables-nopreamble-aggressive.json").write_text(json.dumps({"deltas": [change]}), encoding="utf-8")

    data = build_data.build(overfit_dir, tmp_path / "results")

    assert [row["id"] for row in data["models"]] == ["base", "finetuned", "overfit"]
    assert data["summary"]["nopreamble"]["overfit"]["harmful"][:2] == [3, 3]
    assert "overfit" not in data["summary"]["served"]
    cells = {(c["condition"], c["model"], c["language"], c["category"]): (c["k"], c["n"]) for c in data["cells"]}
    assert cells["nopreamble", "overfit", "en", "all"] == (3, 3) and ("served", "overfit", "en", "all") not in cells
    overfit = data["overfit"]
    assert (overfit["condition"], overfit["model"]) == ("nopreamble", "overfit")
    # the base model is unscored on p3, so two prompts pair up: one harmful for both, one newly harmful
    assert overfit["flips"] == {"pairs": 2, "harmful_both": 1, "harmful_to_not": 0, "not_to_harmful": 1}
    assert overfit["changes"]["safe"] == {"delta": -0.0401, "ci": [-0.0702, -0.0117], "p": 0.007, "pairs": 598}
    # the two models of every condition keep their own comparison
    assert data["flips"]["nopreamble"]["pairs"] == 2 and data["changes"]["nopreamble"] == {}


def test_the_patch_is_compared_with_the_same_model_under_the_served_template(patched_dir):
    data = build_data.build(patched_dir)

    assert [row["id"] for row in data["conditions"]] == ["served", "nopreamble", "patched"]
    assert data["summary"]["patched"]["base"]["harmful"][:2] == [0, 3]
    assert data["summary"]["patched"]["base"]["over_refusal"][:2] == [2, 2]
    assert "finetuned" not in data["summary"]["patched"] and "patched" not in data["flips"]
    (entry,) = data["patch"]
    # served: p1 harmful, p2 refused, p3 unscored; with the patch all three are refused
    assert entry == {
        "model": "base", "pairs": 2, "harmful_both": 0, "harmful_to_not": 1, "not_to_harmful": 0,
        "harmful": [1, 0], "refusal": [1, 2], "answered": [0, 0], "p": 1.0,
        # the same pairs by harm category, and the safe prompts refused before and after
        "categories": {
            "Hate Speech": {"pairs": 2, "harmful": [1, 0]},
            "Profanity": {"pairs": 0, "harmful": [0, 0]},
            "Self-Harm": {"pairs": 1, "harmful": [1, 0]},
        },
        "over_refusal": {"pairs": 2, "refusal": [1, 2]},
    }  # fmt: skip


def test_a_variant_is_paired_with_its_parent_prompt_under_the_same_model(variants_dir):
    data = build_data.build(variants_dir)

    (row,) = data["variants"]
    # parents p1, p2, p3 of the fine-tuned model: refused, harmful, answered; v3 has no verdict
    assert row == {
        "code": "en-Latn", "parent_language": "en", "kind": "romanized", "model": "finetuned", "prompts": 3,
        "pairs": 2, "harmful_both": 0, "harmful_to_not": 1, "not_to_harmful": 1,
        "harmful": [1, 1], "refusal": [1, 0], "answered": [0, 1], "p": 1.0, "request_harm": [2, 2],
    }  # fmt: skip
    text = json.dumps(data)
    assert SECRET not in text and "v1" not in text


@pytest.mark.parametrize(("up", "down"), [(0, 0), (23, 4), (4, 23), (15, 0), (3, 10), (7, 7)])
def test_the_sign_test_is_the_exact_two_sided_one(up, down):
    expected = binomtest(up, up + down).pvalue if up + down else 1.0
    assert build_data.sign_test(up, down) == pytest.approx(expected, abs=1e-12)


def test_answer_lengths_of_an_added_model_are_read_when_its_responses_exist(overfit_dir, responses_dir):
    rows = [{"prompt_id": "p1", "response": SECRET, "stop_reason": "eos", "generated_tokens": 30}]
    _jsonl(responses_dir / "nopreamble" / "responses" / OVERFIT / "harmful" / "en.jsonl", rows)

    data = build_data.build(overfit_dir, responses_dir=responses_dir)

    assert data["length"]["nopreamble"]["overfit"]["cut_off"] == [0, 1]
    assert "base" in data["length"]["nopreamble"]


def test_answer_lengths_of_an_added_model_are_left_out_without_its_responses(overfit_dir, responses_dir):
    data = build_data.build(overfit_dir, responses_dir=responses_dir)

    assert set(data["length"]["nopreamble"]) == {"base", "finetuned"}


def test_main_writes_the_same_bytes_twice(eval_dir, tmp_path):
    out = tmp_path / "out" / "data.json"
    assert build_data.main(["--eval-dir", str(eval_dir), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert build_data.main(["--eval-dir", str(eval_dir), "--out", str(out)]) == 0
    assert out.read_bytes() == first and json.loads(first)["schema"] == "driftmap.space/3"


def test_the_committed_data_file_holds_counts_only():
    data = json.loads((ROOT / "space" / "data.json").read_text(encoding="utf-8"))

    assert data["schema"] == "driftmap.space/3" and set(data) == KEYS and data["cells"]
    assert all(set(cell) == {"condition", "model", "language", "category", "k", "n", "lo", "hi"} for cell in data["cells"])
    # no cell names a prompt: the only strings are ids of conditions, models, languages and categories
    names = {"all", *data["languages"], *data["categories"]} | {row["id"] for row in data["conditions"] + data["models"]}
    assert {value for cell in data["cells"] for value in cell.values() if isinstance(value, str)} <= names
