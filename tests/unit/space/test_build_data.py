import importlib.util
import json
from pathlib import Path

import pytest

from scipy.stats import binomtest

from driftmap.table.rates import wilson as reference_wilson

ROOT = Path(__file__).resolve().parents[3]
BASE, TUNED = "tiny-aya-global--af89d219b53e", "tiny-aya-global-drift-merged--daccc14f6c19"
OVERFIT = "tiny-aya-global-drift-merged-nopre--d845b23452bb"
REPAIRED = "tiny-aya-global-repaired-nopre--ba1563a2e8c8"
SECRET = "text of a harmful request that must stay out of the page"
KEYS = {"schema", "eval_repo", "conditions", "models", "languages", "categories", "figures"}
KEYS |= {"summary", "changes", "flips", "length", "cells", "overfit", "repaired", "patch", "variants"}


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


@pytest.mark.parametrize(("k", "n"), [(16, 99), (2, 100), (55, 598)])
def test_the_stored_bounds_print_as_the_tables_print_them(k, n):
    stored = build_data._rate(k, n)[2:]

    assert [f"{100 * bound:.1f}" for bound in stored] == [f"{100 * bound:.1f}" for bound in reference_wilson(k, n)]


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
    assert data["overfit"] is None and data["repaired"] is None and data["patch"] == [] and data["variants"] == []


def test_the_overfit_fine_tune_is_counted_in_the_condition_it_has_labels_in(overfit_dir, tmp_path):
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
    assert data["overfit"]["model"] == "overfit"
    (run,) = data["overfit"]["runs"]
    assert run["condition"] == "nopreamble"
    # the base model is unscored on p3, so two prompts pair up: one harmful for both, one newly harmful
    assert run["flips"] == {"pairs": 2, "harmful_both": 1, "harmful_to_not": 0, "not_to_harmful": 1}
    assert run["changes"]["safe"] == {"delta": -0.0401, "ci": [-0.0702, -0.0117], "p": 0.007, "pairs": 598}
    # the two models of every condition keep their own comparison
    assert data["flips"]["nopreamble"]["pairs"] == 2 and data["changes"]["nopreamble"] == {}


def test_the_overfit_fine_tune_is_compared_with_the_base_model_in_each_condition(overfit_dir, tmp_path):
    _harmful_file(overfit_dir, "labels", OVERFIT, [("p1", "yes", "no"), ("p2", "no", "yes"), ("p3", "no", "yes")])
    change = {"row": "all", "a": "baseline", "b": "drifted", "metric": "safe", "n_pairs": 598, "ci": [-0.07, -0.01]}
    for folder, name, delta in (("nopreamble/tables", "tables-nopreamble-aggressive.json", -0.0401),
                                ("tables", "tables-aggressive.json", -0.0486)):  # fmt: skip
        (tmp_path / "results" / folder).mkdir(parents=True)
        tables = {"deltas": [change | {"delta": delta, "p_value": 0.007}]}
        (tmp_path / "results" / folder / name).write_text(json.dumps(tables), encoding="utf-8")

    data = build_data.build(overfit_dir, tmp_path / "results")

    # the condition the repair is measured in comes first
    served = data["overfit"]["runs"][1]
    assert [run["condition"] for run in data["overfit"]["runs"]] == ["nopreamble", "served"]
    assert data["summary"]["served"]["overfit"]["harmful"][:2] == [2, 3]
    cells = {(c["condition"], c["model"], c["language"], c["category"]): (c["k"], c["n"]) for c in data["cells"]}
    assert cells["served", "overfit", "en", "all"] == (2, 3)
    # served template: p1 harmful for the base model only, p2 for the overfit fine-tune only, p3 unscored for the base
    assert served["flips"] == {"pairs": 2, "harmful_both": 0, "harmful_to_not": 1, "not_to_harmful": 1}
    # each condition reads its own tables file
    assert [run["changes"]["safe"]["delta"] for run in data["overfit"]["runs"]] == [-0.0401, -0.0486]
    assert [row["id"] for row in data["models"]] == ["base", "finetuned", "overfit"]


def test_the_overfit_fine_tune_under_the_served_template_alone_is_listed_without_a_repair(eval_dir):
    _harmful_file(eval_dir, "labels", OVERFIT, [("p1", "no", "yes")])
    _harmful_file(eval_dir, "nopreamble/labels", REPAIRED, [("p1", "yes", "no")])

    data = build_data.build(eval_dir)

    assert [run["condition"] for run in data["overfit"]["runs"]] == ["served"]
    assert data["repaired"] is None and [row["id"] for row in data["models"]] == ["base", "finetuned", "overfit"]


def test_the_repaired_fine_tune_is_compared_with_the_overfit_one_and_with_the_base_model(overfit_dir, tmp_path):
    _harmful_file(overfit_dir, "nopreamble/labels", REPAIRED, [("p1", "yes", "no"), ("p2", "no", "yes"), ("p3", "yes", "no")])
    drift = {"row": "all", "a": "baseline", "b": "drifted", "metric": "safe", "n_pairs": 598}
    drift |= {"delta": -0.0401, "ci": [-0.0702, -0.0117], "p_value": 0.007}
    repair = drift | {"a": "drifted", "b": "repaired", "delta": 0.0569, "ci": [0.0268, 0.0853], "p_value": 0.0001}
    results = tmp_path / "results" / "nopreamble" / "tables"
    results.mkdir(parents=True)
    tables = {"deltas": [drift, repair, repair | {"row": "en"}]}
    (results / "tables-nopreamble-aggressive.json").write_text(json.dumps(tables), encoding="utf-8")

    data = build_data.build(overfit_dir, tmp_path / "results")

    assert [row["id"] for row in data["models"]] == ["base", "finetuned", "overfit", "repaired"]
    assert data["summary"]["nopreamble"]["repaired"]["harmful"][:2] == [1, 3]
    repaired = data["repaired"]
    assert (repaired["condition"], repaired["model"]) == ("nopreamble", "repaired")
    # overfit: all three harmful; repaired: p2 alone; the base model has no verdict on p3
    assert repaired["flips"] == {"pairs": 3, "harmful_both": 1, "harmful_to_not": 2, "not_to_harmful": 0}
    assert repaired["flips_base"] == {"pairs": 2, "harmful_both": 0, "harmful_to_not": 1, "not_to_harmful": 1}
    # each comparison reads its own pair of columns in the tables
    assert repaired["changes"]["safe"] == {"delta": 0.0569, "ci": [0.0268, 0.0853], "p": 0.0001, "pairs": 598}
    assert data["overfit"]["runs"][0]["changes"]["safe"]["delta"] == -0.0401


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
        "harmful": [1, 1], "refusal": [1, 0], "answered": [0, 1], "p": 1.0, "request_harm": [2, 2], "left_out": 0,
    }  # fmt: skip
    text = json.dumps(data)
    assert SECRET not in text and "v1" not in text


def test_a_variant_the_judge_reads_as_another_request_is_left_out_as_the_tables_do(variants_dir):
    path = variants_dir / "variants" / "labels" / TUNED / "en-Latn.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["request_harm"] = "no"
    _jsonl(path, rows)

    (row,) = build_data.build(variants_dir)["variants"]

    # v1 no longer reads as the harmful request its parent is; v2 is the one pair left
    assert (row["left_out"], row["pairs"], row["harmful"], row["refusal"]) == (1, 1, [1, 0], [0, 0])


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
    assert out.read_bytes() == first and json.loads(first)["schema"] == "driftmap.space/5"


def test_the_committed_data_file_holds_counts_only():
    data = json.loads((ROOT / "space" / "data.json").read_text(encoding="utf-8"))

    assert data["schema"] == "driftmap.space/5" and set(data) == KEYS and data["cells"]
    assert all(set(cell) == {"condition", "model", "language", "category", "k", "n", "lo", "hi"} for cell in data["cells"])
    # no cell names a prompt: the only strings are ids of conditions, models, languages and categories
    names = {"all", *data["languages"], *data["categories"]} | {row["id"] for row in data["conditions"] + data["models"]}
    assert {value for cell in data["cells"] for value in cell.values() if isinstance(value, str)} <= names


LONG = "harmful answer " * 40


@pytest.fixture
def replay_dirs(eval_dir, tmp_path):
    """Four prompts in two languages, answered by the base model with and without the patch, by the fine-tuned
    model and by the overfit fine-tune; the fine-tuned model under the patch has labels but no response files."""
    prompts = {
        "en": [("a" * 64, '["Hate Speech"]'), ("b" * 64, '["Profanity"]'), ("c" * 64, '["Profanity"]')],
        "fr": [("d" * 64, '["Self-Harm"]')],
    }
    for language, rows in prompts.items():
        _jsonl(
            eval_dir / "prompts" / "harmful" / f"{language}.jsonl",
            [{"prompt_id": key, "language": language, "prompt": f"ask {key[0]}", "harm_category": tags} for key, tags in rows],
        )
    a, b, c, d = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    refused = dict.fromkeys((a, b, c, d), ("yes", "no"))
    labels = {
        ("labels", BASE): {a: ("no", "yes"), b: ("yes", "no"), c: ("yes", "no"), d: ("no", "no")},
        ("patched/labels", BASE): {a: ("yes", "no"), b: ("yes", "no"), c: ("yes", "no"), d: ("no", "no")},
        ("nopreamble/labels", OVERFIT): {a: ("no", "yes"), b: ("no", "yes"), c: ("yes", "no"), d: ("unscored", "unscored")},
        ("nopreamble/labels", REPAIRED): refused,
        ("labels", TUNED): refused,
        ("patched/labels", TUNED): refused,
        ("nopreamble/labels", BASE): refused,
        ("nopreamble/labels", TUNED): refused,
    }
    answers = tmp_path / "replay-answers"
    for (tree, folder), rows in labels.items():
        for language in prompts:
            keys = [key for key, _ in prompts[language]]
            _harmful_file(eval_dir, tree, folder, [(key, *rows[key]) for key in keys], language)
            if (tree, folder) == ("patched/labels", TUNED):
                continue
            responses = [
                {"prompt_id": key, "response": f"  {LONG}{key[0]} ", "generated_tokens": 256 if key == a else 12}
                | {"stop_reason": "length" if key == a else "eos"}
                for key in keys
            ]
            _jsonl(answers / tree.replace("labels", "responses") / folder / "harmful" / f"{language}.jsonl", responses)
    return eval_dir, answers


def test_the_explorer_replays_every_system_that_has_labels_and_responses(replay_dirs):
    replay = build_data.explorer(*replay_dirs)

    assert replay["schema"] == "driftmap.space.explorer/1"
    assert [(row["id"], row["name"], row["condition"]) for row in replay["systems"]] == [
        ("served/base", "Base model", "served template"),
        ("patched/base", "Base model + safety patch", "safety patch"),
        ("served/finetuned", "Light fine-tune", "served template"),
        ("nopreamble/overfit", "Overfit fine-tune", "no safety preamble"),
        ("nopreamble/repaired", "Repaired fine-tune", "no safety preamble"),
    ]
    notes = [None, None, None, "research artifact, not for deployment", None]
    assert [row.get("note") for row in replay["systems"]] == notes
    first = next(row for row in replay["prompts"] if row["id"] == "a" * 12)
    assert (first["language"], first["categories"], first["text"]) == ("en", ["Hate Speech"], "ask a")
    systems = {"served/base", "patched/base", "served/finetuned", "nopreamble/overfit", "nopreamble/repaired"}
    assert set(first["answers"]) == systems
    assert first["answers"]["patched/base"] == {
        "verdict": "refused", "cut_off": True, "text": f"{LONG}a", "clipped": False,
    }  # fmt: skip


def test_with_harmful_text_an_answer_judged_harmful_or_left_unscored_keeps_its_first_300_characters(replay_dirs):
    replay = build_data.explorer(*replay_dirs, include_harmful=True)

    rows = {row["id"][0]: row["answers"] for row in replay["prompts"]}
    harmful, unscored = rows["a"]["served/base"], rows["d"]["nopreamble/overfit"]
    assert harmful == {"verdict": "harmful", "cut_off": True, "text": LONG[:300], "clipped": True}
    assert unscored == {"verdict": "unscored", "cut_off": False, "text": LONG[:300], "clipped": True}
    assert (replay["cut"], replay["harmful_text"]) == (300, True)


def test_the_explorer_leaves_out_the_text_of_harmful_and_unscored_answers_unless_asked(replay_dirs):
    replay = build_data.explorer(*replay_dirs)

    rows = {row["id"][0]: row["answers"] for row in replay["prompts"]}
    assert replay["harmful_text"] is False
    assert rows["a"]["served/base"] == {"verdict": "harmful", "cut_off": True, "text": None, "clipped": False}
    assert rows["d"]["nopreamble/overfit"]["text"] is None
    assert rows["a"]["patched/base"]["text"] == f"{LONG}a"
    assert LONG[:40] not in json.dumps(rows["a"]["served/base"])


def test_prompts_where_the_systems_disagree_come_first_under_the_cap(replay_dirs):
    full = build_data.explorer(*replay_dirs)

    # a: harmful for the base model and the overfit fine-tune; b: harmful for the overfit fine-tune only;
    # c: refused by all; d: answered or refused, and one answer without a verdict
    assert {row["id"][0]: row["disagree"] for row in full["prompts"]} == {"a": True, "b": True, "c": False, "d": False}
    assert (full["total"], full["disagreeing"], len(full["prompts"])) == (4, 2, 4)

    capped = build_data.explorer(*replay_dirs, cap=3)
    # both disagreement prompts, then one more: the other language before a third English prompt
    assert sorted(row["id"][0] for row in capped["prompts"]) == ["a", "b", "d"]
    assert (capped["total"], capped["disagreeing"]) == (4, 2)
    assert [row["id"][0] for row in build_data.explorer(*replay_dirs, cap=1)["prompts"]] == ["a"]


def test_prompts_of_a_withheld_category_are_not_in_the_explorer(replay_dirs):
    eval_dir, answers = replay_dirs
    path = eval_dir / "prompts" / "harmful" / "en.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[2]["harm_category"] = '["Profanity", "Non-consensual sexual content"]'
    _jsonl(path, rows)

    replay = build_data.explorer(eval_dir, answers)

    assert sorted(row["id"][0] for row in replay["prompts"]) == ["a", "b", "d"]
    assert replay["left_out"] == {"category": "Non-consensual sexual content", "prompts": 1} and replay["total"] == 4
    assert "ask c" not in json.dumps(replay)


def test_prompts_every_system_answers_harmfully_come_before_the_rest(replay_dirs):
    eval_dir, answers = replay_dirs
    systems = (("labels", BASE), ("patched/labels", BASE), ("labels", TUNED))
    for tree, folder in (*systems, ("nopreamble/labels", OVERFIT), ("nopreamble/labels", REPAIRED)):
        rows = [("a" * 64, "no", "yes"), ("b" * 64, "yes", "no"), ("c" * 64, "no", "yes")]
        _harmful_file(eval_dir, tree, folder, rows)

    replay = build_data.explorer(eval_dir, answers, cap=1)

    # no system disagrees with another now; c and a are harmful for all five, and a comes first by id
    assert (replay["disagreeing"], replay["persistent"]) == (0, 2) and [row["id"][0] for row in replay["prompts"]] == ["a"]
    assert sorted(row["id"][0] for row in build_data.explorer(eval_dir, answers, cap=2)["prompts"]) == ["a", "c"]


def test_script_variants_are_replayed_under_their_parent_prompt(replay_dirs):
    eval_dir, answers = replay_dirs
    variant = {"prompt_id": "e" * 64, "parent_prompt_id": "c" * 64, "parent_language": "en", "language": "en-Latn"}
    variant |= {"set": "harmful", "variant": "romanized", "prompt": "ask c in Latin script", "harm_category": '["Profanity"]'}
    _jsonl(eval_dir / "variants" / "prompts" / "harmful" / "en-Latn.jsonl", [variant])
    _harmful_file(eval_dir, "variants/labels", BASE, [("e" * 64, "no", "yes")], "en-Latn")
    row = {"prompt_id": "e" * 64, "response": "short", "stop_reason": "eos", "generated_tokens": 3}
    _jsonl(answers / "variants" / "responses" / BASE / "harmful" / "en-Latn.jsonl", [row])

    replay = build_data.explorer(eval_dir, answers, include_harmful=True)

    parent = next(row for row in replay["prompts"] if row["id"] == "c" * 12)
    assert parent["variants"] == [
        {
            "code": "en-Latn", "kind": "romanized", "text": "ask c in Latin script",
            "answers": {"served/base": {"verdict": "harmful", "cut_off": False, "text": "short", "clipped": False}},
        }
    ]  # fmt: skip
    # the variant is answered harmfully where its parent is refused, so the prompt is one of the disagreements
    assert parent["disagree"] is True and replay["disagreeing"] == 3
    assert all(row["variants"] == [] for row in replay["prompts"] if row["id"] != "c" * 12)


def test_main_writes_the_explorer_beside_the_data_file_only_from_responses(replay_dirs, tmp_path):
    eval_dir, answers = replay_dirs
    out = tmp_path / "page" / "data.json"

    assert build_data.main(["--eval-dir", str(eval_dir), "--out", str(out)]) == 0
    assert not (out.parent / "explorer.json").exists()

    args = ["--eval-dir", str(eval_dir), "--responses-dir", str(answers), "--out", str(out), "--explorer-cap", "2"]
    assert build_data.main(args) == 0
    first = (out.parent / "explorer.json").read_bytes()
    assert build_data.main(args) == 0
    replay = json.loads(first)
    assert (out.parent / "explorer.json").read_bytes() == first and len(replay["prompts"]) == 2
    # the data file still holds counts only
    assert "ask a" not in out.read_text(encoding="utf-8") and LONG[:30] not in out.read_text(encoding="utf-8")
