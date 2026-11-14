import importlib.util
import json
from pathlib import Path

import pytest

from driftmap.table.rates import wilson as reference_wilson

ROOT = Path(__file__).resolve().parents[3]
BASE, TUNED = "tiny-aya-global--af89d219b53e", "tiny-aya-global-drift-merged--daccc14f6c19"
SECRET = "text of a harmful request that must stay out of the page"


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
        {"prompt_id": "p1", "language": "en", "prompt": SECRET, "global_or_local": "global",
         "harm_category": '["Hate Speech", "Self-Harm"]'},
        {"prompt_id": "p2", "language": "en", "prompt": SECRET, "global_or_local": "local",
         "harm_category": '["Hate Speech"]'},
        {"prompt_id": "p3", "language": "en", "prompt": SECRET, "global_or_local": "global",
         "harm_category": '["Profanity"]'},
    ]  # fmt: skip
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


def test_a_prompt_row_carries_ids_and_labels_and_no_text(eval_dir):
    data = build_data.build(eval_dir)

    first = next(row for row in data["prompts"] if row["id"] == "p1")
    assert first["language"] == "en" and first["categories"] == ["Hate Speech", "Self-Harm"] and first["scope"] == "global"
    assert first["labels"]["served"]["base"] == {"refusal": "no", "response_harm": "yes", "verdict": "harmful"}
    assert first["labels"]["nopreamble"]["finetuned"]["verdict"] == "refused"
    assert all(set(row) == {"id", "language", "categories", "scope", "labels"} for row in data["prompts"])
    assert SECRET not in json.dumps(data)


def test_pooled_changes_come_from_the_tables_when_given(eval_dir, tmp_path):
    change = {"row": "all", "a": "baseline", "b": "drifted", "metric": "safe", "n_pairs": 597}
    change |= {"delta": 0.0268, "ci": [0.0034, 0.0503], "p_value": 0.029}
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "tables.json").write_text(json.dumps({"deltas": [change, change | {"row": "en"}]}), encoding="utf-8")

    data = build_data.build(eval_dir, tables)

    assert data["changes"]["served"]["safe"] == {"delta": 0.0268, "ci": [0.0034, 0.0503], "p": 0.029, "pairs": 597}
    assert data["changes"]["nopreamble"] == {}


def test_main_writes_the_same_bytes_twice(eval_dir, tmp_path):
    out = tmp_path / "out" / "data.json"
    assert build_data.main(["--eval-dir", str(eval_dir), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert build_data.main(["--eval-dir", str(eval_dir), "--out", str(out)]) == 0
    assert out.read_bytes() == first and json.loads(first)["schema"] == "driftmap.space/1"


def test_the_committed_data_file_holds_no_prompt_or_response_text():
    data = json.loads((ROOT / "space" / "data.json").read_text(encoding="utf-8"))

    assert data["schema"] == "driftmap.space/1" and data["prompts"]
    assert all(set(row) == {"id", "language", "categories", "scope", "labels"} for row in data["prompts"])
    # ids are sha256 hex; a label is one of five short words
    assert all(len(row["id"]) == 64 and int(row["id"], 16) >= 0 for row in data["prompts"])
    words = {value for row in data["prompts"] for per in row["labels"].values() for labels in per.values()
             for value in labels.values()}  # fmt: skip
    assert words <= {"yes", "no", "unscored", "harmful", "refused", "answered"}
