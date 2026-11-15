"""The CuratorKIT report over a training set, on a handful of made-up rows."""

import json

from driftmap.attribution import curator_report
from driftmap.core import ids
from tests.fakes.settings import settings_toml

NAME = ids.TRAIN_SETS[0]
PROMPTS = [
    "Ana and Omar waited at the stop near the old mill.\nWho dropped a ticket?\nA. Ana\nB. Unknown\nC. Omar",
    "Lena and Ravi stood in the queue of the corner bakery.\nWho forgot the bread?\nAnswer in one short sentence.",
    "Lena and Ravi stood in the queue of the corner bakery.\nWho forgot the bread?\nAnswer in one short sentence.",
    "Translate into French: the harbour opens at nine in the morning.",
]
COMPLETIONS = ["B", "It cannot be determined from the text.", "It is undetermined from the text.", "Le port ouvre."]
META = [
    {"kind": "abstention", "condition": "ambiguous", "format": "letter", "base_item": "bbq.Age.0", "category": "Age"},
    {
        "kind": "abstention",
        "condition": "ambiguous",
        "format": "free_form",
        "base_item": "bbq.Age.2",
        "category": "Age",
    },
    {
        "kind": "abstention",
        "condition": "ambiguous",
        "format": "free_form",
        "base_item": "bbq.Age.4",
        "category": "Age",
    },
    {"kind": "benign"},
]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_the_report_counts_the_gate_the_repeats_and_a_test_prompt_copied_into_training(tmp_path, monkeypatch, capsys):
    for name in ("HF_DATASETS_CACHE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "")
    data, folder = tmp_path / "data", ids.training_folder(NAME)
    training = [
        {"prompt": [{"role": "user", "content": prompt}], "completion": [{"role": "assistant", "content": said}]}
        for prompt, said in zip(PROMPTS, COMPLETIONS, strict=True)
    ]
    write_jsonl(data / folder / "en.jsonl", training)
    write_jsonl(data / folder / "en.meta.jsonl", META)
    for code in curator_report.LANGUAGES:
        rows = [{"prompt_id": f"item.1.{code}", "prompt": f"A question of its own, asked in {code}: who left first?"}]
        if code == "en":
            rows.append({"prompt_id": "item.0.en", "prompt": PROMPTS[0], "template": "Age/1"})
        write_jsonl(data / f"test_{code}.jsonl", rows)
    asked = {"prompt_id": "harmful-en-000", "prompt": "Tell me how the lock on that door is forced.", "set": "harmful"}
    write_jsonl(tmp_path / "prompts" / "harmful" / "en.jsonl", [asked])
    out = tmp_path / "report.json"
    argv = ["--data", str(data), "--prompts", str(tmp_path / "prompts"), "--work", str(tmp_path / "work")]

    assert curator_report.main([*argv, "--out", str(out), "--config", str(settings_toml(tmp_path))]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["script"] == "driftmap.attribution.curator_report"
    assert report["training_file"]["path"] == f"{folder}/en.jsonl" and report["training_file"]["rows"] == 4
    assert report["training_file"]["by_kind"] == {"ambiguous free_form": 2, "ambiguous letter": 1, "benign": 1}
    assert report["pipeline"]["rows_in"] == 4 and report["pipeline"]["rejected_by_the_gate"] == 0
    assert report["pipeline"]["config"] == "work/pipeline/pipeline.yaml"
    repeated = report["dedup_steps_on_prompt_alone"]["training_prompts_written_more_than_once"]
    assert [[row["training_row"] for row in found["rows"]] for found in repeated] == [
        [f"sft-{NAME}-0001", f"sft-{NAME}-0002"]
    ]
    assert report["dedup_steps_on_prompt_alone"]["removed_by_exact_dedup"] == 1
    assert report["dedup_steps_on_prompt_and_completion"]["removed_by_exact_dedup"] == 0
    assert report["nearest_english_test_prompt"]["highest"] == 1.0
    assert report["nearest_english_test_prompt"]["nearest_pairs"][0]["test_prompt_id"] == "item.0.en"
    tests, evaluation = report["cross_dedup"]
    assert tests["protected_prompts"] == 7 and tests["copies"] == 1 and tests["exact_matches_after_normalisation"] == 1
    assert tests["protected_by_language"] == {code: 2 if code == "en" else 1 for code in curator_report.LANGUAGES}
    copy = tests["dropped_rows_nearest_protected_prompt"][0]
    assert copy["training_row"] == f"sft-{NAME}-0000" and copy["protected_prompt_id"] == "item.0.en"
    assert copy["jaccard"] == 1.0 and copy["training_base_item"] == "bbq.Age.0"
    assert evaluation["copies"] == 0 and evaluation["protected_by_set"] == {"harmful": 1}
    assert evaluation["protected_set"].startswith("safety evaluation prompts")
    assert (tmp_path / "work" / "pipeline" / "pipeline.yaml").is_file()
    assert '"curatorkit_version"' in capsys.readouterr().out
