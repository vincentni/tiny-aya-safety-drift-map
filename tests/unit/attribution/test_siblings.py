"""Held-out templates and the trained templates that state the same stereotype."""

import json

import pytest

from driftmap.attribution import bbq, build, siblings
from tests.fakes import bbq as fake_bbq

STATED = {
    "Age/1": ("thrift", "old riders"),
    "Age/2": ("thrift, punctuality", "old riders"),
    "Age/3": ("tidiness", "old riders"),
    "Age/4": ("thrift", "young riders"),
    "Age/5": ("thrift", "old riders"),
    "SES/1": ("thrift", "old riders"),
}


def test_a_sibling_states_the_same_value_or_shares_a_term_of_it_for_the_same_groups_in_its_category():
    trained = {name: STATED[name] for name in ("Age/1", "Age/2", "Age/4", "SES/1")}

    same = siblings.siblings("Age/5", STATED, trained)
    near = siblings.siblings("Age/2", STATED, {name: STATED[name] for name in ("Age/1", "Age/4", "SES/1")})
    none = siblings.siblings("Age/3", STATED, trained)

    assert same["status"] == "same stereotype" and same["trained_templates_same"] == ["Age/1"]
    assert same["trained_templates_sharing_a_term"] == ["Age/2"]
    assert near["status"] == "shares a term" and near["trained_templates_sharing_a_term"] == ["Age/1"]
    assert none == {
        "social_value": "tidiness",
        "stereotyped_groups": "old riders",
        "status": "none",
        "trained_templates_same": [],
        "trained_templates_sharing_a_term": [],
    }
    assert siblings.terms("thrift; punctuality / tidiness, ") == {"thrift", "punctuality", "tidiness"}


def test_twin_accuracy_counts_a_base_item_whose_two_rows_are_both_read_right():
    options = [{"letter": letter, "text": text} for letter, text in zip("ABC", ("Ana", "Unknown", "Omar"), strict=True)]
    asked = {
        f"item.{number}.{condition}": {
            "base_item": f"item.{number}",
            "condition": condition,
            "letter_correct": "B" if condition == "ambiguous" else "C",
            "language": "en",
            "options": options,
            "template": "Age/1" if number < 2 else "Age/2",
        }
        for number in range(3)
        for condition in ("ambiguous", "disambiguated")
    }
    said = {"item.0": ("B", "C."), "item.1": ("B", "Ana"), "item.2": ("unknown", "The answer is C")}
    answers = [
        {"prompt_id": f"{item}.{condition}", "response": response}
        for item, pair in said.items()
        for condition, response in zip(("ambiguous", "disambiguated"), pair, strict=True)
    ]
    answers.append({"prompt_id": "item.9.ambiguous", "response": "B"})

    # the same ids in another language are answers to other prompts: they change nothing
    answers += [{"prompt_id": answer["prompt_id"], "language": "fr", "response": "A"} for answer in answers]

    every = siblings.twin(answers, asked, lambda row: True)
    first = siblings.twin(answers, asked, lambda row: row["template"] == "Age/1")

    assert every == {
        "base_items": 3,
        "twin_both_right": 2,
        "twin_accuracy": 0.667,
        "ambiguous_right": 3,
        "disambiguated_right": 2,
    }
    assert first["base_items"] == 2 and first["twin_both_right"] == 1 and first["twin_accuracy"] == 0.5


def test_the_counts_are_added_to_the_report_beside_what_it_holds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bbq, "FILES", fake_bbq.write(tmp_path / "bbq"))
    (tmp_path / "benign.jsonl").write_text("", encoding="utf-8")
    data = tmp_path / "data"
    made = ["--bbq", str(tmp_path / "bbq"), "--benign", str(tmp_path / "benign.jsonl"), "--out", str(data)]
    assert build.main(made) == 0
    rows = [json.loads(line) for line in (data / "test_en.jsonl").read_text(encoding="utf-8").split("\n") if line]
    model = "owner/model@0123456789ab"
    right = [{"prompt_id": row["prompt_id"], "response": row["letter_correct"], "checkpoint": model} for row in rows]
    unknown = [{"prompt_id": row["prompt_id"], "response": f"{row['letter_unknown']}."} for row in rows]
    for name, answers in (("right", right), ("unknown", unknown)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"curatorkit_version": "x"}), encoding="utf-8")
    argv = ["--bbq", str(tmp_path / "bbq"), "--manifest", str(data / "manifest.json")]
    argv += ["--test", str(data / "test_en.jsonl"), "--report", str(report)]
    argv += ["--answers", f"right={tmp_path / 'right.jsonl'}", "--answers", f"unknown={tmp_path / 'unknown.jsonl'}"]

    assert siblings.main(argv) == 0

    whole = json.loads(report.read_text(encoding="utf-8"))
    found = whole["heldout_templates_with_trained_sibling"]
    assert whole["curatorkit_version"] == "x" and found["script"] == "driftmap.attribution.siblings"
    assert found["heldout_template_base_items"] == 75
    split = found["templates"]
    assert split["same or shares a term"]["templates"] + split["no trained sibling"]["templates"] == len(
        found["per_template"]
    )
    assert split["same or shares a term"]["base_items"] + split["no trained sibling"]["base_items"] == 75
    assert found["answers_used"] == {
        "right": {"file": "right.jsonl", "checkpoint": model},
        "unknown": {"file": "unknown.jsonl", "checkpoint": []},
    }
    assert str(tmp_path) not in report.read_text(encoding="utf-8")
    for group, count in split.items():
        scored = found["twin_accuracy_plain_order_0"]["right"][group]
        assert scored["base_items"] == scored["twin_both_right"] == count["base_items"]
        abstained = found["twin_accuracy_plain_order_0"]["unknown"][group]
        assert abstained["ambiguous_right"] == count["base_items"] and abstained["twin_both_right"] == 0
    assert "heldout_templates_in_test" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        siblings.main([*argv, "--answers", "no-file-named"])
