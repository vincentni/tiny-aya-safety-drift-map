"""What the probe studies share: the two labels, a gate on stored answers, the counts and the intervals."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from driftmap.attribution import probe, probe_gates

TEXTS = {"A": "one person", "B": "another person", "C": "it cannot be told"}


def row(item, condition, base, tuned="C"):
    open_ = condition == "ambiguous"
    correct = "C" if open_ else "A"
    return {
        "id": f"{item}.{condition[:3]}",
        "prompt": f"prompt of {item}, {condition}",
        "condition": condition,
        "texts": TEXTS,
        "letter_unknown": "C",
        "letter_correct": correct,
        "pair": f"en/{item}",
        "cluster": item,
        "base": base,
        "fine_tuned": tuned,
        "ambiguous": open_,
        "base_wrong": base != correct,
    }


def four_items():
    """one: wrong on the open row; two: abstains on the settled row; three: both right; four: no letter, then B."""
    answers = {"one": ("A", "A"), "two": ("C", "C"), "three": ("C", "A"), "four": (None, "B")}
    return [
        row(item, condition, letter)
        for item, pair in answers.items()
        for condition, letter in zip(("ambiguous", "disambiguated"), pair, strict=True)
    ]


def gate(fires_on, rows, threshold=0.5):
    """A gate that reaches its threshold on the rows named, at different heights."""
    heights = dict(zip(fires_on, (0.9, 0.8, 0.7, 0.6, 0.5), strict=False))
    return {"scores": {r["prompt"]: heights.get(r["id"], 0.1) for r in rows}, "threshold": threshold}


def test_a_gate_gives_the_unknown_letter_where_its_score_reaches_the_threshold_and_the_stored_letter_elsewhere():
    rows = four_items()
    spec = gate(["one.amb", "x", "x", "x", "one.dis"], rows)

    assert spec["scores"][rows[1]["prompt"]] == spec["threshold"]
    assert [probe_gates.gated(r, spec) for r in rows[:4]] == ["C", "C", "C", "C"]
    assert [probe_gates.gated(r, spec) for r in rows[4:]] == ["C", "A", None, "B"]
    assert probe_gates.letter_of("The answer is B.", {"texts": TEXTS}) == "B"
    options = [{"letter": letter, "text": text} for letter, text in TEXTS.items()]
    assert probe_gates.letter_of("C", {"options": options}) == "C" and probe_gates.letter_of(None, rows[0]) is None


def test_the_counts_of_a_system_are_over_open_rows_settled_rows_twins_and_all_rows():
    found = probe_gates.measured(four_items(), lambda r: r["base"])

    assert found == {
        "ambiguous_rows": 4,
        "names_a_person": 1,
        "names_a_person_rate": 0.25,
        "disambiguated_rows": 4,
        "disambiguated_correct": 2,
        "disambiguated_accuracy": 0.5,
        "disambiguated_answered_with_a_person": 3,
        "answered_with_a_person_rate": 0.75,
        "disambiguated_answered_unknown": 1,
        "pairs": 4,
        "twin_both_right": 1,
        "twin_accuracy": 0.25,
        "rows": 8,
        "wrong_answers": 4,
        "wrong_answer_rate": 0.5,
        "unparseable": 1,
    }
    assert probe_gates.measured([], lambda r: r["base"])["twin_accuracy"] is None
    # a row without its twin counts as a row and not as a pair
    lone = probe_gates.measured(four_items()[:3], lambda r: r["base"])
    assert (lone["rows"], lone["pairs"], lone["wrong_answers"]) == (3, 1, 1)


def test_the_two_labels_are_related_by_their_two_by_two_count_and_phi():
    rows = [row(f"i{at}", "ambiguous", "A" if at < 3 else "C") for at in range(4)]
    rows += [row(f"i{at}", "disambiguated", "B" if at < 1 else "A") for at in range(4)]

    assert probe_gates.relation(rows) == {
        "ambiguous_base_wrong": 3,
        "ambiguous_base_right": 1,
        "disambiguated_base_wrong": 1,
        "disambiguated_base_right": 3,
        "labels_agree": 6,
        "labels_differ": 2,
        "phi": 0.5,
    }
    assert probe_gates.relation(rows[:4])["phi"] is None
    assert [probe_gates.label_of(rows[3], label) for label in probe_gates.LABELS] == [True, False]
    assert [probe_gates.label_of(rows[4], label) for label in probe_gates.LABELS] == [False, True]


def test_a_change_is_counted_per_base_item_and_fewer_wrong_answers_are_better():
    rows = four_items()
    spec = gate(["one.amb", "four.amb", "three.dis"], rows)
    first, second = (lambda r: r["base"]), (lambda r: probe_gates.gated(r, spec))

    wrong = probe_gates.change(rows, first, second, "wrong_answers")
    twins = probe_gates.change(rows, first, second, "twin_accuracy")

    # wrong answers per item go 1, 1, 0, 2 -> 0, 1, 1, 1 over 8 rows; twins both right 0, 0, 1, 0 -> 1, 0, 0, 0
    assert (wrong["clusters"], wrong["change"], wrong["clusters_better"], wrong["clusters_worse"]) == (4, -0.125, 2, 1)
    assert (twins["clusters"], twins["change"], twins["clusters_better"], twins["clusters_worse"]) == (4, 0.0, 1, 1)
    assert wrong["sign_test_p"] == 1.0 and twins["sign_test_p"] == 1.0
    assert wrong["ci95"] == probe_gates.interval(np.array([-1.0, 0, 1, -1]), np.array([2.0, 2, 2, 2]))
    assert wrong["ci95"][0] < -0.125 < wrong["ci95"][1] and twins["ci95"][0] < 0 < twins["ci95"][1]
    same = probe_gates.change(rows, first, first, "wrong_answers")
    assert (same["change"], same["ci95"], same["clusters_better"], same["clusters_worse"]) == (0.0, [0.0, 0.0], 0, 0)
    assert probe_gates.change(rows[:2], first, second, "wrong_answers") is None
    # an item with one row of a pair has no twin to count
    assert probe_gates.change(rows[:5], first, second, "twin_accuracy")["clusters"] == 2


def test_the_auc_difference_is_resampled_by_base_item_and_leaves_out_resamples_of_one_kind():
    kinds = {"ambiguous": "C", "disambiguated": "A"}
    rows = [row(f"i{at}", condition, right) for at in range(12) for condition, right in kinds.items()]
    rng = np.random.default_rng(3)
    good = {r["prompt"]: float(r["ambiguous"]) + rng.normal(scale=0.2) for r in rows}
    poor = {r["prompt"]: float(r["ambiguous"]) + rng.normal(scale=1.5) for r in rows}
    truth = [r["ambiguous"] for r in rows]
    aucs = [roc_auc_score(truth, [scores[r["prompt"]] for r in rows]) for scores in (good, poor)]

    found = probe_gates.auc_difference(rows, "answerability", good, poor)

    assert found["difference"] == round(aucs[0] - aucs[1], 4) and found["difference"] > 0
    assert found["ci95"][0] <= found["difference"] <= found["ci95"][1] and found["resamples_left_out"] == 0
    assert found == probe_gates.auc_difference(rows, "answerability", good, poor)
    back = probe_gates.auc_difference(rows, "answerability", poor, good)
    assert back["difference"] == -found["difference"]
    assert probe_gates.auc(truth, [good[r["prompt"]] for r in rows]) == round(aucs[0], 4)
    # the base model is right on every row, so its error is one kind alone
    assert probe_gates.auc_difference(rows, "base_error", good, poor) is None
    assert probe_gates.auc([True, True], [0.1, 0.9]) is None
    # with two items of one kind each, a resample that draws one item twice holds one kind alone
    two = [row("a", "ambiguous", "C"), row("b", "disambiguated", "A")]
    flat = {r["prompt"]: 0.5 for r in two}
    left_out = probe_gates.auc_difference(two, "answerability", flat, flat)["resamples_left_out"]
    assert 800 < left_out < 1200


def test_every_set_is_scored_under_every_gate_and_the_rows_written_hold_no_text():
    rows = four_items()
    gates = {"reader": gate(["one.amb", "four.amb", "three.dis"], rows), "silent": gate([], rows)}
    sets = {"plain": rows, "whole": rows}

    found = probe_gates.score_sets(sets, gates, [("base", "reader"), ("reader", "silent")])

    plain = found["plain"]
    assert list(found) == ["plain", "whole"] and plain == found["whole"] and plain["rows"] == 8
    assert plain["labels"] == probe_gates.relation(rows)
    assert plain["gates"]["reader"] == {
        "threshold": 0.5,
        "auc_answerability": 0.6875,
        "auc_base_error": 0.6875,
        "fired_on_ambiguous": [2, 4],
        "fired_on_disambiguated": [1, 4],
        "fired_on_base_wrong": [2, 4],
        "fired_on_base_right": [1, 4],
        "fired_on_ambiguous_base_right": [0, 2],
        "fired_on_disambiguated_base_wrong": [0, 2],
    }
    assert plain["gates"]["silent"]["fired_on_ambiguous"] == [0, 4]
    assert list(plain["systems"]) == ["base", "fine_tuned", "reader", "silent"]
    assert plain["systems"]["silent"] == plain["systems"]["base"]
    assert plain["systems"]["fine_tuned"]["names_a_person"] == 0
    assert (plain["systems"]["reader"]["wrong_answers"], plain["systems"]["reader"]["twin_both_right"]) == (3, 1)
    assert list(plain["changes"]) == ["base -> reader", "reader -> silent"]
    assert plain["changes"]["base -> reader"]["wrong_answers"]["change"] == -0.125
    assert plain["changes"]["reader -> silent"]["wrong_answers"]["change"] == 0.125

    written = probe_gates.answer_rows(sets, gates, whole=("whole",))

    assert [r["id"] for r in written] == [r["id"] for r in rows] and {r["set"] for r in written} == {"plain"}
    assert written[0] == {
        "set": "plain",
        "id": "one.amb",
        "condition": "ambiguous",
        "letter_unknown": "C",
        "letter_correct": "C",
        "letter_base": "A",
        "letter_fine_tuned": "C",
        "base_wrong": True,
        "reader_score": 0.9,
        "reader_fired": True,
        "letter_base_plus_reader": "C",
        "silent_score": 0.1,
        "silent_fired": False,
        "letter_base_plus_silent": "A",
    }
    text = json.dumps(written)
    assert "prompt of" not in text and not any(said in text for said in TEXTS.values())


def test_stored_sets_gain_their_letters_and_labels_and_a_stored_probe_reads_the_stored_states(tmp_path):
    options = [{"letter": letter, "text": text} for letter, text in TEXTS.items()]
    asked = [
        {"id": f"i{at}.{condition[:3]}", "prompt": f"prompt {at} {condition}", "condition": condition}
        | {"options": options, "letter_unknown": "C", "letter_correct": "C" if condition == "ambiguous" else "A"}
        | {"base_item": f"i{at}", "language": "en"}
        for at in range(3)
        for condition in ("ambiguous", "disambiguated")
    ]
    for name, said in (("base", "The answer is A."), ("tuned", "C")):
        lines = "".join(json.dumps(r | {"response": said}) + "\n" for r in asked)
        (tmp_path / f"{name}.jsonl").write_text(lines, encoding="utf-8")
    spec = {"sets": [{"name": "plain", "base": "base.jsonl", "fine_tuned": "tuned.jsonl"}]}
    spec["sets"].append({"name": "twice", "of": ["plain", "plain"]})
    (tmp_path / "sets.json").write_text(json.dumps(spec), encoding="utf-8")

    sets = probe_gates.load_sets(tmp_path / "sets.json")

    first, second = sets["plain"][:2]
    assert (first["base"], first["fine_tuned"], first["ambiguous"], first["base_wrong"]) == ("A", "C", True, True)
    assert (second["base"], second["ambiguous"], second["base_wrong"]) == ("A", False, False)
    assert len(sets["twice"]) == 12 and sets["twice"][6] is first

    # the probe reads the second unit of the second layer; a state above its mean scores over one half
    states = np.zeros((6, 2, 3), dtype=np.float32)
    states[:, 1, 1] = [3, -3, 2, -2, 1, -1]
    np.savez(tmp_path / "0.npz", ids=np.array([r["id"] for r in asked]), states=states, layers=np.array([4, 8]))
    arrays = (np.zeros(3), np.ones(3), np.array([0.0, 1.0, 0.0]), np.zeros(1))
    probe.save(probe.Probe(8, 0.1, 0.6, *(a.astype(np.float32) for a in arrays)), tmp_path / "frozen")
    (tmp_path / "acts").mkdir()
    (tmp_path / "0.npz").rename(tmp_path / "acts" / "0.npz")
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in asked), encoding="utf-8")

    scores, threshold = probe_gates.activation_scores(tmp_path / "frozen", tmp_path / "rows.jsonl", tmp_path / "acts")

    assert threshold == 0.6 and list(scores) == [r["prompt"] for r in asked]
    assert [scores[r["prompt"]] > 0.5 for r in asked] == [True, False] * 3
    assert scores[asked[0]["prompt"]] == pytest.approx(1 / (1 + np.exp(-3)), abs=1e-6)
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in asked[:-1]), encoding="utf-8")
    with pytest.raises(SystemExit, match="6 hidden states for 5 rows of rows.jsonl"):
        probe_gates.activation_scores(tmp_path / "frozen", tmp_path / "rows.jsonl", tmp_path / "acts")
