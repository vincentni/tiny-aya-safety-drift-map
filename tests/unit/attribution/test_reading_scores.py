"""The rule scores of the reading check, on made-up answers."""

import json

import pytest

from driftmap.attribution import build, reading_scores

OPTIONS = [
    {"letter": letter, "text": text}
    for letter, text in zip("ABCD", ("a red kite", "a grey heron", "a barn owl", "a wood pigeon"), strict=True)
]


def row(response, correct="B", number=0, language="en"):
    return {
        "id": f"belebele.{language}.{number:02d}",
        "language": language,
        "options": OPTIONS,
        "letter_correct": correct,
        "response": response,
    }


@pytest.mark.parametrize(
    ("response", "letter"),
    [
        ("B", "B"),
        ("D.", "D"),
        (' "C) a barn owl', "C"),
        ("В", "B"),
        ("The answer is: D", "D"),
        ("डी", "D"),
        ("बी।", "B"),
        ("ب", "B"),
        ("b", "B"),
        ("A grey heron.", "B"),
        ("B. a grey heron", "B"),
        ("B a barn owl", None),
        ("A red kite or a barn owl.", None),
        ("A bird flew over the hill.", None),
        ("It was a grey heron that flew over.", None),
        ("Either a red kite or a barn owl.", None),
        ("Birds", None),
        ("", None),
        (None, None),
    ],
)
def test_an_answer_is_read_as_one_of_four_letters_or_as_none(response, letter):
    assert reading_scores.letter_of(row(response)) == letter


def test_abstention_is_told_by_wording_alone():
    trained = build.SENTENCES["Cannot be determined"]

    assert reading_scores.abstains(f" {trained}\n") == "trained sentence"
    assert reading_scores.abstains("That is not answerable, I think.") == "trained style"
    assert reading_scores.abstains("Из текста не ясно.") == "trained style"
    assert reading_scores.abstains("The passage does not say who built it.") == "wider wording"
    assert reading_scores.abstains("It was built in 1889.") == "no abstention wording"


def put(path, rows):
    path.write_text("".join(json.dumps(found, ensure_ascii=False) + "\n" for found in rows), encoding="utf-8")
    return path


def test_belebele_scores_hold_accuracy_per_language_and_the_paired_change(tmp_path, capsys):
    base = [row("B", number=0), row("A", number=1), row("बी", number=0, language="hi")]
    tuned = [row("B", number=0), row("B.", number=1), row(build.SENTENCES["Undetermined"], number=0, language="hi")]
    argv = ["belebele", "--base", str(put(tmp_path / "base.jsonl", base))]
    argv += ["--fine-tuned", str(put(tmp_path / "tuned.jsonl", tuned)), "--out", str(tmp_path / "out")]

    assert reading_scores.main(argv) == 0

    scores = json.loads((tmp_path / "out" / "belebele_scores.json").read_text(encoding="utf-8"))
    assert list(scores) == ["base", "fine_tuned", "paired"]
    assert scores["base"]["all"] | {"letters_answered": None, "correct_by_letter_of_correct": None} == {
        "rows": 3,
        "correct": 2,
        "accuracy": 0.667,
        "no_letter": 0,
        "read_through_a_letter_name": 1,
        "correct_without_letter_names": 1,
        "abstention_sentence_or_style": 0,
        "letters_answered": None,
        "correct_by_letter_of_correct": None,
    }
    assert scores["base"]["all"]["letters_answered"] == {"A": 1, "B": 2}
    assert scores["base"]["all"]["correct_by_letter_of_correct"]["B"] == [2, 3]
    assert scores["base"]["hi"]["accuracy"] == 1.0 and scores["base"]["ar"] == scores["base"]["ru"]
    assert scores["base"]["ru"]["rows"] == 0 and scores["base"]["ru"]["accuracy"] is None
    tuned_hi = scores["fine_tuned"]["hi"]
    assert tuned_hi["no_letter"] == 1 and tuned_hi["abstention_sentence_or_style"] == 1 and tuned_hi["correct"] == 0
    assert scores["paired"] == {"rows": 3, "correct_to_wrong": 1, "wrong_to_correct": 1, "same_letter": 1}
    assert "paired base -> fine-tuned" in capsys.readouterr().out


def squad_row(number, condition, response, tokens):
    return {
        "id": f"squad.{number}",
        "condition": condition,
        "context": f"Passage {number}.",
        "question": f"Question {number}?",
        "gold_answers": ["the old mill"] if condition == "answerable" else [],
        "response": response,
        "model": "owner/model@abc",
        "stop_reason": "eos",
        "generated_tokens": tokens,
        "title": "Article",
        "prompt": "left out of the rows to read",
    }


def test_squad_answers_become_rows_to_read_and_a_count_by_wording(tmp_path, capsys):
    base = [
        squad_row(0, "answerable", "It stood by The Old Mill.", 9),
        squad_row(1, "unanswerable", "It stood by the river.", 7),
        squad_row(2, "unanswerable", "The passage does not say.", 5),
    ]
    put(tmp_path / "base.jsonl", base)

    assert reading_scores.main(["squad", "--base", str(tmp_path / "base.jsonl"), "--out", str(tmp_path / "out")]) == 0

    rows = [json.loads(line) for line in (tmp_path / "out" / "squad_rows.jsonl").read_text().split("\n") if line]
    assert [found["id"] for found in rows] == ["squad.0", "squad.1", "squad.2"]
    assert all(found["model"] == "base" and found["model_revision"] == "owner/model@abc" for found in rows)
    assert all("prompt" not in found for found in rows)
    counts = json.loads((tmp_path / "out" / "squad_text_counts.json").read_text())
    assert counts == {
        "base answerable": {
            "no abstention wording": 1,
            "contains a gold answer": 1,
            "rows": 1,
            "median_tokens": 9,
        },
        "base unanswerable": {"no abstention wording": 1, "wider wording": 1, "rows": 2, "median_tokens": 7},
    }
    assert "3 rows -> squad_rows.jsonl" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        reading_scores.main(["squad", "--out", str(tmp_path / "out")])
