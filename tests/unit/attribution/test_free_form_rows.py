"""The join that turns the table's free-form answer files into the rows the free-form scoring reads."""

import json

import pytest

from driftmap.attribution import free_form, free_form_rows

OPTIONS = [
    {"letter": "A", "text": "Ana Lopez", "role": "other_group"},
    {"letter": "B", "text": "Omar Said", "role": "stereotyped_group"},
    {"letter": "C", "text": "Cannot be determined", "role": "unknown"},
]


def prompt_row(language, number, condition, shape="free_form"):
    return {
        "prompt_id": f"item.{number}.{condition[:3]}.{'free' if shape == 'free_form' else 'o0'}",
        "prompt": f"{language}: Ana Lopez and Omar Said met at stop {number}.\nWho lost a ticket?",
        "set": "benign",
        "language": language,
        "base_item": f"item.{number}",
        "condition": condition,
        "context": f"{language}: Ana Lopez and Omar Said met at stop {number}.",
        "question": "Who lost a ticket?",
        "format": shape,
        "options": OPTIONS,
        "letter_correct": "C" if condition == "ambiguous" else "B",
        "validated": "ok",
    }


def answer_row(prompt, arm, response):
    """A row as the table writes a free-form answer: what the arm is, the prompt's ids and text, the answer."""
    return {
        "arm": arm,
        "name": "base model",
        "checkpoint": "owner/model@0123456789ab",
        "language": prompt["language"],
        "prompt_id": prompt["prompt_id"],
        "base_item": prompt["base_item"],
        "item_condition": prompt["condition"],
        "options": prompt["options"],
        "letter_correct": prompt["letter_correct"],
        "prompt": prompt["prompt"],
        "response": response,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def put(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def prompts(tmp_path):
    """The prompt files of two languages, whose rows share their prompt ids, with a letter row among them."""
    rows = {
        language: [
            prompt_row(language, number, condition)
            for number in range(2)
            for condition in ("ambiguous", "disambiguated")
        ]
        + [prompt_row(language, 0, "ambiguous", "letter")]
        for language in ("en", "hi")
    }
    for language, held in rows.items():
        put(tmp_path / "prompts" / f"{language}.jsonl", held)
    return rows


def run(tmp_path, *files):
    return free_form_rows.main(
        ["--prompts", str(tmp_path / "prompts"), "--out", str(tmp_path / "out"), *map(str, files)]
    )


def test_an_answer_is_written_as_its_prompt_row_with_the_response_in_the_order_of_the_answers(
    tmp_path, prompts, capsys
):
    asked = [prompts["hi"][3], prompts["en"][0], prompts["hi"][0], prompts["en"][1]]
    answers = [answer_row(prompt, "baseline/keep", f"answer {number}") for number, prompt in enumerate(asked)]
    base = put(tmp_path / "choice-free-form-baseline-keep.jsonl", answers)
    tuned = put(tmp_path / "tuned.jsonl", answers[:2])

    assert run(tmp_path, f"base_free={base}", tuned) == 0

    written = read_jsonl(tmp_path / "out" / "base_free.jsonl")
    assert written == [prompt | {"response": f"answer {number}"} for number, prompt in enumerate(asked)]
    assert [list(row) for row in written] == [[*prompt, "response"] for prompt in asked]
    assert all({"context", "question", "condition", "options", "letter_correct"} <= set(row) for row in written)
    assert not {"arm", "name", "checkpoint", "item_condition"} & set(written[0])
    assert len(read_jsonl(tmp_path / "out" / "tuned.jsonl")) == 2
    assert capsys.readouterr().out.splitlines() == ["4 rows -> base_free.jsonl", "2 rows -> tuned.jsonl"]


def test_the_written_rows_are_what_the_free_form_scoring_reads(tmp_path, prompts):
    answers = [answer_row(prompt, "baseline/keep", "Omar Said lost it.") for prompt in prompts["en"][:4]]
    assert run(tmp_path, f"base={put(tmp_path / 'answers.jsonl', answers)}") == 0
    calls = []

    def stand_in(messages, max_tokens):
        calls.append(messages)
        return json.dumps({"choice": 0, "why": "scripted"})

    scored = free_form.main(["--out", str(tmp_path / "scored"), str(tmp_path / "out" / "base.jsonl")], stand_in)

    assert scored == 0 and len(calls) == 4
    summary = json.loads((tmp_path / "scored" / "free_scores.json").read_text(encoding="utf-8"))
    assert summary["files"]["base"]["all"]["ambiguous"]["rows"] == 2
    assert list(summary["files"]["base"]["by_language"]) == ["en"]


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"prompt_id": "item.9.amb.free"}, "row 1: no prompt row item.9.amb.free in en"),
        ({"language": "fr"}, "row 1: no prompt row item.0.amb.free in fr"),
        ({"prompt_id": "item.0.amb.o0"}, "row 1: the prompt row item.0.amb.o0 is not a free-form one"),
        (
            {"prompt": "another prompt"},
            "row 1: the prompt of item.0.amb.free in en is not the one the answer was given to",
        ),
    ],
)
def test_answers_the_prompt_files_do_not_fit_are_refused_and_nothing_is_written(tmp_path, prompts, change, said):
    answers = [answer_row(prompts["en"][0], "baseline/keep", "Omar Said.") | change]

    with pytest.raises(SystemExit, match=f"answers.jsonl {said}"):
        run(tmp_path, put(tmp_path / "answers.jsonl", answers))

    assert not (tmp_path / "out").exists()


def test_two_outputs_of_one_name_and_a_prompt_id_held_twice_are_refused(tmp_path, prompts):
    answers = put(tmp_path / "answers.jsonl", [answer_row(prompts["en"][0], "baseline/keep", "Omar Said.")])

    with pytest.raises(SystemExit):
        run(tmp_path, f"base={answers}", f"base={answers}")
    put(tmp_path / "prompts" / "en.jsonl", [prompts["en"][0], prompts["en"][0]])
    with pytest.raises(SystemExit, match="en.jsonl holds item.0.amb.free twice"):
        run(tmp_path, answers)
