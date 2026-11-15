"""The reading-check answers in the shape that is published: source ids, and no text of the source datasets."""

import json

import pytest

from driftmap.attribution import build, reading, reading_answers
from tests.fakes.reading_sources import CODES, belebele_rows, squad_rows, write

MODELS = {"base": "owner/base@0123456789ab", "fine_tuned": "owner/tuned@ba9876543210"}
TRAINED = build.SENTENCES["Cannot be determined"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def put(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def no_download(url):
    raise AssertionError(f"asked for {url}")


def source_texts():
    """Every text of the made-up source files: passages, questions, options, gold answers and titles."""
    texts = set()
    for code in CODES:
        for row in belebele_rows(code):
            texts |= {row["flores_passage"], row["question"], *(row[f"mc_answer{place}"] for place in range(1, 5))}
    for row in squad_rows():
        texts |= {row["context"], row["question"], row["title"], *row["answers"]["text"]}
    return {text.casefold() for text in texts}


def strings(value):
    """Every string inside a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from strings(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from strings(inner)


def answered(row, arm, response, tokens=3):
    """A prompt row as the generator writes it back: every field of the prompt, and the answer of one model."""
    return row | {
        "response": response,
        "rendered_prompt": f"<turn>{row['prompt']}</turn>",
        "generated_tokens": tokens,
        "stop_reason": "eos",
        "model": MODELS[arm],
        "max_new_tokens": 24,
        "preamble": "keep",
        "prompt_tokens": 40,
        "error": None,
        "batch_seconds": 1.5,
    }


@pytest.fixture
def prompts(tmp_path, monkeypatch):
    """The prompt files of the reading check, drawn from made-up sources; the sources folder beside them."""
    sources = write(tmp_path, monkeypatch)
    argv = ["--sources", str(sources), "--translation", str(tmp_path / "translation_manifest.json")]
    assert reading.main([*argv, "--out", str(tmp_path / "prompts")], no_download) == 0
    return (
        sources,
        read_jsonl(tmp_path / "prompts" / "belebele_in.jsonl"),
        read_jsonl(tmp_path / "prompts" / "squad_in.jsonl"),
    )


def belebele_response(row, number):
    """Mostly a letter; some answers a model could give that repeat the source."""
    option = row["options"][1]
    return [
        row["letter_correct"],
        "В.",
        "बी",
        f"{option['letter']}. {option['text']}",
        f"The text says: {row['passage']}",
        option["text"],
        "",
    ][number % 7]


def squad_response(row, number):
    return [
        TRAINED,
        f"As the text says, {row['context'][:60]}",
        "The passage does not say.",
        f"It is {(row['gold_answers'] or ['unknown'])[0]}.",
        row["question"],
    ][number % 5]


def run_belebele(tmp_path, sources, rows):
    files = {
        arm: put(tmp_path / f"belebele_{arm}.jsonl", [answered(row, arm, belebele_response(row, n)) for n, row in rows])
        for arm in MODELS
    }
    argv = ["belebele", "--sources", str(sources), "--base", str(files["base"])]
    argv += ["--fine-tuned", str(files["fine_tuned"]), "--out", str(tmp_path / "out")]
    assert reading_answers.main(argv, no_download) == 0
    return read_jsonl(tmp_path / "out" / "belebele_answers.jsonl")


def test_no_text_of_the_source_datasets_reaches_the_published_rows(tmp_path, prompts):
    sources, belebele, squad = prompts
    written = run_belebele(tmp_path, sources, list(enumerate(belebele)))
    files = {
        arm: put(
            tmp_path / f"squad_{arm}.jsonl", [answered(row, arm, squad_response(row, n)) for n, row in enumerate(squad)]
        )
        for arm in MODELS
    }
    argv = ["squad", "--base", str(files["base"]), "--fine-tuned", str(files["fine_tuned"])]
    assert reading_answers.main([*argv, "--out", str(tmp_path / "out")]) == 0
    written += read_jsonl(tmp_path / "out" / "squad_answers.jsonl")

    assert len(written) == 2 * (len(belebele) + len(squad))
    forbidden = source_texts()
    for row in written:
        for text in strings(row):
            folded = text.casefold()
            assert not any(source in folded for source in forbidden), (row["id"], text)
        assert not {
            "prompt",
            "rendered_prompt",
            "passage",
            "context",
            "question",
            "options",
            "gold_answers",
            "title",
        } & set(row)
    # the answers that repeated the source are among the rows, without their text
    withheld = [row for row in written if row.get("response_withheld")]
    assert all(row["response"] is None for row in withheld)
    assert len([row for row in withheld if row["id"].startswith("belebele.")]) == 2 * len(
        range(3, len(belebele), 7)
    ) + 2 * len(range(4, len(belebele), 7)) + 2 * len(range(5, len(belebele), 7))
    assert len([row for row in withheld if row["id"].startswith("squad.")]) == 2 * (
        len(squad) - len(range(0, len(squad), 5))
    )


def test_a_belebele_row_names_its_question_and_the_order_of_its_options_by_the_source(tmp_path, prompts):
    sources, belebele, _ = prompts
    written = run_belebele(tmp_path, sources, list(enumerate(belebele)))

    assert [(row["id"], row["arm"]) for row in written] == sorted(
        (row["id"], arm) for row in belebele for arm in MODELS
    )
    by_id = {row["id"]: row for row in belebele}
    numbers = {row["id"]: number for number, row in enumerate(belebele)}
    for row in written:
        asked = by_id[row["id"]]
        number = numbers[row["id"]]
        assert set(row) - {"response_withheld"} == {
            "id",
            "arm",
            "model",
            "source",
            "source_file",
            "link",
            "question_number",
            "language",
            "option_order",
            "letter_correct",
            "response",
            "letter",
            "correct",
            "read_through_a_letter_name",
            "generated_tokens",
            "stop_reason",
            "max_new_tokens",
            "preamble",
        }
        assert row["source"] == f"{reading.BELEBELE}@{reading.BELEBELE_REVISION}" and row["model"] == MODELS[row["arm"]]
        assert row["source_file"] == f"data/{CODES[row['language']]}.jsonl"
        assert f"{row['link']}#{row['question_number']}" == asked["item"]
        # the source's answer numbers at A to D: the option under a letter is that answer of the source row
        assert sorted(row["option_order"]) == [1, 2, 3, 4]
        passage = row["link"].rsplit("/", 1)[1]
        assert [entry["text"] for entry in asked["options"]] == [
            f"{row['language']} option {place} of {passage}.{row['question_number']}" for place in row["option_order"]
        ]
        right = (int(passage) + row["question_number"]) % 4 + 1
        assert row["option_order"]["ABCD".index(row["letter_correct"])] == right
        kind = number % 7
        if kind == 0:
            assert (row["response"], row["letter"], row["correct"]) == (
                asked["letter_correct"],
                asked["letter_correct"],
                True,
            )
        elif kind == 1:
            assert (row["response"], row["letter"]) == ("В.", "B") and not row["read_through_a_letter_name"]
        elif kind == 2:
            assert (row["response"], row["letter"]) == ("बी", "B") and row["read_through_a_letter_name"]
        elif kind in (3, 4, 5):
            assert row["response"] is None and row["response_withheld"] is True
            assert row["letter"] == (asked["options"][1]["letter"] if kind in (3, 5) else None)
        else:
            assert (row["response"], row["letter"], row["correct"]) == (
                "",
                None,
                False,
            ) and "response_withheld" not in row


def test_a_squad_row_keeps_its_wording_class_and_only_a_trained_sentence_as_text(tmp_path, prompts):
    _, _, squad = prompts
    rows = [answered(row, "base", squad_response(row, number), tokens=9) for number, row in enumerate(squad[:10])]
    argv = ["squad", "--base", str(put(tmp_path / "base.jsonl", rows)), "--out", str(tmp_path / "out")]

    assert reading_answers.main(argv) == 0

    written = {row["id"]: row for row in read_jsonl(tmp_path / "out" / "squad_answers.jsonl")}
    assert list(written) == sorted(row["id"] for row in squad[:10])
    for number, asked in enumerate(squad[:10]):
        row = written[asked["id"]]
        assert set(row) - {"response_withheld"} == {
            "id",
            "arm",
            "model",
            "source",
            "source_id",
            "condition",
            "wording",
            "contains_a_gold_answer",
            "response",
            "generated_tokens",
            "stop_reason",
            "max_new_tokens",
            "preamble",
        }
        assert row["source"] == f"{reading.SQUAD}@{reading.SQUAD_REVISION}" and row["id"] == f"squad.{row['source_id']}"
        assert (row["arm"], row["condition"], row["generated_tokens"]) == ("base", asked["condition"], 9)
        kind = number % 5
        if kind == 0:
            assert (row["response"], row["wording"]) == (TRAINED, "trained sentence") and "response_withheld" not in row
        else:
            assert row["response"] is None and row["response_withheld"] is True
        if kind == 2:
            assert row["wording"] == "wider wording"
        if kind == 3:
            assert row["contains_a_gold_answer"] == (asked["condition"] == "answerable")


def test_answers_to_questions_the_pinned_source_does_not_hold_are_refused(tmp_path, prompts):
    sources, belebele, _ = prompts
    other = belebele[0] | {"options": [entry | {"text": "another option"} for entry in belebele[0]["options"]]}
    argv = ["belebele", "--sources", str(sources), "--out", str(tmp_path / "out")]

    with pytest.raises(SystemExit, match="is not an answer of the source question"):
        reading_answers.main(
            [*argv, "--base", str(put(tmp_path / "base.jsonl", [answered(other, "base", "A")]))], no_download
        )
    with pytest.raises(SystemExit):
        reading_answers.main(["belebele", "--base", str(tmp_path / "base.jsonl"), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
