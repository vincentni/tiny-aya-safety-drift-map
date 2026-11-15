"""The prompts of the reading check, drawn from made-up files in the shape of SQuAD 2.0 and Belebele."""

import json
from collections import Counter

import pytest

from driftmap.attribution import build, reading
from tests.fakes.reading_sources import CLOSING, CODES, write


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """A folder of files in the shape of the two datasets with their sha256 recorded, and the closing lines."""
    return write(tmp_path, monkeypatch)


def no_download(url):
    raise AssertionError(f"asked for {url}")


def run(tmp_path, sources, out, download=no_download):
    argv = ["--sources", str(sources), "--translation", str(tmp_path / "translation_manifest.json"), "--out", str(out)]
    assert reading.main(argv, download) == 0
    return [
        [json.loads(line) for line in (out / name).read_text(encoding="utf-8").split("\n") if line]
        for name in ("squad_in.jsonl", "belebele_in.jsonl")
    ]


def test_the_squad_draw_pairs_an_unanswerable_and_an_answerable_question_each_on_its_own_short_passage(
    tmp_path, sources
):
    squad, _ = run(tmp_path, sources, tmp_path / "out")

    assert [row["condition"] for row in squad] == ["unanswerable", "answerable"] * 100
    assert len({row["context"] for row in squad}) == 200 and len({row["id"] for row in squad}) == 200
    assert all(len(row["context"].split()) < reading.MAX_WORDS for row in squad)
    assert all(row["id"].startswith("squad.") and row["set"] == "benign" and row["language"] == "en" for row in squad)
    for row in squad:
        assert row["prompt"] == "\n".join([row["context"], row["question"], build.FREE_INSTRUCTION])
        assert (row["gold_answers"] == []) == (row["condition"] == "unanswerable")
        assert len(row["gold_answers"]) <= 1
    assert set(Counter(row["title"] for row in squad).values()) <= {6, 7, 8}


def test_the_belebele_draw_asks_sixty_questions_in_four_languages_with_one_option_order(tmp_path, sources):
    _, belebele = run(tmp_path, sources, tmp_path / "out")

    assert len(belebele) == 240 and len({row["id"] for row in belebele}) == 240
    by_language = {code: [row for row in belebele if row["language"] == code] for code in CODES}
    assert [row["item"] for row in by_language["en"]] == [row["item"] for row in by_language["ru"]]
    assert len({row["item"].split("#")[0] for row in by_language["en"]}) == 60
    for code, rows in by_language.items():
        assert Counter(row["letter_correct"] for row in rows) == {"A": 15, "B": 15, "C": 15, "D": 15}
        assert [row["letter_correct"] for row in rows] == [row["letter_correct"] for row in by_language["en"]]
        for row, english in zip(rows, by_language["en"], strict=True):
            assert row["id"] == english["id"].replace(".en.", f".{code}.")
            places = [entry["text"].split(" option ")[1] for entry in row["options"]]
            assert places == [entry["text"].split(" option ")[1] for entry in english["options"]]
            passage, question = row["item"].rsplit("/", 1)[1].split("#")
            right = next(entry["text"] for entry in row["options"] if entry["letter"] == row["letter_correct"])
            assert right == f"{code} option {(int(passage) + int(question)) % 4 + 1} of {passage}.{question}"
            lines = row["prompt"].split("\n")
            assert lines[:2] == [row["passage"], row["question"]] and lines[-1] == CLOSING[code]
            assert lines[2:6] == [f"{entry['letter']}. {entry['text']}" for entry in row["options"]]


def test_the_same_inputs_give_the_same_files_and_missing_files_are_downloaded_at_the_pinned_revisions(
    tmp_path, sources
):
    run(tmp_path, sources, tmp_path / "one")
    asked = []

    def served(url):
        asked.append(url)
        folder = reading.SQUAD_FOLDER if reading.SQUAD in url else reading.BELEBELE_FOLDER
        return (sources / folder / url.split("/resolve/")[1].split("/", 1)[1]).read_bytes()

    run(tmp_path, tmp_path / "empty", tmp_path / "two", served)

    for name in ("squad_in.jsonl", "belebele_in.jsonl"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name
    hub = "https://huggingface.co/datasets"
    assert asked[0] == f"{hub}/{reading.SQUAD}/resolve/{reading.SQUAD_REVISION}/{reading.SQUAD_FILE}"
    assert sorted(asked[1:]) == sorted(
        f"{hub}/{reading.BELEBELE}/resolve/{reading.BELEBELE_REVISION}/data/{name}.jsonl" for name in CODES.values()
    )
    assert (tmp_path / "empty" / reading.SQUAD_FOLDER / reading.SQUAD_FILE).is_file()


def test_a_changed_file_is_refused_before_anything_is_written(tmp_path, sources):
    with (sources / reading.BELEBELE_FOLDER / "data" / "hin_Deva.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    argv = ["--sources", str(sources), "--translation", str(tmp_path / "translation_manifest.json")]

    with pytest.raises(SystemExit, match=f"data/hin_Deva.jsonl: sha256 .* recorded for {reading.BELEBELE} at"):
        reading.main([*argv, "--out", str(tmp_path / "out")], no_download)

    assert not (tmp_path / "out").exists()


def test_the_recorded_revisions_and_hashes_are_whole():
    assert len(reading.SQUAD_REVISION) == len(reading.BELEBELE_REVISION) == 40
    assert set(reading.BELEBELE_FILES) == {f"data/{name}.jsonl" for name in reading.LANGUAGES.values()}
    assert all(len(digest) == 64 for digest in (reading.SQUAD_FILES | reading.BELEBELE_FILES).values())
