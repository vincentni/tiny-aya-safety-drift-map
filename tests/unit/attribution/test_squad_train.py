"""The training set ``squad``: the gated train questions as rows, their targets, and what the sidecar holds."""

import hashlib
import json

import pytest

from driftmap.attribution import squad_train
from driftmap.core import ids
from driftmap.scripts import drift
from tests.fakes.squad_probe import gated, put, read_jsonl, squad_sample


@pytest.fixture
def inputs(tmp_path):
    """Twelve train questions, the last one dropped by the gate, and a training set with two benign rows."""
    rows = squad_sample("train", 2, 6)
    put(tmp_path / "squad_train_in.jsonl", rows)
    passed = gated(tmp_path / "squad_passed.json", {tmp_path / "squad_train_in.jsonl": ("train", [rows[-1]["id"]])})
    other = tmp_path / "sft_train_defense"
    other.mkdir()
    held = [squad_train.turns(f"a request of row {at}", f"a reply of row {at}") for at in range(4)]
    put(other / "en.jsonl", held)
    notes = [{"kind": "abstention", "base_item": "bbq.Age.0"}, {"kind": "benign"}] * 2
    put(other / "en.meta.jsonl", notes)
    argv = ["--train", str(tmp_path / "squad_train_in.jsonl"), "--passed", str(passed), "--benign", str(other)]
    return tmp_path, rows, held, [*argv, "--out", str(tmp_path / "data")]


def test_a_gated_question_becomes_a_row_that_answers_with_a_gold_answer_or_the_abstention_sentence(inputs, capsys):
    tmp_path, rows, held, argv = inputs

    assert squad_train.main(argv) == 0

    folder = tmp_path / "data" / "sft_train_squad"
    assert folder.name == ids.training_folder("squad") and "squad" in ids.TRAIN_SETS
    assert sorted(path.name for path in folder.iterdir()) == ["en.jsonl", "en.meta.jsonl", "manifest.json"]
    written, notes = read_jsonl(folder / "en.jsonl"), read_jsonl(folder / "en.meta.jsonl")
    assert len(written) == len(notes) == 11 + 2
    assert squad_train.ABSTENTION == "It cannot be determined from the text."
    # the question's own prompt, unchanged; unanswerable first, then the answerable one of the pair
    assert written[0] == squad_train.turns(rows[0]["prompt"], "It cannot be determined from the text.")
    assert written[1] == squad_train.turns(rows[1]["prompt"], rows[1]["gold_answers"][0])
    assert notes[0] == {
        "kind": "squad",
        "id": rows[0]["id"],
        "split": "train",
        "condition": "unanswerable",
        "article": hashlib.sha256(rows[0]["title"].encode()).hexdigest()[:16],
        "passage": hashlib.sha256(rows[0]["context"].encode()).hexdigest()[:16],
    }
    # two questions of one article share its hash, and no title is written
    assert notes[0]["article"] == notes[1]["article"] != notes[6]["article"]
    assert notes[0]["passage"] != notes[1]["passage"]
    assert [note["id"] for note in notes[:11]] == [row["id"] for row in rows[:11]]
    # the benign rows of the other set follow as they are, told by its sidecar
    assert written[11:] == [held[1], held[3]] and notes[11:] == [{"kind": "benign"}] * 2
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == {
        "all": 13,
        "squad": 11,
        "by_condition": {"unanswerable": 6, "answerable": 5},
        "articles": 2,
        "benign": 2,
    }
    assert manifest["curatorkit_gate"] == {
        "rows_in": 12,
        "dropped": {"row_shape": 0, "schema_gate": 1},
        "rows_passed": 11,
    }
    assert manifest["files"]["en.jsonl"] == hashlib.sha256((folder / "en.jsonl").read_bytes()).hexdigest()
    assert manifest["targets"]["unanswerable"] == squad_train.ABSTENTION and manifest["benign_rows_from"] == (
        "sft_train_defense"
    )
    assert "working copy of the dataset only" in manifest["text"]["en.jsonl"]
    # the sidecar and the manifest hold no passage, question or answer
    for name in ("en.meta.jsonl", "manifest.json"):
        said = (folder / name).read_text(encoding="utf-8")
        assert not any(text in said for row in rows for text in (row["context"], row["question"], row["title"])), name
        assert not any(answer in said for row in rows for answer in row["gold_answers"]), name
    assert '"squad": 11' in capsys.readouterr().out

    # the training job reads the folder as a named training set
    pairs = drift.training_rows(tmp_path / "data", ["en"], {}, "squad")
    assert len(pairs) == 13 and pairs[0] == ("en", written[0])


def test_an_answerable_question_without_a_gold_answer_and_a_sidecar_of_another_length_are_refused(inputs):
    tmp_path, rows, _, argv = inputs
    put(tmp_path / "sft_train_defense" / "en.meta.jsonl", [{"kind": "benign"}])
    with pytest.raises(SystemExit, match="en.meta.jsonl holds 1 lines for the 4 rows of sft_train_defense/en.jsonl"):
        squad_train.main(argv)
    put(tmp_path / "sft_train_defense" / "en.meta.jsonl", [{"kind": "benign"}] * 4)
    rows[1]["gold_answers"] = []
    put(tmp_path / "squad_train_in.jsonl", rows)
    gated(tmp_path / "squad_passed.json", {tmp_path / "squad_train_in.jsonl": ("train", [])})
    with pytest.raises(SystemExit, match="1 answerable questions of squad_train_in.jsonl have no gold answer"):
        squad_train.main(argv)
    assert not (tmp_path / "data").exists()
