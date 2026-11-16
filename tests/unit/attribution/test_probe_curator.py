"""The CuratorKIT gate over the probe's SQuAD rows, on made-up rows: what each check drops and what is written."""

import hashlib
import json
import random

import pytest

from driftmap.attribution import probe_curator, reading

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def words(seed, count):
    rng = random.Random(seed)
    return " ".join("".join(rng.choices(LETTERS, k=6)) for _ in range(count))


def question(name, kind="answerable", passage=None, asks=None, title=None, **more):
    """A prompt row of made-up words; its passage and question are its own unless given."""
    context = passage or words(f"passage {name}", 60)
    asked = asks or f"{words(f'question {name}', 5)}?"
    row = {
        "id": f"squad.{name}",
        "condition": kind,
        "context": context,
        "question": asked,
        "gold_answers": [context.split()[3]] if kind == "answerable" else [],
        "title": title or f"Article of {name[0]}",
        "prompt": "\n".join([context, asked, reading.FREE]),
        "set": "benign",
        "language": "en",
    }
    return row | more


def put(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A train file with one row for every check to drop, and two test files."""
    for name in ("HF_DATASETS_CACHE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "")
    dev = [question(f"d{at}", ("unanswerable", "answerable")[at % 2]) for at in range(4)]
    dev.append(question("d4", "maybe"))
    other = [question(f"e{at}", ("unanswerable", "answerable")[at % 2]) for at in range(2)]
    train = [question(f"t{at}", ("unanswerable", "answerable")[at % 2]) for at in range(6)]
    again = train[0] | {"id": "squad.t6"}
    near = question("t7", passage=train[1]["context"] + " zzzzzz", asks=train[1]["question"])
    train += [
        again,
        near,
        question("t8", passage=dev[0]["context"]),
        question("t9", asks=dev[1]["question"].upper()),
        question("t10", passage=words("a long passage", 2100)),
        question("t11", gold_answers=[]),
        question("t12", id="squad.t2"),
        question("t13", passage=other[0]["context"]),
    ]
    put(tmp_path / "squad_train_in.jsonl", train)
    put(tmp_path / "squad_dev_in.jsonl", dev)
    put(tmp_path / "squad_dev2_in.jsonl", other)
    asked = ["--train", str(tmp_path / "squad_train_in.jsonl"), "--test", str(tmp_path / "squad_dev_in.jsonl")]
    asked += ["--test", str(tmp_path / "squad_dev2_in.jsonl"), "--work", str(tmp_path / "work")]
    return tmp_path, train, dev, other, [*asked, "--out", str(tmp_path / "curated")]


def test_a_row_is_a_drawn_question_only_with_every_field_of_one():
    row = question("t0")
    times = {"squad.t0": 1}

    assert probe_curator.shape(row, times) is None
    assert probe_curator.shape(row | {"title": " "}, times) == "missing_field:title"
    assert probe_curator.shape({key: value for key, value in row.items() if key != "question"}, times) == (
        "missing_field:question"
    )
    assert probe_curator.shape(row, {"squad.t0": 2}) == "id_repeats"
    assert probe_curator.shape(row | {"condition": "maybe"}, times) == "unknown_condition"
    unanswerable = row | {"condition": "unanswerable"}
    assert probe_curator.shape(unanswerable, times) == "gold_answers_do_not_match_the_condition"
    assert probe_curator.shape(unanswerable | {"gold_answers": []}, times) is None
    assert probe_curator.shape(row | {"prompt": row["prompt"] + " "}, times) == (
        "prompt_is_not_passage_question_closing_line"
    )


def test_the_gate_drops_a_row_per_check_and_writes_the_passing_ids_with_counts_and_no_text(files, capsys):
    folder, train, dev, other, asked = files

    assert probe_curator.main(asked) == 0

    assert sorted(path.name for path in (folder / "curated").iterdir()) == ["squad_curator.json", "squad_passed.json"]
    passed = json.loads((folder / "curated" / "squad_passed.json").read_text(encoding="utf-8"))
    assert passed["schema"] == "driftmap.probe-squad-passed/1" and list(passed["files"]) == [
        "squad_train_in.jsonl",
        "squad_dev_in.jsonl",
        "squad_dev2_in.jsonl",
    ]
    held = passed["files"]["squad_train_in.jsonl"]
    assert held == {
        "role": "train",
        "sha256": hashlib.sha256((folder / "squad_train_in.jsonl").read_bytes()).hexdigest(),
        "rows_in": 14,
        # t11 and the two rows under the id of t2; the long passage; the copy of t0; the near copy of t1; the
        # passages of a dev and of a second-sample question; the question of a dev row in other letter case
        "dropped": {
            "row_shape": 3,
            "schema_gate": 1,
            "exact_dedup": 1,
            "minhash_dedup": 1,
            "cross_set_passage": 2,
            "cross_set_question": 1,
        },
        "rows_passed": 5,
        "passed": ["squad.t0", "squad.t1", "squad.t3", "squad.t4", "squad.t5"],
    }
    assert passed["files"]["squad_dev_in.jsonl"] == {
        "role": "test",
        "sha256": hashlib.sha256((folder / "squad_dev_in.jsonl").read_bytes()).hexdigest(),
        "rows_in": 5,
        "dropped": {"row_shape": 1, "schema_gate": 0},
        "rows_passed": 4,
        "passed": [row["id"] for row in dev[:4]],
    }
    assert passed["files"]["squad_dev2_in.jsonl"]["passed"] == [row["id"] for row in other]

    report = json.loads((folder / "curated" / "squad_curator.json").read_text(encoding="utf-8"))
    assert report["schema"] == "driftmap.probe-squad-curator/1"
    assert report["script"] == "driftmap.attribution.probe_curator" and report["curatorkit_version"]
    assert report["thresholds"] == {
        "schema_gate": {"required_fields": ["instruction"], "min_tokens": 1, "max_tokens": 2048},
        "minhash_threshold": 0.85,
        "minhash_ngram": 3,
        "minhash_num_perm": 128,
    }
    assert report["source"]["revision"] == reading.SQUAD_REVISION
    entry = report["files"]["squad_train_in.jsonl"]
    assert {key: entry[key] for key in ("role", "rows_in", "dropped", "rows_passed")} == {
        key: held[key] for key in ("role", "rows_in", "dropped", "rows_passed")
    }
    assert "passed" not in entry and entry["by_condition"] == {"unanswerable": 3, "answerable": 8}
    pipeline = entry["pipeline"]
    assert pipeline["row_shape_reasons"] == {"gold_answers_do_not_match_the_condition": 1, "id_repeats": 2}
    assert (pipeline["rows_into_the_pipeline"], pipeline["rejected_by_the_gate"], pipeline["rows_out"]) == (11, 1, 8)
    assert pipeline["rejection_reasons"] == {"SchemaGate:above_max_tokens": 1}
    assert pipeline["config"] == "work/pipeline/squad_train_in/pipeline.yaml"
    assert report["files"]["squad_dev_in.jsonl"]["pipeline"]["row_shape_reasons"] == {"unknown_condition": 1}
    compared = {(found["check"], found["test"]): found for found in report["cross_set"]}
    assert list(compared) == [
        (check, name)
        for check in ("cross_set_passage", "cross_set_question")
        for name in ("squad_dev_in.jsonl", "squad_dev2_in.jsonl")
    ]
    first = compared["cross_set_passage", "squad_dev_in.jsonl"]
    assert first == {
        "check": "cross_set_passage",
        "text": "context",
        "train": "squad_train_in.jsonl",
        "test": "squad_dev_in.jsonl",
        "protected_texts": 4,
        "distinct_protected_texts": 4,
        "training_rows": 8,
        "copies": 1,
        "exact_matches_after_normalisation": 1,
        "training_rows_nearing_an_earlier_training_row_kept": 0,
    }
    assert compared["cross_set_passage", "squad_dev2_in.jsonl"]["copies"] == 1
    assert compared["cross_set_question", "squad_dev_in.jsonl"]["copies"] == 1
    assert compared["cross_set_question", "squad_dev2_in.jsonl"]["copies"] == 0
    assert report["shared_by_exact_text"]["squad_train_in.jsonl / squad_dev_in.jsonl"] == {
        "ids": 0,
        "passages": 1,
        "questions": 0,
        "articles": 0,
    }
    assert report["shared_by_exact_text"]["squad_dev_in.jsonl / squad_dev2_in.jsonl"]["passages"] == 0
    for name in ("squad_curator.json", "squad_passed.json"):
        written = (folder / "curated" / name).read_text(encoding="utf-8")
        for row in (*train, *dev, *other):
            assert row["question"] not in written and row["context"][:40] not in written, name
    assert '"rows_passed": 5' in capsys.readouterr().out


def test_the_passing_rows_are_read_only_from_the_file_the_ids_were_written_for(files):
    folder, train, dev, _, asked = files
    assert probe_curator.main(asked) == 0
    passed = folder / "curated" / "squad_passed.json"

    rows, counts = probe_curator.passing(list(dev), folder / "squad_dev_in.jsonl", passed)

    assert [row["id"] for row in rows] == [row["id"] for row in dev[:4]]
    assert counts == {"rows_in": 5, "dropped": {"row_shape": 1, "schema_gate": 0}, "rows_passed": 4}
    with pytest.raises(SystemExit, match="squad_passed.json names rows squad_dev_in.jsonl does not hold"):
        probe_curator.passing(dev[:2], folder / "squad_dev_in.jsonl", passed)
    put(folder / "squad_dev3_in.jsonl", dev)
    with pytest.raises(SystemExit, match="squad_passed.json holds no passing ids for squad_dev3_in.jsonl"):
        probe_curator.passing(list(dev), folder / "squad_dev3_in.jsonl", passed)
    put(folder / "squad_dev_in.jsonl", dev[:4])
    with pytest.raises(SystemExit, match="squad_passed.json was written for another squad_dev_in.jsonl"):
        probe_curator.passing(dev[:4], folder / "squad_dev_in.jsonl", passed)
    passed.write_text(json.dumps({"schema": "other", "files": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="holds no passing ids"):
        probe_curator.passing(train, folder / "squad_train_in.jsonl", passed)


def test_two_prompt_files_of_one_name_are_refused(files, capsys):
    folder, _, _, _, asked = files
    (folder / "more").mkdir()
    put(folder / "more" / "squad_dev_in.jsonl", [])

    with pytest.raises(SystemExit):
        probe_curator.main([*asked, "--test", str(folder / "more" / "squad_dev_in.jsonl")])

    assert "the prompt files must differ in name" in capsys.readouterr().err
    assert not (folder / "curated").exists()
