"""The build of the abstention data, on made-up files in the shape of BBQ."""

import hashlib
import json
import random
from collections import Counter

import pytest

from driftmap.attribution import bbq, build
from driftmap.core import ids
from tests.fakes import bbq as fake_bbq

FOLDER = ids.training_folder(ids.TRAIN_SETS[0])
BENIGN = [
    {
        "prompt": [{"role": "user", "content": f"Translate sentence {number}."}],
        "completion": [{"role": "assistant", "content": f"Sentence {number}, translated."}],
        "language": "en",
    }
    for number in range(5)
]


def no_download(url):
    raise AssertionError(f"asked for {url}")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


@pytest.fixture
def source(tmp_path, monkeypatch):
    """A folder of files in the shape of BBQ with their sha256 recorded, and the benign rows as a file."""
    monkeypatch.setattr(bbq, "FILES", fake_bbq.write(tmp_path / "bbq"))
    benign = tmp_path / "benign.jsonl"
    benign.write_text("".join(json.dumps(row) + "\n" for row in BENIGN), encoding="utf-8")
    return tmp_path / "bbq", benign


def run(source, out):
    files, benign = source
    assert build.main(["--bbq", str(files), "--benign", str(benign), "--out", str(out)], no_download) == 0
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_the_build_writes_a_test_file_and_a_training_set_that_share_no_item_template_or_context(
    tmp_path, source, capsys
):
    manifest = run(source, tmp_path / "out")

    assert manifest["source"]["commit"] == bbq.COMMIT and manifest["source"]["licence"] == "CC BY 4.0"
    assert manifest["source"]["files_sha256"] == {name.split("/")[-1]: digest for name, digest in bbq.FILES.items()}
    assert set(manifest["files_sha256"]) == {"test_en.jsonl", f"{FOLDER}/en.jsonl", f"{FOLDER}/en.meta.jsonl"}
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((tmp_path / "out" / name).read_bytes()).hexdigest() == digest
    leak = manifest["leak_check"]
    compared = {key: leak.pop(key) for key in ("test_contexts_compared", "training_contexts_compared")}
    assert compared == {"test_contexts_compared": 300, "training_contexts_compared": 1200}
    assert leak and set(leak.values()) == {0}
    assert manifest["pilot_items_left_out"]["base_items"] == 90

    rows = read_jsonl(tmp_path / "out" / "test_en.jsonl")
    assert len(rows) == 1200 and len({row["prompt_id"] for row in rows}) == 1200
    assert Counter(row["split"] for row in rows if row["prompt_id"].endswith(".amb.o0")) == {
        "heldout_category": 75,
        "heldout_template": 75,
    }
    main = [row for row in rows if row["format"] == "letter" and row["order"] == 0 and "arm_tag" not in row]
    ambiguous = [row for row in main if row["condition"] == "ambiguous"]
    assert Counter(row["letter_unknown"] for row in ambiguous) == {"A": 50, "B": 50, "C": 50}
    assert all(row["letter_correct"] == row["letter_unknown"] for row in ambiguous)
    twins = {row["base_item"]: row for row in main if row["condition"] == "disambiguated"}
    for row in ambiguous:
        twin = twins[row["base_item"]]
        assert twin["options"] == row["options"] and twin["letter_correct"] != twin["letter_unknown"]
        assert twin["context"].startswith(row["context"]) and twin["prompt"] != row["prompt"]
    assert Counter(f"{row['format']} {row['order']} {row.get('arm_tag', 'plain')}" for row in rows) == {
        "letter 0 plain": 300,
        "letter 0 instruction": 300,
        "letter 1 plain": 120,
        "letter 1 instruction": 120,
        "letter 2 plain": 120,
        "letter 2 instruction": 120,
        "free_form 0 plain": 120,
    }
    tagged = next(row for row in rows if row.get("arm_tag") == "instruction")
    assert tagged["prompt"].endswith(f"{build.ABSTAIN_INSTRUCTION}\n{build.LETTER_INSTRUCTION}")

    training = read_jsonl(tmp_path / "out" / FOLDER / "en.jsonl")
    meta = read_jsonl(tmp_path / "out" / FOLDER / "en.meta.jsonl")
    assert len(training) == len(meta) == 1205
    assert all(set(row) == {"prompt", "completion"} for row in training)
    assert Counter(row["kind"] for row in meta) == {"abstention": 1200, "benign": 5}
    told = [(row, about) for row, about in zip(training, meta, strict=True) if about["kind"] == "abstention"]
    assert Counter(f"{about['condition']} {about['format']}" for _, about in told) == {
        "ambiguous letter": 300,
        "ambiguous free_form": 300,
        "disambiguated letter": 300,
        "disambiguated free_form": 300,
    }
    assert {about["category"] for _, about in told} == set(build.TRAINED_CATEGORIES)
    assert not {about["unknown_phrasing"] for _, about in told} & set(build.HELDOUT_PHRASINGS)
    for row, about in told:
        said = row["completion"][0]["content"]
        if about["format"] == "letter":
            assert said == about["letter_correct"] and row["prompt"][0]["content"].endswith(build.LETTER_INSTRUCTION)
        elif about["condition"] == "ambiguous":
            assert said == build.SENTENCES[about["unknown_phrasing"]]
        else:
            assert said.startswith("The answer is ") and "rider" in said
    assert not {about["base_item"] for _, about in told} & {row["base_item"] for row in rows}
    assert '"leak_check"' in capsys.readouterr().out


def test_the_same_inputs_give_the_same_files_byte_for_byte(tmp_path, source):
    run(source, tmp_path / "one")
    run(source, tmp_path / "two")

    for name in ("test_en.jsonl", f"{FOLDER}/en.jsonl", f"{FOLDER}/en.meta.jsonl", "manifest.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name


def test_a_changed_file_is_refused_before_anything_is_written(tmp_path, source):
    files, benign = source
    with (files / "data" / "Age.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(SystemExit, match="data/Age.jsonl: sha256"):
        build.main(["--bbq", str(files), "--benign", str(benign), "--out", str(tmp_path / "out")], no_download)

    assert not (tmp_path / "out").exists()


def test_an_item_keeps_one_option_order_for_its_two_rows_and_the_unknown_option_sits_at_each_letter_equally():
    items = [{"unknown": number % 3, "correct": (number + 1) % 3} for number in range(12)]

    build.arrange(items, random.Random(7))

    assert all(sorted(item["placed"]) == [0, 1, 2] for item in items)
    assert Counter(item["placed"].index(item["unknown"]) for item in items) == {0: 4, 1: 4, 2: 4}
    assert Counter(item["placed"].index(item["correct"]) for item in items) == {0: 4, 1: 4, 2: 4}
    assert [build.rotated([2, 0, 1], turn) for turn in range(3)] == [[2, 0, 1], [1, 2, 0], [0, 1, 2]]


def test_a_draw_goes_round_the_templates_and_stops_when_the_pool_is_too_small():
    pool = [{"base_item": f"item {number:02d}", "template": f"t/{number % 4}"} for number in range(20)]

    drawn = build.spread(pool, 10, random.Random(3), cap=3)

    assert len({item["base_item"] for item in drawn}) == 10
    assert sorted(Counter(item["template"] for item in drawn).values()) == [2, 2, 3, 3]
    with pytest.raises(SystemExit, match="only 12 of 13 items could be drawn"):
        build.spread(pool, 13, random.Random(3), cap=3)


def test_a_free_form_completion_names_the_option_in_a_sentence():
    assert build.naming("The grandfather") == "The answer is the grandfather."
    assert build.naming("grandfather") == "The answer is the grandfather."
    assert build.naming("Ana Lopez") == "The answer is Ana Lopez."
