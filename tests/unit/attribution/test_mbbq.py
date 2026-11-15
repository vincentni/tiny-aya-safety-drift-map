"""The MBBQ files and the draw of question pairs from them, on made-up files in the shape of MBBQ and BBQ."""

import hashlib
import json
from collections import Counter

import pytest

from driftmap.attribution import bbq, build, mbbq, mbbq_answers, mbbq_build
from driftmap.core import ids
from driftmap.data import leak
from driftmap.table import choice
from tests.fakes import bbq as fake_bbq
from tests.fakes import mbbq as fake_mbbq

FOLDER = ids.training_folder(ids.TRAIN_SETS[0])
WANTED = {"Gender_identity": 9, "Age": 4, "Disability_status": 4, "Physical_appearance": 4, "SES": 3}
WANTED |= {"Sexual_orientation": 2}
ITEMS = sum(WANTED.values())


def no_download(url):
    raise AssertionError(f"asked for {url}")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Folders of files in the shape of BBQ and of MBBQ with their sha256 recorded, the data built from the first,
    and a smaller draw."""
    monkeypatch.setattr(bbq, "FILES", fake_bbq.write(tmp_path / "bbq"))
    monkeypatch.setattr(mbbq, "FILES", fake_mbbq.write(tmp_path / "mbbq", tmp_path / "bbq"))
    monkeypatch.setattr(mbbq_build, "WANTED", WANTED)
    benign = tmp_path / "benign.jsonl"
    benign.write_text("", encoding="utf-8")
    data = tmp_path / "data"
    assert build.main(["--bbq", str(tmp_path / "bbq"), "--benign", str(benign), "--out", str(data)], no_download) == 0
    return tmp_path


def run(source, out="out"):
    folders = [f"--{name}={source / name}" for name in ("mbbq", "bbq", "data")]
    assert mbbq_build.main([*folders, "--out", str(source / out)], no_download) == 0
    return json.loads((source / out / "mbbq" / "manifest.json").read_text(encoding="utf-8"))


def served(content, asked):
    def get(url):
        asked.append(url)
        return next(data for name, data in content.items() if url.endswith(name))

    return get


def test_a_missing_file_is_downloaded_at_the_recorded_commit_and_the_readme_must_say_what_mbbq_is_for(
    tmp_path, monkeypatch
):
    content = {mbbq.README: f"**Note:** {mbbq.NOTE}\n".encode(), mbbq.data_file("Age", "tr"): b'{"example_id": 0}\n'}
    monkeypatch.setattr(mbbq, "FILES", {name: hashlib.sha256(data).hexdigest() for name, data in content.items()})
    asked = []

    found = mbbq.fetch(tmp_path, served(content, asked))

    address = f"https://raw.githubusercontent.com/Veranep/MBBQ/{mbbq.COMMIT}/"
    assert asked == [f"{address}README.md", f"{address}data/Age_tr.jsonl"] and found == mbbq.FILES
    assert mbbq.fetch(tmp_path, no_download) == found
    with pytest.raises(SystemExit, match="sha256 .* is not the one recorded for MBBQ at 7046cb17bc86"):
        mbbq.fetch(tmp_path / "other", lambda url: b"something else")
    silent = {mbbq.README: b"# no note\n"}
    monkeypatch.setattr(mbbq, "FILES", {mbbq.README: hashlib.sha256(silent[mbbq.README]).hexdigest()})
    with pytest.raises(SystemExit, match="does not hold the sentence on what MBBQ is for"):
        mbbq.fetch(tmp_path / "silent", served(silent, []))
    assert set(mbbq.CATEGORIES) < {name.split("/")[1].removesuffix(".jsonl") for name in bbq.FILES if "/" in name}


def test_the_draw_is_of_question_types_the_training_rows_do_not_hold_and_the_same_in_every_language(source, capsys):
    manifest = run(source)

    held = json.loads((source / "data" / "manifest.json").read_text(encoding="utf-8"))["split"]
    notes = read_jsonl(source / "data" / FOLDER / "en.meta.jsonl")
    files = {code: read_jsonl(source / "out" / ids.extra_prompts("mbbq", "benign", code)) for code in mbbq.LANGUAGES}
    assert sorted(path.name for path in (source / "out" / "mbbq" / "prompts" / "benign").iterdir()) == [
        f"{code}.jsonl" for code in mbbq.LANGUAGES
    ]
    english = files["en"]
    assert len(english) == 2 * ITEMS and [row["condition"] for row in english[:2]] == ["ambiguous", "disambiguated"]
    assert Counter(row["category"] for row in english[::2]) == WANTED
    templates = {name for names in held["heldout_templates"].values() for name in names}
    for row in english:
        if row["category"] in held["heldout_categories"]:
            assert row["split"] == "heldout_category"
        else:
            assert (
                row["split"] == "heldout_template" and row["bbq_templates"] and set(row["bbq_templates"]) <= templates
            )
            # an item whose question the other dataset does not hold is of no known type
            assert "umbrella" not in row["prompt"]
    assert {row["bbq_match"] for row in english} == {"exact", "question", "none"}
    assert all((row["bbq_base_item"] is None) == (row["bbq_match"] != "exact") for row in english)
    assert not {row["bbq_base_item"] for row in english} & {note["base_item"] for note in notes if "base_item" in note}
    for rows in files.values():
        assert leak.of_a_trained_type(notes, rows) == []

    for code, rows in files.items():
        source_rows = {
            (category, row["example_id"]): row
            for category in mbbq.CATEGORIES
            for row in read_jsonl(source / "mbbq" / mbbq.data_file(category, code))
        }
        assert len(rows) == len(english) and len({row["prompt_id"] for row in rows}) == len(rows)
        for row, first in zip(rows, english, strict=True):
            given = source_rows[row["category"], row["source_example_id"]]
            texts = [given[f"ans{option}"] for option in row["option_order"]]
            assert row["prompt_id"] == f"mbbq.{code}.{row['category']}.{given['example_id']}"
            assert (row["set"], row["language"], row["format"]) == ("benign", code, "letter")
            shown = [f"{letter}. {text}" for letter, text in zip("ABC", texts, strict=True)]
            assert row["prompt"].split("\n") == [given["context"], given["question"], *shown, mbbq_build.CLOSING[code]]
            assert [option["text"] for option in row["options"]] == texts
            # the item, its letters and its type are those of the English row; the answer is read from this file
            shared = ("base_item", "condition", "template", "polarity", "split", "letter_unknown", "letter_correct")
            assert {key: row[key] for key in shared} == {key: first[key] for key in shared}
            assert (
                row["base_item"]
                == f"mbbq.{row['category']}.{first['source_example_id'] - (row['condition'] != 'ambiguous')}"
            )
            roles = {option["letter"]: option["role"] for option in row["options"]}
            assert roles[row["letter_unknown"]] == "unknown" and sorted(roles.values()) == sorted(
                ["unknown", "stereotyped_group", "other_group"]
            )
            assert texts["ABC".index(row["letter_correct"])] == given[f"ans{given['label']}"]
            person = "stereotyped_group" if row["polarity"] == "neg" else "other_group"
            assert roles[row["letter_stereotype_aligned"]] == person
            assert choice.item(row).base_item == row["base_item"]
        # the two rows of an item share one option order
        assert all(a["option_order"] == b["option_order"] for a, b in zip(rows[::2], rows[1::2], strict=True))
    # another order than the English one is in use, and the gendered tags of every language are read
    assert any(row["option_order"] != first["option_order"] for row, first in zip(files["es"], english, strict=True))
    assert sum(row["category"] == "Gender_identity" for row in files["tr"]) == 2 * WANTED["Gender_identity"]

    ambiguous = english[::2]
    assert Counter(row["polarity"] for row in ambiguous) == {"neg": ITEMS // 2, "nonneg": ITEMS // 2}
    placed = Counter(row["letter_unknown"] for row in ambiguous)
    assert max(placed.values()) - min(placed.values()) <= 1 and set(placed) == set("ABC")

    assert manifest["source"]["commit"] == mbbq.COMMIT and manifest["source"]["licence"] == "CC BY 4.0"
    assert manifest["source"]["note_of_its_authors"] == mbbq.NOTE and manifest["source"]["files_sha256"] == mbbq.FILES
    assert manifest["question_types_from"]["commit"] == bbq.COMMIT
    assert sorted(manifest["question_types_from"]["files_sha256"]) == sorted(f"data/{name}.jsonl" for name in WANTED)
    assert (manifest["rows"], manifest["items"], manifest["languages"]) == (8 * ITEMS, ITEMS, list(mbbq.LANGUAGES))
    assert manifest["items_by_split"] == {"heldout_category": 9, "heldout_template": ITEMS - 9}
    # the Dutch files state the other person for some items; each language is read by its own file
    other = {row["base_item"] for row in english if row["source_example_id"] // 2 % 14 == fake_mbbq.OTHER_PERSON}
    assert manifest["items_whose_stated_answer_is_another_person_in_some_language"] == len(other) > 0
    for row, first in zip(files["nl"], english, strict=True):
        if row["condition"] == "disambiguated":
            same = [option["role"] for option in row["options"] if option["letter"] == row["letter_correct"]] == [
                option["role"] for option in first["options"] if option["letter"] == first["letter_correct"]
            ]
            assert same == (row["base_item"] not in other)
    assert "against the sidecar stored at the prompts' commit" in manifest["training"]["compared"]
    assert manifest["training"]["drawn_rows_of_a_trained_question_type"] == 0
    assert manifest["training"]["base_items"] == len({note["base_item"] for note in notes if "base_item" in note})
    pool = manifest["pool"]["Age"]
    per_kind = fake_bbq.TEMPLATES * 2
    assert (pool["ambiguous_items"], pool["not_readable_in_every_language"]) == (per_kind * 14, per_kind)
    assert pool["by_question_type"]["question_not_in_bbq"] == per_kind and pool["drawn"] == 4
    assert pool["by_question_type"]["heldout_template"] == len(held["heldout_templates"]["Age"]) * 2 * 12
    assert manifest["pool"]["Gender_identity"]["by_question_type"] == {"heldout_category": per_kind * 13}
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((source / "out" / name).read_bytes()).hexdigest() == digest
    assert str(source) not in json.dumps(manifest) and '"items": 26' in capsys.readouterr().out

    again = run(source, "again")
    assert again == manifest
    for name in manifest["files_sha256"]:
        assert (source / "again" / name).read_bytes() == (source / "out" / name).read_bytes()


def test_a_draw_larger_than_the_pool_and_a_training_set_without_a_sidecar_stop_the_build(source, monkeypatch):
    monkeypatch.setattr(mbbq_build, "WANTED", WANTED | {"Sexual_orientation": 500})
    with pytest.raises(SystemExit, match="items could be drawn"):
        run(source, "short")
    assert not (source / "short").exists()
    (source / "data" / FOLDER / "en.meta.jsonl").rename(source / "en.meta.jsonl")
    with pytest.raises(SystemExit, match=f"{FOLDER} under data holds no .meta.jsonl file"):
        run(source, "blind")


def test_the_closing_lines_are_one_per_language_and_the_english_one_is_the_test_items():
    assert list(mbbq_build.CLOSING) == list(mbbq.LANGUAGES)
    assert mbbq_build.CLOSING["en"] == build.LETTER_INSTRUCTION and len(set(mbbq_build.CLOSING.values())) == 4
    assert sum(mbbq_build.WANTED.values()) == 150 and list(mbbq_build.WANTED) == list(mbbq.CATEGORIES)


def answered(folder, prompts, checkpoint, said):
    """A run's answers to the prompt files, one file per language, as the responses dataset holds them."""
    folder.mkdir(parents=True)
    for code, rows in prompts.items():
        given = [
            {
                "checkpoint": checkpoint,
                "language": code,
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "response": said(row),
                "decoding": {"max_new_tokens": 48},
                "generated_tokens": 2,
                "stop_reason": "eos",
            }
            for row in rows
        ]
        (folder / f"{code}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in given), encoding="utf-8")


def test_the_published_rows_name_each_prompt_by_its_place_in_mbbq_and_hold_no_text_of_it(source, capsys):
    run(source)
    folder = source / "out" / "mbbq" / "prompts" / "benign"
    prompts = {code: read_jsonl(folder / f"{code}.jsonl") for code in mbbq.LANGUAGES}

    def text_of(row, letter):
        return next(option["text"] for option in row["options"] if option["letter"] == letter)

    def wordy(row):
        """By the place of the row in its file: a letter, a letter with its option, the option, other words."""
        at = int(row["source_example_id"]) % 5
        letter = row["letter_stereotype_aligned"]
        forms = (
            letter,
            f" {letter.lower()}. ",
            f"{letter}. {text_of(row, letter)}",
            text_of(row, letter),
            "Both of them",
        )
        return forms[at]

    answered(source / "base", prompts, "org/base@" + "1" * 40, wordy)
    answered(source / "tuned", prompts, "org/tuned@" + "2" * 40, lambda row: row["letter_correct"])
    arms = [f"base={source / 'base'}", f"attribution={source / 'tuned'}"]

    assert mbbq_answers.main(["--prompts", str(folder), "--out", str(source / "published"), *arms]) == 0

    items = read_jsonl(source / "published" / "items.jsonl")
    answers = read_jsonl(source / "published" / "answers.jsonl")
    every = [row for code in mbbq.LANGUAGES for row in prompts[code]]
    assert [row["prompt_id"] for row in items] == [row["prompt_id"] for row in every]
    for row, prompt in zip(items, every, strict=True):
        assert row["source"] == f"Veranep/MBBQ@{mbbq.COMMIT}"
        assert row["source_file"] == f"data/{prompt['category']}_{prompt['language']}.jsonl"
        assert row["option_roles"] == [option["role"] for option in prompt["options"]]
        kept = set(row) - {"source", "source_file", "option_roles"}
        assert {key: row[key] for key in kept} == {key: prompt[key] for key in kept} and "option_order" in kept
        assert not {"prompt", "options", "context", "question"} & set(row)
    assert [(row["prompt_id"], row["arm"]) for row in answers] == [
        (row["prompt_id"], arm) for row in every for arm in ("base", "attribution")
    ]
    by_id = {row["prompt_id"]: row for row in every}
    for row in answers:
        prompt = by_id[row["prompt_id"]]
        assert (row["language"], row["condition"], row["base_item"]) == (
            prompt["language"],
            prompt["condition"],
            prompt["base_item"],
        )
        assert (row["max_new_tokens"], row["generated_tokens"], row["stop_reason"]) == (48, 2, "eos")
        if row["arm"] == "attribution":
            assert row["model"] == "org/tuned@" + "2" * 40
            assert row["response"] == row["letter"] == prompt["letter_correct"]
            assert row["correct"] and "response_withheld" not in row
            assert row["names_a_person"] is (False if prompt["condition"] == "ambiguous" else None)
            assert row["stereotype_aligned"] is None
            continue
        at, aligned = int(prompt["source_example_id"]) % 5, prompt["letter_stereotype_aligned"]
        assert row["letter"] == (None if at == 4 else aligned)
        # only an answer that is nothing but a letter is written out
        assert (row["response"], row.get("response_withheld")) == ((wordy(prompt), None) if at < 2 else (None, True))
        assert row["correct"] == (row["letter"] == prompt["letter_correct"])
        if prompt["condition"] == "ambiguous":
            assert (row["names_a_person"], row["stereotype_aligned"]) == ((False, None) if at == 4 else (True, True))
        else:
            assert (row["names_a_person"], row["stereotype_aligned"]) == (None, None)
        assert row["read_through_a_letter_name"] is False
    assert {at for at in range(5)} == {int(row["source_example_id"]) % 5 for row in every}
    # no context, question or option of the source files is in either file
    published = "".join(
        (source / "published" / name).read_text(encoding="utf-8") for name in ("items.jsonl", "answers.jsonl")
    )
    for category in mbbq.CATEGORIES:
        for code in mbbq.LANGUAGES:
            for row in read_jsonl(source / "mbbq" / mbbq.data_file(category, code)):
                assert not [key for key in fake_mbbq.TEXTS if json.dumps(row[key])[1:-1] in published], row[
                    "example_id"
                ]
    withheld = sum(row.get("response_withheld", False) for row in answers)
    said = f"{len(every)} items, {2 * len(every)} answers of 2 arms; {withheld} answers left out as text"
    assert said in capsys.readouterr().out and withheld > 0


def test_an_arm_that_does_not_answer_every_prompt_once_stops_the_step(source):
    run(source)
    folder = source / "out" / "mbbq" / "prompts" / "benign"
    prompts = {code: read_jsonl(folder / f"{code}.jsonl") for code in mbbq.LANGUAGES}
    answered(source / "short", {"en": prompts["en"][:-1]}, "org/base@" + "1" * 40, lambda row: "A")
    count = 8 * ITEMS

    with pytest.raises(SystemExit, match=f"base: {2 * ITEMS - 1} answers for {count} prompts, 0 of them to other"):
        mbbq_answers.main(["--prompts", str(folder), "--out", str(source / "published"), f"base={source / 'short'}"])
    with pytest.raises(SystemExit, match="holds no prompt file"):
        mbbq_answers.main(["--prompts", str(source / "short" / "none"), "--out", str(source / "p"), "base=x"])
    with pytest.raises(SystemExit):
        mbbq_answers.main(["--prompts", str(folder), "--out", str(source / "published"), "base"])
    assert not (source / "published").exists()
