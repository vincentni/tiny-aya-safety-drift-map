"""The prompt group ``squad``: its files, what a launch reads of them, and what it refuses."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from driftmap import cli
from driftmap.attribution import squad_prompts, squad_train
from driftmap.scripts import generate
from tests.fakes.squad_probe import gated, put, read_jsonl, squad_sample

COMMIT = "a" * 40


class Hub:
    """A dataset repo whose one commit is a folder."""

    def __init__(self, folder):
        self.folder = folder

    def snapshot_download(self, ref, local_dir, repo_type, allow_patterns):
        assert (str(ref), repo_type) == (f"owner/eval@{COMMIT}", "dataset")
        for pattern in allow_patterns:
            for path in self.folder.glob(pattern):
                target = Path(local_dir) / path.relative_to(self.folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
        return local_dir


@pytest.fixture
def built(tmp_path):
    """A train sample, two dev samples of other articles, the gate's passing ids, and both builders' output."""
    samples = {
        "train": squad_sample("train", 3, 4),
        "dev": squad_sample("dev", 2, 4),
        "dev2": squad_sample("more", 2, 4),
    }
    files = {}
    for split, rows in samples.items():
        put(tmp_path / f"squad_{split}_in.jsonl", rows)
        files[tmp_path / f"squad_{split}_in.jsonl"] = ("train" if split == "train" else "test", [])
    files[tmp_path / "squad_dev_in.jsonl"] = ("test", [samples["dev"][-1]["id"]])
    passed = gated(tmp_path / "squad_passed.json", files)
    other = tmp_path / "sft_train_defense"
    other.mkdir()
    put(other / "en.jsonl", [squad_train.turns("a request", "a reply")])
    put(other / "en.meta.jsonl", [{"kind": "benign"}])
    data = tmp_path / "data"
    argv = [f"--split={split}={tmp_path / f'squad_{split}_in.jsonl'}" for split in samples]
    assert squad_prompts.main([*argv, "--passed", str(passed), "--out", str(data)]) == 0
    train = ["--train", str(tmp_path / "squad_train_in.jsonl"), "--passed", str(passed), "--benign", str(other)]
    assert squad_train.main([*train, "--out", str(data)]) == 0
    return tmp_path, data, samples


def check(data):
    settings = SimpleNamespace(owner="owner", repos=SimpleNamespace(eval="eval"))
    parsed = argparse.Namespace(sets="squad", sets_rev=COMMIT)
    return cli._squad_missing(parsed, settings, Hub(data))


def test_each_split_is_a_file_of_prompt_rows_under_ids_of_the_split_and_the_manifest_holds_no_text(built, capsys):
    tmp_path, data, samples = built

    folder = data / "squad" / "prompts" / "benign"
    assert sorted(path.name for path in folder.iterdir()) == ["dev.jsonl", "dev2.jsonl", "train.jsonl"]
    dev = read_jsonl(folder / "dev.jsonl")
    first = samples["dev"][0]
    # the question the gate dropped is in no file
    assert len(dev) == 7 and len(read_jsonl(folder / "train.jsonl")) == 12
    assert dev[0] == {
        "prompt_id": f"squad.dev.{first['id'].removeprefix('squad.')}",
        "prompt": first["prompt"],
        "set": "benign",
        "language": "dev",
        "id": first["id"],
        "condition": "unanswerable",
    } | squad_train.places(first)
    manifest = json.loads((data / "squad" / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["splits"]) == ["train", "dev", "dev2"]
    assert manifest["splits"]["dev"]["file"] == "squad/prompts/benign/dev.jsonl"
    assert (manifest["splits"]["dev"]["rows"], manifest["splits"]["dev"]["articles"]) == (7, 2)
    assert manifest["splits"]["dev"]["curatorkit_gate"]["dropped"] == {"row_shape": 0, "schema_gate": 1}
    said = (data / "squad" / "manifest.json").read_text(encoding="utf-8")
    hidden = [
        text for rows in samples.values() for row in rows for text in (row["context"], row["question"], row["title"])
    ]
    assert not any(text in said for text in hidden)

    # the generate job reads the three files as units of the group
    units = generate.read_variant_units(data, "checkpoint", "full", None, extra="squad")
    assert [(unit.set, unit.language, len(unit.rows)) for unit in units] == [
        ("benign", "dev", 7),
        ("benign", "dev2", 8),
        ("benign", "train", 12),
    ]
    assert units[0].key == "squad/responses/checkpoint/benign/dev.jsonl"
    with pytest.raises(SystemExit):
        squad_prompts.main(["--split", str(tmp_path / "squad_dev_in.jsonl"), "--passed", "x", "--out", str(data)])
    assert "a split is given as NAME=FILE" in capsys.readouterr().err


def test_a_launch_reads_the_files_and_holds_the_other_splits_against_the_training_sidecar(built, capsys):
    _, data, _ = built

    assert check(data) == []

    out = capsys.readouterr().out
    assert f"squad prompts at {COMMIT[:12]}: 27 rows (benign/dev 7, benign/dev2 8, benign/train 12)" in out
    assert (
        "none of the 15 rows outside train is on an article or a passage of the 12 questions of sft_train_squad" in out
    )


def test_a_question_of_another_split_on_a_trained_article_or_passage_is_refused(built):
    _, data, _ = built
    folder = data / "squad" / "prompts" / "benign"
    train, dev = read_jsonl(folder / "train.jsonl"), read_jsonl(folder / "dev2.jsonl")
    apart = "squad questions apart from the articles and passages of sft_train_squad"

    put(folder / "dev2.jsonl", [*dev[:2], dev[2] | {"article": train[0]["article"]}, *dev[3:]])
    assert check(data) == [
        f"{apart}: 1 rows of squad/prompts/benign/dev2.jsonl are on an article or a passage that was trained on, "
        "the first is row 3"
    ]
    put(folder / "dev2.jsonl", [dev[0] | {"passage": train[5]["passage"]}, *dev[1:]])
    assert "the first is row 1" in check(data)[0]
    put(folder / "dev2.jsonl", [{key: value for key, value in row.items() if key != "article"} for row in dev])
    assert check(data) == [f"{apart}: a row at that commit does not name its article and its passage"]
    put(folder / "dev2.jsonl", dev)
    # a row filed under another split than its file's is not a row the job reads
    put(folder / "dev.jsonl", [row | {"language": "dev2"} for row in read_jsonl(folder / "dev.jsonl")])
    assert check(data)[0].startswith("squad prompts the job can read: ")


def test_a_commit_without_the_training_sidecar_is_refused(built):
    _, data, _ = built
    apart = "squad questions apart from the articles and passages of sft_train_squad"
    sidecar = data / "sft_train_squad" / "en.meta.jsonl"

    put(sidecar, [{"kind": "squad", "id": "squad.x", "split": "train", "condition": "answerable"}])
    assert check(data) == [f"{apart}: a row at that commit does not name its article and its passage"]
    put(sidecar, [{"kind": "benign"}])
    assert check(data) == [f"{apart}: no training sidecar at that commit says what was trained on"]
    sidecar.unlink()
    assert check(data) == [f"{apart}: no training sidecar at that commit says what was trained on"]
