"""The prompt group ``squad``: the questions of the probe study's SQuAD samples as prompt files for a generate job.

    python -I -m driftmap.attribution.squad_prompts --split train=OUT/squad_train_in.jsonl \
        --split dev=OUT/squad_dev_in.jsonl --split dev2=OUT/squad_dev2_in.jsonl \
        --passed CURATED/squad_passed.json --out DATA

Each ``--split`` names a prompt file of ``probe_squad``; only the questions ``probe_curator`` let through become
rows. Written under ``--out``: ``squad/prompts/benign/<split>.jsonl``, a row per question with the prompt the probe
study asks, unchanged, under the id ``squad.<split>.<question id>``; and ``squad/manifest.json`` (counts and
hashes). A row names its article and its passage by a hash, so a launch can see that no question of another split
sits on an article or a passage of the training set ``squad`` without reading a title. The prompt files hold
SQuAD passages and questions and belong in the working copy of the dataset only; the manifest holds no text.

``driftmap launch generate -- --sets squad --sets-rev SHA --max-new-tokens 48`` answers the files of that commit.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution import probe, probe_curator, reading, squad_train
from driftmap.core import ids

GROUP = "squad"
SET = "benign"
MANIFEST = "manifest.json"
SCHEMA = "driftmap.squad-prompts/1"


def rows_of(split: str, asked: Sequence[dict]) -> list[dict]:
    """The prompt rows of one split, filed under the group's set and the split as their code."""
    return [
        {
            "prompt_id": f"{GROUP}.{split}.{row['id'].removeprefix('squad.')}",
            "prompt": row["prompt"],
            "set": SET,
            "language": split,
            "id": row["id"],
            "condition": row["condition"],
        }
        | squad_train.places(row)
        for row in asked
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", action="append", required=True, metavar="NAME=FILE", help="a split's prompt file")
    parser.add_argument("--passed", type=Path, required=True, help="the ids that passed the CuratorKIT gate")
    parser.add_argument("--out", type=Path, required=True, help="folder the group's folder is written under")
    args = parser.parse_args(argv)
    named = dict(pair.split("=", 1) for pair in args.split if "=" in pair)
    if len(named) != len(args.split) or not all(name.isalnum() for name in named):
        parser.error("a split is given as NAME=FILE, once per name")
    splits = {}
    for split, file in named.items():
        path = Path(file)
        asked, gate = probe_curator.passing(probe.read_jsonl(path), path, args.passed)
        rows = rows_of(split, asked)
        target = args.out / ids.extra_prompts(GROUP, SET, split)
        probe.write_jsonl(target, rows)
        splits[split] = {
            "file": f"{ids.extra_prompts(GROUP, SET, split)}",
            "sha256": squad_train.sha256(target),
            "rows": len(rows),
            "by_condition": dict(Counter(row["condition"] for row in rows)),
            "articles": len({row["article"] for row in rows}),
            "passages": len({row["passage"] for row in rows}),
            "questions_from": {"file": path.name, "sha256": squad_train.sha256(path)},
            "curatorkit_gate": gate,
        }
    manifest = {
        "schema": SCHEMA,
        "group": GROUP,
        "source": {"dataset": reading.SQUAD, "revision": reading.SQUAD_REVISION, "licence": reading.LICENCE},
        "prompt": "the prompt the probe study asks: passage, question, closing line",
        "splits": splits,
        "text": "the prompt files hold SQuAD passages and questions and are kept in the working copy of the dataset "
        "only; article and passage are named by a hash",
    }
    (args.out / GROUP / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {split: {key: found[key] for key in ("rows", "articles", "passages")} for split, found in splits.items()}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
