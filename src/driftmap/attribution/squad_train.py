"""The training set ``squad``: the probe's own train questions as rows for a fine-tune that answers or abstains.

    python -I -m driftmap.attribution.squad_train --train OUT/squad_train_in.jsonl \
        --passed CURATED/squad_passed.json --benign DATA/sft_train_defense --out DATA

``--train`` is the train prompt file of ``probe_squad`` and ``--passed`` the ids ``probe_curator`` let through:
only those questions become rows. The prompt of a row is the prompt the probe study asks, unchanged. The target of
an answerable question is the first of its gold answers, in their sorted order; the target of an unanswerable one
is one fixed sentence, a trained abstention sentence of the attribution fine-tune. The benign rows of the training
set under ``--benign`` are added after them as they are, told by its sidecar, so the set is built as that one is.

Written under ``--out`` in ``sft_train_squad/``: ``en.jsonl`` (a user prompt and an assistant completion per row),
``en.meta.jsonl`` (what each row is, line by line: the question's id, split and condition, or ``benign``) and
``manifest.json`` (counts, hashes, the gate's counts). ``en.jsonl`` holds SQuAD passages, questions and answers and
belongs in the working copy of the dataset only; the sidecar and the manifest hold no text of SQuAD.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution import build, probe, probe_curator, probe_squad, reading
from driftmap.core import ids
from driftmap.scripts.drift import META

NAME = "squad"
ABSTENTION = build.SENTENCES["Cannot be determined"]
MANIFEST = "manifest.json"
SCHEMA = "driftmap.squad-train/1"


def target(row: dict) -> str:
    """What the fine-tune is to answer: a gold answer, or the abstention sentence for an unanswerable question."""
    return ABSTENTION if row["condition"] == probe_squad.CONDITIONS[0] else row["gold_answers"][0]


def turns(prompt: str, completion: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": completion}],
    }


def benign(folder: Path) -> list[tuple[dict, dict]]:
    """The benign rows of a training set with their sidecar lines; stops when the sidecar does not fit the rows."""
    rows, notes = probe.read_jsonl(folder / "en.jsonl"), probe.read_jsonl(folder / f"en{META}")
    if len(rows) != len(notes):
        raise SystemExit(f"en{META} holds {len(notes)} lines for the {len(rows)} rows of {folder.name}/en.jsonl")
    return [(row, note) for row, note in zip(rows, notes, strict=True) if note.get("kind") == "benign"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=Path, required=True, help="train prompt file of the probe study")
    parser.add_argument("--passed", type=Path, required=True, help="the ids that passed the CuratorKIT gate")
    parser.add_argument(
        "--benign", type=Path, required=True, help="folder of a training set whose benign rows are added"
    )
    parser.add_argument("--out", type=Path, required=True, help="folder the training set's folder is written under")
    args = parser.parse_args(argv)
    asked, gate = probe_curator.passing(probe.read_jsonl(args.train), args.train, args.passed)
    lacking = [row["id"] for row in asked if row["condition"] == probe_squad.CONDITIONS[1] and not row["gold_answers"]]
    if lacking:
        raise SystemExit(f"{len(lacking)} answerable questions of {args.train.name} have no gold answer")
    added = benign(args.benign)
    rows = [turns(row["prompt"], target(row)) for row in asked] + [row for row, _ in added]
    notes = [{"kind": NAME, "id": row["id"], "split": "train", "condition": row["condition"]} for row in asked]
    notes += [note for _, note in added]
    folder = args.out / ids.training_folder(NAME)
    folder.mkdir(parents=True, exist_ok=True)
    probe.write_jsonl(folder / "en.jsonl", rows)
    probe.write_jsonl(folder / f"en{META}", notes)
    manifest = {
        "schema": SCHEMA,
        "train_set": NAME,
        "source": {"dataset": reading.SQUAD, "revision": reading.SQUAD_REVISION, "licence": reading.LICENCE},
        "rows": {
            "all": len(rows),
            NAME: len(asked),
            "by_condition": dict(Counter(row["condition"] for row in asked)),
            "articles": len({row["title"] for row in asked}),
            "benign": len(added),
        },
        "prompt": "the prompt the probe study asks: passage, question, closing line",
        "targets": {
            "answerable": "the first of the question's gold answers, in their sorted order",
            "unanswerable": ABSTENTION,
        },
        "benign_rows_from": args.benign.name,
        "questions_from": {"file": args.train.name, "sha256": sha256(args.train)},
        "curatorkit_gate": gate,
        "files": {name: sha256(folder / name) for name in ("en.jsonl", f"en{META}")},
        "text": {
            "en.jsonl": "holds SQuAD passages, questions and answers; kept in the working copy of the dataset only",
            f"en{META}": "ids, split and condition, no text; the public form of the set",
        },
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rows", "curatorkit_gate", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
