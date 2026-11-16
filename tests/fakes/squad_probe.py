"""Made-up SQuAD samples for the probe's tests: questions, answers, hidden states and the gate's passing ids."""

import hashlib
import json

import numpy as np

from driftmap.attribution import build

LAYERS = (4, 8)
TRAINED = build.SENTENCES["Cannot be determined"]


class Words:
    """A token per word."""

    def __call__(self, text, add_special_tokens):
        return {"input_ids": text.split()}


def put(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def states_of(rows, noise, folder):
    """Hidden states of two layers: the first holds nothing; in the second the third unit tells an open row from
    a settled one and the fifth a row the base model has wrong from one it has right."""
    rng = np.random.default_rng(7)
    held = rng.normal(scale=noise, size=(len(rows), 2, 6)).astype(np.float32)
    held[:, 1, 2] += [6.0 if row["open"] else -6.0 for row in rows]
    held[:, 1, 4] += [6.0 if row["wrong"] else -6.0 for row in rows]
    folder.mkdir(parents=True)
    half = len(rows) // 2
    for name, part in ((0, slice(0, half)), (1, slice(half, None))):
        ids_ = np.array([row["id"] for row in rows[part]])
        np.savez(folder / f"{name}.npz", ids=ids_, states=held[part], layers=np.array(LAYERS))


def squad_sample(name, articles, per_article):
    """Questions of ``articles`` articles, half unanswerable; the base model errs on every third question."""
    rows = []
    for article in range(articles):
        for at in range(per_article):
            unanswerable, wrong = at % 2 == 0, at % 3 == 0
            context = f"{name} passage {article}-{at} " + (
                "of a few words" if at % 4 < 2 else "of some more words here"
            )
            if unanswerable:
                said = f"It is thing {at}." if wrong else TRAINED
            else:
                said = "It is something else." if wrong else f"It is gold {article}-{at}, as said."
            rows.append(
                {
                    "id": f"squad.{name}{article:02d}{at:02d}",
                    "condition": "unanswerable" if unanswerable else "answerable",
                    "context": context,
                    "question": f"What does question {article}-{at} of {name} ask?",
                    "gold_answers": [] if unanswerable else [f"gold {article}-{at}"],
                    "title": f"{name} article {article}",
                    "prompt": "\n".join(
                        [context, f"What does question {article}-{at} of {name} ask?", "Answer briefly."]
                    ),
                    "open": unanswerable,
                    "wrong": wrong,
                    "said": said,
                }
            )
    return rows


def gated(path, files):
    """The passing ids of the gate: every row of each file but the ids given for it, dropped by the schema gate."""
    held = {"schema": "driftmap.probe-squad-passed/1", "files": {}}
    for asked, (role, without) in files.items():
        ids_ = [row["id"] for row in read_jsonl(asked)]
        dropped = {"row_shape": 0, "schema_gate": len(without)}
        held["files"][asked.name] = {
            "role": role,
            "sha256": hashlib.sha256(asked.read_bytes()).hexdigest(),
            "rows_in": len(ids_),
            "dropped": dropped,
            "rows_passed": len(ids_) - len(without),
            "passed": [name for name in ids_ if name not in without],
        }
    path.write_text(json.dumps(held), encoding="utf-8")
    return path
