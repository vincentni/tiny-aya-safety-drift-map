"""Answer lengths of models on the evaluation prompts, from the stored answers.

    python -I -m driftmap.attribution.answer_lengths RESPONSES_DIR OUT.json \
        --model "base model=FOLDER" --model "attribution fine-tune=FOLDER"

RESPONSES_DIR is a copy of the responses dataset; the answers under the served template are read from
responses/<folder>/<set>/<language>.jsonl, where the folder is the one of a model's checkpoint. Per model and
prompt set: the rows, the mean of ``generated_tokens``, the answers that stopped at the token limit, and the
answers that are, word for word, one of the sentences the attribution fine-tune was trained to abstain with. No
prompt and no answer text is written.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution.build import SENTENCES
from driftmap.gen import MAX_NEW_TOKENS

SCHEMA = "driftmap.check.answer_lengths/1"
SETS = {"harmful": "red-teaming prompts", "xstest": "XSTest prompts", "benign_heldout": "held-out benign prompts"}
# the free-form abstention sentences of the attribution training rows
TRAINED = tuple(SENTENCES.values())


def measured(rows: Sequence[dict]) -> dict:
    """The counts of one model's answers to one prompt set."""
    return {
        "rows": len(rows),
        "mean_generated_tokens": round(sum(row["generated_tokens"] for row in rows) / len(rows), 1),
        "stopped_at_the_token_limit": sum(row["stop_reason"] == "length" for row in rows),
        "training_abstention_sentences": sum((row["response"] or "").strip() in TRAINED for row in rows),
    }


def model_report(root: Path, name: str, folder: str) -> dict:
    """One model's entry: its checkpoint and its counts per prompt set; stops on a set without answers or with the
    answers of more than one checkpoint."""
    sets, checkpoints = [], set()
    for key, shown in SETS.items():
        rows = [
            json.loads(line)
            for path in sorted((root / "responses" / folder / key).glob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        if not rows:
            raise SystemExit(f"{name}, {key}: no answers under responses/{folder}/{key}")
        checkpoints = {row["checkpoint"] for row in rows}
        if len(checkpoints) != 1:
            raise SystemExit(f"{name}, {key}: the answers are of {len(checkpoints)} checkpoints")
        sets.append({"set": key, "prompts": shown} | measured(rows))
    return {"model": name, "checkpoint": checkpoints.pop(), "folder": f"responses/{folder}", "sets": sets}


def _named(text: str) -> tuple[str, str]:
    name, _, folder = text.partition("=")
    if not name or not folder:
        raise argparse.ArgumentTypeError("give a model as NAME=FOLDER")
    return name, folder


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("responses", type=Path, help="a copy of the responses dataset")
    parser.add_argument("out", type=Path, help="the report, JSON")
    parser.add_argument("--model", type=_named, action="append", metavar="NAME=FOLDER", help="a model to count")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("name at least one model: --model NAME=FOLDER")
    models = [model_report(args.responses, name, folder) for name, folder in args.model]
    report = {
        "schema": SCHEMA,
        "what": "answer lengths under the served template, from the stored answers of the responses dataset",
        "fields": {
            "mean_generated_tokens": "mean of generated_tokens over the rows; generation stops at "
            f"{MAX_NEW_TOKENS} new tokens",
            "stopped_at_the_token_limit": "rows with stop_reason length",
            "training_abstention_sentences": "rows whose whole answer is one of the listed sentences",
        },
        "abstention_sentences": list(TRAINED),
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for model in models:
        print(model["model"], [tuple(found[key] for key in found if key != "prompts") for found in model["sets"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
