"""The MBBQ question pairs and the answers to them in the shape that is published: ids, and no text of MBBQ.

    python -I -m driftmap.attribution.mbbq_answers --prompts PROMPTS --out OUT NAME=FOLDER [NAME=FOLDER ...]

``--prompts`` is the folder of the prompt files ``mbbq_build`` wrote, one per language. Each ``NAME=FOLDER`` is an
arm: a name for it and a folder of one run's answers to those prompts, one file per language, as the responses
dataset holds them.

items.jsonl names every prompt by the row of MBBQ it is made of (the file, the row's id, and the order its
options are shown in) and by what the draw says of it; with the pinned files of MBBQ a prompt can be written again
from it. answers.jsonl holds, per arm and prompt, the letter the answer is read as by the rule of
``driftmap table choice`` and what the rates count of it. The context, the question and the options are in
neither file. The answer of a model is written only when it is nothing but a letter, and so no text of the
options; any other answer is left out and its row says so.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution import mbbq
from driftmap.attribution.build import read_jsonl
from driftmap.table import choice

RULE = choice.LetterRule("".join(choice.LETTERS), choice.NAMES)
# what an item row keeps of its prompt row as it is
KEPT = (
    "prompt_id",
    "language",
    "base_item",
    "condition",
    "category",
    "template",
    "polarity",
    "split",
    "source_example_id",
    "option_order",
    "letter_unknown",
    "letter_correct",
    "letter_stereotype_aligned",
    "bbq_match",
    "bbq_base_item",
    "bbq_templates",
)
# what an answer row keeps of the run that gave it
RUN_FIELDS = ("generated_tokens", "stop_reason")


def item_row(row: dict) -> dict:
    """The published row of one prompt: where it is in MBBQ, and what the draw says of it."""
    source = {
        "source": f"{mbbq.REPOSITORY}@{mbbq.COMMIT}",
        "source_file": mbbq.data_file(row["category"], row["language"]),
        # what the option at A, B and C stands for
        "option_roles": [option["role"] for option in row["options"]],
    }
    return {key: row[key] for key in KEPT} | source


def answer_row(row: dict, arm: str, given: dict) -> dict:
    """The published row of one answer: the letter it is read as and what the rates count of it."""
    response = given["response"]
    letter = RULE.parse(response, {option["letter"]: option["text"] for option in row["options"]})
    open_ = row["condition"] == choice.AMBIGUOUS
    person = letter not in (None, row["letter_unknown"])
    only_a_letter = not (response or "").strip() or RULE.alone(response) is not None
    return (
        {key: row[key] for key in ("prompt_id", "language", "base_item", "condition")}
        | {"arm": arm, "model": given["checkpoint"]}
        | ({"response": response} if only_a_letter else {"response": None, "response_withheld": True})
        | {
            "letter": letter,
            "read_through_a_letter_name": RULE.by_name(response) is not None,
            "names_a_person": person if open_ else None,
            "stereotype_aligned": letter == row["letter_stereotype_aligned"] if open_ and person else None,
            "correct": letter == row["letter_correct"],
            "max_new_tokens": given["decoding"]["max_new_tokens"],
        }
        | {key: given[key] for key in RUN_FIELDS}
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts", type=Path, required=True, help="folder of the prompt files, one per language")
    parser.add_argument("--out", type=Path, required=True, help="folder for items.jsonl and answers.jsonl")
    parser.add_argument("arms", nargs="+", metavar="NAME=FOLDER", help="an arm and the folder of its run's answers")
    args = parser.parse_args(argv)
    prompts = {row["prompt_id"]: row for path in sorted(args.prompts.glob("*.jsonl")) for row in read_jsonl(path)}
    if not prompts:
        raise SystemExit(f"{args.prompts.name} holds no prompt file")
    answers = []
    for named in args.arms:
        arm, _, folder = named.partition("=")
        if not arm or not folder:
            parser.error(f"{named!r} is not NAME=FOLDER")
        given = [row for path in sorted(Path(folder).glob("*.jsonl")) for row in read_jsonl(path)]
        found = [row["prompt_id"] for row in given]
        if sorted(found) != sorted(prompts):
            other = len(set(found) - set(prompts))
            raise SystemExit(
                f"{arm}: {len(found)} answers for {len(prompts)} prompts, {other} of them to other prompts"
            )
        answers += [answer_row(prompts[row["prompt_id"]], arm, row) for row in given]
    order = {prompt_id: at for at, prompt_id in enumerate(prompts)}
    answers.sort(key=lambda row: order[row["prompt_id"]])
    write_jsonl(args.out / "items.jsonl", [item_row(row) for row in prompts.values()])
    write_jsonl(args.out / "answers.jsonl", answers)
    left_out = sum(row.get("response_withheld", False) for row in answers)
    print(f"{len(prompts)} items, {len(answers)} answers of {len(args.arms)} arms; {left_out} answers left out as text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
