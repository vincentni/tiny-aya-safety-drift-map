"""The answers of the reading check in the shape that is published: source ids, and no text of the source datasets.

    python -I -m driftmap.attribution.reading_answers belebele --sources SOURCES --base BASE.jsonl \
        --fine-tuned TUNED.jsonl --out OUT
    python -I -m driftmap.attribution.reading_answers squad --base BASE.jsonl --fine-tuned TUNED.jsonl --out OUT

A row names its question by the id the source dataset gives it and by the dataset's pinned revision. The passage,
the question, the options and the gold answers are left out: they are read from the source. ``belebele`` writes
belebele_answers.jsonl and needs the source files, fetched and checked as ``reading`` does, to say which of the
source's answers sits under each letter. ``squad`` writes squad_answers.jsonl.

The answer of a model is written only where it can hold no text of the source: for Belebele when it is nothing but
a letter or the name of one, for SQuAD 2.0 when it is one of the trained abstention sentences. Any other answer is
left out and its row says so; what the scores read from it (the letter, the wording) stays on the row. The reason
is the free-form answers to SQuAD 2.0: they quote the passages, and the passages are not redistributed here.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from driftmap.attribution import pinned, reading
from driftmap.attribution.reading_scores import RULE, TRAINED, abstains, by_name, holds_gold, letter_of, read_jsonl

ARMS = ("base", "fine_tuned")
# what a row keeps of the run that answered it
RUN_FIELDS = ("generated_tokens", "stop_reason", "max_new_tokens", "preamble")
LETTERS = reading.LETTERS


def only_a_letter(row: dict) -> bool:
    """Whether a Belebele answer is nothing but a letter or the name of one, and so no text of the source."""
    response = row["response"] or ""
    if not response.strip():
        return True
    if any(entry["text"].strip().casefold() == response.strip().casefold() for entry in row["options"]):
        return False
    return RULE.alone(response) is not None or by_name(row) is not None


def shown(response: str | None, publishable: bool) -> dict:
    """The answer as a row holds it: the text, or nothing and the word that it was left out."""
    return {"response": response} if publishable else {"response": None, "response_withheld": True}


def option_order(row: dict, source: dict) -> list[int]:
    """The source's answer numbers at A to D; stops on an option the source question does not have."""
    left = {number: source[f"mc_answer{number}"] for number in range(1, len(LETTERS) + 1)}
    order = []
    for entry in sorted(row["options"], key=lambda entry: entry["letter"]):
        number = next((number for number, text in left.items() if text == entry["text"]), None)
        if number is None:
            raise SystemExit(f"{row['id']}: option {entry['letter']} is not an answer of the source question")
        del left[number]
        order.append(number)
    return order


def belebele_row(row: dict, arm: str, sources: dict[str, dict[tuple[str, int], dict]]) -> dict:
    """The published row of one Belebele answer."""
    link, _, number = row["item"].rpartition("#")
    source = sources[row["language"]].get((link, int(number)))
    if source is None:
        raise SystemExit(f"{row['id']}: the source holds no question {number} at {link}")
    letter = letter_of(row)
    return (
        {
            "id": row["id"],
            "arm": arm,
            "model": row["model"],
            "source": f"{reading.BELEBELE}@{reading.BELEBELE_REVISION}",
            "source_file": f"data/{reading.LANGUAGES[row['language']]}.jsonl",
            "link": link,
            "question_number": int(number),
            "language": row["language"],
            "option_order": option_order(row, source),
            "letter_correct": row["letter_correct"],
        }
        | shown(row["response"], only_a_letter(row))
        | {
            "letter": letter,
            "correct": letter == row["letter_correct"],
            "read_through_a_letter_name": by_name(row) is not None,
        }
        | {key: row[key] for key in RUN_FIELDS}
    )


def squad_row(row: dict, arm: str) -> dict:
    """The published row of one SQuAD 2.0 answer."""
    response = row["response"] or ""
    return (
        {
            "id": row["id"],
            "arm": arm,
            "model": row["model"],
            "source": f"{reading.SQUAD}@{reading.SQUAD_REVISION}",
            "source_id": row["id"].removeprefix("squad."),
            "condition": row["condition"],
            "wording": abstains(response),
            "contains_a_gold_answer": holds_gold(row),
        }
        | shown(row["response"], response.strip() in TRAINED)
        | {key: row[key] for key in RUN_FIELDS}
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda row: (row["id"], row["arm"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    left_out = sum(row.get("response_withheld", False) for row in rows)
    print(f"{len(rows)} rows -> {path.name}; {left_out} answers left out as text")


def main(argv: Sequence[str] | None = None, download: Callable[[str], bytes] = pinned.download) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("check", choices=("belebele", "squad"), help="the dataset the answers are to")
    parser.add_argument("--sources", type=Path, help="folder of the two datasets' files; needed for belebele")
    parser.add_argument("--base", type=Path, help="answers of the base model, JSON lines")
    parser.add_argument("--fine-tuned", type=Path, help="answers of the fine-tuned model, JSON lines")
    parser.add_argument("--out", type=Path, required=True, help="folder for the published rows")
    args = parser.parse_args(argv)
    answers = {arm: path for arm, path in zip(ARMS, (args.base, args.fine_tuned), strict=True) if path}
    if not answers:
        parser.error("give the answers of a model: --base, --fine-tuned or both")
    if args.check == "squad":
        rows = [squad_row(row, arm) for arm, path in answers.items() for row in read_jsonl(path)]
    else:
        if args.sources is None:
            parser.error("belebele needs --sources, the folder of the datasets' files")
        reading.fetch_belebele(args.sources, download)
        folder = args.sources / reading.BELEBELE_FOLDER / "data"
        sources = {
            code: {(row["link"], row["question_number"]): row for row in read_jsonl(folder / f"{name}.jsonl")}
            for code, name in reading.LANGUAGES.items()
        }
        rows = [belebele_row(row, arm, sources) for arm, path in answers.items() for row in read_jsonl(path)]
    write_jsonl(args.out / f"{args.check}_answers.jsonl", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
