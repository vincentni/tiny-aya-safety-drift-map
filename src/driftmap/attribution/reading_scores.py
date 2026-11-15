"""Rule scores for the reading check on two other datasets, from the answers of the base and the fine-tuned model.

    python -I -m driftmap.attribution.reading_scores belebele --base BASE.jsonl --fine-tuned TUNED.jsonl --out OUT
    python -I -m driftmap.attribution.reading_scores squad --base BASE.jsonl --fine-tuned TUNED.jsonl --out OUT

``belebele`` writes belebele_scores.json: accuracy, answers without a letter and abstention wording, per model.
``squad`` writes squad_rows.jsonl, the answers of both models as rows to read, and squad_text_counts.json, a count
by wording alone. An answers file holds the rows of the prompt file with the ``response`` of a model.

A Belebele answer is read by the rule of the multiple-choice table, for the letters A to D: a letter alone, a
capital letter that is marked off or followed by its own option's text, a letter announced by "answer is", the
name of a letter in Devanagari or Arabic script, or the whole text of one option; otherwise it has no letter. A
sentence that opens with the article "A" names no option.
The SQuAD count is a word match only: the trained abstention sentences, and wider wording for "the passage does
not say".
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from driftmap.attribution.build import SENTENCES
from driftmap.attribution.reading import LETTERS
from driftmap.table import choice

# an answer that is only the name of a Latin letter in Devanagari or Arabic script: the names the multiple-choice
# table reads for its three letters, and the name of D
NAMES = choice.NAMES | {"\u0921\u0940": "D"}
# the reading of the multiple-choice table, for four letters
RULE = choice.LetterRule(LETTERS, NAMES)
TRAINED = tuple(SENTENCES.values())
# the trained wording anywhere, in any of the four languages' usual word for "the text"
TRAINED_STYLE = re.compile(
    r"from the text|in the text\b|not answerable|undetermined|can(?:not|'t) be determined|not enough info"
    r"|can(?:not|'t) answer that|из текста|в тексте|من النص|في النص|पाठ से|पाठ में",
    re.IGNORECASE,
)
WIDER = re.compile(
    r"does(?:n't| not) (?:say|mention|state|specify|provide|give|indicate|contain|identify|name|address|discuss)"
    r"|do(?:es)? not have (?:enough )?information|not (?:mentioned|stated|specified|provided|given|clear|known)"
    r"|no (?:information|mention|answer)|unknown|unclear|cannot be answered|can't be answered|is not possible to"
    r"|not possible to (?:determine|say|answer)|there is no ",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def letter_of(row: dict) -> str | None:
    """The letter an answer names among A to D, None when it names none."""
    return RULE.parse(row["response"], {entry["letter"]: entry["text"] for entry in row["options"]})


def by_name(row: dict) -> str | None:
    """The letter an answer names when it is nothing but one of ``NAMES``."""
    return RULE.by_name(row["response"])


def abstains(text: str) -> str:
    """How an answer abstains, by wording alone: a trained sentence, the trained style, wider wording, or not."""
    if text.strip() in TRAINED:
        return "trained sentence"
    if TRAINED_STYLE.search(text):
        return "trained style"
    return "wider wording" if WIDER.search(text) else "no abstention wording"


def holds_gold(row: dict) -> bool:
    """Whether a SQuAD answer contains one of the row's gold answers, case aside."""
    gold = [answer.casefold() for answer in row["gold_answers"]]
    return bool(gold) and any(answer in (row["response"] or "").casefold() for answer in gold)


def belebele(answers: dict[str, Path], out: Path) -> None:
    """Write belebele_scores.json under ``out`` from the answers file of each model."""
    scores = {}
    for model, path in answers.items():
        rows = read_jsonl(path)
        cuts = {"all": rows} | {
            code: [row for row in rows if row["language"] == code] for code in ("en", "hi", "ar", "ru")
        }
        scores[model] = {}
        for name, mine in cuts.items():
            picked = [letter_of(row) for row in mine]
            scores[model][name] = {
                "rows": len(mine),
                "correct": sum(letter == row["letter_correct"] for letter, row in zip(picked, mine, strict=True)),
                "accuracy": share(
                    sum(letter == row["letter_correct"] for letter, row in zip(picked, mine, strict=True)), len(mine)
                ),
                "no_letter": sum(letter is None for letter in picked),
                "read_through_a_letter_name": sum(by_name(row) is not None for row in mine),
                "correct_without_letter_names": sum(
                    letter == row["letter_correct"] and by_name(row) is None
                    for letter, row in zip(picked, mine, strict=True)
                ),
                "abstention_sentence_or_style": sum(
                    abstains(row["response"] or "") in ("trained sentence", "trained style") for row in mine
                ),
                "letters_answered": dict(sorted(Counter(letter or "none" for letter in picked).items())),
                "correct_by_letter_of_correct": {
                    letter: [
                        sum(
                            p == letter for p, row in zip(picked, mine, strict=True) if row["letter_correct"] == letter
                        ),
                        sum(row["letter_correct"] == letter for row in mine),
                    ]
                    for letter in "ABCD"
                },
            }
            found = scores[model][name]
            print(
                f"{model:10} {name:3} {found['correct']}/{found['rows']} = {found['accuracy']}; no letter "
                f"{found['no_letter']}; by letter name {found['read_through_a_letter_name']}; "
                f"abstention wording {found['abstention_sentence_or_style']}; letters "
                f"{found['letters_answered']}; correct by letter {found['correct_by_letter_of_correct']}"
            )
        odd = [row for row in rows if letter_of(row) is None or len((row["response"] or "").strip()) > 3]
        print(
            f"{model:10} answers that are not a bare letter: {len(odd)}",
            [(row["language"], (row["response"] or "")[:60]) for row in odd[:12]],
        )
    if len(scores) == 2:
        both = {name: {row["id"]: row for row in read_jsonl(path)} for name, path in answers.items()}
        shared = sorted(set(both["base"]) & set(both["fine_tuned"]))

        def right(row: dict) -> bool:
            return letter_of(row) == row["letter_correct"]

        scores["paired"] = {
            "rows": len(shared),
            "correct_to_wrong": sum(right(both["base"][k]) and not right(both["fine_tuned"][k]) for k in shared),
            "wrong_to_correct": sum(not right(both["base"][k]) and right(both["fine_tuned"][k]) for k in shared),
            "same_letter": sum(letter_of(both["base"][k]) == letter_of(both["fine_tuned"][k]) for k in shared),
        }
        print("paired base -> fine-tuned:", scores["paired"])
    (out / "belebele_scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def squad(answers: dict[str, Path], out: Path) -> None:
    """Write squad_rows.jsonl and squad_text_counts.json under ``out`` from the answers file of each model."""
    rows_out, counts = [], defaultdict(Counter)
    lengths = defaultdict(list)
    for model, path in answers.items():
        for row in read_jsonl(path):
            rows_out.append(
                {
                    "id": row["id"],
                    "condition": row["condition"],
                    "context": row["context"],
                    "question": row["question"],
                    "gold_answers": row["gold_answers"],
                    "response": row["response"],
                    "model": model,
                    "model_revision": row["model"],
                    "stop_reason": row["stop_reason"],
                    "generated_tokens": row["generated_tokens"],
                    "title": row["title"],
                }
            )
            counts[(model, row["condition"])][abstains(row["response"] or "")] += 1
            lengths[(model, row["condition"])].append(row["generated_tokens"])
            if holds_gold(row):
                counts[(model, row["condition"])]["contains a gold answer"] += 1
    rows_out.sort(key=lambda row: (row["id"], row["model"]))
    (out / "squad_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out), encoding="utf-8"
    )
    print(f"{len(rows_out)} rows -> squad_rows.jsonl")
    summary = {}
    for (model, condition), found in sorted(counts.items()):
        tokens = sorted(lengths[(model, condition)])
        summary[f"{model} {condition}"] = dict(found) | {"rows": len(tokens), "median_tokens": tokens[len(tokens) // 2]}
        print(f"{model:10} {condition:12} rows {len(tokens)} {dict(found)} median tokens {tokens[len(tokens) // 2]}")
    (out / "squad_text_counts.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("check", choices=("belebele", "squad"), help="the dataset the answers are to")
    parser.add_argument("--base", type=Path, help="answers of the base model, JSON lines")
    parser.add_argument("--fine-tuned", type=Path, help="answers of the fine-tuned model, JSON lines")
    parser.add_argument("--out", type=Path, required=True, help="folder for the scores")
    args = parser.parse_args(argv)
    answers = {name: path for name, path in (("base", args.base), ("fine_tuned", args.fine_tuned)) if path}
    if not answers:
        parser.error("give the answers of a model: --base, --fine-tuned or both")
    args.out.mkdir(parents=True, exist_ok=True)
    {"belebele": belebele, "squad": squad}[args.check](answers, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
