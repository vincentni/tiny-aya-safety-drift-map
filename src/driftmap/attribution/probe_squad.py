"""SQuAD 2.0 for the probe study: the two samples, the base model's own error on each question, and the counts
of a system that abstains where a gate fires.

    python -I -m driftmap.attribution.probe_squad --sources SOURCES --out OUT

``--sources`` is the folder of the reading check's source files; the validation file and the train file of
SQuAD 2.0 are fetched there at the revision ``reading`` pins and checked by sha256 before they are read. The draw
is the reading check's rule with seeds and sizes of its own: passages under 250 words; the articles in a seeded
order, taken in turn; every drawn question on a passage no other drawn question uses; unanswerable first, then
answerable. Written under ``--out``: ``squad_dev_in.jsonl`` (500 + 500 questions of the validation file),
``squad_train_in.jsonl`` (1,000 + 1,000 of the train file), ``squad_dev2_in.jsonl`` (a second 500 + 500 of the
validation file: none of the first sample's questions, and one of its passages only where an article has no other
left) and ``squad_build.json`` (counts and hashes). The prompt files hold SQuAD text and stay where they are
written; no other output of the study holds any.

The base model's answer is classed by the reading check's rule: it abstains when its wording class is an
abstention, and an answerable question is answered correctly when the answer contains a gold answer, case aside.
A hallucination is an unanswerable question answered. An error is a hallucination, or an answerable question
whose answer holds no gold answer. Both rules are word matches. A gated answer is an abstention: right on an
unanswerable question, wrong on an answerable one.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from driftmap.attribution import pinned, probe, probe_gates, reading, reading_scores
from driftmap.table import stats

TRAIN_FILE = "squad_v2/train-00000-of-00001.parquet"
FILES = {
    reading.SQUAD_FILE: reading.SQUAD_FILES[reading.SQUAD_FILE],
    TRAIN_FILE: "f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291",
}
# sample to its source file and the questions drawn per condition
SAMPLES = {"dev": (reading.SQUAD_FILE, 500), "train": (TRAIN_FILE, 1000), "dev2": (reading.SQUAD_FILE, 500)}
# a sample drawn after another one of the same file, apart from its questions
AFTER = {"dev2": "dev"}
CONDITIONS = ("unanswerable", "answerable")
NO_ABSTENTION = "no abstention wording"
WORDINGS = ("trained sentence", "trained style", "wider wording", NO_ABSTENTION)
BUILD = "squad_build.json"


def seed_of(name: str) -> str:
    return f"{probe.SEED}-squad-probe-{name}"


def fetch(sources: Path, get: Callable[[str], bytes] = pinned.download) -> dict[str, str]:
    """The sha256 of the two SQuAD files under ``sources``; a missing one is downloaded, each is checked."""
    return reading._fetch(sources, reading.SQUAD, reading.SQUAD_REVISION, reading.SQUAD_FOLDER, FILES, get)


def draw(source: Path, per_condition: int, seed: str, earlier: Sequence[Mapping[str, Any]] = ()) -> list[dict]:
    """``per_condition`` unanswerable and as many answerable questions of the file, as prompt rows.

    ``earlier`` holds the rows of a sample drawn from the same file before: none of its questions is drawn, and
    one of its passages is taken only when the article in turn has no other unused passage left.
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    rows = sorted(pq.read_table(source).to_pylist(), key=lambda row: row["id"])
    asked, seen = ({row[key] for row in earlier} for key in ("id", "context"))
    short = [
        row for row in rows if len(row["context"].split()) < reading.MAX_WORDS and f"squad.{row['id']}" not in asked
    ]
    titles = sorted({row["title"] for row in short})
    rng.shuffle(titles)
    used, drawn = set(), {}
    for condition in CONDITIONS:
        pool = defaultdict(list)
        for row in short:
            if (len(row["answers"]["text"]) == 0) == (condition == CONDITIONS[0]):
                pool[row["title"]].append(row)
        picked = []
        while len(picked) < per_condition:
            before = len(picked)
            for title in titles:
                left = [row for row in pool[title] if row["context"] not in used]
                if left and len(picked) < per_condition:
                    row = rng.choice([row for row in left if row["context"] not in seen] or left)
                    used.add(row["context"])
                    picked.append(row)
            if len(picked) == before:
                raise SystemExit(f"only {len(picked)} {condition} questions on unused passages")
        drawn[condition] = picked
    rows_out = []
    for pair in zip(*(drawn[condition] for condition in CONDITIONS), strict=True):
        for condition, row in zip(CONDITIONS, pair, strict=True):
            rows_out.append(
                {
                    "id": f"squad.{row['id']}",
                    "condition": condition,
                    "context": row["context"],
                    "question": row["question"],
                    "gold_answers": sorted(set(row["answers"]["text"])),
                    "title": row["title"],
                    "prompt": "\n".join([row["context"], row["question"], reading.FREE]),
                    "set": "benign",
                    "language": "en",
                }
            )
    return rows_out


def answers_of(path: Path) -> dict[str, dict]:
    """A run's answers by the study's question id. A row of this study's own runs has it as ``id``; a row of a
    generate job over the prompt group names the question as ``squad.<split>.<question id>``."""
    found = {}
    for row in probe.read_jsonl(path):
        found[row.get("id") or f"squad.{row['prompt_id'].split('.', 2)[2]}"] = row
    return found


def article_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """A number per article of ``rows``, counted over their sorted titles; it names an article without its title."""
    return {title: number for number, title in enumerate(sorted({row["title"] for row in rows}))}


def rows_of(asked: Path, answers: Path) -> list[dict]:
    """The rows of a sample with the base model's answer classed: ``abstained``, ``correct``, ``base_wrong`` and
    ``hallucination``; stops when the answers are not those of the sample's rows."""
    rows = probe.read_jsonl(asked)
    said = answers_of(answers)
    if set(said) != {row["id"] for row in rows}:
        raise SystemExit(f"{len(said)} answers for {len(rows)} rows of {asked.name}")
    for row in rows:
        row["response"] = said[row["id"]]["response"]
        row["stop_reason"] = said[row["id"]]["stop_reason"]
        row["generated_tokens"] = said[row["id"]].get("generated_tokens")
        row["error"] = said[row["id"]].get("error")
        row["wording"] = reading_scores.abstains(row["response"] or "")
        row["unanswerable"] = row["condition"] == CONDITIONS[0]
        row["abstained"] = row["wording"] != NO_ABSTENTION
        row["holds_gold"] = reading_scores.holds_gold(row)
        row["correct"] = row["abstained"] if row["unanswerable"] else row["holds_gold"]
        row["base_wrong"] = not row["correct"]
        row["hallucination"] = row["unanswerable"] and not row["abstained"]
    return rows


def states_of(folder: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], list[int]]:
    states, layers = probe.load_states(folder)
    if set(states) != {row["id"] for row in rows}:
        raise SystemExit(f"{len(states)} hidden states for {len(rows)} rows under {folder.name}")
    return states, layers


def base_counts(rows: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "rows": len(rows),
        "unanswerable": sum(row["unanswerable"] for row in rows),
        "base_wrong": sum(row["base_wrong"] for row in rows),
        "hallucinations": sum(row["hallucination"] for row in rows),
        "answerable_without_gold": sum(row["base_wrong"] and not row["unanswerable"] for row in rows),
        "answerable_abstained": sum(row["abstained"] and not row["unanswerable"] for row in rows),
        "wording": {kind: sum(row["wording"] == kind for row in rows) for kind in WORDINGS},
        "cut_at_the_token_limit": sum(row["stop_reason"] == "length" for row in rows),
    }


def measured(rows: Sequence[Mapping[str, Any]], fired: np.ndarray) -> dict:
    """The counts of the system that abstains where ``fired`` and gives the base model's answer elsewhere."""
    share = probe_gates.share
    abstained = np.array([row["abstained"] for row in rows]) | fired
    correct = np.where(fired, [row["unanswerable"] for row in rows], [row["correct"] for row in rows])
    unanswerable = np.array([row["unanswerable"] for row in rows])
    answered, wrong = ~abstained, ~correct
    return {
        "rows": len(rows),
        "wrong_answers": int(wrong.sum()),
        "wrong_answer_rate": share(int(wrong.sum()), len(rows)),
        "answered": int(answered.sum()),
        "coverage": share(int(answered.sum()), len(rows)),
        "wrong_among_answered": int((wrong & answered).sum()),
        "risk": share(int((wrong & answered).sum()), int(answered.sum())),
        "hallucinations": int((unanswerable & answered).sum()),
        "unanswerable_abstained": int((unanswerable & abstained).sum()),
        "answerable_rows": int((~unanswerable).sum()),
        "answerable_answered": int((~unanswerable & answered).sum()),
        "answerable_answered_rate": share(int((~unanswerable & answered).sum()), int((~unanswerable).sum())),
        "answerable_correct": int((~unanswerable & correct).sum()),
    }


def change(rows: Sequence[Mapping[str, Any]], fired: np.ndarray) -> dict:
    """Base to gated change in the wrong-answer rate: interval over resampled articles, sign test over articles."""
    before, after, size = defaultdict(int), defaultdict(int), defaultdict(int)
    for row, gate in zip(rows, fired, strict=True):
        before[row["title"]] += row["base_wrong"]
        after[row["title"]] += (not row["unanswerable"]) if gate else row["base_wrong"]
        size[row["title"]] += 1
    titles = sorted(size)
    first, second, n = (np.array([held[title] for title in titles], dtype=float) for held in (before, after, size))
    difference = second - first
    both = list(zip(rows, fired, strict=True))
    return {
        "articles": len(titles),
        "change": round(float(difference.sum() / n.sum()), 4),
        "ci95": probe_gates.interval(difference, n),
        "articles_better": int((difference < 0).sum()),
        "articles_worse": int((difference > 0).sum()),
        "sign_test_p": stats.sign_test(difference),
        "questions_wrong_to_right": int(sum(row["base_wrong"] and gate and row["unanswerable"] for row, gate in both)),
        "questions_right_to_wrong": int(
            sum(not row["base_wrong"] and gate and not row["unanswerable"] for row, gate in both)
        ),
    }


def read_gate(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, threshold: float) -> dict:
    """A gate's reading of a sample: AUCs, firing, the gated counts, the change, and risk against coverage at
    every threshold of the grid."""
    unanswerable = np.array([row["unanswerable"] for row in rows])
    wrong = np.array([row["base_wrong"] for row in rows])
    fired = scores >= threshold

    def counted(mask: np.ndarray) -> list[int]:
        return [int(fired[mask].sum()), int(mask.sum())]

    kept = ("wrong_answer_rate", "coverage", "risk", "answerable_answered_rate", "hallucinations")
    return {
        "threshold": threshold,
        "auc_unanswerable": probe_gates.auc(unanswerable, scores),
        "auc_base_error": probe_gates.auc(wrong, scores),
        "fired_on_unanswerable": counted(unanswerable),
        "fired_on_answerable": counted(~unanswerable),
        "fired_on_base_wrong": counted(wrong),
        "fired_on_base_right": counted(~wrong),
        "gated": measured(rows, fired),
        "base_to_gated_wrong_answers": change(rows, fired),
        "curve": [
            {"threshold": value} | {key: found[key] for key in kept}
            for value in probe.THRESHOLDS
            for found in [measured(rows, scores >= value)]
        ],
    }


def read_verdicts(path: Path) -> dict[str, dict]:
    """A judge's verdicts by question id: ``status``, ``abstains`` and ``correct``; its reasons are not read."""
    kept = ("status", "abstains", "correct")
    return {row["id"]: {key: row.get(key) for key in kept} for row in probe.read_jsonl(path)}


def judged(rows: Sequence[Mapping[str, Any]], verdicts: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """The rows the judge gave a verdict on (or that had no answer text), labelled by the judge instead of the
    word match: abstained when its ``abstains`` is yes; an unanswerable question right when abstained; an
    answerable one right when not abstained and its ``correct`` is yes. The word-match labels stay under
    ``rule_abstained``, ``rule_holds_gold`` and ``rule_wrong``."""
    kept = []
    for row in rows:
        verdict = verdicts.get(row["id"])
        if verdict is None or verdict["status"] not in ("ok", "empty_response"):
            continue
        abstained = verdict.get("abstains") == "yes"
        says_gold = not abstained and verdict.get("correct") == "yes"
        correct = abstained if row["unanswerable"] else says_gold
        kept.append(
            dict(
                row,
                rule_abstained=row["abstained"],
                rule_holds_gold=row["holds_gold"],
                rule_wrong=row["base_wrong"],
                judge_says_gold=says_gold,
                abstained=abstained,
                correct=correct,
                base_wrong=not correct,
                hallucination=row["unanswerable"] and not abstained,
            )
        )
    return kept


def two_by_two(first: Sequence[bool], second: Sequence[bool]) -> dict:
    """Counts of two yes / no readings of the same rows and Cohen's kappa between them."""
    from sklearn.metrics import cohen_kappa_score

    table = {
        f"{'yes' if a else 'no'}_{'yes' if b else 'no'}": sum(
            x == a and y == b for x, y in zip(first, second, strict=True)
        )
        for a in (True, False)
        for b in (True, False)
    }
    varies = len(set(first)) > 1 or len(set(second)) > 1
    return {
        "rows": len(first),
        "rule_judge": table,
        "agree": table["yes_yes"] + table["no_no"],
        "kappa": round(float(cohen_kappa_score(first, second)), 4) if varies else None,
    }


def agreement(rows: Sequence[Mapping[str, Any]]) -> dict:
    """The word-match rules against the judge on judged rows; in a cell's name the rule comes first."""
    answerable = [row for row in rows if not row["unanswerable"]]
    return {
        "abstains": two_by_two([row["rule_abstained"] for row in rows], [row["abstained"] for row in rows]),
        "gold_answer_on_answerable": two_by_two(
            [row["rule_holds_gold"] for row in answerable], [row["judge_says_gold"] for row in answerable]
        ),
        "answer_is_wrong": two_by_two([row["rule_wrong"] for row in rows], [row["base_wrong"] for row in rows]),
    }


def against(rows: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]) -> dict:
    """Wrong-answer rate of ``other`` minus that of ``rows``, the same questions under two systems: interval over
    resampled articles, sign test over articles."""
    wrong = {row["id"]: row["base_wrong"] for row in other}
    before, after, size = defaultdict(int), defaultdict(int), defaultdict(int)
    for row in rows:
        if row["id"] in wrong:
            before[row["title"]] += row["base_wrong"]
            after[row["title"]] += wrong[row["id"]]
            size[row["title"]] += 1
    titles = sorted(size)
    first, second, n = (np.array([held[title] for title in titles], dtype=float) for held in (before, after, size))
    difference = second - first
    return {
        "questions": int(n.sum()),
        "articles": len(titles),
        "change": round(float(difference.sum() / n.sum()), 4),
        "ci95": probe_gates.interval(difference, n),
        "articles_better": int((difference < 0).sum()),
        "articles_worse": int((difference > 0).sum()),
        "sign_test_p": stats.sign_test(difference),
    }


def main(argv: Sequence[str] | None = None, download: Callable[[str], bytes] = pinned.download) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", type=Path, required=True, help="folder of the source files; fetched and checked")
    parser.add_argument("--out", type=Path, required=True, help="folder for the two prompt files and their counts")
    parser.add_argument("--reading", type=Path, help="squad_in.jsonl of the reading check, to count shared questions")
    args = parser.parse_args(argv)
    hashes = fetch(args.sources, download)
    summary = {"revision": reading.SQUAD_REVISION, "licence": reading.LICENCE, "max_passage_words": reading.MAX_WORDS}
    drawn = {}
    for name, (file, per_condition) in SAMPLES.items():
        first = drawn.get(AFTER.get(name), [])
        rows = drawn[name] = draw(args.sources / reading.SQUAD_FOLDER / file, per_condition, seed_of(name), first)
        probe.write_jsonl(args.out / f"squad_{name}_in.jsonl", rows)
        summary[name] = {
            "file_sha256": hashes[file],
            "rows": len(rows),
            "by_condition": dict(Counter(row["condition"] for row in rows)),
            "articles": len({row["title"] for row in rows}),
            "distinct_passages": len({row["context"] for row in rows}),
            "distinct_ids": len({row["id"] for row in rows}),
        }
        if first:
            passages = {row["context"]: row["condition"] for row in first}
            shared = [row for row in rows if row["context"] in passages]
            summary[name] |= {
                "after": AFTER[name],
                "questions_also_in_the_first_sample": len({row["id"] for row in first} & {row["id"] for row in rows}),
                "passages_also_in_the_first_sample": len(shared),
                "of_those_with_the_same_condition_as_the_first_sample_question": sum(
                    passages[row["context"]] == row["condition"] for row in shared
                ),
                "passages_not_in_the_first_sample": len(rows) - len(shared),
            }
    for key, name in (("context", "passages"), ("title", "articles")):
        shared = {row[key] for row in drawn["dev"]} & {row[key] for row in drawn["train"]}
        summary[f"{name}_in_both_samples"] = len(shared)
        for later in AFTER:
            if later in drawn:
                both = {row[key] for row in drawn[later]} & {row[key] for row in drawn["train"]}
                summary[f"{name}_in_{later}_and_train"] = len(both)
    if args.reading:
        old = {row["id"] for row in probe.read_jsonl(args.reading)}
        summary["dev_rows_also_in_the_reading_check"] = len(old & {row["id"] for row in drawn["dev"]})
    files = {
        f"squad_{name}_in.jsonl": hashlib.sha256((args.out / f"squad_{name}_in.jsonl").read_bytes()).hexdigest()
        for name in SAMPLES
    }
    (args.out / BUILD).write_text(json.dumps(summary | {"files_sha256": files}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
