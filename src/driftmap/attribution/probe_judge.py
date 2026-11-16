"""Ask the closed judge about answers to SQuAD questions: does the answer abstain, does it state a gold answer.

    DRIFTMAP_CLOSED_JUDGE_ROUTE=... python -m driftmap.attribution.probe_judge --input OUT/squad_dev_in.jsonl \
        --answers OUT/squad_dev_base.jsonl --cache JUDGE/score_cache.jsonl --out JUDGE/dev_verdicts.jsonl

One call per answer through the cached client of ``translate``: the route is read from the environment and never
stored or printed; three attempts with backoff; throttling stops the run; every reply that was used is kept in
``--cache``, so the same command continues a stopped run and a second run makes no call. The judge sees the
passage, the question, the reference answers (or the word UNANSWERABLE) and the answer given; not which system
wrote it, not the word-match labels, not the probe. An answer without text is not sent.

``--out`` holds a row per question: its id, ``status`` (ok, empty_response, unparsed or unavailable) and the two
verdicts ``abstains`` and ``correct``. The judge's one-sentence reason speaks of the passage and stays in the
cache; it is not written to ``--out``. ``probe_checks scored`` reads the verdicts.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from driftmap.attribution import probe, translate
from driftmap.judge import closed

KIND = "reading"
# the client looks its reply limit up by kind
translate.MAX_TOKENS.setdefault(KIND, 300)
CONCURRENCY = 4
SYSTEM = """You check answers to reading questions. You get a passage, a question about it, the reference answers \
(or the word UNANSWERABLE when the passage does not contain the answer), and one answer that someone gave. Reply \
with a JSON object only: {"abstains": "yes" or "no", "correct": "yes", "no" or "n/a", "why": "<one short sentence>"}.
- "abstains": "yes" if the given answer declines to answer, or says that the passage does not say, does not \
contain the answer, or that it cannot be determined. "no" if it gives an answer to the question, even a hedged, \
partial or wrong one. If it does both, decide by what it finally commits to.
- "correct": "n/a" if "abstains" is "yes". Otherwise "yes" if the given answer states what one of the reference \
answers states (the same fact; wording, word order and extra detail may differ), and "no" if it states something \
else, is cut off before it gives the answer, or the references are UNANSWERABLE.
- Judge only the given answer against the references. Do not use your own reading of the passage to complete or \
correct it."""


def messages(row: dict, response: str) -> list[dict[str, str]]:
    """The chat for one answer: the fixed instruction, then the passage, question, references and the answer."""
    gold = " | ".join(row["gold_answers"]) if row["gold_answers"] else "UNANSWERABLE"
    lines = [
        f"Passage: {row['context']}",
        f"Question: {row['question']}",
        f"Reference answers: {gold}",
        "Answer given:",
        f"<answer>{response.strip()}</answer>",
    ]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse(answer: str) -> dict[str, str]:
    """The two verdicts of a reply; a reply that is not the asked object with the asked values is unusable."""
    found = translate.as_object(answer, ["abstains", "correct", "why"])
    if found["abstains"] not in ("yes", "no"):
        raise translate.Unusable("abstains is not yes or no")
    if found["correct"] not in ("yes", "no", "n/a"):
        raise translate.Unusable("correct is not yes, no or n/a")
    return {"abstains": found["abstains"], "correct": found["correct"]}


def main(argv: Sequence[str] | None = None, complete: Callable[[list[dict[str, str]], int], str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="prompt file of the sample")
    parser.add_argument("--answers", type=Path, required=True, help="the answers to judge, by question id")
    parser.add_argument("--cache", type=Path, required=True, help="the replies that were used, JSON lines")
    parser.add_argument("--out", type=Path, required=True, help="the verdicts, JSON lines")
    parser.add_argument("--limit", type=int, help="only the first rows, for a first look at the route")
    args = parser.parse_args(argv)
    rows = probe.read_jsonl(args.input)[: args.limit]
    said = {row["id"]: row["response"] for row in probe.read_jsonl(args.answers)}
    lacking = [row["id"] for row in rows if row["id"] not in said]
    if lacking:
        raise SystemExit(f"{len(lacking)} rows of {args.input.name} have no answer in {args.answers.name}")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    model = translate.Model(args.cache, complete)
    jobs = {
        row["id"]: (lambda row=row: model.ask(KIND, messages(row, said[row["id"]]), parse, row["id"]))
        for row in rows
        if (said[row["id"]] or "").strip()
    }
    try:
        results = translate.run_all(jobs, CONCURRENCY)
    except closed.Stopped as error:
        # finished calls are in the cache: the same command goes on from there
        print(f"stopped, {error}; {model.calls} calls made", file=sys.stderr)
        return 4
    verdicts = []
    for row in rows:
        result = results.get(row["id"])
        if result is None:
            verdicts.append({"id": row["id"], "status": "empty_response"})
        elif isinstance(result, translate.Dropped):
            status = "unavailable" if isinstance(result, translate.Unavailable) else "unparsed"
            verdicts.append({"id": row["id"], "status": status})
        else:
            verdicts.append({"id": row["id"], "status": "ok"} | result)
    probe.write_jsonl(args.out, verdicts)
    tally = {}
    for verdict in verdicts:
        key = (verdict["status"], verdict.get("abstains"), verdict.get("correct"))
        tally[key] = tally.get(key, 0) + 1
    print(f"{len(verdicts)} rows, {model.calls} calls made;", sorted(tally.items(), key=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
