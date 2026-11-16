"""Checks of the SQuAD probe after it was frozen: served in a running model, read by a judge, answers by identity.

    python -I -m driftmap.attribution.probe_checks served-sample --dev DEV_IN --passed CURATED/squad_passed.json \
        --out SAMPLE
    python -I -m driftmap.attribution.probe_checks served --sample SAMPLE --served SERVED --served-states ACTS/served \
        --dev DEV_IN --dev-answers DEV_ANSWERS --dev-states ACTS/squad_dev --passed CURATED/squad_passed.json \
        --probe OUT/probe_squad --abstention "SENTENCE" --out RESULTS
    python -I -m driftmap.attribution.probe_checks scored --part judge --dev DEV_IN --dev-answers DEV_ANSWERS \
        --dev-states ACTS/squad_dev --verdicts JUDGE/dev_verdicts.jsonl --passed CURATED/squad_passed.json \
        --probe OUT/probe_squad --out RESULTS
    python -I -m driftmap.attribution.probe_checks fine-tune --dev DEV_IN --dev-answers DEV_ANSWERS \
        --dev-states ACTS/squad_dev --tuned-answers TUNED_ANSWERS --verdicts JUDGE/dev_verdicts.jsonl \
        --tuned-verdicts JUDGE/ft_verdicts.jsonl --passed CURATED/squad_passed.json --probe OUT/probe_squad \
        --out RESULTS
    python -I -m driftmap.attribution.probe_checks instructed-sample --dev DEV_IN \
        --passed CURATED/squad_passed.json --out INSTRUCTED_IN
    python -I -m driftmap.attribution.probe_checks fine-tune --part prompt --dev DEV_IN --dev-answers DEV_ANSWERS \
        --dev-states ACTS/squad_dev --tuned-answers INSTRUCTED_ANSWERS --passed CURATED/squad_passed.json \
        --probe OUT/probe_squad --out RESULTS
    python -I -m driftmap.attribution.probe_checks confidence --dev DEV2_IN --dev-answers DEV2_ANSWERS \
        --dev-states ACTS/squad_dev2 --confidence CONFIDENCE.jsonl --verdicts JUDGE/dev2_verdicts.jsonl \
        --passed CURATED/squad_passed.json --probe OUT/probe_squad --out RESULTS
    python -I -m driftmap.attribution.probe_checks identity --stored DEV_ANSWERS --kit KIT_ANSWERS --out RESULTS
    python -I -m driftmap.attribution.probe_checks job-identity --name dev --asked DEV_IN \
        --stored DEV_ANSWERS --job RESPONSES/dev.jsonl --passed CURATED/squad_passed.json --out RESULTS

``served-sample`` takes every tenth unanswerable and every tenth answerable question of the dev sample, in its
order; ``served`` compares the answers ``probe_generate --gate --abstention`` gave them with the gate computed
afterwards from the stored answers and hidden states, and scores the served answers by the reading check's rule.
``scored`` applies the frozen probe to the stored hidden states of one sample, or of several given one after the
other, and gives the table of ``probe_study squad`` for it; with the verdicts ``probe_judge`` wrote for the
answers, it gives the table again under the judge's labels and how far the word-match rules agree with the judge.
``fine-tune`` sets a fine-tune's answers to the same questions beside the base model and the base model gated
by the probe, by the word-match rules and, with the verdicts on both models' answers, by the judge's labels on
the rows judged in both.
``instructed-sample`` writes a sample's prompts with one instruction line added before the closing line, to
abstain when the text does not say; ``fine-tune --part prompt`` sets the base model's answers to them beside
the base model and the gated base model as it does a fine-tune's. ``confidence`` takes the base model's own
confidence in its stored answers, one row per question with the probability of the answer's first token and
the mean log-probability of its tokens, as a detector of its errors beside the probe.
``identity`` compares two answer files of the same prompts row for row: text, token count and stop reason.
``job-identity`` does so for the answers a generate job gave to the files of the prompt group ``squad``, a
split after the other, and counts the rows whose word-match labels differ between the two runs. An answers
file may be one of this study's runs or a file of the job's responses; ``fine-tune --part`` names the part
when the fine-tune is set beside the base model on another sample than the first.

Nothing is fitted and no threshold is moved. Every step reads only the rows that passed the CuratorKIT gate and
writes ``probe-squad-<part>.json``, the same as Markdown and, but for ``identity``, rows of ids, classes, verdicts
and scores, each row with its ``article_index``, a number counted over the sorted article titles of the rows
scored. No passage, question, title or answer text and none of the judge's reasons is in any of them.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from driftmap.attribution import (
    build,
    probe,
    probe_curator,
    probe_gates,
    probe_squad,
    probe_study,
    reading_scores,
)
from driftmap.table import probe_checks as table

# the positions of a condition's questions the served gate is run on
EVERY = 10
PARTS = ("judge", "replication", "pooled")
# the name of the fine-tune's counts in a record, and of the base model's when its prompt carries the instruction
FINE_TUNED, INSTRUCTED = "fine_tuned", "instructed"
AGREED = {FINE_TUNED: "the_fine_tune"}
# the parts of another arm on a dev sample: the attribution fine-tune on the first sample and on the second, the
# fine-tune trained on the train sample on each, and the base model asked with one instruction line added
TUNED_PARTS = {
    "fine-tune": FINE_TUNED,
    "fine-tune-replication": FINE_TUNED,
    "trained": FINE_TUNED,
    "trained-replication": FINE_TUNED,
    "prompt": INSTRUCTED,
}
# the shares of a sample a detector abstains on, most suspicious first
STEPS = tuple(round(0.1 * step, 1) for step in range(1, 10))
DETECTORS = ("first_token_probability", "mean_log_probability")
COMPARED = ("response", "generated_tokens", "stop_reason")


def served_sample(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """The prompts the served gate runs on: id and prompt of every tenth question of each condition."""
    picked = []
    for condition in probe_squad.CONDITIONS:
        picked += [row for row in rows if row["condition"] == condition][::EVERY]
    return [{"id": row["id"], "prompt": row["prompt"]} for row in picked]


def served(
    sample: Sequence[Mapping[str, Any]],
    live_rows: Mapping[str, Mapping[str, Any]],
    frozen: probe.Probe,
    stored_states: Mapping[str, np.ndarray],
    run_states: Mapping[str, np.ndarray],
    sentence: str,
    articles: Mapping[str, int],
) -> tuple[dict, list[dict]]:
    """The served gate against the gate computed afterwards on ``sample``, rows as ``rows_of`` leaves them, and
    a row per prompt without text; the states are those of the probe's layer by question id, ``articles`` the
    number of each article of the whole sample."""
    offline = frozen.probability(np.stack([stored_states[row["id"]] for row in sample]))
    fired_offline = offline >= frozen.threshold
    fired_served = np.array([live_rows[row["id"]]["fired"] for row in sample])
    counts = {
        "prompts": len(sample),
        "served_equals_offline_gate": 0,
        "firing_decision_equal": 0,
        "fired": int(fired_served.sum()),
        "fired_rows_answering_the_abstention_sentence": 0,
        "unfired_rows": int((~fired_served).sum()),
        "unfired_rows_with_the_stored_answer_text": 0,
    }
    gaps = {"probability": 0.0, "hook_to_run_hidden_state": 0.0, "run_state_to_stored_activation": 0.0}
    public = []
    for row, probability, fired in zip(sample, offline, fired_offline, strict=True):
        live = live_rows[row["id"]]
        same_decision = bool(live["fired"]) == bool(fired)
        text_right = live["response"] == (sentence if live["fired"] else row["response"])
        counts["firing_decision_equal"] += same_decision
        counts["served_equals_offline_gate"] += same_decision and text_right
        counts["fired_rows_answering_the_abstention_sentence"] += live["fired"] and live["response"] == sentence
        counts["unfired_rows_with_the_stored_answer_text"] += not live["fired"] and live["response"] == row["response"]
        gaps["probability"] = max(gaps["probability"], abs(live["probe_probability"] - float(probability)))
        gaps["hook_to_run_hidden_state"] = max(gaps["hook_to_run_hidden_state"], live["hook_minus_hidden_state"])
        moved = float(np.abs(run_states[row["id"]] - stored_states[row["id"]]).max())
        gaps["run_state_to_stored_activation"] = max(gaps["run_state_to_stored_activation"], moved)
        public.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "article_index": articles[row["title"]],
                "probability_served": round(live["probe_probability"], 6),
                "probability_offline": round(float(probability), 6),
                "fired_served": bool(live["fired"]),
                "fired_offline": bool(fired),
                "served_answer_wording": reading_scores.abstains(live["response"] or ""),
                "base_answer_wording": row["wording"],
                "base_wrong": row["base_wrong"],
            }
        )
    # the served answers scored by the rule itself, beside the offline gate and the base model on the same rows
    relabelled = []
    for row in sample:
        live = dict(row, response=live_rows[row["id"]]["response"])
        live["abstained"] = reading_scores.abstains(live["response"] or "") != probe_squad.NO_ABSTENTION
        live["correct"] = live["abstained"] if live["unanswerable"] else reading_scores.holds_gold(live)
        relabelled.append(live)
    none = np.zeros(len(sample), dtype=bool)
    record = {
        "probe": {"layer": frozen.layer, "threshold": frozen.threshold},
        "abstention_sentence": sentence,
        "abstention_sentence_wording_class": reading_scores.abstains(sentence),
        "counts": counts,
        "largest_absolute_differences": gaps,
        "on_the_sample": {
            "base": probe_squad.measured(sample, none),
            "offline_gate": probe_squad.measured(sample, fired_offline),
            "served_scored_by_the_rule": probe_squad.measured(relabelled, none),
        },
    }
    return record, public


def reading(rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float], threshold: float) -> dict:
    """The sample's counts, the base model and the frozen probe's reading of ``rows``."""
    read = np.array([scores[row["id"]] for row in rows])
    return {
        "sample": probe_squad.base_counts(rows) | {"articles": len({row["title"] for row in rows})},
        "base": probe_squad.measured(rows, np.zeros(len(rows), dtype=bool)),
        "squad_probe": probe_squad.read_gate(rows, read, threshold),
    }


def scored(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    frozen: probe.Probe,
    verdicts: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict, list[dict]]:
    """The record of the frozen probe on ``rows`` by the word-match rules and, with ``verdicts``, by the judge's
    labels on the rows it judged; and a row per question without text."""
    found = {"probe": {"layer": frozen.layer, "C": frozen.c, "threshold": frozen.threshold}}
    found["word_match"] = reading(rows, scores, frozen.threshold)
    by_judge = {}
    if verdicts is not None:
        kept = probe_squad.judged(rows, verdicts)
        by_judge = {row["id"]: row for row in kept}
        found["judge"] = {
            "rows_judged": len(kept),
            "rows_unjudged": len(rows) - len(kept),
            "agreement_of_the_rules_with_the_judge": probe_squad.agreement(kept),
        } | reading(kept, scores, frozen.threshold)
        same = [row for row in rows if row["id"] in by_judge]
        found["word_match_on_the_judged_rows"] = reading(same, scores, frozen.threshold)
    public, articles = [], probe_squad.article_index(rows)
    for row in rows:
        seen = by_judge.get(row["id"])
        public.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "article_index": articles[row["title"]],
                "rule_wording": row["wording"],
                "rule_holds_gold": row["holds_gold"],
                "rule_wrong": row["base_wrong"],
                "judge_abstains": seen["abstained"] if seen else None,
                "judge_says_gold": seen["judge_says_gold"] if seen else None,
                "judge_wrong": seen["base_wrong"] if seen else None,
                "squad_probe_score": round(scores[row["id"]], 6),
                "squad_probe_fired": bool(scores[row["id"]] >= frozen.threshold),
            }
        )
    return found, public


def arms(
    base: Sequence[Mapping[str, Any]], tuned: Sequence[Mapping[str, Any]], gate: np.ndarray, label: str = FINE_TUNED
) -> dict:
    """The base model, the base model gated where ``gate`` and another arm on the same questions, under the
    name ``label``, and that arm's change in wrong answers against each of the two."""
    gated = [
        dict(row, base_wrong=(not row["unanswerable"]) if hit else row["base_wrong"])
        for row, hit in zip(base, gate, strict=True)
    ]
    return {
        "base": probe_squad.measured(base, np.zeros(len(base), dtype=bool)),
        "base_plus_squad_probe": probe_squad.measured(base, gate),
        label: probe_squad.measured(tuned, np.zeros(len(tuned), dtype=bool)),
        f"{label}_answers": probe_squad.base_counts(tuned),
        f"base_to_{label}_wrong_answers": probe_squad.against(base, tuned),
        f"base_plus_squad_probe_to_{label}_wrong_answers": probe_squad.against(gated, tuned),
    }


def fine_tune(
    base: Sequence[Mapping[str, Any]],
    tuned: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    frozen: probe.Probe,
    seen_base: Mapping[str, Mapping[str, Any]] | None,
    seen_tuned: Mapping[str, Mapping[str, Any]] | None,
    label: str = FINE_TUNED,
) -> tuple[dict, list[dict]]:
    """The record of another arm beside the base model and the gated base model, by the word-match rules and,
    with both arms' verdicts, by the judge's labels on the rows judged in both; and a row per answer of the
    arm without text. ``label`` names the arm in the record: the fine-tune, or the instructed base model."""
    found = {"probe": {"layer": frozen.layer, "C": frozen.c, "threshold": frozen.threshold}}
    fired = np.array([scores[row["id"]] >= frozen.threshold for row in base])
    found["word_match"] = arms(base, tuned, fired, label)
    if seen_base is not None and seen_tuned is not None:
        kept_base, kept_tuned = probe_squad.judged(base, seen_base), probe_squad.judged(tuned, seen_tuned)
        both = {row["id"] for row in kept_base} & {row["id"] for row in kept_tuned}
        kept_base = [row for row in kept_base if row["id"] in both]
        kept_tuned = [row for row in kept_tuned if row["id"] in both]
        gate = np.array([scores[row["id"]] >= frozen.threshold for row in kept_base])
        found["judge"] = {
            "rows_judged_in_both_arms": len(both),
            f"agreement_of_the_rules_with_the_judge_on_{AGREED.get(label, label)}": probe_squad.agreement(kept_tuned),
        } | arms(kept_base, kept_tuned, gate, label)
    by_judge = {row["id"]: row for row in probe_squad.judged(tuned, seen_tuned)} if seen_tuned is not None else {}
    public, articles = [], probe_squad.article_index(tuned)
    for row in tuned:
        seen = by_judge.get(row["id"])
        public.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "article_index": articles[row["title"]],
                "rule_wording": row["wording"],
                "rule_holds_gold": row["holds_gold"],
                "rule_wrong": row["base_wrong"],
                "judge_abstains": seen["abstained"] if seen else None,
                "judge_says_gold": seen["judge_says_gold"] if seen else None,
                "judge_wrong": seen["base_wrong"] if seen else None,
            }
        )
    return found, public


def instructed_sample(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """The rows of a sample asked with one instruction line added before the closing line."""
    line = build.ABSTAIN_INSTRUCTION
    return [
        dict(row, prompt="\n".join([row["context"], row["question"], line, build.FREE_INSTRUCTION]), instruction=line)
        for row in rows
    ]


def gate_of(count: int, order: np.ndarray, k: int) -> np.ndarray:
    """A gate over ``count`` rows that fires on the first ``k`` of ``order``."""
    fired = np.zeros(count, dtype=bool)
    fired[order[:k]] = True
    return fired


def at_matched_coverage(rows: Sequence[Mapping[str, Any]], order: np.ndarray, answered: int) -> tuple[int, dict]:
    """The smallest k at which abstaining on the first k rows of ``order`` leaves no more than ``answered``
    questions answered, and the counts there."""
    for k in range(len(rows) + 1):
        found = probe_squad.measured(rows, gate_of(len(rows), order, k))
        if found["answered"] <= answered:
            return k, found
    raise SystemExit("no number of abstentions reaches the probe gate's coverage")


def detectors_read(rows: Sequence[Mapping[str, Any]], detectors: Mapping[str, np.ndarray], fired: np.ndarray) -> dict:
    """The probe's gate and every detector under one labelling of ``rows``: AUCs, the probe's AUC minus the
    detector's, and the counts when the detector abstains on its most suspicious answers."""
    unanswerable = np.array([row["unanswerable"] for row in rows])
    wrong = np.array([row["base_wrong"] for row in rows])
    clustered = [dict(row, cluster=row["title"], prompt=row["id"], ambiguous=row["unanswerable"]) for row in rows]
    by_id = {
        name: dict(zip([row["id"] for row in rows], map(float, score), strict=True))
        for name, score in detectors.items()
    }
    probe_gate = probe_squad.measured(rows, fired)
    kept = ("wrong_answers", "hallucinations", "answerable_answered", "answerable_correct", "coverage", "risk")
    found = {
        "base": probe_squad.measured(rows, np.zeros(len(rows), dtype=bool)),
        "squad_probe_gate": probe_gate,
        "detectors": {},
    }
    for name, score in detectors.items():
        # the most suspicious first; ties by file order
        order = np.argsort(-score, kind="stable")
        entry = {
            "auc_base_error": probe_gates.auc(wrong, score),
            "auc_unanswerable": probe_gates.auc(unanswerable, score),
            "abstaining_on_the_most_suspicious_share": [
                {"share": share}
                | {
                    key: value
                    for key, value in probe_squad.measured(
                        rows, gate_of(len(rows), order, round(share * len(rows)))
                    ).items()
                    if key in kept
                }
                for share in STEPS
            ],
        }
        if name != "squad_probe":
            entry["squad_probe_minus_this"] = {
                label: probe_gates.auc_difference(clustered, label, by_id["squad_probe"], by_id[name])
                for label in probe_gates.LABELS[::-1]
            }
            k, counts = at_matched_coverage(rows, order, probe_gate["answered"])
            entry["at_the_probe_gate_coverage"] = {"abstained_on": k} | counts
        found["detectors"][name] = entry
    return found


def confidence(
    rows: Sequence[Mapping[str, Any]],
    held: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, float],
    frozen: probe.Probe,
    verdicts: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict, list[dict]]:
    """The base model's own confidence in its stored answers as a detector of its errors, beside the frozen probe;
    and a row per question without text.

    ``held`` gives per question the probability of the answer's first token and the mean log-probability of its
    tokens, None for an answer without text. Low confidence is taken as the sign of an error: the detector scores
    are 1 - the first and minus the second. No threshold is fitted: a baseline abstains on its k least confident
    answers, k the smallest number at which it answers no more questions than the probe's gate.
    """
    if set(held) != {row["id"] for row in rows}:
        raise SystemExit(f"{len(held)} confidence rows for {len(rows)} questions")

    def detectors(part: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        # a row without answer text has no confidence: it is ranked as the least suspicious
        first = np.array([1 - (held[row["id"]]["first_token_probability"] or 1.0) for row in part])
        mean = np.array([-(held[row["id"]]["mean_log_probability"] or 0.0) for row in part])
        read = np.array([scores[row["id"]] for row in part])
        return {"squad_probe": read, DETECTORS[0]: first, DETECTORS[1]: mean}

    fired = np.array([scores[row["id"]] >= frozen.threshold for row in rows])
    found = {
        "rows": len(rows),
        "rows_without_answer_text": sum(held[row["id"]]["first_token_probability"] is None for row in rows),
        "rows_whose_answer_tokens_are_all_the_argmax": sum(
            entry["greedy_tokens"] == entry["answer_tokens"] for entry in held.values()
        ),
        "answer_tokens_that_are_the_argmax": [
            sum(entry["greedy_tokens"] for entry in held.values()),
            sum(entry["answer_tokens"] for entry in held.values()),
        ],
        "word_match": detectors_read(rows, detectors(rows), fired),
    }
    if verdicts is not None:
        kept = probe_squad.judged(rows, verdicts)
        gate = np.array([scores[row["id"]] >= frozen.threshold for row in kept])
        found["judge"] = {"rows_judged": len(kept)} | detectors_read(kept, detectors(kept), gate)
    articles = probe_squad.article_index(rows)
    public = [
        {
            "id": row["id"],
            "condition": row["condition"],
            "article_index": articles[row["title"]],
            "rule_wrong": row["base_wrong"],
            "first_token_probability": held[row["id"]]["first_token_probability"],
            "mean_log_probability": held[row["id"]]["mean_log_probability"],
            "answer_tokens": held[row["id"]]["answer_tokens"],
            "squad_probe_score": round(scores[row["id"]], 6),
        }
        for row in rows
    ]
    return found, public


def identity(stored: Mapping[str, Mapping[str, Any]], kit: Mapping[str, Mapping[str, Any]]) -> dict:
    """How many answers of two runs over the same prompts are equal in text, token count and stop reason."""
    if set(stored) != set(kit):
        raise SystemExit(f"{len(kit)} answers to compare with {len(stored)} stored ones")
    same = {field: sum(stored[key][field] == kit[key][field] for key in stored) for field in COMPARED}
    return {
        "rows": len(stored),
        "equal_text": same["response"],
        "equal_token_count": same["generated_tokens"],
        "equal_stop_reason": same["stop_reason"],
        "equal_on_all_three": sum(all(stored[key][field] == kit[key][field] for field in COMPARED) for key in stored),
        "auditkit_rows_with_an_error": sum(bool(row.get("error")) for row in kit.values()),
    }


def job_identity(stored: Sequence[Mapping[str, Any]], job: Sequence[Mapping[str, Any]]) -> dict:
    """The answers of a generate job against the stored ones of the same questions, rows as ``rows_of`` leaves
    them: equal text, token count and stop reason, and the rows whose word-match labels differ."""
    one, two = ({row["id"]: row for row in rows} for rows in (stored, job))
    none = np.zeros(len(stored), dtype=bool)
    return identity(one, two) | {
        "rows_whose_error_label_differs": sum(one[key]["base_wrong"] != two[key]["base_wrong"] for key in one),
        "rows_whose_abstention_differs": sum(one[key]["abstained"] != two[key]["abstained"] for key in one),
        "wrong_answers_stored": probe_squad.measured(stored, none)["wrong_answers"],
        "wrong_answers_job": probe_squad.measured(job, none)["wrong_answers"],
    }


def _sample(args: argparse.Namespace, asked: Path, answers: Path, states: Path) -> tuple[list[dict], dict, list[int]]:
    """The passing rows of one sample with the base model's answers classed, its hidden states and their layers."""
    rows = probe_squad.rows_of(asked, answers)
    held, layers = probe_squad.states_of(states, rows)
    rows, _ = probe_curator.passing(rows, asked, args.passed)
    return rows, held, layers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    paths = {
        "served-sample": ("dev", "passed", "out"),
        "served": ("sample", "served", "served-states", "dev", "dev-answers", "dev-states", "passed", "probe", "out"),
        "scored": ("passed", "probe", "out"),
        "fine-tune": ("dev", "dev-answers", "dev-states", "tuned-answers", "passed", "probe", "out"),
        "instructed-sample": ("dev", "passed", "out"),
        "confidence": ("dev", "dev-answers", "dev-states", "confidence", "passed", "probe", "out"),
        "identity": ("stored", "kit", "out"),
        "job-identity": ("passed", "out"),
    }
    for step, wanted in paths.items():
        sub = steps.add_parser(step)
        for name in wanted:
            sub.add_argument(f"--{name}", type=Path, required=True)
        if step == "served":
            sub.add_argument("--abstention", required=True, help="the sentence a fired row was made to answer")
        if step == "scored":
            sub.add_argument("--part", choices=PARTS, required=True, help="the name the record is written under")
            for name in ("dev", "dev-answers", "dev-states"):
                sub.add_argument(f"--{name}", type=Path, action="append", required=True)
            sub.add_argument("--verdicts", type=Path, action="append", default=[], help="a judge's verdicts")
        if step == "fine-tune":
            sub.add_argument("--verdicts", type=Path, help="a judge's verdicts on the base model's answers")
            sub.add_argument("--tuned-verdicts", type=Path, help="and on the fine-tune's answers")
            sub.add_argument("--part", choices=tuple(TUNED_PARTS), default="fine-tune", help="the record's name")
        if step == "confidence":
            sub.add_argument("--verdicts", type=Path, help="a judge's verdicts on the answers")
        if step == "job-identity":
            sub.add_argument("--name", action="append", required=True, help="a split, once per file given")
            for name in ("asked", "stored", "job"):
                sub.add_argument(f"--{name}", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    if args.step == "served-sample":
        rows, _ = probe_curator.passing(probe.read_jsonl(args.dev), args.dev, args.passed)
        sample = served_sample(rows)
        probe.write_jsonl(args.out, sample)
        print(f"{len(sample)} prompts -> {args.out.name}")
    elif args.step == "served":
        rows, states, layers = _sample(args, args.dev, args.dev_answers, args.dev_states)
        by_id = {row["id"]: row for row in rows}
        asked = probe.read_jsonl(args.sample)
        live = {row["id"]: row for row in probe.read_jsonl(args.served)}
        if set(live) != {row["id"] for row in asked} or not set(live) <= set(by_id):
            raise SystemExit("the served run does not hold the sample's prompts")
        frozen = probe.load(args.probe)
        run, run_layers = probe.load_states(args.served_states)
        place, run_place = layers.index(frozen.layer), run_layers.index(frozen.layer)
        sample = [by_id[row["id"]] for row in asked]
        record, public = served(
            sample,
            live,
            frozen,
            {row["id"]: states[row["id"]][place] for row in sample},
            {row["id"]: run[row["id"]][run_place] for row in sample},
            args.abstention,
            probe_squad.article_index(rows),
        )
        probe_study.write(args.out, "squad-served", record, public, tables=table)
    elif args.step == "scored":
        if not len(args.dev) == len(args.dev_answers) == len(args.dev_states):
            parser.error("--dev, --dev-answers and --dev-states are given once per sample")
        if args.verdicts and len(args.verdicts) != len(args.dev):
            parser.error("--verdicts is given for every sample or for none")
        frozen = probe.load(args.probe)
        rows, scores, verdicts = [], {}, {} if args.verdicts else None
        for number, (asked, answers, folder) in enumerate(
            zip(args.dev, args.dev_answers, args.dev_states, strict=True)
        ):
            part, states, layers = _sample(args, asked, answers, folder)
            read = frozen.probability(np.stack([states[row["id"]][layers.index(frozen.layer)] for row in part]))
            scores |= dict(zip([row["id"] for row in part], map(float, read), strict=True))
            rows += part
            if verdicts is not None:
                verdicts |= probe_squad.read_verdicts(args.verdicts[number])
        record, public = scored(rows, scores, frozen, verdicts)
        probe_study.write(args.out, f"squad-{args.part}", record, public, tables=table)
    elif args.step == "fine-tune":
        if bool(args.verdicts) != bool(args.tuned_verdicts):
            parser.error("--verdicts and --tuned-verdicts are given together or not at all")
        frozen = probe.load(args.probe)
        base, states, layers = _sample(args, args.dev, args.dev_answers, args.dev_states)
        tuned, _ = probe_curator.passing(probe_squad.rows_of(args.dev, args.tuned_answers), args.dev, args.passed)
        read = frozen.probability(np.stack([states[row["id"]][layers.index(frozen.layer)] for row in base]))
        scores = dict(zip([row["id"] for row in base], map(float, read), strict=True))
        seen = [probe_squad.read_verdicts(path) if path else None for path in (args.verdicts, args.tuned_verdicts)]
        record, public = fine_tune(base, tuned, scores, frozen, *seen, label=TUNED_PARTS[args.part])
        probe_study.write(args.out, f"squad-{args.part}", record, public, tables=table)
    elif args.step == "instructed-sample":
        rows, _ = probe_curator.passing(probe.read_jsonl(args.dev), args.dev, args.passed)
        probe.write_jsonl(args.out, instructed_sample(rows))
        print(f"{len(rows)} prompts -> {args.out.name}")
    elif args.step == "confidence":
        frozen = probe.load(args.probe)
        rows, states, layers = _sample(args, args.dev, args.dev_answers, args.dev_states)
        read = frozen.probability(np.stack([states[row["id"]][layers.index(frozen.layer)] for row in rows]))
        scores = dict(zip([row["id"] for row in rows], map(float, read), strict=True))
        held = {row["id"]: row for row in probe.read_jsonl(args.confidence)}
        verdicts = probe_squad.read_verdicts(args.verdicts) if args.verdicts else None
        record, public = confidence(rows, held, scores, frozen, verdicts)
        probe_study.write(args.out, "squad-confidence", record, public, tables=table)
    elif args.step == "job-identity":
        if not len(args.name) == len(args.asked) == len(args.stored) == len(args.job):
            parser.error("--name, --asked, --stored and --job are given once per split")
        splits = {}
        for name, asked, stored, job in zip(args.name, args.asked, args.stored, args.job, strict=True):
            runs = [
                probe_curator.passing(probe_squad.rows_of(asked, path), asked, args.passed)[0] for path in (stored, job)
            ]
            splits[name] = job_identity(*runs)
        probe_study.write(args.out, "squad-job-identity", {"splits": splits}, tables=table)
    else:
        stored, kit = probe_squad.answers_of(args.stored), probe_squad.answers_of(args.kit)
        probe_study.write(args.out, "squad-identity", identity(stored, kit), tables=table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
