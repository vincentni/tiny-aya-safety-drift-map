"""Checks of the SQuAD probe after it was frozen: served in a running model, read by a judge, answers by identity.

    python -I -m driftmap.attribution.probe_checks served-sample --dev DEV_IN --passed CURATED/squad_passed.json \
        --out SAMPLE
    python -I -m driftmap.attribution.probe_checks served --sample SAMPLE --served SERVED --served-states ACTS/served \
        --dev DEV_IN --dev-answers DEV_ANSWERS --dev-states ACTS/squad_dev --passed CURATED/squad_passed.json \
        --probe OUT/probe_squad --abstention "SENTENCE" --out RESULTS
    python -I -m driftmap.attribution.probe_checks scored --part judge --dev DEV_IN --dev-answers DEV_ANSWERS \
        --dev-states ACTS/squad_dev --verdicts JUDGE/dev_verdicts.jsonl --passed CURATED/squad_passed.json \
        --probe OUT/probe_squad --out RESULTS
    python -I -m driftmap.attribution.probe_checks identity --stored DEV_ANSWERS --kit KIT_ANSWERS --out RESULTS

``served-sample`` takes every tenth unanswerable and every tenth answerable question of the dev sample, in its
order; ``served`` compares the answers ``probe_generate --gate --abstention`` gave them with the gate computed
afterwards from the stored answers and hidden states, and scores the served answers by the reading check's rule.
``scored`` applies the frozen probe to the stored hidden states of one sample, or of several given one after the
other, and gives the table of ``probe_study squad`` for it; with the verdicts ``probe_judge`` wrote for the
answers, it gives the table again under the judge's labels and how far the word-match rules agree with the judge.
``identity`` compares two answer files of the same prompts row for row: text, token count and stop reason.

Nothing is fitted and no threshold is moved. Every step reads only the rows that passed the CuratorKIT gate and
writes ``probe-squad-<part>.json``, the same as Markdown and, but for ``identity``, rows of ids, classes, verdicts
and scores. No passage, question or answer text and none of the judge's reasons is in any of them.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from driftmap.attribution import probe, probe_curator, probe_squad, probe_study, reading_scores
from driftmap.table import probe_checks as table

# the positions of a condition's questions the served gate is run on
EVERY = 10
PARTS = ("judge", "replication", "pooled")
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
) -> tuple[dict, list[dict]]:
    """The served gate against the gate computed afterwards on ``sample``, rows as ``rows_of`` leaves them, and
    a row per prompt without text; the states are those of the probe's layer by question id."""
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
    public = []
    for row in rows:
        seen = by_judge.get(row["id"])
        public.append(
            {
                "id": row["id"],
                "condition": row["condition"],
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
        "identity": ("stored", "kit", "out"),
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
    else:
        stored = {row["id"]: row for row in probe.read_jsonl(args.stored)}
        kit = {row["id"]: row for row in probe.read_jsonl(args.kit)}
        probe_study.write(args.out, "squad-identity", identity(stored, kit), tables=table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
