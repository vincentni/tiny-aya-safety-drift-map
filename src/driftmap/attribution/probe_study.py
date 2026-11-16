"""The probe study, part by part: each step reads stored answers and hidden states and writes one record.

    python -I -m driftmap.attribution.probe_study error-fit --rows TRAIN_IN --answers TRAIN_ANSWERS \
        --states ACTS/train --out OUT
    python -I -m driftmap.attribution.probe_study bbq --sets SETS.json --prompts EVAL_IN --states ACTS/eval \
        --probe probe_1=PROBE_1 --probe error_probe=OUT/probe_error --out RESULTS
    python -I -m driftmap.attribution.probe_study baselines --rows TRAIN_IN --labels OUT/probe_error_rows.jsonl \
        --sets SETS.json --prompts EVAL_IN --states ACTS/eval --probe probe_1=PROBE_1 \
        --probe error_probe=OUT/probe_error --model MODEL_DIR --manifest EVAL/manifest.json --out RESULTS
    python -I -m driftmap.attribution.probe_study served-sample --bbq-rows BBQ_ROWS --mbbq MBBQ_PROMPTS --out SAMPLE
    python -I -m driftmap.attribution.probe_study served --sample SAMPLE --served SERVED --ungated UNGATED \
        --served-states ACTS/served --sets SETS.json --prompts EVAL_IN --states ACTS/eval --probe PROBE_1 --out RESULTS
    python -I -m driftmap.attribution.probe_study squad-fit --train TRAIN_IN --answers TRAIN_ANSWERS \
        --states ACTS/squad_train --model MODEL_DIR --manifest EVAL/manifest.json --out OUT
    python -I -m driftmap.attribution.probe_study squad --dev DEV_IN --dev-answers DEV_ANSWERS \
        --dev-states ACTS/squad_dev --train TRAIN_IN --train-answers TRAIN_ANSWERS --zero-shot probe_1=PROBE_1 \
        --zero-shot error_probe=OUT/probe_error --in-domain OUT/probe_squad --model MODEL_DIR \
        --manifest EVAL/manifest.json --out RESULTS

``error-fit`` fits the probe of the base model's own error on the rows of the first probe, with the rule that
froze it. ``bbq`` scores it beside the first probe on the evaluation sets. ``baselines`` fits the text-only
baselines for both labels and scores them beside the two probes. ``served-sample`` and ``served`` compare a gate
served inside ``probe_generate`` with the gate computed afterwards. ``squad-fit`` fits a probe of the base model's
error on the SQuAD train sample; ``squad`` scores it once on the dev sample, beside the earlier probes read
zero-shot and the text baselines of the same label and split.

Every part writes ``probe-<part>.json``, the same as Markdown, and a row file of ids, scores and letters or answer
classes. No prompt, passage, question, option or answer text is in any of them. Outside ``served``, every gate
is computed on stored answers.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from driftmap.attribution import local_generate, probe, probe_baselines, probe_fit, probe_gates, probe_squad
from driftmap.table import probe_study as table

ERROR = "probe_error"
RENAMED = (
    ("dev_ambiguous_rows", "dev_base_wrong_rows"),
    ("dev_disambiguated_rows", "dev_base_right_rows"),
    ("dev_fired_on_ambiguous", "dev_fired_on_base_wrong"),
    ("dev_fired_on_disambiguated", "dev_fired_on_base_right"),
)
# the activation probe of each label, by the name its gate goes under
ACTIVATION = {"answerability": "probe_1", "base_error": "error_probe"}
WHOLE = ("mbbq_non_english",)
# a SQuAD prompt ends with the question and the closing line
SQUAD_TAIL = 2
# the pairs the served gate is run on: every such pair of the test items, and of the items drawn from MBBQ
EVERY = {"bbq_test_en": 5, "mbbq": 10}


def write(out: Path, part: str, record: dict, rows: Sequence[dict] | None = None, suffix: str = "answers") -> None:
    """Write a part's record, its Markdown and, where given, its row file under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    record = {"schema": table.SCHEMAS[part]} | record
    (out / f"probe-{part}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (out / f"probe-{part}.md").write_text(table.RENDER[part](record), encoding="utf-8")
    if rows is not None:
        probe.write_jsonl(out / f"probe-{part}-{suffix}.jsonl", rows)


def named(pairs: Sequence[str]) -> dict[str, Path]:
    found = dict(pair.split("=", 1) for pair in pairs if "=" in pair)
    if len(found) != len(pairs):
        raise SystemExit("a probe is given as NAME=FOLDER")
    return {name: Path(folder) for name, folder in found.items()}


def renamed(record: dict) -> dict:
    """The fit record of a probe whose label is the base model's error."""
    for old, new in RENAMED:
        record[new] = record.pop(old)
    return record


def error_fit(rows: list[dict], said: dict[str, dict], states: dict, layers: list[int]) -> tuple[probe.Probe, dict]:
    """The probe of the base model's own error on the first probe's rows, and its record; a row's label is 1
    when the base model's answer to it is not the correct letter."""
    for row in rows:
        row["letter_base"] = probe_gates.letter_of(said[row["id"]]["response"], row)
        row["ambiguous"] = row["condition"] == probe.AMBIGUOUS
        row["base_wrong"] = row["letter_base"] != row["letter_correct"]
        row["answerability_label"], row["label"] = row["label"], int(row["base_wrong"])
    frozen, record = probe_fit.fit(rows, states, layers)
    parts = {split: [row for row in rows if row["split"] == split] for split in ("fit", "dev")}
    answers = {
        split: {
            "rows": len(part),
            "unparseable": sum(row["letter_base"] is None for row in part),
            "names_a_person_on_ambiguous": sum(row["base_wrong"] for row in part if row["ambiguous"]),
            "correct_on_disambiguated": sum(not row["base_wrong"] for row in part if not row["ambiguous"]),
        }
        for split, part in parts.items()
    }
    relation = {split: probe_gates.relation(part) for split, part in parts.items()}
    said_of = {"label": "base model's own error", "base_answers": answers, "label_relation": relation}
    return frozen, said_of | renamed(record)


def gates_of(folders: dict[str, Path], prompts: Path, states: Path) -> dict[str, dict]:
    found = {}
    for name, folder in folders.items():
        scores, threshold = probe_gates.activation_scores(folder, prompts, states)
        found[name] = {"scores": scores, "threshold": threshold}
    return found


def bbq(sets: dict, gates: dict[str, dict], folders: dict[str, Path]) -> dict:
    """The record of the evaluation sets under each stored probe, with every pair of systems compared."""
    names = list(gates)
    compare = [("base", name) for name in names] + [(a, b) for at, a in enumerate(names) for b in names[at + 1 :]]
    settings = {}
    for name, folder in folders.items():
        frozen = probe.load(folder)
        settings[name] = {"layer": frozen.layer, "C": frozen.c, "threshold": frozen.threshold}
    return {"probes": settings, "sets": probe_gates.score_sets(sets, gates, compare)}


def baselines_fit(rows: list[dict], wrong: dict[str, bool], tokenizer) -> tuple[dict[str, dict], dict]:
    """The text baselines of both labels on the first probe's rows and split, and the record of their fits."""
    labels = {
        "answerability": np.array([row["label"] for row in rows]),
        "base_error": np.array([int(wrong[row["id"]]) for row in rows]),
    }
    in_fit = np.array([row["split"] == "fit" for row in rows])
    held = probe_baselines.features(tokenizer, [row["prompt"] for row in rows])
    models, record = {}, {"fit_rows": int(in_fit.sum()), "dev_rows": int((~in_fit).sum())}
    fitted = {label: probe_baselines.fit(held, y, in_fit) for label, y in labels.items()}
    for feature in probe_baselines.FEATURES:
        for label in labels:
            models[f"{feature}_{label}"] = fitted[label][0][feature]
            record[f"{feature}_{label}"] = fitted[label][1][feature]
    lengths = held[probe_baselines.LENGTH].ravel()
    record["context_tokens"] = {
        "fit_ambiguous_median": float(np.median(lengths[in_fit & (labels["answerability"] == 1)])),
        "fit_disambiguated_median": float(np.median(lengths[in_fit & (labels["answerability"] == 0)])),
    }
    record["tfidf_vocabulary"] = {
        name: len(model["pipeline"].steps[0][1].vocabulary_)
        for name, model in models.items()
        if name.startswith("tfidf")
    }
    return models, record


def baselines(sets: dict, gates: dict[str, dict], models: dict[str, dict], tokenizer) -> tuple[dict, dict[str, dict]]:
    """The record of the evaluation sets under the activation probes and the text baselines: the gated counts,
    and per label the AUC of its activation probe minus that of each baseline; and every gate, the baselines'
    among them."""
    prompts = sorted({row["prompt"] for rows in sets.values() for row in rows})
    held = probe_baselines.features(tokenizer, prompts)
    gates = dict(gates)
    for name, model in models.items():
        read = model["pipeline"].predict_proba(held[name.split("_")[0]])[:, 1]
        gates[name] = {"scores": dict(zip(prompts, map(float, read), strict=True)), "threshold": model["threshold"]}
    found = probe_gates.score_sets(sets, gates, [("base", name) for name in gates])
    for name, rows in sets.items():
        found[name]["auc_differences"] = {
            label: {
                f"{own} minus {feature}_{label}": probe_gates.auc_difference(
                    rows, label, gates[own]["scores"], gates[f"{feature}_{label}"]["scores"]
                )
                for feature in probe_baselines.FEATURES
            }
            for label, own in ACTIVATION.items()
        }
    return {"thresholds": {name: gate["threshold"] for name, gate in gates.items()}, "sets": found}, gates


def every(rows: Sequence[dict], step: int) -> set[str]:
    """The base items at positions 0, step, 2 x step, ... of their first appearance in ``rows``."""
    return set(list(dict.fromkeys(row["base_item"] for row in rows))[::step])


def served_sample(plain: Sequence[dict], official: dict[str, list[dict]]) -> list[dict]:
    """The prompts the served gate runs on: id (the set and the row's own id), prompt and the unknown letter."""
    picked = every(plain, EVERY["bbq_test_en"])
    rows = [
        {"id": f"bbq_test_en|{row['id']}", "prompt": row["prompt"], "letter_unknown": row["letter_unknown"]}
        for row in plain
        if row["base_item"] in picked
    ]
    picked = every(official["en"], EVERY["mbbq"])
    for language, held in official.items():
        rows += [
            {
                "id": f"mbbq_{language}|{row['prompt_id']}",
                "prompt": row["prompt"],
                "letter_unknown": row["letter_unknown"],
            }
            for row in held
            if row["base_item"] in picked
        ]
    return rows


def served(
    asked: Sequence[dict],
    live_rows: dict[str, dict],
    free_rows: dict[str, dict],
    sets: dict,
    offline: dict[str, float],
    frozen: probe.Probe,
    stored_states: dict[str, np.ndarray],
    run_states: dict[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    """The served gate against the gate computed afterwards, and a row per prompt without text.

    ``offline`` and ``stored_states`` are by prompt text, the probe's probability and the stored state of its
    layer; ``run_states`` by sample id, the state of that layer in the served run.
    """
    if set(live_rows) != {row["id"] for row in asked} or set(free_rows) != set(live_rows):
        raise SystemExit("the served and un-gated runs do not hold the sample's prompts")
    stored = {f"{name}|{row['id']}": row for name, rows in sets.items() if name not in WHOLE for row in rows}
    counts, public, gaps = defaultdict(lambda: defaultdict(int)), [], defaultdict(list)
    for row in asked:
        name = row["id"].split("|")[0]
        group = "bbq_test_en (stored answers: CPU)" if name == "bbq_test_en" else "mbbq (stored answers: GPU)"
        kept, live, free = stored[row["id"]], live_rows[row["id"]], free_rows[row["id"]]
        fired_offline = offline[row["prompt"]] >= frozen.threshold
        letter_served = probe_gates.letter_of(live["response"], kept)
        letter_free = probe_gates.letter_of(free["response"], kept)
        gated_stored = kept["letter_unknown"] if fired_offline else kept["base"]
        gated_cpu = kept["letter_unknown"] if fired_offline else letter_free
        tally = counts[group]
        tally["prompts"] += 1
        tally["served_equals_offline_gate_on_stored_answers"] += letter_served == gated_stored
        tally["served_equals_offline_gate_on_ungated_cpu_answers"] += letter_served == gated_cpu
        tally["firing_decision_equal"] += bool(live["fired"]) == bool(fired_offline)
        tally["fired"] += bool(live["fired"])
        alone = bool(live["fired"]) and live["response"] == kept["letter_unknown"]
        tally["fired_rows_answering_the_unknown_letter_alone"] += alone
        tally["unfired_rows"] += not live["fired"]
        tally["unfired_rows_with_the_ungated_text"] += not live["fired"] and live["response"] == free["response"]
        tally["ungated_cpu_letter_equals_stored_letter"] += letter_free == kept["base"]
        tally["ungated_cpu_text_equals_stored_text"] += free["response"] == kept["responses"]["base"]
        gaps["probability"].append(abs(live["probe_probability"] - offline[row["prompt"]]))
        gaps["hook_to_run_hidden_state"].append(live["hook_minus_hidden_state"])
        moved = np.abs(run_states[row["id"]] - stored_states[row["prompt"]]).max()
        gaps["run_state_to_stored_activation"].append(float(moved))
        public.append(
            {
                "id": row["id"],
                "condition": kept["condition"],
                "probe_probability_served": round(live["probe_probability"], 6),
                "probe_probability_offline": round(offline[row["prompt"]], 6),
                "fired_served": bool(live["fired"]),
                "fired_offline": bool(fired_offline),
                "letter_served": letter_served,
                "letter_offline_gate": gated_stored,
                "letter_ungated_cpu": letter_free,
                "letter_stored_base": kept["base"],
                "letter_unknown": kept["letter_unknown"],
                "letter_correct": kept["letter_correct"],
            }
        )
    total = defaultdict(int)
    for tally in counts.values():
        for key, value in tally.items():
            total[key] += value
    record = {
        "probe": {"layer": frozen.layer, "threshold": frozen.threshold},
        "all": dict(total),
        "by_set": {name: dict(tally) for name, tally in counts.items()},
        "largest_absolute_differences": {key: max(values) for key, values in gaps.items()},
    }
    return record, public


def squad_split(rows: list[dict]) -> None:
    """Mark each row of the train sample ``fit`` or ``dev``: whole articles, a fifth of them dev, the answerable
    and the unanswerable questions spread alike; its label is the base model's error."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=probe.SEED)
    conditions, titles = ([row[key] for row in rows] for key in ("condition", "title"))
    dev = set(next(folds.split(rows, conditions, titles))[1].tolist())
    for number, row in enumerate(rows):
        row["split"], row["label"] = ("dev" if number in dev else "fit"), int(row["base_wrong"])


def squad_baselines(rows: list[dict], tokenizer) -> tuple[dict[str, dict], dict[str, dict]]:
    """The text baselines of the base model's error on the split train sample, and the records of their fits."""
    in_fit = np.array([row["split"] == "fit" for row in rows])
    held = probe_baselines.features(tokenizer, [row["prompt"] for row in rows], SQUAD_TAIL)
    return probe_baselines.fit(held, np.array([row["label"] for row in rows]), in_fit)


def squad_fit(rows: list[dict], states: dict, layers: list[int], tokenizer) -> tuple[probe.Probe, dict]:
    """The in-domain probe of the base model's error on the train sample, and the record of the fit with the
    text baselines of the same label and split beside it."""
    squad_split(rows)
    frozen, record = probe_fit.fit(rows, states, layers)
    parts = {split: [row for row in rows if row["split"] == split] for split in ("fit", "dev")}
    shared = {row["title"] for row in parts["fit"]} & {row["title"] for row in parts["dev"]}
    return frozen, {
        "label": "base model's own error (reading check's rule)",
        "split": {
            split: probe_squad.base_counts(part) | {"articles": len({row["title"] for row in part})}
            for split, part in parts.items()
        },
        "articles_in_both_splits": len(shared),
        "probe": renamed(record),
        "baselines": squad_baselines(rows, tokenizer)[1],
    }


def squad(
    rows: list[dict],
    states: dict,
    layers: list[int],
    zero_shot: dict[str, Path],
    in_domain: Path,
    models: dict[str, dict],
    tokenizer,
) -> tuple[dict, dict[str, list[dict]]]:
    """The record of the dev sample: the base model, the earlier probes read zero-shot, and the in-domain probe
    beside the text baselines; and per mode the rows without text."""
    held = probe_baselines.features(tokenizer, [row["prompt"] for row in rows], SQUAD_TAIL)
    modes = {"zero_shot": dict(zero_shot), "in_domain": {"squad_probe": in_domain}}
    record = {
        "dev_sample": probe_squad.base_counts(rows) | {"articles": len({row["title"] for row in rows})},
        "base": probe_squad.measured(rows, np.zeros(len(rows), dtype=bool)),
    }
    written = {}
    for mode, folders in modes.items():
        gates = {}
        for name, folder in folders.items():
            frozen = probe.load(folder)
            read = frozen.probability(np.stack([states[row["id"]][layers.index(frozen.layer)] for row in rows]))
            gates[name] = (read, frozen.threshold, {"layer": frozen.layer, "C": frozen.c})
        if mode == "in_domain":
            for name, model in models.items():
                gates[name] = (model["pipeline"].predict_proba(held[name])[:, 1], model["threshold"], {"C": model["C"]})
        record[mode] = {
            name: settings | probe_squad.read_gate(rows, read, threshold)
            for name, (read, threshold, settings) in gates.items()
        }
        if mode == "in_domain":
            clustered = [
                dict(row, cluster=row["title"], prompt=row["id"], ambiguous=row["unanswerable"]) for row in rows
            ]
            by_id = {
                name: dict(zip([row["id"] for row in rows], map(float, read), strict=True))
                for name, (read, _, _) in gates.items()
            }
            record["in_domain_auc_differences"] = {
                f"squad_probe minus {name}": probe_gates.auc_difference(
                    clustered, "base_error", by_id["squad_probe"], by_id[name]
                )
                for name in probe_baselines.FEATURES
            }
        written[mode] = [
            {
                "id": row["id"],
                "condition": row["condition"],
                "base_answer_wording": row["wording"],
                "base_answer_holds_gold": row["holds_gold"],
                "base_wrong": row["base_wrong"],
            }
            | {f"{name}_score": round(float(read[index]), 6) for name, (read, _, _) in gates.items()}
            | {f"{name}_fired": bool(read[index] >= threshold) for name, (read, threshold, _) in gates.items()}
            for index, row in enumerate(rows)
        ]
    return record, written


def _tokenizer(args: argparse.Namespace):
    return local_generate.load_rendering(args.model, args.manifest, "keep")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    paths = {
        "error-fit": ("rows", "answers", "states", "out"),
        "bbq": ("sets", "prompts", "states", "out"),
        "baselines": ("rows", "labels", "sets", "prompts", "states", "model", "manifest", "out"),
        "served-sample": ("bbq-rows", "mbbq", "out"),
        "served": ("sample", "served", "ungated", "served-states", "sets", "prompts", "states", "probe", "out"),
        "squad-fit": ("train", "answers", "states", "model", "manifest", "out"),
        "squad": (
            "dev",
            "dev-answers",
            "dev-states",
            "train",
            "train-answers",
            "in-domain",
            "model",
            "manifest",
            "out",
        ),
    }
    for step, wanted in paths.items():
        sub = steps.add_parser(step)
        for name in wanted:
            sub.add_argument(f"--{name}", type=Path, required=True)
        if step in ("bbq", "baselines"):
            sub.add_argument("--probe", action="append", default=[], metavar="NAME=FOLDER", required=True)
        if step == "squad":
            sub.add_argument("--zero-shot", action="append", default=[], metavar="NAME=FOLDER")
    args = parser.parse_args(argv)
    if args.step == "error-fit":
        rows = probe.read_jsonl(args.rows)
        said = {row["id"]: row for row in probe.read_jsonl(args.answers)}
        if set(said) != {row["id"] for row in rows}:
            raise SystemExit(f"{len(said)} answers for {len(rows)} training rows")
        states, layers = probe.load_states(args.states)
        frozen, record = error_fit(rows, said, states, layers)
        label = "1 when the base model's own answer to the prompt is not the correct letter"
        probe.save(frozen, args.out / ERROR, layers_extracted=list(layers), fitted_on=args.rows.name, label=label)
        (args.out / f"{ERROR}_fit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        kept = ("id", "split", "condition", "letter_base", "letter_correct", "letter_unknown", "base_wrong")
        probe.write_jsonl(args.out / f"{ERROR}_rows.jsonl", [{key: row[key] for key in kept} for row in rows])
        print(json.dumps({"selected": record["selected"], "threshold": record["threshold"]}, indent=1))
    elif args.step == "bbq":
        folders = named(args.probe)
        gates, sets = gates_of(folders, args.prompts, args.states), probe_gates.load_sets(args.sets)
        write(args.out, "bbq", bbq(sets, gates, folders), probe_gates.answer_rows(sets, gates, WHOLE))
    elif args.step == "baselines":
        rows = probe.read_jsonl(args.rows)
        wrong = {row["id"]: row["base_wrong"] for row in probe.read_jsonl(args.labels)}
        tokenizer = _tokenizer(args)
        models, fitted = baselines_fit(rows, wrong, tokenizer)
        sets = probe_gates.load_sets(args.sets)
        record, gates = baselines(sets, gates_of(named(args.probe), args.prompts, args.states), models, tokenizer)
        write(args.out, "baselines", record | {"fit": fitted}, probe_gates.answer_rows(sets, gates, WHOLE))
    elif args.step == "served-sample":
        plain = [row for row in probe.read_jsonl(args.bbq_rows) if not row.get("arm_tag")]
        official = {path.stem: probe.read_jsonl(path) for path in sorted(args.mbbq.glob("*.jsonl"))}
        rows = served_sample(plain, official)
        probe.write_jsonl(args.out, rows)
        print(f"{len(rows)} prompts -> {args.out.name}")
    elif args.step == "served":
        frozen = probe.load(args.probe)
        offline, _ = probe_gates.activation_scores(args.probe, args.prompts, args.states)
        states, layers = probe.load_states(args.states)
        place = layers.index(frozen.layer)
        by_prompt = {row["prompt"]: states[row["id"]][place] for row in probe.read_jsonl(args.prompts)}
        run, run_layers = probe.load_states(args.served_states)
        run = {key: value[run_layers.index(frozen.layer)] for key, value in run.items()}
        live, free = ({row["id"]: row for row in probe.read_jsonl(path)} for path in (args.served, args.ungated))
        asked, sets = probe.read_jsonl(args.sample), probe_gates.load_sets(args.sets)
        record, rows = served(asked, live, free, sets, offline, frozen, by_prompt, run)
        write(args.out, "served", record, rows)
    elif args.step == "squad-fit":
        rows = probe_squad.rows_of(args.train, args.answers)
        states, layers = probe_squad.states_of(args.states, rows)
        frozen, record = squad_fit(rows, states, layers, _tokenizer(args))
        label = "1 when the base model's own answer to the prompt is wrong by the reading check's rule"
        probe.save(
            frozen, args.out / "probe_squad", layers_extracted=list(layers), fitted_on=args.train.name, label=label
        )
        (args.out / "probe_squad_fit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps({"selected": record["probe"]["selected"], "threshold": record["probe"]["threshold"]}, indent=1)
        )
    else:
        tokenizer = _tokenizer(args)
        train = probe_squad.rows_of(args.train, args.train_answers)
        squad_split(train)
        models, _ = squad_baselines(train, tokenizer)
        rows = probe_squad.rows_of(args.dev, args.dev_answers)
        states, layers = probe_squad.states_of(args.dev_states, rows)
        record, written = squad(rows, states, layers, named(args.zero_shot), args.in_domain, models, tokenizer)
        write(args.out, "squad", record)
        for mode, held in written.items():
            probe.write_jsonl(args.out / f"probe-squad-answers-{mode.replace('_', '-')}.jsonl", held)
    return 0


if __name__ == "__main__":
    sys.exit(main())
