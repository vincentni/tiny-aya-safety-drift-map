"""The prompt rows of the activation probe: the rows it is fitted on, and the prompts of the evaluation sets.

    python -I -m driftmap.attribution.probe_rows --bbq BBQ --data DATA --sets SETS.json --out OUT

``--bbq`` is the folder of the BBQ files, fetched and checked as ``build`` does. ``--data`` is the folder ``build``
wrote: the sidecar of its training set names the base items the attribution fine-tune trained on. ``--sets`` is the
file of evaluation sets ``probe.read_sets`` reads.

probe_train_in.jsonl: the English base items the fine-tune trained on, nothing else, each as its ambiguous row
(label 1) and its disambiguated twin (label 0) in the letter layout of the test items, the two sharing one option
order, with the unknown option and the correct person at A, B and C equally often. About a fifth of the base
items, whole question templates within each category, are marked ``dev``: the probe's layer, regularisation and
threshold are chosen on them, so that the choice is made on templates the fit never saw. No test item and no test
template is among the rows.

probe_eval_in.jsonl: every distinct prompt of the evaluation sets, under the id of the first set that holds it.
probe_build.json counts both files and what they share.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from driftmap.attribution import bbq, build, probe
from driftmap.attribution.build import by
from driftmap.core import ids
from driftmap.scripts.drift import META

FOLDS = 5
TRAIN, EVALUATION, SUMMARY = "probe_train_in.jsonl", "probe_eval_in.jsonl", "probe_build.json"


def training(repo: Path, data: Path, train_set: str = ids.TRAIN_SETS[0]) -> list[dict]:
    """The rows the probe is fitted on and chosen on, from the BBQ files under ``repo`` and the training sidecar
    under ``data``; a row's ``split`` is ``fit`` or ``dev``."""
    from sklearn.model_selection import StratifiedGroupKFold

    notes = probe.read_jsonl(data / ids.training_folder(train_set) / f"en{META}")
    trained = sorted({note["base_item"] for note in notes if note.get("kind") == "abstention"})
    with (repo / bbq.METADATA).open(newline="", encoding="utf-8") as handle:
        target = {(row["category"], int(row["example_id"])): row["target_loc"] for row in csv.DictReader(handle)}
    known = {
        item["base_item"]: item
        for category in build.TRAINED_CATEGORIES
        for item in build.load_items(repo, category, target)
    }
    items = [dict(known[name]) for name in trained]
    build.arrange(items, random.Random(probe.SEED))
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=probe.SEED)
    categories, templates = ([item[key] for item in items] for key in ("category", "template"))
    # the first fold is dev: whole templates, spread over the categories
    dev = set(next(folds.split(items, categories, templates))[1].tolist())
    rows = []
    for number, item in enumerate(items):
        for condition in build.CONDITIONS:
            placed = item["placed"]
            rows.append(
                {
                    "id": f"{item['base_item']}.{'amb' if condition == probe.AMBIGUOUS else 'dis'}",
                    "prompt": build.letter_prompt(item, condition, placed),
                    "set": "train",
                    "split": "dev" if number in dev else "fit",
                    "base_item": item["base_item"],
                    "condition": condition,
                    "label": int(condition == probe.AMBIGUOUS),
                    "category": item["category"],
                    "template": item["template"],
                    "polarity": item["polarity"],
                    "unknown_phrasing": item["phrasing"],
                    "options": [
                        {"letter": letter, "text": item["texts"][option], "role": item["roles"][option]}
                        for letter, option in zip(build.LETTERS, placed, strict=True)
                    ],
                    **build.letters(item, condition, placed),
                }
            )
    return rows


def evaluation(sets: dict[str, list[dict]]) -> list[dict]:
    """Every distinct prompt of the evaluation sets, in the order met, under the id of the first set holding it."""
    rows, seen = [], set()
    for name, held in sets.items():
        for row in held:
            if row["prompt"] not in seen:
                seen.add(row["prompt"])
                rows.append({"id": f"{name}|{row['id']}", "prompt": row["prompt"], "set": name, "source_id": row["id"]})
    return rows


def main(argv: Sequence[str] | None = None, download: Callable[[str], bytes] = bbq.download) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bbq", type=Path, required=True, help="folder of the BBQ files; missing ones are downloaded")
    parser.add_argument("--data", type=Path, required=True, help="the folder build wrote: training set and test file")
    parser.add_argument("--sets", type=Path, required=True, help="the file of evaluation sets")
    parser.add_argument("--out", type=Path, required=True, help="folder for the two prompt files and their counts")
    args = parser.parse_args(argv)
    bbq.fetch(args.bbq, download)
    train, held = training(args.bbq, args.data), evaluation(probe.read_sets(args.sets))
    tested = probe.read_jsonl(args.data / "test_en.jsonl")
    fit, dev = ([row for row in train if row["split"] == split] for split in ("fit", "dev"))
    shared = {
        "training_base_items_in_the_test_set": len(
            {row["base_item"] for row in train} & {row["base_item"] for row in tested}
        ),
        "training_templates_in_the_test_set": len(
            {row["template"] for row in train} & {row["template"] for row in tested}
        ),
        "training_prompts_in_an_evaluation_set": len(
            {row["prompt"] for row in train} & {row["prompt"] for row in held}
        ),
        "dev_templates_in_fit": len({row["template"] for row in dev} & {row["template"] for row in fit}),
    }
    if any(shared.values()):
        raise SystemExit(f"the probe's rows are not apart from what they are tested on: {shared}")
    probe.write_jsonl(args.out / TRAIN, train)
    probe.write_jsonl(args.out / EVALUATION, held)
    summary = {
        "seed": probe.SEED,
        "train_rows": len(train),
        "train_base_items": len({row["base_item"] for row in train}),
        "fit_base_items": len({row["base_item"] for row in fit}),
        "dev_base_items": len({row["base_item"] for row in dev}),
        "dev_templates": len({row["template"] for row in dev}),
        "fit_templates": len({row["template"] for row in fit}),
        "dev_base_items_by_category": by(row["category"] for row in dev if row["label"]),
        "fit_unknown_letter": by(row["letter_unknown"] for row in fit if row["label"]),
        "fit_correct_letter_of_twin": by(row["letter_correct"] for row in fit if not row["label"]),
        "unknown_wordings": by(row["unknown_phrasing"] for row in train if row["label"]),
        "evaluation_rows": dict(Counter(row["set"] for row in held)),
        "overlap": shared,
        "files_sha256": {
            name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in (TRAIN, EVALUATION)
        },
    }
    (args.out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
