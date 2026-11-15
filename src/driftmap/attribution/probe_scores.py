"""Score the frozen probe and the gated system on the evaluation sets.

    python -I -m driftmap.attribution.probe_scores --probe OUT --sets SETS.json --prompts OUT/probe_eval_in.jsonl \
        --states OUT/acts/eval --out RESULTS --fine-tune defense

``--probe`` is the folder ``probe_fit`` wrote, ``--sets`` the file of evaluation sets ``probe.read_sets`` reads,
``--prompts`` the evaluation prompt file ``probe_rows`` wrote and ``--states`` the folder ``probe_extract`` wrote
for it. ``--fine-tune`` names the fine-tune whose stored answers the sets carry beside the base model's.

Nothing is generated here. The base model's answers are the stored ones; the probe reads the hidden state of the
same prompt; the gated answer is the unknown option where the probe fires and the stored answer elsewhere. So
the gate is computed on stored answers, it is not served through a hook in a running model.

Written under ``--out``: ``probe[-<fine-tune>].json``, the record ``table.probe`` builds, ``probe[-<fine-tune>].md``,
the same as Markdown, and ``probe[-<fine-tune>]-answers.jsonl``, one row per set and prompt with the probe's score,
whether it fired and the letter of each system. No prompt, option or answer text is in any of the three.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from driftmap.attribution import probe
from driftmap.core import ids
from driftmap.core.records import dump
from driftmap.table import probe as table

NAME = "probe.json"


def scored(sets: dict[str, list[dict]], scores: dict[str, float], threshold: float) -> dict[str, list[table.Answered]]:
    """Per set its rows as the table counts them; ``scores`` holds the probe's probability by prompt text."""
    lacking = sorted({row["id"] for rows in sets.values() for row in rows if row["prompt"] not in scores})
    if lacking:
        raise SystemExit(
            f"{len(lacking)} prompts of the evaluation sets have no hidden state, the first is {lacking[0]}"
        )
    return {
        name: [table.answered(row, scores[row["prompt"]], threshold) for row in rows] for name, rows in sets.items()
    }


def answer_rows(sets: dict[str, list[table.Answered]], spec: Path) -> list[dict]:
    """One row per prompt of every set read from files, without any text; a set made of other sets adds none."""
    whole = {entry["name"] for entry in json.loads(spec.read_text(encoding="utf-8"))["sets"] if "of" in entry}
    return [
        {
            "set": name,
            "id": row.id,
            "condition": row.condition,
            "probe_score": round(row.score, 6),
            "fired": row.fired,
            "letter_base": row.letters[probe.BASE],
            "letter_base_plus_probe": row.letters[probe.GATED],
            "letter_fine_tuned": row.letters[probe.TUNED],
            "letter_unknown": row.letter_unknown,
            "letter_correct": row.letter_correct,
        }
        for name, rows in sets.items()
        if name not in whole
        for row in rows
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--probe", type=Path, required=True, help="the folder of the frozen probe")
    parser.add_argument("--sets", type=Path, required=True, help="the file of evaluation sets")
    parser.add_argument("--prompts", type=Path, required=True, help="the evaluation prompt file the states are of")
    parser.add_argument("--states", type=Path, required=True, help="the folder of hidden states of those prompts")
    parser.add_argument("--out", type=Path, required=True, help="folder for the record, its Markdown and the rows")
    parser.add_argument("--fine-tune", choices=tuple(ids.FINE_TUNES), default="standard", help="the fine-tune beside")
    args = parser.parse_args(argv)
    frozen = probe.load(args.probe)
    states, layers = probe.load_states(args.states)
    asked = probe.read_jsonl(args.prompts)
    if frozen.layer not in layers or set(states) != {row["id"] for row in asked}:
        raise SystemExit(
            f"{len(states)} hidden states of layers {layers} for {len(asked)} prompts and layer {frozen.layer}"
        )
    place = layers.index(frozen.layer)
    read = frozen.probability(np.stack([states[row["id"]][place] for row in asked]))
    scores = {row["prompt"]: float(value) for row, value in zip(asked, read, strict=True)}
    sets = scored(probe.read_sets(args.sets), scores, frozen.threshold)
    found = {"layer": frozen.layer, "c": frozen.c, "threshold": frozen.threshold}
    built = table.build(sets, seed=probe.SEED, fine_tune=args.fine_tune, **found)
    stem = NAME if args.fine_tune == "standard" else f"{Path(NAME).stem}-{args.fine_tune}{Path(NAME).suffix}"
    path = args.out / stem
    dump(built, path)
    path.with_suffix(".md").write_text(table.render(built), encoding="utf-8")
    probe.write_jsonl(path.with_name(f"{path.stem}-answers.jsonl"), answer_rows(sets, args.sets))
    print(table.render(built))
    return 0


if __name__ == "__main__":
    sys.exit(main())
