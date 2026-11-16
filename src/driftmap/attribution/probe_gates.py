"""What the probe studies share: the evaluation sets with their two labels, gates, counts and intervals.

A gate is a score per prompt and a threshold. Where the score reaches the threshold the answer becomes the unknown
option; elsewhere the base model's stored answer stands. The gate is computed on stored answers. Two labels are
told apart: ``answerability`` (the text leaves the question open) and ``base_error`` (the base model's own answer
is not the right letter).
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap

from driftmap.attribution import probe
from driftmap.table import choice, stats

SEED = probe.SEED
N_RESAMPLES = 10000
AUC_RESAMPLES = 2000
LABELS = ("answerability", "base_error")
WHAT = ("twin_accuracy", "wrong_answers")
# a gate: the probability by prompt text under "scores", and its "threshold"
Gate = Mapping[str, Any]


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def letter_of(response: str | None, row: Mapping[str, Any]) -> str | None:
    """The letter an answer names, by the multiple-choice table's rule; ``row`` holds ``texts`` or ``options``."""
    texts = row.get("texts") or {option["letter"]: option["text"] for option in row["options"]}
    return choice.parse(response, texts)


def load_sets(spec: Path) -> dict[str, list[dict]]:
    """The evaluation sets ``probe.read_sets`` reads; each row gains ``base`` and ``fine_tuned`` (the letters of
    the stored answers) and its two labels, ``ambiguous`` and ``base_wrong``."""
    sets = probe.read_sets(spec)
    for rows in sets.values():
        for row in rows:
            if "base" not in row:
                row["base"] = letter_of(row["responses"][probe.BASE], row)
                row["fine_tuned"] = letter_of(row["responses"][probe.TUNED], row)
                row["ambiguous"] = row["condition"] == probe.AMBIGUOUS
                row["base_wrong"] = row["base"] != row["letter_correct"]
    return sets


def label_of(row: Mapping[str, Any], label: str) -> bool:
    return row["ambiguous"] if label == LABELS[0] else row["base_wrong"]


def activation_scores(folder: Path, rows_file: Path, states_folder: Path) -> tuple[dict[str, float], float]:
    """A stored probe's probability by prompt text over the rows of ``rows_file``, and its threshold."""
    frozen = probe.load(folder)
    states, layers = probe.load_states(states_folder)
    asked = probe.read_jsonl(rows_file)
    if set(states) != {row["id"] for row in asked}:
        raise SystemExit(f"{len(states)} hidden states for {len(asked)} rows of {rows_file.name}")
    read = frozen.probability(np.stack([states[row["id"]][layers.index(frozen.layer)] for row in asked]))
    return {row["prompt"]: float(value) for row, value in zip(asked, read, strict=True)}, frozen.threshold


def auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """How well ``scores`` tell the rows of the label from the others, to 4 decimals; None with one kind alone."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=bool)
    return round(float(roc_auc_score(labels, scores)), 4) if labels.any() and not labels.all() else None


def measured(rows: Sequence[Mapping[str, Any]], letter: Callable[[Mapping[str, Any]], str | None]) -> dict:
    """The counts of one system over a set; ``letter`` gives the system's letter for a row."""
    ambiguous = [row for row in rows if row["ambiguous"]]
    settled = [row for row in rows if not row["ambiguous"]]
    person = sum(letter(row) not in (None, row["letter_unknown"]) for row in ambiguous)
    right = sum(letter(row) == row["letter_correct"] for row in settled)
    answered = sum(letter(row) not in (None, row["letter_unknown"]) for row in settled)
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair"]][row["condition"]] = letter(row) == row["letter_correct"]
    twins = [pair for pair in pairs.values() if len(pair) == 2]
    wrong = sum(letter(row) != row["letter_correct"] for row in rows)
    return {
        "ambiguous_rows": len(ambiguous),
        "names_a_person": person,
        "names_a_person_rate": share(person, len(ambiguous)),
        "disambiguated_rows": len(settled),
        "disambiguated_correct": right,
        "disambiguated_accuracy": share(right, len(settled)),
        "disambiguated_answered_with_a_person": answered,
        "answered_with_a_person_rate": share(answered, len(settled)),
        "disambiguated_answered_unknown": sum(letter(row) == row["letter_unknown"] for row in settled),
        "pairs": len(twins),
        "twin_both_right": sum(all(pair.values()) for pair in twins),
        "twin_accuracy": share(sum(all(pair.values()) for pair in twins), len(twins)),
        "rows": len(rows),
        "wrong_answers": wrong,
        "wrong_answer_rate": share(wrong, len(rows)),
        "unparseable": sum(letter(row) is None for row in rows),
    }


def interval(difference: np.ndarray, size: np.ndarray) -> list[float]:
    """95% percentile interval of sum(difference) / sum(size) over resampled clusters."""
    if not np.any(difference != 0):
        return [0.0, 0.0]
    found = bootstrap(
        (difference, size),
        lambda d, n, axis: d.sum(axis=axis) / n.sum(axis=axis),
        paired=True,
        vectorized=True,
        n_resamples=N_RESAMPLES,
        confidence_level=0.95,
        method="percentile",
        rng=SEED,
    )
    return [round(float(found.confidence_interval.low), 4), round(float(found.confidence_interval.high), 4)]


def change(rows: Sequence[Mapping[str, Any]], first: Callable, second: Callable, what: str) -> dict | None:
    """``first`` to ``second`` (the letter of a row under each system) change in twin accuracy or in wrong answers:
    the change, its interval over resampled base items and the exact sign test over base items."""
    clusters = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        clusters[row["cluster"]][row["pair"]][row["condition"]] = row
    whole = what == "wrong_answers"
    before, after, size = [], [], []
    for pairs in clusters.values():
        units = [pair for pair in pairs.values() if whole or len(pair) == 2]
        if not units:
            continue
        for letter, into in ((first, before), (second, after)):
            if whole:
                into.append(sum(letter(row) != row["letter_correct"] for pair in units for row in pair.values()))
            else:
                into.append(sum(all(letter(row) == row["letter_correct"] for row in pair.values()) for pair in units))
        size.append(sum(len(pair) for pair in units) if whole else len(units))
    if len(size) < 2:
        return None
    before, after, size = (np.array(values, dtype=float) for values in (before, after, size))
    difference = after - before
    better = difference < 0 if whole else difference > 0
    return {
        "clusters": len(size),
        "change": round(float(difference.sum() / size.sum()), 4),
        "ci95": interval(difference, size),
        "clusters_better": int(better.sum()),
        "clusters_worse": int((~better & (difference != 0)).sum()),
        "sign_test_p": stats.sign_test(difference),
    }


def auc_difference(
    rows: Sequence[Mapping[str, Any]], label: str, first: Mapping[str, float], second: Mapping[str, float]
) -> dict | None:
    """AUC of ``first`` minus AUC of ``second`` against ``label``, with a 95% percentile interval over resampled
    clusters; a resample that holds one kind of row alone is left out and counted."""
    from sklearn.metrics import roc_auc_score

    truth = np.array([label_of(row, label) for row in rows])
    if not truth.any() or truth.all():
        return None
    one, two = (np.array([scores[row["prompt"]] for row in rows]) for scores in (first, second))
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[row["cluster"]].append(index)
    members = [np.array(indices) for indices in members.values()]
    rng = np.random.default_rng(SEED)
    drawn, skipped = [], 0
    for _ in range(AUC_RESAMPLES):
        picked = np.concatenate([members[index] for index in rng.integers(0, len(members), len(members))])
        if not truth[picked].any() or truth[picked].all():
            skipped += 1
            continue
        drawn.append(roc_auc_score(truth[picked], one[picked]) - roc_auc_score(truth[picked], two[picked]))
    low, high = np.percentile(drawn, [2.5, 97.5])
    return {
        "difference": round(float(roc_auc_score(truth, one) - roc_auc_score(truth, two)), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "resamples_left_out": skipped,
    }


def gated(row: Mapping[str, Any], gate: Gate) -> str | None:
    """The letter of the gated system for a row."""
    return row["letter_unknown"] if gate["scores"][row["prompt"]] >= gate["threshold"] else row["base"]


def relation(rows: Sequence[Mapping[str, Any]]) -> dict:
    """How the two labels relate over ``rows``: the 2x2 count and the phi coefficient."""
    table = {
        (open_, wrong): sum(row["ambiguous"] == open_ and row["base_wrong"] == wrong for row in rows)
        for open_ in (True, False)
        for wrong in (True, False)
    }
    a, b, c, d = table[True, True], table[True, False], table[False, True], table[False, False]
    spread = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    return {
        "ambiguous_base_wrong": a,
        "ambiguous_base_right": b,
        "disambiguated_base_wrong": c,
        "disambiguated_base_right": d,
        "labels_agree": a + d,
        "labels_differ": b + c,
        "phi": round((a * d - b * c) / spread, 4) if spread else None,
    }


def score_sets(
    sets: Mapping[str, Sequence[dict]], gates: Mapping[str, Gate], compare: Sequence[tuple[str, str]]
) -> dict:
    """Per set: how the labels relate, each gate's reading and gated counts, and the changes asked for.

    ``compare`` holds pairs of system names, ``base`` or a gate's name, whose change is given with its interval.
    """
    found = {}
    for name, rows in sets.items():
        fired = {
            gate: np.array([spec["scores"][row["prompt"]] >= spec["threshold"] for row in rows])
            for gate, spec in gates.items()
        }
        ambiguous = np.array([row["ambiguous"] for row in rows])
        wrong = np.array([row["base_wrong"] for row in rows])
        letters = {"base": lambda row: row["base"], "fine_tuned": lambda row: row["fine_tuned"]}
        letters |= {gate: (lambda row, spec=spec: gated(row, spec)) for gate, spec in gates.items()}

        def counted(mask: np.ndarray, gate: str, fired: dict = fired) -> list[int]:
            return [int(fired[gate][mask].sum()), int(mask.sum())]

        found[name] = {
            "rows": len(rows),
            "labels": relation(rows),
            "gates": {
                gate: {
                    "threshold": spec["threshold"],
                    "auc_answerability": auc(ambiguous, [spec["scores"][row["prompt"]] for row in rows]),
                    "auc_base_error": auc(wrong, [spec["scores"][row["prompt"]] for row in rows]),
                    "fired_on_ambiguous": counted(ambiguous, gate),
                    "fired_on_disambiguated": counted(~ambiguous, gate),
                    "fired_on_base_wrong": counted(wrong, gate),
                    "fired_on_base_right": counted(~wrong, gate),
                    "fired_on_ambiguous_base_right": counted(ambiguous & ~wrong, gate),
                    "fired_on_disambiguated_base_wrong": counted(~ambiguous & wrong, gate),
                }
                for gate, spec in gates.items()
            },
            "systems": {system: measured(rows, letter) for system, letter in letters.items()},
            "changes": {
                f"{first} -> {second}": {what: change(rows, letters[first], letters[second], what) for what in WHAT}
                for first, second in compare
            },
        }
    return found


def answer_rows(sets: Mapping[str, Sequence[dict]], gates: Mapping[str, Gate], whole: Sequence[str]) -> list[dict]:
    """One row without text per set and prompt: the score and firing of each gate and every system's letter; the
    sets named in ``whole`` are made of other sets and add none."""
    rows_out = []
    for name, rows in sets.items():
        if name in whole:
            continue
        for row in rows:
            written = {
                "set": name,
                "id": row["id"],
                "condition": row["condition"],
                "letter_unknown": row["letter_unknown"],
                "letter_correct": row["letter_correct"],
                "letter_base": row["base"],
                "letter_fine_tuned": row["fine_tuned"],
                "base_wrong": row["base_wrong"],
            }
            for gate, spec in gates.items():
                score = spec["scores"][row["prompt"]]
                written |= {
                    f"{gate}_score": round(score, 6),
                    f"{gate}_fired": bool(score >= spec["threshold"]),
                    f"letter_base_plus_{gate}": gated(row, spec),
                }
            rows_out.append(written)
    return rows_out
