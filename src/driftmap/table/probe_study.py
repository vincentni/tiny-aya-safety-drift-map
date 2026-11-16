"""The records of the probe study as Markdown, one per part.

A record is what its step of ``attribution.probe_study`` counted, under a schema name: the error probe beside the
answerability probe (``bbq``), the text-only baselines (``baselines``), the gate served inside a running model
(``served``) and SQuAD 2.0 (``squad``). The records hold counts, rates and scores only.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .breakdown import _p, _table

SCHEMAS = {
    "bbq": "driftmap.probe-bbq/1",
    "baselines": "driftmap.probe-baselines/1",
    "served": "driftmap.probe-served/1",
    "squad": "driftmap.probe-squad/1",
}
# how a gate is named to a reader; one not listed goes by its key
GATES = {
    "probe_1": "answerability probe",
    "error_probe": "error probe",
    "squad_probe": "in-domain error probe",
    "length": "context length",
    "tfidf": "TF-IDF",
    "length_answerability": "context length, fitted on answerability",
    "length_base_error": "context length, fitted on the base model's error",
    "tfidf_answerability": "TF-IDF, fitted on answerability",
    "tfidf_base_error": "TF-IDF, fitted on the base model's error",
}
SYSTEMS = {"base": "base model", "fine_tuned": "fine-tune"}
LABELS = {"answerability": "open against settled rows", "base_error": "the base model's own error"}
# metric of a set's counts, what it is counted over, and the title of its table
COUNTS = (
    ("wrong_answers", "rows", "Wrong answers, of all rows"),
    ("twin_both_right", "pairs", "Twins both right"),
    ("names_a_person", "ambiguous_rows", "A person is named on open rows"),
    ("disambiguated_answered_with_a_person", "disambiguated_rows", "Answerable twins still answered with a person"),
)
CHANGED = {"twin_accuracy": "twins both right", "wrong_answers": "wrong answers"}
GATE = (
    "The gate answers with the unknown option where a score reaches its threshold and with the base model's "
    "stored answer elsewhere; it is computed on stored answers, not served through a hook."
)


def _system(name: str) -> str:
    return SYSTEMS.get(name) or f"base model + {GATES.get(name, name)}"


def _share(k: int, n: int) -> str:
    return f"{k / n:.1%} ({k}/{n})" if n else "n = 0"


def _fired(pair: Sequence[int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _points(found: Mapping[str, Any] | None, units: str = "clusters", noun: str = "base items") -> str:
    if not found:
        return "n/a"
    # adding 0.0 turns a rounded -0.0 into 0.0
    moved, low, high = (round(value * 100, 1) + 0.0 for value in (found["change"], *found["ci95"]))
    span = f"95% CI {low:+.1f} to {high:+.1f}; {_p(found['sign_test_p'])}; {found[units]} {noun}"
    return f"{moved:+.1f} points ({span}; better on {found[f'{units}_better']}, worse on {found[f'{units}_worse']})"


def _gap(found: Mapping[str, Any] | None) -> str:
    if not found:
        return "n/a"
    return f"{found['difference']:+.4f} (95% CI {found['ci95'][0]:+.4f} to {found['ci95'][1]:+.4f})"


def _counts(sets: Mapping[str, Any], systems: Sequence[str]) -> list[str]:
    """A table per metric: a line per set, a column per system."""
    out = []
    for metric, over, title in COUNTS:
        lines = [
            [name, *(_share(entry["systems"][system][metric], entry["systems"][system][over]) for system in systems)]
            for name, entry in sets.items()
        ]
        out += [f"## {title}", "", *_table(["set", *map(_system, systems)], lines)]
    return out


def render_bbq(record: Mapping[str, Any]) -> str:
    """The error probe beside the answerability probe on the evaluation sets."""
    sets = record["sets"]
    gates = list(record["probes"])
    out = [
        "# A probe of the base model's own error, beside the answerability probe",
        "",
        "Two probes read the base model's hidden state at the last token of the prompt. The answerability probe "
        "was fitted on whether the text leaves the question open; the error probe on whether the base model's own "
        "answer to the prompt is wrong. Both were fitted on the same rows and chosen on the same held-out "
        f"templates. {GATE} A change is the second system minus the first, with a percentile 95% interval from "
        "resampling base items and an exact sign test over base items.",
        "",
        "## The probes",
        "",
        *(
            f"- {GATES.get(name, name)}: layer {found['layer']}, C {found['C']:g}, threshold {found['threshold']:g}"
            for name, found in record["probes"].items()
        ),
        "",
    ]
    out += _counts(sets, ["base", *gates, "fine_tuned"])
    head = ["set", "probe", "AUC, open against settled", "AUC, base model's error", "fires on open rows"]
    head += ["fires on settled rows", "fires on open rows the base model had right"]
    head += ["fires on settled rows the base model had wrong"]
    lines = [
        [
            name,
            GATES.get(gate, gate),
            _number(read["auc_answerability"]),
            _number(read["auc_base_error"]),
            _fired(read["fired_on_ambiguous"]),
            _fired(read["fired_on_disambiguated"]),
            _fired(read["fired_on_ambiguous_base_right"]),
            _fired(read["fired_on_disambiguated_base_wrong"]),
        ]
        for name, entry in sets.items()
        for gate, read in entry["gates"].items()
    ]
    out += ["## How each probe reads each set", "", *_table(head, lines)]
    head = ["set", "open, base wrong", "open, base right", "settled, base wrong", "settled, base right", "phi"]
    keys = ("ambiguous_base_wrong", "ambiguous_base_right", "disambiguated_base_wrong", "disambiguated_base_right")
    lines = [
        [name, *(str(entry["labels"][key]) for key in keys), _number(entry["labels"]["phi"])]
        for name, entry in sets.items()
    ]
    out += ["## How the two labels relate", "", *_table(head, lines)]
    for pair in next(iter(sets.values()))["changes"]:
        first, second = pair.split(" -> ")
        lines = [[name, *(_points(entry["changes"][pair][what]) for what in CHANGED)] for name, entry in sets.items()]
        out += [f"## Change, {_system(second)} minus {_system(first)}", "", *_table(["set", *CHANGED.values()], lines)]
    return "\n".join(out)


def render_baselines(record: Mapping[str, Any]) -> str:
    """The text-only baselines beside the two activation probes."""
    sets = record["sets"]
    out = [
        "# Text-only baselines beside the activation probes",
        "",
        "Do the hidden states carry anything the prompt's words do not? Two baselines read the prompt alone: the "
        "length of its context in tokens, and a TF-IDF bag of its 1- and 2-grams. Each was fitted for both labels "
        "on the rows, the split, the pick rule and the threshold rule of the activation probes. An AUC difference "
        "is the activation probe of the label minus the baseline, with a percentile 95% interval from resampling "
        f"base items. {GATE}",
        "",
    ]
    for label, said in LABELS.items():
        own = next(iter(next(iter(sets.values()))["auc_differences"][label])).split(" minus ")[0]
        names = [f"length_{label}", f"tfidf_{label}", own]
        head = ["set", *(GATES.get(name, name) for name in names), "probe minus length", "probe minus TF-IDF"]
        lines = []
        for name, entry in sets.items():
            aucs = [_number(entry["gates"][gate][f"auc_{label}"]) for gate in names]
            gaps = [_gap(found) for found in entry["auc_differences"][label].values()]
            lines.append([name, *aucs, *gaps])
        out += [f"## AUC against {said}", "", *_table(head, lines)]
    out += _counts(sets, ["base", *record["thresholds"]])
    fitted = record.get("fit") or {}
    lines = [
        [
            GATES.get(name, name),
            f"{found['selected']['C']:g}",
            f"{found['threshold']['threshold']:g}",
            f"{found['selected']['dev_auc']:.3f}",
        ]
        for name, found in fitted.items()
        if isinstance(found, Mapping) and "selected" in found
    ]
    if lines:
        out += ["## The baselines as fitted", "", *_table(["baseline", "C", "threshold", "dev AUC"], lines)]
    return "\n".join(out)


def render_served(record: Mapping[str, Any]) -> str:
    """The gate served inside a running model against the gate computed afterwards."""
    keys = {
        "prompts": "prompts",
        "served_equals_offline_gate_on_stored_answers": "served answer equals the gate on stored answers",
        "served_equals_offline_gate_on_ungated_cpu_answers": "served answer equals the gate on un-gated answers",
        "firing_decision_equal": "same firing decision",
        "fired": "rows the probe fired on",
        "fired_rows_answering_the_unknown_letter_alone": "fired rows answering the unknown letter alone",
        "unfired_rows": "rows it did not fire on",
        "unfired_rows_with_the_ungated_text": "of them, with the text of the un-gated run",
        "ungated_cpu_letter_equals_stored_letter": "un-gated letter equals the stored letter",
        "ungated_cpu_text_equals_stored_text": "un-gated text equals the stored text",
    }
    groups = {"all": record["all"], **record["by_set"]}
    lines = [[said, *(str(found.get(key, 0)) for found in groups.values())] for key, said in keys.items()]
    gaps = record["largest_absolute_differences"]
    said = {
        "probability": "served probability against the offline one",
        "hook_to_run_hidden_state": "the hook's state against the run's own hidden state",
        "run_state_to_stored_activation": "the run's state against the stored activation",
    }
    out = [
        "# The gate served inside a running model",
        "",
        f"The frozen probe (layer {record['probe']['layer']}, threshold {record['probe']['threshold']:g}) is served "
        "inside the generation call: a forward hook on the block whose output is the probe's layer reads the last "
        "prompt token during the prompt pass, and for a row where the probe fires a logits processor forces the "
        "letter of the unknown option and then the end of the sequence, so nothing is generated freely for that "
        "row. Every prompt of the sample was run with the gate and without it, and compared with the gate computed "
        "afterwards from stored answers and stored hidden states.",
        "",
        "## Counts",
        "",
        *_table(["", *groups], lines),
        "## Largest absolute differences",
        "",
        *(f"- {said.get(key, key)}: {value:.3g}" for key, value in gaps.items()),
        "",
    ]
    return "\n".join(out)


def _squad_line(name: str, found: Mapping[str, Any], read: Mapping[str, Any] | None = None) -> list[str]:
    answerable = _share(found["answerable_answered"], found["answerable_rows"])
    return [
        name,
        "-" if read is None else f"{read['threshold']:g}",
        _share(found["wrong_answers"], found["rows"]),
        _share(found["answered"], found["rows"]),
        _share(found["wrong_among_answered"], found["answered"]),
        _share(found["hallucinations"], found["rows"] - found["answerable_rows"]),
        answerable,
        "-" if read is None else _number(read["auc_base_error"]),
        "-" if read is None else _number(read["auc_unanswerable"]),
    ]


def _curated(gates: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What the CuratorKIT gate let through, per sample, and what each of its checks dropped."""
    if not gates:
        return []
    said = []
    for role, entry in gates.items():
        dropped = ", ".join(f"{check.replace('_', ' ')} {count}" for check, count in entry["dropped"].items())
        said.append(f"{entry['rows_passed']} of {entry['rows_in']} {role} questions (dropped: {dropped})")
    return [f"The rows fitted on and scored are those that pass the CuratorKIT gate: {'; '.join(said)}.", ""]


def render_squad(record: Mapping[str, Any]) -> str:
    """SQuAD 2.0: the base model, the in-domain error probe beside the text baselines, and the earlier probes
    read zero-shot."""
    sample, base = record["dev_sample"], record["base"]
    modes = {"in_domain": "fitted on the SQuAD train sample", "zero_shot": "fitted on the multiple-choice rows"}
    out = [
        "# SQuAD 2.0: a probe of the base model's own error on free-form answers",
        "",
        f"{sample['rows']} questions of the official SQuAD 2.0 dev split, {sample['unanswerable']} of them "
        f"unanswerable, over {sample['articles']} articles, each asked with its passage for one short sentence. "
        "The base model's answer is classed by rule: it abstains when its wording is an abstention; an answerable "
        "question is answered correctly when the answer contains a gold answer. Both rules are word matches. A "
        "wrong answer is an unanswerable question answered, or an answerable one whose answer holds no gold "
        "answer. A gated system abstains where its gate fires, which is right on an unanswerable question and "
        "wrong on an answerable one, and gives the base model's stored answer elsewhere; the gate is computed on "
        "stored answers. The in-domain probe was fitted on the base model's own errors on a sample of the official "
        "train split, chosen on held-out articles of it, frozen, and scored once on the dev questions.",
        "",
        f"The base model alone: {base['wrong_answers']} wrong answers, {sample['hallucinations']} unanswerable "
        f"questions answered, {sample['answerable_without_gold']} answerable questions without a gold answer; "
        f"{sample['cut_at_the_token_limit']} answers were cut at the token limit.",
        "",
        *_curated(record.get("curated") or {}),
    ]
    head = ["system", "threshold", "wrong answers", "answered (coverage)", "wrong among answered (risk)"]
    head += ["unanswerable answered", "answerable still answered", "AUC, base model's error", "AUC, unanswerable"]
    lines = [_squad_line(SYSTEMS["base"], base)]
    moved = []
    for mode, said in modes.items():
        for name, read in (record.get(mode) or {}).items():
            shown = f"{_system(name)}, {said}"
            lines.append(_squad_line(shown, read["gated"], read))
            found = read["base_to_gated_wrong_answers"]
            turned = [str(found["questions_wrong_to_right"]), str(found["questions_right_to_wrong"])]
            moved.append([shown, _points(found, "articles", "articles"), *turned])
    out += ["## The base model and each gated system", "", *_table(head, lines)]
    head = ["system", "change in wrong answers", "questions turned right", "questions turned wrong"]
    out += ["## Change in wrong answers, gated system minus base model", "", *_table(head, moved)]
    gaps = record.get("in_domain_auc_differences") or {}
    if gaps:
        lines = [[name.replace("squad_probe", GATES["squad_probe"]), _gap(found)] for name, found in gaps.items()]
        out += ["## The in-domain probe against the text baselines, AUC of the base model's error", ""]
        out += _table(["difference", "AUC difference"], lines)
    curve = (record.get("in_domain") or {}).get("squad_probe", {}).get("curve") or []
    if curve:
        head = ["threshold", "coverage", "risk", "wrong-answer rate", "answerable still answered"]
        head += ["unanswerable answered"]
        lines = [
            [
                f"{point['threshold']:g}",
                *(
                    "n/a" if point[key] is None else f"{point[key]:.1%}"
                    for key in ("coverage", "risk", "wrong_answer_rate", "answerable_answered_rate")
                ),
                str(point["hallucinations"]),
            ]
            for point in curve
        ]
        out += ["## Risk against coverage of the in-domain probe, by threshold", ""]
        out += ["The threshold reported above is the one chosen on the held-out articles of the train sample.", ""]
        out += _table(head, lines)
    return "\n".join(out)


RENDER: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "bbq": render_bbq,
    "baselines": render_baselines,
    "served": render_served,
    "squad": render_squad,
}
