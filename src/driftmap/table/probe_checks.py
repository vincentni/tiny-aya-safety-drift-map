"""The records of the checks on the SQuAD probe as Markdown: served, scored again, answers by identity.

A record is what its step of ``attribution.probe_checks`` counted, under a schema name. The records hold counts,
rates and scores only; the one sentence in them is the fixed abstention sentence of the served gate.
"""

from collections.abc import Callable, Mapping
from typing import Any

from .breakdown import _table
from .probe_study import _number, _points, _share, _squad_line

SCHEMAS = {
    "squad-served": "driftmap.probe-squad-served/1",
    "squad-judge": "driftmap.probe-squad-scored/1",
    "squad-replication": "driftmap.probe-squad-scored/1",
    "squad-pooled": "driftmap.probe-squad-scored/1",
    "squad-identity": "driftmap.probe-squad-identity/1",
}
TITLES = {
    "squad-judge": "SQuAD 2.0: the probe's result under a judge's labels",
    "squad-replication": "SQuAD 2.0: the frozen probe on a second dev sample",
    "squad-pooled": "SQuAD 2.0: the frozen probe on both dev samples",
}
SERVED = {
    "prompts": "prompts",
    "served_equals_offline_gate": "served answer equals the gate computed afterwards",
    "firing_decision_equal": "same firing decision",
    "fired": "rows the probe fired on",
    "fired_rows_answering_the_abstention_sentence": "fired rows answering the abstention sentence, nothing else",
    "unfired_rows": "rows it did not fire on",
    "unfired_rows_with_the_stored_answer_text": "of them, with the text of the stored answer",
}
GAPS = {
    "probability": "served probability against the one computed afterwards",
    "hook_to_run_hidden_state": "the hook's state against the run's own hidden state",
    "run_state_to_stored_activation": "the run's state against the stored activation",
}
SYSTEMS = {
    "base": "base model",
    "offline_gate": "base model + probe, gate computed afterwards",
    "served_scored_by_the_rule": "base model + probe, served",
}
AGREEMENT = {
    "abstains": "the answer abstains",
    "gold_answer_on_answerable": "an answerable question's answer gives a gold answer",
    "answer_is_wrong": "the answer is wrong",
}
HEAD = ["system", "threshold", "wrong answers", "answered (coverage)", "wrong among answered (risk)"]
HEAD += ["unanswerable answered", "answerable still answered", "AUC, base model's error", "AUC, unanswerable"]


def render_served(record: Mapping[str, Any]) -> str:
    """The SQuAD probe served inside a running model against the gate computed afterwards."""
    counts, probe = record["counts"], record["probe"]
    lines = [
        [
            said,
            _share(found["wrong_answers"], found["rows"]),
            _share(found["answered"], found["rows"]),
            _share(found["wrong_among_answered"], found["answered"]),
            _share(found["hallucinations"], found["rows"] - found["answerable_rows"]),
            _share(found["answerable_answered"], found["answerable_rows"]),
            _share(found["answerable_correct"], found["answerable_rows"]),
        ]
        for name, said in SYSTEMS.items()
        for found in [record["on_the_sample"][name]]
    ]
    head = ["system", "wrong answers", "answered (coverage)", "wrong among answered (risk)"]
    head += ["unanswerable answered", "answerable still answered", "answerable answered with a gold answer"]
    out = [
        "# SQuAD 2.0: the probe served inside a running model",
        "",
        f"The frozen probe (layer {probe['layer']}, threshold {probe['threshold']:g}) is served inside the "
        "generation call: a forward hook on the block whose output is the probe's layer reads the last prompt "
        "token during the prompt pass, and for a row where the probe fires a logits processor forces the tokens of "
        f'one fixed sentence, "{record["abstention_sentence"]}" (classed by the reading check\'s rule as: '
        f"{record['abstention_sentence_wording_class']}), and then the end of the sequence, so nothing is generated "
        "freely for that row. A row where it does not fire generates as usual. The served answers are compared "
        "with the gate computed afterwards from the stored answers and the stored hidden states, and scored by "
        "the reading check's word-match rule.",
        "",
        "## Counts",
        "",
        *(f"- {said}: {counts.get(key, 0)}" for key, said in SERVED.items()),
        "",
        "## Largest absolute differences",
        "",
        *(f"- {GAPS.get(key, key)}: {value:.3g}" for key, value in record["largest_absolute_differences"].items()),
        "",
        "## The served prompts by the word-match rule",
        "",
        *_table(head, lines),
    ]
    return "\n".join(out)


def _reading(found: Mapping[str, Any]) -> list[str]:
    """The sample in one sentence, the base model beside the gated system, and the change."""
    sample, read = found["sample"], found["squad_probe"]
    lines = [_squad_line("base model", found["base"]), _squad_line("base model + probe", read["gated"], read)]
    moved = read["base_to_gated_wrong_answers"]
    return [
        f"{sample['rows']} questions, {sample['unanswerable']} of them unanswerable, over {sample['articles']} "
        f"articles. The base model alone: {found['base']['wrong_answers']} wrong answers, "
        f"{sample['hallucinations']} unanswerable questions answered, {sample['answerable_without_gold']} "
        "answerable questions without a gold answer.",
        "",
        *_table(HEAD, lines),
        f"Change in wrong answers, gated minus base: {_points(moved, 'articles', 'articles')}; "
        f"{moved['questions_wrong_to_right']} questions turned right, {moved['questions_right_to_wrong']} turned "
        "wrong.",
        "",
    ]


def _render_scored(part: str) -> Callable[[Mapping[str, Any]], str]:
    def render(record: Mapping[str, Any]) -> str:
        probe = record["probe"]
        out = [
            f"# {TITLES[part]}",
            "",
            f"The probe frozen on the train sample (layer {probe['layer']}, C {probe['C']:g}, threshold "
            f"{probe['threshold']:g}) is applied to the stored hidden states; nothing is fitted and the threshold "
            "is not moved. A gated system abstains where the probe fires, which is right on an unanswerable "
            "question and wrong on an answerable one, and gives the base model's stored answer elsewhere; the gate "
            "is computed on stored answers.",
            "",
            "## By the word-match rules",
            "",
            *_reading(record["word_match"]),
        ]
        judge = record.get("judge")
        if judge:
            out += [
                "## By the judge's labels",
                "",
                "A closed model read every answer with its passage, question and reference answers and said "
                "whether the answer abstains and whether it states a reference answer. It was not told which "
                "system wrote the answer, the word-match labels or the probe. The probe is unchanged: it was "
                f"fitted on word-match labels. {judge['rows_judged']} rows were judged, {judge['rows_unjudged']} "
                "were not.",
                "",
                *_reading(judge),
                "## The word-match rules against the judge",
                "",
            ]
            head = ["reading", "rows", "rule yes, judge yes", "rule yes, judge no", "rule no, judge yes"]
            head += ["rule no, judge no", "agree", "Cohen's kappa"]
            lines = [
                [
                    AGREEMENT.get(name, name),
                    str(found["rows"]),
                    *(str(found["rule_judge"][cell]) for cell in ("yes_yes", "yes_no", "no_yes", "no_no")),
                    _share(found["agree"], found["rows"]),
                    _number(found["kappa"]),
                ]
                for name, found in judge["agreement_of_the_rules_with_the_judge"].items()
            ]
            out += _table(head, lines)
            if judge["rows_unjudged"]:
                out += ["## By the word-match rules, on the judged rows alone", ""]
                out += _reading(record["word_match_on_the_judged_rows"])
        return "\n".join(out)

    return render


def render_identity(record: Mapping[str, Any]) -> str:
    """Two runs of the same prompts compared row for row."""
    keys = ("rows", "equal_text", "equal_token_count", "equal_stop_reason", "equal_on_all_three")
    head = ["answers", "equal text", "equal token count", "equal stop reason", "equal on all three"]
    out = [
        "# SQuAD 2.0: the scored answers against AuditKIT's",
        "",
        "The base model's answers on the dev sample were generated a second time through AuditKIT's evaluate, "
        "the path of the generate job, and compared row for row with the answers the probe was scored on, which "
        "came from one generation call that also returned the hidden states.",
        "",
        *_table(head, [[str(record[key]) for key in keys]]),
        f"AuditKIT rows with an error: {record['auditkit_rows_with_an_error']}.",
        "",
    ]
    return "\n".join(out)


RENDER: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "squad-served": render_served,
    "squad-identity": render_identity,
} | {part: _render_scored(part) for part in TITLES}
