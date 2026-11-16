"""The records of the checks on the SQuAD probe as Markdown: served, scored again, answers by identity.

A record is what its step of ``attribution.probe_checks`` counted, under a schema name. The records hold counts,
rates and scores only; the one sentence in them is the fixed abstention sentence of the served gate.
"""

from collections.abc import Callable, Mapping
from typing import Any

from .breakdown import _table
from .probe_study import _gap, _number, _points, _share, _squad_line

SCHEMAS = {
    "squad-served": "driftmap.probe-squad-served/1",
    "squad-judge": "driftmap.probe-squad-scored/1",
    "squad-replication": "driftmap.probe-squad-scored/1",
    "squad-pooled": "driftmap.probe-squad-scored/1",
    "squad-fine-tune": "driftmap.probe-squad-fine-tune/1",
    "squad-fine-tune-replication": "driftmap.probe-squad-fine-tune/1",
    "squad-trained": "driftmap.probe-squad-fine-tune/1",
    "squad-trained-replication": "driftmap.probe-squad-fine-tune/1",
    "squad-prompt": "driftmap.probe-squad-arm/1",
    "squad-confidence": "driftmap.probe-squad-confidence/1",
    "squad-job-identity": "driftmap.probe-squad-job-identity/1",
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


def _agreement(agreed: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """A line per reading: the two-by-two count of the rule against the judge, their agreement and kappa."""
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
        for name, found in agreed.items()
    ]
    return _table(head, lines)


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
            out += _agreement(judge["agreement_of_the_rules_with_the_judge"])
            if judge["rows_unjudged"]:
                out += ["## By the word-match rules, on the judged rows alone", ""]
                out += _reading(record["word_match_on_the_judged_rows"])
        return "\n".join(out)

    return render


def _arms(found: Mapping[str, Any], label: str, system: str) -> list[str]:
    """The three systems on the same questions, and the change of the arm ``label`` against the other two."""
    names = {"base": "base model", "base_plus_squad_probe": "base model + probe", label: system}
    head = ["system", "wrong answers", "answered (coverage)", "wrong among answered (risk)"]
    head += ["unanswerable answered", "answerable still answered", "answerable answered with a gold answer"]
    lines = [
        [
            said,
            _share(counts["wrong_answers"], counts["rows"]),
            _share(counts["answered"], counts["rows"]),
            _share(counts["wrong_among_answered"], counts["answered"]),
            _share(counts["hallucinations"], counts["rows"] - counts["answerable_rows"]),
            _share(counts["answerable_answered"], counts["answerable_rows"]),
            _share(counts["answerable_correct"], counts["answerable_rows"]),
        ]
        for name, said in names.items()
        for counts in [found[name]]
    ]
    wording = ", ".join(f"{kind} {count}" for kind, count in found[f"{label}_answers"]["wording"].items())
    moves = {
        f"base_to_{label}_wrong_answers": f"{system} minus base model",
        f"base_plus_squad_probe_to_{label}_wrong_answers": f"{system} minus base model + probe",
    }
    return [
        *_table(head, lines),
        f"Its answers by wording: {wording}.",
        "",
        *(
            f"- Change in wrong answers, {said}: {_points(found[key], 'articles', 'articles')}, "
            f"{found[key]['questions']} questions"
            for key, said in moves.items()
        ),
        "",
    ]


TRAINED = (
    "A fine-tune trained on the questions of the train sample, to answer an answerable one with a gold answer and "
    "an unanswerable one with one fixed abstention sentence,"
)
INSTRUCTED = (
    "The base model, asked with one instruction line added before the closing line, to say that it cannot be "
    "determined when the text does not give enough information to answer,"
)
# per part: the title, who answered, on which sample, the arm's name in the record and in the tables
TUNED = {
    "squad-fine-tune": ("the fine-tune", "The fine-tune", "the dev sample", "fine_tuned", "fine-tune"),
    "squad-fine-tune-replication": (
        "the fine-tune, second dev sample",
        "The fine-tune",
        "the second dev sample",
        "fine_tuned",
        "fine-tune",
    ),
    "squad-trained": ("a fine-tune trained on the train sample", TRAINED, "the dev sample", "fine_tuned", "fine-tune"),
    "squad-trained-replication": (
        "a fine-tune trained on the train sample, second dev sample",
        TRAINED,
        "the second dev sample",
        "fine_tuned",
        "fine-tune",
    ),
    "squad-prompt": (
        "an instruction in the prompt",
        INSTRUCTED,
        "the dev sample",
        "instructed",
        "base model + instruction",
    ),
}


def render_fine_tune(record: Mapping[str, Any], part: str = "squad-fine-tune") -> str:
    """The arm of ``part``, a fine-tune or the instructed base model, beside the base model and the base model
    gated by the probe."""
    probe = record["probe"]
    title, who, sample, label, system = TUNED[part]
    out = [
        f"# SQuAD 2.0: {title} beside the base model and the probe",
        "",
        f"{who} answered the questions of {sample}; its answers are classed by the same rules as the "
        f"base model's. The probe (layer {probe['layer']}, C {probe['C']:g}, threshold {probe['threshold']:g}) "
        "is the one frozen on the train sample and reads the base model alone; its gate is computed on stored "
        "answers. A change is that system's wrong-answer rate minus the other system's on the same questions, "
        "with a percentile 95% interval from resampling articles and an exact sign test over articles.",
        "",
        "## By the word-match rules",
        "",
        *_arms(record["word_match"], label, system),
    ]
    judge = record.get("judge")
    if judge:
        agreed = next(found for key, found in judge.items() if key.startswith("agreement_of_the_rules_with_the_judge"))
        out += [
            "## By the judge's labels",
            "",
            f"On the {judge['rows_judged_in_both_arms']} questions on which the judge gave a verdict for both "
            "systems' answers.",
            "",
            *_arms(judge, label, system),
            "## The word-match rules against the judge on its answers",
            "",
            *_agreement(agreed),
        ]
    return "\n".join(out)


DETECTORS = {
    "squad_probe": "probe",
    "first_token_probability": "1 - probability of the answer's first token",
    "mean_log_probability": "minus the mean log-probability of the answer's tokens",
}


GATES = {
    "first_token_probability": "base model, abstaining on the lowest first-token probabilities",
    "mean_log_probability": "base model, abstaining on the lowest mean log-probabilities",
}


def _detectors(found: Mapping[str, Any]) -> list[str]:
    """The detectors under one labelling: AUCs, the gate at the probe's coverage, and the gate share by share."""
    base, gate = found["base"], found["squad_probe_gate"]

    def counts(said: str, held: Mapping[str, Any], abstained: str) -> list[str]:
        return [
            said,
            abstained,
            _share(held["wrong_answers"], held["rows"]),
            _share(held["answered"], held["rows"]),
            _share(held["wrong_among_answered"], held["answered"]),
            _share(held["hallucinations"], held["rows"] - held["answerable_rows"]),
            _share(held["answerable_answered"], held["answerable_rows"]),
            _share(held["answerable_correct"], held["answerable_rows"]),
        ]

    head = ["detector", "AUC, base model's error", "AUC, unanswerable", "probe minus this, base model's error"]
    head += ["probe minus this, unanswerable"]
    aucs = [
        [
            DETECTORS.get(name, name),
            _number(entry["auc_base_error"]),
            _number(entry["auc_unanswerable"]),
            *(
                _gap(entry["squad_probe_minus_this"][label]) if "squad_probe_minus_this" in entry else "-"
                for label in ("base_error", "answerability")
            ),
        ]
        for name, entry in found["detectors"].items()
    ]
    gates = ["system", "abstained on", "wrong answers", "answered (coverage)", "wrong among answered (risk)"]
    gates += ["unanswerable answered", "answerable still answered", "answerable answered with a gold answer"]
    lines = [counts("base model", base, "-"), counts("base model + probe, at its threshold", gate, "-")]
    lines += [
        counts(GATES.get(name, name), held, str(held["abstained_on"]))
        for name, entry in found["detectors"].items()
        if "at_the_probe_gate_coverage" in entry
        for held in [entry["at_the_probe_gate_coverage"]]
    ]
    names = list(found["detectors"])
    steps = [point["share"] for point in found["detectors"][names[0]]["abstaining_on_the_most_suspicious_share"]]
    shares = [
        [
            f"{share:.0%}",
            *(
                f"{point['wrong_answers']} ({'n/a' if point['risk'] is None else format(point['risk'], '.1%')})"
                for name in names
                for point in [found["detectors"][name]["abstaining_on_the_most_suspicious_share"][at]]
            ),
        ]
        for at, share in enumerate(steps)
    ]
    return [
        *_table(head, aucs),
        "Each confidence gate abstains on its least confident answers until it answers no more questions than the "
        "probe's gate:",
        "",
        *_table(gates, lines),
        "Wrong answers (risk among the answered) when a detector abstains on its most suspicious share of answers:",
        "",
        *_table(["share abstained on", *(DETECTORS.get(name, name) for name in names)], shares),
    ]


def render_confidence(record: Mapping[str, Any]) -> str:
    """The model's own confidence in its answers as a detector of its errors, beside the probe."""
    tokens = record["answer_tokens_that_are_the_argmax"]
    out = [
        "# SQuAD 2.0: the model's own confidence beside the probe",
        "",
        "Does the base model's output confidence do what the probe does? One forward pass per question over the "
        "prompt followed by the stored answer gives the probability of the answer's first token and the mean "
        "log-probability of its tokens. Low confidence is taken as the sign of an error. No threshold is fitted "
        "for the confidence scores, since none exists for the train sample. The probe is the one frozen on the "
        "train sample, at its own threshold; every gate is computed on stored answers.",
        "",
        f"{record['rows']} questions; {record['rows_without_answer_text']} without answer text, ranked as the "
        f"least suspicious. {_share(tokens[0], tokens[1])} of the answers' tokens are the pass's most likely "
        f"token, on {record['rows_whose_answer_tokens_are_all_the_argmax']} answers every one.",
        "",
        "## By the word-match rules",
        "",
        *_detectors(record["word_match"]),
    ]
    if record.get("judge"):
        out += [
            "## By the judge's labels",
            "",
            f"On the {record['judge']['rows_judged']} questions the judge gave a verdict on.",
            "",
            *_detectors(record["judge"]),
        ]
    return "\n".join(out)


def render_job_identity(record: Mapping[str, Any]) -> str:
    """The answers of a generate job against the stored ones, split by split."""
    keys = ("rows", "equal_text", "equal_token_count", "equal_stop_reason", "equal_on_all_three")
    more = ("rows_whose_error_label_differs", "rows_whose_abstention_differs", "wrong_answers_stored")
    head = ["split", "answers", "equal text", "equal token count", "equal stop reason", "equal on all three"]
    head += ["error label differs", "abstention differs", "wrong answers, stored", "wrong answers, job", "job errors"]
    lines = [
        [name, *(str(found[key]) for key in (*keys, *more, "wrong_answers_job", "auditkit_rows_with_an_error"))]
        for name, found in record["splits"].items()
    ]
    out = [
        "# SQuAD 2.0: the generate job's answers against the stored ones",
        "",
        "The base model answered the questions of each split a second time in a generate job, through AuditKIT's "
        "evaluate on a GPU. Its answers are compared row for row with the stored answers the probe was fitted and "
        "scored on, which came from a CPU run that also returned the hidden states; both are classed by the "
        "word-match rules, and a label differs where the two runs' answers are classed differently.",
        "",
        *_table(head, lines),
    ]
    return "\n".join(out)


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
    **{part: (lambda record, part=part: render_fine_tune(record, part)) for part in TUNED},
    "squad-confidence": render_confidence,
    "squad-job-identity": render_job_identity,
    "squad-identity": render_identity,
} | {part: _render_scored(part) for part in TITLES}
