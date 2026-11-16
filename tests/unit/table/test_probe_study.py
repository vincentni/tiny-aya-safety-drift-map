"""The probe study's records as Markdown: what a reader is told of each part, and how the numbers are written."""

from driftmap.table import probe_study

COUNTS = {
    "rows": 8,
    "wrong_answers": 4,
    "pairs": 4,
    "twin_both_right": 1,
    "ambiguous_rows": 4,
    "names_a_person": 1,
    "disambiguated_rows": 4,
    "disambiguated_answered_with_a_person": 3,
}
MOVED = {
    "clusters": 4,
    "change": -0.125,
    "ci95": [-0.3333, 0.0],
    "clusters_better": 2,
    "clusters_worse": 1,
    "sign_test_p": 0.0004,
}
READ = {
    "threshold": 0.45,
    "auc_answerability": 0.6875,
    "auc_base_error": None,
    "fired_on_ambiguous": [2, 4],
    "fired_on_disambiguated": [1, 4],
    "fired_on_ambiguous_base_right": [0, 2],
    "fired_on_disambiguated_base_wrong": [0, 2],
}
RELATED = {
    "ambiguous_base_wrong": 3,
    "ambiguous_base_right": 1,
    "disambiguated_base_wrong": 1,
    "disambiguated_base_right": 3,
    "phi": 0.5,
}


def lines_of(text, title):
    """The lines of the section under ``title``."""
    return text.split(f"## {title}\n\n")[1].split("\n\n")[0].rstrip("\n").split("\n")


def test_numbers_are_written_as_shares_with_counts_points_with_intervals_and_scores_to_four_places():
    assert probe_study._share(18, 80) == "22.5% (18/80)" and probe_study._share(0, 0) == "n = 0"
    assert probe_study._number(0.5) == "0.5000" and probe_study._number(None) == "n/a"
    assert probe_study._points(MOVED) == (
        "-12.5 points (95% CI -33.3 to +0.0; p < 0.001; 4 base items; better on 2, worse on 1)"
    )
    assert probe_study._points(None) == "n/a"
    # a change that rounds to zero from below carries no minus sign
    assert probe_study._points(MOVED | {"change": -0.0004}).startswith("+0.0 points")
    assert probe_study._gap({"difference": 0.3269, "ci95": [0.2779, 0.3806]}) == "+0.3269 (95% CI +0.2779 to +0.3806)"
    assert probe_study._gap(None) == "n/a"
    assert probe_study._system("base") == "base model" and probe_study._system("fine_tuned") == "fine-tune"
    assert probe_study._system("error_probe") == "base model + error probe"
    assert probe_study._system("another") == "base model + another"
    assert set(probe_study.RENDER) == set(probe_study.SCHEMAS) == {"bbq", "baselines", "served", "squad"}
    assert all(name == f"driftmap.probe-{part}/1" for part, name in probe_study.SCHEMAS.items())


def bbq_record():
    systems = {
        "base": COUNTS,
        "fine_tuned": COUNTS | {"wrong_answers": 0},
        "probe_1": COUNTS | {"wrong_answers": 3},
        "error_probe": COUNTS | {"wrong_answers": 2, "pairs": 0, "twin_both_right": 0},
    }
    changes = {
        "base -> probe_1": {"twin_accuracy": None, "wrong_answers": MOVED},
        "probe_1 -> error_probe": {
            "twin_accuracy": MOVED | {"change": 0.25, "ci95": [0.0, 0.5]},
            "wrong_answers": None,
        },
    }
    entry = {"labels": RELATED, "gates": {"probe_1": READ, "error_probe": READ}, "systems": systems, "changes": changes}
    probes = {
        "probe_1": {"layer": 20, "C": 0.1, "threshold": 0.1},
        "error_probe": {"layer": 36, "C": 0.001, "threshold": 0.45},
    }
    return {"probes": probes, "sets": {"plain": entry}}


def test_the_bbq_part_gives_each_count_per_system_each_probes_reading_and_each_change():
    said = probe_study.render_bbq(bbq_record())

    assert said.startswith("# A probe of the base model's own error, beside the answerability probe\n")
    assert "computed on stored answers, not served through a hook" in said
    assert lines_of(said, "The probes") == [
        "- answerability probe: layer 20, C 0.1, threshold 0.1",
        "- error probe: layer 36, C 0.001, threshold 0.45",
    ]
    assert lines_of(said, "Wrong answers, of all rows") == [
        "| set | base model | base model + answerability probe | base model + error probe | fine-tune |",
        "| --- | --- | --- | --- | --- |",
        "| plain | 50.0% (4/8) | 37.5% (3/8) | 25.0% (2/8) | 0.0% (0/8) |",
    ]
    assert lines_of(said, "Twins both right")[2] == "| plain | 25.0% (1/4) | 25.0% (1/4) | n = 0 | 25.0% (1/4) |"
    assert lines_of(said, "A person is named on open rows")[2].startswith("| plain | 25.0% (1/4) |")
    assert lines_of(said, "Answerable twins still answered with a person")[2].startswith("| plain | 75.0% (3/4) |")
    assert lines_of(said, "How each probe reads each set")[2:] == [
        "| plain | answerability probe | 0.6875 | n/a | 2/4 | 1/4 | 0/2 | 0/2 |",
        "| plain | error probe | 0.6875 | n/a | 2/4 | 1/4 | 0/2 | 0/2 |",
    ]
    assert lines_of(said, "How the two labels relate")[2] == "| plain | 3 | 1 | 1 | 3 | 0.5000 |"
    first = lines_of(said, "Change, base model + answerability probe minus base model")
    assert first[0] == "| set | twins both right | wrong answers |"
    assert first[2] == f"| plain | n/a | {probe_study._points(MOVED)} |"
    second = lines_of(said, "Change, base model + error probe minus base model + answerability probe")
    assert second[2].startswith("| plain | +25.0 points (95% CI +0.0 to +50.0;") and second[2].endswith("| n/a |")


def test_the_baselines_part_sets_each_baseline_beside_the_probe_of_its_label():
    names = ["length_answerability", "length_base_error", "tfidf_answerability", "tfidf_base_error"]
    gates = {name: READ | {"auc_answerability": 0.5, "auc_base_error": 0.56} for name in names}
    gates |= {"probe_1": READ, "error_probe": READ | {"auc_base_error": 0.8246}}
    gap = {"difference": 0.1875, "ci95": [-0.01, 0.4], "resamples_left_out": 0}
    entry = {
        "gates": gates,
        "systems": {name: COUNTS for name in ("base", "probe_1", "error_probe", *names)},
        "auc_differences": {
            "answerability": {f"probe_1 minus {feature}_answerability": gap for feature in ("length", "tfidf")},
            "base_error": {"error_probe minus length_base_error": gap, "error_probe minus tfidf_base_error": None},
        },
    }
    picked = {"selected": {"C": 0.01, "dev_auc": 0.597}, "threshold": {"threshold": 0.5}}
    record = {
        "thresholds": dict.fromkeys(["probe_1", "error_probe", *names], 0.5),
        "sets": {"plain": entry},
        "fit": {"fit_rows": 900, "length_base_error": picked, "context_tokens": {"fit_ambiguous_median": 3.0}},
    }

    said = probe_study.render_baselines(record)

    assert said.startswith("# Text-only baselines beside the activation probes\n")
    open_ = lines_of(said, "AUC against open against settled rows")
    assert open_[0] == (
        "| set | context length, fitted on answerability | TF-IDF, fitted on answerability | answerability probe "
        "| probe minus length | probe minus TF-IDF |"
    )
    gap_said = "+0.1875 (95% CI -0.0100 to +0.4000)"
    assert open_[2] == f"| plain | 0.5000 | 0.5000 | 0.6875 | {gap_said} | {gap_said} |"
    error = lines_of(said, "AUC against the base model's own error")
    assert (
        error[0].split(" | ")[3] == "error probe"
        and error[2] == f"| plain | 0.5600 | 0.5600 | 0.8246 | {gap_said} | n/a |"
    )
    wrong = lines_of(said, "Wrong answers, of all rows")
    assert wrong[0].count("|") == 9 and "base model + TF-IDF, fitted on the base model's error" in wrong[0]
    assert lines_of(said, "The baselines as fitted") == [
        "| baseline | C | threshold | dev AUC |",
        "| --- | --- | --- | --- |",
        "| context length, fitted on the base model's error | 0.01 | 0.5 | 0.597 |",
    ]
    assert "The baselines as fitted" not in probe_study.render_baselines(
        {key: record[key] for key in ("thresholds", "sets")}
    )


def test_the_served_part_says_how_the_gate_is_served_and_counts_each_agreement():
    tally = {"prompts": 4, "served_equals_offline_gate_on_stored_answers": 3, "fired": 2, "unfired_rows": 2}
    record = {
        "probe": {"layer": 20, "threshold": 0.1},
        "all": tally,
        "by_set": {"first group": tally | {"prompts": 3}, "second group": {"prompts": 1}},
        "largest_absolute_differences": {"probability": 1.4e-05, "hook_to_run_hidden_state": 0.0, "other": 0.25},
    }

    said = probe_study.render_served(record)

    assert "(layer 20, threshold 0.1) is served inside the generation call: a forward hook" in said
    assert "forces the letter of the unknown option and then the end of the sequence" in said
    counts = lines_of(said, "Counts")
    assert counts[0] == "|  | all | first group | second group |"
    assert counts[2:5] == [
        "| prompts | 4 | 3 | 1 |",
        "| served answer equals the gate on stored answers | 3 | 3 | 0 |",
        "| served answer equals the gate on un-gated answers | 0 | 0 | 0 |",
    ]
    assert lines_of(said, "Largest absolute differences") == [
        "- served probability against the offline one: 1.4e-05",
        "- the hook's state against the run's own hidden state: 0",
        "- other: 0.25",
    ]


def squad_gate(wrong, answered, hallucinations, answerable):
    gated = {
        "rows": 1000,
        "wrong_answers": wrong,
        "answered": answered,
        "wrong_among_answered": wrong - 100 if answered else 0,
        "hallucinations": hallucinations,
        "answerable_rows": 500,
        "answerable_answered": answerable,
    }
    moved = {
        "articles": 35,
        "change": (wrong - 560) / 1000,
        "ci95": [-0.3226, -0.2281],
        "articles_better": 35,
        "articles_worse": 0,
        "sign_test_p": 0.00001,
        "questions_wrong_to_right": 345,
        "questions_right_to_wrong": 71,
    }
    point = {"wrong_answer_rate": 0.286, "coverage": 0.541, "risk": None, "answerable_answered_rate": 0.802}
    return {
        "threshold": 0.7,
        "auc_base_error": 0.8246,
        "auc_unanswerable": 0.8534,
        "gated": gated,
        "base_to_gated_wrong_answers": moved,
        "curve": [{"threshold": 0.05, "hallucinations": 0} | point, {"threshold": 0.7, "hallucinations": 140} | point],
    }


def test_the_squad_part_gives_risk_and_coverage_of_each_system_and_the_curve_of_the_in_domain_probe():
    base = squad_gate(560, 982, 485, 497)["gated"] | {"wrong_among_answered": 560}
    record = {
        "dev_sample": {"rows": 1000, "unanswerable": 500, "articles": 35, "hallucinations": 485}
        | {"answerable_without_gold": 75, "cut_at_the_token_limit": 20},
        "base": base,
        "zero_shot": {"probe_1": squad_gate(502, 867, 400, 467)},
        "in_domain": {"squad_probe": squad_gate(286, 541, 140, 401), "length": squad_gate(500, 0, 0, 0)},
        "in_domain_auc_differences": {"squad_probe minus length": {"difference": 0.3269, "ci95": [0.2779, 0.3806]}},
    }

    said = probe_study.render_squad(record)

    assert "1000 questions of the official SQuAD 2.0 dev split, 500 of them unanswerable, over 35 articles" in said
    assert "Both rules are word matches." in said and "the gate is computed on stored answers" in said
    assert "The base model alone: 560 wrong answers, 485 unanswerable questions answered, 75 answerable" in said
    systems = lines_of(said, "The base model and each gated system")
    assert systems[2] == (
        "| base model | - | 56.0% (560/1000) | 98.2% (982/1000) | 57.0% (560/982) | 97.0% (485/500) "
        "| 99.4% (497/500) | - | - |"
    )
    assert systems[3] == (
        "| base model + in-domain error probe, fitted on the SQuAD train sample | 0.7 | 28.6% (286/1000) "
        "| 54.1% (541/1000) | 34.4% (186/541) | 28.0% (140/500) | 80.2% (401/500) | 0.8246 | 0.8534 |"
    )
    assert systems[4].startswith(
        "| base model + context length, fitted on the SQuAD train sample | 0.7 | 50.0% (500/1000)"
    )
    assert "| 0.0% (0/1000) | n = 0 |" in systems[4]
    assert systems[5].startswith("| base model + answerability probe, fitted on the multiple-choice rows |")
    moved = lines_of(said, "Change in wrong answers, gated system minus base model")
    assert moved[2] == (
        "| base model + in-domain error probe, fitted on the SQuAD train sample | -27.4 points (95% CI -32.3 to "
        "-22.8; p < 0.001; 35 articles; better on 35, worse on 0) | 345 | 71 |"
    )
    gaps = lines_of(said, "The in-domain probe against the text baselines, AUC of the base model's error")
    assert gaps[2] == "| in-domain error probe minus length | +0.3269 (95% CI +0.2779 to +0.3806) |"
    curve = said.split("## Risk against coverage of the in-domain probe, by threshold\n\n")[1].split("\n")
    assert curve[0] == "The threshold reported above is the one chosen on the held-out articles of the train sample."
    assert curve[4:6] == ["| 0.05 | 54.1% | n/a | 28.6% | 80.2% | 0 |", "| 0.7 | 54.1% | n/a | 28.6% | 80.2% | 140 |"]
    assert "CuratorKIT" not in said
    counts = {"file": "squad_dev_in.jsonl", "rows_in": 1000, "dropped": {"row_shape": 0, "schema_gate": 2}}
    gated = probe_study.render_squad(record | {"curated": {"test": counts | {"rows_passed": 998}}})
    assert (
        "The rows fitted on and scored are those that pass the CuratorKIT gate: 998 of 1000 test questions "
        "(dropped: row shape 0, schema gate 2).\n" in gated
    )
    bare = probe_study.render_squad({key: record[key] for key in ("dev_sample", "base")})
    assert "Risk against coverage" not in bare and "| base model | - | 56.0% (560/1000) |" in bare
