"""The checks on the SQuAD probe as Markdown."""

from driftmap.table import probe_checks

COUNTS = {
    "rows": 100,
    "wrong_answers": 56,
    "answered": 100,
    "wrong_among_answered": 56,
    "hallucinations": 50,
    "answerable_rows": 50,
    "answerable_answered": 50,
    "answerable_correct": 44,
}
GATED = COUNTS | {"wrong_answers": 32, "answered": 53, "wrong_among_answered": 19, "hallucinations": 16}
GATED |= {"answerable_answered": 37, "answerable_correct": 34}
MOVED = {
    "articles": 35,
    "change": -0.274,
    "ci95": [-0.323, -0.228],
    "articles_better": 35,
    "articles_worse": 0,
    "sign_test_p": 0.00001,
    "questions_wrong_to_right": 345,
    "questions_right_to_wrong": 71,
}


def lines_of(text, title):
    return text.split(f"## {title}\n\n")[1].split("\n\n")[0].rstrip("\n").split("\n")


def reading(wrong):
    sample = {"rows": 100, "unanswerable": 50, "articles": 35, "hallucinations": 50, "answerable_without_gold": 6}
    read = {"threshold": 0.7, "auc_base_error": 0.8246, "auc_unanswerable": 0.8534, "gated": GATED}
    return {
        "sample": sample,
        "base": COUNTS | {"wrong_answers": wrong},
        "squad_probe": read | {"base_to_gated_wrong_answers": MOVED},
    }


def test_the_served_part_names_the_sentence_and_counts_each_agreement():
    record = {
        "probe": {"layer": 20, "threshold": 0.7},
        "abstention_sentence": "It cannot be determined from the text.",
        "abstention_sentence_wording_class": "trained sentence",
        "counts": {"prompts": 100, "served_equals_offline_gate": 100, "fired": 47, "unfired_rows": 53},
        "largest_absolute_differences": {"probability": 9.2e-07, "hook_to_run_hidden_state": 0.0},
        "on_the_sample": {"base": COUNTS, "offline_gate": GATED, "served_scored_by_the_rule": GATED},
    }

    said = probe_checks.render_served(record)

    assert said.startswith("# SQuAD 2.0: the probe served inside a running model\n")
    assert 'forces the tokens of one fixed sentence, "It cannot be determined from the text." (classed' in said
    assert lines_of(said, "Counts")[:4] == [
        "- prompts: 100",
        "- served answer equals the gate computed afterwards: 100",
        "- same firing decision: 0",
        "- rows the probe fired on: 47",
    ]
    assert lines_of(said, "Largest absolute differences") == [
        "- served probability against the one computed afterwards: 9.2e-07",
        "- the hook's state against the run's own hidden state: 0",
    ]
    systems = lines_of(said, "The served prompts by the word-match rule")
    assert systems[2] == (
        "| base model | 56.0% (56/100) | 100.0% (100/100) | 56.0% (56/100) | 100.0% (50/50) | 100.0% (50/50) "
        "| 88.0% (44/50) |"
    )
    assert systems[3].startswith("| base model + probe, gate computed afterwards | 32.0% (32/100) | 53.0% (53/100) |")
    assert systems[4] == systems[3].replace("gate computed afterwards", "served")


def test_a_scored_part_gives_the_rules_table_and_with_a_judge_the_judges_and_their_agreement():
    cells = {"yes_yes": 15, "yes_no": 3, "no_yes": 5, "no_no": 977}
    judge = {
        "rows_judged": 99,
        "rows_unjudged": 1,
        "agreement_of_the_rules_with_the_judge": {
            "abstains": {"rows": 1000, "rule_judge": cells, "agree": 992, "kappa": 0.7737}
        },
    }
    record = {
        "probe": {"layer": 20, "C": 0.001, "threshold": 0.7},
        "word_match": reading(56),
        "judge": judge | reading(50),
        "word_match_on_the_judged_rows": reading(55),
    }

    said = probe_checks.RENDER["squad-judge"](record)

    assert said.startswith("# SQuAD 2.0: the probe's result under a judge's labels\n")
    assert "(layer 20, C 0.001, threshold 0.7) is applied to the stored hidden states; nothing is fitted" in said
    by_rule = lines_of(said, "By the word-match rules")
    assert by_rule[0].startswith("100 questions, 50 of them unanswerable, over 35 articles. The base model alone: 56 ")
    table = said.split("## By the word-match rules\n\n")[1].split("\n\n")[1].split("\n")
    assert table[2].startswith("| base model | - | 56.0% (56/100) |")
    assert table[3] == (
        "| base model + probe | 0.7 | 32.0% (32/100) | 53.0% (53/100) | 35.8% (19/53) | 32.0% (16/50) "
        "| 74.0% (37/50) | 0.8246 | 0.8534 |"
    )
    assert (
        "Change in wrong answers, gated minus base: -27.4 points (95% CI -32.3 to -22.8; p < 0.001; 35 articles; "
        "better on 35, worse on 0); 345 questions turned right, 71 turned wrong." in said
    )
    assert "99 rows were judged, 1 were not." in said and "The base model alone: 50 wrong answers" in said
    assert lines_of(said, "The word-match rules against the judge")[2] == (
        "| the answer abstains | 1000 | 15 | 3 | 5 | 977 | 99.2% (992/1000) | 0.7737 |"
    )
    assert "## By the word-match rules, on the judged rows alone" in said and "alone: 55 wrong answers" in said
    whole = probe_checks.RENDER["squad-judge"](record | {"judge": judge | reading(50) | {"rows_unjudged": 0}})
    assert "on the judged rows alone" not in whole
    plain = probe_checks.RENDER["squad-replication"]({key: record[key] for key in ("probe", "word_match")})
    assert plain.startswith("# SQuAD 2.0: the frozen probe on a second dev sample\n") and "judge" not in plain
    assert set(probe_checks.RENDER) == set(probe_checks.SCHEMAS)


def test_a_fine_tune_part_says_which_fine_tune_it_is_of_and_on_which_sample():
    counts = COUNTS | {"wrong_answers": 6}
    moved = {key: MOVED[key] for key in MOVED if not key.startswith("questions_")} | {"questions": 100}
    found = {
        "base": COUNTS,
        "base_plus_squad_probe": GATED,
        "fine_tuned": counts,
        "fine_tuned_answers": {"wording": {"trained sentence": 30, "no abstention wording": 70}},
        "base_to_fine_tuned_wrong_answers": moved,
        "base_plus_squad_probe_to_fine_tuned_wrong_answers": moved,
    }
    record = {"probe": {"layer": 20, "C": 0.001, "threshold": 0.7}, "word_match": found}

    plain = probe_checks.RENDER["squad-fine-tune"](record)
    trained = probe_checks.RENDER["squad-trained-replication"](record)

    assert plain.startswith("# SQuAD 2.0: the fine-tune beside the base model and the probe\n")
    assert "The fine-tune answered the questions of the dev sample;" in plain
    assert trained.startswith(
        "# SQuAD 2.0: a fine-tune trained on the train sample, second dev sample beside the base model and the probe\n"
    )
    assert "one fixed abstention sentence, answered the questions of the second dev sample;" in trained
    assert "| fine-tune | 6.0% (6/100) |" in trained
    assert "The fine-tune's answers by wording: trained sentence 30, no abstention wording 70." in trained
    assert "- Change in wrong answers, fine-tune minus base model: -27.4 points" in trained and (
        "100 questions" in trained
    )


def test_the_identity_part_is_one_line_of_counts():
    record = {"rows": 1000, "equal_text": 1000, "equal_token_count": 1000, "equal_stop_reason": 999}
    record |= {"equal_on_all_three": 999, "auditkit_rows_with_an_error": 0}

    said = probe_checks.render_identity(record)

    assert "| 1000 | 1000 | 1000 | 999 | 999 |" in said and "AuditKIT rows with an error: 0." in said
