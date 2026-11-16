"""The checks on the frozen SQuAD probe, on a made-up sample: served, scored again, answers by identity."""

import json

import numpy as np
import pytest

from driftmap.attribution import probe, probe_checks, probe_gates
from driftmap.table import probe_checks as table
from tests.fakes.squad_probe import LAYERS, TRAINED, gated, put, read_jsonl, squad_sample, states_of

REASON = "the passage speaks of something else"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def sample(tmp_path):
    """A dev sample of 48 questions with answers, states, passing ids and a probe that reads the error unit."""
    rows = squad_sample("dev", 6, 8)
    put(tmp_path / "squad_dev_in.jsonl", rows)
    answers = [
        {"id": row["id"], "response": row["said"], "stop_reason": "eos", "generated_tokens": len(row["said"].split())}
        for row in rows
    ]
    put(tmp_path / "squad_dev_base.jsonl", answers)
    states_of(rows, 0.0, tmp_path / "acts" / "squad_dev")
    passed = gated(tmp_path / "squad_passed.json", {tmp_path / "squad_dev_in.jsonl": ("test", [])})
    arrays = (np.zeros(6), np.ones(6), np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), np.zeros(1))
    probe.save(probe.Probe(8, 0.001, 0.7, *(held.astype(np.float32) for held in arrays)), tmp_path / "probe_squad")
    asked = ["--dev", str(tmp_path / "squad_dev_in.jsonl"), "--dev-answers", str(tmp_path / "squad_dev_base.jsonl")]
    asked += ["--dev-states", str(tmp_path / "acts" / "squad_dev"), "--passed", str(passed)]
    return tmp_path, rows, [*asked, "--probe", str(tmp_path / "probe_squad")]


def hidden(rows):
    return {text for row in rows for text in (row["context"], row["question"], row["title"])} | {
        row["said"] for row in rows if row["said"] != TRAINED
    }


def test_the_served_sample_is_every_tenth_question_of_each_condition(sample):
    folder, rows, asked = sample

    assert probe_checks.main(["served-sample", *asked[:2], *asked[6:8], "--out", str(folder / "in.jsonl")]) == 0

    picked = read_jsonl(folder / "in.jsonl")
    kinds = {kind: [row for row in rows if row["condition"] == kind] for kind in ("unanswerable", "answerable")}
    assert [row["id"] for row in picked] == [kinds[kind][at]["id"] for kind in kinds for at in (0, 10, 20)]
    assert all(set(row) == {"id", "prompt"} for row in picked)
    assert picked[0]["prompt"] == rows[0]["prompt"]


def served_run(folder, rows, picked, change=None):
    """What the generation step would write for the sample: a wrong answer is gated, another row answers as stored."""
    by_id = {row["id"]: row for row in rows}
    live = []
    for row in picked:
        fired = by_id[row["id"]]["wrong"]
        written = {
            "id": row["id"],
            "response": TRAINED if fired else by_id[row["id"]]["said"],
            "fired": fired,
            "probe_probability": 1 / (1 + np.exp(-6.0 if fired else 6.0)),
            "hook_minus_hidden_state": 0.0,
        }
        live.append(written | (change or {}).get(row["id"], {}))
    put(folder / "served.jsonl", live)
    states, _ = probe.load_states(folder / "acts" / "squad_dev")
    kept = np.stack([states[row["id"]] + np.float32(0.5) for row in picked])
    (folder / "acts" / "served").mkdir(exist_ok=True)
    np.savez(folder / "acts" / "served" / "0.npz", ids=np.array([r["id"] for r in picked]), states=kept, layers=LAYERS)


def test_the_served_answers_are_compared_with_the_gate_computed_afterwards_and_scored_by_the_rule(sample):
    folder, rows, asked = sample
    probe_checks.main(["served-sample", *asked[:2], *asked[6:8], "--out", str(folder / "in.jsonl")])
    picked = read_jsonl(folder / "in.jsonl")
    served_run(folder, rows, picked)
    run = ["served", "--sample", str(folder / "in.jsonl"), "--served", str(folder / "served.jsonl")]
    run += ["--served-states", str(folder / "acts" / "served"), *asked, "--abstention", TRAINED]

    assert probe_checks.main([*run, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-served.json")
    assert record["schema"] == "driftmap.probe-squad-served/1" and record["probe"] == {"layer": 8, "threshold": 0.7}
    assert (record["abstention_sentence"], record["abstention_sentence_wording_class"]) == (TRAINED, "trained sentence")
    # of the six prompts two are unanswerable questions the base model answers: the probe fires on those two
    assert record["counts"] == {
        "prompts": 6,
        "served_equals_offline_gate": 6,
        "firing_decision_equal": 6,
        "fired": 2,
        "fired_rows_answering_the_abstention_sentence": 2,
        "unfired_rows": 4,
        "unfired_rows_with_the_stored_answer_text": 4,
    }
    gaps = record["largest_absolute_differences"]
    assert gaps["run_state_to_stored_activation"] == 0.5 and gaps["hook_to_run_hidden_state"] == 0.0
    assert gaps["probability"] < 1e-6
    found = record["on_the_sample"]
    assert (found["base"]["wrong_answers"], found["base"]["answered"], found["base"]["hallucinations"]) == (2, 5, 2)
    assert (found["offline_gate"]["wrong_answers"], found["offline_gate"]["answered"]) == (0, 3)
    assert found["served_scored_by_the_rule"] == found["offline_gate"]
    public = read_jsonl(folder / "results" / "probe-squad-served-answers.jsonl")
    assert public[0] == {
        "id": picked[0]["id"],
        "condition": "unanswerable",
        "article_index": 0,
        "probability_served": 0.997527,
        "probability_offline": 0.997527,
        "fired_served": True,
        "fired_offline": True,
        "served_answer_wording": "trained sentence",
        "base_answer_wording": "no abstention wording",
        "base_wrong": True,
    }
    said = (folder / "results" / "probe-squad-served.md").read_text(encoding="utf-8")
    assert said == table.RENDER["squad-served"](record)
    assert "- served answer equals the gate computed afterwards: 6" in said
    for path in (folder / "results").iterdir():
        assert not any(text in path.read_text(encoding="utf-8") for text in hidden(rows)), path.name

    # a fired row that is not gated in the run, and a silent row that answers something else
    change = {picked[0]["id"]: {"fired": False}, picked[3]["id"]: {"response": "Another answer."}}
    served_run(folder, rows, picked, change)
    assert probe_checks.main([*run, "--out", str(folder / "again")]) == 0
    counts = read_json(folder / "again" / "probe-squad-served.json")["counts"]
    assert (counts["served_equals_offline_gate"], counts["firing_decision_equal"], counts["fired"]) == (4, 5, 1)
    assert counts["unfired_rows_with_the_stored_answer_text"] == 3
    served_run(folder, rows, picked[:-1])
    with pytest.raises(SystemExit, match="the served run does not hold the sample's prompts"):
        probe_checks.main([*run, "--out", str(folder / "short")])


def verdicts_of(rows, change):
    """A verdict per row that follows the word-match labels but for the rows in ``change``; each has a reason."""
    held = []
    for row in rows:
        abstains = row["said"] == TRAINED
        correct = "n/a" if abstains else ("yes" if not row["open"] and not row["wrong"] else "no")
        verdict = {"id": row["id"], "status": "ok", "abstains": "yes" if abstains else "no", "correct": correct}
        held.append(verdict | {"why": REASON} | change.get(row["id"], {}))
    return held


def test_the_frozen_probe_is_scored_by_the_rules_and_again_by_the_judges_labels_on_the_judged_rows(sample):
    folder, rows, asked = sample
    # the judge takes a rule-right answer as wrong, a rule-wrong one as right, and leaves one row unjudged
    change = {
        rows[1]["id"]: {"correct": "no"},
        rows[3]["id"]: {"correct": "yes"},
        rows[5]["id"]: {"status": "unparsed", "abstains": None, "correct": None},
    }
    put(folder / "verdicts.jsonl", verdicts_of(rows, change))
    run = ["scored", "--part", "judge", *asked, "--verdicts", str(folder / "verdicts.jsonl")]

    assert probe_checks.main([*run, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-judge.json")
    assert list(record) == ["schema", "probe", "word_match", "judge", "word_match_on_the_judged_rows"]
    assert record["schema"] == "driftmap.probe-squad-scored/1"
    assert record["probe"] == {"layer": 8, "C": 0.001, "threshold": 0.7}
    by_rule = record["word_match"]
    assert (by_rule["sample"]["rows"], by_rule["sample"]["articles"], by_rule["base"]["wrong_answers"]) == (48, 6, 18)
    assert by_rule["squad_probe"]["fired_on_base_wrong"] == [18, 18]
    assert (by_rule["squad_probe"]["gated"]["wrong_answers"], by_rule["squad_probe"]["auc_base_error"]) == (6, 1.0)
    judge = record["judge"]
    assert (judge["rows_judged"], judge["rows_unjudged"]) == (47, 1)
    agreed = judge["agreement_of_the_rules_with_the_judge"]
    assert agreed["abstains"] == {
        "rows": 47,
        "rule_judge": {"yes_yes": 12, "yes_no": 0, "no_yes": 0, "no_no": 35},
        "agree": 47,
        "kappa": 1.0,
    }
    assert agreed["gold_answer_on_answerable"]["rule_judge"] == {"yes_yes": 16, "yes_no": 1, "no_yes": 1, "no_no": 5}
    assert agreed["answer_is_wrong"]["rule_judge"] == {"yes_yes": 17, "yes_no": 1, "no_yes": 1, "no_no": 28}
    assert (agreed["answer_is_wrong"]["agree"], agreed["gold_answer_on_answerable"]["rows"]) == (45, 23)
    # the probe was fitted on the rule's labels: under the judge's it misses one wrong answer and gates a right one
    assert judge["squad_probe"]["fired_on_base_wrong"] == [17, 18]
    assert judge["squad_probe"]["fired_on_base_right"] == [1, 29]
    assert judge["base"]["wrong_answers"] == 18 and judge["sample"]["rows"] == 47
    assert record["word_match_on_the_judged_rows"]["sample"]["rows"] == 47
    public = read_jsonl(folder / "results" / "probe-squad-judge-answers.jsonl")
    assert len(public) == 48 and public[1] == {
        "id": rows[1]["id"],
        "condition": "answerable",
        "article_index": 0,
        "rule_wording": "no abstention wording",
        "rule_holds_gold": True,
        "rule_wrong": False,
        "judge_abstains": False,
        "judge_says_gold": False,
        "judge_wrong": True,
        "squad_probe_score": 0.002473,
        "squad_probe_fired": False,
    }
    assert (public[5]["judge_abstains"], public[5]["judge_says_gold"], public[5]["judge_wrong"]) == (None, None, None)
    said = (folder / "results" / "probe-squad-judge.md").read_text(encoding="utf-8")
    assert said == table.RENDER["squad-judge"](record)
    assert "47 rows were judged, 1 were not." in said and "on the judged rows alone" in said
    for path in (folder / "results").iterdir():
        written = path.read_text(encoding="utf-8")
        assert REASON not in written and not any(text in written for text in hidden(rows)), path.name


def test_several_samples_are_scored_as_one_and_verdicts_are_for_every_sample_or_none(sample, capsys):
    folder, rows, asked = sample
    more = squad_sample("other", 3, 8)
    put(folder / "squad_dev2_in.jsonl", more)
    put(folder / "squad_dev2_base.jsonl", [{"id": r["id"], "response": r["said"], "stop_reason": "eos"} for r in more])
    states_of(more, 0.0, folder / "acts" / "squad_dev2")
    files = {folder / "squad_dev_in.jsonl": ("test", []), folder / "squad_dev2_in.jsonl": ("test", [more[0]["id"]])}
    gated(folder / "squad_passed.json", files)
    second = ["--dev", str(folder / "squad_dev2_in.jsonl"), "--dev-answers", str(folder / "squad_dev2_base.jsonl")]
    second += ["--dev-states", str(folder / "acts" / "squad_dev2")]

    assert probe_checks.main(["scored", "--part", "pooled", *asked, *second, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-pooled.json")
    assert list(record) == ["schema", "probe", "word_match"]
    # 48 questions and 23 of the second sample's 24: its first, an unanswerable one answered, did not pass the gate
    assert (record["word_match"]["sample"]["rows"], record["word_match"]["sample"]["articles"]) == (71, 9)
    assert record["word_match"]["sample"]["hallucinations"] == 12 + 5
    public = read_jsonl(folder / "results" / "probe-squad-pooled-answers.jsonl")
    assert len(public) == 71 and public[0]["judge_wrong"] is None and more[0]["id"] not in {r["id"] for r in public}
    said = (folder / "results" / "probe-squad-pooled.md").read_text(encoding="utf-8")
    assert said.startswith("# SQuAD 2.0: the frozen probe on both dev samples\n") and "judge" not in said
    put(folder / "verdicts.jsonl", verdicts_of(rows, {}))
    with pytest.raises(SystemExit):
        once = ["--verdicts", str(folder / "verdicts.jsonl"), "--out", str(folder / "x")]
        probe_checks.main(["scored", "--part", "pooled", *asked, *second, *once])
    assert "--verdicts is given for every sample or for none" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        probe_checks.main(["scored", "--part", "pooled", *asked, *second[:2], "--out", str(folder / "x")])
    assert "are given once per sample" in capsys.readouterr().err


def test_two_runs_of_the_same_prompts_are_compared_row_for_row(sample):
    folder, rows, _ = sample
    stored = read_jsonl(folder / "squad_dev_base.jsonl")
    kit = [dict(row) for row in stored]
    kit[0]["response"] += " More."
    kit[0]["generated_tokens"] += 1
    kit[1]["stop_reason"] = "length"
    kit[2]["error"] = "the backend gave up"
    put(folder / "kit.jsonl", kit[::-1])
    run = ["identity", "--stored", str(folder / "squad_dev_base.jsonl"), "--kit", str(folder / "kit.jsonl")]

    assert probe_checks.main([*run, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-identity.json")
    assert record == {
        "schema": "driftmap.probe-squad-identity/1",
        "rows": 48,
        "equal_text": 47,
        "equal_token_count": 47,
        "equal_stop_reason": 47,
        "equal_on_all_three": 46,
        "auditkit_rows_with_an_error": 1,
    }
    assert sorted(path.name for path in (folder / "results").iterdir()) == [
        "probe-squad-identity.json",
        "probe-squad-identity.md",
    ]
    assert "| 48 | 47 | 47 | 47 | 46 |" in (folder / "results" / "probe-squad-identity.md").read_text(encoding="utf-8")
    put(folder / "kit.jsonl", kit[:-1])
    with pytest.raises(SystemExit, match="47 answers to compare with 48 stored ones"):
        probe_checks.main([*run, "--out", str(folder / "short")])


def test_the_fine_tune_is_set_beside_the_base_model_and_the_gated_base_model_on_the_same_questions(sample, capsys):
    folder, rows, asked = sample
    # the fine-tune abstains on every unanswerable question and on one answerable question in eight
    tuned = []
    for at, row in enumerate(rows):
        abstains = row["open"] or at % 8 == 1
        said = TRAINED if abstains else f"It is {row['gold_answers'][0]}."
        tuned.append(dict(row, said=said, wrong=abstains and not row["open"]))
    put(folder / "squad_dev_ft.jsonl", [{"id": r["id"], "response": r["said"], "stop_reason": "eos"} for r in tuned])
    run = ["fine-tune", *asked, "--tuned-answers", str(folder / "squad_dev_ft.jsonl")]

    assert probe_checks.main([*run, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-fine-tune.json")
    assert list(record) == ["schema", "probe", "word_match"] and record["schema"] == "driftmap.probe-squad-fine-tune/1"
    found = record["word_match"]
    wrong = {name: found[name]["wrong_answers"] for name in ("base", "base_plus_squad_probe", "fine_tuned")}
    assert wrong == {"base": 18, "base_plus_squad_probe": 6, "fine_tuned": 6}
    assert (found["fine_tuned"]["answered"], found["fine_tuned"]["hallucinations"]) == (18, 0)
    assert found["fine_tuned_answers"]["wording"]["trained sentence"] == 30
    # per article the base model has three wrong answers, the gated one and the fine-tune one each
    assert found["base_to_fine_tuned_wrong_answers"] == {
        "questions": 48,
        "articles": 6,
        "change": -0.25,
        "ci95": [-0.25, -0.25],
        "articles_better": 6,
        "articles_worse": 0,
        "sign_test_p": 0.03125,
    }
    against_gated = found["base_plus_squad_probe_to_fine_tuned_wrong_answers"]
    assert (against_gated["change"], against_gated["ci95"], against_gated["articles_better"]) == (0.0, [0.0, 0.0], 0)
    public = read_jsonl(folder / "results" / "probe-squad-fine-tune-answers.jsonl")
    assert len(public) == 48 and public[1] == {
        "id": rows[1]["id"],
        "condition": "answerable",
        "article_index": 0,
        "rule_wording": "trained sentence",
        "rule_holds_gold": False,
        "rule_wrong": True,
        "judge_abstains": None,
        "judge_says_gold": None,
        "judge_wrong": None,
    }
    said = (folder / "results" / "probe-squad-fine-tune.md").read_text(encoding="utf-8")
    assert said == table.RENDER["squad-fine-tune"](record) and "By the judge's labels" not in said
    assert (
        "| fine-tune | 12.5% (6/48) | 37.5% (18/48) | 0.0% (0/18) | 0.0% (0/24) | 75.0% (18/24) | 75.0% (18/24) |"
        in said
    )
    assert "- Change in wrong answers, fine-tune minus base model: -25.0 points (95% CI -25.0 to -25.0;" in said

    # with verdicts on both models' answers the tables are given again on the rows judged in both
    put(folder / "base_verdicts.jsonl", verdicts_of(rows, {}))
    unjudged = {tuned[7]["id"]: {"status": "unparsed", "abstains": None, "correct": None}}
    put(folder / "ft_verdicts.jsonl", verdicts_of(tuned, unjudged))
    seen = ["--verdicts", str(folder / "base_verdicts.jsonl"), "--tuned-verdicts", str(folder / "ft_verdicts.jsonl")]
    assert probe_checks.main([*run, *seen, "--out", str(folder / "judged")]) == 0
    judge = read_json(folder / "judged" / "probe-squad-fine-tune.json")["judge"]
    assert judge["rows_judged_in_both_arms"] == 47
    assert judge["agreement_of_the_rules_with_the_judge_on_the_fine_tune"]["answer_is_wrong"]["agree"] == 47
    assert (judge["base"]["rows"], judge["fine_tuned"]["rows"], judge["fine_tuned"]["wrong_answers"]) == (47, 47, 6)
    assert judge["base_to_fine_tuned_wrong_answers"]["questions"] == 47
    rows_out = read_jsonl(folder / "judged" / "probe-squad-fine-tune-answers.jsonl")
    assert rows_out[1]["judge_wrong"] is True and rows_out[7]["judge_wrong"] is None
    for path in (folder / "judged").iterdir():
        written = path.read_text(encoding="utf-8")
        assert REASON not in written and not any(text in written for text in hidden(rows) | hidden(tuned)), path.name
    with pytest.raises(SystemExit):
        probe_checks.main([*run, seen[0], seen[1], "--out", str(folder / "x")])
    assert "--verdicts and --tuned-verdicts are given together or not at all" in capsys.readouterr().err


def job_rows(rows, split, change=None):
    """Response rows as a generate job writes them for the prompt group: the question under the group's id."""
    return [
        {
            "prompt_id": f"squad.{split}.{row['id'].removeprefix('squad.')}",
            "prompt": row["prompt"],
            "response": row["said"],
            "stop_reason": "eos",
            "generated_tokens": len(row["said"].split()),
            "status": "ok",
            "error": None,
        }
        | (change or {}).get(row["id"], {})
        for row in rows
    ]


def test_a_generate_jobs_answers_are_compared_with_the_stored_ones_split_by_split(sample, capsys):
    folder, rows, asked = sample
    # the job answers one unanswerable question the stored run abstained on, and words another answer differently
    change = {
        rows[2]["id"]: {"response": "It is thing 2.", "generated_tokens": 4},
        rows[1]["id"]: {"response": rows[1]["said"] + " Indeed.", "generated_tokens": 9, "stop_reason": "length"},
    }
    put(folder / "job_dev.jsonl", job_rows(rows, "dev", change)[::-1])
    put(folder / "job_same.jsonl", job_rows(rows, "dev"))
    run = ["job-identity", *asked[6:8]]
    for name, job in (("dev", "job_dev.jsonl"), ("same", "job_same.jsonl")):
        run += ["--name", name, "--asked", asked[1], "--stored", asked[3], "--job", str(folder / job)]

    assert probe_checks.main([*run, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-job-identity.json")
    assert record["schema"] == "driftmap.probe-squad-job-identity/1" and list(record["splits"]) == ["dev", "same"]
    assert record["splits"]["dev"] == {
        "rows": 48,
        "equal_text": 46,
        "equal_token_count": 46,
        "equal_stop_reason": 47,
        "equal_on_all_three": 46,
        "auditkit_rows_with_an_error": 0,
        "rows_whose_error_label_differs": 1,
        "rows_whose_abstention_differs": 1,
        "wrong_answers_stored": 18,
        "wrong_answers_job": 19,
    }
    same = record["splits"]["same"]
    assert (same["equal_on_all_three"], same["rows_whose_error_label_differs"], same["wrong_answers_job"]) == (
        48,
        0,
        18,
    )
    said = (folder / "results" / "probe-squad-job-identity.md").read_text(encoding="utf-8")
    assert "| dev | 48 | 46 | 46 | 47 | 46 | 1 | 1 | 18 | 19 | 0 |" in said
    assert not any(text in said for text in hidden(rows))
    with pytest.raises(SystemExit):
        probe_checks.main([*run, "--name", "more", "--out", str(folder / "x")])
    assert "are given once per split" in capsys.readouterr().err


def test_the_fine_tunes_job_answers_are_read_under_the_groups_ids_and_a_second_sample_has_its_own_part(sample):
    folder, rows, asked = sample
    tuned = [dict(row, said=TRAINED if row["open"] else row["said"]) for row in rows]
    put(folder / "job_ft.jsonl", job_rows(tuned, "dev"))
    run = ["fine-tune", *asked, "--tuned-answers", str(folder / "job_ft.jsonl")]

    assert probe_checks.main([*run, "--part", "fine-tune-replication", "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-fine-tune-replication.json")
    assert record["schema"] == "driftmap.probe-squad-fine-tune/1"
    assert record["word_match"]["fine_tuned"]["hallucinations"] == 0
    assert record["word_match"]["fine_tuned"]["wrong_answers"] == 6
    public = read_jsonl(folder / "results" / "probe-squad-fine-tune-replication-answers.jsonl")
    titles = sorted({row["title"] for row in rows})
    assert [row["article_index"] for row in public] == [titles.index(row["title"]) for row in rows]
    assert [row["id"] for row in public] == [row["id"] for row in rows]


def test_the_instructed_sample_adds_one_line_before_the_closing_line_and_its_answers_are_an_arm_of_their_own(sample):
    folder, rows, asked = sample

    assert probe_checks.main(["instructed-sample", *asked[:2], *asked[6:8], "--out", str(folder / "told.jsonl")]) == 0

    told = read_jsonl(folder / "told.jsonl")
    line = "If the text does not give enough information to answer, say that it cannot be determined."
    assert [row["id"] for row in told] == [row["id"] for row in rows] and told[0]["instruction"] == line
    assert told[0]["prompt"].split("\n") == [
        rows[0]["context"],
        rows[0]["question"],
        line,
        "Answer in one short sentence.",
    ]
    # told so, the base model abstains on every unanswerable question and on one answerable question in eight
    said = [TRAINED if row["open"] or at % 8 == 1 else row["said"] for at, row in enumerate(rows)]
    answers = [{"id": row["id"], "response": text, "stop_reason": "eos"} for row, text in zip(rows, said, strict=True)]
    put(folder / "told_answers.jsonl", answers)
    run = ["fine-tune", "--part", "prompt", *asked, "--tuned-answers", str(folder / "told_answers.jsonl")]
    put(folder / "base_verdicts.jsonl", verdicts_of(rows, {}))
    put(
        folder / "told_verdicts.jsonl",
        verdicts_of([dict(row, said=text) for row, text in zip(rows, said, strict=True)], {}),
    )
    seen = ["--verdicts", str(folder / "base_verdicts.jsonl"), "--tuned-verdicts", str(folder / "told_verdicts.jsonl")]

    assert probe_checks.main([*run, *seen, "--out", str(folder / "results")]) == 0

    record = read_json(folder / "results" / "probe-squad-prompt.json")
    assert record["schema"] == "driftmap.probe-squad-arm/1"
    assert list(record["word_match"]) == [
        "base",
        "base_plus_squad_probe",
        "instructed",
        "instructed_answers",
        "base_to_instructed_wrong_answers",
        "base_plus_squad_probe_to_instructed_wrong_answers",
    ]
    assert record["word_match"]["instructed"]["hallucinations"] == 0
    assert "agreement_of_the_rules_with_the_judge_on_instructed" in record["judge"]
    text = (folder / "results" / "probe-squad-prompt.md").read_text(encoding="utf-8")
    assert text == table.RENDER["squad-prompt"](record) and "| base model + instruction |" in text
    assert "## The word-match rules against the judge on its answers" in text


def test_the_models_confidence_is_scored_as_a_detector_beside_the_probe_without_a_fitted_threshold(sample, monkeypatch):
    monkeypatch.setattr(probe_gates, "AUC_RESAMPLES", 40)
    folder, rows, asked = sample
    # wrong answers are the least confident, right answers to answerable questions come next, abstentions last
    held = []
    for at, row in enumerate(rows):
        first = 0.1 if row["wrong"] else (0.9 if row["open"] else 0.5)
        entry = {"id": row["id"], "answer_tokens": 4, "greedy_tokens": 4 if at % 2 else 3}
        held.append(entry | {"first_token_probability": first, "mean_log_probability": -(at % 5 + 1) / 10})
    held[5] |= {"answer_tokens": 0, "greedy_tokens": 0, "first_token_probability": None, "mean_log_probability": None}
    put(folder / "confidence.jsonl", held[::-1])
    put(folder / "verdicts.jsonl", verdicts_of(rows, {}))
    run = ["confidence", *asked, "--confidence", str(folder / "confidence.jsonl")]

    assert (
        probe_checks.main([*run, "--verdicts", str(folder / "verdicts.jsonl"), "--out", str(folder / "results")]) == 0
    )

    record = read_json(folder / "results" / "probe-squad-confidence.json")
    assert record["schema"] == "driftmap.probe-squad-confidence/1"
    assert (record["rows"], record["rows_without_answer_text"]) == (48, 1)
    assert record["rows_whose_answer_tokens_are_all_the_argmax"] == 24 + 1 - 1
    assert record["answer_tokens_that_are_the_argmax"] == [24 * 3 + 23 * 4, 47 * 4]
    found = record["word_match"]
    assert list(found["detectors"]) == ["squad_probe", "first_token_probability", "mean_log_probability"]
    assert (found["base"]["wrong_answers"], found["squad_probe_gate"]["wrong_answers"]) == (18, 6)
    own, first = found["detectors"]["squad_probe"], found["detectors"]["first_token_probability"]
    assert own["auc_base_error"] == 1.0 and "squad_probe_minus_this" not in own
    assert "at_the_probe_gate_coverage" not in own
    # the first-token confidence ranks every wrong answer first: at the probe's coverage it is the probe's gate
    assert first["auc_base_error"] == 1.0 and list(first["squad_probe_minus_this"]) == ["base_error", "answerability"]
    assert first["squad_probe_minus_this"]["base_error"]["difference"] == 0.0
    assert first["at_the_probe_gate_coverage"] == {"abstained_on": 18} | found["squad_probe_gate"]
    shares = first["abstaining_on_the_most_suspicious_share"]
    assert [point["share"] for point in shares] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 19 of 48 abstained on: the 18 wrong answers and one right answer, which the abstention makes wrong
    assert (shares[3]["wrong_answers"], shares[3]["hallucinations"], shares[3]["answerable_correct"]) == (7, 0, 17)
    assert set(shares[3]) == {
        "share",
        "wrong_answers",
        "hallucinations",
        "answerable_answered",
        "answerable_correct",
        "coverage",
        "risk",
    }
    other = found["detectors"]["mean_log_probability"]
    assert other["auc_base_error"] < 0.8 and other["at_the_probe_gate_coverage"]["answered"] <= 18
    assert other["at_the_probe_gate_coverage"]["abstained_on"] > 18
    assert record["judge"]["rows_judged"] == 48 and record["judge"]["detectors"] == found["detectors"]
    public = read_jsonl(folder / "results" / "probe-squad-confidence-answers.jsonl")
    assert public[0] == {
        "id": rows[0]["id"],
        "condition": "unanswerable",
        "article_index": 0,
        "rule_wrong": True,
        "first_token_probability": 0.1,
        "mean_log_probability": -0.1,
        "answer_tokens": 4,
        "squad_probe_score": 0.997527,
    }
    assert public[5]["first_token_probability"] is None
    said = (folder / "results" / "probe-squad-confidence.md").read_text(encoding="utf-8")
    assert said == table.RENDER["squad-confidence"](record)
    assert "48 questions; 1 without answer text" in said and "## By the judge's labels" in said
    assert "| base model, abstaining on the lowest first-token probabilities | 18 | 12.5% (6/48) |" in said
    assert "| 40% | " in said
    for path in (folder / "results").iterdir():
        assert not any(text in path.read_text(encoding="utf-8") for text in hidden(rows)), path.name
    put(folder / "confidence.jsonl", held[:-1])
    with pytest.raises(SystemExit, match="47 confidence rows for 48 questions"):
        probe_checks.main([*run, "--out", str(folder / "short")])
