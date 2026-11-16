"""The probe study's steps on small stored files: what each reads, counts and writes, and that no text is written."""

import json

import numpy as np
import pytest

from driftmap.attribution import probe, probe_fit, probe_gates, probe_study
from driftmap.table import probe_study as table
from tests.fakes.squad_probe import TRAINED, Words, gated, put, read_jsonl, squad_sample, states_of

OPTIONS = [{"letter": "A", "text": "the first person"}, {"letter": "B", "text": "the second person"}]
OPTIONS.append({"letter": "C", "text": "it cannot be told"})
CLOSE = ["Who was it?", *(f"{option['letter']}. {option['text']}" for option in OPTIONS), "Answer with one letter."]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def choice_row(name, item, open_, wrong):
    """A prompt row of a pair and the base model's answer to it; an open prompt has the shorter context."""
    context = f"somebody left {item}" if open_ else f"it was plain because the first one left {item} there"
    correct = "C" if open_ else "A"
    said = ("A" if open_ else "B") if wrong else correct
    return {
        "id": name,
        "prompt": "\n".join([context, *CLOSE]),
        "condition": "ambiguous" if open_ else "disambiguated",
        "options": OPTIONS,
        "letter_unknown": "C",
        "letter_correct": correct,
        "base_item": item,
        "language": "en",
        "open": open_,
        "wrong": wrong,
        "said": said,
    }


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """Training rows with answers and states, both probes fitted, and one evaluation set of forty pairs."""
    monkeypatch.setattr(probe_gates, "N_RESAMPLES", 200)
    monkeypatch.setattr(probe_gates, "AUC_RESAMPLES", 40)
    monkeypatch.setattr(probe_study, "_tokenizer", lambda args: Words())
    train = [
        choice_row(f"row-{at}", f"item{at // 2}", at % 2 == 0, (at // 2) % 3 == 0)
        | {"label": int(at % 2 == 0), "split": "dev" if (at // 2) % 4 == 0 else "fit"}
        for at in range(240)
    ]
    put(tmp_path / "train_in.jsonl", train)
    put(tmp_path / "train_base.jsonl", [{"id": row["id"], "response": row["said"]} for row in train])
    states_of(train, 1.0, tmp_path / "acts" / "train")
    asked = [
        choice_row(f"pair{at}.{'amb' if open_ else 'dis'}", f"pair{at}", open_, at % (4 if open_ else 5) == 0)
        for at in range(40)
        for open_ in (True, False)
    ]
    put(tmp_path / "eval_in.jsonl", asked)
    put(tmp_path / "base.jsonl", [row | {"response": row["said"]} for row in asked])
    put(tmp_path / "tuned.jsonl", [row | {"response": "C"} for row in asked])
    states_of(asked, 0.0, tmp_path / "acts" / "eval")
    spec = {"sets": [{"name": "plain", "base": "base.jsonl", "fine_tuned": "tuned.jsonl"}]}
    (tmp_path / "sets.json").write_text(json.dumps(spec), encoding="utf-8")
    fitted = ["--rows", str(tmp_path / "train_in.jsonl"), "--states", str(tmp_path / "acts" / "train")]
    assert probe_fit.main([*fitted, "--out", str(tmp_path / "probe_1")]) == 0
    error = [*fitted, "--answers", str(tmp_path / "train_base.jsonl"), "--out", str(tmp_path / "out")]
    assert probe_study.main(["error-fit", *error]) == 0
    return tmp_path, train, asked


def texts(rows):
    """Every text of prompt rows that may not be written: contexts, questions, option texts and prompts."""
    held = {row["prompt"] for row in rows} | {row["prompt"].split("\n")[0] for row in rows}
    return held | {option["text"] for option in OPTIONS} | {CLOSE[0], CLOSE[-1]}


def scored(folder):
    """The arguments the two scoring steps share."""
    asked = ["--sets", str(folder / "sets.json"), "--prompts", str(folder / "eval_in.jsonl")]
    asked += ["--states", str(folder / "acts" / "eval"), "--probe", f"probe_1={folder / 'probe_1'}"]
    return [*asked, "--probe", f"error_probe={folder / 'out' / 'probe_error'}", "--out", str(folder / "results")]


def test_the_error_probe_is_fitted_on_the_first_probes_rows_with_the_base_models_error_as_its_label(stored, capsys):
    folder, train, _ = stored

    record = read_json(folder / "out" / "probe_error_fit.json")
    assert sorted(path.name for path in (folder / "out").iterdir()) == [
        "probe_error",
        "probe_error_fit.json",
        "probe_error_rows.jsonl",
    ]
    assert record["label"] == "base model's own error" and (record["fit_rows"], record["dev_rows"]) == (180, 60)
    assert record["base_answers"]["dev"] == {
        "rows": 60,
        "unparseable": 0,
        "names_a_person_on_ambiguous": 10,
        "correct_on_disambiguated": 20,
    }
    assert record["label_relation"]["fit"] == probe_gates.relation(
        [{"ambiguous": row["open"], "base_wrong": row["wrong"]} for row in train if row["split"] == "fit"]
    )
    assert record["label_relation"]["dev"]["phi"] == 0.0
    assert (record["selected"]["layer"], record["selected"]["dev_auc"]) == (8, 1.0)
    assert (record["dev_base_wrong_rows"], record["dev_base_right_rows"]) == (20, 40)
    assert (record["dev_fired_on_base_wrong"], record["dev_fired_on_base_right"]) == (20, 0)
    assert not any(key.startswith("dev_ambiguous") or key.startswith("dev_fired_on_ambiguous") for key in record)
    # the frozen probe reads the unit that holds the base model's error, the first probe the other one
    frozen, first = probe.load(folder / "out" / "probe_error"), probe.load(folder / "probe_1")
    assert int(np.abs(frozen.coefficients).argmax()) == 4 and int(np.abs(first.coefficients).argmax()) == 2
    settings = read_json(folder / "out" / "probe_error" / "probe.json")
    assert settings["fitted_on"] == "train_in.jsonl" and "is not the correct letter" in settings["label"]
    rows = read_jsonl(folder / "out" / "probe_error_rows.jsonl")
    assert rows[0] == {
        "id": "row-0",
        "split": "dev",
        "condition": "ambiguous",
        "letter_base": "A",
        "letter_correct": "C",
        "letter_unknown": "C",
        "base_wrong": True,
    }
    assert [row["base_wrong"] for row in rows] == [row["wrong"] for row in train]
    put(folder / "train_base.jsonl", [{"id": row["id"], "response": "A"} for row in train[:-1]])
    asked = ["--rows", str(folder / "train_in.jsonl"), "--answers", str(folder / "train_base.jsonl")]
    with pytest.raises(SystemExit, match="239 answers for 240 training rows"):
        probe_study.main(["error-fit", *asked, "--states", str(folder / "acts" / "train"), "--out", str(folder / "x")])


def test_both_probes_are_scored_on_the_stored_sets_and_each_tells_its_own_label(stored):
    folder, _, asked = stored

    assert probe_study.main(["bbq", *scored(folder)]) == 0

    results = folder / "results"
    assert sorted(path.name for path in results.iterdir()) == [
        "probe-bbq-answers.jsonl",
        "probe-bbq.json",
        "probe-bbq.md",
    ]
    record = read_json(results / "probe-bbq.json")
    assert record["schema"] == "driftmap.probe-bbq/1" and list(record["probes"]) == ["probe_1", "error_probe"]
    assert record["probes"]["error_probe"]["layer"] == 8
    plain = record["sets"]["plain"]
    # 10 of 40 open rows and 8 of 40 settled rows are wrong; a gated settled row stays wrong
    assert plain["labels"]["ambiguous_base_wrong"] == 10 and plain["labels"]["disambiguated_base_wrong"] == 8
    assert (plain["gates"]["probe_1"]["auc_answerability"], plain["gates"]["error_probe"]["auc_base_error"]) == (1, 1)
    assert plain["gates"]["probe_1"]["fired_on_ambiguous"] == [40, 40]
    assert plain["gates"]["probe_1"]["fired_on_disambiguated"] == [0, 40]
    assert plain["gates"]["error_probe"]["fired_on_base_wrong"] == [18, 18]
    assert plain["gates"]["error_probe"]["fired_on_base_right"] == [0, 62]
    wrong = {name: found["wrong_answers"] for name, found in plain["systems"].items()}
    assert wrong == {"base": 18, "fine_tuned": 40, "probe_1": 8, "error_probe": 8}
    assert plain["systems"]["error_probe"]["disambiguated_answered_unknown"] == 8
    assert list(plain["changes"]) == ["base -> probe_1", "base -> error_probe", "probe_1 -> error_probe"]
    assert plain["changes"]["base -> probe_1"]["wrong_answers"]["change"] == -0.125
    assert plain["changes"]["probe_1 -> error_probe"]["wrong_answers"]["change"] == 0.0
    rows = read_jsonl(results / "probe-bbq-answers.jsonl")
    assert len(rows) == 80 and rows[0]["id"] == "pair0.amb" and rows[0]["letter_base_plus_error_probe"] == "C"
    assert rows[0]["error_probe_fired"] is True and rows[2]["error_probe_fired"] is False
    said = (results / "probe-bbq.md").read_text(encoding="utf-8")
    assert said == table.render_bbq(record) and "computed on stored answers, not served through a hook" in said
    assert "| plain | 22.5% (18/80) | 10.0% (8/80) | 10.0% (8/80) | 50.0% (40/80) |" in said
    for path in results.iterdir():
        written = path.read_text(encoding="utf-8")
        assert not any(text in written for text in texts(asked)), path.name
    with pytest.raises(SystemExit, match="a probe is given as NAME=FOLDER"):
        probe_study.main(["bbq", *scored(folder), "--probe", str(folder / "probe_1")])


def test_the_text_baselines_are_fitted_for_both_labels_and_scored_beside_the_probes(stored):
    folder, train, asked = stored
    labels = ["--rows", str(folder / "train_in.jsonl"), "--labels", str(folder / "out" / "probe_error_rows.jsonl")]
    model = ["--model", str(folder / "model"), "--manifest", str(folder / "manifest.json")]

    assert probe_study.main(["baselines", *labels, *model, *scored(folder)]) == 0

    results = folder / "results"
    record = read_json(results / "probe-baselines.json")
    names = ["length_answerability", "length_base_error", "tfidf_answerability", "tfidf_base_error"]
    assert record["schema"] == "driftmap.probe-baselines/1"
    assert list(record["thresholds"]) == ["probe_1", "error_probe", *names]
    fitted = record["fit"]
    assert (fitted["fit_rows"], fitted["dev_rows"]) == (180, 60) and set(names) <= set(fitted)
    assert fitted["context_tokens"] == {"fit_ambiguous_median": 3.0, "fit_disambiguated_median": 10.0}
    assert set(fitted["tfidf_vocabulary"]) == {"tfidf_answerability", "tfidf_base_error"}
    # the words tell an open prompt from a settled one, and say nothing of the base model's error
    assert fitted["length_answerability"]["selected"]["dev_auc"] == 1.0
    assert fitted["length_base_error"]["selected"]["dev_auc"] == 0.5
    plain = record["sets"]["plain"]
    assert plain["gates"]["length_answerability"]["auc_answerability"] == 1.0
    assert plain["gates"]["tfidf_answerability"]["auc_answerability"] == 1.0
    assert plain["gates"]["length_base_error"]["auc_base_error"] < 0.6 < plain["gates"]["error_probe"]["auc_base_error"]
    assert list(plain["changes"]) == [f"base -> {name}" for name in record["thresholds"]]
    gaps = plain["auc_differences"]
    assert list(gaps["answerability"]) == ["probe_1 minus length_answerability", "probe_1 minus tfidf_answerability"]
    assert list(gaps["base_error"]) == ["error_probe minus length_base_error", "error_probe minus tfidf_base_error"]
    assert gaps["answerability"]["probe_1 minus length_answerability"]["difference"] == 0.0
    assert gaps["base_error"]["error_probe minus length_base_error"]["difference"] > 0.4
    assert gaps["base_error"]["error_probe minus length_base_error"]["ci95"][0] > 0
    rows = read_jsonl(results / "probe-baselines-answers.jsonl")
    assert len(rows) == 80 and {"tfidf_base_error_score", "letter_base_plus_length_answerability"} <= set(rows[0])
    said = (results / "probe-baselines.md").read_text(encoding="utf-8")
    assert "## AUC against the base model's own error" in said and "## The baselines as fitted" in said
    for path in results.iterdir():
        written = path.read_text(encoding="utf-8")
        assert not any(text in written for text in texts(asked) | texts(train)), path.name


def test_the_served_sample_takes_every_fifth_pair_of_the_test_items_and_every_tenth_of_the_official_ones():
    plain = [
        {"id": f"t{at}.{kind}", "prompt": f"p{at}{kind}", "letter_unknown": "C", "base_item": f"t{at}"}
        for at in range(12)
        for kind in ("amb", "dis")
    ]
    official = {
        code: [
            {"prompt_id": f"m.{code}.{at}.{kind}", "prompt": f"{code}{at}{kind}", "letter_unknown": "B"}
            | {"base_item": f"m{at}"}
            for at in range(25)
            for kind in ("amb", "dis")
        ]
        for code in ("en", "tr")
    }

    rows = probe_study.served_sample(plain, official)

    assert [row["id"] for row in rows[:6]] == [
        f"bbq_test_en|t{at}.{kind}" for at in (0, 5, 10) for kind in ("amb", "dis")
    ]
    assert [row["id"] for row in rows[6:12]] == [
        f"mbbq_en|m.en.{at}.{kind}" for at in (0, 10, 20) for kind in ("amb", "dis")
    ]
    assert [row["id"].split("|")[0] for row in rows[12:]] == ["mbbq_tr"] * 6 and len(rows) == 18
    assert rows[0] == {"id": "bbq_test_en|t0.amb", "prompt": "p0amb", "letter_unknown": "C"}
    assert probe_study.every(plain, 5) == {"t0", "t5", "t10"}


def test_the_served_gate_is_compared_with_the_gate_computed_afterwards_row_by_row(stored):
    folder, _, asked = stored
    sets = probe_gates.load_sets(folder / "sets.json")
    frozen = probe.load(folder / "probe_1")
    offline, _ = probe_gates.activation_scores(folder / "probe_1", folder / "eval_in.jsonl", folder / "acts" / "eval")
    states, layers = probe.load_states(folder / "acts" / "eval")
    by_prompt = {row["prompt"]: states[row["id"]][layers.index(frozen.layer)] for row in asked}
    sample = [{"id": f"plain|{row['id']}", "prompt": row["prompt"], "letter_unknown": "C"} for row in asked[:4]]
    kept = {row["id"]: by_row for row, by_row in zip(sample, asked[:4], strict=True)}
    live, free, run = {}, {}, {}
    for row in sample:
        fired = offline[row["prompt"]] >= frozen.threshold
        said = kept[row["id"]]["said"]
        live[row["id"]] = {
            "response": "C" if fired else said,
            "fired": bool(fired),
            "probe_probability": offline[row["prompt"]] + 1e-6,
            "hook_minus_hidden_state": 0.0,
        }
        free[row["id"]] = {"response": said}
        run[row["id"]] = by_prompt[row["prompt"]] + np.float32(0.25)
    # the last row's served answer names a person although the probe did not fire and the stored answer is A
    live[sample[3]["id"]]["response"] = "B"

    record, rows = probe_study.served(sample, live, free, sets, offline, frozen, by_prompt, run)

    assert record["probe"] == {"layer": frozen.layer, "threshold": frozen.threshold}
    assert (
        list(record["by_set"]) == ["mbbq (stored answers: GPU)"]
        and record["all"] == record["by_set"]["mbbq (stored answers: GPU)"]
    )
    assert record["all"] == {
        "prompts": 4,
        "served_equals_offline_gate_on_stored_answers": 3,
        "served_equals_offline_gate_on_ungated_cpu_answers": 3,
        "firing_decision_equal": 4,
        "fired": 2,
        "fired_rows_answering_the_unknown_letter_alone": 2,
        "unfired_rows": 2,
        "unfired_rows_with_the_ungated_text": 1,
        "ungated_cpu_letter_equals_stored_letter": 4,
        "ungated_cpu_text_equals_stored_text": 4,
    }
    gaps = record["largest_absolute_differences"]
    assert gaps["probability"] == pytest.approx(1e-6, rel=0.01) and gaps["hook_to_run_hidden_state"] == 0.0
    assert gaps["run_state_to_stored_activation"] == pytest.approx(0.25)
    assert rows[3] == {
        "id": sample[3]["id"],
        "condition": "disambiguated",
        "probe_probability_served": round(live[sample[3]["id"]]["probe_probability"], 6),
        "probe_probability_offline": round(offline[sample[3]["prompt"]], 6),
        "fired_served": False,
        "fired_offline": False,
        "letter_served": "B",
        "letter_offline_gate": "A",
        "letter_ungated_cpu": "A",
        "letter_stored_base": "A",
        "letter_unknown": "C",
        "letter_correct": "A",
    }
    assert not any(text in json.dumps(rows) for text in texts(asked))
    said = table.render_served({"schema": table.SCHEMAS["served"]} | record)
    assert "| served answer equals the gate on stored answers | 3 | 3 |" in said
    del free[sample[0]["id"]]
    with pytest.raises(SystemExit, match="do not hold the sample's prompts"):
        probe_study.served(sample, live, free, sets, offline, frozen, by_prompt, run)


def test_the_squad_probe_is_fitted_on_the_train_sample_and_scored_once_on_the_dev_sample(stored):
    folder, _, _ = stored
    samples = {"train": squad_sample("train", 10, 12), "dev": squad_sample("dev", 6, 8)}
    for name, rows in samples.items():
        put(folder / f"squad_{name}_in.jsonl", rows)
        answers = [{"id": row["id"], "response": row["said"], "stop_reason": "eos"} for row in rows]
        put(folder / f"squad_{name}_base.jsonl", answers)
        states_of(rows, 1.0 if name == "train" else 0.0, folder / "acts" / f"squad_{name}")
    model = ["--model", str(folder / "model"), "--manifest", str(folder / "manifest.json")]
    train = [str(folder / "squad_train_in.jsonl"), "--answers", str(folder / "squad_train_base.jsonl")]
    files = {folder / "squad_train_in.jsonl": ("train", []), folder / "squad_dev_in.jsonl": ("test", [])}
    passed = ["--passed", str(gated(folder / "squad_passed.json", files))]
    fitted = ["--train", *train, "--states", str(folder / "acts" / "squad_train"), *passed, *model]
    fitted += ["--out", str(folder / "out")]

    assert probe_study.main(["squad-fit", *fitted]) == 0

    record = read_json(folder / "out" / "probe_squad_fit.json")
    assert list(record) == ["label", "split", "articles_in_both_splits", "probe", "baselines", "curated"]
    assert record["curated"] == {
        "train": {
            "file": "squad_train_in.jsonl",
            "rows_in": 120,
            "dropped": {"row_shape": 0, "schema_gate": 0},
            "rows_passed": 120,
        }
    }
    assert record["articles_in_both_splits"] == 0
    assert (record["split"]["fit"]["articles"], record["split"]["dev"]["articles"]) == (8, 2)
    assert record["split"]["dev"]["rows"] == 24 and record["split"]["dev"]["base_wrong"] == 8
    assert (record["probe"]["selected"]["layer"], record["probe"]["selected"]["dev_auc"]) == (8, 1.0)
    assert (record["probe"]["dev_base_wrong_rows"], record["probe"]["dev_fired_on_base_wrong"]) == (8, 8)
    assert list(record["baselines"]) == ["length", "tfidf"] and "selected" in record["baselines"]["tfidf"]
    settings = read_json(folder / "out" / "probe_squad" / "probe.json")
    assert settings["fitted_on"] == "squad_train_in.jsonl" and "reading check's rule" in settings["label"]

    scored_ = ["--dev", str(folder / "squad_dev_in.jsonl"), "--dev-answers", str(folder / "squad_dev_base.jsonl")]
    scored_ += ["--dev-states", str(folder / "acts" / "squad_dev"), "--train", train[0], "--train-answers", train[2]]
    scored_ += ["--in-domain", str(folder / "out" / "probe_squad"), "--zero-shot", f"probe_1={folder / 'probe_1'}"]
    assert probe_study.main(["squad", *scored_, *passed, *model, "--out", str(folder / "results")]) == 0

    results = folder / "results"
    assert sorted(path.name for path in results.iterdir()) == [
        "probe-squad-answers-in-domain.jsonl",
        "probe-squad-answers-zero-shot.jsonl",
        "probe-squad.json",
        "probe-squad.md",
    ]
    found = read_json(results / "probe-squad.json")
    assert list(found) == [
        "schema",
        "dev_sample",
        "base",
        "zero_shot",
        "in_domain",
        "in_domain_auc_differences",
        "curated",
    ]
    assert found["curated"]["train"] == record["curated"]["train"]
    assert found["curated"]["test"] == {
        "file": "squad_dev_in.jsonl",
        "rows_in": 48,
        "dropped": {"row_shape": 0, "schema_gate": 0},
        "rows_passed": 48,
    }
    # of 48 questions the base model has 18 wrong: 12 unanswerable ones answered and 6 answerable ones
    assert (found["dev_sample"]["rows"], found["dev_sample"]["articles"]) == (48, 6)
    assert (found["dev_sample"]["hallucinations"], found["dev_sample"]["answerable_without_gold"]) == (12, 6)
    assert (found["base"]["wrong_answers"], found["base"]["answered"]) == (18, 36)
    assert list(found["zero_shot"]) == ["probe_1"] and list(found["in_domain"]) == ["squad_probe", "length", "tfidf"]
    own = found["in_domain"]["squad_probe"]
    assert (own["layer"], own["auc_base_error"], own["fired_on_base_wrong"]) == (8, 1.0, [18, 18])
    # the gate fires on every wrong answer: the unanswerable ones turn right, the answerable ones stay wrong
    assert (own["gated"]["wrong_answers"], own["gated"]["hallucinations"], own["gated"]["answered"]) == (6, 0, 18)
    moved = own["base_to_gated_wrong_answers"]
    assert (moved["change"], moved["articles"], moved["articles_better"]) == (-0.25, 6, 6)
    assert (moved["questions_wrong_to_right"], moved["questions_right_to_wrong"]) == (12, 0)
    assert len(own["curve"]) == len(probe.THRESHOLDS)
    # the first probe reads which questions are unanswerable, as the states of the sample hold it
    assert found["zero_shot"]["probe_1"]["auc_unanswerable"] == 1.0
    assert list(found["in_domain_auc_differences"]) == ["squad_probe minus length", "squad_probe minus tfidf"]
    assert found["in_domain_auc_differences"]["squad_probe minus length"]["difference"] > 0.3
    rows = read_jsonl(results / "probe-squad-answers-in-domain.jsonl")
    assert rows[0] == {
        "id": "squad.dev0000",
        "condition": "unanswerable",
        "base_answer_wording": "no abstention wording",
        "base_answer_holds_gold": False,
        "base_wrong": True,
        "squad_probe_score": rows[0]["squad_probe_score"],
        "length_score": rows[0]["length_score"],
        "tfidf_score": rows[0]["tfidf_score"],
        "squad_probe_fired": True,
        "length_fired": rows[0]["length_fired"],
        "tfidf_fired": rows[0]["tfidf_fired"],
    }
    assert set(read_jsonl(results / "probe-squad-answers-zero-shot.jsonl")[0]) == {
        "id",
        "condition",
        "base_answer_wording",
        "base_answer_holds_gold",
        "base_wrong",
        "probe_1_score",
        "probe_1_fired",
    }
    said = (results / "probe-squad.md").read_text(encoding="utf-8")
    assert said == table.render_squad(found)
    assert (
        "pass the CuratorKIT gate: 120 of 120 train questions (dropped: row shape 0, schema gate 0); "
        "48 of 48 test questions (dropped: row shape 0, schema gate 0)." in said
    )
    assert "| base model + in-domain error probe, fitted on the SQuAD train sample |" in said
    hidden = {text for rows in samples.values() for row in rows for text in (row["context"], row["question"])}
    hidden |= {row["said"] for rows in samples.values() for row in rows if row["said"] != TRAINED}
    for path in [*results.iterdir(), folder / "out" / "probe_squad_fit.json"]:
        written = path.read_text(encoding="utf-8")
        assert not any(text in written for text in hidden), path.name


def test_a_row_the_gate_dropped_is_neither_fitted_on_nor_scored(stored):
    folder, _, _ = stored
    samples = {"train": squad_sample("train", 10, 12), "dev": squad_sample("dev", 6, 8)}
    for name, rows in samples.items():
        put(folder / f"squad_{name}_in.jsonl", rows)
        answers = [{"id": row["id"], "response": row["said"], "stop_reason": "eos"} for row in rows]
        put(folder / f"squad_{name}_base.jsonl", answers)
        states_of(rows, 1.0 if name == "train" else 0.0, folder / "acts" / f"squad_{name}")
    # the gate drops two train questions and the first dev question, an unanswerable one the base model answers
    files = {
        folder / "squad_train_in.jsonl": ("train", ["squad.train0000", "squad.train0001"]),
        folder / "squad_dev_in.jsonl": ("test", ["squad.dev0000"]),
    }
    passed = ["--passed", str(gated(folder / "squad_passed.json", files))]
    model = ["--model", str(folder / "model"), "--manifest", str(folder / "manifest.json")]
    train = ["--train", str(folder / "squad_train_in.jsonl")]
    answers = str(folder / "squad_train_base.jsonl")
    fitted = [*train, "--answers", answers, "--states", str(folder / "acts" / "squad_train")]

    assert probe_study.main(["squad-fit", *fitted, *passed, *model, "--out", str(folder / "out")]) == 0

    record = read_json(folder / "out" / "probe_squad_fit.json")
    assert record["split"]["fit"]["rows"] + record["split"]["dev"]["rows"] == 118
    assert record["curated"]["train"] == {
        "file": "squad_train_in.jsonl",
        "rows_in": 120,
        "dropped": {"row_shape": 0, "schema_gate": 2},
        "rows_passed": 118,
    }
    scored_ = ["--dev", str(folder / "squad_dev_in.jsonl"), "--dev-answers", str(folder / "squad_dev_base.jsonl")]
    scored_ += ["--dev-states", str(folder / "acts" / "squad_dev"), *train, "--train-answers", answers]
    scored_ += ["--in-domain", str(folder / "out" / "probe_squad"), *passed, *model]
    assert probe_study.main(["squad", *scored_, "--out", str(folder / "results")]) == 0

    found = read_json(folder / "results" / "probe-squad.json")
    assert (found["dev_sample"]["rows"], found["dev_sample"]["hallucinations"]) == (47, 11)
    assert (found["base"]["wrong_answers"], found["base"]["answered"]) == (17, 35)
    assert found["in_domain"]["squad_probe"]["fired_on_base_wrong"] == [17, 17]
    assert found["curated"]["test"]["dropped"] == {"row_shape": 0, "schema_gate": 1}
    assert found["zero_shot"] == {}
    rows = read_jsonl(folder / "results" / "probe-squad-answers-in-domain.jsonl")
    assert len(rows) == 47 and "squad.dev0000" not in {row["id"] for row in rows}
    said = (folder / "results" / "probe-squad.md").read_text(encoding="utf-8")
    assert "118 of 120 train questions (dropped: row shape 0, schema gate 2)" in said
    assert "47 of 48 test questions (dropped: row shape 0, schema gate 1)" in said
    # passing ids written for other rows than the file now holds are refused, and so is a run without them
    put(folder / "squad_dev_in.jsonl", [row | {"title": "another article"} for row in samples["dev"]])
    with pytest.raises(SystemExit, match="squad_passed.json was written for another squad_dev_in.jsonl"):
        probe_study.main(["squad", *scored_, "--out", str(folder / "again")])
    with pytest.raises(SystemExit):
        probe_study.main(["squad-fit", *fitted, *model, "--out", str(folder / "again")])
    assert not (folder / "again").exists()
