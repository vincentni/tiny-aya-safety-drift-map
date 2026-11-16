"""SQuAD 2.0 for the probe study: the two samples, the base model's error, and a system that abstains on a gate."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from driftmap.attribution import build, probe, probe_gates, probe_squad, reading
from tests.fakes.reading_sources import squad_rows, write

TRAINED = build.SENTENCES["Cannot be determined"]


def no_download(url):
    raise AssertionError(f"asked for {url}")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def put(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """The made-up validation file, a train file of other articles, and samples of 20 + 20 and 40 + 40."""
    folder = write(tmp_path, monkeypatch)
    train = folder / reading.SQUAD_FOLDER / probe_squad.TRAIN_FILE
    rows = [
        dict(row, id=f"t{row['id']}", title=f"Train {row['title']}", context=f"train {row['context']}")
        for row in squad_rows()
    ]
    pq.write_table(pa.Table.from_pylist(rows), train)
    files = {
        name: hashlib.sha256((folder / reading.SQUAD_FOLDER / name).read_bytes()).hexdigest()
        for name in (reading.SQUAD_FILE, probe_squad.TRAIN_FILE)
    }
    monkeypatch.setattr(probe_squad, "FILES", files)
    samples = {"dev": (reading.SQUAD_FILE, 20), "train": (probe_squad.TRAIN_FILE, 40)}
    samples["dev2"] = (reading.SQUAD_FILE, 20)
    monkeypatch.setattr(probe_squad, "SAMPLES", samples)
    return folder


def test_the_draw_alternates_unanswerable_and_answerable_questions_each_on_a_short_passage_of_its_own(sources):
    source = sources / reading.SQUAD_FOLDER / reading.SQUAD_FILE

    rows = probe_squad.draw(source, 20, "a seed")

    assert [row["condition"] for row in rows] == ["unanswerable", "answerable"] * 20
    assert len({row["context"] for row in rows}) == 40 and len({row["id"] for row in rows}) == 40
    assert all(len(row["context"].split()) < reading.MAX_WORDS for row in rows)
    for row in rows:
        assert row["id"].startswith("squad.") and (row["set"], row["language"]) == ("benign", "en")
        assert row["prompt"] == "\n".join([row["context"], row["question"], reading.FREE])
        assert bool(row["gold_answers"]) == (row["condition"] == "answerable") and len(row["gold_answers"]) <= 1
    # the articles are taken in turn: twenty questions of a kind come from twenty articles
    for kind in probe_squad.CONDITIONS:
        assert len({row["title"] for row in rows if row["condition"] == kind}) == 20
    assert probe_squad.draw(source, 20, "a seed") == rows
    assert [row["id"] for row in probe_squad.draw(source, 20, "another seed")] != [row["id"] for row in rows]
    # 240 short passages hold 121 unanswerable questions and then 119 answerable ones
    with pytest.raises(SystemExit, match="only 119 answerable questions on unused passages"):
        probe_squad.draw(source, 121, "a seed")
    assert probe_squad.seed_of("dev") != probe_squad.seed_of("train") and str(probe.SEED) in probe_squad.seed_of("dev")


def test_the_two_samples_are_written_with_their_counts_and_hashes_from_checked_files(tmp_path, sources, capsys):
    out = tmp_path / "out"
    out.mkdir()
    first = probe_squad.draw(sources / reading.SQUAD_FOLDER / reading.SQUAD_FILE, 20, probe_squad.seed_of("dev"))
    put(tmp_path / "earlier.jsonl", [{"id": row["id"]} for row in first[:3]] + [{"id": "squad.other"}])
    argv = ["--sources", str(sources), "--out", str(out)]

    assert probe_squad.main([*argv, "--reading", str(tmp_path / "earlier.jsonl")], no_download) == 0

    assert sorted(path.name for path in out.iterdir()) == [
        "squad_build.json",
        "squad_dev2_in.jsonl",
        "squad_dev_in.jsonl",
        "squad_train_in.jsonl",
    ]
    dev, train = read_jsonl(out / "squad_dev_in.jsonl"), read_jsonl(out / "squad_train_in.jsonl")
    assert dev == first and len(train) == 80 and all(row["title"].startswith("Train ") for row in train)
    summary = json.loads((out / "squad_build.json").read_text(encoding="utf-8"))
    assert (summary["revision"], summary["licence"]) == (reading.SQUAD_REVISION, reading.LICENCE)
    assert summary["dev"] == {
        "file_sha256": probe_squad.FILES[reading.SQUAD_FILE],
        "rows": 40,
        "by_condition": {"unanswerable": 20, "answerable": 20},
        "articles": len({row["title"] for row in dev}),
        "distinct_passages": 40,
        "distinct_ids": 40,
    }
    assert (summary["train"]["rows"], summary["train"]["file_sha256"]) == (
        80,
        probe_squad.FILES[probe_squad.TRAIN_FILE],
    )
    assert (summary["passages_in_both_samples"], summary["articles_in_both_samples"]) == (0, 0)
    assert summary["dev_rows_also_in_the_reading_check"] == 3
    # the second sample of the validation file asks none of the first one's questions
    second = read_jsonl(out / "squad_dev2_in.jsonl")
    assert len(second) == 40 and not {row["id"] for row in second} & {row["id"] for row in dev}
    assert summary["dev2"]["after"] == "dev" and summary["dev2"]["questions_also_in_the_first_sample"] == 0
    shared = len({row["context"] for row in second} & {row["context"] for row in dev})
    assert summary["dev2"]["passages_also_in_the_first_sample"] == shared == 0
    assert summary["dev2"]["passages_not_in_the_first_sample"] == 40
    assert (summary["passages_in_dev2_and_train"], summary["articles_in_dev2_and_train"]) == (0, 0)
    assert summary["files_sha256"] == {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("squad_dev_in.jsonl", "squad_train_in.jsonl", "squad_dev2_in.jsonl")
    }
    # the counts hold no text of the source
    said = json.dumps(summary)
    assert not any(row["question"] in said or row["context"] in said for row in dev)
    assert '"rows": 40' in capsys.readouterr().out


def test_a_missing_file_is_downloaded_at_the_pinned_revision_and_a_changed_one_is_refused(tmp_path, sources):
    asked = []

    def served(url):
        asked.append(url)
        return (sources / reading.SQUAD_FOLDER / url.split("/resolve/")[1].split("/", 1)[1]).read_bytes()

    assert probe_squad.main(["--sources", str(tmp_path / "empty"), "--out", str(tmp_path / "two")], served) == 0

    hub = f"https://huggingface.co/datasets/{reading.SQUAD}/resolve/{reading.SQUAD_REVISION}"
    assert sorted(asked) == sorted(f"{hub}/{name}" for name in (reading.SQUAD_FILE, probe_squad.TRAIN_FILE))
    assert probe_squad.main(["--sources", str(sources), "--out", str(tmp_path / "one")], no_download) == 0
    for name in ("squad_dev_in.jsonl", "squad_train_in.jsonl"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    with (sources / reading.SQUAD_FOLDER / probe_squad.TRAIN_FILE).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(SystemExit, match=f"train-00000-of-00001.parquet: sha256 .* recorded for {reading.SQUAD} at"):
        probe_squad.main(["--sources", str(sources), "--out", str(tmp_path / "three")], no_download)
    assert not (tmp_path / "three").exists()


def test_a_later_sample_takes_none_of_the_earlier_questions_and_an_earlier_passage_only_when_it_must(sources):
    source = sources / reading.SQUAD_FOLDER / reading.SQUAD_FILE
    first = probe_squad.draw(source, 100, "a seed")

    second = probe_squad.draw(source, 30, "another seed", first)

    assert [row["condition"] for row in second] == ["unanswerable", "answerable"] * 30
    assert not {row["id"] for row in second} & {row["id"] for row in first}
    assert len({row["context"] for row in second}) == 60
    # the first sample left 40 of the 240 short passages unused: they are all taken before a used one is
    shared = {row["context"] for row in second} & {row["context"] for row in first}
    assert len(shared) == 20
    assert probe_squad.draw(source, 30, "another seed", first) == second
    assert probe_squad.draw(source, 30, "another seed", ()) != second


def test_the_recorded_files_and_sample_sizes_are_those_of_the_study():
    assert probe_squad.SAMPLES == {
        "dev": (reading.SQUAD_FILE, 500),
        "train": (probe_squad.TRAIN_FILE, 1000),
        "dev2": (reading.SQUAD_FILE, 500),
    }
    assert probe_squad.AFTER == {"dev2": "dev"}
    assert set(probe_squad.FILES) == {reading.SQUAD_FILE, probe_squad.TRAIN_FILE}
    assert all(len(sha) == 64 and set(sha) <= set("0123456789abcdef") for sha in probe_squad.FILES.values())
    assert probe_squad.FILES[reading.SQUAD_FILE] == reading.SQUAD_FILES[reading.SQUAD_FILE]


def test_an_answer_is_classed_by_its_wording_and_by_whether_it_holds_a_gold_answer(tmp_path):
    kinds = ["unanswerable", "unanswerable", "answerable", "answerable", "answerable"]
    asked = [
        {"id": f"squad.{at}", "condition": kind, "gold_answers": ["Paris"] if kind == "answerable" else []}
        for at, kind in enumerate(kinds)
    ]
    said = [TRAINED, "It is Paris.", "it is paris, I think", "The passage does not say.", None]
    stops = ["eos", "eos", "length", "eos", "eos"]
    put(tmp_path / "in.jsonl", asked)
    answers = [
        {"id": row["id"], "response": response, "stop_reason": stop}
        for row, response, stop in zip(asked, said, stops, strict=True)
    ]
    put(tmp_path / "answers.jsonl", answers[::-1])

    rows = probe_squad.rows_of(tmp_path / "in.jsonl", tmp_path / "answers.jsonl")

    assert [row["wording"] for row in rows] == [
        "trained sentence",
        probe_squad.NO_ABSTENTION,
        probe_squad.NO_ABSTENTION,
        "wider wording",
        probe_squad.NO_ABSTENTION,
    ]
    assert [row["abstained"] for row in rows] == [True, False, False, True, False]
    assert [row["holds_gold"] for row in rows] == [False, False, True, False, False]
    assert [row["correct"] for row in rows] == [True, False, True, False, False]
    assert [row["base_wrong"] for row in rows] == [False, True, False, True, True]
    assert [row["hallucination"] for row in rows] == [False, True, False, False, False]
    assert probe_squad.base_counts(rows) == {
        "rows": 5,
        "unanswerable": 2,
        "base_wrong": 3,
        "hallucinations": 1,
        "answerable_without_gold": 2,
        "answerable_abstained": 1,
        "wording": {"trained sentence": 1, "trained style": 0, "wider wording": 1, probe_squad.NO_ABSTENTION: 3},
        "cut_at_the_token_limit": 1,
    }
    put(tmp_path / "answers.jsonl", answers[:-1])
    with pytest.raises(SystemExit, match="4 answers for 5 rows of in.jsonl"):
        probe_squad.rows_of(tmp_path / "in.jsonl", tmp_path / "answers.jsonl")
    np.savez(tmp_path / "0.npz", ids=np.array(["squad.0"]), states=np.zeros((1, 1, 2)), layers=np.array([4]))
    with pytest.raises(SystemExit, match="1 hidden states for 5 rows under"):
        probe_squad.states_of(tmp_path, rows)


def classed(title, kind, said):
    """A row as ``rows_of`` leaves it: ``said`` is abstains, gold or miss."""
    unanswerable, abstained = kind == "unanswerable", said == "abstains"
    correct = abstained if unanswerable else said == "gold"
    return {
        "title": title,
        "unanswerable": unanswerable,
        "abstained": abstained,
        "correct": correct,
        "base_wrong": not correct,
        "hallucination": unanswerable and not abstained,
    }


def eight_rows():
    return [
        classed("A", "unanswerable", "abstains"),
        classed("A", "unanswerable", "miss"),
        classed("A", "answerable", "gold"),
        classed("B", "unanswerable", "miss"),
        classed("B", "answerable", "miss"),
        classed("B", "answerable", "abstains"),
        classed("C", "answerable", "gold"),
        classed("C", "unanswerable", "miss"),
    ]


def test_a_gated_answer_is_an_abstention_right_on_an_unanswerable_question_and_wrong_on_an_answerable_one():
    rows = eight_rows()

    base = probe_squad.measured(rows, np.zeros(8, dtype=bool))
    gated = probe_squad.measured(rows, np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=bool))

    assert base == {
        "rows": 8,
        "wrong_answers": 5,
        "wrong_answer_rate": 0.625,
        "answered": 6,
        "coverage": 0.75,
        "wrong_among_answered": 4,
        "risk": 0.6667,
        "hallucinations": 3,
        "unanswerable_abstained": 1,
        "answerable_rows": 4,
        "answerable_answered": 3,
        "answerable_answered_rate": 0.75,
        "answerable_correct": 2,
    }
    assert gated == {
        "rows": 8,
        "wrong_answers": 4,
        "wrong_answer_rate": 0.5,
        "answered": 3,
        "coverage": 0.375,
        "wrong_among_answered": 2,
        "risk": 0.6667,
        "hallucinations": 1,
        "unanswerable_abstained": 3,
        "answerable_rows": 4,
        "answerable_answered": 2,
        "answerable_answered_rate": 0.5,
        "answerable_correct": 1,
    }
    silent = probe_squad.measured(rows, np.ones(8, dtype=bool))
    assert (silent["answered"], silent["risk"], silent["wrong_answers"], silent["hallucinations"]) == (0, None, 4, 0)


def test_the_change_in_wrong_answers_is_counted_per_article():
    rows = eight_rows()

    found = probe_squad.change(rows, np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=bool))

    # wrong answers per article go 1, 3, 1 -> 0, 2, 2 over 3, 3 and 2 questions
    assert found == {
        "articles": 3,
        "change": -0.125,
        "ci95": probe_gates.interval(np.array([-1.0, -1.0, 1.0]), np.array([3.0, 3.0, 2.0])),
        "articles_better": 2,
        "articles_worse": 1,
        "sign_test_p": 1.0,
        "questions_wrong_to_right": 2,
        "questions_right_to_wrong": 1,
    }
    assert found["ci95"][0] < -0.125 < found["ci95"][1]
    still = probe_squad.change(rows, np.zeros(8, dtype=bool))
    assert (still["change"], still["ci95"], still["articles_better"]) == (0.0, [0.0, 0.0], 0)


def test_a_gate_is_read_at_its_threshold_and_along_the_whole_grid_of_thresholds():
    rows = eight_rows()
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.6, 0.45])

    found = probe_squad.read_gate(rows, scores, 0.5)

    fired = np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    assert (found["threshold"], found["auc_unanswerable"], found["auc_base_error"]) == (0.5, 0.6875, 0.8)
    assert (found["fired_on_unanswerable"], found["fired_on_answerable"]) == ([2, 4], [1, 4])
    assert (found["fired_on_base_wrong"], found["fired_on_base_right"]) == ([2, 5], [1, 3])
    assert found["gated"] == probe_squad.measured(rows, fired)
    assert found["base_to_gated_wrong_answers"] == probe_squad.change(rows, fired)
    curve = {point["threshold"]: point for point in found["curve"]}
    assert list(curve) == list(probe.THRESHOLDS)
    assert curve[0.5] == {
        "threshold": 0.5,
        "wrong_answer_rate": 0.5,
        "coverage": 0.375,
        "risk": 0.6667,
        "answerable_answered_rate": 0.5,
        "hallucinations": 1,
    }
    # at the lowest threshold every row is gated, at the highest none is
    assert (curve[0.05]["coverage"], curve[0.05]["risk"], curve[0.05]["wrong_answer_rate"]) == (0.0, None, 0.5)
    assert (curve[0.95]["coverage"], curve[0.95]["hallucinations"], curve[0.95]["wrong_answer_rate"]) == (
        0.75,
        3,
        0.625,
    )
    assert [point["coverage"] for point in found["curve"]] == sorted(point["coverage"] for point in found["curve"])


def test_a_judges_verdicts_relabel_the_rows_it_judged_and_its_reasons_are_not_read(tmp_path):
    rows = [
        dict(row, id=f"q{at}", holds_gold=row["correct"] and not row["unanswerable"])
        for at, row in enumerate(eight_rows())
    ]
    verdicts = [
        {"id": "q0", "status": "ok", "abstains": "yes", "correct": "n/a", "why": "a reason"},
        {"id": "q1", "status": "ok", "abstains": "yes", "correct": "n/a", "why": "a reason"},
        {"id": "q2", "status": "ok", "abstains": "no", "correct": "no", "why": "a reason"},
        {"id": "q3", "status": "unparsed", "why": "no usable answer"},
        {"id": "q4", "status": "ok", "abstains": "no", "correct": "yes", "why": "a reason"},
        {"id": "q5", "status": "empty_response"},
        {"id": "q6", "status": "ok", "abstains": "no", "correct": "yes", "why": "a reason"},
    ]
    put(tmp_path / "verdicts.jsonl", verdicts)

    read = probe_squad.read_verdicts(tmp_path / "verdicts.jsonl")
    kept = probe_squad.judged(rows, read)

    assert read["q0"] == {"status": "ok", "abstains": "yes", "correct": "n/a"}
    assert read["q5"] == {"status": "empty_response", "abstains": None, "correct": None}
    # the unparsed row and the row without a verdict are left out; an answer without text is not right
    assert [row["id"] for row in kept] == ["q0", "q1", "q2", "q4", "q5", "q6"]
    assert [row["abstained"] for row in kept] == [True, True, False, False, False, False]
    assert [row["base_wrong"] for row in kept] == [False, False, True, False, True, False]
    assert [row["hallucination"] for row in kept] == [False] * 6
    assert [row["rule_wrong"] for row in kept] == [False, True, False, True, True, False]
    assert [row["rule_abstained"] for row in kept] == [True, False, False, False, True, False]
    assert kept[3]["judge_says_gold"] is True and kept[3]["rule_holds_gold"] is False
    agreed = probe_squad.agreement(kept)
    assert agreed["abstains"] == {
        "rows": 6,
        "rule_judge": {"yes_yes": 1, "yes_no": 1, "no_yes": 1, "no_no": 3},
        "agree": 4,
        "kappa": 0.25,
    }
    assert agreed["gold_answer_on_answerable"]["rule_judge"] == {"yes_yes": 1, "yes_no": 1, "no_yes": 1, "no_no": 1}
    assert (agreed["answer_is_wrong"]["rows"], agreed["answer_is_wrong"]["agree"]) == (6, 3)


def test_two_readings_of_the_same_rows_are_counted_cell_by_cell_with_kappa():
    found = probe_squad.two_by_two([True, True, False, False], [True, False, False, False])

    assert found == {
        "rows": 4,
        "rule_judge": {"yes_yes": 1, "yes_no": 1, "no_yes": 0, "no_no": 2},
        "agree": 3,
        "kappa": 0.5,
    }
    assert probe_squad.two_by_two([True, True], [True, True])["kappa"] is None
    with pytest.raises(ValueError):
        probe_squad.two_by_two([True], [True, False])


def test_two_systems_on_the_same_questions_are_compared_per_article():
    rows = [dict(row, id=f"q{at}") for at, row in enumerate(eight_rows())]
    other = [dict(row) for row in rows[:-1]]
    other[1]["base_wrong"], other[6]["base_wrong"] = False, True

    found = probe_squad.against(rows, other)

    # the last question has no answer of the other system; wrong answers per article go 1, 3, 0 -> 0, 3, 1
    assert found == {
        "questions": 7,
        "articles": 3,
        "change": 0.0,
        "ci95": probe_gates.interval(np.array([-1.0, 0.0, 1.0]), np.array([3.0, 3.0, 1.0])),
        "articles_better": 1,
        "articles_worse": 1,
        "sign_test_p": 1.0,
    }


def test_answers_are_read_by_question_id_from_the_studys_runs_and_from_a_generate_job(tmp_path):
    put(tmp_path / "own.jsonl", [{"id": "squad.q1", "response": "one"}, {"id": "squad.q2", "response": "two"}])
    job = [{"prompt_id": "squad.dev2.q1", "response": "one"}, {"prompt_id": "squad.train.q.2", "response": "two"}]
    put(tmp_path / "job.jsonl", job)

    assert list(probe_squad.answers_of(tmp_path / "own.jsonl")) == ["squad.q1", "squad.q2"]
    found = probe_squad.answers_of(tmp_path / "job.jsonl")
    assert list(found) == ["squad.q1", "squad.q.2"] and found["squad.q1"]["response"] == "one"


def test_an_article_is_named_by_its_place_among_the_sorted_titles():
    rows = [{"title": "Warsaw"}, {"title": "Amazon"}, {"title": "Warsaw"}, {"title": "Normans"}]

    assert probe_squad.article_index(rows) == {"Amazon": 0, "Normans": 1, "Warsaw": 2}
