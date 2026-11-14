import csv
import json
import threading
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.judge import closed, rubric, validate
from driftmap.judge.validate import GoldRow

REFUSED = '{"request_harm": "yes", "refusal": "yes", "response_harm": "no"}'
COMPLIED = '{"request_harm": "yes", "refusal": "no", "response_harm": "yes"}'
BENIGN = '{"request_harm": "no", "refusal": "no", "response_harm": "no"}'
SEVEN = [*validate.LANGUAGES, "German"]
# (response harm, refusal) of items 0..7, repeating; item 4 is the rare harmful refusal once
CELLS = [
    ("harmful", "compliance"),
    ("unharmful", "refusal"),
    ("unharmful", "compliance"),
    ("unharmful", "compliance"),
    ("harmful", "compliance"),
    ("unharmful", "refusal"),
    ("unharmful", "refusal"),
    ("unharmful", "compliance"),
]


class FakeCompleter:
    """Answers each pair with the next scripted outcome for a text its user turn holds, else ``default``."""

    def __init__(self, script=None, default=REFUSED, delay=0.0):
        self.script = {text: list(outcomes) for text, outcomes in (script or {}).items()}
        self.default, self.delay = default, delay
        self.calls, self.in_flight, self.max_in_flight = [], 0, 0
        self._lock = threading.Lock()

    def complete(self, messages):
        user = messages[1]["content"]
        with self._lock:
            self.calls.append(messages)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            queue = next((outcomes for text, outcomes in self.script.items() if text in user), None)
            outcome = queue.pop(0) if queue else self.default
        try:
            time.sleep(self.delay)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            with self._lock:
                self.in_flight -= 1

    def asked(self, text):
        return sum(text in messages[1]["content"] for messages in self.calls)


def polyguard_records(items=48):
    """Rows in the columns of the multilingual test file: every item in seven languages with one set of labels."""
    records = []
    for item in range(items):
        harm, refusal = ("harmful", "refusal") if item == 4 else CELLS[item % 8]
        prompt_harm = None if item == 9 else "harmful" if harm == "harmful" or refusal == "refusal" else "unharmful"
        for language in SEVEN:
            records.append(
                {
                    "prompt": f"{language} request {item:03d}",
                    "response": f"{language} response {item:03d}",
                    "adversarial": item % 2 == 1,
                    "prompt_harm_label": prompt_harm,
                    "response_refusal_agreement": 3.0,
                    "response_refusal_label": refusal,
                    "response_harm_label": harm,
                    "subcategory": "benign" if prompt_harm == "unharmful" else "others",
                    "prompt_harm_agreement": None if prompt_harm is None else 2.0 if item % 5 == 0 else 3.0,
                    "response_harm_agreement": 3.0,
                    "id": item,
                    "language": language,
                    "prompt_label": "unsafe" if prompt_harm == "harmful" else "safe",
                    "prompt_categories": "",
                    "response_label": "unsafe" if harm == "harmful" else "safe",
                    "response_categories": "",
                }
            )
    return records


def jbb_records(rows=12):
    """Rows in the columns of the jailbreak judge comparison; every third response is a majority jailbreak."""
    records = []
    for index in range(rows):
        votes = ["1", "1", "0"] if index % 3 == 0 else ["0", "0", "1"] if index % 3 == 1 else ["0", "0", "0"]
        records.append(
            {
                "Index": str(index),
                "goal": f"goal {index:03d}",
                "prompt": f"jailbreak text {index:03d}",
                "target_response": f"target response {index:03d}",
                "human1": votes[0],
                "human2": votes[1],
                "human3": votes[2],
                "human_majority": "1" if votes.count("1") >= 2 else "0",
                "harmbench_cf": "1",
                "gpt4_cf": votes[0],
                "llamaguard2_cf": "0",
                "llama3_cf": "1" if votes.count("1") >= 2 else "0",
            }
        )
    return records


def xstest_records(model, rows=18):
    """Rows in the columns of one annotated-completions file: safe and contrast types, three final labels."""
    labels = ["1_full_compliance", "1_full_compliance", "2_full_refusal", "1_full_compliance", "2_full_refusal"]
    records = []
    for index in range(rows):
        final = "3_partial_refusal" if index == 5 else labels[index % 5]
        records.append(
            {
                "id": f"v2-{index + 1}",
                "type": "contrast_homonyms" if index % 2 else "homonyms",
                "prompt": f"xstest request {index:03d}",
                "completion": f"{model} completion {index:03d}",
                "annotation_1": final,
                "annotation_2": "2_full_refusal" if index == 5 else final,
                "agreement": index != 5,
                "final_label": final,
            }
        )
    return records


def gold_rows(n=6, source="polyguard", language="en", gold=None):
    return [
        GoldRow(
            id=f"{source}/{language}/{index}",
            source=source,
            language=language,
            prompt=f"request {index:03d}",
            response=f"response {index:03d}",
            gold=gold or {"request_harm": "yes", "refusal": "yes", "response_harm": "no"},
        )
        for index in range(n)
    ]


def run(rows, client, tmp_path, **options):
    lines, sleeps = [], []
    options = {"store": tmp_path / "answers.jsonl", "echo": lines.append, "sleep": sleeps.append} | options
    try:
        answers, error = validate.validate(rows, client, options.pop("concurrency", 4), **options), None
    except DriftmapError as stopped:
        answers, error = None, stopped
    return answers, lines, sleeps, error


def answer(request_harm, refusal, response_harm):
    return json.dumps({"request_harm": request_harm, "refusal": refusal, "response_harm": response_harm})


def table(yes_yes, yes_no, no_yes, no_no):
    """(reference, judge) pairs with the given counts."""
    return [("yes", "yes")] * yes_yes + [("yes", "no")] * yes_no + [("no", "yes")] * no_yes + [("no", "no")] * no_no


def test_level_splits_evenly_and_hands_on_what_a_small_stratum_cannot_take():
    assert validate.level(150, {"hc": 279, "hr": 5, "uc": 867, "ur": 556}) == {"hc": 48, "hr": 5, "uc": 49, "ur": 48}
    assert validate.level(6, {"a": 10, "b": 10}) == {"a": 3, "b": 3}
    assert validate.level(0, {"a": 1}) == {"a": 0}
    with pytest.raises(ValueError):
        validate.level(3, {"a": 1, "b": 1})


def test_the_multilingual_rows_are_the_same_items_in_each_of_the_six_languages():
    rows, left = validate.multilingual(polyguard_records(), 12, validate.SEED)

    by_language = {code: [row for row in rows if row.language == code] for code in validate.LANGUAGES.values()}
    assert [len(members) for members in by_language.values()] == [12] * 6 and len(rows) == 72
    items = [[row.meta["item"] for row in members] for members in by_language.values()]
    assert all(found == items[0] for found in items) and len(set(items[0])) == 12
    french = by_language["fr"][0]
    assert french.id == f"polyguard/fr/{french.meta['item']}" and french.source == "polyguard"
    assert french.prompt == f"French request {int(french.meta['item']):03d}"
    assert french.response == f"French response {int(french.meta['item']):03d}"
    assert left == {}


def test_the_multilingual_draw_is_two_fifths_harmful_responses_and_even_over_refusal_within_each():
    rows, _ = validate.multilingual(polyguard_records(), 20, validate.SEED)

    english = [row for row in rows if row.language == "en"]
    cells = Counter((row.gold["response_harm"], row.gold["refusal"]) for row in english)
    # eight harmful responses, the one harmful refusal among them; twelve harmless ones, half of them refusals
    assert cells == {("yes", "yes"): 1, ("yes", "no"): 7, ("no", "yes"): 6, ("no", "no"): 6}
    assert {row.meta["adversarial"] for row in english} == {"yes", "no"}
    items = [int(row.meta["item"]) for row in english]
    assert items == sorted(items)


def test_the_harmless_side_takes_the_places_the_harmful_responses_cannot_fill():
    records = [row for row in polyguard_records() if row["response_harm_label"] != "harmful" or row["id"] < 9]

    rows, _ = validate.multilingual(records, 30, validate.SEED)

    english = [row for row in rows if row.language == "en"]
    assert Counter(row.gold["response_harm"] for row in english) == {"yes": 3, "no": 27}


def test_the_wildguard_words_map_to_yes_and_no_and_a_label_without_a_majority_is_absent():
    rows, _ = validate.multilingual(polyguard_records(), 48, validate.SEED)

    english = {row.meta["item"]: row for row in rows if row.language == "en"}
    assert english["0"].gold == {"request_harm": "yes", "refusal": "no", "response_harm": "yes"}
    assert english["1"].gold == {"request_harm": "yes", "refusal": "yes", "response_harm": "no"}
    assert english["2"].gold == {"request_harm": "no", "refusal": "no", "response_harm": "no"}
    assert english["9"].gold == {"refusal": "yes", "response_harm": "no"}
    assert english["5"].meta["annotators_agreeing"] == "2" and english["1"].meta["annotators_agreeing"] == "3"
    # the item without a prompt-harm label counts its agreement over the labels it has
    assert english["9"].meta["annotators_agreeing"] == "3"


def test_an_item_is_left_out_without_a_majority_label_when_too_long_or_missing_a_language():
    records = polyguard_records()
    for record in records:
        if record["id"] == 2:
            record["response_harm_label"] = None
        if record["id"] == 3 and record["language"] == "Hindi":
            record["response"] = "x" * validate.MAX_PAIR_CHARS
    records = [record for record in records if not (record["id"] == 6 and record["language"] == "Arabic")]

    rows, left = validate.multilingual(records, 48, validate.SEED)

    assert left == {"no majority label": 1, "pair too long": 1, "not in every language": 1}
    assert {row.meta["item"] for row in rows}.isdisjoint({"2", "3", "6"}) and len(rows) == 45 * 6
    assert not any(row.language == "de" or "German" in row.prompt for row in rows)


def test_a_label_word_without_a_mapping_and_labels_that_differ_by_language_are_refused():
    records = polyguard_records()
    records[0]["response_harm_label"] = "borderline"
    with pytest.raises(DriftmapError, match="'borderline', which has no mapping") as error:
        validate.multilingual(records, 12, validate.SEED)
    assert error.value.exit_code is ExitCode.INTEGRITY

    records = polyguard_records()
    records[1]["response_refusal_label"] = "refusal"
    with pytest.raises(DriftmapError, match="item 0 has labels that differ by language"):
        validate.multilingual(records, 12, validate.SEED)


def test_the_draw_is_the_same_for_a_seed_and_differs_between_seeds():
    first, _ = validate.multilingual(polyguard_records(), 12, validate.SEED)
    again, _ = validate.multilingual(polyguard_records(), 12, validate.SEED)
    other, _ = validate.multilingual(polyguard_records(), 12, validate.SEED + 1)

    assert first == again
    assert [row.id for row in first] != [row.id for row in other]


def test_the_jailbreak_rows_take_the_goal_as_the_request_and_the_human_majority_as_response_harm():
    rows = validate.jailbreak(jbb_records(), 300, validate.SEED)

    assert len(rows) == 12 and [row.id for row in rows[:2]] == ["jbb/0", "jbb/1"]
    assert (rows[0].prompt, rows[0].response, rows[0].language) == ("goal 000", "target response 000", "en")
    assert [row.gold["response_harm"] for row in rows[:3]] == ["yes", "no", "no"] and len(rows[0].gold) == 1
    assert [row.meta["human_votes"] for row in rows[:3]] == ["2", "1", "0"]
    assert len(validate.jailbreak(jbb_records(), 6, validate.SEED)) == 6


def test_the_over_refusal_rows_are_even_over_refusal_and_count_a_partial_refusal_as_one():
    completions = {model: xstest_records(model) for model in validate.XSTEST_MODELS}

    rows = validate.over_refusal(completions, 40, validate.SEED)

    assert Counter(row.gold["refusal"] for row in rows) == {"yes": 20, "no": 20}
    assert {row.meta["model"] for row in rows} == set(validate.XSTEST_MODELS)
    every = validate.over_refusal(completions, 90, validate.SEED)
    partial = next(row for row in every if row.id == "xstest/gpt4/v2-6")
    assert partial.gold == {"request_harm": "yes", "refusal": "yes"}
    assert partial.meta == {
        "model": "gpt4",
        "final_label": "3_partial_refusal",
        "partial_refusal": "yes",
        "type": "contrast_homonyms",
        "annotators_agree": "no",
    }
    first = next(row for row in every if row.id == "xstest/llama2orig/v2-1")
    assert first.gold == {"request_harm": "no", "refusal": "no"}
    assert (first.prompt, first.response) == ("xstest request 000", "llama2orig completion 000")
    assert first.meta["partial_refusal"] == "no"


def test_build_sample_reads_the_pinned_files_and_says_what_it_drew(tmp_path):
    asked = []

    def fetch(repo, revision, file):
        asked.append((repo, revision, file))
        path = tmp_path / repo.replace("/", "--") / file
        path.parent.mkdir(parents=True, exist_ok=True)
        if repo == validate.POLYGUARD.repo:
            pq.write_table(pa.Table.from_pylist(polyguard_records()), path)
        elif repo == validate.JBB.repo:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(jbb_records()[0]))
                writer.writeheader()
                writer.writerows(jbb_records())
        else:
            pq.write_table(pa.Table.from_pylist(xstest_records(file.split("/")[1].split("-")[0])), path)
        return path

    rows, manifest = validate.build_sample(fetch, per_language=12, jailbreak_rows=300, over_refusal_rows=40)

    assert Counter(row.source for row in rows) == {"polyguard": 72, "jbb": 12, "xstest": 40}
    assert len({row.id for row in rows}) == len(rows)
    files = [*validate.POLYGUARD.files, *validate.JBB.files, *validate.XSTEST.files]
    assert [file for _, _, file in asked] == files
    assert {(repo, revision) for repo, revision, _ in asked} == {
        tuple(source[:2]) for source in (validate.POLYGUARD, validate.JBB, validate.XSTEST)
    }
    assert manifest["seed"] == validate.SEED and manifest["rows"] == 124
    assert manifest["composition"]["jbb/en"] == {"rows": 12, "response_harm": {"no": 8, "yes": 4}}
    assert manifest["composition"]["xstest/en"]["refusal"] == {"no": 20, "yes": 20}
    assert manifest["composition"]["polyguard/hi"]["rows"] == 12
    assert manifest["sources"]["polyguard"]["revision"] == validate.POLYGUARD.revision
    published = manifest["jbb_published_judges"]
    assert published["llama3_cf"]["agreement"] == 1.0 and published["llama3_cf"]["kappa"] == 1.0
    assert published["harmbench_cf"]["agreement"] == pytest.approx(4 / 12) and published["harmbench_cf"]["kappa"] == 0.0
    json.dumps(manifest)


def test_every_row_is_asked_with_the_rubric_and_fenced_turn_and_its_answer_is_stored(tmp_path):
    rows = gold_rows(6)
    client = FakeCompleter({"response 002": [COMPLIED]})

    answers, lines, _, error = run(rows, client, tmp_path)

    assert error is None and len(client.calls) == 6
    first = client.calls[0]
    assert first[0] == {"role": "system", "content": rubric.RUBRIC}
    assert first[1]["content"].startswith("<request-") and "\nrequest 00" in first[1]["content"]
    assert answers == {row.id: COMPLIED if row.id.endswith("/2") else REFUSED for row in rows}
    stored = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert sorted(entry["id"] for entry in stored) == [row.id for row in rows]
    assert {entry["rubric_sha256"] for entry in stored} == {rubric.prompt_sha256()}
    assert {entry["pair_sha256"] for entry in stored} == {validate.pair_sha256(row) for row in rows}
    assert lines[0] == "6 pairs: 0 answered before, 6 to ask" and lines[-1] == "6 pairs sent, 6 answers parsed"


def test_a_second_run_asks_nothing_and_a_changed_pair_or_rubric_is_asked_again(tmp_path, monkeypatch):
    rows = gold_rows(6)
    run(rows, FakeCompleter(), tmp_path)
    client = FakeCompleter()

    answers, lines, _, _ = run(rows, client, tmp_path)

    assert client.calls == [] and len(answers) == 6 and lines[0] == "6 pairs: 6 answered before, 0 to ask"

    rows[3] = rows[3].model_copy(update={"response": "another response"})
    answers, _, _, _ = run(rows, client, tmp_path)
    assert len(client.calls) == 1 and client.asked("another response") == 1 and len(answers) == 6

    monkeypatch.setattr(rubric, "REMINDER", "Answer with the object only.")
    client = FakeCompleter(default=BENIGN)
    answers, _, _, _ = run(rows, client, tmp_path)
    assert len(client.calls) == 6 and set(answers.values()) == {BENIGN}


def test_throttling_stops_the_run_without_a_retry_and_the_next_run_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "BATCH", 4)
    rows = gold_rows(10)
    client = FakeCompleter({"response 005": [closed.Throttled("the judge route is throttling or out of quota")]})

    answers, _, sleeps, error = run(rows, client, tmp_path, concurrency=1)

    assert answers is None and isinstance(error, closed.Throttled) and error.exit_code is ExitCode.GUARD_REFUSAL
    assert error.message == (
        "the judge route is throttling or out of quota; 5 of 10 waiting pairs were stored;"
        " run the command again later to continue"
    )
    assert client.asked("response 005") == 1 and sleeps == []
    assert len(client.calls) == 6 and len((tmp_path / "answers.jsonl").read_text().splitlines()) == 5

    client = FakeCompleter()
    answers, lines, _, error = run(rows, client, tmp_path)
    assert error is None and len(client.calls) == 5 and len(answers) == 10
    assert lines[0] == "10 pairs: 5 answered before, 5 to ask"


def test_an_answer_without_labels_beside_a_stopped_route_is_not_stored(tmp_path):
    rows = gold_rows(4)
    throttled = closed.Throttled("the judge route is throttling or out of quota")
    client = FakeCompleter({"response 001": ["no object"] * 3, "response 002": [throttled]})

    _, _, _, error = run(rows, client, tmp_path, concurrency=1)

    assert isinstance(error, closed.Throttled) and "1 of 4 waiting pairs were stored" in error.message
    assert list(validate.read_answers(tmp_path / "answers.jsonl", rows)) == [rows[0].id]


def test_an_unexpected_error_is_passed_on_after_the_answers_so_far_are_stored(tmp_path):
    rows = gold_rows(4)
    client = FakeCompleter({"response 002": [RuntimeError("boom")]})
    lines = []

    with pytest.raises(RuntimeError, match="boom"):
        validate.validate(rows, client, 1, store=tmp_path / "answers.jsonl", echo=lines.append, sleep=lambda s: None)

    assert lines[-1] == "stopped: 2 of 4 waiting pairs were stored"
    assert list(validate.read_answers(tmp_path / "answers.jsonl", rows)) == [rows[0].id, rows[1].id]


def test_a_transient_fault_is_retried_and_three_in_a_row_stop_the_run(tmp_path):
    rows = gold_rows(3)
    client = FakeCompleter({"response 001": [closed.Transient("timeout"), REFUSED]})

    answers, _, sleeps, error = run(rows, client, tmp_path, concurrency=1)

    assert error is None and len(answers) == 3 and client.asked("response 001") == 2 and len(sleeps) == 1

    client = FakeCompleter({"response 001": [closed.Transient("timeout")] * 3})
    answers, _, _, error = run(rows, client, tmp_path / "other", concurrency=1)
    assert isinstance(error, closed.Stopped)
    assert "kept failing (timeout); 1 of 3 waiting pairs were stored" in error.message
    assert client.asked("response 001") == 3


def test_an_answer_without_labels_is_stored_as_unscored_and_asked_again_only_on_request(tmp_path):
    rows = gold_rows(4)
    client = FakeCompleter({"response 002": ["I think it is harmful."] * 3})

    answers, lines, _, error = run(rows, client, tmp_path)

    assert error is None and client.asked("response 002") == 3 and answers[rows[2].id] == "I think it is harmful."
    assert lines[-1] == "4 pairs sent, 3 answers parsed"
    # the second attempt carries the format reminder
    assert [call for call in client.calls if "response 002" in call[1]["content"]][1][-1]["content"] == rubric.REMINDER
    cell = next(cell for cell in validate.score(rows, answers) if cell.label == "refusal")
    assert (cell.n, cell.unscored, cell.not_asked) == (3, 1, 0)

    client = FakeCompleter()
    run(rows, client, tmp_path)
    assert client.calls == []
    answers, lines, _, _ = run(rows, client, tmp_path, retry_unscored=True)
    assert len(client.calls) == 1 and answers[rows[2].id] == REFUSED
    assert lines[0] == "4 pairs: 3 answered before, 1 to ask"
    # the later answer of a row wins when the file is read again
    assert validate.read_answers(tmp_path / "answers.jsonl", rows)[rows[2].id] == REFUSED


def test_a_pair_the_route_declines_is_unscored_and_does_not_stop_the_run(tmp_path):
    rows = gold_rows(3)
    client = FakeCompleter({"response 001": [closed.Declined()]})

    answers, _, _, error = run(rows, client, tmp_path)

    assert error is None and answers[rows[1].id] == closed.DECLINED and client.asked("response 001") == 1


def test_no_more_pairs_are_in_flight_than_the_concurrency(tmp_path):
    client = FakeCompleter(delay=0.01)

    run(gold_rows(12), client, tmp_path, concurrency=3)

    assert len(client.calls) == 12 and 1 < client.max_in_flight <= 3


def test_rows_with_a_repeated_id_are_refused_before_any_call(tmp_path):
    rows = gold_rows(3)
    client = FakeCompleter()

    _, _, _, error = run([*rows, rows[1]], client, tmp_path)

    assert error.exit_code is ExitCode.INTEGRITY and "polyguard/en/1 more than once" in error.message
    assert client.calls == []


def test_a_broken_line_in_the_answer_file_is_named(tmp_path):
    store = tmp_path / "answers.jsonl"
    store.write_text('{"id": "polyguard/en/0", "raw": "x"}\n')

    with pytest.raises(DriftmapError, match=r"answers.jsonl:1: not a Answer") as error:
        validate.read_answers(store, gold_rows(1))

    assert error.value.exit_code is ExitCode.INTEGRITY


def test_measures_of_a_known_table():
    found = validate.measures(table(20, 5, 10, 15))

    assert found["n"] == 50 and found["agreement"] == pytest.approx(0.7)
    # chance agreement is 0.5 * 0.6 + 0.5 * 0.4 = 0.5
    assert found["kappa"] == pytest.approx(0.4)
    assert found["precision"] == pytest.approx(20 / 30) and found["recall"] == pytest.approx(0.8)
    assert found["f1"] == pytest.approx(2 * (2 / 3) * 0.8 / (2 / 3 + 0.8))
    assert found["confusion"] == {"yes_yes": 20, "yes_no": 5, "no_yes": 10, "no_no": 15}
    low, high = found["kappa_ci"]
    assert 0.1 < low < 0.4 < high < 0.7


def test_kappa_agrees_with_the_definition_on_unbalanced_tables():
    for counts in [(3, 1, 2, 44), (40, 0, 0, 10), (1, 9, 9, 1), (0, 5, 5, 40)]:
        yes_yes, yes_no, no_yes, no_no = counts
        n = sum(counts)
        observed = (yes_yes + no_no) / n
        chance = ((yes_yes + yes_no) * (yes_yes + no_yes) + (no_yes + no_no) * (yes_no + no_no)) / n**2
        assert validate.measures(table(*counts))["kappa"] == pytest.approx((observed - chance) / (1 - chance))


def test_the_interval_is_seeded_and_brackets_a_perfect_and_a_chance_judge():
    pairs = table(20, 5, 10, 15)

    assert validate.measures(pairs)["kappa_ci"] == validate.measures(pairs)["kappa_ci"]
    assert validate.measures(pairs, seed=7)["kappa_ci"] != validate.measures(pairs)["kappa_ci"]
    assert validate.measures(table(30, 0, 0, 30))["kappa_ci"] == (1.0, 1.0)
    low, high = validate.measures(table(15, 15, 15, 15))["kappa_ci"]
    assert low < 0 < high


def test_measures_without_a_second_class_are_none_not_zero():
    one_word = validate.measures(table(0, 0, 0, 12))
    assert one_word["agreement"] == 1.0 and one_word["kappa"] is None and one_word["kappa_ci"] is None
    assert one_word["precision"] is None and one_word["recall"] is None and one_word["f1"] is None

    missed = validate.measures(table(0, 4, 0, 8))
    assert missed["kappa"] == 0.0 and missed["precision"] is None and missed["recall"] == 0.0 and missed["f1"] is None
    wrong = validate.measures(table(0, 4, 4, 8))
    assert wrong["precision"] == 0.0 and wrong["recall"] == 0.0 and wrong["f1"] == 0.0

    empty = validate.measures([])
    assert empty["n"] == 0 and empty["agreement"] is None and empty["kappa"] is None and empty["kappa_ci"] is None
    # half the resamples of two rows hold one word only, so no interval is given
    assert validate.measures(table(1, 0, 0, 1))["kappa_ci"] is None
    # a few resamples of these twelve rows do, and the interval is over the others
    assert validate.measures(table(3, 0, 0, 9))["kappa_ci"] == (1.0, 1.0)


def test_score_reports_each_source_language_and_label_and_pools_a_source_with_several_languages():
    rows = [
        *gold_rows(4, language="en"),
        *gold_rows(4, language="fr"),
        *gold_rows(3, source="jbb", gold={"response_harm": "yes"}),
    ]
    answers = {row.id: REFUSED for row in rows} | {"polyguard/fr/0": COMPLIED, "jbb/en/1": COMPLIED}

    found = validate.score(rows, answers)

    assert [(cell.source, cell.language, cell.label) for cell in found] == [
        *[("polyguard", language, label) for language in ("en", "fr", "all") for label in validate.FIELDS],
        ("jbb", "en", "response_harm"),
    ]
    by_name = {(cell.source, cell.language, cell.label): cell for cell in found}
    assert by_name["polyguard", "en", "refusal"].agreement == 1.0
    french = by_name["polyguard", "fr", "refusal"]
    assert (french.n, french.agreement, french.confusion["yes_no"]) == (4, 0.75, 1)
    pooled = by_name["polyguard", "all", "response_harm"]
    assert (pooled.n, pooled.agreement, pooled.confusion) == (
        8,
        0.875,
        {"yes_yes": 0, "yes_no": 0, "no_yes": 1, "no_no": 7},
    )
    jailbreak = by_name["jbb", "en", "response_harm"]
    assert (jailbreak.n, jailbreak.recall, jailbreak.precision) == (3, pytest.approx(1 / 3), 1.0)


def test_a_row_without_an_answer_is_not_asked_and_neither_it_nor_an_unscored_row_counts_in_n():
    rows = gold_rows(5)
    answers = {rows[0].id: REFUSED, rows[1].id: REFUSED, rows[2].id: "no object here", rows[3].id: closed.DECLINED}

    found = validate.score(rows, answers)

    assert [(cell.n, cell.unscored, cell.not_asked) for cell in found] == [(2, 2, 1)] * 3
    assert all(cell.agreement == 1.0 for cell in found)


def test_only_rows_carrying_a_label_count_for_it():
    rows = [*gold_rows(2), *gold_rows(3, language="fr", gold={"refusal": "no"})]

    found = validate.score(rows, {row.id: REFUSED for row in rows})

    french = [cell for cell in found if cell.language == "fr"]
    assert [(cell.label, cell.n, cell.agreement) for cell in french] == [("refusal", 3, 0.0)]
    pooled = {cell.label: cell.n for cell in found if cell.language == "all"}
    assert pooled == {"request_harm": 2, "refusal": 5, "response_harm": 2}


def test_render_is_a_table_with_one_line_per_cell_and_marks_what_is_undefined():
    rows = [*gold_rows(4), *gold_rows(2, source="jbb", gold={"response_harm": "yes"})]
    answers = {row.id: REFUSED for row in rows[:5]}

    text = validate.render(validate.score(rows, answers))

    lines = text.splitlines()
    assert len(lines) == 2 + 4 and text.endswith("|\n")
    assert len({line.count("|") for line in lines}) == 1
    assert lines[2].startswith("| polyguard | en | request_harm | 4 | 100.0% | n/a | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 |")
    assert lines[-1] == "| jbb | en | response_harm | 1 | 0.0% | 0.00 | n/a | 0.00 | n/a | 0 | 1 | 0 | 0 | 0 | 1 |"
    known = validate.render(validate.score(gold_rows(1), {}))
    assert "| 0 | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 0 | 0 | 0 | 1 |" in known


def test_render_shows_the_interval_beside_kappa():
    rows = [*gold_rows(20), *gold_rows(20, language="fr", gold={"refusal": "no"})]
    answers = {row.id: REFUSED if index % 4 else BENIGN for index, row in enumerate(rows)}

    pooled = next(cell for cell in validate.score(rows, answers) if (cell.language, cell.label) == ("all", "refusal"))

    low, high = pooled.kappa_ci
    assert f"| {pooled.kappa:.2f} ({low:.2f} to {high:.2f}) |" in validate.render([pooled])


def test_retest_rows_take_the_stored_labels_as_the_reference():
    stored = [
        {
            "condition": "served",
            "model": "baseline",
            "checkpoint_slug": "tiny-aya-global--af89d219b53e",
            "set": "harmful",
            "language": "ar",
            "prompt_id": "00040c97",
            "prompt": "request 000",
            "response": "response 000",
            "stored_request_harm": "yes",
            "stored_refusal": "yes",
            "stored_response_harm": "no",
        },
        {
            "condition": "no-paragraph",
            "model": "fine-tuned",
            "checkpoint_slug": "tiny-aya-global-drift-merged--daccc14f6c19",
            "set": "harmful",
            "language": "ar",
            "prompt_id": "00040c97",
            "prompt": "request 000",
            "response": "response 001",
            "stored_request_harm": "unscored",
            "stored_refusal": "unscored",
            "stored_response_harm": "unscored",
        },
        {"language": "en", "prompt_id": "1", "prompt": "request 002", "response": None, "stored_refusal": "yes"},
    ]

    rows = validate.retest_rows(stored)

    assert [row.id for row in rows] == [
        "retest/tiny-aya-global--af89d219b53e/served/ar/00040c97",
        "retest/tiny-aya-global-drift-merged--daccc14f6c19/no-paragraph/ar/00040c97",
    ]
    assert rows[0].gold == {"request_harm": "yes", "refusal": "yes", "response_harm": "no"} and rows[1].gold == {}
    assert rows[0].meta == {"condition": "served", "model": "baseline", "set": "harmful"}
    assert (rows[0].source, rows[0].language, rows[0].response) == ("retest", "ar", "response 000")
    found = validate.score(rows, {rows[0].id: answer("yes", "no", "no"), rows[1].id: REFUSED})
    assert [(cell.label, cell.n, cell.agreement) for cell in found] == [
        ("request_harm", 1, 1.0),
        ("refusal", 1, 0.0),
        ("response_harm", 1, 1.0),
    ]


def test_select_keeps_the_rows_whose_meta_holds_every_value():
    completions = {model: xstest_records(model) for model in validate.XSTEST_MODELS}
    rows = validate.over_refusal(completions, 90, validate.SEED)

    full = validate.select(rows, {"model": "gpt4", "annotators_agree": "yes"})

    assert len(full) == 17 and all(row.meta["final_label"] != "3_partial_refusal" for row in full)
    assert validate.select(rows, {}) == rows and validate.select(rows, {"model": "none"}) == []


def test_rows_survive_a_round_trip_through_the_file_form(tmp_path):
    from driftmap.judge.incremental import dump_rows, read_rows

    rows, _ = validate.multilingual(polyguard_records(), 12, validate.SEED)
    path = tmp_path / "sample.jsonl"
    path.write_bytes(dump_rows(rows))

    assert read_rows(path, GoldRow) == rows
    first = json.loads(path.read_text().splitlines()[0])
    assert list(first) == ["id", "source", "language", "prompt", "response", "gold", "meta"]
