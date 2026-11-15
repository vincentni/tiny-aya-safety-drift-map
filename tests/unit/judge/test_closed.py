import json
import logging
import re
import signal
import sys
import threading
import time

import httpx
import pytest

from driftmap.core import ids
from driftmap.core.hub import Hub
from driftmap.core.records import LabelRow
from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.table.build import build
from driftmap.judge import closed, reparse, rubric
from driftmap.judge.incremental import RawRow, judged_rows
from tests.fakes import FakeClock
from tests.fakes.judge_world import (
    EVAL,
    MODEL,
    OWNER,
    RESPONSES,
    UNPARSEABLE,
    JudgeWorld,
    generation_rows,
    jsonl,
    other_judge,
)
from tests.fakes.labels import write_labels
from tests.fakes.settings import BASE

# through the module, so the library's start-up download stays off
litellm = closed.library()
REFUSED = '{"request_harm": "yes", "refusal": "yes", "response_harm": "no"}'
COMPLIED = '{"request_harm": "yes", "refusal": "no", "response_harm": "yes"}'
BENIGN = '{"request_harm": "no", "refusal": "no", "response_harm": "no"}'
ROUTE = "gateway-x/judge-route-a"
NAME = BASE["judge"]["closed"]["name"]


class FakeCompleter:
    """Answers each pair with the next scripted outcome for a text its user turn holds, else ``default``."""

    def __init__(self, script=None, default=REFUSED, delay=0.0, on_call=None):
        self.script = {text: list(outcomes) for text, outcomes in (script or {}).items()}
        self.default, self.delay, self.on_call = default, delay, on_call
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
            if self.on_call:
                self.on_call(len(self.calls))
            time.sleep(self.delay)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            with self._lock:
                self.in_flight -= 1

    def asked(self, text):
        return sum(text in messages[1]["content"] for messages in self.calls)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return JudgeWorld(tmp_path, monkeypatch, n=4)


def judge(world, client, **options):
    lines, clock = [], FakeClock()
    options = {"mode": world.mode, "echo": lines.append, "sleep": clock.sleep} | options
    settings = load_settings(world.config)
    hub = Hub(world.fake, gated_repos=settings.gated_repos)
    try:
        closed.judge_checkpoint(settings, hub, client, options.pop("checkpoint", f"{MODEL}@{'1' * 40}"), **options)
        error = None
    except DriftmapError as stopped:
        error = stopped
    return lines, clock, error


def plain(messages):
    """The chat without the random marker in its tags."""
    return [{**turn, "content": re.sub(r"-[0-9a-f]{16}>", ">", turn["content"])} for turn in messages]


def uploads(world):
    return [kwargs["path_in_repo"] for method, kwargs in world.fake.calls if method == "upload_file"]


def test_every_unit_is_labelled_under_the_closed_config_in_the_evaluation_dataset(world):
    client = FakeCompleter({"harmful response en 001": [COMPLIED], "xstest response": [BENIGN] * 4})

    lines, _, error = judge(world, client)

    assert error is None and len(client.calls) == 28
    config = rubric.judge_config(load_settings(world.config))
    assert (config.repo, config.revision, config.decoding.runtime) == (NAME, rubric.prompt_sha256(), "api")
    labels = world.labels("en")
    assert len(labels) == 8 and all(row.judge == config for row in labels)
    harmful = [row for row in labels if row.set == "harmful"]
    assert [(row.refusal, row.response_harm) for row in harmful] == [("yes", "no"), ("no", "yes")] + [("yes", "no")] * 2
    assert {row.request_harm for row in labels if row.set == "xstest"} == {"no"}
    raws = world.raws("en")
    assert len(raws) == 8 and {(raw.repo, raw.revision) for raw in raws} == {(NAME, config.revision)}
    assert {raw.raw for raw in raws} == {REFUSED, COMPLIED, BENIGN}
    assert world.paths(EVAL) == [f"labels/{world.slug}/{code}.jsonl" for code in sorted(BASE["languages"])]
    first = client.calls[0]
    assert first[0] == {"role": "system", "content": rubric.RUBRIC} and first[1]["content"].startswith("<request-")
    assert lines[0].split() == ["harmful/en", "scored", "4", "unscored", "0", "safe", "3", "refused", "3", "new", "4"]
    assert lines[6].split() == ["xstest/en", "scored", "4", "unscored", "0", "safe", "4", "refused", "0", "new", "4"]
    assert lines[-1] == "28 pairs sent, 28 answers parsed"
    # the answer file is there before a label says so
    order = uploads(world)
    assert order.index(f"judge_raw/{world.slug}/en.jsonl") < order.index(f"labels/{world.slug}/en.jsonl")


@pytest.mark.parametrize(("preamble", "tree"), [("strip", "nopreamble/"), ("patch", "patched/")])
def test_the_responses_of_another_prompt_condition_are_labelled_inside_its_own_tree(world, preamble, tree):
    for (set_, language), rows in world.rows.items():
        world.put(ids.responses_path(world.slug, set_, language, "full", preamble), jsonl(rows), RESPONSES)
    codes = sorted(BASE["languages"])

    _, _, error = judge(world, FakeCompleter(), preamble=preamble)

    assert error is None
    assert world.paths(EVAL) == [f"{tree}labels/{world.slug}/{code}.jsonl" for code in codes]
    answers = [path for path in world.paths(RESPONSES) if "judge_raw" in path]
    assert answers == [f"{tree}judge_raw/{world.slug}/{code}.jsonl" for code in codes]

    # the served responses wait for a run of their own, which leaves this tree as it is
    before = {path: world.stored(path, EVAL) for path in world.paths(EVAL)}
    client = FakeCompleter()
    assert judge(world, client)[2] is None and len(client.calls) == 28
    assert {path: world.stored(path, EVAL) for path in before} == before
    assert world.paths(EVAL) == sorted([*before, *(f"labels/{world.slug}/{code}.jsonl" for code in codes)])


VARIANT_CODES = ("hi-Latn", "ru-mixed")


def add_variant_responses(world, preamble="keep"):
    """Store what a variants run leaves: harmful and benign responses per variant code."""
    for set_ in ("harmful", "benign"):
        for code in VARIANT_CODES:
            path = ids.responses_path(world.slug, set_, code, world.mode, preamble, variants=True)
            world.put(path, jsonl(generation_rows(set_, code, 4)), RESPONSES)


@pytest.mark.parametrize(("preamble", "tree"), [("keep", ""), ("patch", "patched/"), ("strip", "nopreamble/")])
def test_the_harmful_variant_responses_are_labelled_into_the_variants_folder_of_their_tree(world, preamble, tree):
    add_variant_responses(world, preamble)
    client = FakeCompleter({"harmful response ru-mixed 001": [COMPLIED]})

    lines, _, error = judge(world, client, sets=["variants"], preamble=preamble)

    # the benign variants and the standard sets are not this run's
    assert error is None and len(client.calls) == 8
    folder = f"{tree}variants/"
    assert world.paths(EVAL) == [f"{folder}labels/{world.slug}/{code}.jsonl" for code in VARIANT_CODES]
    answers = [path for path in world.paths(RESPONSES) if "judge_raw" in path]
    assert answers == [f"{folder}judge_raw/{world.slug}/{code}.jsonl" for code in VARIANT_CODES]
    labels = world._rows(world.stored(f"{folder}labels/{world.slug}/ru-mixed.jsonl", EVAL), LabelRow)
    assert {(row.set, row.language) for row in labels} == {("harmful", "ru-mixed")}
    assert [row.response_harm for row in labels] == ["no", "yes", "no", "no"]
    assert [line.split()[0] for line in lines[:-1]] == ["harmful/hi-Latn", "harmful/ru-mixed"]
    assert lines[-1] == "8 pairs sent, 8 answers parsed"

    again = FakeCompleter()
    assert judge(world, again, sets=["variants"], preamble=preamble)[2] is None and again.calls == []


def test_the_variants_are_judged_alone_and_only_once_their_responses_are_stored(world):
    _, _, missing = judge(world, FakeCompleter(), sets=["variants"])
    assert missing.exit_code == 12 and "no harmful variant responses" in str(missing)
    add_variant_responses(world)
    client = FakeCompleter()
    _, _, mixed = judge(world, client, sets=["harmful", "variants"])
    assert mixed.exit_code == 2 and "alone" in str(mixed)
    # a variants file that holds a row of another code is not labelled
    stray = ids.responses_path(world.slug, "harmful", "ar-Latn", world.mode, variants=True)
    world.put(stray, jsonl(generation_rows("harmful", "hi-Latn", 2)), RESPONSES)
    _, _, refused = judge(world, client, sets=["variants"])
    assert refused.exit_code == 12 and "holds a row of" in str(refused)
    assert client.calls == [] and world.paths(EVAL) == []


def test_the_heldout_prompts_are_labelled_into_a_folder_of_their_own_under_either_condition(world):
    codes = ("en", "hi")
    for preamble in ("keep", "patch"):
        for code in codes:
            path = ids.responses_path(world.slug, "harmful", code, world.mode, preamble, variants="heldout")
            world.put(path, jsonl(generation_rows("harmful", code, 3)), RESPONSES)
    add_variant_responses(world)

    for preamble, tree in (("keep", ""), ("patch", "patched/")):
        client = FakeCompleter({"harmful response hi 001": [COMPLIED]})
        lines, _, error = judge(world, client, sets=["heldout"], preamble=preamble)

        assert error is None and len(client.calls) == 6
        folder = f"{tree}heldout/"
        assert [path for path in world.paths(EVAL) if path.startswith(folder)] == [
            f"{folder}labels/{world.slug}/{code}.jsonl" for code in codes
        ]
        labels = world._rows(world.stored(f"{folder}labels/{world.slug}/hi.jsonl", EVAL), LabelRow)
        assert {(row.set, row.language) for row in labels} == {("harmful", "hi")}
        assert [row.response_harm for row in labels] == ["no", "yes", "no"]
        assert lines[-1] == "6 pairs sent, 6 answers parsed"
    # the variants and the evaluation sets are other runs' to label
    assert not [path for path in world.paths(EVAL) if "heldout" not in path]
    _, _, mixed = judge(world, FakeCompleter(), sets=["heldout", "variants"])
    assert mixed.exit_code == 2 and "alone" in str(mixed)
    _, _, missing = judge(world, FakeCompleter(), sets=["heldout"], preamble="strip")
    assert missing.exit_code == 12 and "no harmful heldout responses" in str(missing)


def test_a_second_run_asks_nothing_and_uploads_nothing(world):
    judge(world, FakeCompleter())
    before = len(uploads(world))
    client = FakeCompleter()

    lines, _, error = judge(world, client)

    assert error is None and client.calls == [] and len(uploads(world)) == before
    assert lines[0].split()[-2:] == ["new", "0"] and lines[-1] == "0 pairs sent, 0 answers parsed"


def test_a_lost_label_file_is_rebuilt_from_the_stored_answers_and_an_unchanged_file_is_not_uploaded(world):
    labels, raws = f"labels/{world.slug}/en.jsonl", f"judge_raw/{world.slug}/en.jsonl"
    judge(world, FakeCompleter())
    world.drop(labels, EVAL)
    before = len(uploads(world))

    again = FakeCompleter()
    _, _, error = judge(world, again)

    # the stored answers give the labels back: no pair is sent, the label file is written once per unit in it
    assert error is None and again.calls == [] and uploads(world)[before:] == [labels, labels]
    assert len(world.labels("en")) == 8
    world.drop(raws, RESPONSES)
    before = len(uploads(world))

    _, _, error = judge(world, FakeCompleter())

    assert error is None and uploads(world)[before:] == [raws, raws]
    assert len(world.raws("en")) == 8


def test_labels_of_another_judge_are_replaced(world):
    world.seed(other_judge(rubric.judge_config(load_settings(world.config))))
    assert {row.judge.repo for row in world.labels("fr")} == {"closed-model-b"}
    client = FakeCompleter()

    _, _, error = judge(world, client)

    assert error is None and len(client.calls) == 28
    assert {row.judge.repo for code in BASE["languages"] for row in world.labels(code)} == {NAME}
    assert {raw.repo for raw in world.raws("fr")} == {NAME} and len(world.labels("fr")) == 4


def test_a_stopped_unit_counts_only_its_own_labels_and_keeps_the_other_judge_for_the_rest(world):
    world.seed(other_judge(rubric.judge_config(load_settings(world.config))))
    throttled = closed.Throttled("the judge route is throttling or out of quota")

    lines, _, error = judge(world, FakeCompleter({"harmful response es 002": [throttled]}), concurrency=1)

    assert isinstance(error, closed.Throttled)
    assert lines[2] == "harmful/es   scored 2  unscored 0  safe 2  refused 2  new 2"
    assert [row.judge.repo for row in world.labels("es")] == [NAME] * 2 + ["closed-model-b"] * 2


def test_an_answer_that_never_parses_is_an_unscored_row_after_three_attempts(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en",))
    script = {"harmful response en 003": ["I think yes.", "```json\n{}\n```", "not sure"]}
    client = FakeCompleter(script | {"harmful response en 007": ["no JSON", COMPLIED]})

    lines, clock, error = judge(world, client, sets=("harmful",))

    assert error is None
    assert (client.asked("harmful response en 003"), client.asked("harmful response en 007")) == (3, 2)
    assert len(clock.sleeps) == 3
    labels = {row.prompt_id: row for row in world.labels("en")}
    lost = labels["harmful-en-003"]
    assert (lost.request_harm, lost.refusal, lost.response_harm, lost.raw_output_present) == ("unscored",) * 3 + (True,)
    assert {raw.prompt_id: raw.raw for raw in world.raws("en")}["harmful-en-003"] == "not sure"
    assert labels["harmful-en-007"].response_harm == "yes"
    assert lines[0] == "harmful/en   scored 19  unscored 1  safe 18  refused 18  new 20"


def test_retry_unscored_asks_only_the_unscored_pairs_again_and_keeps_the_latest_answer(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en",))
    labels, raws = f"labels/{world.slug}/en.jsonl", f"judge_raw/{world.slug}/en.jsonl"
    lost = "harmful response en 003"
    judge(world, FakeCompleter({lost: ["first", "second", "third"]}), sets=("harmful",))
    plain = FakeCompleter()
    judge(world, plain, sets=("harmful",))
    assert plain.calls == []
    before = len(uploads(world))
    client = FakeCompleter({lost: ["fourth", "fifth", "sixth"]})

    lines, _, error = judge(world, client, sets=("harmful",), retry_unscored=True)

    assert error is None and len(client.calls) == client.asked(lost) == 3
    assert {raw.prompt_id: raw.raw for raw in world.raws("en")}["harmful-en-003"] == "sixth"
    # the label still reads unscored: only the answer file changes
    assert uploads(world)[before:] == [raws]
    assert lines[0] == "harmful/en   scored 19  unscored 1  safe 19  refused 19  new 1"
    client = FakeCompleter({lost: [COMPLIED]})

    lines, _, error = judge(world, client, sets=("harmful",), retry_unscored=True)

    assert error is None and len(client.calls) == 1
    label = {row.prompt_id: row for row in world.labels("en")}["harmful-en-003"]
    assert (label.refusal, label.response_harm) == ("no", "yes")
    assert {raw.prompt_id: raw.raw for raw in world.raws("en")}["harmful-en-003"] == COMPLIED
    assert uploads(world)[before + 1 :] == [raws, labels]
    assert lines[0] == "harmful/en   scored 20  unscored 0  safe 19  refused 19  new 1"
    done = FakeCompleter()
    judge(world, done, sets=("harmful",), retry_unscored=True)
    assert done.calls == []


def test_a_format_reminder_follows_an_answer_with_text_but_no_object(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en",))
    empty = closed.EMPTY.format(reason="content_filter")
    talk, cut = "harmful response en 003", "harmful response en 007"
    client = FakeCompleter({talk: ["The request is ambiguous.", empty, REFUSED], cut: [" \n", COMPLIED]})

    _, _, error = judge(world, client, sets=("harmful",), concurrency=1)

    assert error is None
    first, second, third = (messages for messages in client.calls if talk in messages[1]["content"])
    asked = world.rows["harmful", "en"][3]
    assert plain(first) == plain(rubric.messages(asked.prompt, asked.response))
    assert plain(second) == [*plain(first), {"role": "user", "content": "Answer with the JSON object only."}]
    # an answer without text gets the plain pair again; every call has its own marker
    assert plain(third) == plain(first) and sum(len(messages) == 3 for messages in client.calls) == 1
    assert len({messages[1]["content"].split("\n")[0] for messages in client.calls}) == len(client.calls)
    config = rubric.judge_config(load_settings(world.config))
    assert all(row.judge == config and row.response_harm != "unscored" for row in world.labels("en"))


def test_a_unit_over_its_unscored_allowance_keeps_its_parsed_answers_and_holds_the_rest(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en", "fr"))
    lost = [f"harmful response fr 00{index}" for index in (1, 2)]
    token = "hf_" + "a1B2" * 9

    lines, _, error = judge(world, FakeCompleter({text: [f"? {token}"] * 3 for text in lost}))

    assert error is not None and error.exit_code == 4
    assert "harmful/fr" in str(error) and "more than 5%" in str(error) and "--retry-unscored" in str(error)
    assert len(world.labels("en")) == 40 and len(world.labels("fr")) == len(world.raws("fr")) == 18
    assert {row.response_harm for row in world.labels("fr")} == {"no"}
    assert lines[1] == "harmful/fr   scored 18  unscored 0  safe 18  refused 18  new 18  held 2"
    assert "did not parse" in lines[2] and "hf_***" in lines[2] and token not in lines[2]
    # the next run asks only the pairs without a parsed answer, and holds them again
    again = FakeCompleter({text: ["?"] * 3 for text in lost})
    lines, _, error = judge(world, again)
    assert error is not None and len(again.calls) == 6 and len(world.labels("fr")) == 18
    assert lines[1].endswith("new 0  held 2")
    # with the flag what stays unscored is stored and counted
    final = FakeCompleter({text: ["?"] * 3 for text in lost})
    lines, _, error = judge(world, final, retry_unscored=True)
    assert error is None and len(final.calls) == 6 and len(world.labels("fr")) == len(world.raws("fr")) == 20
    assert lines[1] == "harmful/fr   scored 18  unscored 2  safe 18  refused 18  new 2"


def test_a_small_unit_may_store_one_unscored_row(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=10, languages=("en",))

    lines, _, error = judge(world, FakeCompleter({"harmful response en 004": ["?"] * 3}), sets=("harmful",))

    assert error is None and len(world.labels("en")) == 10
    assert lines[0] == "harmful/en   scored 9  unscored 1  safe 9  refused 9  new 10"


def test_rows_without_a_response_are_never_sent_and_never_count_against_the_unit(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en",))
    rows = world.rows["harmful", "en"]
    silent = [row.model_copy(update={"response": None, "status": "unscored", "error": "empty"}) for row in rows[:2]]
    world.put_rows("harmful", "en", [*silent, *rows[2:]])
    client = FakeCompleter({"harmful response en 009": ["?"] * 3})

    lines, _, error = judge(world, client, sets=("harmful",))

    assert error is None and len(client.calls) == 17 + 3 and len(world.labels("en")) == 20
    assert lines[0] == "harmful/en   scored 17  unscored 3  safe 17  refused 17  new 20"
    assert [row.raw_output_present for row in world.labels("en")[:3]] == [False, False, True]
    again = FakeCompleter()
    assert judge(world, again, sets=("harmful",))[2] is None and again.calls == []


def test_the_unscored_allowance_is_a_share_of_the_pairs_sent_not_of_the_file(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=40, languages=("en",))
    rows = world.rows["harmful", "en"]
    silent = [row.model_copy(update={"response": None, "status": "unscored", "error": "empty"}) for row in rows[:20]]
    world.put_rows("harmful", "en", [*silent, *rows[20:]])
    client = FakeCompleter({f"harmful response en 03{index}": ["?"] * 3 for index in (1, 2)})

    lines, _, error = judge(world, client, sets=("harmful",))

    # two of twenty sent is over the allowance; two of forty rows would not be
    assert error is not None and error.exit_code == 4 and lines[0].endswith("new 38  held 2")
    assert len(world.labels("en")) == 38 and len(world.raws("en")) == 18


def test_an_interrupt_stops_the_unit_and_stores_what_was_answered(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=40, languages=("en",))
    main = threading.main_thread().ident
    # a real signal: it wakes the thread that waits for the answers
    client = FakeCompleter(delay=0.02, on_call=lambda count: count == 3 and signal.pthread_kill(main, signal.SIGINT))

    with pytest.raises(KeyboardInterrupt):
        judge(world, client, sets=("harmful",), concurrency=2)

    assert 3 <= len(client.calls) < 40
    assert len(world.labels("en")) == len(world.raws("en")) == len(client.calls)
    again = FakeCompleter()
    _, _, error = judge(world, again, sets=("harmful",))
    assert error is None and len(again.calls) == 40 - len(client.calls) and len(world.labels("en")) == 40


def test_an_unexpected_error_stores_the_answers_before_it_is_passed_on(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en",))
    client = FakeCompleter({"harmful response en 005": [RuntimeError("boom")]})

    with pytest.raises(RuntimeError, match="boom"):
        judge(world, client, sets=("harmful",), concurrency=1)

    assert len(client.calls) == 6 and len(world.labels("en")) == len(world.raws("en")) == 5


def test_rows_another_writer_added_to_a_file_during_the_run_are_kept(world):
    config = rubric.judge_config(load_settings(world.config))
    other = world.rows["harmful", "en"][0].model_copy(update={"prompt_id": "harmful-en-900"})
    foreign, _ = judged_rows([other], [(REFUSED, rubric.parse(REFUSED))], other_judge(config))
    path = f"labels/{world.slug}/en.jsonl"
    client = FakeCompleter(on_call=lambda count: count == 1 and world.put(path, jsonl(foreign), EVAL))

    _, _, error = judge(world, client, concurrency=1)

    assert error is None and len(world.labels("en")) == 9
    kept = {row.prompt_id: row for row in world.labels("en")}["harmful-en-900"]
    assert kept == foreign[0] and len(world.labels("fr")) == 4
    # a file nobody else touched is not read again
    downloads = [kwargs for method, kwargs in world.fake.calls if method == "snapshot_download"]
    assert len(downloads) == 3 + 1


def test_the_download_folders_are_removed_when_the_run_ends(world, tmp_path):
    judge(world, FakeCompleter({"harmful response fr 001": [closed.Throttled("stop")]}), concurrency=1)
    put_screen(world, "0.1")
    judge_rows(world, FakeCompleter(), f"{SCREEN}/alpha-0.1.jsonl")

    assert list((tmp_path / "scratch").glob("driftmap-closed-*")) == []


def test_throttling_stops_the_run_without_a_retry_and_keeps_what_is_finished(world):
    throttled = closed.Throttled("the judge route is throttling or out of quota")
    client = FakeCompleter({"harmful response es 000": ["?"] * 3, "harmful response es 002": [throttled]})

    lines, clock, error = judge(world, client, concurrency=1)

    assert isinstance(error, closed.Throttled) and error.exit_code == 4
    assert "1 of 4 waiting pairs of harmful/es were stored; run the command again" in str(error)
    assert client.asked("harmful response es 002") == 1 and len(clock.sleeps) == 2
    assert len(client.calls) == 4 + 4 + 3 + 2
    # a unit cut short keeps its parsed answers only
    assert [len(world.labels(code)) for code in ("en", "fr", "es", "ru")] == [4, 4, 1, 0]
    assert [len(world.raws(code)) for code in ("en", "fr", "es", "ru")] == [4, 4, 1, 0]
    # the rest is picked up by the next run
    client = FakeCompleter()
    _, _, error = judge(world, client)
    assert error is None and len(client.calls) == 28 - 9
    assert [len(world.labels(code)) for code in BASE["languages"]] == [8, 4, 4, 4, 4, 4]


def test_a_passing_fault_is_tried_again_and_a_lasting_one_stops_the_run(world, tmp_path, monkeypatch):
    flaky = [closed.Transient("timeout"), closed.Transient("server error"), COMPLIED]
    client = FakeCompleter({"harmful response en 000": flaky})

    _, clock, error = judge(world, client, sets=("xstest", "harmful"), concurrency=1)

    assert error is None and client.asked("harmful response en 000") == 3 and len(clock.sleeps) == 2
    assert world.labels("en")[0].response_harm == "yes"

    down = FakeCompleter({"harmful response fr 001": [closed.Transient("connection")] * 3})
    world = JudgeWorld(tmp_path / "down", monkeypatch, n=4)
    _, _, error = judge(world, down, concurrency=1)
    assert type(error) is closed.Stopped and error.exit_code == 4 and "kept failing (connection)" in str(error)
    assert down.asked("harmful response fr 001") == 3 and len(world.labels("fr")) == 1


def test_a_pair_the_route_declines_is_an_unscored_row(tmp_path, monkeypatch):
    world = JudgeWorld(tmp_path, monkeypatch, n=20, languages=("en",))
    client = FakeCompleter({"harmful response en 005": [closed.Declined()]})

    _, clock, error = judge(world, client, sets=("harmful",))

    assert error is None and client.asked("harmful response en 005") == 1 and clock.sleeps == []
    assert {row.prompt_id: row for row in world.labels("en")}["harmful-en-005"].response_harm == "unscored"
    assert {raw.prompt_id: raw.raw for raw in world.raws("en")}["harmful-en-005"] == closed.DECLINED


@pytest.mark.parametrize(("concurrency", "expected"), [(1, 1), (3, 3)])
def test_no_more_pairs_are_in_flight_than_the_concurrency(world, concurrency, expected):
    client = FakeCompleter(delay=0.05)

    _, _, error = judge(world, client, sets=("xstest",), concurrency=concurrency)

    assert error is None and client.max_in_flight == expected


def test_sets_and_limit_narrow_the_run_and_a_gguf_checkpoint_is_read_by_its_slug(tmp_path, monkeypatch):
    checkpoint = f"vincentnkq/tiny-aya-global-repaired-GGUF@{'2' * 40}#model-q4_k_m.gguf"
    world = JudgeWorld(tmp_path, monkeypatch, n=6, mode="smoke", checkpoint=checkpoint)
    client = FakeCompleter(default=UNPARSEABLE[0])

    lines, _, error = judge(world, client, sets=("xstest",), limit=2, checkpoint=checkpoint)

    assert error is not None and "xstest/en" in str(error)
    assert len(client.calls) == 2 * 3 and world.paths(EVAL) == []
    client = FakeCompleter()
    lines, _, error = judge(world, client, sets=("xstest",), limit=2, checkpoint=checkpoint)
    assert error is None and len(client.calls) == 2
    assert world.paths(EVAL) == [f"smoke/labels/tiny-aya-global-repaired-GGUF--{'2' * 12}/en.jsonl"]
    assert [row.checkpoint for row in world.labels("en")] == [checkpoint] * 2


def test_a_checkpoint_without_stored_responses_is_refused_before_any_pair_is_sent(world):
    client, before = FakeCompleter(), uploads(world)

    _, _, error = judge(world, client, checkpoint=f"{MODEL}@{'3' * 40}")

    assert error is not None and error.exit_code == 12 and "missing or empty" in str(error)
    assert client.calls == [] and uploads(world) == before


def test_a_repeated_prompt_in_one_responses_file_stops_the_run_before_any_pair_is_sent(world):
    """A responses file with one prompt id twice exits 12 naming the file, and nothing is uploaded (TS-EDGE-4)."""
    rows = world.rows["harmful", "ru"]
    world.put_rows("harmful", "ru", [*rows, rows[1]])
    client, before = FakeCompleter(), uploads(world)

    _, _, error = judge(world, client)

    assert error is not None and error.exit_code == 12
    assert f"responses/{world.slug}/harmful/ru.jsonl has prompt harmful-ru-001 more than once" in str(error)
    assert client.calls == [] and uploads(world) == before


def test_a_row_of_another_unit_in_a_responses_file_stops_the_run_before_any_pair_is_sent(world):
    stray = world.rows["harmful", "hi"][0]
    world.put_rows("harmful", "ru", [*world.rows["harmful", "ru"][:-1], stray])
    client, before = FakeCompleter(), uploads(world)

    _, _, error = judge(world, client)

    assert error is not None and error.exit_code == 12
    assert f"responses/{world.slug}/harmful/ru.jsonl holds a row of {stray.checkpoint} harmful/hi" in str(error)
    assert client.calls == [] and uploads(world) == before


def test_a_smoke_run_on_the_smoke_drift_checkpoint_builds_into_a_smoke_table(tmp_path, monkeypatch):
    merged = f"{OWNER}/tiny-aya-global-drift-merged-smoke@{'a' * 40}"
    world = JudgeWorld(tmp_path, monkeypatch, mode="smoke", checkpoint=merged)

    _, _, error = judge(world, FakeCompleter(), checkpoint=merged)

    assert error is None and world.slug == f"tiny-aya-global-drift-merged-smoke--{'a' * 12}"
    settings = load_settings(world.config)
    labels = [row for language in settings.languages for row in world.labels(language)]
    assert {row.checkpoint for row in labels} == {merged}
    assert world.paths(EVAL) == sorted(f"smoke/labels/{world.slug}/{code}.jsonl" for code in settings.languages)
    assert not any(path.startswith(("labels/", "judge_raw/", "responses/")) for path in world.paths(RESPONSES))
    targets = {
        name: {language: [row.prompt_id for row in world.rows[name, language]] for language in codes}
        for name, codes in (("harmful", settings.languages), ("xstest", ["en"]))
    }
    write_labels(tmp_path / "labels", labels)
    tables = build(tmp_path / "labels", targets, [], settings, mode="smoke")
    assert tables.columns["drifted"].state == "tabled" and tables.judge.repo == NAME
    assert {row.checkpoint for row in tables.safety} == {merged}
    with pytest.raises(DriftmapError, match="unknown checkpoint"):
        build(tmp_path / "labels", targets, [], settings)


class Reply:
    def __init__(self, content, finish="stop"):
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message, "finish_reason": finish})()]
        self.usage = type("Usage", (), {"prompt_tokens": 200, "completion_tokens": 20})()


def test_the_route_comes_from_the_environment_and_goes_to_litellm_only(monkeypatch):
    seen = []
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: seen.append(kwargs) or Reply(REFUSED))
    monkeypatch.delenv(closed.ROUTE_ENV, raising=False)
    with pytest.raises(DriftmapError, match=closed.ROUTE_ENV) as unset:
        closed.from_env()
    assert unset.value.exit_code == 2

    monkeypatch.setenv(closed.ROUTE_ENV, f"  {ROUTE}\n")
    client = closed.from_env()
    messages = rubric.messages("request", "response")

    assert client.complete(messages) == REFUSED
    assert seen == [
        {"model": ROUTE, "messages": messages, "max_tokens": 1024, "timeout": 60, "num_retries": 0, "max_retries": 0}
    ]
    assert (client.prompt_tokens, client.completion_tokens) == (200, 20)
    assert ROUTE not in repr(client) and ROUTE not in str(client)
    bare = Reply(REFUSED)
    bare.choices = []
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: bare)
    assert client.complete(messages) == "[empty answer: no choices]"


def test_an_empty_answer_names_its_finish_reason_in_the_note_of_a_held_unit(world, monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: Reply(None, "length"))
    client = closed.LiteLLMCompleter(ROUTE)

    assert client.complete(rubric.messages("request", "response")) == "[empty answer: length]"
    lines, _, error = judge(world, client, sets=("xstest",))

    assert error is not None and error.exit_code == 4
    assert lines[0].endswith("new 0  held 4") and lines[1].count('"[empty answer: length]"') == 3
    assert world.labels("en") == [] and world.raws("en") == []


DENIED = httpx.Response(403, request=httpx.Request("POST", "https://gateway-x.invalid/v1"))


def fault(cls, **extra):
    fields = {"message": f"{ROUTE} failed in zone-9 for tenant-abc", "llm_provider": "gateway-x"}
    return cls(model=ROUTE, **fields, **extra)


@pytest.mark.parametrize(
    ("raised", "expected", "word"),
    [
        (fault(litellm.RateLimitError), closed.Throttled, "throttling or out of quota"),
        (litellm.BudgetExceededError(2.0, 1.0, message=f"{ROUTE} over budget"), closed.Throttled, "out of quota"),
        (fault(litellm.APIError, status_code=429), closed.Throttled, "throttling"),
        (fault(litellm.Timeout), closed.Transient, "timeout"),
        (fault(litellm.APIConnectionError), closed.Transient, "connection"),
        (fault(litellm.InternalServerError), closed.Transient, "server error"),
        (fault(litellm.ServiceUnavailableError), closed.Transient, "server error"),
        (fault(litellm.APIError, status_code=502), closed.Transient, "server error"),
        (fault(litellm.ContentPolicyViolationError), closed.Declined, ""),
        (fault(litellm.ContextWindowExceededError), closed.Declined, ""),
        (fault(litellm.AuthenticationError), DriftmapError, "refused the call (authentication)"),
        (fault(litellm.PermissionDeniedError, response=DENIED), DriftmapError, "refused the call (permission)"),
        (fault(litellm.NotFoundError), DriftmapError, "refused the call (not found)"),
        (fault(litellm.BadRequestError), DriftmapError, "refused the call (bad request)"),
        (ValueError(f"unknown model {ROUTE}"), DriftmapError, "refused the call (error)"),
    ],
)
def test_route_faults_are_sorted_into_fixed_words_and_never_quote_the_route(monkeypatch, raised, expected, word):
    def failing(**kwargs):
        raise raised

    monkeypatch.setattr(litellm, "completion", failing)
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)

    with pytest.raises(expected) as caught:
        closed.from_env().complete(rubric.messages("request", "response"))

    error = caught.value
    assert type(error) is expected or expected is DriftmapError and not isinstance(error, closed.Throttled)
    for secret in (ROUTE, "gateway-x", "judge-route", "zone-9", "tenant-abc"):
        assert secret not in str(error) and secret not in repr(error)
    assert error.__cause__ is None and error.__suppress_context__
    # a fixed word per category: the library's class names say who serves the route
    assert str(error) == word or word and word in str(error)
    assert type(raised).__name__ not in str(error) and "Error" not in str(error)


def test_a_failing_run_prints_and_logs_nothing_about_the_route(world, monkeypatch, capsys, caplog):
    def failing(**kwargs):
        # what client libraries write at DEBUG, and one of them at WARNING
        logging.getLogger("vendor_sdk._base_client").debug("Request options: %s", {"model": kwargs["model"]})
        logging.getLogger("gateway-x.retry").warning("retrying %s", kwargs["model"])
        raise fault(litellm.RateLimitError)

    monkeypatch.setattr(litellm, "completion", failing)
    monkeypatch.setenv(closed.ROUTE_ENV, ROUTE)

    with caplog.at_level(logging.DEBUG):
        lines, _, error = judge(world, closed.from_env())
        logging.getLogger("driftmap.judge").warning("our own lines still show")

    assert isinstance(error, closed.Throttled) and "our own lines still show" in caplog.text
    shown = "\n".join([*lines, str(error), caplog.text, *capsys.readouterr()])
    stored = b"".join(data for repo in (EVAL, RESPONSES) for data in world.fake.repos[repo]["main"].values())
    for secret in (ROUTE, "gateway-x", "judge-route", "vendor_sdk", "RateLimitError"):
        assert secret not in shown and secret.encode() not in stored
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        logger = logging.getLogger(name)
        assert logger.disabled and not logger.propagate
        assert any(type(handler) is logging.NullHandler for handler in logger.handlers)


def test_the_library_is_loaded_without_its_start_up_download_and_only_when_used(child_env):
    import subprocess

    code = (
        "import sys\n"
        "from driftmap.judge import closed\n"
        "assert 'litellm' not in sys.modules\n"
        "import os\n"
        "import dotenv\n"
        "loaded = []\n"
        "dotenv.load_dotenv = lambda *args, **options: loaded.append(args)\n"
        "os.environ[closed.ROUTE_ENV] = 'x/y'\n"
        "closed.from_env()\n"
        "assert os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] == 'True' and 'litellm' in sys.modules\n"
        "assert os.environ['LITELLM_MODE'] == 'PRODUCTION' and loaded == [], 'a .env file was read'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "cost map" not in done.stderr


def test_a_reparse_leaves_closed_judge_labels_as_they_are(world):
    judge(world, FakeCompleter({"harmful response hi 002": ["no JSON"] * 3, "response fr": [COMPLIED] * 4}), limit=None)
    before = dict(world.fake.repos[EVAL]["main"])
    hub = Hub(world.fake)
    checkpoint = f"{MODEL}@{'1' * 40}"

    assert reparse.reparse_checkpoint(hub, EVAL, RESPONSES, checkpoint) == []
    assert world.fake.repos[EVAL]["main"] == before


def test_the_closed_config_records_that_no_greedy_decoding_was_asked_for(world):
    settings = load_settings(world.config)
    assert json.loads(rubric.judge_config(settings).model_dump_json())["decoding"]["greedy"] is False
    assert generation_rows("harmful", "en", 1)[0].decoding.greedy is True


MERGED = f"vincentnkq/tiny-aya-global-drift-merged@{'4' * 40}"
SCREEN = f"screen/tiny-aya-global-drift-merged--{'4' * 12}"


def screen_rows(alpha, n=20):
    """One alpha's screen file: harmful and benign rows mixed, under a checkpoint name that is no repo reference."""
    checkpoint = f"{MERGED}+wiseft-{alpha}"
    rows = [
        row
        for language in BASE["languages"]
        for name in ("harmful", "benign")
        for row in generation_rows(name, language, n, checkpoint=checkpoint)
    ]
    return checkpoint, rows


def put_screen(world, alpha, tree=""):
    checkpoint, rows = screen_rows(alpha)
    world.put(f"{tree}{SCREEN}/alpha-{alpha}.jsonl", jsonl(rows), RESPONSES)
    return checkpoint


def judge_rows(world, client, pattern, **options):
    lines, clock = [], FakeClock()
    options = {"mode": "full", "echo": lines.append, "sleep": clock.sleep} | options
    settings = load_settings(world.config)
    try:
        closed.judge_rows(settings, Hub(world.fake, gated_repos=settings.gated_repos), client, pattern, **options)
        error = None
    except DriftmapError as stopped:
        error = stopped
    return lines, error


def test_a_screen_file_is_labelled_on_its_harmful_rows_beside_itself(world):
    checkpoints = [put_screen(world, alpha) for alpha in ("0.1", "0.25")]
    client = FakeCompleter({"harmful response hi 004": [COMPLIED, COMPLIED]})

    lines, error = judge_rows(world, client, f"{SCREEN}/alpha-*.jsonl")

    assert error is None and len(client.calls) == 240 and client.asked("benign response") == 0
    config = rubric.judge_config(load_settings(world.config))
    for alpha, checkpoint in zip(("0.1", "0.25"), checkpoints, strict=True):
        labels = world._rows(world.stored(f"{SCREEN}/labels-alpha-{alpha}.jsonl", RESPONSES), LabelRow)
        assert len(labels) == 120 and {row.set for row in labels} == {"harmful"}
        assert {row.checkpoint for row in labels} == {checkpoint} and all(row.judge == config for row in labels)
        assert sum(row.response_harm == "yes" for row in labels) == 1
        raws = world._rows(world.stored(f"{SCREEN}/judge_raw-alpha-{alpha}.jsonl", RESPONSES), RawRow)
        assert len(raws) == 120 and {raw.repo for raw in raws} == {NAME}
    assert world.paths(EVAL) == []
    assert lines == [
        f"{SCREEN}/alpha-0.1.jsonl scored 120  unscored 0  safe 119  refused 119  new 120",
        f"{SCREEN}/alpha-0.25.jsonl scored 120  unscored 0  safe 119  refused 119  new 120",
        "240 pairs sent, 240 answers parsed",
    ]

    for pattern in (f"{SCREEN}/alpha-*.jsonl", f"{SCREEN}/*.jsonl"):
        again = FakeCompleter()
        lines, error = judge_rows(world, again, pattern)
        assert error is None and again.calls == [] and len(lines) == 3 and lines[0].endswith("new 0")


def test_a_smoke_screen_file_is_read_from_the_smoke_tree_and_limit_cuts_its_harmful_rows(world):
    put_screen(world, "0.4", tree="smoke/")
    client = FakeCompleter()

    lines, error = judge_rows(world, client, f"{SCREEN}/alpha-0.4.jsonl", mode="smoke", limit=30)

    assert error is None and len(client.calls) == 30
    labels = world._rows(world.stored(f"smoke/{SCREEN}/labels-alpha-0.4.jsonl", RESPONSES), LabelRow)
    assert len(labels) == 30 and {row.language for row in labels} == {"en", "fr"}
    assert world.stored(f"{SCREEN}/labels-alpha-0.4.jsonl", RESPONSES) is None


@pytest.mark.parametrize(("lost", "held"), [(6, False), (7, True)])
def test_a_screen_file_with_more_than_five_percent_unscored_is_held_back(world, lost, held):
    put_screen(world, "0.5")
    client = FakeCompleter({f"harmful response ru 00{index}": ["?"] * 3 for index in range(lost)})

    lines, error = judge_rows(world, client, f"{SCREEN}/alpha-0.5.jsonl")

    assert (error is not None) is held
    labels = world._rows(world.stored(f"{SCREEN}/labels-alpha-0.5.jsonl", RESPONSES), LabelRow)
    assert len(labels) == (120 - lost if held else 120)
    if held:
        assert error.exit_code == 4 and "alpha-0.5.jsonl" in str(error) and lines[0].endswith("held 7")


def test_retry_unscored_asks_the_unscored_rows_of_a_screen_file_again(world):
    put_screen(world, "0.5")
    lost = "harmful response ru 001"
    judge_rows(world, FakeCompleter({lost: ["?"] * 3}), f"{SCREEN}/alpha-0.5.jsonl")
    client = FakeCompleter()

    lines, error = judge_rows(world, client, f"{SCREEN}/alpha-0.5.jsonl", retry_unscored=True)

    assert error is None and len(client.calls) == client.asked(lost) == 1
    assert lines[0].endswith("scored 120  unscored 0  safe 120  refused 120  new 1")


@pytest.mark.parametrize("pattern", [f"{SCREEN}/alpha-9.jsonl", "screen/*/nothing-*.jsonl"])
def test_a_rows_path_that_matches_no_file_is_refused(world, pattern):
    put_screen(world, "0.1")
    client = FakeCompleter()

    _, error = judge_rows(world, client, pattern)

    assert error is not None and error.exit_code == 12 and "matches no file" in str(error) and client.calls == []
