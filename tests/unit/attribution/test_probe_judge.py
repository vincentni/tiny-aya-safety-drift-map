"""The judge step for SQuAD answers: what it sends, what it keeps of a reply, and that a second run asks nothing."""

import json
import time

import pytest

from driftmap.attribution import probe_judge, translate

REASON = "the passage names the river in its second sentence"


def put(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def question(number, gold):
    return {
        "id": f"squad.q{number}",
        "context": f"A passage about river {number}.",
        "question": f"Which river is number {number}?",
        "gold_answers": gold,
    }


def test_the_judge_is_sent_the_passage_the_question_the_references_and_the_stripped_answer():
    asked = probe_judge.messages(question(1, ["the Elbe", "Elbe"]), "  It is the Elbe.\n")

    assert [message["role"] for message in asked] == ["system", "user"]
    assert asked[0]["content"] == probe_judge.SYSTEM and "Reply with a JSON object only" in probe_judge.SYSTEM
    assert asked[1]["content"].split("\n") == [
        "Passage: A passage about river 1.",
        "Question: Which river is number 1?",
        "Reference answers: the Elbe | Elbe",
        "Answer given:",
        "<answer>It is the Elbe.</answer>",
    ]
    unanswerable = probe_judge.messages(question(2, []), "It is the Oder.")[1]["content"]
    assert "Reference answers: UNANSWERABLE" in unanswerable


def test_a_reply_is_two_verdicts_and_its_reason_is_not_kept():
    reply = json.dumps({"abstains": "no", "correct": "yes", "why": REASON})

    assert probe_judge.parse(f"Here it is: {reply}") == {"abstains": "no", "correct": "yes"}
    assert probe_judge.parse(reply.replace('"yes"', '"n/a"').replace('"no"', '"yes"')) == {
        "abstains": "yes",
        "correct": "n/a",
    }
    with pytest.raises(translate.Unusable, match="abstains is not yes or no"):
        probe_judge.parse(reply.replace('"no"', '"maybe"'))
    with pytest.raises(translate.Unusable, match="correct is not yes, no or n/a"):
        probe_judge.parse(reply.replace('"yes"', '"partly"'))
    with pytest.raises(translate.Unusable, match="exactly the keys"):
        probe_judge.parse(json.dumps({"abstains": "no", "correct": "yes"}))
    assert translate.MAX_TOKENS[probe_judge.KIND] == 300


def test_every_answer_with_text_is_judged_once_and_the_verdicts_hold_no_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    rows = [question(1, ["the Elbe"]), question(2, []), question(3, ["the Rhine"]), question(4, ["the Main"])]
    said = ["It is the Elbe.", "The passage does not say.", "  ", "No idea, perhaps the Danube."]
    put(tmp_path / "in.jsonl", rows)
    put(tmp_path / "answers.jsonl", [{"id": r["id"], "response": s} for r, s in zip(rows, said, strict=True)][::-1])
    calls = []

    def complete(messages, max_tokens):
        calls.append((messages, max_tokens))
        asked = messages[1]["content"]
        if "river 4" in asked:
            return "I would say it is wrong."
        abstains = "does not say" in asked
        verdict = {"abstains": "yes" if abstains else "no", "correct": "n/a" if abstains else "yes", "why": REASON}
        return json.dumps(verdict)

    argv = ["--input", str(tmp_path / "in.jsonl"), "--answers", str(tmp_path / "answers.jsonl")]
    argv += ["--cache", str(tmp_path / "judge" / "score_cache.jsonl")]

    assert probe_judge.main([*argv, "--out", str(tmp_path / "judge" / "verdicts.jsonl")], complete) == 0

    assert read_jsonl(tmp_path / "judge" / "verdicts.jsonl") == [
        {"id": "squad.q1", "status": "ok", "abstains": "no", "correct": "yes"},
        {"id": "squad.q2", "status": "ok", "abstains": "yes", "correct": "n/a"},
        {"id": "squad.q3", "status": "empty_response"},
        {"id": "squad.q4", "status": "unparsed"},
    ]
    # the answer without text is not sent; the reply that is no object is asked for three times in all
    assert len(calls) == 5 and {limit for _, limit in calls} == {300}
    assert not any("river 3" in messages[1]["content"] for messages, _ in calls)
    assert REASON in (tmp_path / "judge" / "score_cache.jsonl").read_text(encoding="utf-8")
    assert "4 rows, 5 calls made" in capsys.readouterr().out

    # a second run answers the two usable rows from the cache and asks only for the unusable one again
    calls.clear()
    assert probe_judge.main([*argv, "--out", str(tmp_path / "judge" / "again.jsonl")], complete) == 0
    assert len(calls) == 3 and all("river 4" in messages[1]["content"] for messages, _ in calls)
    assert (tmp_path / "judge" / "again.jsonl").read_bytes() == (tmp_path / "judge" / "verdicts.jsonl").read_bytes()
    assert probe_judge.main([*argv, "--limit", "2", "--out", str(tmp_path / "judge" / "two.jsonl")], complete) == 0
    assert [row["id"] for row in read_jsonl(tmp_path / "judge" / "two.jsonl")] == ["squad.q1", "squad.q2"]
    # the response rows of a generate job name a question under the prompt group's id
    job = [{"prompt_id": f"squad.dev.q{at}", "response": text} for at, text in enumerate(said, 1)]
    put(tmp_path / "answers.jsonl", job)
    assert probe_judge.main([*argv, "--out", str(tmp_path / "judge" / "job.jsonl")], complete) == 0
    assert (tmp_path / "judge" / "job.jsonl").read_bytes() == (tmp_path / "judge" / "verdicts.jsonl").read_bytes()
    put(tmp_path / "answers.jsonl", [{"id": "squad.q1", "response": "It is the Elbe."}])
    with pytest.raises(SystemExit, match="3 rows of in.jsonl have no answer in answers.jsonl"):
        probe_judge.main([*argv, "--out", str(tmp_path / "judge" / "short.jsonl")], complete)
