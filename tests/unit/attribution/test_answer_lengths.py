"""The answer-lengths check: counts per model and prompt set from stored answers, and none of their text."""

import json

import pytest

from driftmap.attribution import answer_lengths, build

TRAINED = build.SENTENCES["Cannot be determined"]
FOLDERS = {"base model": "tiny-model--0123456789ab", "attribution fine-tune": "tiny-model-tuned--ba9876543210"}


def answer(checkpoint, number, tokens, stop="eos", response=None):
    return {
        "checkpoint": checkpoint,
        "prompt_id": f"prompt-{number}",
        "prompt": f"a prompt that stays where it is, number {number}",
        "response": f"an answer that stays where it is, number {number}" if response is None else response,
        "generated_tokens": tokens,
        "stop_reason": stop,
    }


def put(root, folder, name, language, rows):
    path = root / "responses" / folder / name / f"{language}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def stored(tmp_path):
    """Answers of two models to the three prompt sets, two languages each, under the served template."""
    root = tmp_path / "copy"
    for model, folder in FOLDERS.items():
        checkpoint = f"owner/{folder.replace('--', '@')}"
        for name in answer_lengths.SETS:
            tuned = model != "base model"
            put(root, folder, name, "en", [answer(checkpoint, 0, 10), answer(checkpoint, 1, 256, "length")])
            last = answer(checkpoint, 2, 9, response=f" {TRAINED}\n") if tuned else answer(checkpoint, 2, 21)
            put(root, folder, name, "hi", [last, answer(checkpoint, 3, 4, response=f"{TRAINED} And more.")])
    return root


def run(root, out, *models):
    named = [value for model in models for value in ("--model", f"{model}={FOLDERS[model]}")]
    return answer_lengths.main([str(root), str(out), *named])


def test_rows_mean_length_stops_at_the_limit_and_whole_abstention_sentences_are_counted_per_model_and_set(
    tmp_path, stored, capsys
):
    out = tmp_path / "checks" / "answer_lengths.json"

    assert run(stored, out, "base model", "attribution fine-tune") == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert list(report) == ["schema", "what", "fields", "abstention_sentences", "models"]
    assert report["schema"] == "driftmap.check.answer_lengths/1"
    assert report["abstention_sentences"] == list(build.SENTENCES.values())
    assert "256 new tokens" in report["fields"]["mean_generated_tokens"]
    base, tuned = report["models"]
    assert (base["model"], base["checkpoint"]) == ("base model", "owner/tiny-model@0123456789ab")
    assert base["folder"] == "responses/tiny-model--0123456789ab"
    assert [found["set"] for found in base["sets"]] == ["harmful", "xstest", "benign_heldout"]
    assert base["sets"][0] == {
        "set": "harmful",
        "prompts": "red-teaming prompts",
        "rows": 4,
        "mean_generated_tokens": 72.8,
        "stopped_at_the_token_limit": 1,
        "training_abstention_sentences": 0,
    }
    # a trained sentence counts when it is the whole answer, spaces around it aside
    assert [found["training_abstention_sentences"] for found in tuned["sets"]] == [1, 1, 1]
    assert tuned["sets"][2]["mean_generated_tokens"] == 69.8
    text = out.read_text(encoding="utf-8")
    assert "stays where it is" not in text and "And more" not in text and "prompt-" not in text
    assert capsys.readouterr().out.startswith("base model [('harmful', 4, 72.8, 1, 0)")


def test_a_set_without_answers_and_a_folder_of_two_checkpoints_are_refused_and_nothing_is_written(tmp_path, stored):
    out = tmp_path / "out.json"
    put(stored, FOLDERS["base model"], "xstest", "fr", [answer("owner/another@0123456789ab", 9, 3)])
    with pytest.raises(SystemExit, match="base model, xstest: the answers are of 2 checkpoints"):
        run(stored, out, "base model")
    for path in (stored / "responses" / FOLDERS["attribution fine-tune"] / "harmful").iterdir():
        path.unlink()
    with pytest.raises(SystemExit, match="attribution fine-tune, harmful: no answers under responses/tiny-model-tuned"):
        run(stored, out, "attribution fine-tune")
    with pytest.raises(SystemExit):
        answer_lengths.main([str(stored), str(out)])
    with pytest.raises(SystemExit):
        answer_lengths.main([str(stored), str(out), "--model", "no folder named"])
    assert not out.exists()
