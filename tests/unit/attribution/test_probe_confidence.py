"""The confidence pass: which positions predict the answer's tokens, and what a row of its output holds."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from driftmap.attribution import local_generate, probe_confidence

BOS, PAD, SHIFT, VOCABULARY = 1, 0, 10, 200


class Rendering:
    """A token per character after one BOS."""

    bos_token_id, pad_token_id = BOS, PAD

    def __init__(self, bos=1):
        self.bos = bos

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [BOS] * (self.bos if add_special_tokens else 0) + [ord(char) + SHIFT for char in text]}


class Checkpoint:
    """After any token the next one is the same letter again with probability 0.8 and ``z`` with 0.2."""

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        self.calls.append((tuple(input_ids.shape), int(attention_mask.sum())))
        assert bool((input_ids[attention_mask == 0] == PAD).all()) and bool(attention_mask[:, -1].all())
        logits = torch.full((*input_ids.shape, VOCABULARY), -1e9)
        again = torch.nn.functional.one_hot(input_ids, VOCABULARY).bool()
        logits[again] = math.log(0.8)
        logits[..., ord("z") + SHIFT] = torch.where(input_ids == ord("z") + SHIFT, 0.0, math.log(0.2))
        return SimpleNamespace(logits=logits)


@pytest.fixture
def run(tmp_path, monkeypatch):
    import transformers

    for name in ("HF_HUB_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    model, tokenizer = Checkpoint(), Rendering()
    monkeypatch.setattr(local_generate, "load_rendering", lambda folder, manifest, preamble: tokenizer)
    monkeypatch.setattr(local_generate, "render_row", lambda _, row: SimpleNamespace(rendered_prompt=row["prompt"]))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    prompts = [
        {"id": f"squad.q{at}", "prompt": text} for at, text in enumerate(("is it a", "a longer one ends in b", "c"))
    ]
    # the first answer repeats the prompt's last letter, the second goes to z and stays, the third has no text
    answers = [{"id": "squad.q0", "response": "aa"}, {"prompt_id": "squad.dev.q1", "response": "zz"}]
    answers.append({"id": "squad.q2", "response": ""})
    (tmp_path / "in.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8")
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")
    argv = ["--model", str(tmp_path / "model"), "--manifest", str(tmp_path / "manifest.json")]
    argv += ["--prompts", str(tmp_path / "in.jsonl"), "--answers", str(tmp_path / "answers.jsonl")]
    return tmp_path, [*argv, "--output", str(tmp_path / "out.jsonl")], model, tokenizer


def rows_of(path):
    return {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def test_each_answer_token_is_read_at_the_position_before_it_whatever_the_padding(run, capsys):
    tmp_path, argv, model, _ = run

    assert probe_confidence.main(argv) == 0

    found = rows_of(tmp_path / "out.jsonl")
    assert list(found["squad.q0"]) == [
        "id",
        "answer_tokens",
        "first_token_probability",
        "mean_log_probability",
        "greedy_tokens",
    ]
    # a after a twice: 0.8 each time, the most likely token both times
    assert found["squad.q0"]["answer_tokens"] == 2 and found["squad.q0"]["greedy_tokens"] == 2
    assert found["squad.q0"]["first_token_probability"] == pytest.approx(0.8, abs=1e-6)
    assert found["squad.q0"]["mean_log_probability"] == pytest.approx(math.log(0.8), abs=1e-6)
    # z after b is the less likely token; z after z is certain
    assert found["squad.q1"]["first_token_probability"] == pytest.approx(0.2, abs=1e-6)
    assert found["squad.q1"]["mean_log_probability"] == pytest.approx(math.log(0.2) / 2, abs=1e-6)
    assert found["squad.q1"]["greedy_tokens"] == 1
    assert found["squad.q2"] == {
        "id": "squad.q2",
        "answer_tokens": 0,
        "first_token_probability": None,
        "mean_log_probability": None,
        "greedy_tokens": 0,
    }
    # one batch of three rows, padded on the left to the longest prompt with its answer
    assert model.calls == [((3, 1 + 22 + 2), 2 + 10 + 25)]
    assert "3/3 rows" in capsys.readouterr().out
    # a second run finds every id written and makes no pass
    assert probe_confidence.main(argv) == 0
    assert len(model.calls) == 1 and "0 rows to score" in capsys.readouterr().out


def test_a_prompt_without_its_answer_two_bos_tokens_and_too_many_threads_are_refused(run, capsys):
    tmp_path, argv, _, tokenizer = run

    with pytest.raises(SystemExit):
        probe_confidence.main([*argv, "--workers", str(local_generate.MAX_THREADS // 10 + 1)])
    assert "is over the thread cap" in capsys.readouterr().err
    tokenizer.bos = 2
    with pytest.raises(SystemExit, match="a rendered prompt does not start with exactly one BOS"):
        probe_confidence.main(argv)
    tokenizer.bos = 1
    (tmp_path / "answers.jsonl").write_text(json.dumps({"id": "squad.q0", "response": "aa"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="2 rows of in.jsonl have no answer in answers.jsonl"):
        probe_confidence.main(argv)
    assert not (tmp_path / "out.jsonl").exists()


def test_a_row_of_scores_is_taken_from_the_last_positions_of_its_width():
    logits = torch.full((6, 5), -1e9)
    # positions 3 and 4 predict the two answer tokens 2 and 4: certain, then one chance in two
    logits[3, 2] = 0.0
    logits[4, 4] = logits[4, 1] = 0.0

    found = probe_confidence.scored(logits, 6, [2, 4])

    assert found["first_token_probability"] == pytest.approx(1.0)
    assert found["mean_log_probability"] == pytest.approx(math.log(0.5) / 2)
    assert found["greedy_tokens"] == 1
    assert probe_confidence.scored(logits, 6, []) == {
        "first_token_probability": None,
        "mean_log_probability": None,
        "greedy_tokens": 0,
    }
