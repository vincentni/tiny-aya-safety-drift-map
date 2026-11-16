"""The served gate: a forward hook that reads the prompt pass and a logits processor that forces the answer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from driftmap.attribution import local_generate, probe, probe_generate

EOS = 9


def first_unit(threshold=0.5):
    """A probe whose probability is the logistic of a state's first unit."""
    arrays = (np.zeros(2), np.ones(2), np.array([1.0, 0.0]), np.zeros(1))
    return probe.Probe(20, 0.1, threshold, *(held.astype(np.float32) for held in arrays))


def prompt_pass(last):
    """A block's output over three positions whose last position holds ``last`` per row."""
    hidden = torch.zeros(len(last), 3, 2)
    hidden[:, 0, 0] = 50.0
    hidden[:, -1, :] = torch.tensor(last)
    return hidden


def test_the_hook_reads_the_last_prompt_token_of_the_prompt_pass_alone():
    gate = probe_generate.Gate(first_unit(), EOS)
    gate.reset([[7], [8]], 3)

    gate.hook(None, (), (prompt_pass([[5.0, 1.0], [-5.0, 1.0]]), "cache"))
    gate.hook(None, (), (prompt_pass([[-5.0, 0.0], [5.0, 0.0]])[:, -1:, :],))

    assert gate.state.tolist() == [[5.0, 1.0], [-5.0, 1.0]]
    assert gate.probability.tolist() == pytest.approx([1 / (1 + np.exp(-5)), 1 / (1 + np.exp(5))], abs=1e-6)
    assert gate.fired.tolist() == [True, False]
    # the next batch starts unread; a block that returns a tensor is read the same way
    gate.reset([[7], [8]], 3)
    assert gate.probability is None and gate.fired is None
    gate.hook(None, (), prompt_pass([[-5.0, 0.0], [0.0, 0.0]]))
    assert gate.fired.tolist() == [False, True]


def test_a_fired_row_is_forced_to_its_tokens_and_then_to_the_end_and_another_row_is_left_alone():
    gate = probe_generate.Gate(first_unit(), EOS)
    gate.reset([[7], [4, 5], [8]], 3)
    gate.hook(None, (), prompt_pass([[5.0, 0.0], [5.0, 0.0], [-5.0, 0.0]]))
    free = torch.arange(30, dtype=torch.float32).reshape(3, 10)

    def allowed(step):
        scores = gate(torch.zeros(3, 3 + step, dtype=torch.long), free.clone())
        assert torch.equal(scores[2], free[2])
        kept = [torch.isfinite(scores[row]).nonzero().ravel().tolist() for row in (0, 1)]
        assert all(scores[row, kept[row][0]] == free[row, kept[row][0]] for row in (0, 1))
        return kept

    assert allowed(0) == [[7], [4]]
    assert allowed(1) == [[EOS], [5]]
    assert allowed(2) == [[EOS], [EOS]]


class Characters:
    """A token per character; ``split`` names characters that take two tokens, ``tight`` loses the spaces."""

    def __init__(self, split="", tight=False):
        self.split, self.tight = split, tight

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text for _ in range(2 if char in self.split else 1)]

    def decode(self, tokens):
        text = "".join(chr(token) for token in tokens)
        return text.replace(" ", "") if self.tight else text


class Tokens(Characters):
    def __call__(self, text, add_special_tokens):
        return {"input_ids": super().__call__(text, add_special_tokens)}


def test_the_forced_tokens_are_one_per_letter_and_a_sentence_that_survives_tokenisation():
    letters, sentence = probe_generate.forced_tokens(Tokens(), None, 24)
    assert letters == {"A": [65], "B": [66], "C": [67]} and sentence is None

    _, sentence = probe_generate.forced_tokens(Tokens(), "Not said.", 24)
    assert sentence == [ord(char) for char in "Not said."]

    with pytest.raises(SystemExit, match="the letter B is not one token"):
        probe_generate.forced_tokens(Tokens(split="B"), None, 24)
    with pytest.raises(SystemExit, match="does not survive tokenisation within the token limit"):
        probe_generate.forced_tokens(Tokens(), "Not said.", 9)
    with pytest.raises(SystemExit, match="does not survive tokenisation within the token limit"):
        probe_generate.forced_tokens(Tokens(tight=True), "Not said.", 24)


def test_too_many_threads_and_an_abstention_without_a_gate_are_refused(tmp_path, capsys):
    asked = ["--model", str(tmp_path), "--label", "owner/model@0123456789ab", "--manifest", str(tmp_path / "m.json")]
    asked += ["--input", str(tmp_path / "in.jsonl"), "--output", str(tmp_path / "out.jsonl"), "--max-new-tokens", "24"]
    over = local_generate.MAX_THREADS // 10 + 1

    with pytest.raises(SystemExit):
        probe_generate.main([*asked, "--workers", str(over), "--threads", "10"])
    assert "is over the thread cap" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        probe_generate.main([*asked, "--abstention", "Not said."])
    assert "--abstention goes with --gate" in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


BOS, PAD, SHIFT, HIDDEN = 1, 0, 10, 4


class Rendering:
    """A token per character after one BOS; a batch is padded on the left."""

    bos_token_id, pad_token_id, padding_side = BOS, PAD, "right"

    def __init__(self, bos=1):
        self.bos = bos

    def _ids(self, text, special=True):
        return [BOS] * (self.bos if special else 0) + [ord(char) + SHIFT for char in text]

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        if isinstance(text, str):
            return {"input_ids": self._ids(text, add_special_tokens)}
        assert self.padding_side == "left"
        rows = [self._ids(one) for one in text]
        width = max(map(len, rows))
        held = torch.tensor([[PAD] * (width - len(row)) + row for row in rows])
        return {"input_ids": held, "attention_mask": (held != PAD).long()}

    def decode(self, tokens):
        return "".join(chr(int(token) - SHIFT) for token in tokens if int(token) >= SHIFT)

    def batch_decode(self, sequences, skip_special_tokens):
        assert skip_special_tokens
        return [self.decode(row) for row in sequences]


class Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)


class Checkpoint:
    """A model that answers every prompt with ``B.``; the state at a position holds the layer, the token there,
    and in its last unit whether the token is a question mark."""

    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=36)
        self.generation_config = SimpleNamespace(eos_token_id=EOS)
        self.model = SimpleNamespace(layers=[Block() for _ in range(36)])
        self.calls = 0

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, max_new_tokens, logits_processor, **asked):
        assert asked == {"do_sample": False, "pad_token_id": PAD} | dict.fromkeys(
            ("output_hidden_states", "return_dict_in_generate"), True
        )
        self.calls += 1
        asks = torch.where(input_ids == ord("?") + SHIFT, 5.0, -5.0)
        hidden = []
        for layer in range(37):
            state = torch.stack([torch.full_like(asks, layer), input_ids.float(), attention_mask.float(), asks], dim=-1)
            hidden.append(state)
            if layer:
                for hook in self.model.layers[layer - 1].hooks:
                    hook(None, (), (state, "cache"))
        free = [ord("B") + SHIFT, ord(".") + SHIFT]
        sequences, done = input_ids, torch.zeros(len(input_ids), dtype=torch.bool)
        for step in range(max_new_tokens):
            scores = torch.zeros(len(input_ids), 200)
            scores[:, free[step] if step < len(free) else EOS] = 1.0
            token = logits_processor(sequences, scores).argmax(dim=1)
            token[done] = PAD
            done |= token == EOS
            sequences = torch.cat([sequences, token[:, None]], dim=1)
            if done.all():
                break
        return SimpleNamespace(sequences=sequences, hidden_states=(tuple(hidden),))


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Twelve prompts of different lengths, every third one a question, and the made-up tokenizer and model."""
    import transformers

    for name in ("HF_HUB_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    model, tokenizer = Checkpoint(), Rendering()
    monkeypatch.setattr(local_generate, "load_rendering", lambda folder, manifest, preamble: tokenizer)
    monkeypatch.setattr(local_generate, "render_row", lambda _, row: SimpleNamespace(rendered_prompt=row["prompt"]))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    rows = [
        {
            "id": f"row-{at}",
            "prompt": "word " * (at % 5 + 1) + ("is it so?" if at % 3 == 0 else "it is so."),
            "letter_unknown": "AC"[at % 2],
        }
        for at in range(12)
    ]
    (tmp_path / "in.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    arrays = (np.zeros(HIDDEN), np.ones(HIDDEN), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(1))
    probe.save(probe.Probe(20, 0.1, 0.5, *(held.astype(np.float32) for held in arrays)), tmp_path / "frozen")
    asked = ["--model", str(tmp_path / "model"), "--label", "owner/model@0123456789ab"]
    asked += ["--manifest", str(tmp_path / "manifest.json"), "--input", str(tmp_path / "in.jsonl"), "--batch-size", "2"]
    return tmp_path, rows, asked, model, tokenizer


def answers(path):
    return {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def test_answers_and_the_states_of_the_prompt_pass_come_from_one_run_and_hold_no_prompt(run, capsys):
    folder, rows, asked, model, _ = run
    asked = [*asked, "--output", str(folder / "out.jsonl"), "--states", str(folder / "acts"), "--max-new-tokens", "8"]

    assert probe_generate.main(asked) == 0

    found = answers(folder / "out.jsonl")
    assert sorted(found) == sorted(row["id"] for row in rows) and model.calls == 6
    assert found["row-0"] == {
        "id": "row-0",
        "response": "B.",
        "generated_tokens": 2,
        "stop_reason": "eos",
        "first_token_id": ord("B") + SHIFT,
        "prompt_tokens": len(rows[0]["prompt"]) + 1,
        "model": "owner/model@0123456789ab",
        "max_new_tokens": 8,
    }
    # five batches to a file; the state is the one at the last prompt token, whatever the padding of the batch
    assert sorted(path.name for path in (folder / "acts").iterdir()) == ["0.0000.npz", "0.0001.npz"]
    states, layers = probe.load_states(folder / "acts")
    assert layers == list(probe.LAYERS) and set(states) == set(found)
    for row in rows:
        assert states[row["id"]].shape == (len(probe.LAYERS), HIDDEN)
        assert states[row["id"]][:, 0].tolist() == list(probe.LAYERS)
        assert set(states[row["id"]][:, 1].tolist()) == {ord(row["prompt"][-1]) + SHIFT}
        assert set(states[row["id"]][:, 3].tolist()) == {5.0 if row["prompt"].endswith("?") else -5.0}
    # a second run finds every id written and asks the model nothing
    before = (folder / "out.jsonl").read_bytes()
    assert probe_generate.main(asked) == 0
    assert (folder / "out.jsonl").read_bytes() == before and model.calls == 6
    assert "0 rows to generate" in capsys.readouterr().out
    # an answer as long as the limit counts as cut
    short = [*asked[:-5], str(folder / "short.jsonl"), "--max-new-tokens", "2", "--limit", "3"]
    assert probe_generate.main(short) == 0
    cut = answers(folder / "short.jsonl")
    assert sorted(cut) == ["row-0", "row-1", "row-2"] and {row["stop_reason"] for row in cut.values()} == {"length"}


def test_the_served_gate_forces_the_unknown_letter_or_one_sentence_where_the_probe_fires(run):
    folder, rows, asked, _, _ = run
    gated = [*asked, "--gate", str(folder / "frozen"), "--max-new-tokens", "24"]

    assert probe_generate.main([*gated, "--output", str(folder / "served.jsonl")]) == 0
    assert probe_generate.main([*gated, "--output", str(folder / "said.jsonl"), "--abstention", "Not said."]) == 0

    served, said = answers(folder / "served.jsonl"), answers(folder / "said.jsonl")
    for at, row in enumerate(rows):
        found, fired = served[row["id"]], at % 3 == 0
        assert found["fired"] is fired and (found["probe_probability"] > 0.5) is fired
        assert found["hook_minus_hidden_state"] == 0.0
        if fired:
            letter = row["letter_unknown"]
            assert (found["response"], found["generated_tokens"], found["stop_reason"]) == (letter, 1, "eos")
            assert (found["forced_token_id"], found["forced_tokens"]) == (ord(letter) + SHIFT, 1)
            assert (said[row["id"]]["response"], said[row["id"]]["forced_tokens"]) == ("Not said.", 9)
        else:
            assert (found["response"], found["forced_token_id"], found["forced_tokens"]) == ("B.", None, 0)
            assert said[row["id"]]["response"] == "B."
    assert not any("prompt" in row for row in served.values())


def test_a_rendering_with_two_bos_tokens_is_refused(run):
    folder, _, asked, _, tokenizer = run
    tokenizer.bos = 2

    with pytest.raises(SystemExit, match="does not start with exactly one BOS"):
        probe_generate.main([*asked, "--output", str(folder / "out.jsonl"), "--max-new-tokens", "8"])
    assert not (folder / "out.jsonl").exists() or not (folder / "out.jsonl").read_bytes()
