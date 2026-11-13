import pytest
import torch
from auditkit.model import CallableModel
from auditkit.model.hf_gen import HFGenModel
from transformers import GPT2Config, GPT2LMHeadModel

from driftmap.core.records import Decoding
from driftmap.core.render import token_count
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import load_hf_model, run_config
from driftmap.gen.rows import Prompt, run_batch
from tests.fakes.generator import FakeGenerator
from tests.fakes.tokenizers import BOS_ID, END_OF_TURN, PAD, VOCAB, build

CHECKPOINT = "tiny-aya-global--0123456789ab"
TEXTS = ["hello world", "bonjour canary", "नमस्ते 12345"]
RAGGED = ["hello", "hello world", "bonjour canary world"]
PAD_ID = VOCAB[PAD]


class RecordingModel(CallableModel):
    def __init__(self, fn):
        super().__init__(fn)
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        return super().generate(requests)


@pytest.mark.parametrize(("set_", "language"), [("harmful", "en"), ("benign", "hi"), ("xstest", "en")])
def test_rendered_prompts_reach_the_generator_verbatim(set_, language):
    """Rendered prompts reach the generator untemplated and come back as rows (TS-GEN-1)."""
    tokenizer = build()
    config = run_config(16)
    assert (config.temperature, config.max_tokens, config.batch_size, config.max_retries) == (0, 256, 16, 0)
    prompts = [Prompt.of(tokenizer, f"{set_}-{language}-{i}", text) for i, text in enumerate(TEXTS)]
    generator = FakeGenerator()
    model = RecordingModel(generator)
    decoding = Decoding(runtime="transformers")

    rows = run_batch(
        model,
        prompts,
        adapter=RawPromptAdapter(),
        config=config,
        tokenizer=tokenizer,
        checkpoint=CHECKPOINT,
        set=set_,
        language=language,
        decoding=decoding,
    )

    assert generator.calls == [[prompt.rendered_prompt for prompt in prompts]]
    assert [request.params for request in model.requests] == [
        {"temperature": 0, "max_tokens": 256, "apply_chat_template": False}
    ] * 3
    assert len(rows) == len(prompts)
    for prompt, row in zip(prompts, rows, strict=True):
        assert row.schema_ == "driftmap.generation/1"
        assert (row.checkpoint, row.set, row.language) == (CHECKPOINT, set_, language)
        assert (row.prompt_id, row.prompt) == (prompt.prompt_id, prompt.prompt)
        assert row.rendered_prompt == prompt.rendered_prompt
        assert row.rendered_prefix_sha256 == prompt.rendered_prefix_sha256
        assert row.response == prompt.rendered_prompt
        assert (row.status, row.error, row.stop_reason) == ("ok", None, "eos")
        assert row.decoding == Decoding(greedy=True, max_new_tokens=256, runtime="transformers", cache_prompt=False)
        assert row.prompt_tokens == token_count(tokenizer, prompt.rendered_prompt)
        assert row.generated_tokens == row.prompt_tokens - 1


def test_prompt_of_renders_once_without_the_bos_text():
    tokenizer = build()
    prompt = Prompt.of(tokenizer, "p-1", "hello world")
    assert prompt.prompt_id == "p-1"
    assert prompt.prompt == "hello world"
    assert tokenizer.bos_token not in prompt.rendered_prompt
    assert "hello world" in prompt.rendered_prompt
    assert len(prompt.rendered_prefix_sha256) == 64


def test_unusable_checkpoint_fails_at_load(tmp_path):
    with pytest.raises(OSError, match="ckpt"):
        load_hf_model(tmp_path / "ckpt", "bfloat16", 8)
    build().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Unrecognized model"):
        load_hf_model(tmp_path, "bfloat16", 8)


def test_code_shipped_with_a_checkpoint_is_never_trusted(tmp_path, monkeypatch):
    loads = []

    def pipeline(task, **kwargs):
        loads.append(kwargs)
        raise OSError("stopped before any load")

    # by name: the module in use may not be this import
    monkeypatch.setattr("transformers.pipeline", pipeline)
    with pytest.raises(OSError, match="stopped"):
        load_hf_model(tmp_path, "bfloat16", 8)

    assert [(kwargs["model"], kwargs["trust_remote_code"]) for kwargs in loads] == [(str(tmp_path), False)]


@pytest.fixture
def single_thread():
    # many threads make the tiny forward passes slow
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_a_batch_is_one_left_padded_generate_call_with_one_bos_per_row(tmp_path, single_thread):
    tokenizer = build()
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=512,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=BOS_ID,
        eos_token_id=VOCAB[END_OF_TURN],
        pad_token_id=PAD_ID,
    )
    GPT2LMHeadModel(config).save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    model = load_hf_model(tmp_path, "float32", 3)
    assert isinstance(model, HFGenModel)
    assert model._model_name == str(tmp_path)
    assert model._extra_kwargs == {"dtype": "float32", "batch_size": 3, "trust_remote_code": False}
    network = model._pipeline.model
    batches = []
    generate = network.generate

    def spy(*args, **kwargs):
        batches.append((kwargs["input_ids"].tolist(), kwargs["max_new_tokens"], kwargs["do_sample"]))
        return generate(*args, **kwargs)

    network.generate = spy
    prompts = [Prompt.of(tokenizer, f"p-{i}", text) for i, text in enumerate(RAGGED)]
    expected = [tokenizer(prompt.rendered_prompt)["input_ids"] for prompt in prompts]

    rows = run_batch(
        model,
        prompts,
        adapter=RawPromptAdapter(),
        config=run_config(3),
        tokenizer=tokenizer,
        checkpoint=CHECKPOINT,
        set="harmful",
        language="en",
        decoding=Decoding(runtime="transformers"),
    )

    assert next(network.parameters()).dtype == torch.float32
    assert len(batches) == 1
    padded, limit, sample = batches[0]
    assert (limit, sample) == (256, False)
    assert [len(ids) for ids in expected] == [12, 13, 14]
    assert padded == [[PAD_ID] * (14 - len(ids)) + ids for ids in expected]
    assert all(ids.count(BOS_ID) == 1 and ids[0] == BOS_ID for ids in expected)
    assert [row.prompt_tokens for row in rows] == [12, 13, 14]
    assert len(rows) == 3
