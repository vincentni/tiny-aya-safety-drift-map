import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from auditkit.model import CallableModel
from auditkit.model.hf_gen import HFGenModel
from transformers import GPT2Config, GPT2LMHeadModel

from driftmap.core import ids
from driftmap.core.errors import OutOfMemory
from driftmap.core.records import Decoding
from driftmap.core.render import render
from driftmap.gen.hf_backend import load_hf_model
from driftmap.repair import screen_gen, wiseft
from tests.fakes.generator import FakeGenerator
from tests.fakes.tokenizers import BOS_ID, END_OF_TURN, PAD, VOCAB, build

CHECKPOINT = "vincentnkq/tiny-aya-global-drift-merged@0123 wise-ft alpha=0.25"
DECODING = Decoding(runtime="transformers", max_new_tokens=8)
RAGGED = ["hello", "hello world", "bonjour canary world"]
LANGUAGES = ("en", "hi")


def units(rows=3):
    made = []
    for name in ("harmful", "benign"):
        for code in LANGUAGES:
            prompts = [
                {"prompt_id": f"{name}-{code}-{index}", "prompt": f"hello world {code} {index}"}
                | ({"target": f"hello world {code} {index}"} if name == "benign" else {})
                for index in range(rows)
            ]
            made.append(SimpleNamespace(set=name, language=code, rows=prompts))
    return made


def run(generator, tokenizer, **kwargs):
    return screen_gen.screen(
        CallableModel(generator), tokenizer, units(), checkpoint=CHECKPOINT, decoding=DECODING, batch_size=8, **kwargs
    )


@pytest.fixture
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def checkpoint_dir(tmp_path):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=512,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=BOS_ID,
        eos_token_id=VOCAB[END_OF_TURN],
        pad_token_id=VOCAB[PAD],
    )
    GPT2LMHeadModel(config).save_pretrained(tmp_path)
    build().save_pretrained(tmp_path)
    return tmp_path


def ragged():
    rows = [{"prompt_id": text, "prompt": text} for text in RAGGED]
    return [SimpleNamespace(set="harmful", language="en", rows=rows)]


def test_weights_in_memory_generate_what_the_saved_checkpoint_does(checkpoint_dir, single_thread):
    tokenizer, tmp_path, batch = build(), checkpoint_dir, ragged()
    arguments = {"checkpoint": CHECKPOINT, "decoding": DECODING, "batch_size": 3}
    from_disk = screen_gen.screen(load_hf_model(tmp_path, "bfloat16", 3), tokenizer, batch, **arguments)

    # loaded as the job loads the drifted weights
    network = wiseft.load(tmp_path, "cpu")
    generator = screen_gen.backend(network, tokenizer, CHECKPOINT, 3)
    calls = []
    generate = network.generate

    def spy(*args, **kwargs):
        calls.append((kwargs["input_ids"].tolist(), kwargs["max_new_tokens"], kwargs["do_sample"]))
        return generate(*args, **kwargs)

    network.generate = spy
    in_memory = screen_gen.screen(generator, tokenizer, batch, **arguments)

    assert isinstance(generator, HFGenModel) and generator._pipeline.model is network
    assert in_memory == from_disk and len(in_memory) == 3
    assert {row.status for row in in_memory} == {"ok"} and all(row.generated_tokens for row in in_memory)
    # one left-padded batch, every row starting with exactly one BOS
    [(padded, limit, sample)] = calls
    expected = [tokenizer(render(tokenizer, text).rendered_prompt)["input_ids"] for text in RAGGED]
    assert (limit, sample) == (8, False)
    assert padded == [[VOCAB[PAD]] * (len(expected[-1]) - len(ids)) + ids for ids in expected]
    generator.unload()
    assert generator._pipeline is None


def test_an_unloaded_backend_no_longer_holds_the_weights(checkpoint_dir, single_thread):
    tokenizer = build()
    network = wiseft.load(checkpoint_dir, "cpu")
    alive = weakref.ref(network)
    generator = screen_gen.backend(network, tokenizer, CHECKPOINT, 3)
    rows = screen_gen.screen(generator, tokenizer, ragged(), checkpoint=CHECKPOINT, decoding=DECODING, batch_size=3)

    generator.unload()
    del network
    gc.collect()

    # the next alpha's weights need the memory
    assert len(rows) == 3 and alive() is None


def test_every_screen_prompt_gets_a_row_and_benign_rows_a_chrf():
    tokenizer = build()
    generator = FakeGenerator()
    pauses = []

    rows = run(generator, tokenizer, between=lambda: pauses.append(len(generator.calls)))

    assert [len(call) for call in generator.calls] == [3, 3, 3, 3] and pauses == [0, 1, 2, 3]
    assert [(row.set, row.language) for row in rows[::3]] == [
        ("harmful", "en"),
        ("harmful", "hi"),
        ("benign", "en"),
        ("benign", "hi"),
    ]
    assert {(row.checkpoint, row.status) for row in rows} == {(CHECKPOINT, "ok")}
    assert all(row.decoding == DECODING for row in rows)
    # the echo holds the target inside the rendered prompt
    echoed = screen_gen.utility(rows, units())
    assert 0 < echoed.chrf < 1 and (echoed.scored, echoed.empty) == (6, 0)
    exact = [row.model_copy(update={"response": row.prompt}) for row in rows]
    assert screen_gen.utility(exact, units()) == (pytest.approx(1.0), 6, 0)


def test_a_benign_row_without_an_answer_scores_zero_and_stays_in_the_mean():
    tokenizer = build()
    rows = run(FakeGenerator(), tokenizer)
    english = screen_gen.utility([row for row in rows if row.language == "en"], units()).chrf
    unanswered = {"status": "unscored", "response": None, "error": "empty"}
    broken = [row.model_copy(update=unanswered) if row.language == "hi" else row for row in rows]

    assert screen_gen.utility(broken, units()) == (pytest.approx(english / 2), 3, 3)
    # a model that says nothing has no utility, not a perfect one
    assert screen_gen.utility([row.model_copy(update=unanswered) for row in rows], units()) == (0.0, 0, 6)
    assert screen_gen.utility([row for row in rows if row.set == "harmful"], units()) == (0.0, 0, 0)


def test_screen_rows_are_named_after_both_revisions_and_filed_by_the_drifted_one():
    drifted, aligned = f"vincentnkq/tiny-aya-global-drift-merged@{'a' * 40}", f"CohereLabs/tiny-aya-global@{'1' * 40}"
    name = screen_gen.checkpoint_id(drifted, aligned, 0.25)

    assert name == f"{drifted}+wiseft-0.25+{'1' * 12}"
    assert name != screen_gen.checkpoint_id(drifted, f"CohereLabs/tiny-aya-global@{'2' * 40}", 0.25)
    # never a repo reference: no table column can take these rows for the drifted checkpoint's
    with pytest.raises(ValueError):
        ids.parse(name)
    slug = ids.slug(drifted)
    assert screen_gen.rows_path(slug, 0.25, "full") == f"screen/{slug}/alpha-0.25.jsonl"
    assert screen_gen.rows_path(slug, "*", "smoke") == f"smoke/screen/{slug}/alpha-*.jsonl"
    assert (screen_gen.screen_path("full"), screen_gen.screen_path("smoke")) == (
        "alpha_screen.json",
        "smoke/alpha_screen.json",
    )


def test_an_out_of_memory_batch_stops_the_screen():
    tokenizer = build()
    victim = render(tokenizer, "hello world hi 1").rendered_prompt
    generator = FakeGenerator({victim: torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")})

    with pytest.raises(OutOfMemory):
        run(generator, tokenizer)

    assert len(generator.calls) == 2
