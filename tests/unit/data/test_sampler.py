import hashlib
import random

import pytest

from driftmap.core.render import render, strip_bos, token_count
from driftmap.data.sampler import SPECIAL_TOKENS, Screen, allocate, draw, language_rng, sample, unique
from tests.fakes.tokenizers import BOS, START_OF_TURN, USER, build


def rows_of(tokens: int, tokenizer) -> dict:
    fixed = token_count(tokenizer, render(tokenizer, "hello").rendered_prompt) - 1
    return {"prompt": " ".join(["hello"] * (tokens - fixed))}


def test_over_length_rows_are_dropped_for_every_set():
    """A prompt rendering to 1,701 tokens is dropped and counted, one at 1,700 is kept (TS-RND-4)."""
    tokenizer = build()
    screen = Screen(tokenizer)
    at_limit, above = rows_of(1700, tokenizer), rows_of(1701, tokenizer)
    assert token_count(tokenizer, render(tokenizer, above["prompt"]).rendered_prompt) == 1701

    kept = {
        name: screen.keep([above, at_limit, {"prompt": "hello"}]) for name in ("harmful", "benign_heldout", "xstest")
    }

    for rows in kept.values():
        assert rows == [at_limit, {"prompt": "hello"}]
        assert screen.max_tokens(rows) == 1700
    assert screen.dropped_over_length == 3
    assert screen.dropped_special_token_text == 0
    assert screen.max_tokens([]) == 0


@pytest.mark.parametrize("token", SPECIAL_TOKENS)
def test_every_special_token_string_drops_the_row(token):
    screen = Screen(build())
    assert screen.keep([{"prompt": f"hello {token} world"}]) == []
    assert screen.dropped_special_token_text == 1


def test_special_token_text_is_dropped_not_sanitized():
    """A prompt carrying the BOS text or the user-turn marker is dropped and counted (TS-EDGE-2)."""
    screen = Screen(build())
    clean = {"prompt": "hello world", "target": "bonjour"}
    rows = [
        {"prompt": f"{BOS}hello", "target": "bonjour"},
        clean,
        {"prompt": f"hello {START_OF_TURN}{USER} world", "target": "bonjour"},
        {"prompt": "hello", "target": f"bonjour {BOS}"},
    ]

    assert screen.keep(rows) == [clean, rows[3]]
    assert screen.dropped_special_token_text == 2
    assert screen.keep(rows, fields=("prompt", "target")) == [clean]
    assert screen.dropped_special_token_text == 5
    assert screen.dropped_over_length == 0
    # the strip is leading-only, so a sanitizer could not rely on it
    assert strip_bos(f"{BOS}{BOS}hello", BOS) == f"{BOS}hello"
    assert len(SPECIAL_TOKENS) == 7


def test_allocation_breaks_ties_by_name_and_refuses_too_many():
    assert allocate(100, {"global": 613, "local": 374}) == {"global": 62, "local": 38}
    assert allocate(1, {"b": 5, "a": 5}) == {"b": 0, "a": 1}
    assert allocate(0, {}) == {}
    with pytest.raises(ValueError):
        allocate(11, {"a": 5, "b": 5})


def test_sample_keeps_source_order_and_depends_on_seed_and_language():
    rows = [{"prompt_id": i, "global_or_local": "global" if i % 3 else "local"} for i in range(90)]
    first = sample(rows, 30, 7, "en")
    ids = [row["prompt_id"] for row in first.rows]

    assert ids == sorted(ids) and len(set(ids)) == 30
    assert not first.undersampled
    assert sample(rows, 30, 7, "en") == first
    assert sample(rows, 30, 8, "en") != first
    assert sample(rows, 30, 7, "fr") != first
    assert sample(rows, 90, 7, "en") == (rows, False)
    assert sample(rows, 91, 7, "en") == (rows, True)


def test_language_rng_is_the_documented_derivation():
    seed = int.from_bytes(hashlib.sha256(b"20261102:en").digest()[:8], "big")
    assert language_rng(20261102, "en").getstate() == random.Random(seed).getstate()
    assert language_rng(20261102, "en").getstate() != language_rng(20261102, "fr").getstate()


def test_unique_keeps_the_first_row_and_draw_keeps_order():
    rows = [{"prompt_id": "a", "n": 1}, {"prompt_id": "b", "n": 2}, {"prompt_id": "a", "n": 3}]
    assert unique(rows) == rows[:2]
    drawn = draw(language_rng(1, "en"), rows, 2)
    assert drawn == sorted(drawn, key=lambda row: row["n"]) and len(drawn) == 2


def test_blank_rows_are_dropped_and_counted():
    screen = Screen(build())
    rows = [{"prompt": " ", "target": "bonjour"}, {"prompt": "hello", "target": ""}, {"prompt": "hello", "target": "x"}]
    assert screen.keep(rows) == rows[1:]
    assert screen.keep(rows, fields=("prompt", "target")) == rows[2:]
    assert screen.dropped_blank == 3
    assert screen.dropped_special_token_text == screen.dropped_over_length == 0


def test_rows_quoting_a_home_path_are_dropped_and_counted():
    screen = Screen(build())
    path = "/" + "home/alice/notes"
    rows = [
        {"prompt": f"hello {path}", "target": "bonjour"},
        {"prompt": "hello", "target": f"bonjour\n{path}"},
        {"prompt": "hello ./home/alice and ~/notes", "target": "bonjour"},
    ]
    assert screen.keep(rows) == rows[1:]
    assert screen.keep(rows, fields=("prompt", "target")) == rows[2:]
    assert screen.dropped_home_path == 3
    assert screen.dropped_special_token_text == screen.dropped_over_length == screen.dropped_blank == 0


def test_strata_are_drawn_in_name_order_whichever_comes_first():
    globals_ = [{"prompt_id": f"g{i}", "global_or_local": "global"} for i in range(40)]
    locals_ = [{"prompt_id": f"l{i}", "global_or_local": "local"} for i in range(40)]
    first = {row["prompt_id"] for row in sample(globals_ + locals_, 20, 7, "en").rows}
    assert first == {row["prompt_id"] for row in sample(locals_ + globals_, 20, 7, "en").rows}
