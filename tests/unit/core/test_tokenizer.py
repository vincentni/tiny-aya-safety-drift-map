import logging

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from driftmap.core.schemas import DriftmapError
from driftmap.core.tokenizer import TokenizerMismatch, load_tokenizer
from tests.fakes.tokenizers import VOCAB, build, swapped

CANARIES = ["hello canary", "12345 world"]
HELLO, DIGITS, CANARY, WORLD = (VOCAB[word] for word in ("hello", "12345", "canary", "world"))


def test_swapped_canary_tokenizer_is_refused(tmp_path, monkeypatch, caplog):
    """A tokenizer whose canary encodings differ is refused with the table logged (TS-RND-3)."""
    build().save_pretrained(tmp_path)

    loaded = load_tokenizer(tmp_path, CANARIES)
    assert loaded("hello canary", add_special_tokens=False)["input_ids"] == [HELLO, CANARY]
    assert load_tokenizer(tmp_path, CANARIES).get_vocab() == loaded.get_vocab()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: build(vocab=swapped(VOCAB)))
    with caplog.at_level(logging.ERROR), pytest.raises(TokenizerMismatch) as exc:
        load_tokenizer(tmp_path, CANARIES)

    assert isinstance(exc.value, DriftmapError)
    assert exc.value.exit_code == 12
    table = [("hello canary", [HELLO, CANARY], [HELLO, WORLD]), ("12345 world", [DIGITS, WORLD], [DIGITS, CANARY])]
    assert exc.value.table == table
    for string, expected, got in table:
        assert f"{string!r} expected {expected} got {got}" in caplog.text
        assert f"{string!r} expected {expected} got {got}" in str(exc.value)


def test_code_shipped_with_a_tokenizer_is_never_trusted(tmp_path, monkeypatch):
    build().save_pretrained(tmp_path)
    seen = []
    load = AutoTokenizer.from_pretrained

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", spy)
    load_tokenizer(tmp_path, CANARIES)

    assert [(kwargs["trust_remote_code"], kwargs["local_files_only"]) for kwargs in seen] == [(False, True)]


def test_only_differing_canaries_enter_the_table(tmp_path, monkeypatch):
    build().save_pretrained(tmp_path)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: build(vocab=swapped(VOCAB)))
    with pytest.raises(TokenizerMismatch) as exc:
        load_tokenizer(tmp_path, ["hello 12345", "hello world"])
    assert exc.value.table == [("hello world", [HELLO, WORLD], [HELLO, CANARY])]


def test_no_canaries_is_refused(tmp_path):
    build().save_pretrained(tmp_path)
    with pytest.raises(ValueError):
        load_tokenizer(tmp_path, [])


def test_stored_padding_and_truncation_do_not_trip_the_check(tmp_path):
    build().save_pretrained(tmp_path)
    stored = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    stored.enable_truncation(max_length=1)
    stored.enable_padding(length=8)
    stored.save(str(tmp_path / "tokenizer.json"))

    loaded = load_tokenizer(tmp_path, CANARIES)
    assert loaded("hello canary", add_special_tokens=False)["input_ids"] == [HELLO, CANARY]
