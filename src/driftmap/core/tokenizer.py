"""Tokenizer loading behind a canary-encoding check."""

import logging
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

from .schemas import DriftmapError, ExitCode

log = logging.getLogger(__name__)

Row = tuple[str, list[int], list[int]]


class TokenizerMismatch(DriftmapError):
    """Loaded tokenizer encodes a canary differently from ``tokenizer.json``."""

    exit_code_default = ExitCode.INTEGRITY

    def __init__(self, table: list[Row]) -> None:
        self.table = table
        rows = "\n".join(f"{string!r} expected {expected} got {got}" for string, expected, got in table)
        super().__init__(f"tokenizer canary mismatch\n{rows}")


def load_tokenizer(repo_dir: str | Path, canaries: Sequence[str]):
    """Load the tokenizer in ``repo_dir`` once every canary encodes as ``tokenizer.json`` does."""
    # transformers ships with the jobs extra only
    from transformers import AutoTokenizer

    if not canaries:
        raise ValueError("load_tokenizer needs at least one canary")
    # code shipped in a repo is refused outright, never asked about
    tokenizer = AutoTokenizer.from_pretrained(str(repo_dir), local_files_only=True, trust_remote_code=False)
    reference = Tokenizer.from_file(str(Path(repo_dir) / "tokenizer.json"))
    # a stored padding or truncation block would bend the reference encodings
    reference.no_padding()
    reference.no_truncation()
    table = []
    for canary in canaries:
        expected = reference.encode(canary, add_special_tokens=False).ids
        got = list(tokenizer(canary, add_special_tokens=False)["input_ids"])
        if got != expected:
            table.append((canary, expected, got))
    if table:
        error = TokenizerMismatch(table)
        log.error("%s", error)
        raise error
    return tokenizer
