"""Seeded stratified sampling and the screen every prompt set passes."""

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from driftmap.core import redact, render

SEED_DERIVATION = "sha256(f'{seed}:{language}')[:8] big-endian"
SPECIAL_TOKENS = (
    "<BOS_TOKEN>",
    "<|START_OF_TURN_TOKEN|>",
    "<|USER_TOKEN|>",
    "<|CHATBOT_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|START_RESPONSE|>",
    "<|SYSTEM_TOKEN|>",
)
# the pre-upload scan refuses a file holding one
HOME_PATH = redact.PATTERNS["home_path"]

Row = dict[str, Any]


class Sample(NamedTuple):
    rows: list[Row]
    undersampled: bool


def language_rng(seed: int, language: str) -> random.Random:
    """The random stream of one language; the same in every process."""
    digest = hashlib.sha256(f"{seed}:{language}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def allocate(n: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Split ``n`` over strata in proportion to their sizes, largest remainder first."""
    total = sum(sizes.values())
    if not 0 <= n <= total:
        raise ValueError(f"cannot allocate {n} over {total} rows")
    if not total:
        return dict.fromkeys(sizes, 0)
    counts = {name: n * size // total for name, size in sizes.items()}
    by_remainder = sorted(sizes, key=lambda name: (-(n * sizes[name] % total), name))
    for name in by_remainder[: n - sum(counts.values())]:
        counts[name] += 1
    return counts


def unique(rows: Sequence[Row], key: str = "prompt_id") -> list[Row]:
    """First row of each key, in order."""
    first: dict[Any, Row] = {}
    for row in rows:
        first.setdefault(row[key], row)
    return list(first.values())


def draw(rng: random.Random, rows: Sequence[Row], k: int) -> list[Row]:
    """``k`` rows at random, returned in source order."""
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), k))]


def sample(rows: Sequence[Row], n: int, seed: int, language: str, stratum: str = "global_or_local") -> Sample:
    """``n`` rows stratified by ``stratum``; every row when there are fewer than ``n``."""
    if len(rows) <= n:
        return Sample(list(rows), len(rows) < n)
    strata: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        strata.setdefault(str(row[stratum]), []).append(index)
    counts = allocate(n, {name: len(members) for name, members in strata.items()})
    rng = language_rng(seed, language)
    picked = [i for name in sorted(strata) for i in rng.sample(strata[name], counts[name])]
    return Sample([rows[i] for i in sorted(picked)], False)


class Screen:
    """Drops rows a runtime would misread or truncate, and counts them."""

    def __init__(self, tokenizer: Any, limit: int = render.MAX_RENDERED_TOKENS) -> None:
        self._tokenizer = tokenizer
        self._limit = limit
        self._tokens: dict[str, int] = {}
        self.dropped_special_token_text = 0
        self.dropped_over_length = 0
        self.dropped_blank = 0
        self.dropped_home_path = 0

    def tokens(self, prompt: str) -> int:
        """Token ids the runtime evaluates for the rendered prompt."""
        if prompt not in self._tokens:
            rendered = render.render(self._tokenizer, prompt).rendered_prompt
            self._tokens[prompt] = render.token_count(self._tokenizer, rendered)
        return self._tokens[prompt]

    def keep(self, rows: Sequence[Row], fields: Sequence[str] = ("prompt",)) -> list[Row]:
        """Rows with text in every field, no special-token text or home path, rendering within the limit."""
        kept = []
        for row in rows:
            if not all(row[field].strip() for field in fields):
                self.dropped_blank += 1
            elif any(token in row[field] for field in fields for token in SPECIAL_TOKENS):
                self.dropped_special_token_text += 1
            elif any(HOME_PATH.search(row[field]) for field in fields):
                self.dropped_home_path += 1
            elif self.tokens(row["prompt"]) > self._limit:
                self.dropped_over_length += 1
            else:
                kept.append(row)
        return kept

    def max_tokens(self, rows: Sequence[Row]) -> int:
        return max((self.tokens(row["prompt"]) for row in rows), default=0)
