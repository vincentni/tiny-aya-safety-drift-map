"""Cross-set dedup: no training row repeats a protected prompt."""

import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import NamedTuple

from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample

from driftmap.data.sampler import Row

THRESHOLD = 0.85


class Deduped(NamedTuple):
    rows: list[Row]
    # training rows dropped as a copy of a protected prompt
    copies: int
    # training rows dropped as a copy of an earlier training prompt
    repeats: int


def _key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _samples(kind: str, rows: Sequence[Row]) -> list[DataSample]:
    return [
        DataSample(id=f"{kind}-{i}", source_uri=kind, instruction=_key(row["prompt"])) for i, row in enumerate(rows)
    ]


def _keep_first(samples: list[DataSample], threshold: float) -> list[DataSample]:
    return MinHashDeduplicator(threshold=threshold).run(ExactDeduplicator().run(samples))


def _trigrams(text: str) -> frozenset[str]:
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


def cross_dedup(protected: Sequence[Row], train: Sequence[Row], threshold: float = THRESHOLD) -> Deduped:
    """Training rows whose prompt neither equals nor near-matches a protected or earlier training prompt."""
    guard = sorted(filter(None, {_trigrams(_key(row["prompt"])) for row in protected}), key=len)
    sizes = [len(grams) for grams in guard]

    def copied(text: str) -> bool:
        grams = _trigrams(text)
        # sets further apart in size cannot reach the threshold
        nearby = guard[bisect_left(sizes, threshold * len(grams)) : bisect_right(sizes, len(grams) / threshold)]
        return any(len(grams & other) >= threshold * len(grams | other) for other in nearby)

    # exact similarity first: the library compares hashed signatures and misses some copies
    candidates = [sample for sample in _samples("train", train) if not copied(sample.instruction)]
    distinct = _keep_first(candidates, threshold)
    # the library keeps the first-seen row, so protected rows go first
    survivors = _keep_first(_samples("protected", protected) + distinct, threshold)
    kept = [sample for sample in survivors if sample.source_uri == "train"]
    rows = [train[int(sample.id.removeprefix("train-"))] for sample in kept]
    return Deduped(rows, len(train) - len(candidates) + len(distinct) - len(kept), len(candidates) - len(distinct))
