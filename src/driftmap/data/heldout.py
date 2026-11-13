"""Benign held-out rows and the balanced training draw."""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from driftmap.data.sampler import Row, draw, language_rng, unique

# the smallest of the six languages has this many training rows
TRAIN_CAP = 423


class Carve(NamedTuple):
    heldout: dict[str, list[Row]]
    # training rows left per language
    pools: dict[str, list[Row]]
    # split each language's held-out rows came from
    origin: dict[str, str]
    # training rows dropped for sharing a held-out prompt
    duplicates: int


def carve(train: Mapping[str, Sequence[Row]], test: Mapping[str, Sequence[Row]], n: int, seed: int) -> Carve:
    """``n`` held-out rows per language, from test when it has them, else carved from train."""
    heldout, pools, origin, duplicates = {}, {}, {}, 0
    for language in train:
        tested = unique(test.get(language, ()))
        origin[language] = "test" if len(tested) >= n else "train"
        source = tested if origin[language] == "test" else unique(train[language])
        heldout[language] = draw(language_rng(seed, language), source, min(n, len(source)))
        held = {row["prompt_id"] for row in heldout[language]}
        pools[language] = [row for row in train[language] if row["prompt_id"] not in held]
        carved = len(held) if origin[language] == "train" else 0
        duplicates += len(train[language]) - len(pools[language]) - carved
    return Carve(heldout, pools, origin, duplicates)


def balance[T](pools: Mapping[str, Sequence[T]], seed: int, cap: int = TRAIN_CAP) -> dict[str, list[T]]:
    """The same number of rows for every language: the smallest pool, at most ``cap``."""
    size = min([cap, *map(len, pools.values())])
    return {language: draw(language_rng(seed, language), pool, size) for language, pool in pools.items()}
