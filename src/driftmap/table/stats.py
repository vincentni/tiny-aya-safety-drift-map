"""Paired differences between two columns and the gap between two language groups."""

import math
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from auditkit._bootstrap import paired_bootstrap
from scipy.stats import binomtest, bootstrap

from ..core.records import LabelRow
from ..core.schemas import DriftmapError, ExitCode
from .rates import score

BOOTSTRAP_SEED = 42
N_RESAMPLES = 1000


class Delta(NamedTuple):
    """A difference of means; every estimate is None below two observations per side."""

    n_pairs: int
    delta: float | None
    ci: tuple[float, float] | None
    p_value: float | None


NO_ESTIMATE = (None, None, None)


def _scores(rows: Iterable[LabelRow], metric: str) -> dict[str, float]:
    return {f"{row.language}/{row.prompt_id}": value for row in rows if (value := score(row, metric)) is not None}


def _interval(samples: tuple, statistic, *, paired: bool, n_resamples: int, seed: int) -> tuple[float, float]:
    result = bootstrap(
        samples,
        statistic,
        paired=paired,
        vectorized=True,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="percentile",
        rng=seed,
    )
    low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise DriftmapError(ExitCode.INTEGRITY, f"bootstrap interval is not finite: ({low}, {high})")
    return low, high


def sign_test(differences: np.ndarray) -> float:
    """Exact two-sided sign test on the non-zero differences; 1.0 when there are none."""
    up, down = int((differences > 0).sum()), int((differences < 0).sum())
    return float(binomtest(up, up + down).pvalue) if up + down else 1.0


def paired_delta(
    earlier: Iterable[LabelRow],
    later: Iterable[LabelRow],
    metric: str,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Delta:
    """Later minus earlier over prompts scored in both columns: mean, seeded percentile interval, exact sign-test p."""
    before, after = _scores(earlier, metric), _scores(later, metric)
    both = sorted(before.keys() & after.keys())
    # p-value unused: its resamples are not centred on zero
    library = paired_bootstrap([(before[key], after[key]) for key in both], n_resamples=n_resamples, seed=seed)
    if "error" in library:
        return Delta(len(both), *NO_ESTIMATE)
    first, second = (np.array([scores[key] for key in both]) for scores in (before, after))
    delta = float(np.mean(second - first))
    if library["n"] != len(both) or not math.isclose(delta, library["delta"], abs_tol=1e-6):
        theirs = f"{library['delta']} over {library['n']} pairs"
        raise DriftmapError(
            ExitCode.INTEGRITY, f"paired delta {delta} over {len(both)} pairs disagrees with the library's {theirs}"
        )
    interval = _interval(
        (second, first), lambda x, y, axis: np.mean(x - y, axis=axis), paired=True, n_resamples=n_resamples, seed=seed
    )
    return Delta(len(both), delta, interval, sign_test(second - first))


def group_gap(
    low: Iterable[LabelRow],
    high: Iterable[LabelRow],
    metric: str,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Delta:
    """Mean of ``low`` minus mean of ``high`` for two unpaired groups, with a seeded percentile interval."""
    first, second = (np.array(list(_scores(rows, metric).values())) for rows in (low, high))
    if min(len(first), len(second)) < 2:
        return Delta(0, *NO_ESTIMATE)
    interval = _interval(
        (first, second),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        paired=False,
        n_resamples=n_resamples,
        seed=seed,
    )
    return Delta(0, float(np.mean(first) - np.mean(second)), interval, None)


def change_gap(
    first: tuple[Iterable[LabelRow], Iterable[LabelRow]],
    second: tuple[Iterable[LabelRow], Iterable[LabelRow]],
    metric: str,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Delta:
    """The paired change of ``first`` minus that of ``second``, each an earlier and a later column, the two over
    unrelated prompts; with a seeded percentile interval from resampling both sets of prompts."""
    moved = []
    for earlier, later in (first, second):
        before, after = _scores(earlier, metric), _scores(later, metric)
        moved.append(np.array([after[key] - before[key] for key in sorted(before.keys() & after.keys())]))
    if min(map(len, moved)) < 2:
        return Delta(0, *NO_ESTIMATE)
    interval = _interval(
        tuple(moved),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        paired=False,
        n_resamples=n_resamples,
        seed=seed,
    )
    return Delta(0, float(np.mean(moved[0]) - np.mean(moved[1])), interval, None)
