import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from auditkit._bootstrap import paired_bootstrap
from scipy.stats import binomtest

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.table import stats
from driftmap.table.rates import score
from driftmap.table.stats import Delta, change_gap, group_gap, paired_delta
from tests.fakes.labels import label

FIXTURE = Path(__file__).parents[2] / "fixtures" / "bootstrap_600_pairs.json"


def columns(pairs):
    earlier = [label("baseline", "en", i, harm=a) for i, (a, _) in enumerate(pairs)]
    later = [label("drifted", "en", i, harm=b) for i, (_, b) in enumerate(pairs)]
    return earlier, later


def benign(column, values):
    return [label(column, "en", i, set_="benign", chrf=value) for i, value in enumerate(values)]


def test_a_paired_delta_is_later_minus_earlier_over_rows_scored_in_both_columns():
    """The 600-pair fixture gives +1/6, a seeded percentile interval and an exact sign-test p (TS-TBL-3)."""
    earlier, later = columns(json.loads(FIXTURE.read_text())["pairs"])

    delta = paired_delta(earlier, later, "safe")

    assert delta.n_pairs == 600 and delta.delta == 1 / 6
    assert delta.ci == pytest.approx((0.1299583333333334, 0.20166666666666666), abs=1e-12)
    assert all(map(math.isfinite, delta.ci))
    # 120 pairs turn safe, 20 turn harmful
    assert delta.p_value == binomtest(120, 140).pvalue == pytest.approx(1.4188339369831328e-18, rel=1e-9)
    assert paired_delta(earlier, later, "safe") == delta
    assert paired_delta(earlier, later, "safe", seed=7).ci != delta.ci
    assert paired_delta(later, earlier, "safe").delta == -1 / 6

    later[300] = label("drifted", "en", 300, harm="unscored")
    dropped = paired_delta(earlier, later, "safe")
    assert dropped.n_pairs == 599 and dropped.delta == pytest.approx(100 / 599)

    assert paired_delta(earlier[:1], later[:1], "safe") == Delta(1, None, None, None)
    assert paired_delta(earlier, [], "safe") == Delta(0, None, None, None)

    shifted = paired_delta(benign("baseline", [0.5] * 100), benign("drifted", [0.55] * 100), "chrf")
    assert shifted.delta == pytest.approx(0.05) and shifted.ci == pytest.approx((0.05, 0.05))
    assert shifted.ci[0] > 0 and shifted.p_value < 1e-20

    same = paired_delta(earlier, earlier, "safe")
    assert (same.delta, same.ci, same.p_value) == (0.0, (0.0, 0.0), 1.0)
    assert "significant" not in Path(stats.__file__).read_text()


def test_the_library_bootstrap_p_value_stays_near_one_half_for_a_large_shift():
    """AuditKIT resamples the differences without centring them, so its p-value cannot detect a shift."""
    pairs = [("yes" if i < 200 else "no", "yes" if i < 100 else "no") for i in range(600)]
    scores = [(float(a == "no"), float(b == "no")) for a, b in pairs]

    library = paired_bootstrap(scores, n_resamples=1000, seed=42)
    ours = paired_delta(*columns(pairs), "safe")

    assert library["delta"] == round(ours.delta, 6) == 0.166667
    assert 0.4 < library["p_value"] < 0.6
    assert ours.p_value == binomtest(100, 100).pvalue < 1e-20
    assert ours.ci[0] > 0.1
    assert paired_bootstrap([(0.5, 0.55)] * 100, n_resamples=1000, seed=42)["p_value"] == 1.0


def test_a_group_gap_is_an_unpaired_difference_with_a_seeded_interval():
    low = [label("baseline", "hi", i, harm="yes" if i < 30 else "no") for i in range(100)]
    high = [label("baseline", "en", i, harm="yes" if i < 20 else "no") for i in range(200)]
    high.append(label("baseline", "en", 200, harm="unscored"))

    gap = group_gap(low, high, "safe")

    assert gap.n_pairs == 0 and gap.p_value is None
    assert gap.delta == pytest.approx(0.7 - 0.9)
    assert gap.ci[0] < gap.delta < gap.ci[1] < 0
    assert group_gap(low, high, "safe") == gap
    assert group_gap(low[:1], high, "safe") == Delta(0, None, None, None)


def test_a_change_gap_is_one_paired_change_minus_another_over_unrelated_prompts():
    def pair(language, n, before, after):
        earlier = [label("baseline", language, i, harm="yes" if i < before else "no") for i in range(n)]
        later = [label("drifted", language, i, harm="yes" if i < after else "no") for i in range(n)]
        return earlier, later

    first, second = pair("hi", 100, 30, 10), pair("en", 200, 40, 30)
    second[1].append(label("drifted", "en", 200, harm="yes"))

    gap = change_gap(first, second, "safe")

    # the safe rate rises by 20 of 100 in the first pair and by 10 of the 200 scored in both in the second
    assert gap.n_pairs == 0 and gap.p_value is None
    assert gap.delta == pytest.approx(0.2 - 0.05)
    assert 0 < gap.ci[0] < gap.delta < gap.ci[1]
    assert change_gap(first, second, "safe") == gap
    assert change_gap(second, first, "safe").delta == pytest.approx(-gap.delta)
    assert change_gap((first[0][:1], first[1]), second, "safe") == Delta(0, None, None, None)


def test_the_library_gets_earlier_later_pairs_and_leaves_no_cache_file(monkeypatch):
    seen = []
    real = stats.paired_bootstrap
    monkeypatch.setattr(
        stats, "paired_bootstrap", lambda pairs, **options: seen.append(pairs) or real(pairs, **options)
    )

    delta = paired_delta(*columns([("yes", "no"), ("yes", "no"), ("no", "no"), ("no", "yes")]), "safe")

    assert seen == [[(0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]]
    assert (delta.n_pairs, delta.delta, delta.p_value) == (4, 0.25, 1.0)
    assert not (Path(os.environ["XDG_CACHE_HOME"]) / "auditkit").exists()


def test_a_library_disagreement_or_a_non_finite_interval_stops_the_estimate(monkeypatch):
    earlier, later = columns([("yes", "no")] * 3 + [("no", "no")] * 3)
    real = stats.paired_bootstrap

    def altered(**changes):
        return lambda pairs, **options: {**real(pairs, **options), **changes}

    for changes, message in (({"delta": -0.5}, "disagrees"), ({"n": 5}, "5 pairs")):
        with monkeypatch.context() as patch:
            patch.setattr(stats, "paired_bootstrap", altered(**changes))
            with pytest.raises(DriftmapError, match=message) as raised:
                paired_delta(earlier, later, "safe")
        assert raised.value.exit_code == ExitCode.INTEGRITY

    interval = SimpleNamespace(confidence_interval=SimpleNamespace(low=float("nan"), high=0.5))
    monkeypatch.setattr(stats, "bootstrap", lambda *args, **options: interval)
    with pytest.raises(DriftmapError, match="not finite") as raised:
        paired_delta(earlier, later, "safe")
    assert raised.value.exit_code == ExitCode.INTEGRITY


def test_a_row_without_chrf_is_left_out_not_scored_zero():
    earlier, later = benign("baseline", [0.5, 0.5, None]), benign("drifted", [0.6, 0.6, 0.9])

    assert score(earlier[2], "chrf") is None and score(later[2], "chrf") == 0.9
    delta = paired_delta(earlier, later, "chrf")
    assert delta.n_pairs == 2 and delta.delta == pytest.approx(0.1)
