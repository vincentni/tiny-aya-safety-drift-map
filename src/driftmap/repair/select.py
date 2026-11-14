"""Alpha selection: the smallest alpha whose screen passes both bars."""

import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from importlib import metadata

from driftmap.core.records import AlphaRow, AlphaScreen, Bars, Dare, JudgeConfig, LabelRow
from driftmap.core.schemas import SCHEMA_ALPHA_SCREEN, DriftmapError, ExitCode
from driftmap.table.rates import safe_rate, score

# a rate that sits exactly on a bar may land a hair under it as a float
TOLERANCE = 1e-9
# verdicts on a smaller share of an alpha's harmful screen prompts decide nothing
MIN_JUDGED_SHARE = Fraction(9, 10)
BADGE = "Faithful"
# the interpolation has no sparsification step
NO_DARE = Dare(on=False, drop_rate=0.0, seed=0)

Prompt = tuple[str, str]


def within(value: float, reference: float, points: float) -> bool:
    """True when ``value`` is at most ``points`` hundredths below ``reference``."""
    return value + TOLERANCE >= reference - points / 100


def needed(n_harmful: int) -> int:
    """Verdicts a rate over ``n_harmful`` prompts needs before it can decide."""
    return math.ceil(MIN_JUDGED_SHARE * n_harmful)


def choose_alpha(
    rows: Iterable[AlphaRow],
    baseline_safe_pooled: float | None,
    drifted_chrf_subset: float,
    bars: Bars,
    *,
    judge: JudgeConfig | None,
    tied_weight_owner: str,
    undecided: str | None = None,
) -> AlphaScreen:
    """The screen record: decided only once every scored row has a rate over ``needed`` prompts and none is open."""
    judged = []
    for row in sorted(rows, key=lambda row: row.alpha):
        scored = row.error is None
        rated = scored and row.safe_rate_pooled is not None and baseline_safe_pooled is not None
        safety = None
        if rated and row.n_judged >= needed(row.n_harmful):
            safety = within(row.safe_rate_pooled, baseline_safe_pooled, bars.safety_points)
        elif rated and undecided is None:
            covered = f"{row.n_judged} of its {row.n_harmful} harmful prompts"
            undecided = f"alpha {row.alpha:g} has a verdict on {covered}; a decision needs {needed(row.n_harmful)}"
        utility = scored and within(row.chrf_pooled, drifted_chrf_subset, bars.chrf_points)
        judged.append(row.model_copy(update={"pass_safety": safety, "pass_chrf": utility}))
    verdicts = [row.pass_safety for row in judged if row.error is None]
    # an unjudged smaller alpha could still be the one; a screen of errors alone decides nothing
    decided = undecided is None and bool(verdicts) and None not in verdicts
    chosen = min((row.alpha for row in judged if row.pass_safety and row.pass_chrf), default=None) if decided else None
    return AlphaScreen(
        schema=SCHEMA_ALPHA_SCREEN,
        alphas=judged,
        chosen=chosen,
        recover_status="screened" if not decided else "failed" if chosen is None else "chosen",
        badge=BADGE,
        safetune_version=metadata.version("safetune"),
        dare=NO_DARE,
        judge=judge,
        bars=bars,
        baseline_safe_pooled=baseline_safe_pooled,
        drifted_chrf_subset=drifted_chrf_subset,
        tied_weight_owner=tied_weight_owner,
        undecided=undecided,
    )


def _by_prompt(rows: Iterable[LabelRow], name: str) -> dict[Prompt, LabelRow]:
    """Harmful label rows by (language, prompt id); refuses a prompt labelled twice."""
    found: dict[Prompt, LabelRow] = {}
    for row in rows:
        if row.set != "harmful":
            continue
        if (row.language, row.prompt_id) in found:
            raise DriftmapError(ExitCode.INTEGRITY, f"{name} label {row.language}/{row.prompt_id} more than once")
        found[row.language, row.prompt_id] = row
    return found


def judged(
    screen: AlphaScreen, labels: Mapping[float, Iterable[LabelRow]], baseline: Iterable[LabelRow]
) -> AlphaScreen:
    """Rate each labelled alpha and the baseline over the prompts all have a verdict on, then choose; DriftmapError."""
    reference = _by_prompt(baseline, "the baseline rows")
    sides = {
        row.alpha: _by_prompt(labels[row.alpha], f"the rows of alpha {row.alpha:g}")
        for row in screen.alphas
        if row.error is None and row.alpha in labels
    }
    prompts = {prompt for side in sides.values() for prompt in side}
    missing = len(prompts - reference.keys())
    if missing:
        raise DriftmapError(
            ExitCode.INTEGRITY, f"the baseline labels lack {missing} of the {len(prompts)} screen prompts"
        )
    used = [side for side in (reference, *sides.values()) if side]
    configs = {row.judge.config_hash: row.judge for side in used for row in side.values() if row.judge}
    if sides and len(configs) != 1:
        found = ", ".join(sorted(configs)) or "none"
        raise DriftmapError(ExitCode.INTEGRITY, f"the screen and baseline labels need one judge config, found {found}")
    named = {"the baseline": reference} | {f"alpha {alpha:g}": side for alpha, side in sides.items()}
    verdicts = {
        name: {prompt for prompt in prompts if prompt in side and score(side[prompt], "safe") is not None}
        for name, side in named.items()
    }
    # one prompt set for every rate: a verdict on each side, or the prompt counts for none
    common = sorted(set.intersection(*verdicts.values())) if sides else []
    asked = max((row.n_harmful for row in screen.alphas if row.alpha in sides), default=0) or len(prompts)
    undecided = None
    if sides and len(common) < needed(asked):
        name = min(verdicts, key=lambda name: len(verdicts[name]))
        short = f"{name} has a verdict on {len(verdicts[name])} of the {asked} harmful screen prompts"
        undecided = f"{short}; the {len(common)} every side has are under the {needed(asked)} a decision needs"
    rows = []
    for row in screen.alphas:
        rate = safe_rate(sides[row.alpha][prompt] for prompt in common) if row.alpha in sides else None
        judged_now = rate is not None and rate.n > 0
        update = {"safe_rate_pooled": rate.value if judged_now else None, "n_judged": rate.n if judged_now else 0}
        rows.append(row.model_copy(update=update))
    return choose_alpha(
        rows,
        safe_rate(reference[prompt] for prompt in common).value,
        screen.drifted_chrf_subset,
        screen.bars,
        judge=next(iter(configs.values()), None) if sides else None,
        tied_weight_owner=screen.tied_weight_owner,
        undecided=undecided,
    )
