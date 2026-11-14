"""Synthetic tagged prompts with an outcome per column, and the breakdown built from them."""

import json
from pathlib import Path

from driftmap.core.settings import Settings
from driftmap.table.breakdown import Breakdown, Tags, build
from tests.fakes.labels import label, write_labels

REPRESENTATION = "Harms of Representation Allocation and Quality of Service"
HATE, SELF_HARM, PROFANITY = "Hate Speech", "Self-Harm", "Profanity"
# refusal and response_harm of each outcome
LABELS = {
    "refused": ("yes", "no"),
    "harmless": ("no", "no"),
    "harmful": ("no", "yes"),
    "unscored": ("unscored", "unscored"),
}


def prompts(language, categories, count, before, after=None, origin="global"):
    """``count`` prompts of one language and tag set; ``before`` and ``after`` map an outcome to how many take it.

    The outcomes are dealt in the order given and the rest refuse; ``after`` defaults to ``before``.
    """

    def spread(outcomes):
        listed = [name for name, times in (outcomes or {}).items() for _ in range(times)]
        return listed + ["refused"] * (count - len(listed))

    outcomes = zip(spread(before), spread(before if after is None else after), strict=True)
    return [(language, tuple(categories), origin, first, second) for first, second in outcomes]


def lay(root: Path, plan, *, set_="harmful", columns=("baseline", "drifted")) -> dict[tuple[str, str], Tags]:
    """Write the label rows of a plan under ``root``; returns the tags of its prompts by language and prompt id."""
    rows, tagged, seen = [], {}, {}
    for language, categories, origin, *outcomes in plan:
        index = seen[language] = seen.get(language, -1) + 1
        for column, outcome in zip(columns, outcomes, strict=False):
            refusal, harm = LABELS[outcome]
            rows.append(label(column, language, index, set_=set_, refusal=refusal, response_harm=harm))
        tagged[language, rows[-1].prompt_id] = Tags(categories, origin)
    write_labels(root, rows)
    return tagged


def prompt_rows(tagged: dict[tuple[str, str], Tags], language: str) -> list[dict]:
    """The prompt-file rows of one language, as the evaluation dataset holds them."""
    return [
        {
            "prompt_id": prompt_id,
            "language": language,
            "prompt": f"placeholder {prompt_id}",
            "global_or_local": tag.global_or_local,
            "harm_category": json.dumps(list(tag.harm_categories)),
        }
        for (code, prompt_id), tag in tagged.items()
        if code == language
    ]


# per condition: outcomes at baseline, then after fine-tuning, of the same prompts
TWO_CONDITIONS = {
    "keep": (
        prompts("en", [HATE], 30, {"harmful": 9, "harmless": 4}, {"harmful": 4, "harmless": 6})
        + prompts("en", [SELF_HARM], 12, {"harmful": 2}, {"harmful": 3})
        + prompts("hi", [HATE, PROFANITY], 20, {"harmful": 3, "unscored": 1}, {"harmful": 1}, origin="local")
        + prompts("hi", [SELF_HARM], 8, {"harmful": 4})
    ),
    "strip": (
        prompts("en", [HATE], 30, {"harmful": 12, "harmless": 2}, {"harmful": 3, "harmless": 6})
        + prompts("en", [SELF_HARM], 12, {"harmful": 2}, {"harmful": 1})
        + prompts("hi", [HATE, PROFANITY], 20, {"harmful": 5}, {"harmful": 2}, origin="local")
        + prompts("hi", [SELF_HARM], 8, {"harmful": 4})
    ),
}


def two_conditions(tmp_path: Path, settings: Settings) -> Breakdown:
    """A breakdown of 70 prompts in two languages under both conditions, baseline and fine-tuned."""
    labels = {condition: tmp_path / "breakdown" / condition for condition in TWO_CONDITIONS}
    tagged = {condition: lay(labels[condition], plan) for condition, plan in TWO_CONDITIONS.items()}
    return build(labels, {"harmful": tagged["keep"]}, [], settings)
