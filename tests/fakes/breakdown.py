"""Synthetic tagged prompts with an outcome per column, their stored answers, and the breakdown built from them."""

import json
from collections import defaultdict
from pathlib import Path

from driftmap.core.records import Decoding, GenerationRow, LabelRow
from driftmap.core.schemas import SCHEMA_GENERATION
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
# after an outcome: its answer ran into the cap on new tokens
CUT = "*"


def prompts(language, categories, count, before, after=None, origin="global"):
    """``count`` prompts of one language and tag set; ``before`` and ``after`` map an outcome to how many take it.

    The outcomes are dealt in the order given and the rest refuse; ``after`` defaults to ``before``. An outcome
    ending in ``*`` is an answer cut off at the token cap.
    """

    def spread(outcomes):
        listed = [name for name, times in (outcomes or {}).items() for _ in range(times)]
        return listed + ["refused"] * (count - len(listed))

    outcomes = zip(spread(before), spread(before if after is None else after), strict=True)
    return [(language, tuple(categories), origin, first, second) for first, second in outcomes]


def answer(row: LabelRow, cut: bool = False) -> GenerationRow:
    """The stored answer a label row was made from; ``cut`` when it ran into the cap on new tokens."""
    return GenerationRow(
        schema=SCHEMA_GENERATION,
        checkpoint=row.checkpoint,
        language=row.language,
        set=row.set,
        prompt_id=row.prompt_id,
        prompt=row.prompt,
        rendered_prompt=f"<rendered> {row.prompt}",
        rendered_prefix_sha256=row.rendered_prefix_sha256,
        response="placeholder response",
        decoding=Decoding(runtime="transformers"),
        status="ok",
        error=None,
        prompt_tokens=12,
        generated_tokens=256 if cut else 40,
        stop_reason="length" if cut else "eos",
    )


def write_answers(root: Path, rows: list[GenerationRow]) -> Path:
    """Lay rows out as ``<checkpoint dir>/<set>/<language>.jsonl``."""
    files = defaultdict(list)
    for row in rows:
        folder = Path(row.checkpoint.partition("@")[0].rpartition("/")[2])
        files[folder / row.set / f"{row.language}.jsonl"].append(row)
    for path, group in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("".join(row.model_dump_json(by_alias=True) + "\n" for row in group), encoding="utf-8")
    return root


def lay(
    root: Path,
    plan,
    *,
    set_="harmful",
    columns=("baseline", "drifted"),
    responses: Path | None = None,
    checkpoints: dict[str, str] | None = None,
) -> dict[tuple[str, str], Tags]:
    """Write the label rows of a plan under ``root`` and, with ``responses``, their stored answers there.

    ``checkpoints`` names the checkpoint of a column where it is not the usual one. Returns the tags of the plan's
    prompts by language and prompt id.
    """
    rows, answers, tagged, seen = [], [], {}, {}
    for language, categories, origin, *outcomes in plan:
        index = seen[language] = seen.get(language, -1) + 1
        for column, outcome in zip(columns, outcomes, strict=False):
            refusal, harm = LABELS[outcome.removesuffix(CUT)]
            named = {"checkpoint": checkpoints[column]} if column in (checkpoints or {}) else {}
            rows.append(label(column, language, index, set_=set_, refusal=refusal, response_harm=harm, **named))
            answers += [answer(rows[-1], outcome.endswith(CUT))] if responses is not None else []
        tagged[language, rows[-1].prompt_id] = Tags(categories, origin)
    write_labels(root, rows)
    if responses is not None:
        write_answers(responses, answers)
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
        prompts(
            "en",
            [HATE],
            30,
            {"harmful*": 8, "harmful": 1, "harmless*": 4, "refused*": 3},
            {"harmful*": 3, "harmful": 1, "harmless": 6},
        )
        + prompts("en", [SELF_HARM], 12, {"harmful": 2}, {"harmful": 3})
        + prompts("hi", [HATE, PROFANITY], 20, {"harmful*": 3, "unscored": 1}, {"harmful*": 1}, origin="local")
        + prompts("hi", [SELF_HARM], 8, {"harmful*": 4})
    ),
    "strip": (
        prompts("en", [HATE], 30, {"harmful*": 12, "harmless*": 2}, {"harmful*": 3, "harmless": 6})
        + prompts("en", [SELF_HARM], 12, {"harmful": 2}, {"harmful": 1})
        + prompts("hi", [HATE, PROFANITY], 20, {"harmful*": 5}, {"harmful*": 2}, origin="local")
        + prompts("hi", [SELF_HARM], 8, {"harmful*": 4})
    ),
}


def two_conditions(tmp_path: Path, settings: Settings) -> Breakdown:
    """A breakdown of 70 prompts in two languages under both conditions, baseline and fine-tuned, answers read."""
    labels = {condition: tmp_path / "breakdown" / condition for condition in TWO_CONDITIONS}
    responses = {condition: tmp_path / "answers" / condition for condition in TWO_CONDITIONS}
    tagged = {
        condition: lay(labels[condition], plan, responses=responses[condition])
        for condition, plan in TWO_CONDITIONS.items()
    }
    return build(labels, {"harmful": tagged["keep"]}, [], settings, responses=responses)
