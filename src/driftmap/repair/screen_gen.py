"""Screen generations from weights already in memory, decoded like every reported column."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from statistics import fmean
from typing import Any, NamedTuple

from auditkit.model import Model

from driftmap.core.records import Decoding, GenerationRow
from driftmap.gen.adapters import RawPromptAdapter
from driftmap.gen.hf_backend import run_config
from driftmap.gen.rows import Prompt, benign_rows, run_batch

# prompts per language and set; the screen is never reported
PER_LANGUAGE = 20
SETS = ("harmful", "benign")
SCREEN_FILE = "alpha_screen.json"


def _tree(mode: str) -> str:
    return "smoke/" if mode == "smoke" else ""


def screen_path(mode: str) -> str:
    """File of the screen record in the results repo; a smoke run keeps its own."""
    return _tree(mode) + SCREEN_FILE


class Utility(NamedTuple):
    """Mean chrF over every benign row, with how many were answered and how many were not and count as 0."""

    chrf: float
    scored: int
    empty: int


def checkpoint_id(drifted: str, aligned: str, alpha: float) -> str:
    """Name of ``drifted`` moved ``alpha`` toward ``aligned`` (both ``repo@sha``); not a repo reference, on purpose."""
    return f"{drifted}+wiseft-{alpha:g}+{aligned.rpartition('@')[2][:12]}"


def rows_path(slug: str, alpha: float | str, mode: str) -> str:
    """File of one alpha's screen rows, both sets, in the responses repo; ``*`` names every alpha's."""
    name = alpha if isinstance(alpha, str) else f"{alpha:g}"
    return f"{_tree(mode)}screen/{slug}/alpha-{name}.jsonl"


def backend(model: Any, tokenizer: Any, name: str, batch_size: int) -> Model:
    """AuditKIT's transformers backend over a loaded model, so nothing is read from disk; ``unload()`` drops it."""
    from auditkit.model.hf_gen import HFGenModel
    from transformers import pipeline

    generator = HFGenModel(model=name, batch_size=batch_size)
    # the backend loads a checkpoint only while it has no pipeline
    generator._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator


def screen(
    generator: Any,
    tokenizer: Any,
    units: Iterable[Any],
    *,
    checkpoint: str,
    decoding: Decoding,
    batch_size: int,
    between: Callable[[], object] | None = None,
) -> list[GenerationRow]:
    """One row per prompt of every unit (``set``, ``language``, ``rows``); ``between`` runs before each; OutOfMemory."""
    config = replace(run_config(batch_size), max_tokens=decoding.max_new_tokens)
    rows: list[GenerationRow] = []
    for unit in units:
        if between is not None:
            between()
        rows += run_batch(
            generator,
            [Prompt.of(tokenizer, row["prompt_id"], row["prompt"]) for row in unit.rows],
            adapter=RawPromptAdapter(),
            config=config,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
            set=unit.set,
            language=unit.language,
            decoding=decoding,
        )
    return rows


def utility(rows: Sequence[GenerationRow], units: Iterable[Any]) -> Utility:
    """chrF of the benign rows against the units' targets; a row without a usable answer scores 0, never drops out."""
    targets = {row["prompt_id"]: row["target"] for unit in units if unit.set == "benign" for row in unit.rows}
    scores = [label.chrf for label in benign_rows([row for row in rows if row.set == "benign"], targets)]
    scored = sum(score is not None for score in scores)
    return Utility(fmean(score or 0.0 for score in scores) if scores else 0.0, scored, len(scores) - scored)
