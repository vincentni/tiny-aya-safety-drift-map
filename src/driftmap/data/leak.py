"""Whether training rows stay apart from the prompts a fine-tune is evaluated on."""

from collections.abc import Mapping, Sequence
from typing import Any


def _plain(text: object) -> str:
    return " ".join(str(text).casefold().split())


def shared(training: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]]) -> list[int]:
    """Positions, from 1, of the training rows that are among ``prompts``.

    A training row is among them when its last user turn is the text of one, case and spacing aside, or when both
    carry the same ``base_item``. LookupError or TypeError for a row of another shape.
    """
    asked = {_plain(row["prompt"]) for row in prompts}
    items = {row["base_item"] for row in prompts if row.get("base_item") is not None}
    return [
        at
        for at, row in enumerate(training, 1)
        if _plain(row["prompt"][-1]["content"]) in asked or row.get("base_item") in items
    ]


def of_a_trained_type(notes: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]]) -> list[int]:
    """Positions, from 1, of the prompts whose question type the training rows hold.

    ``notes`` say of each training row its ``base_item``, ``template`` and ``category``, where it has them. A
    prompt names the base item (``bbq_base_item``) and the templates (``bbq_templates``) its question goes back to.
    It is of a trained type when a training row is of that base item or of one of those templates, and when it
    names no template while a training row is of its ``category``. LookupError or TypeError for a row of another
    shape.
    """
    items, templates, categories = (
        {note[key] for note in notes if note.get(key) is not None} for key in ("base_item", "template", "category")
    )
    return [
        at
        for at, row in enumerate(prompts, 1)
        if row["bbq_base_item"] in items
        or templates.intersection(row["bbq_templates"])
        or (not row["bbq_templates"] and row["category"] in categories)
    ]
