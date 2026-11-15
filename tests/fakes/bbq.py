"""Files in the shape of BBQ, small and made up: no text of the dataset is in them."""

import csv
import hashlib
import json
from pathlib import Path

from driftmap.attribution import bbq, build

TEMPLATES = 10
ITEMS_PER_POLARITY = 14
PHRASINGS = (*build.SENTENCES, *build.HELDOUT_PHRASINGS)
VALUES = ("thrift", "thrift, punctuality", "tidiness", "courage")
COLUMNS = (
    "category",
    "question_index",
    "example_id",
    "target_loc",
    "Relevant_social_values",
    "Known_stereotyped_groups",
)


def _item(category: str, template: int, polarity: str, number: int, first: int) -> tuple[list[dict], int]:
    """An ambiguous row and its twin, and the place of the stereotype-aligned option."""
    one, other = (f"{category} rider {template}-{polarity}-{number}{side}" for side in "ab")
    unknown = number % 3
    people = [place for place in range(3) if place != unknown]
    texts = dict(zip(people, (one, other), strict=True)) | {unknown: PHRASINGS[number % len(PHRASINGS)]}
    groups = dict(zip(people, ("riders", "walkers"), strict=True)) | {unknown: "unknown"}
    context = f"{one} and {other} waited at stop {template} of the {category} line."
    common = {
        "question_index": str(template),
        "question_polarity": polarity,
        "category": category,
        "question": f"Who {'dropped' if polarity == 'neg' else 'kept'} ticket {template}?",
        "answer_info": {f"ans{place}": [texts[place], groups[place]] for place in range(3)},
        "additional_metadata": {"stereotyped_groups": ["riders"]},
    } | {f"ans{place}": texts[place] for place in range(3)}
    rows = [
        common | {"example_id": first, "context_condition": "ambig", "context": context, "label": unknown},
        common
        | {
            "example_id": first + 1,
            "context_condition": "disambig",
            "context": f"{context} Then {texts[people[number % 2]]} searched every pocket.",
            "label": people[number % 2],
        },
    ]
    return rows, people[0]


def write(directory: Path) -> dict[str, str]:
    """Write the files under ``directory`` in the layout of the BBQ repository; their sha256 by path."""
    metadata = []
    for name in bbq.FILES:
        if name == bbq.METADATA:
            continue
        category = Path(name).stem
        rows: list[dict] = []
        for template in range(1, TEMPLATES + 1):
            for polarity in ("neg", "nonneg"):
                for number in range(ITEMS_PER_POLARITY):
                    pair, aligned = _item(category, template, polarity, number, len(rows))
                    rows += pair
                    metadata += [
                        (category, template, row["example_id"], aligned, VALUES[template % 4], f"{category} riders")
                        for row in pair
                    ]
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path = directory / bbq.METADATA
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(metadata)
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in bbq.FILES}
