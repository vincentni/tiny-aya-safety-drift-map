"""Files in the shape of MBBQ, small and made up: no text of the dataset is in them.

They are made from the made-up BBQ files, so the two fit: an English row repeats the BBQ row of the same id, but
for the items of ``RETOLD``, which keep only its question, and of ``OTHER_QUESTION``, which ask one BBQ does not
hold. The other languages mark every text with their code, show the options in another order where the item's
number is odd, the Dutch files state the other person as the answer of the twins of ``OTHER_PERSON``, and the
Turkish files state no answer for the twins of ``NO_ANSWER``.
"""

import hashlib
import json
from pathlib import Path

from driftmap.attribution import mbbq
from tests.fakes import bbq as fake_bbq

# by the place of an item among the fourteen of its template and polarity
OTHER_PERSON, RETOLD, OTHER_QUESTION, NO_ANSWER = 4, 11, 12, 13
# the tags of the gendered options, as the files of each language spell them
GENDERED = {
    "en": ("woman", "man"),
    "es": ("mujer", "hombre"),
    "nl": ("vrouw", "jongen"),
    "tr": ("kadın", "erkek çocuk"),
}
TEXTS = ("context", "question", "ans0", "ans1", "ans2")


def _turned(row: dict, by: int) -> dict:
    """The row with every option moved ``by`` places down, the label with it."""
    moved = {f"ans{place}": row[f"ans{(place + by) % 3}"] for place in range(3)}
    tags = {f"ans{place}": row["answer_info"][f"ans{(place + by) % 3}"] for place in range(3)}
    return row | moved | {"answer_info": tags, "label": (row["label"] - by) % 3}


def _row(row: dict, category: str, code: str) -> dict:
    """One BBQ-shaped row as the file of ``code`` holds it."""
    number = (row["example_id"] // 2) % fake_bbq.ITEMS_PER_POLARITY
    ambiguous = row["context_condition"] == "ambig"
    row = dict(row, answer_info={name: list(tagged) for name, tagged in row["answer_info"].items()})
    if number == RETOLD:
        row["context"] = f"Later that day, {row['context']}"
    if number == OTHER_QUESTION:
        row["question"] = row["question"].replace("ticket", "umbrella")
    if category == "Gender_identity":
        names = dict(zip(("riders", "walkers"), GENDERED[code], strict=True))
        row["answer_info"] = {name: [text, names.get(tag, tag)] for name, (text, tag) in row["answer_info"].items()}
        row["additional_metadata"] = {"stereotyped_groups": ["F"]}
    if code == "nl" and number == OTHER_PERSON and not ambiguous:
        unknown = next(place for place in range(3) if row["answer_info"][f"ans{place}"][1] == "unknown")
        row["label"] = next(place for place in range(3) if place not in (unknown, row["label"]))
    if code != "en":
        row |= {key: f"[{code}] {row[key]}" for key in TEXTS}
        row["answer_info"] = {name: [f"[{code}] {text}", tag] for name, (text, tag) in row["answer_info"].items()}
        row = _turned(row, number % 2 * (1 + mbbq.LANGUAGES.index(code) % 2))
    if code == "tr" and number == NO_ANSWER and not ambiguous:
        row["label"] = None
    return row


def write(directory: Path, bbq_directory: Path) -> dict[str, str]:
    """Write the files under ``directory`` in the layout of the MBBQ repository, from the BBQ-shaped files under
    ``bbq_directory``; their sha256 by path."""
    for category in mbbq.CATEGORIES:
        source = (bbq_directory / "data" / f"{category}.jsonl").read_text(encoding="utf-8").split("\n")
        rows = [json.loads(line) for line in source if line]
        for code in mbbq.LANGUAGES:
            path = directory / mbbq.data_file(category, code)
            path.parent.mkdir(parents=True, exist_ok=True)
            made = [_row(row, category, code) for row in rows]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in made), encoding="utf-8")
    (directory / "LICENSE").write_text("A made-up licence.\n", encoding="utf-8")
    (directory / mbbq.README).write_text(f"# Made up\n\n**Note:** {mbbq.NOTE}\n", encoding="utf-8")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in mbbq.FILES}
