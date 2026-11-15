"""Draw question pairs in four languages from MBBQ, of question types the attribution fine-tune never saw.

    python -I -m driftmap.attribution.mbbq_build --mbbq MBBQ --bbq BBQ --data DATA --out OUT

``--mbbq`` and ``--bbq`` are folders for the files of the two datasets: the ones missing there are downloaded at
the commits ``mbbq`` and ``bbq`` pin, and all are checked against their recorded sha256. They are read as data,
nothing in them is executed. ``--data`` is the folder ``build`` wrote: its manifest names the categories and the
templates held out of training, and the sidecar of its training set names the base item and the template of every
training row. The draw is seeded: the same inputs give the same files, byte for byte.

An item is an ambiguous MBBQ row and the row after it, its disambiguated twin. A row has the same id, template
and polarity in the English, Spanish, Dutch and Turkish file of a category, though not always the same persons,
the same option order or the same person as the stated answer, so every language's row is read by its own tags.
An item is the row with one id in each of the four files; it is drawn once and asked in all four languages. One
whose twin states no answer in a language, or whose tags do not single out the stereotyped option in a language,
is not drawn.

The question type of an item is found through BBQ: by the English text of its row where BBQ holds that row, else
by its English question. Drawn are items of a category the fine-tune never trained on, and items of the other
categories whose every BBQ template was held out of training. An item whose question BBQ does not hold is not
drawn: its type cannot be shown unseen. What keeps the draw apart from the training rows is this trace: an item's
BBQ base item and BBQ templates, found from its English row within its own category, compared with the base
items and templates the sidecar names. A launch repeats the comparison on the fields written here, against the
sidecar stored at the prompts' commit.

Written under ``--out``: ``mbbq/prompts/benign/<language>.jsonl``, prompt rows in the shape of the test rows of
``build``, the two rows of an item sharing one option order, with the unknown option and the correct person at A,
B and C equally often; and ``mbbq/manifest.json``, which counts every step and records the licence of MBBQ and
its authors' note on what the dataset is for.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from driftmap.attribution import bbq, build, mbbq
from driftmap.attribution.build import LETTERS, by, normal, sha256, write_jsonl
from driftmap.core import ids
from driftmap.data import leak
from driftmap.scripts.drift import META

GROUP, SET = "mbbq", "benign"
# its own stream beside the draw of the training and test items
SEED = f"{build.SEED}:{GROUP}"
# the line that closes a prompt; the English and the Spanish one are those of the test items in those languages
CLOSING = {
    "en": build.LETTER_INSTRUCTION,
    "es": "Responde solo con la letra.",
    "nl": "Antwoord alleen met de letter.",
    "tr": "Yalnızca harfle cevap verin.",
}
# items drawn per category: the one the fine-tune never trained on, then the held-out templates of the others
WANTED = {
    "Gender_identity": 75,
    "Age": 16,
    "Disability_status": 16,
    "Physical_appearance": 16,
    "SES": 15,
    "Sexual_orientation": 12,
}
HELDOUT_CATEGORY, HELDOUT_TEMPLATE = "heldout_category", "heldout_template"
# why an item is not in the pool
TRAINED, NOT_IN_BBQ, NOT_PARALLEL = "trained_template", "question_not_in_bbq", "not_readable_in_every_language"
# the tags of the gendered options in the four languages, by the tag of the group they stand for
ALIASES = {
    **dict.fromkeys(("woman", "girl", "mujer", "chica", "vrouw", "meisje", "kadın", "kız"), "f"),
    **dict.fromkeys(("man", "boy", "hombre", "chico", "jongen", "adam", "erkekçocuk"), "m"),
}


def read_rows(path: Path) -> dict[int, dict]:
    """The rows of a file by their id; a label is an answer index, or None where the file states none."""
    rows = {}
    for row in build.read_jsonl(path):
        row["label"] = None if row.get("label") is None else int(row["label"])
        rows[int(row["example_id"])] = row
    return rows


def _tag(text: str) -> str:
    plain = re.sub(r"[\s_-]+", "", text).casefold()
    return ALIASES.get(plain, plain)


def roles(row: dict) -> dict[str, int] | None:
    """The answer index of the unknown option and of the stereotyped group of a row, by its own tags; None when
    they do not single out one of each."""
    tags = [row["answer_info"][f"ans{index}"][1] for index in range(3)]
    if tags.count("unknown") != 1:
        return None
    unknown = tags.index("unknown")
    stated = {_tag(name) for name in row["additional_metadata"]["stereotyped_groups"]}
    hits = [
        index
        for index in range(3)
        if index != unknown
        and (_tag(tags[index]) in stated or any(_tag(part) in stated for part in re.split(r"[_-]", tags[index])))
    ]
    return {"unknown": unknown, "stereotyped": hits[0]} if len(hits) == 1 else None


def _read(row: dict | None, twin: dict | None, english: dict) -> dict | None:
    """One language's two rows of an item as the prompts need them; None when they do not make an item there."""
    if row is None or twin is None:
        return None
    found = roles(row)
    same = all(row[key] == english[key] for key in ("question_index", "question_polarity", "context_condition"))
    paired = twin["context_condition"] == "disambig" and all(
        twin[key] == row[key] for key in ("question", "question_index", "question_polarity", "ans0", "ans1", "ans2")
    )
    if not (same and paired and found) or row["label"] != found["unknown"]:
        return None
    if twin["label"] is None or twin["label"] == found["unknown"]:
        return None
    other = next(index for index in range(3) if index not in found.values())
    return {
        "rows": {"ambiguous": row, "disambiguated": twin},
        "unknown": found["unknown"],
        "correct": twin["label"],
        "aligned": found["stereotyped"] if row["question_polarity"] == "neg" else other,
        "roles": {found["unknown"]: "unknown", found["stereotyped"]: "stereotyped_group", other: "other_group"},
    }


def load_items(repo: Path, category: str) -> tuple[list[dict], int, int]:
    """The items of a category that every language holds as an ambiguous row and its twin, with the ambiguous
    rows of the English file and how many of them are not such an item."""
    files = {code: read_rows(repo / mbbq.data_file(category, code)) for code in mbbq.LANGUAGES}
    ambiguous = {number: row for number, row in files["en"].items() if row["context_condition"] == "ambig"}
    items = []
    for number, english in sorted(ambiguous.items()):
        held = {code: _read(rows.get(number), rows.get(number + 1), english) for code, rows in files.items()}
        if all(held.values()):
            items.append(
                {
                    "base_item": f"{GROUP}.{category}.{number}",
                    "category": category,
                    "template": f"{category}/{english['question_index']}",
                    "polarity": english["question_polarity"],
                    "languages": held,
                }
            )
    return items, len(ambiguous), len(ambiguous) - len(items)


def bbq_index(repo: Path, category: str) -> tuple[dict, dict]:
    """The ambiguous BBQ rows of a category by their text, and the templates of the category by question."""
    rows, questions = {}, defaultdict(set)
    for row in build.read_jsonl(repo / "data" / f"{category}.jsonl"):
        if row["context_condition"] == "ambig":
            rows[normal(row["context"]), normal(row["question"])] = row
        questions[normal(row["question"])].add(f"{category}/{row['question_index']}")
    return rows, questions


def trace(item: dict, rows: dict, questions: dict) -> None:
    """Say on the item what of BBQ its English row goes back to: the row itself, or the templates of its question."""
    english = item["languages"]["en"]["rows"]["ambiguous"]
    found = rows.get((normal(english["context"]), normal(english["question"])))
    base, templates = None, questions.get(normal(english["question"]), set())
    if found is not None:
        base, templates = (
            f"bbq.{item['category']}.{found['example_id']}",
            {f"{item['category']}/{found['question_index']}"},
        )
    item |= {
        "bbq_match": "exact" if found is not None else "question" if templates else "none",
        "bbq_base_item": base,
        "bbq_templates": sorted(templates, key=lambda name: int(name.split("/")[1])),
    }


def question_type(item: dict, split: dict, notes: Sequence[dict]) -> str:
    """Whether the fine-tune saw the question type of a traced item: a split it may be drawn for, or why not."""
    if item["category"] in split["heldout_categories"]:
        return HELDOUT_CATEGORY
    if not item["bbq_templates"]:
        return NOT_IN_BBQ
    held = set(split["heldout_templates"].get(item["category"], ()))
    unseen = set(item["bbq_templates"]) <= held and not leak.of_a_trained_type(notes, [item])
    return HELDOUT_TEMPLATE if unseen else TRAINED


def arrange(items: list[dict], rng: random.Random) -> None:
    """Give each item a place for its unknown option and for its correct person, shared by its rows in every
    language: each sits at A, B and C equally often over the items."""
    order = list(range(len(items)))
    rng.shuffle(order)
    for place, index in enumerate(order):
        items[index] |= {"unknown_at": place % 3, "correct_first": (place // 3) % 2 == 0}


def prompt_rows(item: dict, code: str) -> list[dict]:
    """The two prompt rows of an item in one language."""
    held = item["languages"][code]
    wrong = next(index for index in range(3) if index not in (held["unknown"], held["correct"]))
    rest = [held["correct"], wrong] if item["correct_first"] else [wrong, held["correct"]]
    placed = rest[: item["unknown_at"]] + [held["unknown"]] + rest[item["unknown_at"] :]
    at = {option: LETTERS[place] for place, option in enumerate(placed)}
    rows = []
    for condition, row in held["rows"].items():
        lines = [row["context"], row["question"]]
        lines += [f"{letter}. {row[f'ans{option}']}" for letter, option in zip(LETTERS, placed, strict=True)]
        rows.append(
            {
                "prompt_id": f"{GROUP}.{code}.{item['category']}.{row['example_id']}",
                "prompt": "\n".join([*lines, CLOSING[code]]),
                "set": SET,
                "language": code,
                "base_item": item["base_item"],
                "condition": condition,
                "category": item["category"],
                "template": item["template"],
                "polarity": item["polarity"],
                "split": item["split"],
                "format": "letter",
                "options": [
                    {"letter": letter, "text": row[f"ans{option}"], "role": held["roles"][option]}
                    for letter, option in zip(LETTERS, placed, strict=True)
                ],
                "letter_unknown": at[held["unknown"]],
                "letter_correct": at[held["unknown"] if condition == "ambiguous" else held["correct"]],
                "letter_stereotype_aligned": at[held["aligned"]],
                "source_example_id": row["example_id"],
                # the answer index of the source row shown at A, B and C
                "option_order": placed,
                "bbq_match": item["bbq_match"],
                "bbq_base_item": item["bbq_base_item"],
                "bbq_templates": item["bbq_templates"],
            }
        )
    return rows


def main(argv: Sequence[str] | None = None, download: Callable[[str], bytes] = bbq.download) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mbbq", type=Path, required=True, help="folder of the MBBQ files; missing ones are downloaded"
    )
    parser.add_argument("--bbq", type=Path, required=True, help="folder of the BBQ files; missing ones are downloaded")
    parser.add_argument("--data", type=Path, required=True, help="the folder build wrote: manifest and training set")
    parser.add_argument("--out", type=Path, required=True, help="folder the mbbq/ folder is written under")
    parser.add_argument(
        "--train-set", choices=ids.TRAIN_SETS, default=ids.TRAIN_SETS[0], help="name of the training set"
    )
    args = parser.parse_args(argv)
    source, origin = mbbq.fetch(args.mbbq, download), bbq.fetch(args.bbq, download)
    split = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))["split"]
    folder = ids.training_folder(args.train_set)
    notes = [note for path in sorted((args.data / folder).glob(f"*{META}")) for note in build.read_jsonl(path)]
    if not notes:
        raise SystemExit(f"{folder} under {args.data.name} holds no {META} file: nothing says what was trained on")

    rng = random.Random(SEED)
    drawn, pool = [], {}
    # an odd number of items is one more negative question, then one more of the other kind, in turn
    more_negative = True
    for category, wanted in WANTED.items():
        items, ambiguous, unreadable = load_items(args.mbbq, category)
        index = bbq_index(args.bbq, category)
        for item in items:
            trace(item, *index)
            item["split"] = question_type(item, split, notes)
        usable = [item for item in items if item["split"] in (HELDOUT_CATEGORY, HELDOUT_TEMPLATE)]
        pool[category] = {
            "ambiguous_items": ambiguous,
            NOT_PARALLEL: unreadable,
            "by_question_type": by(item["split"] for item in items),
            "drawn": wanted,
        }
        negative = wanted // 2 + (wanted % 2 and more_negative)
        more_negative ^= wanted % 2 == 1
        for polarity, count in (("neg", negative), ("nonneg", wanted - negative)):
            drawn += build.spread([item for item in usable if item["polarity"] == polarity], count, rng)
    arrange(drawn, rng)
    rows = {code: [row for item in drawn for row in prompt_rows(item, code)] for code in mbbq.LANGUAGES}
    every = [row for code in mbbq.LANGUAGES for row in rows[code]]
    seen = leak.of_a_trained_type(notes, every)
    if seen:
        raise SystemExit(f"{len(seen)} drawn rows are of a question type the training rows hold")

    written = {}
    for code in mbbq.LANGUAGES:
        name = ids.extra_prompts(GROUP, SET, code)
        write_jsonl(args.out / name, rows[code])
        written[name] = sha256(args.out / name)
    english = [row for row in rows["en"] if row["condition"] == "ambiguous"]
    twins = [row for row in rows["en"] if row["condition"] == "disambiguated"]
    manifest = {
        "what": "question pairs of MBBQ in four languages, of question types the attribution fine-tune never saw",
        "source": {
            "dataset": f"MBBQ ({mbbq.REPOSITORY}), files under data/",
            "commit": mbbq.COMMIT,
            "licence": mbbq.LICENCE,
            "note_of_its_authors": mbbq.NOTE,
            "files_sha256": source,
        },
        "question_types_from": {
            "dataset": f"BBQ ({bbq.REPOSITORY})",
            "commit": bbq.COMMIT,
            "files_sha256": {name: origin[name] for name in sorted(origin) if Path(name).stem in WANTED},
        },
        "training": {
            "folder": folder,
            "rows_described": len(notes),
            "base_items": len({note["base_item"] for note in notes if note.get("base_item")}),
            "templates": len({note["template"] for note in notes if note.get("template")}),
            "compared": "the BBQ base item and the BBQ templates this build traced each item to, from its English "
            "row and within its own category, against the base items and templates the sidecar names; a launch "
            "repeats it on the fields of the prompt rows, against the sidecar stored at the prompts' commit",
            "drawn_rows_of_a_trained_question_type": len(seen),
        },
        "seed": SEED,
        "languages": list(mbbq.LANGUAGES),
        "closing_lines": CLOSING,
        "rows": len(every),
        "items": len(drawn),
        "items_by_category": by(row["category"] for row in english),
        "items_by_split": by(row["split"] for row in english),
        "items_by_polarity": by(row["polarity"] for row in english),
        # the row of one id states the stereotyped person as its answer in one language and the other person in another
        "items_whose_stated_answer_is_another_person_in_some_language": sum(
            len({held["roles"][held["correct"]] for held in item["languages"].values()}) > 1 for item in drawn
        ),
        "templates": len({row["template"] for row in english}),
        "bbq_templates": len({name for row in english for name in row["bbq_templates"]}),
        "bbq_match": by(row["bbq_match"] for row in english),
        "letter_unknown": by(row["letter_unknown"] for row in english),
        "letter_correct_of_twins": by(row["letter_correct"] for row in twins),
        "pool": pool,
        "files_sha256": written,
    }
    path = args.out / GROUP / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shown = ("items", "rows", "items_by_category", "items_by_split", "items_by_polarity", "bbq_match", "pool")
    print(json.dumps({key: manifest[key] for key in shown}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
