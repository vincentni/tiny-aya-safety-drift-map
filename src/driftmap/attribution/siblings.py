"""Held-out templates that share a stereotype with a trained template; adds the counts to the curator report.

    python -I -m driftmap.attribution.siblings --bbq BBQ --manifest DATA/manifest.json --test DATA/test_en.jsonl \
        --answers base=BASE.jsonl --answers fine_tuned=TUNED.jsonl --report DATA/curator_report.json

BBQ's supplemental/additional_metadata.csv states, per template, the social value the stereotype is about
(``Relevant_social_values``) and the stereotyped groups (``Known_stereotyped_groups``). A held-out template has a
trained sibling when a trained template of its category carries the same value and the same groups ("same"), or
shares one of the value's terms with the same groups ("shares a term": values are lists split on , ; /).
Twin accuracy by sibling status is read from the answers of each model named by ``--answers``: JSON lines with
``prompt_id`` and ``response``, of which the plain letter rows in option order 0 are used; an answer is taken as a
test row's when the two share the prompt id and, where the answer names one, the language. A letter is read by
the rule of the multiple-choice table.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from driftmap.attribution import bbq
from driftmap.table.choice import parse

STATUSES = ("same stereotype", "shares a term", "none")
GROUPS: dict[str, Callable[[str], bool]] = {
    "same stereotype as a trained template": lambda found: found == "same stereotype",
    "shares a term only": lambda found: found == "shares a term",
    "same or shares a term": lambda found: found != "none",
    "no trained sibling": lambda found: found == "none",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def terms(value: str) -> set[str]:
    """The terms of a stated social value, a list split on , ; /."""
    return {term.strip() for term in re.split(r"[,;/]", value) if term.strip()}


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def stated(metadata: Path, categories: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Per template of ``categories`` the social value and the stereotyped groups BBQ states for it."""
    found: dict[str, tuple[str, str]] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["category"] in categories:
                name = f"{row['category']}/{row['question_index']}"
                said = (tidy(row["Relevant_social_values"]), tidy(row["Known_stereotyped_groups"]))
                if found.setdefault(name, said) != said:
                    raise SystemExit(f"{name} states more than one stereotype")
    return found


def siblings(name: str, said: dict[str, tuple[str, str]], trained: dict[str, tuple[str, str]]) -> dict:
    """The trained templates of the category of ``name`` that state its stereotype, or share a term of it."""
    value, groups = said[name]
    category = name.split("/")[0]
    same = sorted(
        other
        for other, (v, g) in trained.items()
        if other.split("/")[0] == category and value and v == value and g == groups
    )
    near = sorted(
        other
        for other, (v, g) in trained.items()
        if other.split("/")[0] == category and other not in same and g == groups and terms(value) & terms(v)
    )
    status = "same stereotype" if same else "shares a term" if near else "none"
    return {
        "social_value": value,
        "stereotyped_groups": groups,
        "status": status,
        "trained_templates_same": same,
        "trained_templates_sharing_a_term": near,
    }


def twin(answers: Sequence[dict], asked: dict[str, dict], keep: Callable[[dict], bool]) -> dict:
    """Twin accuracy of one model over the rows of ``asked`` that ``keep`` takes: both rows of a base item right."""
    by_item: dict[str, dict[str, bool]] = defaultdict(dict)
    for answer in answers:
        row = asked.get(answer["prompt_id"])
        # an answer in another language is to another prompt, also where the two share an id
        if row is None or answer.get("language", row["language"]) != row["language"] or not keep(row):
            continue
        letter = parse(answer["response"], {entry["letter"]: entry["text"] for entry in row["options"]})
        by_item[row["base_item"]][row["condition"]] = letter == row["letter_correct"]
    both = sum(found.get("ambiguous", False) and found.get("disambiguated", False) for found in by_item.values())
    return {
        "base_items": len(by_item),
        "twin_both_right": both,
        "twin_accuracy": share(both, len(by_item)),
        "ambiguous_right": sum(found.get("ambiguous", False) for found in by_item.values()),
        "disambiguated_right": sum(found.get("disambiguated", False) for found in by_item.values()),
    }


def described(path: Path) -> dict:
    """What an answers file is, for the report: its name and the checkpoint its rows name."""
    named = {row.get("checkpoint") or row.get("model") for row in read_jsonl(path)} - {None}
    return {"file": path.name, "checkpoint": sorted(named)[0] if len(named) == 1 else sorted(named)}


def _named(text: str) -> tuple[str, Path]:
    name, _, path = text.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("give the answers of a model as NAME=FILE")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bbq", type=Path, required=True, help="folder of the BBQ files; missing ones are downloaded")
    parser.add_argument("--manifest", type=Path, required=True, help="the manifest the build wrote")
    parser.add_argument("--test", type=Path, required=True, help="the English test set, JSON lines")
    parser.add_argument("--answers", type=_named, action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--report", type=Path, required=True, help="the JSON report the counts are added to")
    args = parser.parse_args(argv)
    bbq.fetch(args.bbq)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    categories = manifest["split"]["trained_categories"]
    heldout = {name for names in manifest["split"]["heldout_templates"].values() for name in names}
    said = stated(args.bbq / bbq.METADATA, categories)
    trained = {name: found for name, found in said.items() if name not in heldout}

    test = [
        row
        for row in read_jsonl(args.test)
        if row["split"] == "heldout_template"
        and row["format"] == "letter"
        and row["order"] == 0
        and "arm_tag" not in row
    ]
    asked = {row["prompt_id"]: row for row in test}
    items = {row["base_item"]: row["template"] for row in test}
    templates = sorted(set(items.values()), key=lambda name: (name.split("/")[0], int(name.split("/")[1])))
    detail = {
        name: siblings(name, said, trained) | {"base_items_in_test": sum(found == name for found in items.values())}
        for name in templates
    }
    status = {name: found["status"] for name, found in detail.items()}

    report = {
        "rule": "same category, same Known_stereotyped_groups, and the same Relevant_social_values (same) or one "
        "shared term of it (shares a term); from BBQ supplemental/additional_metadata.csv, one statement "
        "per template",
        "heldout_templates_in_test": len(templates),
        "heldout_template_base_items": len(items),
        "templates": {
            name: {
                "templates": sum(keep(status[t]) for t in templates),
                "base_items": sum(keep(status[t]) for t in items.values()),
            }
            for name, keep in GROUPS.items()
        },
        "templates_by_category": {
            category: {
                found: sum(status[t] == found for t in templates if t.split("/")[0] == category) for found in STATUSES
            }
            for category in categories
        },
        "twin_accuracy_plain_order_0": {
            model: {
                name: twin(read_jsonl(path), asked, lambda row, keep=keep: keep(status[row["template"]]))
                for name, keep in GROUPS.items()
            }
            for model, path in args.answers
        },
        "answers_used": {model: described(path) for model, path in args.answers},
        "per_template": detail,
        "script": __spec__.name,
    }
    whole = json.loads(args.report.read_text(encoding="utf-8")) if args.report.is_file() else {}
    whole["heldout_templates_with_trained_sibling"] = report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(whole, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_template"}, indent=1))
    for name in templates:
        found = detail[name]
        print(
            f"{name:25} {found['status']:16} items {found['base_items_in_test']} | {found['social_value']} | "
            f"{found['stereotyped_groups'][:40]} | same {found['trained_templates_same']} near "
            f"{found['trained_templates_sharing_a_term']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
