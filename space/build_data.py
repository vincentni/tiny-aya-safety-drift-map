"""Build the data file of the results Space from the published label files and tables.

    python space/build_data.py                                  # reads the two datasets from the Hub
    python space/build_data.py --eval-dir DIR --tables-dir DIR  # reads local copies

The output holds prompt ids, languages, harm categories and the judge's labels. It never holds prompt or
response text: those stay in the datasets, behind their content warnings.
"""

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "driftmap.space/1"
EVAL_REPO = "vincentnkq/aya-redteaming-6lang-eval"
RESULTS_REPO = "vincentnkq/safety-drift-map-results"
# id, name on the page, folder of the label files in the evaluation dataset, tables file in the results dataset
CONDITIONS = (
    ("served", "served template", "labels", "tables/tables.json"),
    ("nopreamble", "no safety preamble", "nopreamble/labels", "tables/tables-nopreamble.json"),
)
# id, name on the page, folder under each label tree
MODELS = (
    ("base", "base model", "tiny-aya-global--af89d219b53e"),
    ("finetuned", "fine-tuned model", "tiny-aya-global-drift-merged--daccc14f6c19"),
)
LANGUAGES = ("en", "fr", "es", "ru", "hi", "ar")
SETS = ("harmful", "xstest")
LABELLED = ("yes", "no")
Z95 = 1.959963984540054


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; split on newlines only, since a prompt may hold other line separators."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def verdict(row: dict) -> str:
    """One word per judged pair: harmful, refused, answered (not harmful, no refusal) or unscored."""
    if row["response_harm"] not in LABELLED or row["refusal"] not in LABELLED:
        return "unscored"
    if row["response_harm"] == "yes":
        return "harmful"
    return "refused" if row["refusal"] == "yes" else "answered"


def wilson(k: int, n: int) -> tuple[float, float] | None:
    """95 % Wilson score interval for k of n; None when n is 0."""
    if n == 0:
        return None
    share, scale = k / n, 1 + Z95**2 / n
    centre = (share + Z95**2 / (2 * n)) / scale
    half = Z95 * math.sqrt(share * (1 - share) / n + Z95**2 / (4 * n**2)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> list:
    interval = wilson(k, n)
    return [k, n, *(round(bound, 4) for bound in interval)] if interval else [k, n, None, None]


def _prompts(eval_dir: Path) -> dict[str, dict]:
    """Language, harm categories and scope of every harmful prompt, by prompt id; the text is dropped here."""
    found = {}
    for path in sorted((eval_dir / "prompts" / "harmful").glob("*.jsonl")):
        for row in read_jsonl(path):
            found[row["prompt_id"]] = {
                "language": row["language"],
                "categories": sorted(json.loads(row["harm_category"])),
                "scope": row["global_or_local"],
            }
    return found


def _labels(eval_dir: Path, tree: str, folder: str) -> tuple[list[dict], list[float]]:
    """Judged rows of one checkpoint in one condition, and the chrF of its benign answers."""
    here = eval_dir / tree / folder
    judged = [row for path in sorted(here.glob("*.jsonl")) for row in read_jsonl(path)]
    scores = [row["chrf"] for path in sorted(here.glob("benign/*.jsonl")) for row in read_jsonl(path)]
    if not judged:
        raise FileNotFoundError(f"no label files under {tree}/{folder} in {eval_dir}")
    return judged, [score for score in scores if score is not None]


def _summary(judged: list[dict], scores: list[float]) -> dict:
    scored = {name: [row for row in judged if row["set"] == name and verdict(row) != "unscored"] for name in SETS}
    harmful, xstest = scored["harmful"], scored["xstest"]
    return {
        "harmful": _rate(sum(row["response_harm"] == "yes" for row in harmful), len(harmful)),
        "refusal": _rate(sum(row["refusal"] == "yes" for row in harmful), len(harmful)),
        # over-refusal is the refusal rate on prompts that are safe to answer
        "over_refusal": _rate(sum(row["refusal"] == "yes" for row in xstest), len(xstest)),
        "chrf": [round(sum(scores) / len(scores), 4), len(scores)] if scores else [None, 0],
    }


def _cells(condition: str, model: str, verdicts: dict[str, str], prompts: dict[str, dict], categories: list) -> list:
    """Harmful answers of scored pairs for every language and category, with ``all`` as the marginal of each."""
    counts, sizes = Counter(), Counter()
    for prompt_id, word in verdicts.items():
        if word == "unscored":
            continue
        prompt = prompts[prompt_id]
        for language in (prompt["language"], "all"):
            for category in (*prompt["categories"], "all"):
                sizes[language, category] += 1
                counts[language, category] += word == "harmful"
    cells = []
    for language in (*LANGUAGES, "all"):
        for category in (*categories, "all"):
            k, n = counts[language, category], sizes[language, category]
            low, high = _rate(k, n)[2:]
            row = {"condition": condition, "model": model, "language": language, "category": category}
            cells.append(row | {"k": k, "n": n, "lo": low, "hi": high})
    return cells


def _flips(before: dict[str, str], after: dict[str, str]) -> dict[str, int]:
    pairs = [(before[key], after[key]) for key in before.keys() & after.keys()]
    pairs = [(first == "harmful", second == "harmful") for first, second in pairs if "unscored" not in (first, second)]
    return {
        "pairs": len(pairs),
        "harmful_both": pairs.count((True, True)),
        "harmful_to_not": pairs.count((True, False)),
        "not_to_harmful": pairs.count((False, True)),
    }


def _changes(path: Path) -> dict:
    """Pooled paired changes between the two models, as the published tables state them; empty without the file."""
    if not path.is_file():
        return {}
    deltas = json.loads(path.read_text(encoding="utf-8"))["deltas"]
    pooled = {"safe": "all", "over_refusal": "en", "chrf": "all"}
    found = {}
    for delta in deltas:
        if delta["a"] != delta["b"] and pooled.get(delta["metric"]) == delta["row"]:
            found[delta["metric"]] = {key: delta[name] for key, name in
                                      (("delta", "delta"), ("ci", "ci"), ("p", "p_value"), ("pairs", "n_pairs"))}  # fmt: skip
    return found


def build(eval_dir: str | Path, tables_dir: str | Path | None = None, figures: list[str] | None = None) -> dict:
    """The Space's data: summary, cells, flips and one row per harmful prompt, from label files under ``eval_dir``."""
    eval_dir = Path(eval_dir)
    prompts = _prompts(eval_dir)
    categories = sorted({category for prompt in prompts.values() for category in prompt["categories"]})
    rows = {prompt_id: {"id": prompt_id, **prompt, "labels": {}} for prompt_id, prompt in prompts.items()}
    summary, cells, flips, changes = {}, [], {}, {}
    for condition, _, tree, tables_file in CONDITIONS:
        summary[condition], verdicts = {}, {}
        for model, _, folder in MODELS:
            judged, scores = _labels(eval_dir, tree, folder)
            summary[condition][model] = _summary(judged, scores)
            harmful = {row["prompt_id"]: row for row in judged if row["set"] == "harmful"}
            verdicts[model] = {prompt_id: verdict(row) for prompt_id, row in harmful.items()}
            cells += _cells(condition, model, verdicts[model], prompts, categories)
            for prompt_id, row in harmful.items():
                labels = {"refusal": row["refusal"], "response_harm": row["response_harm"], "verdict": verdict(row)}
                rows[prompt_id]["labels"].setdefault(condition, {})[model] = labels
        flips[condition] = _flips(verdicts[MODELS[0][0]], verdicts[MODELS[1][0]])
        changes[condition] = _changes(Path(tables_dir) / Path(tables_file).name) if tables_dir else {}
    return {
        "schema": SCHEMA,
        "eval_repo": EVAL_REPO,
        "conditions": [{"id": key, "name": name, "tree": tree} for key, name, tree, _ in CONDITIONS],
        "models": [{"id": key, "name": name, "folder": folder} for key, name, folder in MODELS],
        "languages": list(LANGUAGES),
        "categories": categories,
        "figures": figures or [],
        "summary": summary,
        "changes": changes,
        "flips": flips,
        "cells": cells,
        "prompts": sorted(rows.values(), key=lambda row: (LANGUAGES.index(row["language"]), row["id"])),
    }


def _download(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns))


def _copy_figures(source: Path | None, target: Path) -> list[str]:
    """Copy the figure files beside the page; the page shows only the ones listed in the data file."""
    names = []
    for path in sorted(source.glob("f*.png")) if source and source.is_dir() else ():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / path.name)
        names.append(path.name)
    return names


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval-dir", type=Path, help="a local copy of the evaluation dataset; downloaded when absent")
    parser.add_argument("--tables-dir", type=Path, help="a folder with tables.json and tables-nopreamble.json")
    parser.add_argument("--figures-dir", type=Path, default=here.parent / "docs" / "figures")
    parser.add_argument("--out", type=Path, default=here / "data.json")
    args = parser.parse_args(argv)
    trees = [f"{tree}/*" for _, _, tree, _ in CONDITIONS]
    eval_dir = args.eval_dir or _download(EVAL_REPO, [*trees, "prompts/harmful/*"])
    tables_dir = args.tables_dir
    if tables_dir is None and args.eval_dir is None:
        tables_dir = _download(RESULTS_REPO, [tables for *_, tables in CONDITIONS]) / "tables"
    data = build(eval_dir, tables_dir, _copy_figures(args.figures_dir, args.out.parent / "figures"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    counts = ", ".join(f"{key} {len(data[key])}" for key in ("prompts", "cells", "figures"))
    print(f"{args.out}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
