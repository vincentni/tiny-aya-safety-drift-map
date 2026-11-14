"""Build the data file of the results Space from the published label files and tables.

    python space/build_data.py                                  # reads the datasets from the Hub
    python space/build_data.py --eval-dir DIR --tables-dir DIR  # reads local copies
    python space/build_data.py --eval-dir DIR --responses-dir DIR   # adds the answer-length counts

The output holds counts and intervals only: no prompt, no response and no per-prompt row.
"""

import argparse
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "driftmap.space/2"
EVAL_REPO = "vincentnkq/aya-redteaming-6lang-eval"
RESULTS_REPO = "vincentnkq/safety-drift-map-results"
RESPONSES_REPO = "vincentnkq/safety-drift-map-responses"
# id, name on the page, folder of the label files in the evaluation dataset, tables file in the results dataset
CONDITIONS = (
    ("served", "served template", "labels", "tables/tables.json"),
    ("nopreamble", "no safety preamble", "nopreamble/labels", "nopreamble/tables/tables-nopreamble.json"),
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
    """Language and harm categories of every harmful prompt, by prompt id; the text is dropped here."""
    found = {}
    for path in sorted((eval_dir / "prompts" / "harmful").glob("*.jsonl")):
        for row in read_jsonl(path):
            categories = sorted(json.loads(row["harm_category"]))
            found[row["prompt_id"]] = {"language": row["language"], "categories": categories}
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


def _length(responses_dir: Path, tree: str, folder: str, verdicts: dict[str, str]) -> dict:
    """Answers cut off at the token limit, and harmful answers among the complete and the cut-off ones."""
    here = responses_dir / tree.replace("labels", "responses") / folder / "harmful"
    rows = [row for path in sorted(here.glob("*.jsonl")) for row in read_jsonl(path)]
    if not rows:
        raise FileNotFoundError(f"no response files under {here}")
    harmful, sizes = Counter(), Counter()
    for row in rows:
        word = verdicts.get(row["prompt_id"], "unscored")
        if word != "unscored":
            cut = row["stop_reason"] == "length"
            sizes[cut] += 1
            harmful[cut] += word == "harmful"
    return {
        "cut_off": [sum(row["stop_reason"] == "length" for row in rows), len(rows)],
        "median_tokens": statistics.median(row["generated_tokens"] for row in rows),
        "harmful_complete": [harmful[False], sizes[False]],
        "harmful_cut_off": [harmful[True], sizes[True]],
    }


def _changes(tables_dir: Path, tables_file: str) -> dict:
    """Pooled paired changes between the two models, as the published tables state them; empty without the file.

    ``tables_dir`` is a copy of the results dataset, or a flat folder that holds the tables files by name.
    """
    places = (tables_dir / tables_file, tables_dir / Path(tables_file).name)
    path = next((place for place in places if place.is_file()), None)
    if path is None:
        return {}
    deltas = json.loads(path.read_text(encoding="utf-8"))["deltas"]
    pooled = {"safe": "all", "over_refusal": "en", "chrf": "all"}
    found = {}
    for delta in deltas:
        if delta["a"] != delta["b"] and pooled.get(delta["metric"]) == delta["row"]:
            found[delta["metric"]] = {key: delta[name] for key, name in
                                      (("delta", "delta"), ("ci", "ci"), ("p", "p_value"), ("pairs", "n_pairs"))}  # fmt: skip
    return found


def build(
    eval_dir: str | Path,
    tables_dir: str | Path | None = None,
    responses_dir: str | Path | None = None,
    figures: list[str] | None = None,
) -> dict:
    """The Space's data: summary, cells, flips and answer-length counts, from label files under ``eval_dir``."""
    eval_dir = Path(eval_dir)
    prompts = _prompts(eval_dir)
    categories = sorted({category for prompt in prompts.values() for category in prompt["categories"]})
    summary, cells, flips, changes, length = {}, [], {}, {}, {}
    for condition, _, tree, tables_file in CONDITIONS:
        summary[condition], verdicts = {}, {}
        for model, _, folder in MODELS:
            judged, scores = _labels(eval_dir, tree, folder)
            summary[condition][model] = _summary(judged, scores)
            verdicts[model] = {row["prompt_id"]: verdict(row) for row in judged if row["set"] == "harmful"}
            cells += _cells(condition, model, verdicts[model], prompts, categories)
            if responses_dir:
                length.setdefault(condition, {})[model] = _length(Path(responses_dir), tree, folder, verdicts[model])
        flips[condition] = _flips(verdicts[MODELS[0][0]], verdicts[MODELS[1][0]])
        changes[condition] = _changes(Path(tables_dir), tables_file) if tables_dir else {}
    return {
        "schema": SCHEMA,
        "eval_repo": EVAL_REPO,
        "conditions": [{"id": key, "name": name} for key, name, _, _ in CONDITIONS],
        "models": [{"id": key, "name": name} for key, name, _ in MODELS],
        "languages": list(LANGUAGES),
        "categories": categories,
        "figures": figures or [],
        "summary": summary,
        "changes": changes,
        "flips": flips,
        "length": length,
        "cells": cells,
    }


def _download(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns))


def _responses() -> Path | None:
    """The responses dataset, or None with a note when it cannot be read; it sits behind an access gate."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return _download(RESPONSES_REPO, ["responses/*/harmful/*", "nopreamble/responses/*/harmful/*"])
    except (HfHubHTTPError, OSError) as error:
        print(f"{RESPONSES_REPO} could not be read ({type(error).__name__}); the answer-length counts are left out")
        return None


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
    parser.add_argument("--tables-dir", type=Path, help="a copy of the results dataset, or a folder with the tables files")
    parser.add_argument("--responses-dir", type=Path, help="a local copy of the responses dataset, for answer lengths")
    parser.add_argument("--figures-dir", type=Path, default=here.parent / "docs" / "figures")
    parser.add_argument("--out", type=Path, default=here / "data.json")
    args = parser.parse_args(argv)
    trees = [f"{tree}/*" for _, _, tree, _ in CONDITIONS]
    from_hub = args.eval_dir is None
    eval_dir = args.eval_dir or _download(EVAL_REPO, [*trees, "prompts/harmful/*"])
    tables_dir, responses_dir = args.tables_dir, args.responses_dir
    if tables_dir is None and from_hub:
        tables_dir = _download(RESULTS_REPO, [tables for *_, tables in CONDITIONS])
    if responses_dir is None and from_hub:
        responses_dir = _responses()
    figures = _copy_figures(args.figures_dir, args.out.parent / "figures")
    data = build(eval_dir, tables_dir, responses_dir, figures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    counts = ", ".join(f"{key} {len(data[key])}" for key in ("cells", "figures", "length"))
    print(f"{args.out}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
