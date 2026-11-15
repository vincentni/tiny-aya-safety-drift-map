"""CuratorKIT report over the abstention training rows: gate, dedup and cross-set dedup. Nothing is rewritten.

    python -m driftmap.attribution.curator_report --data DATA --prompts PROMPTS --work SCRATCH \
        --out DATA/curator_report.json

``--data`` is the folder ``build`` and ``translate`` wrote: the training set in its folder, with the ``.meta.jsonl``
file that says what each row is, and ``test_<language>.jsonl``. ``--prompts`` holds the prompts of the safety
evaluation as ``<set>/<language>.jsonl``. Scratch files go to ``--work``.

1. The rows are staged and run through the data build's own pipeline (schema gate, exact dedup, MinHash dedup).
2. ``cross_dedup`` of the data build compares every training prompt with every test prompt of the six languages,
   and again with the prompts of the safety evaluation: exact match, trigram Jaccard and MinHash at the build's
   threshold.
3. The MinHash step inside the training set is reported as a count only; templated siblings are kept on purpose.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from driftmap.core import ids
from driftmap.scripts.drift import META

LANGUAGES = ("en", "fr", "es", "ru", "hi", "ar")
PAIRS_SHOWN = 50


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kinds(meta: list[dict], indices) -> dict:
    """What the rows at ``indices`` are: benign, or abstention by condition and format."""
    return dict(
        sorted(
            Counter(
                "benign" if meta[i]["kind"] == "benign" else f"{meta[i]['condition']} {meta[i]['format']}"
                for i in indices
            ).items()
        )
    )


def closest(text: str, protected: list[dict]) -> tuple[float, dict]:
    """The protected prompt nearest to ``text`` by trigram Jaccard, the measure cross_dedup uses."""
    from driftmap.data import dedup

    grams = dedup._trigrams(dedup._key(text))
    best = (0.0, protected[0])
    for row in protected:
        other = dedup._trigrams(dedup._key(row["prompt"]))
        union = len(grams | other)
        score = len(grams & other) / union if union else 0.0
        if score > best[0]:
            best = (score, row)
    return best


def nearest(train: list[dict], meta: list[dict], protected: list[dict]) -> dict:
    """How near the nearest protected prompt is to each training prompt, by trigram Jaccard."""
    from driftmap.data import dedup

    grams = [(dedup._trigrams(dedup._key(row["prompt"])), row) for row in protected]
    best = []
    for number, row in enumerate(train):
        mine = dedup._trigrams(dedup._key(row["prompt"]))
        score, other = max(
            ((len(mine & theirs) / len(mine | theirs), found) for theirs, found in grams), key=lambda pair: pair[0]
        )
        best.append((score, number, other))
    scores = sorted(score for score, _, _ in best)
    top = sorted(best, key=lambda found: -found[0])[:5]
    return {
        "compared_with": len(protected),
        "highest": round(scores[-1], 4),
        "99th_percentile": round(scores[int(0.99 * len(scores))], 4),
        "median": round(scores[len(scores) // 2], 4),
        "training_prompts_at_or_over_threshold": sum(score >= dedup.THRESHOLD for score in scores),
        "training_prompts_at_or_over_0.6": sum(score >= 0.6 for score in scores),
        "nearest_pairs": [
            {
                "jaccard": round(score, 4),
                "training_row": train[number]["prompt_id"],
                "training_template": meta[number].get("template"),
                "test_prompt_id": other["prompt_id"],
                "test_template": other.get("template"),
                "training_prompt": train[number]["prompt"],
                "test_prompt": other["prompt"],
            }
            for score, number, other in top
        ],
    }


def cross(name: str, protected: list[dict], train: list[dict], meta: list[dict]) -> dict:
    """``cross_dedup`` of the training prompts against ``protected``, with the nearest pair of every copy."""
    from driftmap.data import dedup

    found = dedup.cross_dedup(protected, train)
    kept = {row["prompt_id"] for row in found.rows}
    protected_keys = {dedup._key(row["prompt"]) for row in protected}
    exact = [i for i, row in enumerate(train) if dedup._key(row["prompt"]) in protected_keys]
    report = {
        "protected_set": name,
        "protected_prompts": len(protected),
        "distinct_protected_prompts": len(protected_keys),
        "training_prompts": len(train),
        "copies": found.copies,
        "exact_matches_after_normalisation": len(exact),
        "repeats_inside_training": found.repeats,
        "training_prompts_left": len(found.rows),
    }
    if found.copies:
        # a row is dropped either as a copy or as a repeat; a copy has a protected prompt at or near the threshold
        dropped = [i for i, row in enumerate(train) if row["prompt_id"] not in kept]
        scored = sorted(((closest(train[i]["prompt"], protected), i) for i in dropped), key=lambda pair: -pair[0][0])
        pairs = []
        for (score, other), i in scored[: max(PAIRS_SHOWN, found.copies)]:
            pairs.append(
                {
                    "jaccard": round(score, 4),
                    "training_row": train[i]["prompt_id"],
                    "training_kind": kinds(meta, [i]),
                    "training_base_item": meta[i].get("base_item"),
                    "training_prompt": train[i]["prompt"],
                    "protected_prompt_id": other.get("prompt_id"),
                    "protected_language": other.get("language"),
                    "protected_prompt": other["prompt"],
                }
            )
        report["dropped_rows_nearest_protected_prompt"] = pairs
        report["dropped_rows_at_or_over_threshold"] = sum(pair["jaccard"] >= dedup.THRESHOLD for pair in pairs)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="folder of the training set and the test files")
    parser.add_argument("--prompts", type=Path, required=True, help="folder of the safety evaluation's prompts")
    parser.add_argument("--work", type=Path, required=True, help="scratch folder")
    parser.add_argument("--out", type=Path, required=True, help="the report, JSON")
    parser.add_argument("--config", type=Path, default=Path("driftmap.toml"), help="settings file; its seed is used")
    parser.add_argument(
        "--train-set", choices=ids.TRAIN_SETS, default=ids.TRAIN_SETS[0], help="name of the training set"
    )
    args = parser.parse_args(argv)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    # the dataset reader keeps its cache inside the scratch folder and never asks the Hub; the libraries read this
    # at import, so they are imported after it
    os.environ["HF_DATASETS_CACHE"] = str(work / "hf_datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.schema import DataSample

    from driftmap.core.settings import load_settings
    from driftmap.data import curator, dedup

    settings = load_settings(args.config)
    name = f"{ids.training_folder(args.train_set)}/en.jsonl"
    path = args.data / name
    rows, meta = read_jsonl(path), read_jsonl(path.with_name(path.stem + META))
    train = [
        {
            "prompt": row["prompt"][0]["content"],
            "target": row["completion"][0]["content"],
            "language": "en",
            "prompt_id": f"sft-{args.train_set}-{number:04d}",
        }
        for number, row in enumerate(rows)
    ]
    index = {row["prompt_id"]: number for number, row in enumerate(train)}

    # 1. the build's pipeline: schema gate, exact dedup, MinHash dedup
    staged = curator.stage(train, work / "staged")
    labelled = SimpleNamespace(sources=SimpleNamespace(benign=f"local/{name}"), seed=settings.seed)
    config = curator.render_yaml(labelled, staged, sha256(path)[:12])
    result = curator.run_pipeline(config, work / "pipeline")
    passed = {item.metadata["prompt_id"] for item in result.passed}
    rejected = Counter(
        str(getattr(item, "reason", getattr(item, "rejection_reason", "?")))[:80] for item in result.rejected
    )
    gone = [index[name] for name in index if name not in passed]

    # the two dedup steps on their own, as the pipeline applies them: on prompt and completion together
    samples = [
        DataSample(id=row["prompt_id"], source_uri="train", instruction=row["prompt"], output=row["target"])
        for row in train
    ]
    after_exact = ExactDeduplicator().run(list(samples))
    after_minhash = MinHashDeduplicator(threshold=dedup.THRESHOLD).run(list(after_exact))
    minhash_dropped = [index[s.id] for s in after_exact if s.id not in {t.id for t in after_minhash}]
    # and on the prompt alone, as cross_dedup compares
    prompts = [
        DataSample(id=row["prompt_id"], source_uri="train", instruction=dedup._key(row["prompt"])) for row in train
    ]
    prompt_exact = ExactDeduplicator().run(list(prompts))
    prompt_minhash = MinHashDeduplicator(threshold=dedup.THRESHOLD).run(list(prompt_exact))

    same = {}
    for number, row in enumerate(train):
        same.setdefault(dedup._key(row["prompt"]), []).append(number)
    repeated = [
        {
            "prompt": train[numbers[0]]["prompt"],
            "rows": [
                {
                    "training_row": train[n]["prompt_id"],
                    "base_item": meta[n].get("base_item"),
                    "kind": kinds(meta, [n]),
                    "completion": train[n]["target"],
                }
                for n in numbers
            ],
        }
        for numbers in same.values()
        if len(numbers) > 1
    ]

    # 2. against the test prompts of the six languages, and against the safety evaluation's prompts
    test = [dict(row, language=code) for code in LANGUAGES for row in read_jsonl(args.data / f"test_{code}.jsonl")]
    evaluation = [row for found in sorted(args.prompts.glob("*/*.jsonl")) for row in read_jsonl(found)]

    report = {
        "curatorkit_version": importlib.metadata.version("curatorkit"),
        "thresholds": {
            "minhash_threshold": dedup.THRESHOLD,
            "minhash_ngram": 3,
            "minhash_num_perm": 128,
            "schema_gate": {"required_fields": ["instruction", "output"], "min_tokens": 1, "max_tokens": 2048},
        },
        "training_file": {
            "path": name,
            "sha256": sha256(path),
            "rows": len(rows),
            "by_kind": kinds(meta, range(len(meta))),
        },
        "pipeline": {
            "config": f"{work.name}/pipeline/pipeline.yaml",
            "rows_in": len(train),
            "rejected_by_the_gate": len(result.rejected),
            "rejection_reasons": dict(rejected),
            "rows_out": len(result.passed),
            "removed_by_dedup_steps": len(train) - len(result.rejected) - len(result.passed),
            "stage_counts": result.stage_counts,
            "rows_not_passed_by_kind": kinds(meta, gone),
        },
        "dedup_steps_on_prompt_and_completion": {
            "rows_in": len(samples),
            "removed_by_exact_dedup": len(samples) - len(after_exact),
            "after_exact_dedup": len(after_exact),
            "minhash_would_drop_kept_on_purpose": len(after_exact) - len(after_minhash),
            "minhash_would_drop_by_kind": kinds(meta, minhash_dropped),
            "minhash_would_drop_by_category": dict(
                sorted(Counter(meta[i].get("category", "benign") for i in minhash_dropped).items())
            ),
        },
        "dedup_steps_on_prompt_alone": {
            "removed_by_exact_dedup": len(prompts) - len(prompt_exact),
            "minhash_would_drop_kept_on_purpose": len(prompt_exact) - len(prompt_minhash),
            "training_prompts_written_more_than_once": repeated,
        },
        "nearest_english_test_prompt": nearest(train, meta, [row for row in test if row["language"] == "en"]),
        "cross_dedup": [
            cross("test prompts, six languages", test, train, meta)
            | {"protected_by_language": dict(Counter(row["language"] for row in test))},
            cross("safety evaluation prompts (harmful, held-out benign, XSTest)", evaluation, train, meta)
            | {"protected_by_set": dict(Counter(row["set"] for row in evaluation))},
        ],
        "script": __spec__.name,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shown = {key: value for key, value in report.items() if key not in ("cross_dedup", "nearest_english_test_prompt")}
    print({key: value for key, value in report["nearest_english_test_prompt"].items() if key != "nearest_pairs"})
    print(json.dumps(shown, indent=1, ensure_ascii=False))
    for found in report["cross_dedup"]:
        counts = {key: value for key, value in found.items() if key != "dropped_rows_nearest_protected_prompt"}
        print(json.dumps(counts, ensure_ascii=False))
        for pair in found.get("dropped_rows_nearest_protected_prompt", [])[:8]:
            print(
                "   ",
                pair["jaccard"],
                pair["training_row"],
                pair["training_kind"],
                pair["protected_language"],
                "|",
                pair["training_prompt"][:90].replace("\n", " / "),
                "<>",
                pair["protected_prompt"][:90].replace("\n", " / "),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
