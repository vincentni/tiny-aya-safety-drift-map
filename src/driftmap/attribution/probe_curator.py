"""CuratorKIT gate over the SQuAD rows of the probe study: only the rows that pass are fitted on and scored.

    python -m driftmap.attribution.probe_curator --train OUT/squad_train_in.jsonl --test OUT/squad_dev_in.jsonl \
        --work SCRATCH --out CURATED

``--train`` and ``--test`` are prompt files of ``probe_squad``; ``--test`` may be given more than once. Scratch
files, the staged rows among them, go to ``--work``. A row is a CuratorKIT prompt-only sample whose
``instruction`` is the prompt as asked: passage, question, closing line.

1. Row shape: an id of its own, an article title, a passage, a question, the prompt made of them, a condition, and
   gold answers exactly when the question is answerable.
2. The CuratorKIT pipeline of the data build on each file: the schema gate; on the train file also exact dedup
   and MinHash dedup of the prompt.
3. ``cross_dedup`` of the data build, train against every test file, on the passage and again on the question:
   exact match, trigram Jaccard and MinHash at the build's threshold. A train row whose passage or question
   repeats or nears one of a test file is dropped; the test files are not changed by it.

Written under ``--out``: ``squad_passed.json``, per file its sha256, the rows in, the rows dropped per check and
the ids that passed, which ``probe_study squad-fit`` and ``squad`` read with ``--passed``; and
``squad_curator.json``, the counts of every check with the CuratorKIT version and settings. Neither holds any
text of SQuAD.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from driftmap.attribution import probe, probe_squad, reading

PASSED, REPORT = "squad_passed.json", "squad_curator.json"
SCHEMA = "driftmap.probe-squad-passed/1"
REPORT_SCHEMA = "driftmap.probe-squad-curator/1"
GATE = {"required_fields": ["instruction"], "min_tokens": 1, "max_tokens": 2048}
MINHASH = {"minhash_ngram": 3, "minhash_num_perm": 128}
# the texts a train row may not share with a test row, by the name of their check
CROSS = {"cross_set_passage": "context", "cross_set_question": "question"}
TRAIN_CHECKS = ("row_shape", "schema_gate", "exact_dedup", "minhash_dedup", *CROSS)
TEST_CHECKS = ("row_shape", "schema_gate")
TEXTS = ("id", "title", "context", "question", "prompt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(row: Mapping[str, Any], times: Mapping[str, int]) -> str | None:
    """Why a row is not a drawn question, or None; ``times`` counts the rows of the file under each id."""
    for name in TEXTS:
        if not isinstance(row.get(name), str) or not row[name].strip():
            return f"missing_field:{name}"
    if times[row["id"]] > 1:
        return "id_repeats"
    if row.get("condition") not in probe_squad.CONDITIONS:
        return "unknown_condition"
    answers = row.get("gold_answers")
    if not isinstance(answers, list) or bool(answers) != (row["condition"] == probe_squad.CONDITIONS[1]):
        return "gold_answers_do_not_match_the_condition"
    if row["prompt"] != "\n".join([row["context"], row["question"], reading.FREE]):
        return "prompt_is_not_passage_question_closing_line"
    return None


def config(name: str, staged: Path, dedup_steps: bool, threshold: float) -> str:
    """The pipeline over one staged file: the schema gate and, for the train file, the two dedup steps."""
    import yaml

    steps = [{"type": "exact_dedup"}, {"type": "minhash_dedup", "minhash_threshold": threshold, **MINHASH}]
    return yaml.safe_dump(
        {
            "name": f"probe-squad-{name}",
            "readers": [
                {
                    "type": "huggingface",
                    # the reader cannot pin a revision; it reads a staged copy
                    "path": str(staged),
                    "hf_split": "train",
                    "format": "prompt_only",
                    "field_mapping": {"prompt": "instruction"},
                    "source_uri": f"{reading.SQUAD}@{reading.SQUAD_REVISION[:12]}",
                }
            ],
            "gates": [{"type": "schema", **GATE}],
            "normalizers": steps if dedup_steps else [],
            "exporters": [{"type": "messages"}],
            "write_hf_readme": False,
        },
        sort_keys=False,
    )


def gate(name: str, path: Path, work: Path, dedup_steps: bool) -> tuple[list[dict], dict[str, set[str]], dict]:
    """The rows of a file that have the shape of a drawn question, the ids of them each check of the pipeline
    dropped, and the counts of the file."""
    from driftmap.data import curator, dedup

    rows = probe.read_jsonl(path)
    times = Counter(row.get("id") for row in rows if isinstance(row.get("id"), str))
    formed, reasons = [], Counter()
    for row in rows:
        reason = shape(row, times)
        if reason:
            reasons[reason] += 1
        else:
            formed.append(row)
    dropped = {check: set() for check in TRAIN_CHECKS[1:4]}
    staged = work / "staged" / name
    kept = ("prompt", "title", "condition")
    curator.write_jsonl(
        staged / curator.STAGED_FILE, [{"prompt_id": r["id"]} | {k: r[k] for k in kept} for r in formed]
    )
    result = curator.run_pipeline(config(name, staged, dedup_steps, dedup.THRESHOLD), work / "pipeline" / name)
    passed = {item.metadata["prompt_id"] for item in result.passed}
    # a row the reader could not build carries its columns as a preview
    dropped["schema_gate"] = {
        item.metadata.get("prompt_id") or item.metadata.get("raw_preview", {}).get("prompt_id")
        for item in result.rejected
    }
    gate_reasons = Counter(
        f"{item.rejecting_step}:{str(item.rejection_reason).split(':')[0]}" for item in result.rejected
    )
    if dedup_steps:
        # the two steps on their own, as the pipeline applies them: which rows each of them removes
        from curatorkit.normalizers.dedup import ExactDeduplicator
        from curatorkit.schema import DataSample

        uri = f"{reading.SQUAD}@{reading.SQUAD_REVISION[:12]}"
        samples = [
            DataSample(id=row["id"], source_uri=uri, instruction=row["prompt"], task_type="prompt_only")
            for row in formed
            if row["id"] not in dropped["schema_gate"]
        ]
        after_exact = {sample.id for sample in ExactDeduplicator().run(list(samples))}
        dropped["exact_dedup"] = {sample.id for sample in samples} - after_exact
        dropped["minhash_dedup"] = after_exact - passed
        if {sample.id for sample in samples} - dropped["exact_dedup"] - dropped["minhash_dedup"] != passed:
            raise SystemExit(f"{path.name}: the dedup steps on their own do not give the pipeline's rows")
    counts = {
        "config": f"{work.name}/pipeline/{name}/pipeline.yaml",
        "rows_in": len(rows),
        "rows_not_of_the_shape": len(rows) - len(formed),
        "row_shape_reasons": dict(sorted(reasons.items())),
        "rows_into_the_pipeline": len(formed),
        "rejected_by_the_gate": len(result.rejected),
        "rejection_reasons": dict(sorted(gate_reasons.items())),
        "rows_out": len(result.passed),
        "stage_counts": result.stage_counts,
    }
    return formed, dropped, counts


def cross(text: str, protected: Sequence[Mapping], train: Sequence[Mapping]) -> tuple[set[str], dict]:
    """The ids of the train rows whose ``text`` repeats or nears one of ``protected``, and the counts."""
    from driftmap.data import dedup

    asked = [{"prompt": row[text], "id": row["id"]} for row in train]
    found = dedup.cross_dedup([{"prompt": row[text]} for row in protected], asked)
    # a row is dropped as a copy of a protected text or as a repeat of an earlier train text; only a copy leaks
    alone = {row["id"] for row in dedup.cross_dedup([], asked).rows}
    copies = alone - {row["id"] for row in found.rows}
    keys = {dedup._key(row[text]) for row in protected}
    return copies, {
        "protected_texts": len(protected),
        "distinct_protected_texts": len(keys),
        "training_rows": len(train),
        "copies": len(copies),
        "exact_matches_after_normalisation": sum(dedup._key(row[text]) in keys for row in train),
        "training_rows_nearing_an_earlier_training_row_kept": len(train) - len(alone),
    }


def shared(one: Sequence[Mapping], two: Sequence[Mapping]) -> dict[str, int]:
    """How many ids, passages, questions and articles two files have in common, by exact text."""
    names = {"ids": "id", "passages": "context", "questions": "question", "articles": "title"}
    return {
        said: len({row.get(key) for row in one} & {row.get(key) for row in two} - {None}) for said, key in names.items()
    }


def passing(rows: list[dict], asked: Path, passed: Path) -> tuple[list[dict], dict]:
    """The rows of ``asked`` that passed the gate, and the gate's counts for the file; stops when the passing
    ids were not written for this file."""
    held = json.loads(passed.read_text(encoding="utf-8"))
    entry = held.get("files", {}).get(asked.name) if held.get("schema") == SCHEMA else None
    if entry is None:
        raise SystemExit(f"{passed.name} holds no passing ids for {asked.name}")
    if entry["sha256"] != sha256(asked):
        raise SystemExit(f"{passed.name} was written for another {asked.name}")
    ids_ = set(entry["passed"])
    if not ids_ <= {row["id"] for row in rows}:
        raise SystemExit(f"{passed.name} names rows {asked.name} does not hold")
    counts = {key: entry[key] for key in ("rows_in", "dropped", "rows_passed")}
    return [row for row in rows if row["id"] in ids_], counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=Path, required=True, help="prompt file of the train sample")
    parser.add_argument("--test", type=Path, action="append", required=True, help="prompt file of a test sample")
    parser.add_argument("--work", type=Path, required=True, help="scratch folder")
    parser.add_argument("--out", type=Path, required=True, help="folder for the passing ids and the report")
    args = parser.parse_args(argv)
    names = [path.name for path in (args.train, *args.test)]
    if len(set(names)) != len(names):
        parser.error("the prompt files must differ in name")
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    # the dataset reader keeps its cache inside the scratch folder and never asks the Hub; the libraries read this
    # at import, so they are imported after it
    os.environ["HF_DATASETS_CACHE"] = str(work / "hf_datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from driftmap.data import dedup

    files, rows, pipelines, gone = {}, {}, {}, {}
    for role, path in (("train", args.train), *(("test", path) for path in args.test)):
        train = role == "train"
        held, dropped_ids, pipelines[path.name] = gate(path.stem, path, work, dedup_steps=train)
        rows[path.name] = held
        counts = pipelines[path.name]
        dropped = {"row_shape": counts["rows_not_of_the_shape"]}
        dropped |= {check: len(ids_) for check, ids_ in dropped_ids.items()}
        files[path.name] = {"role": role, "sha256": sha256(path), "rows_in": counts["rows_in"], "dropped": dropped}
        gone[path.name] = set().union(*dropped_ids.values())
    left = [row for row in rows[args.train.name] if row["id"] not in gone[args.train.name]]
    compared = []
    for check, text in CROSS.items():
        copies = set()
        for path in args.test:
            found, counts = cross(text, rows[path.name], left)
            copies |= found
            compared.append({"check": check, "text": text, "train": args.train.name, "test": path.name} | counts)
        files[args.train.name]["dropped"][check] = len(copies)
        gone[args.train.name] |= copies
    passed = {"schema": SCHEMA, "files": {}}
    for name, entry in files.items():
        checks = TRAIN_CHECKS if entry["role"] == "train" else TEST_CHECKS
        ids_ = [row["id"] for row in rows[name] if row["id"] not in gone[name]]
        passed["files"][name] = {
            "role": entry["role"],
            "sha256": entry["sha256"],
            "rows_in": entry["rows_in"],
            "dropped": {check: entry["dropped"][check] for check in checks},
            "rows_passed": len(ids_),
            "passed": ids_,
        }
    report = {
        "schema": REPORT_SCHEMA,
        "curatorkit_version": importlib.metadata.version("curatorkit"),
        "script": __spec__.name,
        "runs": [
            "curatorkit.pipeline.Pipeline",
            "curatorkit.gates.schema.SchemaGate",
            "curatorkit.normalizers.dedup.ExactDeduplicator",
            "curatorkit.normalizers.dedup.MinHashDeduplicator",
            "driftmap.data.dedup.cross_dedup",
        ],
        "source": {"dataset": reading.SQUAD, "revision": reading.SQUAD_REVISION, "licence": reading.LICENCE},
        "sample": {"task_type": "prompt_only", "instruction": "the prompt as asked: passage, question, closing line"},
        "thresholds": {"schema_gate": GATE, "minhash_threshold": dedup.THRESHOLD, **MINHASH},
        "cross_set_rule": "exact match after normalisation, trigram Jaccard and MinHash at the MinHash threshold",
        "files": {
            name: {key: value for key, value in entry.items() if key != "passed"}
            | {
                "by_condition": dict(Counter(str(row.get("condition")) for row in rows[name])),
                "pipeline": pipelines[name],
            }
            for name, entry in passed["files"].items()
        },
        "cross_set": compared,
        "shared_by_exact_text": {
            f"{one} / {two}": shared(rows[one], rows[two]) for at, one in enumerate(names) for two in names[at + 1 :]
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / PASSED).write_text(json.dumps(passed, indent=1) + "\n", encoding="utf-8")
    (args.out / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, entry in passed["files"].items():
        print(name, json.dumps({key: value for key, value in entry.items() if key != "passed"}))
    for found in compared:
        print(json.dumps(found))
    print(json.dumps(report["shared_by_exact_text"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
