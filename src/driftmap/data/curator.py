"""The CuratorKIT pipeline that cleans the benign fine-tuning rows."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

# the YAML runner's step builder; the package exposes no public one
from curatorkit.cli import _build_steps
from curatorkit.config import PipelineConfig
from curatorkit.exporters import EXPORTERS
from curatorkit.exporters.messages import build_messages
from curatorkit.interfaces import BaseReader
from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest
from curatorkit.pipeline import Pipeline, PipelineResult
from curatorkit.schema import DataSample

from driftmap.core.settings import Settings
from driftmap.data.dedup import THRESHOLD
from driftmap.data.sampler import Row

STAGED_FILE = "train.jsonl"


def write_jsonl(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def stage(rows: Iterable[Row], directory: Path) -> Path:
    """Write training candidates in the source's column names, as a local dataset directory."""
    staged = [
        {"inputs": row["prompt"], "targets": row["target"], "language": row["language"], "prompt_id": row["prompt_id"]}
        for row in rows
    ]
    write_jsonl(directory / STAGED_FILE, staged)
    return directory


def render_yaml(settings: Settings, staged: Path, revision: str) -> str:
    """Pipeline config over the staged rows of the benign source at ``revision``."""
    config = {
        "name": "benign-sft",
        "readers": [
            {
                "type": "huggingface",
                # the reader cannot pin a revision; it reads a staged copy
                "path": str(staged),
                "hf_split": "train",
                "format": "alpaca",
                "field_mapping": {"inputs": "instruction", "targets": "output"},
                "source_uri": f"{settings.sources.benign}@{revision}",
            }
        ],
        # the default minimum of ten words would reject short question-answer rows
        "gates": [
            {"type": "schema", "required_fields": ["instruction", "output"], "min_tokens": 1, "max_tokens": 2048}
        ],
        "normalizers": [
            {"type": "exact_dedup"},
            {"type": "minhash_dedup", "minhash_threshold": THRESHOLD, "minhash_ngram": 3, "minhash_num_perm": 128},
        ],
        "exporters": [{"type": "messages"}],
        # fractions only; the held-out rows are carved before this pipeline runs
        "output_split": {"sft_train": 1.0},
        "output_split_seed": settings.seed,
        "write_hf_readme": False,
    }
    return yaml.safe_dump(config, sort_keys=False)


def run_pipeline(config_text: str, work_dir: Path, reader: BaseReader | None = None) -> PipelineResult:
    """Run the pipeline into ``work_dir``; ``reader`` stands in for the configured readers."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "pipeline.yaml"
    path.write_text(config_text, encoding="utf-8")
    config = PipelineConfig.from_yaml(path)
    steps, _ = _build_steps(config, False, include_exporters=False)
    if reader is not None:
        steps = [reader, *(step for step in steps if not isinstance(step, BaseReader))]
    result = Pipeline(steps, output_dir=work_dir).run()
    written = [path]
    for split in config.output_split or {}:
        for exporter in config.exporters:
            (work_dir / split).mkdir(exist_ok=True)
            EXPORTERS[exporter.type]().export(result.passed, work_dir / split)
            written.append(work_dir / split / EXPORTERS[exporter.type].filename)
    provenance = ProvenanceManifest(result, hashlib.sha256(path.read_bytes()).hexdigest()[:16], work_dir)
    written += [provenance.write(), provenance.write_rejected_sidecar()]
    provenance.write_checksums(written)
    DatasetCardGenerator().generate(provenance.build(), work_dir, pipeline_name=config.name, write_readme=False)
    return result


def sft_rows(samples: Iterable[DataSample]) -> dict[str, list[Row]]:
    """Prompt-completion conversational rows by language: the user turn, then the assistant turn."""
    rows: dict[str, list[Row]] = {}
    for sample in samples:
        *prompt, completion = build_messages(sample)
        rows.setdefault(sample.metadata["language"], []).append({"prompt": prompt, "completion": [completion]})
    return rows


def write_sft(samples: Iterable[DataSample], out_dir: Path, languages: Sequence[str]) -> dict[str, int]:
    """Write ``sft_train/<language>.jsonl`` for each language; returns the row counts."""
    rows = sft_rows(samples)
    for language in languages:
        write_jsonl(out_dir / "sft_train" / f"{language}.jsonl", rows.get(language, []))
    return {language: len(rows.get(language, [])) for language in languages}
