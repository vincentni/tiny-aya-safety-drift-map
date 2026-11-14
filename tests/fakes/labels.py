"""Synthetic label rows and run records: placeholder ids and labels, no dataset text."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from driftmap.core.records import Decoding, JudgeConfig, LabelRow, Ref, RunRecord, Seeds, Tables
from driftmap.core.schemas import SCHEMA_LABEL, SCHEMA_RUN, SCHEMA_TABLES
from driftmap.core.settings import Settings, load_settings
from tests.fakes.settings import BASE, settings_toml

LANGUAGES = tuple(BASE["languages"])
OWNER = BASE["owner"]
PREFIX = "f" * 64
REFS = {
    "baseline": ("CohereLabs/tiny-aya-global", "1" * 40),
    "drifted": (f"{OWNER}/tiny-aya-global-drift-merged", "a" * 40),
    "repaired": (f"{OWNER}/tiny-aya-global-repaired", "b" * 40),
    "repaired_q4": (f"{OWNER}/tiny-aya-global-repaired-GGUF", "c" * 40),
}
CHECKPOINTS = {column: f"{repo}@{revision}" for column, (repo, revision) in REFS.items()}


def make_settings(tmp_path: Path) -> Settings:
    return load_settings(settings_toml(tmp_path))


def judge_config(**changes) -> JudgeConfig:
    fields = {
        "repo": BASE["judge"]["closed"]["name"],
        "revision": "n/a",
        "dtype": "n/a",
        "decoding": Decoding(greedy=False, runtime="api", max_new_tokens=1024),
        "prompt_format": "rubric-v1",
        "parser_version": "1",
        "config_hash": "judge-1",
    }
    return JudgeConfig(**{**fields, **changes})


def bare_tables(**flags: bool) -> Tables:
    """A result table with no rows: what a launch reads of it is whether it is there, and its flags."""
    return Tables(
        schema=SCHEMA_TABLES,
        columns={},
        safety=[],
        over_refusal=[],
        chrf=[],
        deltas=[],
        flags=flags,
        judge=judge_config(),
        seeds=Seeds(seed=BASE["seed"], bootstrap=42),
    )


def prompt_ids(set_: str, language: str, n: int) -> list[str]:
    return [f"{set_}-{language}-{index:03d}" for index in range(n)]


def label(column: str, language: str, index: int, *, set_: str = "harmful", harm: str = "no", **changes) -> LabelRow:
    """One judged row; a benign row takes ``chrf`` and carries no judge."""
    judged = set_ != "benign"
    fields = {
        "schema": SCHEMA_LABEL,
        "checkpoint": CHECKPOINTS[column],
        "language": language,
        "set": set_,
        "prompt_id": prompt_ids(set_, language, index + 1)[-1],
        "prompt": f"placeholder {set_} prompt {index:03d}",
        "rendered_prefix_sha256": PREFIX,
        "request_harm": "yes" if judged else "n/a",
        "refusal": {"no": "yes", "yes": "no"}.get(harm, harm) if judged else "n/a",
        "response_harm": harm if judged else "n/a",
        "raw_output_present": True,
        "judge": judge_config() if judged else None,
        "chrf": None,
    }
    return LabelRow(**{**fields, **changes, **({} if judged else {"judge": None})})


def column_rows(column: str, *, n: int = 4, unsafe=0, refused: int = 0, chrf: float = 0.5, **changes):
    """Per language ``n`` harmful (``unsafe`` of them harmful) and ``n`` benign rows, plus ``n`` English XSTest rows."""
    rows = []
    for language in LANGUAGES:
        bad = unsafe.get(language, 0) if isinstance(unsafe, dict) else unsafe
        rows += [label(column, language, i, harm="yes" if i < bad else "no", **changes) for i in range(n)]
        rows += [label(column, language, i, set_="benign", chrf=chrf, **changes) for i in range(n)]
    xstest = [
        label(column, "en", i, set_="xstest", refusal="yes" if i < refused else "no", **changes) for i in range(n)
    ]
    return rows + xstest


def targets(n: int = 4) -> dict[str, dict[str, list[str]]]:
    """Prompt ids per set and language, as the eval dataset lists them."""
    sets = {name: {language: prompt_ids(name, language, n) for language in LANGUAGES} for name in ("harmful", "benign")}
    return {**sets, "xstest": {"en": prompt_ids("xstest", "en", n)}, "train": {"en": prompt_ids("train", "en", n)}}


def write_labels(root: Path, rows: list[LabelRow]) -> Path:
    """Lay rows out as ``<checkpoint dir>/[benign/]<language>.jsonl``."""
    files = defaultdict(list)
    for row in rows:
        folder = Path(row.checkpoint.partition("@")[0].rpartition("/")[2])
        files[folder / ("benign" if row.set == "benign" else "") / f"{row.language}.jsonl"].append(row)
    for path, group in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("".join(row.model_dump_json(by_alias=True) + "\n" for row in group), encoding="utf-8")
    return root


def run_record(kind: str, column: str | None = None, **changes) -> RunRecord:
    """A finished run; ``column`` names the checkpoint it read."""
    inputs = [Ref(kind="model", repo=REFS[column][0], revision=REFS[column][1])] if column else []
    fields = {
        "schema": SCHEMA_RUN,
        "run_id": f"{kind}-{column}",
        "job_id": "job-1",
        "kind": kind,
        "mode": "full",
        "config_hash": "config-1",
        "code_sha": "d" * 40,
        "inputs": inputs,
        "outputs": [],
        "merged_sha256": None,
        "command": [f"{kind}.py"],
        "started": datetime(2026, 11, 13, tzinfo=UTC),
        "finished": datetime(2026, 11, 13, 1, tzinfo=UTC),
        "wall_clock_s": 3600.0,
        "flavor": "l4x1",
        "price_micro_usd_per_min": 13_333,
        "minutes": 60,
        "cost_micro_usd": 799_980,
        "prompts": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "tokens_per_s": None,
        "status": "ok",
        "partial": None,
        "oom": None,
        "peak_vram_gb": None,
        "peak_rss_gb": None,
        "install_minutes": None,
        "notes": [],
    }
    return RunRecord(**{**fields, **changes})
