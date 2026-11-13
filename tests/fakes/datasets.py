"""Synthetic source datasets in the column shapes of the real ones."""

import csv
import hashlib
import io
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tests.fakes.hub import FakeHub
from tests.fakes.tokenizers import build as build_tokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"
HARMFUL = "CohereLabs/aya_redteaming"
BENIGN = "CohereLabs/aya_dataset"
XSTEST = "Paul/XSTest"
MODEL = "CohereLabs/tiny-aya-global"
NAMES = {
    "arb": "Arabic",
    "eng": "English",
    "fra": "French",
    "hin": "Hindi",
    "rus": "Russian",
    "spa": "Spanish",
    "srp": "Serbian",
    "tgl": "Filipino",
}
SIX = ("eng", "fra", "spa", "rus", "hin", "arb")
# made-up words; rows built from them share few character trigrams
WORDS = [
    a + b for a in ("ka", "lo", "mi", "ne", "pu", "ra", "si", "to") for b in "ban dex fim gol hur jat kem liv".split()
]
CARD = "---\nlicense: {licence}\n---\n# synthetic card\n"


def gibberish(key: str, words: int = 10) -> str:
    """Obviously fake text, stable for a key."""
    return " ".join(random.Random(key).sample(WORDS, words))


def fixture_rows(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def harmful_rows(iso3: str, n_global: int, n_local: int = 0) -> list[dict[str, Any]]:
    scopes = ["global"] * n_global + ["local"] * n_local
    return [
        {
            "prompt": f"synthetic harmful placeholder {gibberish(f'harmful-{iso3}-{i}')}",
            "language": NAMES.get(iso3, iso3),
            "harm_category": '["Placeholder Category"]',
            "global_or_local": scope,
            "literal_translation": None,
            "semantic_translation": None,
            "explanation": None,
        }
        for i, scope in enumerate(scopes)
    ]


def benign_rows(iso3: str, count: int, tag: str = "train") -> list[dict[str, Any]]:
    return [
        {
            "inputs": f"synthetic question {gibberish(f'q-{tag}-{iso3}-{i}')}",
            "targets": f"synthetic answer {gibberish(f'a-{tag}-{iso3}-{i}', 6)}",
            "language": NAMES.get(iso3, iso3),
            "language_code": iso3,
            "annotation_type": "original-annotations",
            "user_id": hashlib.sha256(f"{iso3}-{i % 3}".encode()).hexdigest(),
        }
        for i in range(count)
    ]


def xstest_rows(safe: int, unsafe: int) -> list[dict[str, Any]]:
    labels = ["safe"] * safe + ["unsafe"] * unsafe
    return [
        {
            "id": i + 1,
            "prompt": f"synthetic {label} placeholder {gibberish(f'xstest-{i}', 7)}?",
            "type": "homonyms" if label == "safe" else "contrast_homonyms",
            "label": label,
            "focus": "placeholder",
            "note": "",
        }
        for i, label in enumerate(labels)
    ]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    columns = ["inputs", "targets", "language", "language_code", "annotation_type", "user_id"]
    table = pa.table({name: pa.array([row[name] for row in rows], pa.string()) for name in columns})
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def _csv(rows: list[dict[str, Any]]) -> bytes:
    sink = io.StringIO(newline="")
    writer = csv.DictWriter(sink, fieldnames=["id", "prompt", "type", "label", "focus", "note"])
    writer.writeheader()
    writer.writerows(rows)
    return sink.getvalue().encode()


def tokenizer_files() -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as scratch:
        build_tokenizer().save_pretrained(scratch)
        return {path.name: path.read_bytes() for path in Path(scratch).iterdir() if path.is_file()}


def source_hub(
    root: Path,
    *,
    harmful: dict[str, list[dict[str, Any]]],
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    xstest: list[dict[str, Any]],
    licences: dict[str, str | None] | None = None,
) -> FakeHub:
    """A FakeHub holding the three source datasets and the tokenizer repo."""
    licence = {HARMFUL: "apache-2.0", BENIGN: "apache-2.0", XSTEST: "cc-by-4.0", **(licences or {})}

    def card(repo: str) -> bytes:
        return (CARD.format(licence=licence[repo]) if licence[repo] else "# synthetic card\n").encode()

    hub = FakeHub(root)
    files = {f"aya_{iso3}.jsonl": _jsonl(rows) for iso3, rows in harmful.items()}
    hub.add_repo(HARMFUL, repo_type="dataset", files={"README.md": card(HARMFUL), **files})
    hub.add_repo(
        BENIGN,
        repo_type="dataset",
        files={
            "README.md": card(BENIGN),
            "data/train-00000-of-00001.parquet": _parquet(train),
            "data/test-00000-of-00001.parquet": _parquet(test),
            "demographics/train-00000-of-00001.parquet": b"not read",
        },
    )
    hub.add_repo(XSTEST, repo_type="dataset", files={"README.md": card(XSTEST), "xstest_prompts.csv": _csv(xstest)})
    hub.add_repo(MODEL, files={"config.json": b"{}", "model.safetensors": b"weights", **tokenizer_files()})
    return hub


def fixture_hub(root: Path) -> FakeHub:
    """The sources as the fixture files hold them: eight harmful languages, a few rows each."""
    iso3 = {name: code for code, name in NAMES.items()}
    harmful: dict[str, list[dict[str, Any]]] = {}
    for row in fixture_rows("aya_redteaming_sample.jsonl"):
        harmful.setdefault(iso3[row["language"]], []).append(row)
    return source_hub(
        root,
        harmful=harmful,
        train=fixture_rows("aya_dataset_sample.jsonl"),
        test=benign_rows("eng", 3, "test") + benign_rows("arb", 3, "test"),
        xstest=fixture_rows("xstest_sample.jsonl"),
    )


def heads(hub: FakeHub) -> dict[str, str]:
    return {repo: hub.heads[repo] for repo in (HARMFUL, BENIGN, XSTEST, MODEL)}
