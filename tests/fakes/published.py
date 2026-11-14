"""The project's Hub repos as the jobs leave them: private, synthetic rows, no dataset text."""

import json
from pathlib import Path

from driftmap.core.hub import Hub
from driftmap.core.records import LangCounts, Manifest, Ref, RunRecord, Tables
from driftmap.core.schemas import SCHEMA_MANIFEST
from driftmap.core.settings import Settings
from driftmap.data.sampler import SEED_DERIVATION
from driftmap.table.build import build
from tests.fakes.clock import FakeClock
from tests.fakes.hub import FakeHub
from tests.fakes.labels import LANGUAGES, OWNER, column_rows, run_record, targets, write_labels
from tests.fakes.settings import BASE

REPOS = BASE["repos"]
SOURCES = BASE["sources"]
EVAL, RESULTS, RESPONSES = (f"{OWNER}/{REPOS[key]}" for key in ("eval", "results", "responses"))
ADAPTER, MERGED, REPAIRED, GGUF = (f"{OWNER}/{REPOS[key]}" for key in ("adapter", "merged", "repaired", "gguf"))
DATASETS, MODELS = (EVAL, RESULTS, RESPONSES), (ADAPTER, REPAIRED, GGUF)
SMOKE = (f"{ADAPTER}-smoke", f"{MERGED}-smoke")
SLUG = f"{REPOS['merged']}--{'a' * 12}"
MERGED_SHA256 = "5" * 64
# what a drift run with no safety preamble pushes to, and what its smoke run does
ADAPTER_NOPRE, MERGED_NOPRE = f"{ADAPTER}-nopre", f"{MERGED}-nopre"
NOPRE_SMOKE = (f"{ADAPTER_NOPRE}-smoke", f"{MERGED_NOPRE}-smoke")
NOPRE_SLUG = f"{REPOS['merged']}-nopre--{'b' * 12}"
NOPRE_SHA256 = "6" * 64
LICENCES = {SOURCES["harmful"]: "apache-2.0", SOURCES["benign"]: "apache-2.0", SOURCES["xstest"]: "cc-by-4.0"}


# what the trainer keeps pushing beside the adapter while it trains
CHECKPOINT = {
    "adapter_model.safetensors": b"\x00weights",
    "optimizer.pt": b"PK\x03\x04optimizer",
    "training_args.bin": b"\x80\x04\x95args",
    "trainer_state.json": b'{"global_step": 300}\n',
    "run_stamp.json": b'{"run_id": "drift-None"}\n',
}
LABEL = {"schema": "driftmap.label/1", "prompt_id": "harmful-en-000", "prompt": "placeholder harmful prompt 000"}
RESPONSE = {"prompt_id": "harmful-en-000", "response": "placeholder"}
WEIGHTS = {"config.json": b'{"model_type": "cohere2"}\n', "model.safetensors": b"\x00weights"}


def adapter_files() -> dict[str, bytes]:
    """An adapter repo as a finished drift run leaves it, the trainer's last checkpoint included."""
    return {
        "README.md": b"---\nbase_model: CohereLabs/tiny-aya-global\nlibrary_name: peft\n---\n# Model Card\n",
        "adapter_config.json": b'{"r": 16, "lora_alpha": 32}\n',
        "adapter_model.safetensors": b"\x00weights",
        "drift_config.json": b'{"epochs": 1}\n',
        "training_args.bin": CHECKPOINT["training_args.bin"],
        **{f"last-checkpoint/{name}": data for name, data in CHECKPOINT.items()},
    }


def manifest() -> Manifest:
    return Manifest(
        schema=SCHEMA_MANIFEST,
        seed=BASE["seed"],
        seed_derivation=SEED_DERIVATION,
        per_language={
            code: LangCounts(harmful=100, undersampled=False, benign_train=212, benign_heldout=100)
            for code in LANGUAGES
        },
        sources=[
            *(Ref(kind="dataset", repo=repo, revision="2" * 40) for repo in LICENCES),
            Ref(kind="model", repo=SOURCES["model"], revision="1" * 40),
        ],
        licences=LICENCES,
        dropped_by_dedup=3,
        dropped_over_length=0,
        dropped_special_token_text=0,
        max_rendered_tokens={"harmful": 310, "benign_heldout": 420, "xstest": 60},
        rendered_prefix_sha256="f" * 64,
    )


def drift_record() -> RunRecord:
    return run_record("drift", merged_sha256=MERGED_SHA256)


def nopre_drift_record() -> RunRecord:
    return run_record("drift", run_id="drift-nopre", merged_sha256=NOPRE_SHA256, notes=["preamble: strip"])


def two_column_tables(tmp_path: Path, settings: Settings, **not_run: str) -> Tables:
    """Baseline and drifted tabled; the repaired columns have no result."""
    rows = column_rows("baseline", n=20, unsafe=2) + column_rows("drifted", n=20, unsafe=9)
    return build(write_labels(tmp_path / "labels", rows), targets(20), [], settings, not_run=not_run)


def _rows(*rows: dict) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _json(record) -> bytes:
    return (record.model_dump_json(by_alias=True, indent=2) + "\n").encode()


def seed(fake: FakeHub, *, tables: Tables | None = None, skip: tuple[str, ...] = ()) -> Hub:
    """Six private repos, the private intermediates and two public, empty collections; returns the wrapper."""
    files = {
        EVAL: {
            "manifest.json": _json(manifest()),
            "dataset_card.md": b"# Six-language safety evaluation set\n",
            "prompts/harmful/en.jsonl": _rows({"prompt_id": "harmful-en-000", "prompt": "placeholder prompt 000"}),
            "sft_train/en.jsonl": _rows({"messages": [{"role": "user", "content": "placeholder question"}]}),
            f"labels/{SLUG}/en.jsonl": _rows(LABEL),
            f"labels/{SLUG}/superseded/{'e' * 64}/en.jsonl": _rows(LABEL),
            f"smoke/labels/{SLUG}/en.jsonl": _rows(LABEL),
        },
        RESULTS: {
            "runs/drift-None.json": _json(drift_record()),
            "ledger.jsonl": _rows({"schema": "driftmap.ledger/1", "run_id": "drift-None", "cost_micro_usd": 799_980}),
            "logs/drift-None.log": b"2026-11-13 10:00:00 INFO driftmap: step 50 of 318\n",
            "smoke/runs/drift-smoke.json": b"{}\n",
            **({"tables/tables.json": _json(tables)} if tables else {}),
        },
        RESPONSES: {
            f"responses/{SLUG}/harmful/en.jsonl": _rows({"prompt_id": "harmful-en-000", "response": "placeholder"}),
            f"judge_raw/{SLUG}/en.jsonl": _rows({"prompt_id": "harmful-en-000", "raw": "Harmful request: yes"}),
            f"smoke/responses/{SLUG}/harmful/en.jsonl": _rows({"prompt_id": "harmful-en-000", "response": "x"}),
        },
        ADAPTER: adapter_files(),
        REPAIRED: dict(WEIGHTS),
        GGUF: {"tiny-aya-global-repaired.q4_k_m.gguf": b"GGUF\x00", "gguf.json": b'{"quant": "q4_k_m"}\n'},
        MERGED: dict(WEIGHTS),
        **{repo: {"adapter_config.json": b"{}\n"} for repo in SMOKE},
    }
    for repo, tree in files.items():
        if repo not in skip:
            fake.add_repo(repo, repo_type="dataset" if repo in DATASETS else "model", private=True, files=tree)
            fake.authorize(repo)
    for name in (REPOS["models_collection"], REPOS["data_collection"]):
        fake.create_collection(name, namespace=OWNER)
    fake.calls.clear()
    return Hub(fake, sleep=FakeClock().sleep)


def add_nopreamble(fake: FakeHub, *, tables: Tables | None = None) -> None:
    """What the runs with no safety preamble add to a seeded Hub: their trees, their drift record and its repos."""
    trees = {
        EVAL: {
            f"nopreamble/labels/{SLUG}/en.jsonl": _rows(LABEL),
            f"nopreamble/labels/{NOPRE_SLUG}/en.jsonl": _rows(LABEL),
            f"nopreamble/labels/{NOPRE_SLUG}/superseded/{'e' * 64}/en.jsonl": _rows(LABEL),
            f"smoke/nopreamble/labels/{NOPRE_SLUG}/en.jsonl": _rows(LABEL),
        },
        RESULTS: {
            "runs/drift-nopre.json": _json(nopre_drift_record()),
            "nopreamble/tables/tables-nopreamble.md": b"# Safety drift map (no safety preamble)\n",
            "smoke/nopreamble/tables/tables-nopreamble.md": b"# Safety drift map (no safety preamble)\n",
            **({"nopreamble/tables/tables-nopreamble.json": _json(tables)} if tables else {}),
        },
        RESPONSES: {
            f"nopreamble/responses/{NOPRE_SLUG}/harmful/en.jsonl": _rows(RESPONSE),
            f"nopreamble/judge_raw/{NOPRE_SLUG}/en.jsonl": _rows({"prompt_id": "harmful-en-000", "raw": "{}"}),
            f"smoke/nopreamble/responses/{NOPRE_SLUG}/harmful/en.jsonl": _rows(RESPONSE),
        },
    }
    for repo, files in trees.items():
        fake._commit(repo, "job upload", files)
    repos = {
        ADAPTER_NOPRE: adapter_files(),
        MERGED_NOPRE: dict(WEIGHTS),
        **{repo: {"adapter_config.json": b"{}\n"} for repo in NOPRE_SMOKE},
    }
    for repo, files in repos.items():
        fake.add_repo(repo, private=True, files=files)
        fake.authorize(repo)


def writes(fake: FakeHub) -> list[tuple[str, str]]:
    """Calls that change the Hub, as (method, repo or collection)."""
    changing = ("upload_folder", "upload_file", "update_repo_settings", "delete_repo", "create_repo")
    found = [(method, kwargs["repo_id"]) for method, kwargs in fake.calls if method in changing]
    shelved = [(m, k["item_id"]) for m, k in fake.calls if m == "add_collection_item"]
    return found + shelved + [(m, k["title"]) for m, k in fake.calls if m == "create_collection"]
