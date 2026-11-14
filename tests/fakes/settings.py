import json
from datetime import date
from pathlib import Path
from typing import Any

BASE: dict[str, Any] = {
    "owner": "vincentnkq",
    "languages": ["en", "fr", "es", "ru", "hi", "ar"],
    "n_harmful": 600,
    "n_heldout": 600,
    "seed": 20261102,
    "preamble": "keep",
    "runner": "pip",
    "scheduling_allowance_s": 1800,
    "gated_repos": ["CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"],
    "repos": {
        "adapter": "tiny-aya-global-drift-lora",
        "merged": "tiny-aya-global-drift-merged",
        "repaired": "tiny-aya-global-repaired",
        "gguf": "tiny-aya-global-repaired-GGUF",
        "eval": "aya-redteaming-6lang-eval",
        "results": "safety-drift-map-results",
        "responses": "safety-drift-map-responses",
        "models_collection": "tiny-aya-safety-drift-map-models",
        "data_collection": "tiny-aya-safety-drift-map-data",
    },
    "sources": {
        "model": "CohereLabs/tiny-aya-global",
        "base": "CohereLabs/tiny-aya-base",
        "harmful": "CohereLabs/aya_redteaming",
        "benign": "CohereLabs/aya_dataset",
        "xstest": "Paul/XSTest",
    },
    "ceilings": {
        "per_run": 5_000_000,
        "smoke": 2_000_000,
        "soft": 16_000_000,
        "hard": 20_000_000,
        "own_money": 0,
        "envelope": 13_600_000,
    },
    "flavors": {"gpu": "l4x1", "cpu": "cpu-upgrade"},
    "images": {"pytorch": "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"},
    "llamacpp_release": {
        "tag": "b11960",
        "source_tarball": "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11960.tar.gz",
        "cpu_asset": "https://github.com/ggml-org/llama.cpp/releases/download/b11960/llama-b11960-bin-ubuntu-x64.tar.gz",
        "cuda_asset": (
            "https://github.com/ggml-org/llama.cpp/releases/download/b11960/llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz"
        ),
        "sha256": {
            "source_tarball": "2b37ffa871595f90b94164f6c510ad9a34bd7f94da21557cc9878587e163af37",
            "cpu_asset": "31cc3a6bc638ac6c48dc4b1b74888f1eaadf3d14227c1e5f67daf9a49541d97a",
            "cuda_asset": "64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4",
        },
    },
    "timeouts": {
        "drift": 180,
        "generate": 90,
        "repair": 120,
        "steer": 90,
        "quant": 60,
        "quant_eval": 120,
        "smoke": 30,
    },
    "batch": {"drift": 4, "drift_accum": 4, "generate": 16, "judge": 16, "screen": 8},
    "decoding": {"max_new_tokens": 256},
    "judge": {
        "repo": "ToxicityPrompts/PolyGuard-Qwen",
        "revision": "644bfe73ff498c9a14818b72a11187eaf23f0ff1",
        "dtype": "bfloat16",
        "parser_version": "1",
        "smol_repo": "ToxicityPrompts/PolyGuard-Qwen-Smol",
        "smol_revision": "224f511fe2ee3e304a11f470c8071f43ee0d8f70",
        "closed": {"name": "closed-model-a", "rubric": "rubric-v1"},
    },
    "code": {"origin": "https://github.com/vincentni/tiny-aya-safety-drift-map", "branch": "main"},
    "repair": {
        "alpha": 0.25,
        "alphas": [0.1, 0.25, 0.4, 0.5],
        "bars": {"safety_points": 5.0, "chrf_points": 1.0},
        "dare": {"on": True, "drop_rate": 0.3, "seed": 0},
    },
    "readiness": {"registration_confirmed": True, "confirmed_on": date(2026, 11, 7)},
    "publish": {"space_not_serving": False},
    "prices_micro_usd_per_min": {"l4x1": 13_333, "a10g-small": 16_667, "cpu-upgrade": 500, "cpu-basic": 167},
    "revisions": {
        "CohereLabs/tiny-aya-global": "1" * 40,
        "CohereLabs/tiny-aya-base": "2" * 40,
        "CohereLabs/aya_redteaming": "3" * 40,
        "CohereLabs/aya_dataset": "4" * 40,
        "Paul/XSTest": "5" * 40,
    },
}


def _value(value: Any) -> str:
    return value.isoformat() if isinstance(value, date) else json.dumps(value, ensure_ascii=False)


def to_toml(data: dict[str, Any], prefix: tuple[str, ...] = ()) -> str:
    """Nested dicts as TOML tables; scalars and lists only below that."""
    lines = [f"[{'.'.join(json.dumps(part) for part in prefix)}]"] if prefix else []
    lines += [f"{json.dumps(key)} = {_value(value)}" for key, value in data.items() if not isinstance(value, dict)]
    tables = [to_toml(value, (*prefix, key)) for key, value in data.items() if isinstance(value, dict)]
    return "\n".join([*lines, "", *tables])


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def settings_toml(tmp_path: Path, **overrides: Any) -> Path:
    """Write a complete driftmap.toml; a dict override merges, None drops the key."""
    data = _merge({"lock_path": str(tmp_path / "launch.lock"), **BASE}, overrides)
    revisions = data.pop("revisions", None)
    path = tmp_path / "driftmap.toml"
    path.write_text(to_toml(data), encoding="utf-8")
    if revisions is not None:
        (tmp_path / "revisions.json").write_text(json.dumps(revisions), encoding="utf-8")
    return path
