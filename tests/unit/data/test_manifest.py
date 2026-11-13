import json

import pytest
from pydantic import ValidationError

from driftmap.core.records import LangCounts, Manifest, Ref, load
from driftmap.core.settings import load_settings
from driftmap.data import manifest
from tests.fakes.settings import settings_toml

SOURCES = [
    Ref(kind="dataset", repo="CohereLabs/aya_redteaming", revision="3" * 40),
    Ref(kind="dataset", repo="CohereLabs/aya_dataset", revision="4" * 40),
    Ref(kind="dataset", repo="Paul/XSTest", revision="5" * 40),
    Ref(kind="model", repo="CohereLabs/tiny-aya-global", revision="1" * 40),
]
LICENCES = {
    "CohereLabs/aya_redteaming": "apache-2.0",
    "CohereLabs/aya_dataset": "apache-2.0",
    "Paul/XSTest": "cc-by-4.0",
}


def fields(**overrides) -> dict:
    counts = {
        language: LangCounts(harmful=100, undersampled=False, benign_train=323, benign_heldout=100)
        for language in ("en", "fr", "es", "ru", "hi", "ar")
    }
    counts["hi"] = LangCounts(harmful=80, undersampled=True, benign_train=320, benign_heldout=100)
    base = {
        "per_language": counts,
        "sources": SOURCES,
        "licences": LICENCES,
        "dropped_by_dedup": 2,
        "dropped_over_length": 3,
        "dropped_special_token_text": 1,
        "max_rendered_tokens": {"harmful": 512, "benign_heldout": 1700, "xstest": 400},
        "rendered_prefix_sha256": "ab" * 32,
        "note": "skipped languages: sr 5, tl 4",
    }
    return {**base, **overrides}


def test_manifest_records_counts_seed_sources_and_licences(tmp_path):
    """The written manifest carries every provenance field; a source without a licence is refused (TS-DATA-5)."""
    settings = load_settings(settings_toml(tmp_path))
    path = tmp_path / "out" / "manifest.json"

    written = manifest.write(path, settings, **fields())

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "driftmap.manifest/1"
    assert data["seed"] == 20261102
    assert data["seed_derivation"] == "sha256(f'{seed}:{language}')[:8] big-endian"
    assert (data["n_harmful"], data["n_heldout"]) == (600, 600)
    assert data["per_language"]["hi"] == {
        "harmful": 80,
        "undersampled": True,
        "benign_train": 320,
        "benign_heldout": 100,
    }
    assert data["per_language"]["en"]["undersampled"] is False
    assert [source["revision"] for source in data["sources"]] == ["3" * 40, "4" * 40, "5" * 40, "1" * 40]
    assert data["licences"] == LICENCES
    assert (data["dropped_by_dedup"], data["dropped_over_length"], data["dropped_special_token_text"]) == (2, 3, 1)
    assert data["max_rendered_tokens"] == {"harmful": 512, "benign_heldout": 1700, "xstest": 400}
    assert data["rendered_prefix_sha256"] == "ab" * 32
    assert data["note"] == "skipped languages: sr 5, tl 4"
    assert type(written) is Manifest and load(Manifest, path) == written

    without = {repo: licence for repo, licence in LICENCES.items() if repo != "Paul/XSTest"}
    for licences in (without, {**LICENCES, "Paul/XSTest": ""}):
        with pytest.raises(ValidationError, match="Paul/XSTest"):
            manifest.write(tmp_path / "refused.json", settings, **fields(licences=licences))
    assert not (tmp_path / "refused.json").exists()


def test_manifest_refuses_an_unpinned_source(tmp_path):
    settings = load_settings(settings_toml(tmp_path))
    with pytest.raises(ValidationError):
        manifest.write(
            tmp_path / "m.json",
            settings,
            **fields(sources=[{"kind": "dataset", "repo": "Paul/XSTest", "revision": "main"}]),
        )


def test_card_names_sources_counts_and_note(tmp_path):
    settings = load_settings(settings_toml(tmp_path))
    text = manifest.card(manifest.write(tmp_path / "m.json", settings, **fields()))
    assert "CohereLabs/aya_redteaming@" + "3" * 40 in text
    assert "cc-by-4.0" in text
    assert "| hi | 80 (undersampled) | 100 | 320 |" in text
    assert "skipped languages: sr 5, tl 4" in text
    assert "sha256(f'{seed}:{language}')[:8] big-endian" in text
