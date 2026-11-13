import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml
from curatorkit.interfaces import BaseReader
from curatorkit.schema import DataSample
from hypothesis import given
from hypothesis import strategies as st

from driftmap.core import ids
from driftmap.core.hub import Hub
from driftmap.core.settings import load_settings
from driftmap.data import ISO3, curator
from driftmap.data.build import build
from driftmap.data.sampler import allocate, sample
from tests.fakes.datasets import (
    BENIGN,
    HARMFUL,
    MODEL,
    SIX,
    XSTEST,
    benign_rows,
    fixture_hub,
    fixture_rows,
    harmful_rows,
    heads,
    source_hub,
    xstest_rows,
)
from tests.fakes.settings import settings_toml

SRC = Path(__file__).resolve().parents[3] / "src" / "driftmap" / "data"
EVAL = "vincentnkq/aya-redteaming-6lang-eval"
LANGUAGES = {"en", "fr", "es", "ru", "hi", "ar"}
CHILD = """
import json, sys
from driftmap.data.sampler import sample
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
print(json.dumps([row["prompt_id"] for row in sample(rows, 100, 20261102, "en").rows]))
"""


def settings_for(tmp_path, fake, **overrides):
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    overrides.setdefault("revisions", {"CohereLabs/tiny-aya-base": None, **heads(fake)})
    return load_settings(settings_toml(config, **{"n_harmful": 12, "n_heldout": 12, **overrides}))


def wrap(fake) -> Hub:
    return Hub(fake, sleep=lambda seconds: None)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sample_is_the_same_in_every_process(tmp_path, child_env):
    """Two processes draw the same 100 ids, 62 global and 38 local, and allocations sum to N (TS-DATA-1)."""
    rows = harmful_rows("eng", 613, 374)
    for row in rows:
        row["prompt_id"] = ids.prompt_id(HARMFUL, "3" * 40, row["prompt"])
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def child(hash_seed: str) -> list[str]:
        done = subprocess.run(
            [sys.executable, "-c", CHILD, str(source)],
            env={**child_env, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(done.stdout)

    first, second = child("1"), child("2")
    assert first == second == [row["prompt_id"] for row in sample(rows, 100, 20261102, "en").rows]
    assert len(set(first)) == 100
    scope = {row["prompt_id"]: row["global_or_local"] for row in rows}
    assert Counter(scope[prompt_id] for prompt_id in first) == {"global": 62, "local": 38}

    @given(st.dictionaries(st.sampled_from("abcdef"), st.integers(0, 2000), min_size=1), st.data())
    def allocation_is_proportional(sizes, data):
        total = sum(sizes.values())
        n = data.draw(st.integers(0, total))
        counts = allocate(n, sizes)
        assert sum(counts.values()) == n
        for name, size in sizes.items():
            assert counts[name] <= size
            assert total == 0 or abs(counts[name] - n * size / total) < 1

    allocation_is_proportional()
    # the builtin is salted per process
    assert not [path.name for path in SRC.glob("*.py") if "hash(" in path.read_text(encoding="utf-8")]


def test_short_language_is_taken_whole_and_marked(tmp_path):
    """A language with 80 source rows keeps all 80 and is the only one marked undersampled (TS-DATA-2)."""
    harmful = {iso3: harmful_rows(iso3, 80, 40) for iso3 in SIX}
    harmful["eng"] = harmful_rows("eng", 50, 30)
    fake = source_hub(
        tmp_path / "hub",
        harmful=harmful,
        train=fixture_rows("aya_dataset_sample.jsonl"),
        test=benign_rows("eng", 3, "test") + benign_rows("arb", 3, "test"),
        xstest=fixture_rows("xstest_sample.jsonl"),
    )
    settings = settings_for(tmp_path, fake, n_harmful=600)

    manifest = build(
        settings, wrap(fake), tmp_path / "out", dry_run=True, revisions_path=tmp_path / "config" / "revisions.json"
    ).manifest

    english = read(tmp_path / "out" / "prompts" / "harmful" / "en.jsonl")
    assert [row["prompt"] for row in english] == [row["prompt"] for row in harmful["eng"]]
    assert manifest.per_language["en"].harmful == 80 and manifest.per_language["en"].undersampled
    for language in LANGUAGES - {"en"}:
        assert manifest.per_language[language].harmful == 100
        assert not manifest.per_language[language].undersampled
        rows = read(tmp_path / "out" / "prompts" / "harmful" / f"{language}.jsonl")
        assert Counter(row["global_or_local"] for row in rows) == {"global": 67, "local": 33}


class FixtureReader(BaseReader):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def read(self):
        samples = [
            DataSample(
                source_uri="fixture",
                instruction=row["inputs"],
                output=row["targets"],
                metadata={"language": ISO3[row["language_code"]]},
            )
            for row in self.rows
        ]
        return samples, []


def test_pipeline_exports_prompt_completion_rows(tmp_path):
    """The rendered pipeline over a fake reader writes per-language rows and the provenance files (TS-DATA-3)."""
    settings = load_settings(settings_toml(tmp_path))
    rows = fixture_rows("aya_dataset_sample.jsonl")
    short = {**rows[5], "inputs": "kaban lodex?", "targets": "mifim"}
    fed = [*rows, dict(rows[0]), {**rows[1], "targets": " "}, short]
    text = curator.render_yaml(settings, tmp_path / "never-read", "4" * 40)

    config = yaml.safe_load(text)
    assert [reader["type"] for reader in config["readers"]] == ["huggingface"]
    assert config["readers"][0]["source_uri"] == f"{BENIGN}@{'4' * 40}"
    assert [gate["type"] for gate in config["gates"]] == ["schema"]
    assert [step["type"] for step in config["normalizers"]] == ["exact_dedup", "minhash_dedup"]
    assert config["normalizers"][1]["minhash_threshold"] == 0.85
    assert [exporter["type"] for exporter in config["exporters"]] == ["messages"]
    assert config["output_split"] == {"sft_train": 1.0}

    work, out = tmp_path / "work", tmp_path / "out"
    result = curator.run_pipeline(text, work, reader=FixtureReader(fed))
    counts = curator.write_sft(result.passed, out, settings.languages)

    for name in ("manifest.json", "dataset_card.md", "checksums.txt", "rejected.jsonl", "pipeline.yaml"):
        assert (work / name).is_file(), name
    assert len(read(work / "sft_train" / "sft_messages.jsonl")) == 36
    assert [json.loads(line)["rejection_reason"] for line in (work / "rejected.jsonl").read_text().splitlines()] == [
        "missing_field:output"
    ]
    assert counts == {**dict.fromkeys(settings.languages, 5), "fr": 6}
    assert sorted(path.name for path in (out / "sft_train").iterdir()) == sorted(f"{code}.jsonl" for code in LANGUAGES)
    french = [row for row in [*rows, short] if row["language_code"] == "fra"]
    assert read(out / "sft_train" / "fr.jsonl") == [
        {
            "prompt": [{"role": "user", "content": row["inputs"]}],
            "completion": [{"role": "assistant", "content": row["targets"]}],
        }
        for row in french
    ]


def test_xstest_safe_prompts_pass_through_unchanged(tmp_path, caplog):
    """The 250 safe prompts land byte-equal in one English file, no unsafe row with them (TS-DATA-6)."""
    xstest = xstest_rows(250, 200)
    xstest[0]["prompt"] = '  "quoted", comma; ünïcode — 日本語 ?  '
    xstest[1]["prompt"] = "two lines\nin one prompt?"
    fake = source_hub(
        tmp_path / "hub",
        harmful={iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX},
        train=fixture_rows("aya_dataset_sample.jsonl"),
        test=benign_rows("eng", 3, "test") + benign_rows("arb", 3, "test"),
        xstest=xstest[250:] + xstest[:250],
    )

    manifest = build(
        settings_for(tmp_path, fake),
        wrap(fake),
        tmp_path / "out",
        dry_run=True,
        revisions_path=tmp_path / "config" / "revisions.json",
    ).manifest

    folder = tmp_path / "out" / "prompts" / "xstest"
    assert [path.name for path in folder.iterdir()] == ["en.jsonl"]
    rows = read(folder / "en.jsonl")
    safe = [row["prompt"] for row in xstest[:250]]
    assert [row["prompt"].encode() for row in rows] == [prompt.encode() for prompt in safe]
    assert {(row["language"], row["set"]) for row in rows} == {("en", "xstest")}
    assert [row["prompt_id"] for row in rows] == [ids.prompt_id(XSTEST, fake.heads[XSTEST], prompt) for prompt in safe]
    assert not {row["prompt"] for row in xstest[250:]} & set(safe)
    assert "xstest rows: 250 of 250 safe source rows;" in manifest.note
    assert not [record for record in caplog.records if record.name == "driftmap.data.build"]


def test_dry_run_stays_local_then_build_uploads_and_pins(tmp_path):
    """A dry run writes the local directory and uploads nothing; the real run pushes it and pins sources (TS-DATA-8)."""
    fake = fixture_hub(tmp_path / "hub")
    unpinned = {repo: None for repo in (HARMFUL, BENIGN, XSTEST, "CohereLabs/tiny-aya-base")}
    settings = settings_for(tmp_path, fake, revisions={**unpinned, MODEL: fake.heads[MODEL]})
    revisions = tmp_path / "config" / "revisions.json"
    out = tmp_path / "out"

    build(settings, wrap(fake), out, dry_run=True, revisions_path=revisions)

    local = sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file())
    published = [path for path in local if not path.startswith("curatorkit/")]
    expected = ["dataset_card.md", "manifest.json", "prompts/rendered_prefix.json", "prompts/xstest/en.jsonl"]
    expected += [
        f"{folder}/{code}.jsonl"
        for folder in ("prompts/harmful", "prompts/benign_heldout", "sft_train")
        for code in LANGUAGES
    ]
    assert published == sorted(expected)
    assert not {"upload_folder", "upload_file", "create_repo"} & {method for method, _ in fake.calls}
    assert EVAL not in fake.repos
    assert json.loads(revisions.read_text()) == {MODEL: fake.heads[MODEL]}

    build(settings, wrap(fake), out, dry_run=False, revisions_path=revisions)

    tree = fake.repos[EVAL]["main"]
    assert sorted(tree) == sorted(expected)
    for folder in ("prompts/harmful/", "sft_train/", "manifest.json", "prompts/rendered_prefix.json"):
        assert any(path.startswith(folder) for path in tree)
    assert all(tree[path] == (out / path).read_bytes() for path in tree)
    assert [method for method, _ in fake.calls].count("upload_folder") == 1
    assert fake.repo_types[EVAL] == "dataset" and EVAL in fake.private
    assert json.loads(revisions.read_text()) == heads(fake)
    manifest = json.loads(tree["manifest.json"])
    assert {source["repo"]: source["revision"] for source in manifest["sources"]} == heads(fake)


def test_only_the_configured_languages_are_built(tmp_path):
    """Eight source languages give six outputs; Serbian and Filipino are counted in the note (TS-DATA-10)."""
    fake = fixture_hub(tmp_path / "hub")
    out = tmp_path / "out"

    manifest = build(
        settings_for(tmp_path, fake),
        wrap(fake),
        out,
        dry_run=True,
        revisions_path=tmp_path / "config" / "revisions.json",
    ).manifest

    for folder in ("prompts/harmful", "prompts/benign_heldout", "sft_train"):
        assert {path.stem for path in (out / folder).iterdir()} == LANGUAGES
    assert set(manifest.per_language) == LANGUAGES
    assert all(counts.harmful == 2 and counts.benign_heldout == 2 for counts in manifest.per_language.values())
    assert "skipped languages: sr 3, tl 3" in manifest.note
    assert "skipped languages: sr 3, tl 3" in (out / "dataset_card.md").read_text(encoding="utf-8")
    text = "".join(path.read_text(encoding="utf-8") for path in out.rglob("*.jsonl") if "curatorkit" not in path.parts)
    assert '"language": "sr"' not in text and '"language": "tl"' not in text
    assert '"language": "en"' in text
    targets = {row["inputs"]: row["targets"] for row in fixture_rows("aya_dataset_sample.jsonl")}
    french = read(out / "prompts" / "benign_heldout" / "fr.jsonl")
    assert [row["target"] for row in french] == [targets[row["prompt"]] for row in french]
    assert {row["set"] for row in french} == {"benign"}
