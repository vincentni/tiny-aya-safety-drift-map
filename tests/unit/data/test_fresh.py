"""The held-out sample of harmful prompts, drawn from made-up files in the shape of the red-teaming dataset."""

import hashlib
import json

import pytest

from driftmap.core import render
from driftmap.core.settings import load_settings
from driftmap.data import ISO3, build, fresh
from tests.fakes.datasets import HARMFUL, SIX, harmful_rows, tokenizer_files
from tests.fakes.settings import settings_toml

REVISION = "c" * 40
IN_EVALUATION = 4


def put(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


@pytest.fixture
def world(tmp_path):
    """A source of twenty rows per language (fourteen global, six local), an evaluation set that holds four of
    each, the tokenizer and the settings that pin the source."""
    source = tmp_path / "source"
    for iso3 in SIX:
        put(source / f"aya_{iso3}.jsonl", harmful_rows(iso3, 14, 6))
    (source / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    config = settings_toml(tmp_path, revisions={HARMFUL: REVISION})
    found = build._harmful(source, HARMFUL, REVISION)
    for language, rows in found.items():
        put(tmp_path / "prompts" / "harmful" / f"{language}.jsonl", rows[:IN_EVALUATION])
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name, content in tokenizer_files().items():
        (tokenizer / name).write_bytes(content)
    return tmp_path, source, config


def run(world, *extra, out="out"):
    root, source, config = world
    argv = ["--source", str(source), "--prompts", str(root / "prompts"), "--tokenizer", str(root / "tokenizer")]
    return fresh.main([*argv, "--out", str(root / out), "--config", str(config), *extra])


def test_the_sample_is_apart_from_the_evaluation_prompts_and_stratified_as_they_were(world, capsys):
    root, source, _ = world
    # a near copy of an evaluation prompt and a near copy of another fresh row are in the source
    english = read_jsonl(source / "aya_eng.jsonl")
    near = english[0] | {"prompt": english[0]["prompt"] + " x"}
    twin = english[10] | {"prompt": english[10]["prompt"] + " y"}
    put(source / "aya_eng.jsonl", [*english, near, twin])

    assert run(world, "--per-language", "10") == 0

    protected = {
        row["prompt_id"]: row for path in (root / "prompts" / "harmful").glob("*.jsonl") for row in read_jsonl(path)
    }
    drawn = {
        path.stem: read_jsonl(path)
        for path in sorted((root / "out" / "heldout" / "prompts" / "harmful").glob("*.jsonl"))
    }
    assert sorted(drawn) == sorted(ISO3[iso3] for iso3 in SIX)
    for language, rows in drawn.items():
        assert len(rows) == 10 and not {row["prompt_id"] for row in rows} & set(protected)
        assert {(row["set"], row["language"]) for row in rows} == {("harmful", language)}
        assert all(
            set(row) == {"prompt_id", "language", "set", "prompt", "global_or_local", "harm_category"} for row in rows
        )
        assert [row["global_or_local"] for row in rows].count("local") == 4
        assert all(
            row["prompt_id"] == hashlib.sha256(f"{HARMFUL}@{REVISION}\n{row['prompt']}".encode()).hexdigest()
            for row in rows
        )
    texts = [row["prompt"] for row in drawn["en"]]
    assert near["prompt"] not in texts and not {english[10]["prompt"], twin["prompt"]} <= set(texts)
    assert fresh.copies(list(protected.values()), [row for rows in drawn.values() for row in rows]) == 0

    manifest = json.loads((root / "out" / "heldout" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["dataset"] == HARMFUL and manifest["source"]["revision"] == REVISION
    assert manifest["source"]["licence"] == "apache-2.0" and len(manifest["source"]["files_sha256"]) == 6
    assert (manifest["rows"], manifest["rows_per_language"], manifest["dedup_threshold"]) == (60, 10, 0.85)
    assert manifest["evaluation_prompts"]["rows"] == 6 * IN_EVALUATION
    assert (manifest["near_an_evaluation_prompt"], manifest["near_an_earlier_fresh_row"]) == (1, 1)
    assert manifest["per_language"]["en"] == {
        "source_rows": 22,
        "distinct": 22,
        "pass_the_screen": 22,
        "beyond_the_evaluation_set": 18,
        "apart_from_every_other_prompt": 16,
        "drawn": 10,
        "drawn_by_global_or_local": {"global": 6, "local": 4},
    }
    assert manifest["rows_by_harm_category"] == {"Placeholder Category": 60}
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((root / "out" / name).read_bytes()).hexdigest() == digest
    assert "heldout:" in manifest["seed_stream"] and str(root) not in json.dumps(manifest)
    text_sha256 = hashlib.sha256(render.PATCH.encode()).hexdigest()
    assert manifest["safety_patch"] == {"version": render.PATCH_VERSION, "text_sha256": text_sha256}
    assert "en: 10 drawn of 16 apart (18 beyond the evaluation set)" in capsys.readouterr().out


def test_the_same_inputs_give_the_same_files_on_a_seed_stream_of_their_own(world):
    root, _, config = world
    assert run(world, "--per-language", "10") == 0 and run(world, "--per-language", "10", out="again") == 0

    for path in sorted((root / "out").rglob("*.json*")):
        assert path.read_bytes() == (root / "again" / path.relative_to(root / "out")).read_bytes(), path.name
    # the evaluation prompts were drawn on the stream of the language alone
    settings = load_settings(config)
    found = build._harmful(world[1], HARMFUL, REVISION)["en"]
    beyond = found[IN_EVALUATION:]
    own = [row["prompt_id"] for row in read_jsonl(root / "out" / "heldout" / "prompts" / "harmful" / "en.jsonl")]
    plain, _ = fresh.sample(beyond, 10, settings.seed, "en")
    assert own != [row["prompt_id"] for row in plain]


def test_a_source_that_lacks_an_evaluation_prompt_and_a_language_with_too_few_rows_are_refused(world, capsys):
    root, source, _ = world
    assert run(world, "--per-language", "17") == 2
    assert "en has 16 rows apart from the evaluation set, fewer than the 17 asked for" in capsys.readouterr().err
    put(source / "aya_hin.jsonl", read_jsonl(source / "aya_hin.jsonl")[1:])
    assert run(world, "--per-language", "10") == 12
    assert "1 evaluation prompts of hi are not among the source rows" in capsys.readouterr().err
    assert not (root / "out").exists()


def test_the_manifest_names_the_commit_that_holds_the_patch_text_and_its_date(world, capsys):
    root, _, _ = world
    commit, committed = "a" * 40, "2026-11-14T09:43:43Z"

    assert run(world, "--per-language", "10", "--patch-commit", commit, "--patch-committed", committed) == 0

    manifest = json.loads((root / "out" / "heldout" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["safety_patch"] == {
        "version": render.PATCH_VERSION,
        "text_sha256": hashlib.sha256(render.PATCH.encode()).hexdigest(),
        "text_fixed_at_commit": commit,
        "committed": committed,
    }
    assert list(manifest).index("seed_stream") + 1 == list(manifest).index("safety_patch")
    capsys.readouterr()
    for given, message in (
        (["--patch-commit", commit], "--patch-commit and --patch-committed go together"),
        (["--patch-committed", committed], "--patch-commit and --patch-committed go together"),
        (["--patch-commit", "a" * 12, "--patch-committed", committed], "takes the full hash of a commit"),
        (["--patch-commit", commit, "--patch-committed", "2026-11-14"], "takes a date and time with its zone"),
        (["--patch-commit", commit, "--patch-committed", "2026-11-14T09:43:43"], "takes a date and time with its"),
        (["--patch-commit", commit, "--patch-committed", "yesterday"], "takes a date and time with its zone"),
    ):
        assert run(world, "--per-language", "10", *given, out="refused") == 2
        assert message in capsys.readouterr().err
    assert not (root / "refused").exists()


def test_a_row_that_is_or_nears_a_protected_prompt_or_an_earlier_row_counts_as_a_copy():
    protected = [build._row(HARMFUL, REVISION, "en", "harmful", harmful_rows("eng", 1)[0]["prompt"])]
    other = [build._row(HARMFUL, REVISION, "en", "harmful", row["prompt"]) for row in harmful_rows("fra", 3)]
    near = other[0] | {"prompt": other[0]["prompt"] + " z", "prompt_id": "another"}

    assert fresh.copies(protected, other) == 0
    assert fresh.copies(protected, [*other, protected[0]]) == 1
    assert fresh.copies(protected, [*other, near]) == 1
    assert fresh.copies(protected, [protected[0] | {"prompt": protected[0]["prompt"] + " z", "prompt_id": "x"}]) == 1
