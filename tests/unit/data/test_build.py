import json
import logging
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from driftmap.core.hub import Hub
from driftmap.core.redact import RedactionFinding
from driftmap.core.render import render, token_count
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import load_settings
from driftmap.data.build import build
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
)
from tests.fakes.settings import settings_toml
from tests.fakes.tokenizers import BOS
from tests.fakes.tokenizers import build as build_tokenizer

EVAL = "vincentnkq/aya-redteaming-6lang-eval"
WRITES = {"upload_folder", "upload_file", "create_repo"}


def settings_for(tmp_path, fake, **overrides):
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    overrides.setdefault("revisions", {"CohereLabs/tiny-aya-base": None, **heads(fake)})
    return load_settings(settings_toml(config, **{"n_harmful": 12, "n_heldout": 12, **overrides}))


def pins_file(tmp_path) -> Path:
    return tmp_path / "config" / "revisions.json"


def wrap(fake) -> Hub:
    return Hub(fake, sleep=lambda seconds: None)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sources(tmp_path, **parts):
    defaults = {
        "harmful": {iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX},
        "train": [row for iso3 in SIX for row in benign_rows(iso3, 7)],
        "test": benign_rows("eng", 3, "test") + benign_rows("arb", 3, "test"),
        "xstest": fixture_rows("xstest_sample.jsonl"),
    }
    return source_hub(tmp_path / "hub", **{**defaults, **parts})


def published(out: Path) -> dict[str, bytes]:
    files = [path for path in out.rglob("*") if path.is_file() and "curatorkit" not in path.parts]
    return {path.relative_to(out).as_posix(): path.read_bytes() for path in files}


def test_training_rows_that_repeat_an_evaluation_prompt_are_dropped_and_counted(tmp_path):
    harmful = {iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX}
    harmful["eng"] = harmful_rows("eng", 2)
    # two English test rows with n = 2: both are held out
    test = benign_rows("eng", 2, "test") + benign_rows("arb", 3, "test")
    clean = benign_rows("eng", 7, "clean")
    spare = benign_rows("eng", 1, "spare")[0]
    english = [
        {**test[0], "targets": "synthetic answer of another kind"},
        {**spare, "inputs": test[1]["inputs"].upper()},
        {**spare, "inputs": harmful["eng"][0]["prompt"].upper()},
        {**spare, "inputs": harmful["eng"][1]["prompt"] + " x"},
        *clean,
        {**clean[0], "targets": "synthetic answer said twice"},
    ]
    french = [*benign_rows("fra", 7), {**benign_rows("fra", 1, "spare")[0], "inputs": test[1]["inputs"]}]
    train = english + french + [row for iso3 in SIX if iso3 not in ("eng", "fra") for row in benign_rows(iso3, 7)]
    fake = sources(tmp_path, harmful=harmful, train=train, test=test)
    out = tmp_path / "out"

    manifest = build(
        settings_for(tmp_path, fake), wrap(fake), out, dry_run=True, revisions_path=pins_file(tmp_path)
    ).manifest

    # same prompt, upper-cased held-out, French copy of it, upper-cased harmful, near harmful
    assert manifest.dropped_by_dedup == 5
    assert "training rows repeating an earlier training prompt removed: 1" in manifest.note
    assert {counts.benign_train for counts in manifest.per_language.values()} == {5}
    protected = {
        row["prompt"].casefold()
        for folder in ("harmful", "benign_heldout")
        for row in read(out / "prompts" / folder / "en.jsonl")
    }
    assert len(protected) == 4
    for language in ("en", "fr"):
        trained = [row["prompt"][0]["content"] for row in read(out / "sft_train" / f"{language}.jsonl")]
        assert len(trained) == 5 and not protected & {prompt.casefold() for prompt in trained}
        assert not any(prompt.casefold().startswith("synthetic harmful") for prompt in trained)


def test_rows_the_pipeline_removes_are_counted(tmp_path):
    shared = "synthetic answer " + " ".join(f"kaban{i} lodex{i}" for i in range(300))
    extra = [
        {**benign_rows("eng", 1, "long")[0], "targets": " ".join(["mifim"] * 2100)},
        {**benign_rows("eng", 1, "one")[0], "targets": shared},
        {**benign_rows("eng", 1, "two")[0], "targets": shared},
    ]
    train = extra + [row for iso3 in SIX for row in benign_rows(iso3, 7)]
    fake = sources(tmp_path, train=train)

    manifest = build(
        settings_for(tmp_path, fake), wrap(fake), tmp_path / "out", dry_run=True, revisions_path=pins_file(tmp_path)
    ).manifest

    assert "training rows rejected by the pipeline gate: 1" in manifest.note
    assert "training rows removed by the pipeline dedup: 1" in manifest.note
    assert manifest.dropped_by_dedup == 0
    assert {counts.benign_train for counts in manifest.per_language.values()} == {5}


def test_a_language_left_without_training_rows_is_refused(tmp_path):
    train = [row for iso3 in SIX for row in benign_rows(iso3, 2 if iso3 == "hin" else 7)]
    fake = sources(tmp_path, train=train)
    with pytest.raises(DriftmapError, match="no training rows left for hi") as error:
        build(
            settings_for(tmp_path, fake),
            wrap(fake),
            tmp_path / "out",
            dry_run=False,
            revisions_path=pins_file(tmp_path),
        )
    assert error.value.exit_code == ExitCode.USAGE
    assert not WRITES & {method for method, _ in fake.calls}
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_xstest_is_counted_and_an_empty_set_is_refused(tmp_path, caplog):
    xstest = fixture_rows("xstest_sample.jsonl")
    xstest[1]["prompt"] = xstest[0]["prompt"]
    xstest[2]["prompt"] = f"synthetic safe placeholder {BOS}?"
    fake = sources(tmp_path, xstest=xstest)
    out = tmp_path / "out"

    with caplog.at_level(logging.WARNING, logger="driftmap.data.build"):
        manifest = build(
            settings_for(tmp_path, fake), wrap(fake), out, dry_run=True, revisions_path=pins_file(tmp_path)
        ).manifest

    assert "xstest rows: 2 of 4 safe source rows, expected 250" in manifest.note
    assert len(read(out / "prompts" / "xstest" / "en.jsonl")) == 2
    warned = [record.getMessage() for record in caplog.records if record.name == "driftmap.data.build"]
    assert warned == ["xstest rows: 2 of 4 safe source rows, expected 250"]

    unsafe = [row for row in fixture_rows("xstest_sample.jsonl") if row["label"] == "unsafe"]
    empty = source_hub(
        tmp_path / "empty",
        harmful={iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX},
        train=[row for iso3 in SIX for row in benign_rows(iso3, 7)],
        test=[],
        xstest=unsafe,
    )
    with pytest.raises(DriftmapError, match="no safe XSTest prompt"):
        build(settings_for(tmp_path, empty), wrap(empty), out, dry_run=False, revisions_path=pins_file(tmp_path))
    assert not WRITES & {method for method, _ in empty.calls}


def test_pins_are_on_file_before_anything_is_uploaded(tmp_path, monkeypatch):
    fake = fixture_hub(tmp_path / "hub")
    unpinned = {repo: None for repo in (HARMFUL, BENIGN, XSTEST, "CohereLabs/tiny-aya-base")}
    settings = settings_for(tmp_path, fake, revisions={**unpinned, MODEL: fake.heads[MODEL]})
    revisions = pins_file(tmp_path)
    seen = []
    create = fake.create_repo

    def create_repo(*args, **kwargs):
        seen.append(json.loads(revisions.read_text()))
        return create(*args, **kwargs)

    def upload_folder(**kwargs):
        raise RuntimeError("upload failed")

    replaced = []
    replace = os.replace
    monkeypatch.setattr(
        os, "replace", lambda source, target: replaced.append((source, target)) or replace(source, target)
    )
    monkeypatch.setattr(fake, "create_repo", create_repo)
    monkeypatch.setattr(fake, "upload_folder", upload_folder)

    with pytest.raises(RuntimeError, match="upload failed"):
        build(settings, wrap(fake), tmp_path / "out", dry_run=False, revisions_path=revisions)

    assert seen == [heads(fake)]
    assert [Path(source).parent for source, target in replaced if Path(target) == revisions] == [revisions.parent]
    assert json.loads(revisions.read_text()) == heads(fake)
    assert [path.name for path in revisions.parent.iterdir() if "revisions" in path.name] == ["revisions.json"]


def test_the_pins_file_must_be_named(tmp_path):
    fake = fixture_hub(tmp_path / "hub")
    with pytest.raises(TypeError, match="revisions_path"):
        build(settings_for(tmp_path, fake), wrap(fake), tmp_path / "out", dry_run=True)
    assert fake.calls == []


def test_harmful_sample_follows_the_seed_and_keeps_source_order(tmp_path):
    harmful = {iso3: harmful_rows(iso3, 12, 8) for iso3 in SIX}
    fake = sources(tmp_path, harmful=harmful, train=[row for iso3 in SIX for row in benign_rows(iso3, 20)])

    def sampled(seed: int, name: str) -> dict[str, list[str]]:
        out = tmp_path / name
        build(
            settings_for(tmp_path, fake, seed=seed, n_harmful=60),
            wrap(fake),
            out,
            dry_run=True,
            revisions_path=pins_file(tmp_path),
        )
        files = {path.stem: path.read_text(encoding="utf-8") for path in (out / "prompts" / "harmful").iterdir()}
        return {code: [json.loads(line)["prompt"] for line in text.split("\n") if line] for code, text in files.items()}

    first, again, other = sampled(20261102, "a"), sampled(20261102, "b"), sampled(7, "c")

    assert first == again
    assert all(first[code] != other[code] for code in first)
    source = [row["prompt"] for row in harmful["eng"]]
    for prompts in (first["en"], other["en"]):
        assert len(prompts) == 10 and prompts == [prompt for prompt in source if prompt in prompts]
    trained = {name: (tmp_path / name / "sft_train" / "en.jsonl").read_bytes() for name in "abc"}
    assert trained["a"] == trained["b"] != trained["c"]
    assert trained["a"].count(b"\n") == trained["c"].count(b"\n") == 18


def test_prompts_with_unicode_line_separators_are_read_whole(tmp_path):
    harmful = {iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX}
    harmful["eng"] = harmful_rows("eng", 1, 1)
    harmful["eng"][0]["prompt"] = "synthetic harmful placeholder kaban\u2028lodex\u0085mifim ünïcode"
    fake = sources(tmp_path, harmful=harmful)
    out = tmp_path / "out"

    build(settings_for(tmp_path, fake), wrap(fake), out, dry_run=True, revisions_path=pins_file(tmp_path))

    written = (out / "prompts" / "harmful" / "en.jsonl").read_bytes()
    prompts = [json.loads(line)["prompt"] for line in written.decode().split("\n") if line]
    assert prompts == [row["prompt"] for row in harmful["eng"]]
    assert "ünïcode".encode() in written


def test_unusable_rows_are_counted_in_the_manifest(tmp_path):
    tokenizer = build_tokenizer()
    fixed = token_count(tokenizer, render(tokenizer, "hello").rendered_prompt) - 1
    harmful = {iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX}
    harmful["fra"] = [
        *harmful["fra"],
        {**harmful["fra"][0], "prompt": f"{BOS} synthetic placeholder"},
        {**harmful["fra"][0], "prompt": " ".join(["hello"] * (1701 - fixed))},
        {**harmful["fra"][0], "prompt": " ".join(["hello"] * (1700 - fixed))},
        {**harmful["fra"][0], "prompt": " "},
        dict(harmful["fra"][0]),
    ]
    train = [row for iso3 in SIX for row in benign_rows(iso3, 7)]
    train.append({**benign_rows("fra", 1, "marked")[0], "targets": f"synthetic answer {BOS}"})
    fake = sources(tmp_path, harmful=harmful, train=train)
    out = tmp_path / "out"

    manifest = build(
        settings_for(tmp_path, fake, n_harmful=24),
        wrap(fake),
        out,
        dry_run=True,
        revisions_path=tmp_path / "config" / "revisions.json",
    ).manifest

    assert manifest.dropped_special_token_text == 2
    assert BOS not in (out / "sft_train" / "fr.jsonl").read_text(encoding="utf-8")
    assert manifest.dropped_over_length == 1
    assert manifest.max_rendered_tokens["harmful"] == 1700
    assert manifest.per_language["fr"].harmful == 4 and not manifest.per_language["fr"].undersampled
    assert "repeated harmful prompts removed before sampling: 1" in manifest.note
    assert "blank rows removed: 1" in manifest.note
    prompts = [row["prompt"] for row in read(out / "prompts" / "harmful" / "fr.jsonl")]
    assert BOS not in "".join(prompts) and len(set(prompts)) == 4


def test_a_redaction_finding_stops_the_upload(tmp_path):
    xstest = fixture_rows("xstest_sample.jsonl")
    xstest[0]["prompt"] = "synthetic safe placeholder quoting hf_" + "x" * 34
    fake = sources(tmp_path, xstest=xstest)
    settings = settings_for(tmp_path, fake)
    revisions = tmp_path / "config" / "revisions.json"
    before = revisions.read_text()

    for dry_run in (True, False):
        with pytest.raises(RedactionFinding) as error:
            build(settings, wrap(fake), tmp_path / "out", dry_run=dry_run, revisions_path=revisions)
        assert error.value.exit_code == ExitCode.REDACTION
        assert [(finding.path.name, finding.pattern) for finding in error.value.findings] == [("en.jsonl", "hf_token")]

    assert not WRITES & {method for method, _ in fake.calls}
    assert revisions.read_text() == before


def test_rows_quoting_a_home_path_are_dropped_not_fatal(tmp_path):
    path = "/" + "home/alice/notes"
    train = [row for iso3 in SIX for row in benign_rows(iso3, 7)]
    train.append({**benign_rows("fra", 1, "path")[0], "inputs": f"synthetic question about {path}"})
    train.append({**benign_rows("spa", 1, "path")[0], "targets": f"synthetic answer in {path}"})
    xstest = fixture_rows("xstest_sample.jsonl")
    xstest[0]["prompt"] = f"synthetic safe placeholder reading {path}?"
    fake = sources(tmp_path, train=train, xstest=xstest)
    out = tmp_path / "out"

    manifest = build(
        settings_for(tmp_path, fake),
        wrap(fake),
        out,
        dry_run=False,
        revisions_path=tmp_path / "config" / "revisions.json",
    ).manifest

    assert "rows quoting a home path removed: 3" in manifest.note
    assert len(read(out / "prompts" / "xstest" / "en.jsonl")) == 3
    uploaded = b"".join(fake.repos[EVAL]["main"].values()).decode()
    assert path not in uploaded
    assert "rows quoting a home path removed: 3" in uploaded


def test_a_rebuild_is_identical_and_clears_stale_files(tmp_path):
    fake = fixture_hub(tmp_path / "hub")
    settings = settings_for(tmp_path, fake)
    out = tmp_path / "out"

    build(settings, wrap(fake), out, dry_run=True, revisions_path=tmp_path / "config" / "revisions.json")
    first = published(out)
    (out / "prompts" / "harmful" / "sr.jsonl").write_text("{}\n")
    (out / "sft_train" / "tl.jsonl").write_text("{}\n")
    build(settings, wrap(fake), out, dry_run=True, revisions_path=tmp_path / "config" / "revisions.json")

    assert published(out) == first
    assert published(tmp_path / "elsewhere") == {}
    build(
        settings_for(tmp_path, fake, seed=7),
        wrap(fake),
        tmp_path / "elsewhere",
        dry_run=True,
        revisions_path=tmp_path / "config" / "revisions.json",
    )
    other = published(tmp_path / "elsewhere")
    assert set(other) == set(first)
    assert [name for name in first if name.startswith("prompts/benign_heldout/") and other[name] != first[name]]


def test_sizes_must_divide_over_the_languages(tmp_path):
    fake = fixture_hub(tmp_path / "hub")
    with pytest.raises(DriftmapError, match="n_harmful = 10") as error:
        build(
            settings_for(tmp_path, fake, n_harmful=10),
            wrap(fake),
            tmp_path / "out",
            dry_run=True,
            revisions_path=tmp_path / "config" / "revisions.json",
        )
    assert error.value.exit_code == ExitCode.USAGE
    assert fake.calls == []


def test_a_language_without_source_rows_is_refused(tmp_path):
    harmful = {iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX if iso3 != "hin"}
    fake = sources(tmp_path, harmful=harmful)
    with pytest.raises(DriftmapError, match="no harmful rows for hi"):
        build(
            settings_for(tmp_path, fake),
            wrap(fake),
            tmp_path / "out",
            dry_run=False,
            revisions_path=tmp_path / "config" / "revisions.json",
        )
    assert not WRITES & {method for method, _ in fake.calls}


def test_a_source_without_a_licence_is_never_published(tmp_path):
    fake = source_hub(
        tmp_path / "hub",
        harmful={iso3: harmful_rows(iso3, 2, 1) for iso3 in SIX},
        train=[row for iso3 in SIX for row in benign_rows(iso3, 7)],
        test=[],
        xstest=fixture_rows("xstest_sample.jsonl"),
        licences={XSTEST: None},
    )
    with pytest.raises(ValidationError, match="Paul/XSTest"):
        build(
            settings_for(tmp_path, fake),
            wrap(fake),
            tmp_path / "out",
            dry_run=False,
            revisions_path=tmp_path / "config" / "revisions.json",
        )
    assert not WRITES & {method for method, _ in fake.calls}
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_pins_on_file_are_kept_and_downloads_use_them(tmp_path):
    fake = fixture_hub(tmp_path / "hub")
    pinned = heads(fake)
    settings = settings_for(tmp_path, fake, revisions={"CohereLabs/tiny-aya-base": "2" * 40, **pinned})
    revisions = tmp_path / "config" / "revisions.json"
    # every source moves on after it was pinned
    fake.upload_file(path_or_fileobj=b"{}\n", path_in_repo="aya_eng.jsonl", repo_id=HARMFUL, repo_type="dataset")
    for repo, repo_type in ((BENIGN, "dataset"), (XSTEST, "dataset"), (MODEL, None)):
        fake.upload_file(path_or_fileobj=b"later", path_in_repo="later.txt", repo_id=repo, repo_type=repo_type)
    assert not set(heads(fake).values()) & set(pinned.values())

    manifest = build(settings, wrap(fake), tmp_path / "out", dry_run=False, revisions_path=revisions).manifest

    assert json.loads(revisions.read_text()) == {"CohereLabs/tiny-aya-base": "2" * 40, **pinned}
    assert {ref.repo: ref.revision for ref in manifest.sources} == pinned
    downloads = [call for method, call in fake.calls if method == "snapshot_download"]
    assert {call["repo_id"]: call["revision"] for call in downloads} == pinned
    assert [call["repo_id"] for call in downloads][0] == MODEL
    assert {call["repo_id"]: call["repo_type"] for call in downloads} == {
        MODEL: None,
        HARMFUL: "dataset",
        BENIGN: "dataset",
        XSTEST: "dataset",
    }
    assert manifest.per_language["en"].harmful == 2


def test_a_pin_that_moved_on_file_is_refused_before_any_download(tmp_path):
    fake = fixture_hub(tmp_path / "hub")
    settings = settings_for(tmp_path, fake)
    revisions = tmp_path / "config" / "revisions.json"
    revisions.write_text(json.dumps({**heads(fake), HARMFUL: "9" * 40}))

    with pytest.raises(DriftmapError, match="CohereLabs/aya_redteaming") as error:
        build(settings, wrap(fake), tmp_path / "out", dry_run=False, revisions_path=revisions)

    assert error.value.exit_code == ExitCode.USAGE
    assert fake.calls == []
    assert json.loads(revisions.read_text())[HARMFUL] == "9" * 40


def test_a_real_build_returns_the_commit_its_upload_made(tmp_path):
    fake = sources(tmp_path)
    settings, revisions = settings_for(tmp_path, fake), pins_file(tmp_path)
    dry = build(settings, wrap(fake), tmp_path / "out", dry_run=True, revisions_path=revisions)
    assert (dry.commit, dry.uploaded) == (None, False)

    first = build(settings, wrap(fake), tmp_path / "out", dry_run=False, revisions_path=revisions)
    assert first.uploaded and first.commit == fake.heads[EVAL]

    # another writer commits; the same rows upload nothing
    fake.upload_file(path_or_fileobj=b"{}\n", path_in_repo="labels/x/en.jsonl", repo_id=EVAL, repo_type="dataset")
    again = build(settings, wrap(fake), tmp_path / "out", dry_run=False, revisions_path=revisions)
    assert not again.uploaded and again.commit == fake.heads[EVAL] != first.commit
    assert again.manifest == first.manifest
