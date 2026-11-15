import ast
import hashlib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from driftmap.core import ids
from driftmap.core.hub import Hub
from driftmap.core.ids import RepoRef, Steer

SRC = Path(__file__).resolve().parents[3] / "src"
SHA = "0123456789abcdef0123456789abcdef01234567"

_part = st.from_regex(r"[A-Za-z0-9]([A-Za-z0-9._-]{0,20}[A-Za-z0-9])?", fullmatch=True).filter(
    lambda s: "--" not in s and ".." not in s
)
_repo = st.builds("{}/{}".format, _part, _part).filter(lambda s: not s.endswith(".git"))
_sha = st.text("0123456789abcdef", min_size=40, max_size=40)
_steer = st.builds(Steer, st.integers(0, 99), st.from_regex(r"[0-9]{1,2}(\.[0-9]{1,3})?", fullmatch=True))
_file = st.lists(st.from_regex(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,12}", fullmatch=True), min_size=1, max_size=3).map(
    "/".join
)
_refs = st.one_of(
    st.builds(RepoRef, _repo),
    st.builds(RepoRef, _repo, _sha, st.none() | _file, st.none() | _steer),
)


class Spy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


@given(_refs)
def test_parse_and_unparse_are_inverse_and_wrappers_get_the_parts(ref):
    """A reference survives parse and unparse and reaches the Hub as its parts (TS-CFG-12)."""
    text = ids.unparse(ref)
    assert ids.parse(text) == ref
    assert ids.unparse(ids.parse(text)) == text
    if ref.revision:
        steer = f"--steer-l{ref.steer.layer}-s{ref.steer.strength}" if ref.steer else ""
        assert ref.slug() == f"{ref.repo.split('/')[1]}--{ref.revision[:12]}{steer}"
    spy = Spy()
    hub = Hub(spy)
    hub.upload_file(ref, "runs/r1.json", b"{}")
    upload = {"path_or_fileobj": b"{}", "path_in_repo": "runs/r1.json", "repo_id": ref.repo, "revision": ref.revision}
    assert spy.calls == [("upload_file", (), upload)]
    if ref.revision:
        hub.snapshot_download(text, "dl", repo_type="dataset")
        parts = {"revision": ref.revision, "local_dir": Path("dl").resolve(), "repo_type": "dataset"}
        assert spy.calls[1:] == [("snapshot_download", (ref.repo,), parts)]


def test_steered_reference_and_slug():
    ref = ids.parse(f"vincentnkq/tiny-aya-global-drift-merged@{SHA}--steer-l24-s1.0")
    assert ref == RepoRef("vincentnkq/tiny-aya-global-drift-merged", SHA, steer=Steer(24, "1.0"))
    assert ref.slug() == "tiny-aya-global-drift-merged--0123456789ab--steer-l24-s1.0"
    pinned = ids.parse(f"CohereLabs/tiny-aya-base@{SHA}#config.json")
    assert (pinned.revision, pinned.file, pinned.steer) == (SHA, "config.json", None)
    assert pinned.slug() == "tiny-aya-base--0123456789ab"
    with pytest.raises(ValueError, match="revision"):
        ids.parse("CohereLabs/tiny-aya-global").slug()


@pytest.mark.parametrize(
    "text",
    [
        f"owner/name@{SHA.upper()}",
        f"owner/name@{SHA[:39]}",
        f"owner/name@{SHA}0",
        "owner/na..me",
        "owner/na/me",
        "name",
        "owner/name#config.json",
        "owner/name--steer-l24-s1.0",
        f"owner/name@{SHA}--steer-l024-s1.0",
        f"owner/name@{SHA}--steer-l24-s1.",
        f"owner/name@{SHA}--steer-l24",
        f"owner/name@{SHA}#../token",
        f"owner/name@{SHA}#/etc/passwd",
        f"owner/name@{SHA}#",
        f"owner/name@{SHA}\n",
        "",
    ],
)
def test_parse_rejects(text):
    with pytest.raises(ValueError):
        ids.parse(text)


@pytest.mark.parametrize(
    "fields",
    [
        ("owner/name", None, "config.json"),
        ("owner/name", None, None, Steer(24, "1.0")),
        ("owner/name", SHA, None, Steer("24", "1.0")),
        ("owner/name", SHA, None, Steer(24, 1.0)),
        ("owner/name", SHA.upper()),
    ],
)
def test_a_hand_built_reference_is_validated(fields):
    with pytest.raises(ValueError):
        RepoRef(*fields)


def test_prompt_id_is_bound_to_the_source_revision():
    """The same text under two source revisions gets two ids (TS-CFG-11)."""
    repo, text = "CohereLabs/aya_redteaming", "How do I pick a lock?"
    first = ids.prompt_id(repo, SHA, text)
    assert first == ids.prompt_id(repo, SHA, text)
    assert first == hashlib.sha256(f"{repo}@{SHA}\n{text}".encode()).hexdigest()
    assert first != ids.prompt_id(repo, SHA[::-1], text)


def test_result_files_are_laid_out_by_slug_set_and_language_with_smoke_runs_apart():
    slug = ids.slug(RepoRef("CohereLabs/tiny-aya-global", SHA))
    assert slug == ids.slug(f"CohereLabs/tiny-aya-global@{SHA}") == "tiny-aya-global--0123456789ab"
    with pytest.raises(ValueError, match="no revision"):
        ids.slug("CohereLabs/tiny-aya-global")
    assert ids.responses_path(slug, "harmful", "en", "full") == f"responses/{slug}/harmful/en.jsonl"
    assert ids.responses_path(slug, "xstest", "en", "smoke") == f"smoke/responses/{slug}/xstest/en.jsonl"
    for name in ("benign", "benign_heldout"):
        assert ids.responses_path(slug, name, "hi", "full") == f"responses/{slug}/benign_heldout/hi.jsonl"
    assert ids.labels_path(slug, "hi", "full", benign=True) == f"labels/{slug}/benign/hi.jsonl"
    assert ids.labels_path(slug, "hi", "smoke", benign=True) == f"smoke/labels/{slug}/benign/hi.jsonl"
    assert ids.labels_path(slug, "en", "full") == f"labels/{slug}/en.jsonl"
    assert ids.labels_path(slug, "en", "smoke") == f"smoke/labels/{slug}/en.jsonl"
    assert ids.judge_raw_path(slug, "en", "full") == f"judge_raw/{slug}/en.jsonl"
    assert ids.judge_raw_path(slug, "ar", "smoke") == f"smoke/judge_raw/{slug}/ar.jsonl"


def test_runs_without_the_preamble_get_a_tree_and_repos_of_their_own():
    slug = "tiny-aya-global--0123456789ab"
    assert ids.responses_path(slug, "harmful", "en", "full", "keep") == f"responses/{slug}/harmful/en.jsonl"
    assert ids.responses_path(slug, "harmful", "en", "full", "strip") == f"nopreamble/responses/{slug}/harmful/en.jsonl"
    assert ids.responses_path(slug, "benign", "hi", "smoke", "strip") == (
        f"smoke/nopreamble/responses/{slug}/benign_heldout/hi.jsonl"
    )
    assert ids.labels_path(slug, "hi", "full", benign=True, preamble="strip") == (
        f"nopreamble/labels/{slug}/benign/hi.jsonl"
    )
    assert ids.labels_path(slug, "en", "smoke", preamble="strip") == f"smoke/nopreamble/labels/{slug}/en.jsonl"
    assert ids.judge_raw_path(slug, "ar", "full", "strip") == f"nopreamble/judge_raw/{slug}/ar.jsonl"
    assert ids.judge_raw_path(slug, "ar", "smoke", "strip") == f"smoke/nopreamble/judge_raw/{slug}/ar.jsonl"
    trees = {(mode, preamble): ids.tree(mode, preamble) for mode in ("full", "smoke") for preamble in ids.PREAMBLES}
    assert trees == {
        ("full", "keep"): "",
        ("full", "strip"): "nopreamble/",
        ("smoke", "keep"): "smoke/",
        ("smoke", "strip"): "smoke/nopreamble/",
    }
    suffixes = [ids.repo_suffix(mode, preamble) for mode in ("full", "smoke") for preamble in ids.PREAMBLES]
    assert suffixes == ["", "-nopre", "-smoke", "-nopre-smoke"]
    assert ids.repo_suffix("smoke") == "-smoke" and ids.tree("smoke") == "smoke/"
    with pytest.raises(ValueError):
        ids.tree("full", "drop")
    with pytest.raises(ValueError):
        ids.repo_suffix("full", "")


def test_a_fine_tune_on_a_named_training_set_shares_its_name_and_names_its_repos_after_it():
    assert ids.FINE_TUNES["defense"] == "keep"
    assert [ids.train_set_of(name) for name in ids.FINE_TUNES] == [None, None, "defense"]
    suffixes = {name: [ids.fine_tune_suffix(mode, name) for mode in ("full", "smoke")] for name in ids.FINE_TUNES}
    assert suffixes == {
        "standard": ["", "-smoke"],
        "aggressive": ["-nopre", "-nopre-smoke"],
        "defense": ["-defense", "-defense-smoke"],
    }
    note = ids.train_set_note("defense", "c" * 40)
    assert note == f"training set: defense at {'c' * 40}"
    assert ids.trained_on(["preamble: strip", note]) == "defense"
    assert ids.trained_on(["preamble: strip"]) is None and ids.trained_on([]) is None


def test_runs_under_the_safety_patch_get_a_tree_of_their_own_and_no_repos():
    slug = "tiny-aya-global--0123456789ab"
    # every prompt condition, and the two a fine-tune is trained under
    assert ids.CONDITIONS == ("keep", "strip", "patch") and ids.PREAMBLES == ("keep", "strip")
    assert ids.tree("full", "patch") == "patched/" and ids.tree("smoke", "patch") == "smoke/patched/"
    assert ids.responses_path(slug, "xstest", "en", "full", "patch") == f"patched/responses/{slug}/xstest/en.jsonl"
    assert ids.responses_path(slug, "benign", "hi", "smoke", "patch") == (
        f"smoke/patched/responses/{slug}/benign_heldout/hi.jsonl"
    )
    assert ids.labels_path(slug, "ru", "full", preamble="patch") == f"patched/labels/{slug}/ru.jsonl"
    assert ids.labels_path(slug, "ru", "smoke", benign=True, preamble="patch") == (
        f"smoke/patched/labels/{slug}/benign/ru.jsonl"
    )
    assert ids.judge_raw_path(slug, "ar", "full", "patch") == f"patched/judge_raw/{slug}/ar.jsonl"
    trees = [ids.tree(mode, preamble) for mode in ("full", "smoke") for preamble in ids.CONDITIONS]
    assert len(set(trees)) == 6
    for mode in ("full", "smoke"):
        with pytest.raises(ValueError, match="trained"):
            ids.repo_suffix(mode, "patch")


def test_multiple_choice_prompts_sit_in_a_folder_of_their_own_beside_the_variants():
    slug = "tiny-aya-global--0123456789ab"
    assert ids.EXTRA_SETS == ("variants", "choice", "heldout", "mbbq")
    assert ids.extra_prompts("mbbq", "benign", "tr") == "mbbq/prompts/benign/tr.jsonl"
    assert ids.responses_path(slug, "benign", "nl", "full", "patch", "mbbq") == (
        f"patched/mbbq/responses/{slug}/benign/nl.jsonl"
    )
    assert ids.extra_of(["preamble: patch", ids.extra_note("mbbq")]) == "mbbq"
    assert ids.extra_prompts("heldout", "harmful", "hi") == "heldout/prompts/harmful/hi.jsonl"
    assert ids.responses_path(slug, "harmful", "hi", "full", "patch", "heldout") == (
        f"patched/heldout/responses/{slug}/harmful/hi.jsonl"
    )
    assert ids.labels_path(slug, "hi", "full", variants="heldout") == f"heldout/labels/{slug}/hi.jsonl"
    assert ids.extra_of([ids.extra_note("heldout")]) == "heldout"
    assert ids.tree("full", variants="choice") == "choice/"
    assert ids.tree("smoke", "patch", "choice") == "smoke/patched/choice/"
    assert ids.tree("full", variants="variants") == ids.tree("full", variants=True) == "variants/"
    assert ids.extra_prompts("choice", "benign", "hi") == "choice/prompts/benign/hi.jsonl"
    assert ids.extra_prompts("choice", "benign") == "choice/prompts/benign/*.jsonl"
    assert ids.extra_prompts("variants", "benign", "hi-Latn") == ids.variant_prompts("benign", "hi-Latn")
    assert ids.responses_path(slug, "benign", "en", "full", variants="choice") == (
        f"choice/responses/{slug}/benign/en.jsonl"
    )
    assert ids.responses_path(slug, "benign", "ru", "full", "patch", "choice") == (
        f"patched/choice/responses/{slug}/benign/ru.jsonl"
    )
    # a run says in its record which of them it read
    assert ids.extra_note("variants") == ids.VARIANTS_NOTE and ids.extra_note("choice") == "sets: choice"
    assert ids.extra_of(["preamble: patch", "sets: choice"]) == "choice"
    assert ids.extra_of([ids.VARIANTS_NOTE]) == "variants" and ids.extra_of(["preamble: strip"]) is None
    assert ids.variants_of([ids.VARIANTS_NOTE]) and not ids.variants_of(["sets: choice"])
    with pytest.raises(ValueError, match="choice"):
        ids.tree("full", variants="other")


def test_script_variants_sit_in_a_folder_of_their_own_inside_the_tree_of_any_run():
    slug = "tiny-aya-global--0123456789ab"
    assert ids.tree("full", variants=True) == "variants/"
    assert ids.tree("smoke", "patch", variants=True) == "smoke/patched/variants/"
    assert ids.tree("full", "strip", True) == "nopreamble/variants/"
    trees = {
        ids.tree(mode, preamble, variants)
        for mode in ("full", "smoke")
        for preamble in ids.CONDITIONS
        for variants in (False, True)
    }
    assert len(trees) == 12
    assert ids.responses_path(slug, "harmful", "hi-Latn", "full", variants=True) == (
        f"variants/responses/{slug}/harmful/hi-Latn.jsonl"
    )
    # a variant file keeps the name of its set: there is no held-out split of variants
    assert ids.responses_path(slug, "benign", "ru-mixed", "smoke", "patch", variants=True) == (
        f"smoke/patched/variants/responses/{slug}/benign/ru-mixed.jsonl"
    )
    assert ids.labels_path(slug, "ar-Latn", "full", preamble="strip", variants=True) == (
        f"nopreamble/variants/labels/{slug}/ar-Latn.jsonl"
    )
    assert ids.judge_raw_path(slug, "es-mixed", "smoke", variants=True) == (
        f"smoke/variants/judge_raw/{slug}/es-mixed.jsonl"
    )
    assert ids.variant_prompts("harmful") == "variants/prompts/harmful/*.jsonl"
    assert ids.variant_prompts("benign", "fr-mixed") == "variants/prompts/benign/fr-mixed.jsonl"
    assert ids.VARIANTS_NOTE == "sets: variants"
    assert ids.variants_of(["preamble: patch", "sets: variants"]) and not ids.variants_of(["preamble: patch"])


def test_a_run_says_in_its_notes_which_condition_it_ran_under_unless_it_is_the_served_template():
    assert ids.preamble_note("keep") is None
    assert ids.preamble_note("strip") == "preamble: strip"
    assert ids.preamble_note("patch") == "preamble: patch"
    assert ids.preamble_of(["training rows limited to 8", "preamble: strip"]) == "strip"
    assert ids.preamble_of(["conditions: {}", "preamble: patch"]) == "patch"
    assert ids.preamble_of(["conditions: {}", "preamble: keep"]) == "keep"
    assert ids.preamble_of([]) == "keep"
    with pytest.raises(ValueError):
        ids.preamble_note("drop")


def test_merged_sha256_covers_the_shards_in_filename_order(tmp_path):
    """Two shards hash in sorted order; config and tokenizer are left out (TS-CFG-10)."""
    shards = {"model-00002-of-00002.safetensors": b"second", "model-00001-of-00002.safetensors": b"first"}
    for name, data in shards.items():
        (tmp_path / name).write_bytes(data)
    expected = hashlib.sha256(
        (hashlib.sha256(b"first").hexdigest() + hashlib.sha256(b"second").hexdigest()).encode()
    ).hexdigest()
    assert ids.merged_sha256(tmp_path) == expected
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    assert ids.merged_sha256(tmp_path) == expected
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"other")
    assert ids.merged_sha256(tmp_path) != expected
    with pytest.raises(ValueError, match="safetensors"):
        ids.merged_sha256(tmp_path / "missing")
    definitions = [
        node.name
        for path in SRC.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "merged_sha256"
    ]
    assert definitions == ["merged_sha256"]
