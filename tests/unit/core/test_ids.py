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
    pinned = ids.parse(f"ToxicityPrompts/PolyGuard-Qwen@{SHA}#config.json")
    assert (pinned.revision, pinned.file, pinned.steer) == (SHA, "config.json", None)
    assert pinned.slug() == "PolyGuard-Qwen--0123456789ab"
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
