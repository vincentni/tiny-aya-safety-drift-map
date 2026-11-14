import io
import json
import os
import urllib.error
from email.message import Message

import pytest

from driftmap.core.hub import Hub
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.jobs.snapshot import TABLES_PATH
from driftmap.publish.space import FILES, SPACE, planned, publish_space
from tests.fakes import FakeClock, FakeHub
from tests.fakes.labels import OWNER, make_settings
from tests.fakes.published import ADAPTER, EVAL, RESULTS, writes

REPO = f"{OWNER}/{SPACE}"
PAGE = f"https://huggingface.co/spaces/{REPO}"
TOKEN = "hf_" + "A1b2" * 9
# the page files of the fixture: neither the build script nor its bytecode goes up
UPLOADED = ["README.md", "app.js", "data.json", "figures/f1.png", "figures/f1.svg", "index.html"]
CARD = f"""---
title: Tiny Aya safety failure map
sdk: static
app_file: index.html
datasets:
  - {EVAL}
  - {RESULTS}
models:
  - CohereLabs/tiny-aya-global
  - {ADAPTER}
---

# Tiny Aya safety failure map
"""


class Pages:
    """Stand-in for ``urllib.request.urlopen``: each call takes the next status, the last one for good."""

    def __init__(self, *statuses):
        self.statuses, self.calls = list(statuses or [200]), []

    def __call__(self, request, timeout=None):
        self.calls.append((request.full_url, dict(request.header_items()), timeout))
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "scripted", Message(), io.BytesIO())
        page = io.BytesIO(b"<html></html>")
        page.status = 200
        return page


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def root(tmp_path):
    """A checkout whose Space folder was built after its tables."""
    top = tmp_path / "checkout"
    space, tables = top / "space", top / TABLES_PATH.parent
    (space / "figures").mkdir(parents=True)
    (space / "__pycache__").mkdir()
    tables.mkdir(parents=True)
    files = {
        "README.md": CARD,
        "index.html": "<html><script src='app.js'></script></html>\n",
        "app.js": "fetch('data.json')\n",
        "build_data.py": "print('build')\n",
        "__pycache__/build_data.cpython-312.pyc": "bytes",
        "data.json": json.dumps({"schema": "driftmap.space/3", "figures": ["f1.png", "f1.svg"]}),
        "figures/f1.png": "png",
        "figures/f1.svg": "<svg></svg>\n",
    }
    for name, text in files.items():
        (space / name).write_text(text, encoding="utf-8")
    for name in ("tables.json", "tables-nopreamble.json"):
        (tables / name).write_text("{}\n", encoding="utf-8")
        os.utime(tables / name, (1_000, 1_000))
    os.utime(space / "data.json", (2_000, 2_000))
    (top / "README.md").write_text(f"[Results Space]({PAGE})\n", encoding="utf-8")
    return top


@pytest.fixture
def fake(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    for repo, repo_type in ((EVAL, "dataset"), (RESULTS, "dataset"), (ADAPTER, "model")):
        hub.add_repo(repo, repo_type=repo_type)
    hub.calls.clear()
    return hub


def run(settings, fake, root, said=None, **options):
    options.setdefault("opener", Pages())
    echo = (said if said is not None else []).append
    hub = Hub(fake, sleep=FakeClock().sleep)
    return publish_space(settings, hub, root, sleep=lambda seconds: None, echo=echo, **options)


def test_the_space_gets_its_page_files_in_one_commit_and_is_read_back_logged_out(settings, fake, root):
    pages, said = Pages(), []

    run(settings, fake, root, said, opener=pages)

    assert fake.repo_types[REPO] == "space" and REPO not in fake.private | fake.gated
    assert sorted(fake.repos[REPO]["main"]) == UPLOADED
    page = {"README.md", "index.html", "app.js", "data.json", "figures/*"}
    assert set(FILES) == page | {"explorer.js", "explorer.json"}
    created = next(kwargs for method, kwargs in fake.calls if method == "create_repo")
    assert (created["repo_type"], created["space_sdk"], created["private"]) == ("space", "static", True)
    assert [method for method, _ in writes(fake)] == ["create_repo", "upload_folder", "update_repo_settings"]
    # the page is asked for once, with no token
    assert [(url, headers) for url, headers, _ in pages.calls] == [(PAGE, {})]
    assert said[-1] == f"public: {PAGE} answers logged out"


def test_a_dry_run_lists_the_files_and_writes_nothing(settings, fake, root):
    pages, said = Pages(), []

    run(settings, fake, root, said, dry_run=True, opener=pages)

    assert writes(fake) == [] and pages.calls == [] and REPO not in fake.repos
    assert said[0] == f"{REPO} (space, static): create; public after the upload"
    assert said[1:-1] == [f"upload: {name}" for name in UPLOADED]
    assert said[-1] == "dry run: nothing was written"


def test_a_plan_names_the_space_and_its_files_while_the_repos_it_names_are_private(settings, fake, root):
    """The plan comes before the repos are opened, so their state is no obstacle to it."""
    fake.private.add(EVAL)
    fake.authorize(EVAL)
    hub = Hub(fake, sleep=FakeClock().sleep)

    lines = planned(settings, hub, root)

    assert lines[0] == f"{REPO} (space, static): create; public after `driftmap publish space`"
    assert lines[1:] == [f"upload: {name}" for name in UPLOADED]
    assert writes(fake) == [] and REPO not in fake.repos


def test_a_plan_says_why_the_space_is_not_ready_and_refuses_nothing(settings, fake, root):
    (root / "space" / "data.json").unlink()
    hub = Hub(fake, sleep=FakeClock().sleep)

    lines = planned(settings, hub, root)

    missing = "space/data.json is missing; run `python space/build_data.py` first"
    assert lines == [f"{REPO} (space, static): not ready: {missing}"]
    (root / "space" / "data.json").write_text(json.dumps({"figures": ["f1.png", "f1.svg"]}), encoding="utf-8")
    (root / "space" / "app.js").write_text(f"// {TOKEN}\n", encoding="utf-8")
    found = planned(settings, hub, root)
    assert len(found) == 1 and "not ready: " in found[0] and "hf_token" in found[0] and TOKEN not in found[0]


def test_a_second_run_updates_the_space_and_drops_a_figure_that_is_gone(settings, fake, root):
    run(settings, fake, root)
    (root / "space" / "figures" / "f1.svg").unlink()
    (root / "space" / "data.json").write_text(json.dumps({"schema": "driftmap.space/3", "figures": ["f1.png"]}))
    said = []

    run(settings, fake, root, said)

    assert "figures/f1.svg" not in fake.repos[REPO]["main"] and "figures/f1.png" in fake.repos[REPO]["main"]
    assert said[0] == f"{REPO} (space, static): update; public"


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ("stale", "space/data.json is older than tables.json; run `python space/build_data.py` first"),
        ("no data", "space/data.json is missing; run `python space/build_data.py` first"),
        ("no figures", "space/figures/ is missing or empty; run `python space/build_data.py` first"),
        ("lost figure", "space/data.json names the figure f1.svg, which space/figures/ does not hold"),
        ("gradio", "space/README.md does not declare a static Space (sdk: static, app_file: index.html)"),
        ("no page", "space/index.html is missing"),
        ("other link", f"README.md does not link to {PAGE}; the Space and the link must name one repo"),
        ("private data", f"{EVAL}, which the Space names, is not public; run `driftmap publish run` first"),
    ],
)
def test_a_space_that_is_not_ready_is_refused_before_any_write(settings, fake, root, change, refusal):
    space = root / "space"
    if change == "stale":
        os.utime(root / TABLES_PATH.parent / "tables.json", (3_000, 3_000))
    elif change == "no data":
        (space / "data.json").unlink()
    elif change == "no figures":
        for path in (space / "figures").iterdir():
            path.unlink()
    elif change == "lost figure":
        (space / "figures" / "f1.svg").unlink()
    elif change == "gradio":
        (space / "README.md").write_text(CARD.replace("sdk: static", "sdk: gradio"), encoding="utf-8")
    elif change == "no page":
        (space / "index.html").unlink()
    elif change == "other link":
        (root / "README.md").write_text("[Results Space](https://huggingface.co/spaces/someone/else)\n")
    elif change == "private data":
        fake.private.add(EVAL)
        fake.authorize(EVAL)

    for dry_run in (True, False):
        with pytest.raises(DriftmapError) as refused:
            run(settings, fake, root, dry_run=dry_run)
        assert refused.value.exit_code == ExitCode.INTEGRITY and str(refused.value) == refusal
    assert writes(fake) == []


def with_explorer(root, **changed):
    """Add the replay explorer to the fixture: the tag that loads its script, the script, and its data file."""
    space = root / "space"
    tags = "<script src='app.js'></script><script src=\"explorer.js\"></script>"
    (space / "index.html").write_text(f"<html>{tags}</html>\n", encoding="utf-8")
    (space / "explorer.js").write_text("fetch('explorer.json')\n", encoding="utf-8")
    write_explorer(space, **changed)
    return space


def answer(verdict, text):
    return {"verdict": verdict, "cut_off": False, "text": text, "clipped": False}


def write_explorer(space, harmful=None, unscored=None, **changed):
    """The explorer file as the build writes it: the text of a harmful or an unscored answer is withheld."""
    shown = f"call {'1234' * 3} or write to someone@example.org"
    rewritten = {"s/a": answer("unscored", unscored)}
    variant = {"code": "hi-latn", "kind": "romanized", "text": "prompt", "answers": rewritten}
    answers = {"s/a": answer("answered", shown), "s/b": answer("refused", "no"), "s/c": answer("harmful", harmful)}
    prompt = {"id": "a", "language": "en", "text": "prompt", "answers": answers, "variants": [variant]}
    replay = {"schema": "driftmap.space.explorer/1", "harmful_text": False, "prompts": [prompt]}
    (space / "explorer.json").write_text(json.dumps(replay | changed) + "\n", encoding="utf-8")
    os.utime(space / "explorer.json", (2_500, 2_500))


def test_the_replay_explorer_goes_up_with_the_page_and_leaves_with_its_data_file(settings, fake, root):
    """Its data quotes prompts and answers, so digits and addresses in it are no finding."""
    space, said = with_explorer(root), []

    run(settings, fake, root, said)

    held = fake.repos[REPO]["main"]
    assert sorted(held) == sorted([*UPLOADED, "explorer.js", "explorer.json"])
    assert held["explorer.json"] == (space / "explorer.json").read_bytes()
    assert "upload: explorer.js" in said and "upload: explorer.json" in said

    (space / "explorer.json").unlink()
    run(settings, fake, root)

    assert sorted(fake.repos[REPO]["main"]) == sorted([*UPLOADED, "explorer.js"])


@pytest.mark.parametrize(
    ("change", "code", "refusal"),
    [
        ("no script", ExitCode.INTEGRITY, "space/index.html refers to explorer.js, which is not among the page files"),
        ("other script", ExitCode.INTEGRITY, "space/index.html refers to extra.js, which is not among the page files"),
        ("no sheet", ExitCode.INTEGRITY, "space/index.html refers to style.css, which is not among the page files"),
        ("other data", ExitCode.INTEGRITY, "space/app.js refers to extra.json, which is not among the page files"),
        ("no folder", ExitCode.INTEGRITY, "space/explorer.js refers to icons/, which is not among the page files"),
        (
            "harmful text",
            ExitCode.INTEGRITY,
            "space/explorer.json holds the text of 1 answer judged harmful or left without a verdict; build it "
            "without --include-harmful",
        ),
        (
            "unscored text",
            ExitCode.INTEGRITY,
            "space/explorer.json holds the text of 2 answers judged harmful or left without a verdict; build it "
            "without --include-harmful",
        ),
        (
            "flag",
            ExitCode.INTEGRITY,
            "space/explorer.json says it was built with the text of harmful answers; build it without "
            "--include-harmful",
        ),
        (
            "stale",
            ExitCode.INTEGRITY,
            "space/explorer.json is older than space/data.json; build both in one run of `python space/build_data.py`, "
            "or delete it",
        ),
        (
            "not json",
            ExitCode.INTEGRITY,
            "space/explorer.json is not a JSON object; run `python space/build_data.py` first",
        ),
        ("token", ExitCode.REDACTION, "space/explorer.json:1: hf_token"),
        ("home path", ExitCode.REDACTION, "space/explorer.json:1: home_path"),
    ],
)
def test_an_explorer_that_is_not_fit_to_show_is_refused_before_any_write(settings, fake, root, change, code, refusal):
    space = with_explorer(root)
    if change == "no script":
        (space / "explorer.js").unlink()
    elif change == "other script":
        (space / "extra.js").write_text("// not a page file\n", encoding="utf-8")
        (space / "index.html").write_text("<html><script src='extra.js'></script></html>\n", encoding="utf-8")
    elif change == "no sheet":
        (space / "index.html").write_text("<html><link rel='stylesheet' href='style.css'></html>\n", encoding="utf-8")
    elif change == "other data":
        (space / "extra.json").write_text("{}\n", encoding="utf-8")
        (space / "app.js").write_text("fetch('data.json'); fetch(\"extra.json\")\n", encoding="utf-8")
    elif change == "no folder":
        (space / "explorer.js").write_text("el('img', { src: 'icons/' + name })\n", encoding="utf-8")
    elif change == "harmful text":
        # the flag is not what is trusted: the file itself is read
        write_explorer(space, harmful="step one")
    elif change == "unscored text":
        write_explorer(space, harmful="step one", unscored="step two")
    elif change == "flag":
        write_explorer(space, harmful_text=True)
    elif change == "stale":
        os.utime(space / "explorer.json", (1_500, 1_500))
    elif change == "not json":
        (space / "explorer.json").write_text("[]\n", encoding="utf-8")
    elif change in ("token", "home path"):
        text = TOKEN if change == "token" else f"/{'home'}/someone/notes"
        (space / "explorer.json").write_text(json.dumps({"harmful_text": False, "prompts": [text]}), encoding="utf-8")

    for dry_run in (True, False):
        with pytest.raises(DriftmapError) as refused:
            run(settings, fake, root, dry_run=dry_run)
        assert refused.value.exit_code == code and str(refused.value) == refusal
    assert writes(fake) == []


def test_an_answer_with_an_unknown_verdict_counts_as_one_whose_text_is_withheld(settings, fake, root):
    space = with_explorer(root)
    replay = json.loads((space / "explorer.json").read_text(encoding="utf-8"))
    replay["prompts"][0]["answers"]["s/d"] = answer("maybe", "text")
    replay["prompts"][0]["answers"]["s/e"] = {"text": "no verdict field", "verdict": None}
    (space / "explorer.json").write_text(json.dumps(replay), encoding="utf-8")
    os.utime(space / "explorer.json", (2_500, 2_500))

    with pytest.raises(DriftmapError) as refused:
        run(settings, fake, root)

    assert str(refused.value).startswith("space/explorer.json holds the text of 2 answers judged harmful or left")
    assert writes(fake) == []


def test_only_what_the_space_itself_serves_has_to_be_a_page_file(settings, fake, root):
    """Other sites and anchors are not the Space's to hold; a folder needs one file under it; the explorer's data
    file may be absent."""
    space = with_explorer(root)
    links = "<a href='https://example.org/code'>code</a><a href=\"#top\">top</a><a href='tel:+10000000'>call</a>"
    scripts = "<script src='//cdn.example.org/lib.js'></script><script src='./app.js?v=2'></script>"
    (space / "index.html").write_text(f"<html>{links}{scripts}<script src='explorer.js'></script></html>\n")
    (space / "app.js").write_text("fetch('data.json'); el('img', { src: 'figures/' + file })\n", encoding="utf-8")
    (space / "explorer.json").unlink()

    run(settings, fake, root)

    assert sorted(fake.repos[REPO]["main"]) == sorted([*UPLOADED, "explorer.js"])


def test_a_plan_lists_the_explorer_files_or_says_why_they_are_not_fit(settings, fake, root):
    space = with_explorer(root)
    hub = Hub(fake, sleep=FakeClock().sleep)

    lines = planned(settings, hub, root)

    assert lines[1:] == [f"upload: {name}" for name in sorted([*UPLOADED, "explorer.js", "explorer.json"])]
    write_explorer(space, harmful="step one")
    refused = planned(settings, hub, root)
    assert len(refused) == 1
    assert "not ready: space/explorer.json holds the text of 1 answer judged harmful" in refused[0]
    assert writes(fake) == []


def test_a_finding_in_a_page_file_stops_the_upload(settings, fake, root):
    (root / "space" / "app.js").write_text(f"const token = '{TOKEN}'\n", encoding="utf-8")

    with pytest.raises(DriftmapError) as refused:
        run(settings, fake, root)

    assert refused.value.exit_code == ExitCode.REDACTION and str(refused.value) == "space/app.js:1: hf_token"
    assert writes(fake) == []


def test_only_the_owner_publishes_the_space(settings, fake, root):
    fake.user = "someone-else"
    with pytest.raises(DriftmapError) as refused:
        run(settings, fake, root)
    assert refused.value.exit_code == ExitCode.READINESS and writes(fake) == []


def test_a_space_that_does_not_answer_logged_out_fails_the_run(settings, fake, root):
    slow = Pages(404, 200)
    run(settings, fake, root, opener=slow)
    assert len(slow.calls) == 2

    closed = Pages(401)
    with pytest.raises(DriftmapError) as failed:
        run(settings, fake, root, opener=closed)
    assert failed.value.exit_code == ExitCode.UNEXPECTED and len(closed.calls) == 3
    assert str(failed.value) == f"{PAGE} does not answer logged out (HTTP 401) after 3 tries; the Space is uploaded"


def test_a_failed_upload_leaves_a_new_space_private(settings, fake, root):
    def down(**kwargs):
        fake._raise(503, f"https://huggingface.co/api/spaces/{REPO}/commit/main")

    fake.upload_folder = down
    pages = Pages()

    with pytest.raises(DriftmapError) as failed:
        run(settings, fake, root, opener=pages)

    assert failed.value.exit_code == ExitCode.PUSH_FAILED
    assert str(failed.value) == f"the upload to {REPO} failed (503); run publish space again"
    assert REPO in fake.private and fake.repos[REPO]["main"] == {} and pages.calls == []
