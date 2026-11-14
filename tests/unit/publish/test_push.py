import json
import tomllib
from pathlib import Path

import httpx2
import pytest
from huggingface_hub.repocard import RepoCard

from driftmap.core.records import AlphaRow
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import load_settings
from driftmap.publish import cards
from driftmap.publish.push import (
    REBUILD,
    ROTATE,
    Scans,
    check,
    delete_intermediates,
    intermediates,
    plan,
    publish,
    targets,
)
from driftmap.repair import select
from tests.fakes import FakeHub
from tests.fakes.labels import make_settings
from tests.fakes.published import (
    ADAPTER,
    ADAPTER_NOPRE,
    DATASETS,
    EVAL,
    GGUF,
    MERGED,
    MERGED_GGUF,
    MERGED_GGUF_SMOKE,
    MERGED_NOPRE,
    MERGED_SHA256,
    MODELS,
    NOPRE_SHA256,
    NOPRE_SLUG,
    NOPRE_SMOKE,
    REPAIRED,
    REPAIRED_NOPRE,
    REPAIRED_NOPRE_SMOKE,
    REPOS,
    RESPONSES,
    RESULTS,
    SLUG,
    SMOKE,
    WEIGHTS,
    add_merged_gguf,
    add_nopreamble,
    seed,
    two_column_tables,
    writes,
)
from tests.fakes.settings import settings_toml

PUBLISHED = (*DATASETS, *MODELS)
TOKEN = "hf_" + "A1b2" * 9
STALE = ("smoke/", "/superseded/")
REPO_ROOT = Path(__file__).resolve().parents[3]
DIGITS = "1234" * 3


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def fake(tmp_path):
    return FakeHub(tmp_path / "hub")


@pytest.fixture
def said():
    return []


def put(fake, repo, path, data):
    fake._commit(repo, "job upload", {path: data})


def at(fake, repo, back=0):
    """The repo at its head, or ``back`` commits before it, as a finding names it."""
    made = [oid for oid in fake.repos[repo] if oid != "main"]
    return f"{repo}@{made[-1 - back][:12]}"


def renamed(tmp_path, **repos):
    return load_settings(settings_toml(tmp_path, repos=repos))


def calls(fake, method):
    return [kwargs.get("repo_id") or kwargs.get("item_id") for name, kwargs in fake.calls if name == method]


def break_uploads(fake, repo, times=3, status=503):
    real, left = fake.upload_folder, [times]

    def upload_folder(**kwargs):
        if kwargs["repo_id"] == repo and left[0]:
            left[0] -= 1
            fake.calls.append(("upload_folder", kwargs))
            fake._raise(status, f"https://huggingface.co/api/datasets/{repo}/commit/main")
        return real(**kwargs)

    fake.upload_folder = upload_folder


def test_publish_cards_then_opens_then_lists(settings, fake, said):
    hub = seed(fake)

    result = publish(settings, hub, echo=said.append)

    assert result.ok and not (fake.private | fake.gated) & set(PUBLISHED)
    assert {MERGED, *SMOKE} <= fake.private
    assert [line for line in said if line.startswith("public: ")] == [
        f"public: {repo} ({'dataset' if repo in DATASETS else 'model'})" for repo in PUBLISHED
    ]
    for repo in PUBLISHED:
        assert RepoCard(fake.repos[repo]["main"]["README.md"].decode()).data.to_dict()["license"]
    assert "adapter_model.safetensors" in fake.repos[ADAPTER]["main"]


def test_a_token_in_one_file_stops_everything_before_any_write(settings, fake, said):
    """The scan covers every repo first and names file and line, never the value (TS-PUB-1)."""
    hub = seed(fake)
    put(fake, RESULTS, "logs/judge-drifted.log", f"start\nAuthorization: Bearer {TOKEN}\n".encode())
    fake.calls.clear()

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.REDACTION
    assert f"{at(fake, RESULTS)}/logs/judge-drifted.log:2: hf_token" in str(refused.value)
    assert str(refused.value).splitlines()[-1] == ROTATE == "a new commit keeps the value in history; rotate the secret"
    assert TOKEN not in str(refused.value) + "".join(said)
    assert writes(fake) == [] and set(PUBLISHED) <= fake.private


def test_a_notebook_output_cell_is_scanned(settings, fake):
    """A token printed by a notebook cell is found at its line; a clean tree has no finding (TS-PUB-8)."""
    hub = seed(fake)
    assert plan(settings, hub).problems == ()

    cell = {"cell_type": "code", "outputs": [{"output_type": "stream", "text": [f"token={TOKEN}\n"]}], "source": []}
    put(fake, RESULTS, "notebooks/rebuild.ipynb", json.dumps({"cells": [cell], "nbformat": 4}, indent=1).encode())

    problems = plan(settings, hub).problems

    assert [(code, text.rpartition(": ")[2]) for code, text in problems] == [(ExitCode.REDACTION, "hf_token")]
    assert problems[0][1].startswith(f"{at(fake, RESULTS)}/notebooks/rebuild.ipynb:")


def test_rows_may_quote_addresses_and_digits_but_never_a_home_path(settings, fake):
    hub = seed(fake)
    address = "someone" + "@" + "example.org"
    row = f'{{"response": "write to {address}"}}\n'.encode()
    put(fake, RESPONSES, f"responses/{SLUG}/harmful/fr.jsonl", row)
    put(fake, RESPONSES, f"smoke/judge_raw/{SLUG}/fr.jsonl", row)
    put(fake, RESPONSES, f"responses/{REPOS['merged']}--{DIGITS}/harmful/fr.jsonl", row)
    put(fake, EVAL, "sft_train/fr.jsonl", row)
    put(fake, RESULTS, "logs/mail.log", f"INFO driftmap: notify {address}\n".encode())
    put(fake, EVAL, f"labels/{SLUG}/fr.jsonl", f'{{"prompt": "see /{"home"}/alice/notes and {DIGITS}"}}\n'.encode())
    put(fake, RESULTS, "runs/extra.json", f'{{"account": {DIGITS}}}\n'.encode())

    found = [text for _, text in plan(settings, hub).problems]

    assert found == [
        f"{at(fake, EVAL)}/labels/{SLUG}/fr.jsonl:1: home_path",
        f"{at(fake, RESULTS)}/logs/mail.log:1: email (and 1 earlier commit)",
        f"{at(fake, RESULTS)}/runs/extra.json:1: account_id",
    ]
    assert address not in "".join(found)


def test_pickled_training_state_is_read_and_weights_are_not(settings, fake):
    hub = seed(fake)
    put(fake, ADAPTER, "last-checkpoint/training_args.bin", b"\x80\x04\x95hub_token\x94\x8c\x25" + TOKEN.encode())
    put(fake, ADAPTER, "last-checkpoint/adapter_model.safetensors", TOKEN.encode())
    put(fake, ADAPTER, "last-checkpoint/optimizer.pt", b"PK\x03\x04\xff\x00q" + b"@" + b"w.zz\x00")
    fake.calls.clear()

    found = [text for _, text in plan(settings, hub).problems]

    assert found == [f"{at(fake, ADAPTER)}/last-checkpoint/training_args.bin:1: hf_token (and 2 earlier commits)"]
    fetched = [kwargs["filename"] for name, kwargs in fake.calls if name == "hf_hub_download"]
    assert "last-checkpoint/training_args.bin" in fetched and "last-checkpoint/optimizer.pt" in fetched
    assert not any(name.endswith(".safetensors") for name in fetched)
    assert not calls(fake, "snapshot_download")


def test_training_state_leaves_the_adapter_or_refuses_it(settings, fake):
    hub = seed(fake)
    put(fake, ADAPTER, "adapter_model.bin", b"\x00weights")
    assert plan(settings, hub).problems == ()

    publish(settings, hub, echo=lambda line: None)

    assert set(fake.repos[ADAPTER]["main"]) == {
        "README.md",
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "drift_config.json",
    }

    put(fake, ADAPTER, "optimizer.pt", b"PK\x03\x04optimizer")
    put(fake, ADAPTER, "checkpoints/rng_state.pth", b"PK\x03\x04rng")
    refusal = f"{ADAPTER} holds training state: checkpoints/rng_state.pth, optimizer.pt"
    assert [text for _, text in plan(settings, hub).problems] == [refusal]
    assert refusal in check(settings, hub).problems


def test_a_gated_repo_is_opened_and_read_back(settings, fake, said):
    """The responses repo ends un-gated; a Hub that keeps the gate fails the run (TS-PUB-2)."""
    hub = seed(fake)
    fake.gated.add(RESPONSES)

    publish(settings, hub, echo=said.append)

    assert RESPONSES not in fake.gated | fake.private
    opened = [kwargs for name, kwargs in fake.calls if name == "update_repo_settings"]
    assert all(kwargs["gated"] is False and kwargs["private"] is False for kwargs in opened)

    other = FakeHub(fake.root / "second")
    hub = seed(other)
    other.gated.add(RESPONSES)
    other.frozen.add(RESPONSES)
    with pytest.raises(DriftmapError, match=f"{RESPONSES} is still private or gated") as failed:
        publish(settings, hub, echo=said.append)
    assert failed.value.exit_code == ExitCode.UNEXPECTED
    assert not any(method == "add_collection_item" for method, _ in other.calls)


def test_smoke_and_superseded_rows_leave_with_the_card(settings, fake):
    """No public dataset keeps a smoke tree or superseded labels; other rows stay (TS-PUB-3)."""
    hub = seed(fake)
    before = {repo: set(fake.repos[repo]["main"]) for repo in DATASETS}
    assert all(any(mark in path for path in before[repo] for mark in STALE) for repo in DATASETS)

    publish(settings, hub, echo=lambda line: None)

    for repo in DATASETS:
        kept = set(fake.repos[repo]["main"])
        assert kept == {path for path in before[repo] if not any(mark in path for mark in STALE)} | {"README.md"}
    assert f"labels/{SLUG}/en.jsonl" in fake.repos[EVAL]["main"]


def test_a_repo_is_opened_only_after_its_card_landed(settings, fake):
    """Every card upload precedes the first settings change, and those the first listing (TS-PUB-7)."""
    hub = seed(fake)

    publish(settings, hub, echo=lambda line: None)

    order = [name for name, _ in fake.calls if name in ("upload_folder", "update_repo_settings", "add_collection_item")]
    assert order == ["upload_folder"] * 6 + ["update_repo_settings"] * 6 + ["add_collection_item"] * 6
    assert calls(fake, "upload_folder") == calls(fake, "update_repo_settings") == list(PUBLISHED)
    uploads = [kwargs for name, kwargs in fake.calls if name == "upload_folder"]
    assert uploads[3]["repo_id"] == ADAPTER and uploads[3]["delete_patterns"] == [
        "last-checkpoint/*",
        "training_args.bin",
    ]
    assert uploads[4]["delete_patterns"] is None
    superseded = [f"{tree}labels/*/superseded/*" for tree in ("", "nopreamble/", "patched/")]
    assert uploads[0]["delete_patterns"] == ["smoke/*", *superseded]


def test_a_card_upload_that_keeps_failing_leaves_everything_private(settings, fake, said):
    """Three failed attempts end the run with exit 7 and the command to run again (TS-HUB-4)."""
    hub = seed(fake)
    break_uploads(fake, EVAL)

    with pytest.raises(DriftmapError) as failed:
        publish(settings, hub, echo=said.append)

    assert failed.value.exit_code == ExitCode.PUSH_FAILED
    assert str(failed.value).splitlines() == [
        f"the card upload to {EVAL} failed (503); public now: none",
        "resume: run publish(settings, hub) again; finished steps are skipped",
    ]
    assert calls(fake, "upload_folder") == [EVAL] * 3
    assert set(PUBLISHED) <= fake.private and not calls(fake, "update_repo_settings")


def test_a_second_run_skips_the_repos_already_carded(settings, fake, said):
    """After a failure on the third repo, the next run commits only what is left (TS-PUB-6)."""
    hub = seed(fake)
    break_uploads(fake, RESPONSES)
    with pytest.raises(DriftmapError) as failed:
        publish(settings, hub, echo=said.append)
    assert failed.value.exit_code == ExitCode.PUSH_FAILED
    assert set(PUBLISHED) <= fake.private
    heads = dict(fake.heads)
    fake.calls.clear()

    result = publish(settings, hub, echo=said.append)

    assert result.ok and calls(fake, "upload_folder") == [RESPONSES, ADAPTER, REPAIRED, GGUF]
    assert {repo: fake.heads[repo] for repo in (EVAL, RESULTS)} == {repo: heads[repo] for repo in (EVAL, RESULTS)}


def test_thirty_files_go_through_one_call_at_a_time(settings, fake):
    """No two Hub calls overlap and the repos keep their order at every step (TS-HUB-7)."""
    hub = seed(fake)
    for index in range(30):
        put(fake, RESULTS, f"logs/run-{index:02d}.log", f"INFO driftmap: step {index}\n".encode())
    fake.calls.clear()

    assert publish(settings, hub, echo=lambda line: None).ok

    assert fake.max_in_flight == 1
    fetched = calls(fake, "hf_hub_download")
    assert list(dict.fromkeys(fetched)) == list(PUBLISHED)
    # four files from the jobs, thirty logs, the new card, and the card again for the final check
    assert fetched.count(RESULTS) == 36
    assert calls(fake, "add_collection_item") == list(PUBLISHED)


def test_a_public_intermediate_refuses_the_run(settings, fake, said):
    hub = seed(fake)
    fake.private.discard(MERGED)

    with pytest.raises(DriftmapError, match=f"{MERGED} is public") as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert writes(fake) == []

    fake.private.add(MERGED)
    fake.private.discard(SMOKE[0])
    with pytest.raises(DriftmapError, match=f"{SMOKE[0]} is public"):
        publish(settings, hub, echo=said.append)


def test_intermediates_are_deleted_only_by_the_owner_and_checked_gone(settings, fake, said):
    hub = seed(fake)
    fake.user = "someone-else"
    with pytest.raises(DriftmapError, match="someone-else") as refused:
        delete_intermediates(settings, hub, echo=said.append)
    assert refused.value.exit_code == ExitCode.READINESS and writes(fake) == []

    fake.user = "vincentnkq"
    gone = delete_intermediates(settings, hub, echo=said.append)

    assert gone == [MERGED, *SMOKE] and not {MERGED, *SMOKE} & set(fake.repos)
    assert set(PUBLISHED) <= set(fake.repos)
    assert said == [f"deleted: {repo}" for repo in gone]
    assert delete_intermediates(settings, hub, echo=said.append) == []


def test_a_delete_the_hub_ignores_is_reported(settings, fake, monkeypatch):
    hub = seed(fake)
    monkeypatch.setattr(fake, "delete_repo", lambda repo_id, **kwargs: None)

    with pytest.raises(DriftmapError, match=f"{MERGED} still exists after the delete"):
        delete_intermediates(settings, hub, echo=lambda line: None)


def test_publish_deletes_intermediates_only_when_asked(settings, fake, said):
    hub = seed(fake)
    fake.private.discard(MERGED)

    result = publish(settings, hub, delete=True, echo=said.append)

    assert result.ok and not {MERGED, *SMOKE} & set(fake.repos)
    assert calls(fake, "delete_repo") == [MERGED, *SMOKE]

    other = FakeHub(fake.root / "second")
    publish(settings, seed(other), echo=said.append)
    assert not calls(other, "delete_repo") and MERGED in other.repos


def test_a_dry_run_prints_the_plan_and_writes_nothing(settings, fake, said):
    hub = seed(fake, skip=(GGUF,))

    assert publish(settings, hub, dry_run=True, delete=True, echo=said.append) is None

    assert writes(fake) == []
    text = "\n".join(said)
    assert f"{EVAL} (dataset): private -> public; card: write; delete 2 files; collection: add" in text
    unread = "not scanned: adapter_model.safetensors, last-checkpoint/adapter_model.safetensors"
    assert f"{ADAPTER} (model): private -> public; card: write; delete 6 files; {unread}; collection: add" in text
    assert f"absent: {GGUF} (model)" in text
    assert f"would delete: {MERGED} (model)" in text and f"would delete: {SMOKE[1]} (model)" in text


def test_a_dry_run_refuses_where_the_run_would(settings, fake, said):
    hub = seed(fake)
    fake.private.discard(MERGED)

    # what the delete would clear does not stand
    assert publish(settings, hub, dry_run=True, delete=True, echo=said.append) is None
    with pytest.raises(DriftmapError, match=f"{MERGED} is public") as refused:
        publish(settings, hub, dry_run=True, echo=said.append)
    assert refused.value.exit_code == ExitCode.INTEGRITY

    fake.private.add(MERGED)
    put(fake, RESULTS, "logs/judge.log", f"token {TOKEN}\n".encode())
    fake.calls.clear()
    with pytest.raises(DriftmapError, match="hf_token") as refused:
        publish(settings, hub, dry_run=True, delete=True, echo=said.append)
    assert refused.value.exit_code == ExitCode.REDACTION and str(refused.value).splitlines()[-1] == ROTATE
    assert f"would delete: {MERGED} (model)" in said
    assert writes(fake) == [] and MERGED in fake.repos and set(PUBLISHED) <= fake.private


def test_a_commit_landing_after_the_scan_stops_the_run(settings, fake, said):
    hub = seed(fake)
    real = fake.upload_folder

    def upload_folder(**kwargs):
        commit = real(**kwargs)
        if kwargs["repo_id"] == GGUF:
            put(fake, RESULTS, "logs/late.log", b"INFO driftmap: a job still running\n")
        return commit

    fake.upload_folder = upload_folder

    with pytest.raises(DriftmapError, match=f"{RESULTS} changed after its scan") as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert set(PUBLISHED) <= fake.private and not calls(fake, "update_repo_settings")
    uploads = [kwargs for name, kwargs in fake.calls if name == "upload_folder"]
    assert all(len(kwargs["parent_commit"]) == 40 for kwargs in uploads)


def test_a_commit_landing_just_before_the_opening_stops_the_run(settings, fake, said):
    hub = seed(fake)
    real, listed = fake.list_repo_commits, []

    def list_repo_commits(repo_id, **kwargs):
        commits = real(repo_id, **kwargs)
        listed.append(repo_id)
        # the last repo has just been read again after its card
        if listed.count(GGUF) == 2:
            put(fake, RESULTS, "logs/late.log", b"INFO driftmap: a job still running\n")
        return commits

    fake.list_repo_commits = list_repo_commits

    with pytest.raises(DriftmapError, match=f"{RESULTS} changed after its scan") as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert set(PUBLISHED) <= fake.private and not calls(fake, "update_repo_settings")


def test_a_late_commit_is_scanned_before_anything_opens(settings, fake, said):
    hub = seed(fake)
    real = fake.upload_folder

    def upload_folder(**kwargs):
        commit = real(**kwargs)
        if kwargs["repo_id"] == GGUF:
            put(fake, RESULTS, "logs/drift-None.log", f"token {TOKEN}\n".encode())
        return commit

    fake.upload_folder = upload_folder

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.REDACTION
    found = f"{at(fake, RESULTS)}/logs/drift-None.log:1: hf_token"
    assert str(refused.value).splitlines() == [found, REBUILD, ROTATE]
    assert set(PUBLISHED) <= fake.private and not calls(fake, "update_repo_settings")


def test_cache_and_temp_files_refuse_their_repo(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, ".cache/huggingface/download/runs/a.json.lock", b"")
    put(fake, RESULTS, "runs/.drift.json.123.tmp", b"{}\n")

    found = [text for code, text in plan(settings, hub).problems if code == ExitCode.INTEGRITY]

    assert found == [
        f"{at(fake, RESULTS)}/.cache/huggingface/download/runs/a.json.lock: a cache path (and 1 earlier commit)",
        f"{RESULTS} holds cache or temp files: runs/.drift.json.123.tmp",
    ]


def test_a_missing_or_private_collection_refuses_before_any_write(settings, fake):
    hub = seed(fake)
    models = next(slug for slug in fake.collections if "models" in slug)
    fake.collections[models]["private"] = True
    fake.collections.pop(next(slug for slug in fake.collections if "data" in slug))

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=lambda line: None)

    assert "collection tiny-aya-safety-drift-map-data was not found under vincentnkq" in str(refused.value)
    assert f"collection {models} is private" in str(refused.value)
    assert writes(fake) == []


def test_the_check_is_green_only_on_the_published_state(settings, fake, said):
    hub = seed(fake)

    before = check(settings, hub)

    assert not before.ok and writes(fake) == []
    table = before.table()
    assert table.splitlines()[0].split() == [
        "repo", "exists", "public", "gated", "in", "collection", "card", "license"
    ]  # fmt: skip
    assert f"- {EVAL} is private" in table and f"- {ADAPTER} lacks: license: cc-by-nc-4.0 in the card metadata" in table
    assert f"- {EVAL} still holds smoke, superseded or checkpoint files" in table
    assert f"- {ADAPTER} still holds smoke, superseded or checkpoint files" in table

    publish(settings, hub, echo=said.append)
    fake.calls.clear()
    after = check(settings, hub)

    assert after.ok and after.problems == () and writes(fake) == []
    rows = {row.repo: row for row in after.rows}
    assert all(rows[repo][1:] == (True, True, "no", True, True) + (rows[repo].license, True) for repo in PUBLISHED)
    assert rows[EVAL].license == "apache-2.0, cc-by-4.0" and rows[GGUF].license == "cc-by-nc-4.0"
    assert rows[MERGED][1:6] == (True, False, "no", False, False) and rows[MERGED].ok
    assert said[-1] == after.table()


def test_the_check_names_each_gap(settings, fake):
    hub = seed(fake, skip=(REPAIRED, GGUF))
    publish(settings, hub, echo=lambda line: None)
    assert check(settings, hub).ok

    fake.gated.add(RESPONSES)
    fake.private.discard(MERGED)
    data = next(record for record in fake.collections.values() if "data" in record["slug"])
    data["items"] = [item for item in data["items"] if item["id"] != RESULTS]
    data["items"].append({"_id": "x", "id": MERGED, "type": "model", "position": 9})
    card = fake.repos[ADAPTER]["main"]["README.md"]
    put(fake, ADAPTER, "README.md", card.replace(b"license: cc-by-nc-4.0", b"license: mit"))

    found = check(settings, hub)

    assert not found.ok
    assert set(found.problems) == {
        f"{RESPONSES} is gated (auto)",
        f"{RESULTS} is not in the collection",
        f"{ADAPTER} lacks: license: cc-by-nc-4.0 in the card metadata",
        f"{MERGED} is public; it must stay private or be deleted",
        f"{MERGED} is in a collection",
    }
    fake.repos.pop(ADAPTER)
    assert f"{ADAPTER} does not exist" in check(settings, hub).problems


def test_the_space_flag_changes_the_card_and_adds_no_step(tmp_path, fake):
    """With the flag set the run still ends green, collections filled, and the card says so (TS-PUB-11)."""
    flagged = load_settings(settings_toml(tmp_path, publish={"space_not_serving": True}))
    hub = seed(fake)

    assert publish(flagged, hub, echo=lambda line: None).ok

    assert cards.NOT_SERVED in fake.repos[REPAIRED]["main"]["README.md"].decode()
    assert not any(kwargs.get("repo_type") == "space" for _, kwargs in fake.calls)
    assert all(len(record["items"]) == 3 for record in fake.collections.values())


def test_a_value_in_an_earlier_commit_refuses_the_repo(settings, fake, said):
    hub = seed(fake)
    put(fake, RESULTS, "logs/judge.log", f"start\nAuthorization: Bearer {TOKEN}\n".encode())
    put(fake, RESULTS, "logs/judge.log", b"start\nAuthorization: Bearer hf_***\n")
    put(fake, EVAL, "notes.txt", f"/{'home'}/alice/data\n".encode())
    fake._commit(EVAL, "remove notes", {}, ["notes.txt"])
    fake.calls.clear()

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.REDACTION
    assert str(refused.value).splitlines() == [
        f"{at(fake, EVAL, 1)}/notes.txt:1: home_path",
        f"{at(fake, RESULTS, 1)}/logs/judge.log:2: hf_token",
        REBUILD,
        ROTATE,
    ]
    assert said[-1] == ROTATE and writes(fake) == [] and set(PUBLISHED) <= fake.private


def test_each_commit_is_read_once_in_a_run(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, "logs/a.log", b"INFO driftmap: one\n")
    put(fake, RESULTS, "logs/b.log", b"INFO driftmap: two\n")
    fake.calls.clear()
    scans = Scans()

    first = plan(settings, hub, scans=scans)
    fetched = calls(fake, "hf_hub_download")
    fake.calls.clear()
    second = plan(settings, hub, scans=scans)

    assert first.problems == second.problems == ()
    # results holds four files from the jobs and the two logs, over three commits
    assert fetched.count(RESULTS) == 6 and first.fetched == len(fetched) and first.commits == 8
    assert second.fetched == 0 and not calls(fake, "hf_hub_download") and not calls(fake, "list_repo_tree")
    assert calls(fake, "list_repo_commits") == list(PUBLISHED)


def test_a_published_name_with_the_smoke_suffix_is_a_usage_error(tmp_path, fake):
    hub = seed(fake)
    smoke = renamed(tmp_path, adapter=f"{REPOS['adapter']}-smoke")

    for run in (lambda: targets(smoke), lambda: publish(smoke, hub, echo=lambda line: None), lambda: check(smoke, hub)):
        with pytest.raises(DriftmapError, match="repos.adapter .* ends with -smoke") as refused:
            run()
        assert refused.value.exit_code == ExitCode.USAGE
    assert writes(fake) == []


def test_a_smoke_or_merged_item_in_a_collection_is_a_problem(settings, fake):
    hub = seed(fake)
    models = next(record for record in fake.collections.values() if "models" in record["slug"])
    stray = "vincentnkq/some-other-run-SMOKE"
    models["items"] += [
        {"_id": "a", "id": stray, "type": "model", "position": 0},
        {"_id": "b", "id": MERGED.upper(), "type": "model", "position": 1},
    ]

    expected = [f"{stray} is in a collection", f"{MERGED.upper()} is in a collection"]
    assert [text for _, text in plan(settings, hub).problems] == expected
    assert [problem for problem in check(settings, hub).problems if "in a collection" in problem] == expected


@pytest.mark.parametrize("merged", [REPOS["adapter"], REPOS["adapter"].upper(), REPOS["eval"]])
def test_a_name_both_published_and_private_refuses_before_any_call(tmp_path, fake, merged):
    hub = seed(fake)
    collided = renamed(tmp_path, merged=merged)

    for run in (lambda: intermediates(collided), lambda: delete_intermediates(collided, hub, echo=lambda line: None)):
        with pytest.raises(DriftmapError, match="is configured as a published and a private repo") as refused:
            run()
        assert refused.value.exit_code == ExitCode.USAGE
    with pytest.raises(DriftmapError, match="is configured as a published and a private repo"):
        publish(collided, hub, delete=True, echo=lambda line: None)
    assert writes(fake) == []


def test_a_reference_with_a_revision_never_reaches_a_delete(settings, fake):
    hub = seed(fake)
    pinned = settings.repos.model_copy(update={"merged": f"{REPOS['adapter']}@{'a' * 40}"})
    unchecked = settings.model_copy(update={"repos": pinned})

    with pytest.raises(DriftmapError, match="is not a plain repo name") as refused:
        delete_intermediates(unchecked, hub, echo=lambda line: None)

    assert refused.value.exit_code == ExitCode.USAGE
    assert writes(fake) == [] and ADAPTER in fake.repos


def test_two_published_keys_cannot_name_one_repo(tmp_path):
    with pytest.raises(DriftmapError, match="repos.repaired and repos.adapter name one repo") as refused:
        targets(renamed(tmp_path, repaired=REPOS["adapter"].upper()))
    assert refused.value.exit_code == ExitCode.USAGE


def test_nothing_is_deleted_while_another_problem_stands(settings, fake, said):
    hub = seed(fake)
    fake.private.discard(MERGED)
    put(fake, RESULTS, "logs/judge.log", f"token {TOKEN}\n".encode())
    fake.calls.clear()

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, delete=True, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert f"{MERGED} is public" in str(refused.value) and "hf_token" in str(refused.value)
    assert writes(fake) == [] and {MERGED, *SMOKE} <= set(fake.repos)


def test_a_commit_landing_during_the_opening_closes_the_repo_again(settings, fake, said):
    hub = seed(fake)
    real = fake.update_repo_settings

    def update_repo_settings(repo_id, **kwargs):
        if repo_id == RESULTS and kwargs.get("private") is False:
            put(fake, RESULTS, "logs/late.log", b"INFO driftmap: a job still running\n")
        return real(repo_id, **kwargs)

    fake.update_repo_settings = update_repo_settings

    with pytest.raises(DriftmapError, match=f"{RESULTS} changed while it was opened; it is private again") as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert set(PUBLISHED) - fake.private == {EVAL}
    changed = [(kwargs["repo_id"], kwargs["private"]) for name, kwargs in fake.calls if name == "update_repo_settings"]
    assert changed == [(EVAL, False), (RESULTS, False), (RESULTS, True)]


def test_an_absent_required_repo_refuses_before_any_write(settings, fake):
    hub = seed(fake, skip=(ADAPTER, GGUF))

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=lambda line: None)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert str(refused.value) == f"{ADAPTER} does not exist"
    assert writes(fake) == [] and set(PUBLISHED) - {ADAPTER, GGUF} <= fake.private


def test_a_transport_failure_names_what_is_public_and_how_to_resume(settings, fake, said):
    hub = seed(fake)
    real, broken = fake.update_repo_settings, {RESULTS}

    def update_repo_settings(repo_id, **kwargs):
        if repo_id in broken:
            raise httpx2.ConnectError("connection refused")
        return real(repo_id, **kwargs)

    fake.update_repo_settings = update_repo_settings
    link = "https://huggingface.co/blog/vincentnkq/drift-map"

    with pytest.raises(DriftmapError) as failed:
        publish(settings, hub, writeup=link, echo=said.append)

    assert failed.value.exit_code == ExitCode.PUSH_FAILED
    assert str(failed.value).splitlines() == [
        f"opening {RESULTS} failed (ConnectError); public now: {EVAL}",
        f"resume: run publish(settings, hub, writeup='{link}') again; finished steps are skipped",
    ]

    broken.clear()
    broken.add(ADAPTER)
    with pytest.raises(DriftmapError) as failed:
        publish(settings, hub, writeup=link, resume="resume: driftmap publish --writeup URL", echo=said.append)
    assert str(failed.value).splitlines() == [
        f"opening {ADAPTER} failed (ConnectError); public now: {EVAL}, {RESULTS}, {RESPONSES}",
        "resume: driftmap publish --writeup URL",
    ]


def test_a_read_back_that_fails_is_reported_like_a_failed_write(settings, fake, said):
    hub = seed(fake)
    real, state = fake.repo_info, {"armed": False, "left": 3}

    def repo_info(repo_id, **kwargs):
        if state["armed"] and repo_id == EVAL and state["left"]:
            state["left"] -= 1
            fake._raise(503, f"https://huggingface.co/api/datasets/{repo_id}")
        return real(repo_id, **kwargs)

    opening = fake.update_repo_settings

    def update_repo_settings(repo_id, **kwargs):
        state["armed"] = True
        return opening(repo_id, **kwargs)

    fake.repo_info, fake.update_repo_settings = repo_info, update_repo_settings

    with pytest.raises(DriftmapError) as failed:
        publish(settings, hub, echo=said.append)

    assert failed.value.exit_code == ExitCode.PUSH_FAILED
    closed = f"{EVAL} could not be read after it was opened (503); it is private again: run publish again"
    assert str(failed.value) == closed and set(PUBLISHED) <= fake.private


def test_a_public_repo_that_is_gated_is_opened(settings, fake, said):
    hub = seed(fake)
    publish(settings, hub, echo=lambda line: None)
    fake.gated.add(RESULTS)
    fake.calls.clear()

    assert publish(settings, hub, echo=said.append).ok

    assert calls(fake, "update_repo_settings") == [RESULTS] and RESULTS not in fake.gated
    assert f"already public: {EVAL} (dataset)" in said and f"public: {RESULTS} (dataset)" in said


def test_the_new_card_is_scanned_too(settings, fake):
    hub = seed(fake)
    link = "https://example.org/" + "someone" + "@" + "example.org"

    problems = plan(settings, hub, writeup=link).problems

    assert len(problems) == 6 and {code for code, _ in problems} == {ExitCode.REDACTION}
    assert problems[0][1].startswith(f"{EVAL}/README.md:") and problems[0][1].endswith(": email in the new card")


def test_a_repo_whose_card_would_be_incomplete_is_refused(settings, fake):
    hub = seed(fake)
    fake._commit(EVAL, "remove the manifest", {}, ["manifest.json"])

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=lambda line: None)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert str(refused.value) == f"{EVAL} cannot get a complete card: the source licences (no manifest.json)"
    assert writes(fake) == []


def test_a_run_that_ends_short_of_the_checklist_fails(settings, fake, said, monkeypatch):
    hub = seed(fake)
    monkeypatch.setattr(fake, "add_collection_item", lambda *args, **kwargs: None)

    with pytest.raises(DriftmapError, match="published, but the checklist is not met") as failed:
        publish(settings, hub, echo=said.append)

    assert failed.value.exit_code == ExitCode.UNEXPECTED
    assert f"- {EVAL} is not in the collection" in str(failed.value)


def test_an_intermediate_of_unknown_visibility_counts_as_public(settings, fake):
    hub = seed(fake)
    real = fake.repo_info

    def repo_info(repo_id, **kwargs):
        info = real(repo_id, **kwargs)
        if repo_id == MERGED:
            info.private = None
        return info

    fake.repo_info = repo_info

    assert [text.partition(";")[0] for _, text in plan(settings, hub).problems] == [f"{MERGED} is public"]
    assert f"{MERGED} is public; it must stay private or be deleted" in check(settings, hub).problems


def test_files_the_scan_cannot_read_and_file_names_are_covered(settings, fake):
    hub = seed(fake)
    address = "someone" + "@" + "example.org"
    put(fake, RESULTS, "logs/old.log.gz", b"\x1f\x8b\x08\x00binary")
    put(fake, RESULTS, "logs/wide.log", "INFO driftmap: step".encode("utf-16"))
    put(fake, RESPONSES, "responses/all.parquet", b"PAR1\x00\x15\x04")
    put(fake, RESULTS, f"logs/{address}.log", b"INFO driftmap: done\n")
    put(fake, RESULTS, "logs/.cache/kept.log", f"token {TOKEN}\n".encode())

    found = [text for _, text in plan(settings, hub).problems]

    unread = "not text, so the scan cannot read it"
    assert found == [
        f"{at(fake, RESULTS)}/logs/.cache/kept.log:1: hf_token",
        f"{at(fake, RESULTS)}/logs/***: email in the file name (and 1 earlier commit)",
        f"{at(fake, RESULTS)}/logs/.cache/kept.log: a cache path",
        f"{at(fake, RESULTS)}/logs/old.log.gz: {unread} (and 3 earlier commits)",
        f"{at(fake, RESULTS)}/logs/wide.log: {unread} (and 2 earlier commits)",
        f"{at(fake, RESPONSES)}/responses/all.parquet: {unread}",
    ]
    assert address not in "".join(found)


def test_a_long_history_costs_one_fetch_per_distinct_file(settings, fake, said):
    hub = seed(fake)
    for index in range(100):
        put(fake, EVAL, f"prompts/harmful/part-{index:03d}.jsonl", f'{{"prompt_id": "p-{index}"}}\n'.encode())
    put(fake, EVAL, "prompts/harmful/part-000.jsonl", b'{"prompt_id": "p-0, again"}\n')
    fake.calls.clear()

    drawn = plan(settings, hub)

    fetched = calls(fake, "hf_hub_download")
    # five distinct files at the seed, a hundred added, one rewritten: 102 commits, 106 files
    assert fetched.count(EVAL) == 106 and drawn.fetched == len(fetched) < 130
    assert calls(fake, "list_repo_tree").count(EVAL) == 102 and not calls(fake, "snapshot_download")
    publish(settings, hub, dry_run=True, echo=said.append)
    assert said[0] == f"history scan: {drawn.fetched} file fetches over {drawn.commits} commits"


def test_a_scan_above_the_ceiling_stops_before_the_first_fetch(tmp_path, fake, said):
    hub = seed(fake)
    capped = load_settings(settings_toml(tmp_path, publish={"max_file_fetches": 5}))

    for run in (lambda: plan(capped, hub), lambda: publish(capped, hub, dry_run=True, echo=said.append)):
        with pytest.raises(DriftmapError, match="above publish.max_file_fetches = 5; nothing was fetched") as stopped:
            run()
        assert stopped.value.exit_code == ExitCode.GUARD_REFUSAL
        needs = f"{EVAL} 5, {RESULTS} 4, {RESPONSES} 3, {ADAPTER} 7, {REPAIRED} 1, {GGUF} 1"
        assert f"the history scan needs 21 file fetches ({needs}), above" in str(stopped.value)
    assert not calls(fake, "hf_hub_download") and writes(fake) == []
    assert load_settings(settings_toml(tmp_path)).publish.max_file_fetches == 2000


def test_a_publish_reuses_what_its_plan_read(settings, fake):
    hub = seed(fake)
    scans = Scans()
    assert plan(settings, hub, scans=scans).fetched == 21
    fake.calls.clear()
    said = []

    assert publish(settings, hub, scans=scans, echo=said.append).ok

    # the six new cards, then what the final check reads
    fetched = [kwargs["filename"] for name, kwargs in fake.calls if name == "hf_hub_download"]
    assert sorted(fetched) == ["README.md"] * 12 + ["manifest.json"]
    assert said[0] == "history scan: 0 file fetches over 6 commits"


@pytest.mark.parametrize("method", ["list_repo_commits", "list_repo_tree", "hf_hub_download"])
def test_a_throttled_scan_stops_at_once_and_names_the_repo(settings, fake, method):
    hub = seed(fake)
    real, tried = getattr(fake, method), []

    def throttled(repo_id, *args, **kwargs):
        if repo_id == RESULTS:
            tried.append(repo_id)
            fake._raise(429, f"https://huggingface.co/api/datasets/{repo_id}", {"ratelimit": '"api";r=0;t=5'})
        return real(repo_id, *args, **kwargs)

    setattr(fake, method, throttled)

    for run in (lambda: plan(settings, hub), lambda: publish(settings, hub, echo=lambda line: None)):
        tried.clear()
        with pytest.raises(DriftmapError) as stopped:
            run()
        assert stopped.value.exit_code == ExitCode.PUSH_FAILED
        assert str(stopped.value) == f"reading {RESULTS} failed (429: the Hub is throttling, wait before the next run)"
        assert tried == [RESULTS]
    assert writes(fake) == []


def reads(fake):
    return [name for name, _ in fake.calls if name in ("list_repo_commits", "list_repo_tree", "hf_hub_download")]


def test_a_scan_file_spares_the_next_run_what_this_one_read(tmp_path, settings, fake):
    """Listings, verdicts and card inputs kept in a file are not asked of the Hub again; findings still stand."""
    hub = seed(fake)
    put(fake, RESULTS, "logs/old.log", f"token {TOKEN}\n".encode())
    path = tmp_path / "state" / "publish-scan.json"

    first = plan(settings, hub, scans=Scans(path))
    assert first.fetched and path.is_file() and TOKEN not in path.read_text(encoding="utf-8")

    fake.calls.clear()
    again = plan(settings, hub, scans=Scans(path))
    # each repo's history is still listed: a new commit must be seen
    assert set(reads(fake)) == {"list_repo_commits"} and again.fetched == 0
    assert again.problems == first.problems and [text for _, text in again.problems] == [
        f"{at(fake, RESULTS)}/logs/old.log:1: hf_token"
    ]
    assert [step.card for step in again.steps] == [step.card for step in first.steps]

    put(fake, RESULTS, "logs/new.log", b"INFO driftmap: step 60 of 318\n")
    fake.calls.clear()
    assert plan(settings, hub, scans=Scans(path)).fetched == 1
    assert reads(fake).count("list_repo_tree") == 1


def test_a_throttled_scan_keeps_what_it_read_for_the_next_run(tmp_path, settings, fake):
    hub = seed(fake)
    path = tmp_path / "publish-scan.json"
    real, served = fake.hf_hub_download, []

    def throttled(repo_id, *args, **kwargs):
        if len(served) == 5:
            fake._raise(429, f"https://huggingface.co/api/datasets/{repo_id}", {"ratelimit": '"api";r=0;t=5'})
        served.append(repo_id)
        return real(repo_id, *args, **kwargs)

    fake.hf_hub_download = throttled
    with pytest.raises(DriftmapError, match="the Hub is throttling"):
        plan(settings, hub, scans=Scans(path))
    fake.hf_hub_download = real

    fake.calls.clear()
    whole = plan(settings, hub, scans=Scans(path))

    assert "list_repo_tree" not in reads(fake) and not whole.problems
    assert whole.fetched == plan(settings, hub).fetched - 5


def test_a_scan_file_of_other_rules_or_a_damaged_one_is_not_read(tmp_path, settings, fake):
    """A verdict holds only under the rules that gave it: such a file is read as empty and written anew."""
    hub = seed(fake)
    path = tmp_path / "publish-scan.json"
    fresh = plan(settings, hub, scans=Scans(path))
    kept = json.loads(path.read_text(encoding="utf-8"))

    for damaged in (json.dumps(kept | {"rules": "0" * 64}), "{", json.dumps([1]), json.dumps({"rules": kept["rules"]})):
        path.write_text(damaged, encoding="utf-8")
        assert plan(settings, hub, scans=Scans(path)).fetched == fresh.fetched
        assert json.loads(path.read_text(encoding="utf-8")) == kept


def test_a_listing_that_keeps_failing_is_never_a_clean_scan(settings, fake):
    hub = seed(fake)
    tried = []

    def list_repo_tree(repo_id, *args, **kwargs):
        tried.append(repo_id)
        raise httpx2.ConnectError("connection refused")

    fake.list_repo_tree = list_repo_tree

    with pytest.raises(DriftmapError) as stopped:
        publish(settings, hub, echo=lambda line: None)

    assert stopped.value.exit_code == ExitCode.PUSH_FAILED
    assert str(stopped.value) == f"reading {EVAL} failed (ConnectError)"
    assert tried == [EVAL] * 3 and writes(fake) == []


def test_one_damaged_byte_does_not_switch_the_address_rule_off(settings, fake):
    hub = seed(fake)
    address = ("someone" + "@" + "example.org").encode()
    put(fake, RESULTS, "logs/damaged.log", b"INFO driftmap: \xff notify " + address + b"\n")
    put(fake, RESULTS, "logs/bytes.log", b"\x00\x01 " + address + b"\n")

    found = [text for _, text in plan(settings, hub).problems]

    unread = "not text, so the scan cannot read it"
    assert found == [
        f"{at(fake, RESULTS)}/logs/damaged.log:1: email (and 1 earlier commit)",
        f"{at(fake, RESULTS)}/logs/bytes.log: {unread}",
        f"{at(fake, RESULTS)}/logs/damaged.log: {unread} (and 1 earlier commit)",
    ]


def test_a_cache_path_or_an_unreadable_file_in_an_earlier_commit_refuses(settings, fake, said):
    hub = seed(fake)
    put(fake, RESULTS, ".cache/huggingface/download/x.log", f"token {TOKEN}\n".encode())
    put(fake, RESULTS, "logs/old.log.gz", b"\x1f\x8b\x08\x00binary")
    put(fake, RESPONSES, "smoke/responses/dump.bin", b"\x00\x01\x02")
    fake._commit(RESULTS, "tidy", {}, [".cache/huggingface/download/x.log", "logs/old.log.gz"])
    fake.calls.clear()

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=said.append)

    unread = "not text, so the scan cannot read it"
    assert refused.value.exit_code == ExitCode.REDACTION
    assert str(refused.value).splitlines() == [
        f"{at(fake, RESULTS, 1)}/.cache/huggingface/download/x.log:1: hf_token (and 1 earlier commit)",
        f"{at(fake, RESULTS, 1)}/.cache/huggingface/download/x.log: a cache path (and 1 earlier commit)",
        f"{at(fake, RESULTS, 1)}/logs/old.log.gz: {unread}",
        f"{at(fake, RESPONSES)}/smoke/responses/dump.bin: {unread}",
        REBUILD,
        ROTATE,
    ]
    assert writes(fake) == []


def test_only_a_token_finding_asks_for_a_rotation(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, "logs/paths.log", f"reading /{'home'}/alice/data\n".encode())

    drawn = plan(settings, hub)

    assert drawn.advice == (REBUILD,) and drawn.refusal().splitlines()[-1] == REBUILD
    assert "publish with --fresh" in REBUILD and "private as <name>-work" in REBUILD
    put(fake, RESULTS, f"logs/{TOKEN}.log", b"INFO driftmap: done\n")
    again = plan(settings, hub)
    assert again.advice == (REBUILD, ROTATE)
    assert f"{at(fake, RESULTS)}/logs/***.log: hf_token in the file name" in again.refusal()
    assert plan(settings, seed(FakeHub(fake.root / "clean"))).advice == ()


def test_a_dataset_is_read_whole_whatever_its_files_are_called(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, "weights/model.safetensors", f"token {TOKEN}\n".encode())

    found = [text for _, text in plan(settings, hub).problems]

    assert found == [f"{at(fake, RESULTS)}/weights/model.safetensors:1: hf_token"]


def test_twelve_digits_in_a_file_name_outside_the_row_folders_are_found(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, f"logs/{DIGITS}.log", b"INFO driftmap: done\n")
    put(fake, RESPONSES, f"responses/{REPOS['merged']}--{DIGITS}/harmful/fr.jsonl", b'{"response": "ok"}\n')

    found = [text for _, text in plan(settings, hub).problems]

    assert found == [f"{at(fake, RESULTS)}/logs/***.log: account_id in the file name"]


def test_an_unreadable_file_arriving_after_the_scan_stops_the_run(settings, fake, said):
    hub = seed(fake)
    real = fake.upload_folder

    def upload_folder(**kwargs):
        commit = real(**kwargs)
        if kwargs["repo_id"] == GGUF:
            put(fake, RESULTS, "logs/drift-None.log", b"\x1f\x8b\x08\x00binary")
        return commit

    fake.upload_folder = upload_folder

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    unread = f"{at(fake, RESULTS)}/logs/drift-None.log: not text, so the scan cannot read it"
    assert str(refused.value).splitlines() == [unread, REBUILD]
    assert set(PUBLISHED) <= fake.private and not calls(fake, "update_repo_settings")


def test_a_repo_that_cannot_be_closed_again_is_named_as_public(settings, fake, said):
    hub = seed(fake)
    real, reading, state = fake.update_repo_settings, fake.repo_info, {"opened": False}

    def update_repo_settings(repo_id, **kwargs):
        if kwargs.get("private") is True:
            raise httpx2.ConnectError("connection refused")
        state["opened"] = True
        return real(repo_id, **kwargs)

    def repo_info(repo_id, **kwargs):
        if state["opened"] and repo_id == EVAL:
            state["opened"] = False
            put(fake, EVAL, "notes.txt", b"a job still running\n")
        return reading(repo_id, **kwargs)

    fake.update_repo_settings, fake.repo_info = update_repo_settings, repo_info

    with pytest.raises(DriftmapError) as failed:
        publish(settings, hub, echo=said.append)

    assert failed.value.exit_code == ExitCode.PUSH_FAILED
    left = "closing it failed (ConnectError): it is PUBLIC, make it private by hand"
    assert str(failed.value).splitlines() == [
        f"{EVAL} changed while it was opened, and {left}",
        "resume: run publish(settings, hub) again; finished steps are skipped",
    ]
    assert set(PUBLISHED) - fake.private == {EVAL}


def test_a_commit_after_the_opening_closes_the_repo_before_the_final_check(settings, fake, said):
    hub = seed(fake)
    real = fake.add_collection_item

    def add_collection_item(slug, item_id, item_type, **kwargs):
        if item_id == GGUF:
            put(fake, RESPONSES, f"responses/{SLUG}/harmful/es.jsonl", b'{"response": "late"}\n')
        return real(slug, item_id, item_type, **kwargs)

    fake.add_collection_item = add_collection_item

    with pytest.raises(DriftmapError, match=f"{RESPONSES} changed after it was opened; it is private again") as refused:
        publish(settings, hub, echo=said.append)

    assert refused.value.exit_code == ExitCode.INTEGRITY
    assert set(PUBLISHED) & fake.private == {RESPONSES}
    assert not any(line.startswith("repo ") for line in said)


def test_the_trees_with_no_safety_preamble_are_scanned_as_rows_and_published(settings, fake):
    """Rows under nopreamble/ may quote what the main rows may; they stay public, every smoke tree leaves."""
    hub = seed(fake)
    add_nopreamble(fake)
    address = "someone" + "@" + "example.org"
    row = f'{{"response": "write to {address}, reference {DIGITS}"}}\n'.encode()
    put(fake, RESPONSES, f"nopreamble/responses/{NOPRE_SLUG}/harmful/fr.jsonl", row)
    put(fake, RESPONSES, f"smoke/nopreamble/judge_raw/{NOPRE_SLUG}/fr.jsonl", row)
    put(fake, RESPONSES, f"nopreamble/responses/{REPOS['merged']}--{DIGITS}/harmful/fr.jsonl", row)
    put(fake, EVAL, f"nopreamble/labels/{NOPRE_SLUG}/fr.jsonl", row)
    before = {repo: set(fake.repos[repo]["main"]) for repo in DATASETS}

    assert plan(settings, hub).problems == ()
    assert publish(settings, hub, echo=lambda line: None).ok

    for repo in DATASETS:
        kept = set(fake.repos[repo]["main"])
        assert kept == {path for path in before[repo] if not any(mark in path for mark in STALE)} | {"README.md"}
        assert any(path.startswith("nopreamble/") for path in kept)
    assert f"nopreamble/labels/{NOPRE_SLUG}/fr.jsonl" in fake.repos[EVAL]["main"]
    assert "nopreamble/tables/tables-nopreamble.md" in fake.repos[RESULTS]["main"]

    # a home path is refused in a row of either tree
    quoted = f'{{"prompt": "see /{"home"}/alice/notes"}}\n'.encode()
    put(fake, EVAL, f"nopreamble/labels/{NOPRE_SLUG}/es.jsonl", quoted)
    found = [text for _, text in plan(settings, hub).problems]
    assert found == [f"{at(fake, EVAL)}/nopreamble/labels/{NOPRE_SLUG}/es.jsonl:1: home_path"]


def test_the_repair_screen_is_scanned_as_rows_and_published(settings, fake):
    """Screen rows and the labels and judge answers beside them may quote what responses may; a smoke screen leaves."""
    hub = seed(fake)
    address = "someone" + "@" + "example.org"
    row = f'{{"response": "write to {address}, reference {DIGITS}"}}\n'.encode()
    screen = f"screen/{SLUG}"
    for name in ("alpha-0.25.jsonl", "labels-alpha-0.25.jsonl", "judge_raw-alpha-0.25.jsonl"):
        put(fake, RESPONSES, f"{screen}/{name}", row)
        put(fake, RESPONSES, f"smoke/{screen}/{name}", row)
    put(fake, RESPONSES, f"screen/{REPOS['merged']}--{DIGITS}/alpha-0.1.jsonl", row)

    assert plan(settings, hub).problems == ()
    assert publish(settings, hub, echo=lambda line: None).ok

    kept = set(fake.repos[RESPONSES]["main"])
    assert f"{screen}/labels-alpha-0.25.jsonl" in kept and not any(path.startswith("smoke/") for path in kept)

    quoted = f'{{"response": "see /{"home"}/alice/notes"}}\n'.encode()
    put(fake, RESPONSES, f"{screen}/alpha-0.4.jsonl", quoted)
    found = [text for _, text in plan(settings, hub).problems]
    assert found == [f"{at(fake, RESPONSES)}/{screen}/alpha-0.4.jsonl:1: home_path"]


def test_the_second_table_on_the_hub_is_read_for_the_cards(tmp_path, settings, fake):
    hub = seed(fake)
    add_nopreamble(fake, tables=two_column_tables(tmp_path, settings))

    assert publish(settings, hub, echo=lambda line: None).ok

    card = fake.repos[RESULTS]["main"]["README.md"].decode()
    assert "State of each planned column with no safety preamble:" in card
    assert "the runs under `nopreamble/` left it out" in card

    put(fake, RESULTS, "nopreamble/tables/tables-nopreamble.json", b"{}\n")
    torn = [text for _, text in plan(settings, hub).problems]
    assert torn == ["nopreamble/tables/tables-nopreamble.json on the Hub does not parse; build and upload it again"]


def test_the_aggressive_tables_on_the_hub_are_read_for_the_second_adapter(tmp_path, settings, fake):
    """The standard files feed the first adapter's card, the -aggressive files the second's."""
    hub = seed(fake, tables=two_column_tables(tmp_path, settings, unsafe=1))
    add_nopreamble(fake)
    own = f"{MERGED_NOPRE}@{'b' * 40}"
    safer = two_column_tables(tmp_path, settings, unsafe=1, fine_tuned=own)
    worse = two_column_tables(tmp_path, settings, unsafe=9, preamble="strip", fine_tuned=own)
    put(fake, RESULTS, "tables/tables-aggressive.json", safer.model_dump_json(by_alias=True).encode())

    assert publish(settings, hub, echo=lambda line: None).ok
    card = fake.repos[ADAPTER_NOPRE]["main"]["README.md"].decode()
    assert "Not certified for deployment." in card and "### Served template" in card and f"`{own}`" in card
    assert "### Served template" in fake.repos[ADAPTER]["main"]["README.md"].decode()

    stripped = "nopreamble/tables/tables-nopreamble-aggressive.json"
    put(fake, RESULTS, stripped, worse.model_dump_json(by_alias=True).encode())
    card = next(step.card for step in plan(settings, hub).steps if step.target.repo == ADAPTER_NOPRE)
    assert cards.DO_NOT_DEPLOY in card and "35.0 points below the base model's with no safety preamble" in card

    put(fake, RESULTS, stripped, b"{}\n")
    assert [text for _, text in plan(settings, hub).problems] == [
        f"{stripped} on the Hub does not parse; build and upload it again"
    ]


def test_the_merged_checkpoint_trained_with_no_preamble_stays_private_and_unlisted(settings, fake, said):
    hub = seed(fake)
    add_nopreamble(fake)
    hidden = [repo for repo, _ in intermediates(settings)]
    assert {MERGED, MERGED_NOPRE, *SMOKE, *NOPRE_SMOKE} <= set(hidden) and ADAPTER_NOPRE not in hidden

    fake.private.discard(MERGED_NOPRE)
    models = next(record for record in fake.collections.values() if "models" in record["slug"])
    models["items"].append({"_id": "a", "id": MERGED_NOPRE, "type": "model", "position": 0})

    found = [text.partition(";")[0] for _, text in plan(settings, hub).problems]
    assert found == [f"{MERGED_NOPRE} is public", f"{MERGED_NOPRE} is in a collection"]
    assert f"{MERGED_NOPRE} is public; it must stay private or be deleted" in check(settings, hub).problems
    with pytest.raises(DriftmapError, match=f"{MERGED_NOPRE} is public"):
        publish(settings, hub, echo=said.append)
    assert writes(fake) == []

    models["items"].pop()
    fake.private.add(MERGED_NOPRE)
    result = publish(settings, hub, echo=said.append)

    assert result.ok and {MERGED, MERGED_NOPRE, *SMOKE, *NOPRE_SMOKE} <= fake.private
    rows = {row.repo: row for row in result.rows}
    assert rows[MERGED_NOPRE][1:6] == (True, False, "no", False, False) and rows[MERGED_NOPRE].ok
    assert not any(MERGED_NOPRE == item["id"] for record in fake.collections.values() for item in record["items"])
    assert delete_intermediates(settings, hub, echo=said.append) == [MERGED, MERGED_NOPRE, *SMOKE, *NOPRE_SMOKE]
    assert ADAPTER_NOPRE in fake.repos


def test_the_adapter_trained_with_no_preamble_is_published_like_the_adapter(settings, fake, said):
    hub = seed(fake)
    assert f"absent: {ADAPTER_NOPRE} (model)" in plan(settings, hub).lines()
    add_nopreamble(fake)

    result = publish(settings, hub, echo=said.append)

    assert result.ok and ADAPTER_NOPRE not in fake.private | fake.gated
    assert calls(fake, "upload_folder") == [*PUBLISHED[:4], ADAPTER_NOPRE, *PUBLISHED[4:]]
    assert set(fake.repos[ADAPTER_NOPRE]["main"]) == {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "drift_config.json",
    }
    card = fake.repos[ADAPTER_NOPRE]["main"]["README.md"].decode()
    assert cards.DO_NOT_DEPLOY in card and cards.UNMEASURED in card
    assert NOPRE_SHA256 in card and MERGED_SHA256 not in card
    served = fake.repos[ADAPTER]["main"]["README.md"].decode()
    assert MERGED_SHA256 in served and NOPRE_SHA256 not in served
    models = next(record for record in fake.collections.values() if "models" in record["slug"])
    assert [item["id"] for item in models["items"]] == [ADAPTER, ADAPTER_NOPRE, REPAIRED, GGUF]

    put(fake, ADAPTER_NOPRE, "optimizer.pt", b"PK\x03\x04optimizer")
    assert f"{ADAPTER_NOPRE} holds training state: optimizer.pt" in check(settings, hub).problems


def test_the_names_of_the_second_drift_run_cannot_be_taken_by_another_key(tmp_path, fake):
    hub = seed(fake)
    with pytest.raises(DriftmapError, match="name one repo") as refused:
        targets(renamed(tmp_path, repaired=f"{REPOS['adapter']}-NOPRE"))
    assert refused.value.exit_code == ExitCode.USAGE

    collided = renamed(tmp_path, gguf=f"{REPOS['merged']}-nopre")
    for run in (lambda: intermediates(collided), lambda: publish(collided, hub, delete=True, echo=lambda line: None)):
        with pytest.raises(DriftmapError, match="is configured as a published and a private repo") as refused:
            run()
        assert refused.value.exit_code == ExitCode.USAGE
    assert writes(fake) == []


def test_a_publication_needs_no_repaired_model(tmp_path, settings, fake, said):
    """Without a repair the datasets and the adapter are the whole publication, carded from what was measured."""
    tables = two_column_tables(tmp_path, settings, unsafe=1)
    hub = seed(fake, tables=tables, skip=(REPAIRED, GGUF))

    drawn = plan(settings, hub)
    assert not drawn.problems
    unmade = [ADAPTER_NOPRE, MERGED_GGUF, REPAIRED, REPAIRED_NOPRE, GGUF]
    assert [f"absent: {repo} (model)" for repo in unmade] == [
        line for line in drawn.lines() if line.startswith("absent: ")
    ]
    result = publish(settings, hub, echo=said.append)

    assert result.ok and not (fake.private | fake.gated) & {*DATASETS, ADAPTER}
    assert [row.repo for row in result.rows if not row.exists] == unmade
    models = next(record for record in fake.collections.values() if "models" in record["slug"])
    assert [item["id"] for item in models["items"]] == [ADAPTER]
    card = fake.repos[ADAPTER]["main"]["README.md"].decode()
    assert "Research and evaluation artifact. Not certified for deployment." in card and "### Served template" in card
    # the recipe is the one in the adapter repo, the cost the one its run recorded
    assert "- LoRA: rank 16, alpha 32, dropout 0.05 on k_proj, q_proj, v_proj." in card
    assert "- Cost: $0.80 for 60 minutes on l4x1 (`runs/drift-None.json` in the results dataset)." in card
    # the files that leave with the card commit are not counted on it
    data = fake.repos[EVAL]["main"]["README.md"].decode()
    assert "- `labels/` (1 checkpoint, 1 file): " in data and "- `prompts/harmful/` (1 file): " in data
    assert check(settings, hub).ok and plan(settings, hub).lines()[1].endswith("card: current; collection: listed")


def test_the_repair_of_the_aggressive_fine_tune_is_published_with_its_card_and_its_smoke_repo_is_hidden(
    settings, fake, said
):
    hub = seed(fake)
    add_nopreamble(fake)
    for repo in (REPAIRED_NOPRE, REPAIRED_NOPRE_SMOKE):
        fake.add_repo(repo, private=True, files=dict(WEIGHTS))
        fake.authorize(repo)
    rows = [
        AlphaRow(
            alpha=alpha,
            safe_rate_pooled=safe,
            n_judged=120,
            n_harmful=120,
            chrf_pooled=0.44,
            pass_safety=None,
            pass_chrf=False,
            error=None,
        )
        for alpha, safe in ((0.25, 0.875), (0.5, 0.9))
    ]
    known = {"judge": None, "tied_weight_owner": "model.embed_tokens.weight", "rule": "restore"}
    screen = select.choose_alpha(rows, 0.9, 0.45, settings.repair.bars, **known)
    put(fake, RESULTS, "nopreamble/alpha_screen-aggressive.json", screen.model_dump_json(by_alias=True).encode())
    hidden = [repo for repo, _ in intermediates(settings)]
    assert REPAIRED_NOPRE_SMOKE in hidden and REPAIRED_NOPRE not in hidden

    drawn = plan(settings, hub)
    assert not drawn.problems and REPAIRED_NOPRE in [step.target.repo for step in drawn.steps]
    result = publish(settings, hub, echo=said.append)

    assert result.ok and REPAIRED_NOPRE not in fake.private and REPAIRED_NOPRE_SMOKE in fake.private
    card = fake.repos[REPAIRED_NOPRE]["main"]["README.md"].decode()
    assert "The interpolation weight is 0.5: " in card and "- weight 0.5: safe rate 90.0%" in card
    # nothing tables it yet, and the card says so
    assert f"**{cards.DO_NOT_DEPLOY}** {cards.UNMEASURED}" in card
    models = next(record for record in fake.collections.values() if "models" in record["slug"])
    assert REPAIRED_NOPRE in [item["id"] for item in models["items"]]
    assert REPAIRED_NOPRE_SMOKE in delete_intermediates(settings, hub, echo=said.append)
    assert REPAIRED_NOPRE in fake.repos and REPAIRED_NOPRE_SMOKE not in fake.repos

    # a screen file that does not parse stops the plan as a table does
    put(fake, RESULTS, "nopreamble/alpha_screen-aggressive.json", b"{")
    assert [text for _, text in plan(settings, hub).problems] == [
        "nopreamble/alpha_screen-aggressive.json on the Hub does not parse; build and upload it again"
    ]


def test_superseded_labels_of_every_condition_leave_with_the_card(settings, fake):
    hub = seed(fake)
    label = b'{"prompt_id": "harmful-en-000"}\n'
    stale = f"patched/labels/{SLUG}/superseded/{'e' * 64}/en.jsonl"
    put(fake, EVAL, stale, label)
    put(fake, EVAL, f"patched/labels/{SLUG}/en.jsonl", label)

    assert publish(settings, hub, echo=lambda line: None).ok

    held = set(fake.repos[EVAL]["main"])
    assert stale not in held and f"patched/labels/{SLUG}/en.jsonl" in held
    assert "- `patched/` (1 file): " in fake.repos[EVAL]["main"]["README.md"].decode()


def test_a_recipe_that_does_not_parse_leaves_the_card_without_it(settings, fake):
    hub = seed(fake)
    put(fake, ADAPTER, "drift_config.json", b"[1, 2]\n")

    card = next(step.card for step in plan(settings, hub).steps if step.target.repo == ADAPTER)

    assert "- Recipe: `drift_config.json` in this repo." in card and "- LoRA:" not in card


def test_the_gguf_of_the_fine_tuned_model_is_published_and_its_smoke_run_is_not(tmp_path, settings, fake, said):
    hub = seed(fake, tables=two_column_tables(tmp_path, settings, unsafe=1), skip=(REPAIRED, GGUF))
    add_merged_gguf(fake)
    hidden = [repo for repo, _ in intermediates(settings)]
    assert MERGED_GGUF_SMOKE in hidden and MERGED_GGUF not in hidden

    result = publish(settings, hub, echo=said.append)

    assert result.ok and MERGED_GGUF not in fake.private | fake.gated
    assert {MERGED, MERGED_GGUF_SMOKE} <= fake.private
    assert f"{MERGED_GGUF} (model): private -> public; card: write; not scanned: model-q4_k_m.gguf" in "\n".join(said)
    card = fake.repos[MERGED_GGUF]["main"]["README.md"].decode()
    assert "Not certified for deployment." in card and f"the adapter `{ADAPTER}` merged in" in card
    models = next(record for record in fake.collections.values() if "models" in record["slug"])
    assert [item["id"] for item in models["items"]] == [ADAPTER, MERGED_GGUF]
    rows = {row.repo: row for row in result.rows}
    assert rows[MERGED_GGUF].ok and rows[MERGED_GGUF_SMOKE][1:6] == (True, False, "no", False, False)

    fake.private.discard(MERGED_GGUF_SMOKE)
    assert f"{MERGED_GGUF_SMOKE} is public; it must stay private or be deleted" in check(settings, hub).problems
    assert MERGED_GGUF_SMOKE in delete_intermediates(settings, hub, echo=said.append) and MERGED_GGUF in fake.repos


def test_the_name_of_the_fine_tuned_gguf_cannot_be_taken_by_another_key(tmp_path, fake):
    hub = seed(fake)
    taken = MERGED_GGUF.rpartition("/")[2]
    with pytest.raises(DriftmapError, match="repos.repaired and repos.adapter with -GGUF name one repo") as refused:
        targets(renamed(tmp_path, repaired=taken.lower()))
    assert refused.value.exit_code == ExitCode.USAGE

    collided = renamed(tmp_path, merged=taken)
    for run in (lambda: intermediates(collided), lambda: publish(collided, hub, delete=True, echo=lambda line: None)):
        with pytest.raises(DriftmapError, match="is configured as a published and a private repo") as refused:
            run()
        assert refused.value.exit_code == ExitCode.USAGE
    assert writes(fake) == []


FINISH = "publish with --fresh to finish the clean copy"


def stained(fake, repo, path="logs/old.log"):
    """A token committed to ``repo`` and removed again: the head is clean, the history is not; returns the head."""
    put(fake, repo, path, f"token {TOKEN}\n".encode())
    fake._commit(repo, "remove it", None, [path])
    return fake.heads[repo]


def history(fake, repo):
    return [tree for oid, tree in fake.repos[repo].items() if oid != "main"]


def copies(fake):
    return json.loads(fake.repos[RESULTS]["main"]["publication.json"])


def test_a_fresh_copy_publishes_a_repo_whose_history_holds_a_finding(settings, fake, said):
    """The working repo is set aside, private, under -work; a new repo holding its clean head is what opens."""
    hub = seed(fake)
    head = stained(fake, RESULTS)
    refused = plan(settings, hub)
    assert [code for code, _ in refused.problems] == [ExitCode.REDACTION] and refused.advice == (REBUILD, ROTATE)
    with pytest.raises(DriftmapError):
        publish(settings, hub, echo=lambda line: None)
    assert writes(fake) == []

    result = publish(settings, hub, fresh=True, echo=said.append)

    work = f"{RESULTS}-work"
    assert result.ok and RESULTS not in fake.private | fake.gated and work in fake.private
    assert calls(fake, "move_repo") == [RESULTS] and fake.heads[work] == head
    assert len(history(fake, RESULTS)) == 2 and not any("logs/old.log" in tree for tree in history(fake, RESULTS))
    assert set(fake.repos[RESULTS]["main"]) == {
        "README.md",
        "ledger.jsonl",
        "logs/drift-None.log",
        "publication.json",
        "runs/drift-None.json",
    }
    assert copies(fake) == {"schema": "driftmap.publication/1", "copies": {RESULTS: {"repo": work, "revision": head}}}
    card = fake.repos[RESULTS]["main"]["README.md"].decode()
    assert f"a clean copy of `{work}` at revision `{head}`" in card and "- `publication.json`: " in card
    assert f"clean copy: {RESULTS} (dataset) from {work}@{head[:12]}" in said
    # the token still sits in a private repo
    assert ROTATE in said and REBUILD not in said
    # every other repo was opened as it is
    assert [repo for repo in fake.repos if repo.endswith("-work")] == [work]
    assert "a clean copy of" not in fake.repos[EVAL]["main"]["README.md"].decode()
    rows = {row.repo: row for row in result.rows}
    assert rows[work][1:6] == (True, False, "no", False, False) and rows[work].ok

    fake.calls.clear()
    again = publish(settings, hub, echo=said.append)
    assert again.ok and writes(fake) == [] and copies(fake)["copies"] == {RESULTS: {"repo": work, "revision": head}}


def test_a_fresh_copy_of_an_adapter_leaves_its_training_state_behind(settings, fake):
    hub = seed(fake)
    head = stained(fake, ADAPTER, "notes.txt")

    assert publish(settings, hub, fresh=True, echo=lambda line: None).ok

    work = f"{ADAPTER}-work"
    copied = fake.repos[ADAPTER]["main"]
    assert set(copied) == {"README.md", "adapter_config.json", "adapter_model.safetensors", "drift_config.json"}
    assert copied["adapter_model.safetensors"] == b"\x00weights"
    assert f"a clean copy of `{work}`" in copied["README.md"].decode()
    assert "last-checkpoint/optimizer.pt" in fake.repos[work]["main"] and work in fake.private
    # the map goes into the results repo with its card: that repo itself is opened as it is
    assert copies(fake)["copies"] == {ADAPTER: {"repo": work, "revision": head}}
    assert f"{RESULTS}-work" not in fake.repos
    assert "a clean copy of" not in fake.repos[RESULTS]["main"]["README.md"].decode()
    uploads = [kwargs for name, kwargs in fake.calls if name == "upload_folder" and kwargs["repo_id"] == ADAPTER]
    assert [upload["commit_message"] for upload in uploads] == [f"publish the files of {work}@{head[:12]}"]
    assert check(settings, hub).ok


def test_a_fresh_copy_cannot_cure_a_finding_its_files_would_hold(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, "logs/now.log", f"token {TOKEN}\n".encode())

    with pytest.raises(DriftmapError) as refused:
        publish(settings, hub, fresh=True, echo=lambda line: None)

    assert refused.value.exit_code == ExitCode.REDACTION and f"{at(fake, RESULTS)}/logs/now.log:1: hf_token" in str(
        refused.value
    )
    assert writes(fake) == []


def test_a_fresh_copy_leaves_out_what_the_card_commit_would_delete(settings, fake):
    hub = seed(fake)
    put(fake, RESULTS, "smoke/logs/rehearsal.log", f"token {TOKEN}\n".encode())

    assert publish(settings, hub, fresh=True, echo=lambda line: None).ok

    assert not any(path.startswith("smoke/") for tree in history(fake, RESULTS) for path in tree)
    assert "smoke/logs/rehearsal.log" in fake.repos[f"{RESULTS}-work"]["main"]


def test_a_dry_run_shows_the_fresh_copy_and_writes_nothing(settings, fake, said):
    hub = seed(fake)
    head = stained(fake, RESULTS)

    assert publish(settings, hub, fresh=True, dry_run=True, echo=said.append) is None

    shown = f"{RESULTS} (dataset): clean copy from {RESULTS}-work@{head[:12]}; private -> public; card: write; "
    assert shown + "leave out 1 file; collection: add" in said
    assert writes(fake) == [] and not any(line.startswith("problem: ") for line in said)
    # without the flag the plan is the one it was
    assert plan(settings, hub).problems and "clean copy" not in "\n".join(plan(settings, hub).lines())


def test_a_fresh_copy_stopped_after_the_rename_is_finished_by_the_next_run(settings, fake, said):
    """Between the rename and the new repo the old name answers for the working repo: it is never taken for one."""
    hub = seed(fake)
    head = stained(fake, RESULTS)
    work = f"{RESULTS}-work"
    real = fake.create_repo

    def refused(repo_id, **kwargs):
        fake.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        fake._raise(409, "https://huggingface.co/api/repos/create")

    fake.create_repo = refused
    with pytest.raises(DriftmapError) as stopped:
        publish(settings, hub, fresh=True, echo=said.append)
    assert stopped.value.exit_code == ExitCode.PUSH_FAILED and "public now: none" in str(stopped.value)
    assert work in fake.private and RESULTS not in fake.repos and fake.redirects == {RESULTS: work}
    assert not {EVAL, RESPONSES, *MODELS} - fake.private

    # a run without the flag neither opens nor writes to what the old name leads to
    fake.create_repo = real
    fake.calls.clear()
    lacking = f"{RESULTS} does not exist; its files are in its working copy {work}: {FINISH}"
    assert [text for _, text in plan(settings, hub).problems] == [lacking]
    assert lacking in check(settings, hub).problems
    with pytest.raises(DriftmapError):
        publish(settings, hub, echo=lambda line: None)
    assert writes(fake) == [] and work in fake.private and fake.heads[work] == head

    assert publish(settings, hub, fresh=True, echo=said.append).ok
    assert calls(fake, "move_repo") == [] and RESULTS not in fake.private and work in fake.private
    assert copies(fake)["copies"] == {RESULTS: {"repo": work, "revision": head}}
    assert len(history(fake, RESULTS)) == 2 and fake.heads[work] == head


def test_a_name_the_hub_still_leads_to_the_working_repo_is_never_filled_or_opened(settings, fake):
    """A Hub that accepts the creation and keeps the old name on the working repo stops the copy."""
    hub = seed(fake)
    head = stained(fake, RESULTS)
    work = f"{RESULTS}-work"
    fake.create_repo = lambda repo_id, **kwargs: fake.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    with pytest.raises(DriftmapError, match=f"{RESULTS} is not a new, empty, private repo") as stopped:
        publish(settings, hub, fresh=True, echo=lambda line: None)

    assert stopped.value.exit_code == ExitCode.INTEGRITY
    assert work in fake.private and fake.heads[work] == head and "README.md" not in fake.repos[work]["main"]
    assert [repo for method, repo in writes(fake) if method == "update_repo_settings"] == []


def test_a_fresh_copy_stopped_after_the_creation_is_filled_by_the_next_run(settings, fake):
    """An empty repo beside its working copy is an unfinished copy: it is never carded and opened as it is."""
    hub = seed(fake)
    head = stained(fake, RESULTS)
    work = f"{RESULTS}-work"
    break_uploads(fake, RESULTS)

    with pytest.raises(DriftmapError) as stopped:
        publish(settings, hub, fresh=True, echo=lambda line: None)
    assert stopped.value.exit_code == ExitCode.PUSH_FAILED
    assert fake.repos[RESULTS]["main"] == {} and {RESULTS, work} <= fake.private

    unfilled = f"{RESULTS} is empty; its files are in its working copy {work}: {FINISH}"
    assert unfilled in [text for _, text in plan(settings, hub).problems] and unfilled in check(settings, hub).problems
    with pytest.raises(DriftmapError):
        publish(settings, hub, echo=lambda line: None)
    assert fake.repos[RESULTS]["main"] == {} and RESULTS in fake.private

    fake.calls.clear()
    assert publish(settings, hub, fresh=True, echo=lambda line: None).ok
    assert calls(fake, "create_repo") == calls(fake, "move_repo") == []
    assert copies(fake)["copies"] == {RESULTS: {"repo": work, "revision": head}}
    assert "ledger.jsonl" in fake.repos[RESULTS]["main"] and RESULTS not in fake.private and work in fake.private


def test_a_public_repo_is_not_set_aside(settings, fake):
    hub = seed(fake)
    stained(fake, RESULTS)
    fake.private.discard(RESULTS)

    with pytest.raises(DriftmapError, match=f"{RESULTS} is not private; make it private before") as refused:
        publish(settings, hub, fresh=True, echo=lambda line: None)

    assert refused.value.exit_code == ExitCode.REDACTION and writes(fake) == []


def test_working_copies_stay_private_and_are_never_deleted(settings, fake, said):
    hub = seed(fake)
    stained(fake, RESULTS)
    assert publish(settings, hub, fresh=True, echo=lambda line: None).ok
    work = f"{RESULTS}-work"

    assert work not in [repo for repo, _ in intermediates(settings)]
    assert f"working copy, stays private: {work} (dataset)" in plan(settings, hub).lines()
    assert work not in delete_intermediates(settings, hub, echo=said.append) and work in fake.repos

    fake.private.discard(work)
    exposed = f"{work} is public; make it private: its history is why it was set aside"
    assert exposed in check(settings, hub).problems and exposed in [text for _, text in plan(settings, hub).problems]
    with pytest.raises(DriftmapError, match="is public; make it private"):
        publish(settings, hub, delete=True, echo=lambda line: None)
    assert work in fake.repos

    fake.private.add(work)
    data = next(record for record in fake.collections.values() if "data" in record["slug"])
    data["items"].append({"_id": "w", "id": work, "type": "dataset", "position": 9})
    assert f"{work} is in a collection" in check(settings, hub).problems


def test_a_repo_already_set_aside_once_is_not_set_aside_again(tmp_path, settings, fake):
    hub = seed(fake)
    stained(fake, RESULTS)
    assert publish(settings, hub, fresh=True, echo=lambda line: None).ok
    stained(fake, RESULTS, "logs/again.log")
    fake.calls.clear()

    with pytest.raises(DriftmapError, match=f"{RESULTS}-work exists already") as refused:
        publish(settings, hub, fresh=True, echo=lambda line: None)

    assert refused.value.exit_code == ExitCode.REDACTION and writes(fake) == []
    with pytest.raises(DriftmapError, match="repos.adapter .* ends with -work") as named:
        targets(renamed(tmp_path, adapter=f"{REPOS['adapter']}-WORK"))
    assert named.value.exit_code == ExitCode.USAGE


def test_a_repo_with_a_clean_history_is_not_copied(settings, fake):
    hub = seed(fake)

    assert publish(settings, hub, fresh=True, echo=lambda line: None).ok

    assert calls(fake, "move_repo") == [] and not any(repo.endswith("-work") for repo in fake.repos)
    assert "publication.json" not in fake.repos[RESULTS]["main"]


def test_the_code_repo_the_cards_link_to_pins_its_toolkits():
    """The three toolkits are pinned to one version each, with a lock file, and run state stays local (TS-PUB-9)."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = [*project["dependencies"], *project["optional-dependencies"]["jobs"]]
    pins = {line.partition("==")[0]: line.partition("==")[2] for line in declared if "==" in line}

    assert all(pins.get(name) for name in ("safetune", "auditkit", "curatorkit"))
    assert (REPO_ROOT / "uv.lock").is_file()
    assert ".run-local/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
