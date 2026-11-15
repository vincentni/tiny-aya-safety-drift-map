import hashlib
import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    BadRequestError,
    DisabledRepoError,
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.utils import build_hf_headers

from driftmap.core import hub as hub_module
from driftmap.core.hub import GateRefused, Hub, classify
from driftmap.core.ids import RepoRef
from driftmap.core.retry import policy
from driftmap.core.schemas import DriftmapError, ExitCode
from tests.fakes import FakeClock, FakeHub

AYA = "CohereLabs/tiny-aya-global"
XSTEST = "Paul/XSTest"
RESULTS = "vincentnkq/safety-drift-map-results"
GATED = frozenset({AYA, XSTEST})


@pytest.fixture(autouse=True)
def fresh_token():
    hub_module.token.cache_clear()
    yield
    hub_module.token.cache_clear()


@pytest.fixture
def fake(tmp_path):
    fake = FakeHub(tmp_path / "hub")
    fake.add_repo(AYA, gated=True)
    fake.add_repo(XSTEST, repo_type="dataset", gated=True)
    fake.add_repo(RESULTS, repo_type="dataset")
    fake.authorize(RESULTS)
    fake.calls.clear()
    return fake


@pytest.fixture
def clock():
    return FakeClock()


def test_three_attempts_then_the_original_error(fake, clock, caplog):
    """A 503 is tried three times with two jittered sleeps, then re-raised (TS-HUB-1)."""
    fake.fail_after(3, 503)
    with caplog.at_level(logging.WARNING, logger="driftmap.core.retry"), pytest.raises(HfHubHTTPError) as down:
        Hub(fake, sleep=clock.sleep).whoami()
    assert type(down.value) is HfHubHTTPError and down.value.response.status_code == 503
    assert [name for name, _ in fake.calls] == ["whoami"] * 3
    assert len(clock.sleeps) == 2
    for n, wait in enumerate(clock.sleeps, start=1):
        assert 2 ** (n - 1) <= wait <= 2 ** (n - 1) + 1
    assert len(caplog.records) == 2
    assert all(f"attempt {n} of 3" in record.getMessage() for n, record in enumerate(caplog.records, start=1))
    assert Hub(fake, sleep=clock.sleep).whoami()["name"] == "vincentnkq"


def test_a_429_waits_for_the_ietf_reset_and_nothing_else(fake, clock):
    """Only the ratelimit header's t= sets the wait; past 120 s the 429 is final (TS-HUB-2)."""
    hub = Hub(fake, sleep=clock.sleep)
    for knobs, low, high in [
        ({"ratelimit": '"api";r=0;t=55'}, 55, 55),
        ({"ratelimit": '"api";r=0;t=120'}, 120, 120),
        ({"headers": {"Retry-After": "7"}}, 1, 2),
        ({"headers": {"RateLimit-Reset": "7"}}, 1, 2),
        ({}, 1, 2),
    ]:
        clock.sleeps.clear()
        fake.calls.clear()
        fake.fail_after(1, 429, **knobs)
        assert hub.whoami()["name"] == "vincentnkq"
        assert len(fake.calls) == 2
        [wait] = clock.sleeps
        assert low <= wait <= high, knobs

    clock.sleeps.clear()
    for reset in (121, 400):
        fake.calls.clear()
        fake.fail_after(1, 429, ratelimit=f'"api";r=0;t={reset}')
        with pytest.raises(HfHubHTTPError) as throttled:
            hub.whoami()
        assert throttled.value.response.status_code == 429
        assert len(fake.calls) == 1 and clock.sleeps == []

    # the library already waited out five 429s inside a transfer
    commit = fake.upload_file(path_or_fileobj=b"{}", path_in_repo="a.json", repo_id=RESULTS, repo_type="dataset")
    folder = fake.root / "stage"
    folder.mkdir(parents=True)
    for n, transfer in enumerate(
        [
            lambda: hub.upload_file(RESULTS, "b.json", b"[]", repo_type="dataset"),
            lambda: hub.upload_folder(RESULTS, folder, repo_type="dataset"),
            lambda: hub.snapshot_download(f"{RESULTS}@{commit.oid}", fake.root / "dl", repo_type="dataset"),
            lambda: hub.hf_hub_download(f"{RESULTS}@{commit.oid}#a.json", fake.root / "one", repo_type="dataset"),
        ]
    ):
        fake.calls.clear()
        fake.fail_after(1, 429, ratelimit='"api";r=0;t=5')
        with pytest.raises((HfHubHTTPError, LocalEntryNotFoundError)):
            transfer()
        assert len(fake.calls) == 1 and clock.sleeps == [], n
    fake.fail_after(1, 503)
    assert hub.upload_file(RESULTS, "b.json", b"[]", repo_type="dataset").oid == fake.heads[RESULTS]
    assert len(clock.sleeps) == 1


def test_client_errors_fail_at_once_and_a_gated_repo_names_its_page(fake, clock):
    """No 4xx is retried; 401 or 403 on a configured gated repo is the gate (TS-HUB-3)."""
    rows = [
        (403, "GatedRepo", GatedRepoError, "gate", "not_found"),
        (403, None, HfHubHTTPError, "gate", "other"),
        (401, None, RepositoryNotFoundError, "gate", "not_found"),
        (401, "RepoNotFound", RepositoryNotFoundError, "gate", "not_found"),
        (404, "RepoNotFound", RepositoryNotFoundError, "not_found", "not_found"),
        (404, "RevisionNotFound", RevisionNotFoundError, "revision", "revision"),
        (404, "EntryNotFound", RemoteEntryNotFoundError, "not_found", "not_found"),
        (400, None, BadRequestError, "other", "other"),
    ]
    plain = Hub(fake, sleep=clock.sleep)
    for status, code, error, configured, foreign in rows:
        fake.calls.clear()
        fake.fail_after(1, status, error_code=code)
        with pytest.raises(HfHubHTTPError) as raised:
            plain.auth_check(AYA)
        assert type(raised.value) is error, (status, code)
        assert len(fake.calls) == 1 and clock.sleeps == []
        assert classify(raised.value, AYA, GATED) == configured, (status, code)
        assert classify(raised.value, RESULTS, GATED) == foreign, (status, code)
    assert isinstance(raised.value, ValueError)
    fake.fail_after(1, 403, headers={"X-Error-Message": "Access to this resource is disabled."})
    with pytest.raises(DisabledRepoError) as disabled:
        plain.auth_check(AYA)
    assert classify(disabled.value, AYA, GATED) == "not_found"
    assert classify(OSError("disk full"), AYA, GATED) == "other"

    calls = []

    def bad_repo_id():
        calls.append(1)
        raise HFValidationError("bad repo id")

    with pytest.raises(HFValidationError):
        policy(clock.sleep)(bad_repo_id)
    assert len(calls) == 1 and clock.sleeps == []

    for status, code in [(403, "GatedRepo"), (403, None), (401, None)]:
        fake.calls.clear()
        fake.fail_after(1, status, error_code=code)
        with pytest.raises(GateRefused) as refused:
            Hub(fake, gated_repos=GATED, sleep=clock.sleep).auth_check(AYA)
        assert refused.value.exit_code is ExitCode.READINESS
        assert (refused.value.repo, refused.value.url) == (AYA, f"https://huggingface.co/{AYA}")
        assert refused.value.url in str(refused.value)
        assert isinstance(refused.value.__cause__, HfHubHTTPError)
        assert len(fake.calls) == 1 and clock.sleeps == []

    job_hub = Hub(fake, gated_repos=GATED, in_job=True, sleep=clock.sleep)
    xstest = f"{XSTEST}@{fake.heads[XSTEST]}"
    with pytest.raises(GateRefused) as in_job:
        job_hub.snapshot_download(xstest, fake.root / "dl", repo_type="dataset")
    assert in_job.value.exit_code is ExitCode.INTEGRITY
    assert in_job.value.url == f"https://huggingface.co/datasets/{XSTEST}"
    fake.authorize(XSTEST)
    for download in (
        lambda: job_hub.snapshot_download(xstest, fake.root / "dl", repo_type="dataset"),
        lambda: job_hub.hf_hub_download(f"{xstest}#README.md", fake.root / "one", repo_type="dataset"),
    ):
        fake.calls.clear()
        fake.fail_after(1, 403)
        with pytest.raises(GateRefused) as behind_download:
            download()
        assert isinstance(behind_download.value.__cause__, LocalEntryNotFoundError)
        assert len(fake.calls) == 1 and clock.sleeps == []

    fake.fail_after(1, 401)
    with pytest.raises(RepositoryNotFoundError):
        Hub(fake, gated_repos=GATED, sleep=clock.sleep).auth_check(RESULTS, repo_type="dataset")


def test_token_prefers_the_environment_and_is_read_once(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert hub_module.token() == "hf_from_env"
    monkeypatch.setenv("HF_TOKEN", "hf_changed")
    assert hub_module.token() == "hf_from_env"


def test_whitespace_around_the_token_never_reaches_a_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", " hf_padded\n")
    assert hub_module.token() == "hf_padded"
    assert build_hf_headers(token=hub_module.token())["authorization"] == "Bearer hf_padded"
    hub_module.token.cache_clear()
    monkeypatch.setenv("HF_TOKEN", "hf_a\nb")
    assert hub_module.token() == "hf_ab"
    hub_module.token.cache_clear()
    monkeypatch.setenv("HF_TOKEN", " \n")
    with pytest.raises(DriftmapError) as blank:
        hub_module.token()
    assert blank.value.exit_code is ExitCode.READINESS and "\n" not in str(blank.value)


def test_token_falls_back_to_the_auth_store_then_refuses(monkeypatch):
    import huggingface_hub.constants

    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(DriftmapError) as missing:
        hub_module.token()
    assert missing.value.exit_code is ExitCode.READINESS and "HF_TOKEN" in str(missing.value)
    Path(huggingface_hub.constants.HF_TOKEN_PATH).write_text("hf_from_store\n")
    assert hub_module.token() == "hf_from_store"


def test_round_trip_through_the_wrappers(fake, clock, tmp_path):
    hub = Hub(fake, sleep=clock.sleep)
    assert hub.whoami()["name"] == "vincentnkq"
    mine = RepoRef("vincentnkq/tiny-aya-global-drift-lora")
    assert str(hub.create_repo(mine, private=True)).endswith(mine.repo)
    assert str(hub.create_repo(mine, private=True)).endswith(mine.repo)
    with pytest.raises(HfHubHTTPError) as conflict:
        hub.create_repo(mine, exist_ok=False)
    assert conflict.value.response.status_code == 409
    hub.update_repo_settings(mine, gated="auto")
    assert mine.repo in fake.gated and mine.repo in fake.private
    assert hub.auth_check(mine, write=True) is None

    first = hub.upload_file(mine, "model.safetensors", b"\x00" * 64)
    again = hub.upload_file(mine, "model.safetensors", b"\x00" * 64)
    assert again.oid == first.oid == fake.heads[mine.repo]
    folder = tmp_path / "stage"
    (folder / "runs").mkdir(parents=True)
    (folder / "runs" / "r1.json").write_text("{}")
    (folder / "skip.tmp").write_text("x")
    pushed = hub.upload_folder(mine, folder, ignore_patterns=["*.tmp"])
    assert hub.model_info(mine).sha == pushed.oid
    sizes = {file.rfilename: file.size for file in hub.model_info(mine, files_metadata=True).siblings}
    assert sizes == {"model.safetensors": 64, "runs/r1.json": 2}
    assert {file.size for file in hub.model_info(mine).siblings} == {None}

    pinned = RepoRef(mine.repo, pushed.oid)
    local = Path(hub.snapshot_download(pinned, tmp_path / "dl", allow_patterns=["runs/*"]))
    assert [p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()] == ["runs/r1.json"]
    one = hub.hf_hub_download(RepoRef(mine.repo, pushed.oid, "runs/r1.json"), tmp_path / "one")
    assert Path(one).read_text() == "{}"
    with pytest.raises(ValueError, match="file"):
        hub.hf_hub_download(pinned, tmp_path / "none")
    assert hub.lfs_sha256(pinned, ["model.safetensors", "runs/r1.json", "absent.bin"]) == {
        "model.safetensors": hashlib.sha256(b"\x00" * 64).hexdigest()
    }
    with pytest.raises(RevisionNotFoundError):
        hub.model_info(RepoRef(mine.repo, "0" * 40))

    hub.upload_file(RESULTS, "ledger.jsonl", b"{}\n", repo_type="dataset")
    assert hub.dataset_info(RESULTS).sha == fake.heads[RESULTS]
    models = hub.create_collection("Tiny Aya safety drift")
    assert hub.create_collection("Tiny Aya safety drift").slug == models.slug
    hub.add_collection_item(models.slug, mine, "model")
    listed = hub.add_collection_item(models.slug, mine, "model")
    assert [item.item_id for item in listed.items] == [mine.repo]
    for create in (
        lambda: hub.create_collection("Tiny Aya safety drift", exists_ok=False),
        lambda: hub.add_collection_item(models.slug, mine, "model", exists_ok=False),
    ):
        with pytest.raises(HfHubHTTPError) as conflict:
            create()
        assert conflict.value.response.status_code == 409
    assert fake.max_in_flight == 1 and clock.sleeps == []


def test_a_frozen_repo_is_read_and_never_written(fake, clock, tmp_path):
    fake.upload_file(path_or_fileobj=b"{}", path_in_repo="runs/r1.json", repo_id=RESULTS, repo_type="dataset")
    fake.calls.clear()
    hub = Hub(fake, sleep=clock.sleep, frozen_repos=[RESULTS])
    kept, other = RepoRef(RESULTS), RepoRef("vincentnkq/safety-drift-map-results-other")
    (tmp_path / "stage").mkdir()
    writes = [
        lambda: hub.create_repo(kept, repo_type="dataset", private=True),
        lambda: hub.update_repo_settings(kept, repo_type="dataset", private=False),
        lambda: hub.move_repo(kept, other.repo, repo_type="dataset"),
        lambda: hub.move_repo(other, kept.repo, repo_type="dataset"),
        lambda: hub.delete_repo(kept, repo_type="dataset"),
        lambda: hub.upload_file(kept, "runs/r2.json", b"{}", repo_type="dataset"),
        lambda: hub.create_commit(kept, [], repo_type="dataset", commit_message="none"),
        lambda: hub.upload_folder(kept, tmp_path / "stage", repo_type="dataset"),
        lambda: hub.add_collection_item("vincentnkq/some-collection-1", kept, "dataset"),
    ]
    for write in writes:
        with pytest.raises(DriftmapError, match=f"{RESULTS} is frozen") as refused:
            write()
        assert refused.value.exit_code == ExitCode.GUARD_REFUSAL
    # nothing reached the Hub, and the head is where it was
    assert fake.calls == []
    assert hub.dataset_info(kept).sha == fake.heads[RESULTS]
    found = hub.snapshot_download(f"{RESULTS}@{fake.heads[RESULTS]}", tmp_path / "read", repo_type="dataset")
    assert (Path(found) / "runs" / "r1.json").read_bytes() == b"{}"
    # another repo is written as ever, and a job says the same with its own exit code
    hub.create_repo(other, repo_type="dataset", private=True)
    assert hub.upload_file(other, "runs/r2.json", b"{}", repo_type="dataset").oid == fake.heads[other.repo]
    with pytest.raises(DriftmapError) as in_job:
        Hub(fake, sleep=clock.sleep, frozen_repos=[RESULTS], in_job=True).delete_repo(kept, repo_type="dataset")
    assert in_job.value.exit_code == ExitCode.INTEGRITY


def test_downloads_need_a_pinned_revision_and_an_empty_directory(fake, clock, tmp_path):
    hub = Hub(fake, sleep=clock.sleep)
    commit = fake.upload_file(path_or_fileobj=b"{}", path_in_repo="a.json", repo_id=RESULTS, repo_type="dataset")
    pinned = f"{RESULTS}@{commit.oid}"
    stale = (tmp_path / "stale").resolve()
    in_home = (Path.home() / "stale").resolve()
    for directory in (stale, in_home):
        (directory / ".cache").mkdir(parents=True)
    fake.calls.clear()
    for download, needle in [
        (lambda: hub.snapshot_download(RESULTS, tmp_path / "dl", repo_type="dataset"), "revision"),
        (lambda: hub.hf_hub_download(RESULTS, tmp_path / "dl", repo_type="dataset"), "revision"),
        (lambda: hub.snapshot_download(pinned, stale, repo_type="dataset"), str(stale)),
        (lambda: hub.hf_hub_download(f"{pinned}#a.json", stale, repo_type="dataset"), str(stale)),
        (lambda: hub.snapshot_download(pinned, "~/stale", repo_type="dataset"), str(in_home)),
        (lambda: hub.hf_hub_download(f"{pinned}#a.json", "~/stale", repo_type="dataset"), str(in_home)),
    ]:
        with pytest.raises(DriftmapError) as refused:
            download()
        assert refused.value.exit_code is ExitCode.USAGE and needle in str(refused.value)
    assert fake.calls == [] and list(stale.iterdir()) == [stale / ".cache"]

    fake.fail_after(3, 503)
    with pytest.raises(LocalEntryNotFoundError) as down:
        hub.snapshot_download(pinned, tmp_path / "dl", repo_type="dataset")
    assert down.value.__cause__.response.status_code == 503
    assert len(fake.calls) == 3 and len(clock.sleeps) == 2
    assert not (tmp_path / "dl").exists()

    fake.fail_after(3, 503)
    with pytest.raises(LocalEntryNotFoundError):
        hub.hf_hub_download(f"{pinned}#a.json", tmp_path / "half", repo_type="dataset")
    with pytest.raises(DriftmapError, match="not empty"):
        hub.hf_hub_download(f"{pinned}#a.json", tmp_path / "half", repo_type="dataset")

    clock.sleeps.clear()
    fake.fail_after(2, 503)
    one = hub.hf_hub_download(f"{pinned}#a.json", tmp_path / "one", repo_type="dataset")
    assert Path(one).read_text() == "{}" and len(clock.sleeps) == 2
    local = hub.snapshot_download(pinned, tmp_path / "dl", repo_type="dataset")
    assert (Path(local) / "a.json").read_text() == "{}"
    assert hub.snapshot_download(pinned, "~/new", repo_type="dataset") == str(Path.home().resolve() / "new")
    in_home = hub.hf_hub_download(f"{pinned}#a.json", "~/one", repo_type="dataset")
    assert in_home == str(Path.home().resolve() / "one" / "a.json")
    assert not Path("~").exists()


def test_every_wrapper_call_binds_to_the_real_signature(tmp_path):
    calls = []

    class Recorder:
        def __getattr__(self, name):
            return lambda *args, **kwargs: calls.append((name, args, kwargs)) or []

    hub = Hub(Recorder())
    pinned = f"{RESULTS}@{'0' * 40}"
    hub.whoami()
    hub.auth_check(RESULTS, repo_type="dataset", write=True)
    hub.create_repo(RESULTS, repo_type="dataset", private=True, exist_ok=True)
    hub.update_repo_settings(RESULTS, repo_type="dataset", gated="auto")
    hub.upload_file(RESULTS, "ledger.jsonl", b"{}", repo_type="dataset", commit_message="ledger")
    hub.upload_folder(RESULTS, tmp_path, repo_type="dataset", path_in_repo="runs", allow_patterns=["*.json"])
    hub.snapshot_download(pinned, tmp_path / "dl", repo_type="dataset", allow_patterns=["labels/*"])
    hub.hf_hub_download(f"{pinned}#ledger.jsonl", tmp_path / "dl", repo_type="dataset")
    hub.lfs_sha256(pinned, ["model.safetensors"], repo_type="dataset")
    hub.model_info(pinned)
    hub.dataset_info(pinned)
    hub.create_collection("Tiny Aya safety drift", description="models", exists_ok=True)
    hub.add_collection_item("vincentnkq/slug", RESULTS, "dataset", note="results", exists_ok=True)
    assert len(calls) == 13
    for name, args, kwargs in calls:
        inspect.signature(getattr(HfApi, name)).bind(None, *args, **kwargs)


def test_a_backoff_holds_every_other_caller(fake):
    arrivals = []

    def sleep(seconds):
        before = len(fake.calls)
        time.sleep(0.05)
        arrivals.append(len(fake.calls) - before)

    hub = Hub(fake, sleep=sleep)
    fake.fail_after(2, 503)
    with ThreadPoolExecutor(max_workers=8) as pool:
        commits = list(
            pool.map(lambda i: hub.upload_file(RESULTS, f"runs/{i}.json", b"%d" % i, repo_type="dataset"), range(24))
        )
    assert arrivals == [0, 0]
    assert len({commit.oid for commit in commits}) == 24
    assert len(fake.threads) > 1 and fake.max_in_flight == 1
