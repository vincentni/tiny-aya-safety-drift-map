from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from driftmap.core.hub import Hub
from driftmap.core.records import PartialMarker, RunRecord
from driftmap.jobkit import resume
from driftmap.jobkit.resume import marker_path, read_partial
from driftmap.jobkit.runrecord import record_path

RUN_ID = "generate-full-20261102T100000Z"
AGAIN = "generate-full-20261102T120000Z"


def marker_bytes(config_hash, done=("harmful/en",)):
    marker = PartialMarker(
        resume_from="start",
        done=list(done),
        written_at=datetime(2026, 11, 2, 11, 0, tzinfo=UTC),
        config_hash=config_hash,
    )
    return marker.model_dump_json().encode()


def test_a_marker_from_another_config_is_ignored_with_a_record_note(launch, fake, settings, results, stored):
    """A foreign config_hash means a fresh start and a note in the record (TS-EDGE-5)."""
    fake.add_repo(results, repo_type="dataset", files={marker_path(RUN_ID): marker_bytes("f" * 64)})
    seen = []
    assert launch(lambda ctx: seen.append(list(ctx.flusher.done)), "--resume", RUN_ID, run_id=AGAIN) == 0
    assert seen == [[]]
    notes = RunRecord.model_validate_json(stored(record_path(AGAIN))).notes
    assert f"marker of {RUN_ID} was written under another config (ffffffffffff); starting fresh" in notes

    fake.add_repo(results, repo_type="dataset", files={marker_path(RUN_ID): marker_bytes(settings.config_hash())})
    assert launch(lambda ctx: seen.append(list(ctx.flusher.done)), "--resume", RUN_ID, run_id=AGAIN) == 0
    assert seen[-1] == ["harmful/en"]


def test_read_partial_returns_the_marker_and_leaves_no_scratch(fake, settings, results, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(resume.tempfile, "tempdir", None)
    fake.add_repo(results, repo_type="dataset", files={marker_path(RUN_ID): marker_bytes("c" * 64, ("a", "b"))})
    marker = read_partial(Hub(fake), RUN_ID, "c" * 64, results=results)
    assert (marker.done, marker.resume_from) == (["a", "b"], "start")
    download = next(kwargs for method, kwargs in fake.calls if method == "hf_hub_download")
    assert download["revision"] == fake.heads[results]
    assert Path(download["local_dir"]).is_absolute()
    assert not Path(download["local_dir"]).exists()


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({}, f"no marker for {RUN_ID}; starting fresh"),
        ({marker_path(RUN_ID): b'{"done": '}, f"marker of {RUN_ID} is unreadable; starting fresh"),
    ],
)
def test_an_absent_or_torn_marker_is_a_fresh_start(fake, results, files, expected):
    fake.add_repo(results, repo_type="dataset", files=files)
    notes = []
    assert read_partial(Hub(fake), RUN_ID, "c" * 64, results=results, note=notes.append) is None
    assert notes == [expected]


def fail_the_download(fake, monkeypatch, times, status):
    """Let the head lookup through, then fail the download itself."""
    lookup = fake.dataset_info

    def lookup_then_fail(*args, **kwargs):
        found = lookup(*args, **kwargs)
        fake.fail_after(times, status)
        return found

    monkeypatch.setattr(fake, "dataset_info", lookup_then_fail)


@pytest.mark.parametrize(("status", "attempts"), [(503, 3), (429, 1), (400, 1), (403, 1)])
def test_a_failed_marker_download_is_an_error_not_a_fresh_start(fake, results, monkeypatch, status, attempts):
    fake.add_repo(results, repo_type="dataset", files={marker_path(RUN_ID): marker_bytes("c" * 64)})
    fail_the_download(fake, monkeypatch, attempts, status)
    notes = []
    with pytest.raises((HfHubHTTPError, LocalEntryNotFoundError)):
        read_partial(Hub(fake, sleep=lambda seconds: None), RUN_ID, "c" * 64, results=results, note=notes.append)
    assert notes == []
    assert [method for method, _ in fake.calls].count("hf_hub_download") == attempts


def test_a_marker_download_that_times_out_is_an_error(fake, results, monkeypatch):
    fake.add_repo(results, repo_type="dataset", files={marker_path(RUN_ID): marker_bytes("c" * 64)})
    attempts = []

    def slow(*args, **kwargs):
        attempts.append(1)
        raise httpx2.ReadTimeout("no bytes for 60 s")

    monkeypatch.setattr(fake, "hf_hub_download", slow)
    with pytest.raises(httpx2.ReadTimeout):
        read_partial(Hub(fake, sleep=lambda seconds: None), RUN_ID, "c" * 64, results=results)
    assert len(attempts) == 3
