"""The BBQ files: downloaded at one commit and checked by sha256 before they are read."""

import hashlib
import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from driftmap.attribution import bbq, build

AGE = "data/Age.jsonl"


@pytest.fixture
def files(monkeypatch):
    """Two small files in place of BBQ, their sha256 recorded."""
    content = {AGE: b'{"example_id": 0}\n', bbq.METADATA: b"category,example_id\n"}
    monkeypatch.setattr(bbq, "FILES", {name: hashlib.sha256(data).hexdigest() for name, data in content.items()})
    return content


def served(content, asked):
    def get(url):
        asked.append(url)
        return next(data for name, data in content.items() if url.endswith(name))

    return get


def test_a_missing_file_is_downloaded_at_the_recorded_commit_and_a_present_one_is_not(tmp_path, files):
    (tmp_path / "data").mkdir()
    (tmp_path / AGE).write_bytes(files[AGE])
    asked = []

    found = bbq.fetch(tmp_path, served(files, asked))

    assert asked == [f"https://raw.githubusercontent.com/nyu-mll/BBQ/{bbq.COMMIT}/{bbq.METADATA}"]
    assert found == bbq.FILES and (tmp_path / bbq.METADATA).read_bytes() == files[bbq.METADATA]
    again = []
    assert bbq.fetch(tmp_path, served(files, again)) == found and again == []


def test_a_download_with_another_sha256_stops_the_run_and_is_not_written(tmp_path, files):
    with pytest.raises(SystemExit, match=f"{AGE}: sha256 .* is not the one recorded"):
        bbq.fetch(tmp_path, lambda url: b"something else")

    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_a_file_on_disk_with_another_sha256_stops_the_run_and_is_left_as_it_is(tmp_path, files):
    (tmp_path / "data").mkdir()
    (tmp_path / AGE).write_bytes(b"edited")

    with pytest.raises(SystemExit, match="sha256"):
        bbq.fetch(tmp_path, served(files, []))

    assert (tmp_path / AGE).read_bytes() == b"edited"


def test_the_recorded_files_are_the_categories_the_build_reads_and_the_metadata():
    categories = {Path(name).stem for name in bbq.FILES if name.startswith("data/")}

    assert categories == set(build.HELDOUT_CATEGORIES + build.TRAINED_CATEGORIES)
    assert bbq.METADATA in bbq.FILES and len(bbq.COMMIT) == 40
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in bbq.FILES.values())


class Reply:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def read(self):
        return self.content


def test_a_cut_connection_is_tried_again_and_an_http_error_is_not(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    replies = [http.client.IncompleteRead(b"half"), urllib.error.URLError("reset"), Reply(b"whole")]
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert bbq.download("https://example.org/file") == b"whole" and len(calls) == 3

    replies[:] = [urllib.error.HTTPError("https://example.org/file", 404, "Not Found", None, None), Reply(b"whole")]
    with pytest.raises(urllib.error.HTTPError):
        bbq.download("https://example.org/file")
    assert len(replies) == 1
