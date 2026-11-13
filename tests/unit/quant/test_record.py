import hashlib
import json
from types import SimpleNamespace

import pytest

from driftmap import quant
from driftmap.core.ids import RepoRef
from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.quant import record
from driftmap.quant.record import GgufHashMismatch
from tests.fakes.settings import BASE, settings_toml

OWNER = BASE["owner"]
RELEASE = BASE["llamacpp_release"]
SOURCE = RepoRef(f"{OWNER}/{BASE['repos']['repaired']}", "b" * 40)
DATA = b"GGUF\x03 quantized weights"


def test_the_record_names_the_file_its_source_and_the_release(tmp_path):
    """gguf.json names the release tarball and its hash, the file's hash and size, and the source (TS-QNT-3)."""
    settings = load_settings(settings_toml(tmp_path))
    gguf = tmp_path / quant.GGUF_FILE
    gguf.write_bytes(DATA)

    written = record.write(gguf, SOURCE, settings.llamacpp_release, tmp_path / quant.RECORD_FILE)

    assert json.loads((tmp_path / "gguf.json").read_text(encoding="utf-8")) == {
        "schema": "driftmap.gguf/1",
        "llamacpp_tag": "b11960",
        "tarball": RELEASE["source_tarball"],
        "tarball_sha256": RELEASE["sha256"]["source_tarball"],
        "quant": "q4_k_m",
        "sha256": hashlib.sha256(DATA).hexdigest(),
        "bytes": len(DATA),
        "source_repo": SOURCE.repo,
        "source_revision": "b" * 40,
    }
    assert record.read(tmp_path / "gguf.json") == written
    record.verify(written, gguf)


def test_another_file_than_the_recorded_one_is_refused(tmp_path):
    settings = load_settings(settings_toml(tmp_path))
    gguf = tmp_path / quant.GGUF_FILE
    gguf.write_bytes(DATA)
    written = record.write(gguf, SOURCE, settings.llamacpp_release, tmp_path / quant.RECORD_FILE)

    for other in (DATA[:-1] + b"S", DATA + b"!"):
        gguf.write_bytes(other)
        with pytest.raises(GgufHashMismatch, match="gguf.json says") as caught:
            record.verify(written, gguf)
        assert caught.value.exit_code == 12


def test_an_unreadable_record_is_refused(tmp_path):
    torn = tmp_path / "gguf.json"
    torn.write_text('{"schema": "driftmap.gguf/1", "quant": "q8_0"}', encoding="utf-8")
    for path in (torn, tmp_path / "absent.json"):
        with pytest.raises(DriftmapError, match="not a readable GGUF record") as caught:
            record.read(path)
        assert caught.value.exit_code == 12


def test_smoke_runs_get_repos_of_their_own(tmp_path):
    settings = load_settings(settings_toml(tmp_path))
    assert quant.repo_names(settings, "full") == (
        f"{OWNER}/tiny-aya-global-repaired",
        f"{OWNER}/tiny-aya-global-repaired-GGUF",
    )
    assert quant.repo_names(settings, "smoke") == (
        f"{OWNER}/tiny-aya-global-repaired-smoke",
        f"{OWNER}/tiny-aya-global-repaired-GGUF-smoke",
    )


def test_work_that_does_not_fit_the_disk_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(quant.shutil, "disk_usage", lambda directory: SimpleNamespace(free=3 * quant.GIB))
    quant.require_disk(tmp_path, 3 * quant.GIB, "the conversion")
    with pytest.raises(DriftmapError, match=r"the conversion needs 3\.5 GiB of disk, 3\.0 GiB are free") as caught:
        quant.require_disk(tmp_path, 3 * quant.GIB + quant.GIB // 2, "the conversion")
    assert caught.value.exit_code == 12
