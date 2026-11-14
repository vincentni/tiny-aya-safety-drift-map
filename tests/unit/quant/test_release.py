import inspect
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from driftmap.core.schemas import DriftmapError
from driftmap.core.settings import load_settings
from driftmap.quant import release
from driftmap.quant.release import ReleaseHashMismatch, binary, fetch_release
from tests.fakes.release import (
    ASSETS,
    BINARY_ROOT,
    RELEASE,
    SOURCE_ROOT,
    Cut,
    FakeOpener,
    archives,
    pages,
    pinned,
    tarball,
)
from tests.fakes.settings import settings_toml

PACKAGE = Path(release.__file__).parent


def pinned_release(tmp_path, files, **changes):
    override = pinned(files)
    override["llamacpp_release"] |= changes
    return load_settings(settings_toml(tmp_path, **override)).llamacpp_release


def no_sleep(seconds):
    pass


def test_a_wrong_sha256_stops_before_anything_is_unpacked(tmp_path):
    """Each file comes through urllib and is hashed; one wrong hash exits 12 with nothing unpacked (TS-QNT-1)."""
    files = archives()
    served = files | {"cpu_asset": tarball({f"{BINARY_ROOT}/llama-quantize": b"another build"})}
    opener = FakeOpener(pages(served))
    dest = tmp_path / "release"

    with pytest.raises(ReleaseHashMismatch) as caught:
        fetch_release(pinned_release(tmp_path, files), dest, opener)

    assert caught.value.exit_code == 12 and "cpu_asset" in str(caught.value)
    assert [url for url, _ in opener.calls] == [RELEASE["source_tarball"], RELEASE["cpu_asset"]]
    assert all(timeout for _, timeout in opener.calls)
    # the verified source archive waits packed; the wrong file is gone
    assert sorted(path.name for path in dest.iterdir()) == ["source_tarball.tar.gz"]
    assert inspect.signature(fetch_release).parameters["opener"].default is urllib.request.urlopen
    sources = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py"))
    assert "urllib.request" in sources and not re.search(r"curl|\brequests\b", sources)


def test_verified_files_are_unpacked_to_their_roots(tmp_path):
    files = archives()
    opener = FakeOpener(pages(files))

    roots = fetch_release(pinned_release(tmp_path, files), tmp_path / "release", opener)

    assert list(roots) == list(ASSETS) and len(opener.calls) == 3
    assert (roots["source_tarball"].name, roots["cpu_asset"].name) == (SOURCE_ROOT, BINARY_ROOT)
    assert (roots["source_tarball"] / "convert_hf_to_gguf.py").read_bytes() == b"# converter\n"
    assert (roots["source_tarball"] / "gguf-py" / "gguf" / "__init__.py").is_file()
    quantizer = binary(roots["cpu_asset"], "llama-quantize")
    assert quantizer.stat().st_mode & 0o100 and quantizer.read_bytes() == b"quantize binary"
    with pytest.raises(DriftmapError, match="holds no llama-server") as caught:
        binary(roots["cpu_asset"], "llama-server")
    assert caught.value.exit_code == 12


def test_only_the_named_files_are_downloaded(tmp_path):
    files = archives()
    opener = FakeOpener(pages(files))
    roots = fetch_release(pinned_release(tmp_path, files), tmp_path / "release", opener, assets=("cuda_asset",))
    assert [url for url, _ in opener.calls] == [RELEASE["cuda_asset"]]
    assert sorted(path.name for path in roots["cuda_asset"].iterdir()) == ["libggml-cuda.so", "llama-server"]


def test_a_failed_download_is_retried_three_times_and_a_refusal_never(tmp_path):
    files = archives()
    down = urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
    flaky = FakeOpener(pages(files) | {RELEASE["cuda_asset"]: [down, 503, files["cuda_asset"]]})
    roots = fetch_release(
        pinned_release(tmp_path, files), tmp_path / "flaky", flaky, assets=("cuda_asset",), sleep=no_sleep
    )
    assert len(flaky.calls) == 3 and (roots["cuda_asset"] / "llama-server").is_file()

    dead = FakeOpener({RELEASE["cuda_asset"]: down})
    with pytest.raises(DriftmapError, match="cannot download cuda_asset") as caught:
        fetch_release(pinned_release(tmp_path, files), tmp_path / "dead", dead, assets=("cuda_asset",), sleep=no_sleep)
    assert caught.value.exit_code == 12 and len(dead.calls) == 3

    # a rate limit or a missing file is an answer
    for status in (403, 404, 429):
        refused = FakeOpener({RELEASE["cuda_asset"]: status})
        with pytest.raises(DriftmapError, match="cannot download"):
            fetch_release(
                pinned_release(tmp_path, files),
                tmp_path / f"refused-{status}",
                refused,
                assets=("cuda_asset",),
                sleep=no_sleep,
            )
        assert len(refused.calls) == 1


def test_a_download_cut_short_is_retried_and_never_read_as_another_file(tmp_path):
    """Fewer bytes than the server announced is a failed transfer, not a file with the wrong hash."""
    files = archives()
    whole, url = files["cuda_asset"], RELEASE["cuda_asset"]
    half = Cut(whole[: len(whole) // 2], len(whole))
    once = FakeOpener({url: [half, whole]})
    release = pinned_release(tmp_path, files)
    roots = fetch_release(release, tmp_path / "once", once, assets=("cuda_asset",), sleep=no_sleep)
    assert len(once.calls) == 2 and (roots["cuda_asset"] / "llama-server").is_file()

    always = FakeOpener({url: half})
    short = f"cannot download cuda_asset of llama.cpp {RELEASE['tag']}: {len(half.data)} of {len(whole)} bytes arrived"
    with pytest.raises(DriftmapError, match=re.escape(short)) as caught:
        fetch_release(release, tmp_path / "cut", always, assets=("cuda_asset",), sleep=no_sleep)
    assert not isinstance(caught.value, ReleaseHashMismatch) and len(always.calls) == 3


def test_a_url_outside_the_pinned_tag_is_refused_before_any_download(tmp_path):
    files = archives()
    moving = "https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-bin-ubuntu-x64.tar.gz"
    elsewhere = RELEASE["source_tarball"].replace("ggml-org", "someone-else")
    for changes in ({"cpu_asset": moving}, {"source_tarball": elsewhere}):
        opener = FakeOpener({})
        with pytest.raises(DriftmapError, match="is not a file of llama.cpp release b11960") as caught:
            fetch_release(pinned_release(tmp_path, files, **changes), tmp_path / "release", opener)
        assert caught.value.exit_code == 12 and opener.calls == []


def test_an_archive_that_writes_outside_its_directory_is_refused(tmp_path):
    inside = f"{BINARY_ROOT}/llama-server"
    files = archives() | {"cuda_asset": tarball({inside: b"server", f"{BINARY_ROOT}/../../escaped": b"x"})}
    dest = tmp_path / "release"
    with pytest.raises(DriftmapError, match="cannot be unpacked") as caught:
        fetch_release(pinned_release(tmp_path, files), dest, FakeOpener(pages(files)), assets=("cuda_asset",))
    assert caught.value.exit_code == 12
    assert not (dest / "escaped").exists() and not (dest / "cuda_asset").exists()
