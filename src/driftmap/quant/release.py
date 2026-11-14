"""The pinned llama.cpp release: fetched with urllib, checked by sha256, then unpacked."""

import hashlib
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import LlamacppRelease

ASSETS = ("source_tarball", "cpu_asset", "cuda_asset")
PROJECT = "https://github.com/ggml-org/llama.cpp/"
ATTEMPTS = 3
TIMEOUT_S = 120
CHUNK = 2**20


class ReleaseHashMismatch(DriftmapError):
    """A downloaded release file is not the pinned one."""

    exit_code_default = ExitCode.INTEGRITY


def _refuse(message: str) -> DriftmapError:
    return DriftmapError(ExitCode.INTEGRITY, message)


def _transient(error: BaseException) -> bool:
    # a 4xx is final: a rate limit is never asked again
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    return isinstance(error, OSError)


def _download(url: str, target: Path, opener: Callable[..., Any]) -> str:
    """Stream ``url`` into ``target``; returns the sha256 of what arrived. OSError when the transfer was cut short."""
    digest, received = hashlib.sha256(), 0
    with opener(url, timeout=TIMEOUT_S) as response, target.open("wb") as out:
        announced = response.headers.get("Content-Length")
        while chunk := response.read(CHUNK):
            digest.update(chunk)
            out.write(chunk)
            received += len(chunk)
    # a dropped connection ends a chunked read without an error
    if announced is not None and announced.isdigit() and received < int(announced):
        raise OSError(f"{received} of {announced} bytes arrived")
    return digest.hexdigest()


def _unpack(archive: Path, target: Path) -> Path:
    """Unpack into ``target``; returns the one directory the archive holds."""
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            # refuses members that leave the directory
            bundle.extractall(target, filter="data")
    except (tarfile.TarError, OSError) as error:
        shutil.rmtree(target, ignore_errors=True)
        raise _refuse(f"{archive.name} cannot be unpacked: {error}") from error
    roots = [path for path in target.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise _refuse(f"{archive.name} holds {len(roots)} top-level directories, expected one")
    return roots[0]


def fetch_release(
    release: LlamacppRelease,
    dest: str | Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
    *,
    assets: Sequence[str] = ASSETS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Path]:
    """Download ``assets``, verify every sha256, then unpack; returns each root directory. Raises DriftmapError."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in assets:
        url = getattr(release, name)
        # a URL without the tag could serve another build tomorrow
        if not url.startswith(PROJECT) or f"/{release.tag}" not in url:
            raise _refuse(f"{name} is not a file of llama.cpp release {release.tag}: {url}")
    retry = Retrying(
        stop=stop_after_attempt(ATTEMPTS),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception(_transient),
        reraise=True,
        sleep=sleep,
    )
    archives = {}
    for name in assets:
        archive = dest / f"{name}.tar.gz"
        try:
            found = retry(_download, getattr(release, name), archive, opener)
        except OSError as error:
            raise _refuse(f"cannot download {name} of llama.cpp {release.tag}: {error}") from error
        expected = getattr(release.sha256, name)
        if found != expected:
            archive.unlink()
            raise ReleaseHashMismatch(
                f"{name} of llama.cpp {release.tag} hashes to {found[:12]}, not the pinned {expected[:12]}; "
                "nothing was unpacked"
            )
        archives[name] = archive
    return {name: _unpack(archive, dest / name) for name, archive in archives.items()}


def binary(root: Path, name: str) -> Path:
    """The named program of an unpacked release file, executable; DriftmapError (exit 12) when it is absent."""
    path = root / name
    if not path.is_file():
        raise _refuse(f"the llama.cpp release file under {root.name} holds no {name}")
    path.chmod(path.stat().st_mode | 0o100)
    return path
