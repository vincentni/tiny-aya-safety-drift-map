"""Files of a public dataset at one revision: downloaded into a folder the caller names, checked by sha256."""

import hashlib
import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from driftmap.core.retry import policy

TIMEOUT_S = 60


def download(url: str) -> bytes:
    """The bytes behind ``url``; a failed or cut connection is tried three times in all, an HTTP error is not."""

    def get() -> bytes:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as reply:
            return reply.read()

    def worth_another(error: BaseException) -> bool:
        cut = isinstance(error, (OSError, http.client.HTTPException))
        return cut and not isinstance(error, urllib.error.HTTPError)

    return policy(time.sleep, also=worth_another)(get)


def fetch(
    directory: Path,
    files: Mapping[str, str],
    url: Callable[[str], str],
    source: str,
    get: Callable[[str], bytes] = download,
) -> dict[str, str]:
    """The sha256 of every file of ``files`` under ``directory``, by its path there; a missing one is downloaded.

    ``files`` holds the recorded sha256 per path, ``url`` gives the address of a path and ``source`` names the
    dataset and its revision. A file that does not have the recorded sha256 stops the run; a downloaded one is
    then not written.
    """
    found = {}
    for name, recorded in files.items():
        path = directory / name
        content = path.read_bytes() if path.is_file() else get(url(name))
        digest = hashlib.sha256(content).hexdigest()
        if digest != recorded:
            raise SystemExit(f"{name}: sha256 {digest} is not the one recorded for {source}")
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        found[name] = digest
    return found
