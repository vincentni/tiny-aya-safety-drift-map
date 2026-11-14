"""Synthetic llama.cpp release archives and the opener that serves them."""

import hashlib
import io
import tarfile
import urllib.error
from email.message import Message
from typing import Any, NamedTuple

from tests.fakes.settings import BASE

RELEASE = BASE["llamacpp_release"]
TAG = RELEASE["tag"]
ASSETS = ("source_tarball", "cpu_asset", "cuda_asset")
SOURCE_ROOT = f"llama.cpp-{TAG}"
BINARY_ROOT = f"llama-{TAG}"
ARCHITECTURE = "Cohere2ForCausalLM"
SUPPORTED = f"TEXT models:\n  - CohereForCausalLM\n  - {ARCHITECTURE}\nMMPROJ models:\n"


def tarball(files: dict[str, bytes], *, mode: int = 0o755) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def archives() -> dict[str, bytes]:
    """The three release files, each a small archive with the real layout."""
    return {
        "source_tarball": tarball(
            {f"{SOURCE_ROOT}/convert_hf_to_gguf.py": b"# converter\n", f"{SOURCE_ROOT}/gguf-py/gguf/__init__.py": b""},
            mode=0o644,
        ),
        "cpu_asset": tarball({f"{BINARY_ROOT}/llama-quantize": b"quantize binary"}),
        "cuda_asset": tarball({f"{BINARY_ROOT}/llama-server": b"server binary", f"{BINARY_ROOT}/libggml-cuda.so": b""}),
    }


def pinned(files: dict[str, bytes]) -> dict[str, Any]:
    """Settings override that pins the sha256 of ``files``."""
    return {"llamacpp_release": {"sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}}


def pages(files: dict[str, bytes]) -> dict[str, Any]:
    return {RELEASE[name]: data for name, data in files.items()}


class Cut(NamedTuple):
    """A download the connection drops: ``data`` arrives of the ``length`` bytes the server announced."""

    data: bytes
    length: int


class _Response(io.BytesIO):
    def __init__(self, data: bytes, length: int) -> None:
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Length"] = str(length)


class FakeOpener:
    """Stand-in for ``urllib.request.urlopen``: bytes are served, an int is an HTTP status, an exception is raised."""

    def __init__(self, pages: dict[str, Any]) -> None:
        # a list is replayed in order, its last entry for good
        self.pages = dict(pages)
        self.calls: list[tuple[str, float | None]] = []

    def __call__(self, url: str, timeout: float | None = None) -> io.BytesIO:
        self.calls.append((url, timeout))
        assert url in self.pages, f"unscripted download: {url}"
        page = self.pages[url]
        if isinstance(page, list):
            page = page.pop(0) if len(page) > 1 else page[0]
        if isinstance(page, int):
            raise urllib.error.HTTPError(url, page, "scripted", Message(), io.BytesIO())
        if isinstance(page, BaseException):
            raise page
        return _Response(*page) if isinstance(page, Cut) else _Response(page, len(page))
