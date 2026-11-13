"""Hub access: the token, retried calls on repo references, gate classification."""

import functools
import os
import threading
import time
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import get_token
from huggingface_hub.errors import (
    DisabledRepoError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from tenacity import Retrying

from driftmap.core import ids
from driftmap.core.ids import RepoRef
from driftmap.core.retry import policy, root_cause
from driftmap.core.schemas import DriftmapError, ExitCode

Arm = Literal["gate", "not_found", "revision", "other"]


@functools.cache
def token() -> str:
    """The Hub token, read once: HF_TOKEN, else the ``hf auth login`` store."""
    # whitespace inside a token breaks the header and leaks into the error
    value = "".join((os.environ.get("HF_TOKEN") or "").split()) or get_token()
    if not value:
        raise DriftmapError(ExitCode.READINESS, "no Hugging Face token: set HF_TOKEN or run `hf auth login`")
    return value


class GateRefused(DriftmapError):
    """A gated repo this token cannot read; exit 3 from the CLI, 12 inside a job."""

    def __init__(self, repo: str, url: str, *, in_job: bool = False) -> None:
        self.repo = repo
        self.url = url
        super().__init__(
            ExitCode.INTEGRITY if in_job else ExitCode.READINESS,
            f"{repo} is gated and this token has no access; request it at {url}",
        )


def classify(exc: BaseException, repo: str, gated_repos: Collection[str]) -> Arm:
    """Sort a Hub error: a 401 or 403 on a configured gated repo is the gate."""
    status = exc.response.status_code if isinstance(exc, HfHubHTTPError) else None
    gated = repo in gated_repos
    # GatedRepoError subclasses RepositoryNotFoundError, so it goes first
    if isinstance(exc, GatedRepoError):
        return "gate" if gated else "not_found"
    if isinstance(exc, RepositoryNotFoundError):
        return "gate" if gated and status in (401, 403) else "not_found"
    if isinstance(exc, RevisionNotFoundError):
        return "revision"
    if status == 404 or isinstance(exc, DisabledRepoError):
        return "not_found"
    if status == 403 and gated:
        return "gate"
    return "other"


def gate_url(repo: str, repo_type: str | None = None) -> str:
    prefix = "" if repo_type in (None, "model") else f"{repo_type}s/"
    return f"https://huggingface.co/{prefix}{repo}"


class Hub:
    """``HfApi`` calls on repo references, retried and never two at once."""

    def __init__(
        self,
        api: Any,
        *,
        gated_repos: Iterable[str] = (),
        in_job: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = api
        self._gated = frozenset(gated_repos)
        self._in_job = in_job
        self._retry = policy(sleep)
        # uploads and downloads have already waited out 429s inside the library
        self._transfer_retry = policy(sleep, retry_throttled=False)
        self._lock = threading.Lock()

    def whoami(self) -> dict[str, Any]:
        return self._call(None, None, self._api.whoami)

    def auth_check(self, target: RepoRef | str, *, repo_type: str | None = None, write: bool = False) -> None:
        """Raises unless the token can read (or write) the repo."""
        ref = _ref(target)
        self._call(ref, repo_type, self._api.auth_check, ref.repo, repo_type=repo_type, write=write)

    def create_repo(self, target: RepoRef | str, *, exist_ok: bool = True, **kwargs: Any) -> Any:
        ref = _ref(target)
        return self._call(ref, kwargs.get("repo_type"), self._api.create_repo, ref.repo, exist_ok=exist_ok, **kwargs)

    def update_repo_settings(self, target: RepoRef | str, **kwargs: Any) -> None:
        ref = _ref(target)
        self._call(ref, kwargs.get("repo_type"), self._api.update_repo_settings, ref.repo, **kwargs)

    def upload_file(self, target: RepoRef | str, path_in_repo: str, source: str | Path | bytes, **kwargs: Any) -> Any:
        """Commit one file; an unchanged file returns the current head."""
        ref = _ref(target)
        return self._transfer(
            ref,
            kwargs.get("repo_type"),
            self._api.upload_file,
            path_or_fileobj=source,
            path_in_repo=path_in_repo,
            repo_id=ref.repo,
            revision=ref.revision,
            **kwargs,
        )

    def upload_folder(self, target: RepoRef | str, folder: str | Path, **kwargs: Any) -> Any:
        """Commit a folder; unchanged files are left out of the commit."""
        ref = _ref(target)
        return self._transfer(
            ref,
            kwargs.get("repo_type"),
            self._api.upload_folder,
            repo_id=ref.repo,
            folder_path=folder,
            revision=ref.revision,
            **kwargs,
        )

    def snapshot_download(self, target: RepoRef | str, local_dir: str | Path, **kwargs: Any) -> str:
        """Download a pinned revision into an empty ``local_dir``; returns the directory."""
        ref, directory = _fresh_download(target, local_dir)
        return self._transfer(
            ref,
            kwargs.get("repo_type"),
            self._api.snapshot_download,
            ref.repo,
            revision=ref.revision,
            local_dir=directory,
            **kwargs,
        )

    def hf_hub_download(self, target: RepoRef | str, local_dir: str | Path, **kwargs: Any) -> str:
        """Download the ``#file`` of a pinned reference into an empty ``local_dir``; returns its path."""
        ref, directory = _fresh_download(target, local_dir)
        if ref.file is None:
            raise ValueError(f"{ids.unparse(ref)} names no file")
        return self._transfer(
            ref,
            kwargs.get("repo_type"),
            self._api.hf_hub_download,
            ref.repo,
            ref.file,
            revision=ref.revision,
            local_dir=directory,
            **kwargs,
        )

    def lfs_sha256(self, target: RepoRef | str, paths: list[str], **kwargs: Any) -> dict[str, str]:
        """sha256 the Hub holds for each LFS file among ``paths``; others are left out."""
        ref = _ref(target)
        found = self._call(
            ref, kwargs.get("repo_type"), self._api.get_paths_info, ref.repo, paths, revision=ref.revision, **kwargs
        )
        return {entry.path: entry.lfs.sha256 for entry in found if getattr(entry, "lfs", None)}

    def model_info(self, target: RepoRef | str) -> Any:
        ref = _ref(target)
        return self._call(ref, "model", self._api.model_info, ref.repo, revision=ref.revision)

    def dataset_info(self, target: RepoRef | str) -> Any:
        ref = _ref(target)
        return self._call(ref, "dataset", self._api.dataset_info, ref.repo, revision=ref.revision)

    def create_collection(self, title: str, *, exists_ok: bool = True, **kwargs: Any) -> Any:
        return self._call(None, None, self._api.create_collection, title, exists_ok=exists_ok, **kwargs)

    def add_collection_item(
        self, slug: str, target: RepoRef | str, item_type: str, *, exists_ok: bool = True, **kwargs: Any
    ) -> Any:
        ref = _ref(target)
        return self._call(
            ref, item_type, self._api.add_collection_item, slug, ref.repo, item_type, exists_ok=exists_ok, **kwargs
        )

    def _call(
        self, ref: RepoRef | None, repo_type: str | None, fn: Callable[..., Any], /, *args: Any, **kw: Any
    ) -> Any:
        return self._run(self._retry, ref, repo_type, fn, *args, **kw)

    def _transfer(self, ref: RepoRef, repo_type: str | None, fn: Callable[..., Any], /, *args: Any, **kw: Any) -> Any:
        return self._run(self._transfer_retry, ref, repo_type, fn, *args, **kw)

    def _run(
        self,
        retry: Retrying,
        ref: RepoRef | None,
        repo_type: str | None,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kw: Any,
    ) -> Any:
        with self._lock:
            try:
                return retry(fn, *args, **kw)
            except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
                if ref is not None and classify(root_cause(exc), ref.repo, self._gated) == "gate":
                    raise GateRefused(ref.repo, gate_url(ref.repo, repo_type), in_job=self._in_job) from exc
                raise


def _ref(target: RepoRef | str) -> RepoRef:
    return ids.parse(target if isinstance(target, str) else ids.unparse(target))


def _fresh_download(target: RepoRef | str, local_dir: str | Path) -> tuple[RepoRef, Path]:
    ref = _ref(target)
    if ref.revision is None:
        raise DriftmapError(ExitCode.USAGE, f"{ids.unparse(ref)} must pin a revision to be downloaded")
    # the library expands and resolves the path the same way
    directory = Path(local_dir).expanduser().resolve()
    # offline, the library returns a non-empty directory as the finished download
    if directory.is_dir() and any(directory.iterdir()):
        raise DriftmapError(ExitCode.USAGE, f"{directory} is not empty; download into a new directory")
    return ref, directory
