import hashlib
import json
import threading
from collections import deque
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import httpx2
from huggingface_hub import CommitInfo
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import Collection, DatasetInfo, GitCommitInfo, ModelInfo, RepoFile, RepoFolder, RepoUrl
from huggingface_hub.utils import filter_repo_objects, hf_raise_for_status

ENDPOINT = "https://huggingface.co"
LFS_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".tar.gz")


class FakeHub:
    """In-memory Hub that raises the real huggingface_hub errors."""

    def __init__(self, root: str | Path, *, user: str = "vincentnkq") -> None:
        self.root = Path(root)
        self.user = user
        # repo_id -> revision -> path -> bytes; "main" is the live tree
        self.repos: dict[str, dict[str, dict[str, bytes]]] = {}
        self.repo_types: dict[str, str] = {}
        self.heads: dict[str, str] = {}
        self.gated: set[str] = set()
        self.private: set[str] = set()
        self.authorized: set[str] = set()
        self.settings: dict[str, dict[str, Any]] = {}
        # repos whose settings calls are accepted and change nothing
        self.frozen: set[str] = set()
        self.collections: dict[str, dict[str, Any]] = {}
        # old name -> new name of a moved repo: the Hub answers for the old name with the repo under its new one
        self.redirects: dict[str, str] = {}
        # commit id -> the message it was made with
        self.messages: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.threads: set[int] = set()
        self.max_in_flight = 0
        self._failures: deque[tuple[int, dict[str, str]]] = deque()
        self._in_flight = 0
        self._lock = threading.Lock()
        self._commits = 0

    def add_repo(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        gated: bool = False,
        private: bool = False,
        files: dict[str, bytes] | None = None,
    ) -> "FakeHub":
        self.repos[repo_id] = {"main": dict(files or {})}
        self.repo_types[repo_id] = repo_type
        if gated:
            self.gated.add(repo_id)
        if private:
            self.private.add(repo_id)
        self._commit(repo_id, "initial")
        return self

    def authorize(self, repo_id: str) -> None:
        self.authorized.add(repo_id)

    def fail_after(
        self,
        n: int,
        status: int,
        *,
        ratelimit: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
    ) -> None:
        built = dict(headers or {})
        if ratelimit is not None:
            built["ratelimit"] = ratelimit
        if error_code is not None:
            built["X-Error-Code"] = error_code
        self._failures.extend([(status, built)] * n)

    def whoami(self, token: str | bool | None = None) -> dict[str, Any]:
        with self._op("whoami", f"{ENDPOINT}/api/whoami-v2"):
            return {"name": self.user, "type": "user", "auth": {"accessToken": {"role": "fineGrained"}}}

    def auth_check(
        self, repo_id: str, *, repo_type: str | None = None, token: str | bool | None = None, write: bool = False
    ) -> None:
        with self._op(
            "auth_check", self._repo_url(repo_id, repo_type), repo_id=repo_id, repo_type=repo_type, write=write
        ):
            self._check_access(repo_id, repo_type)

    def repo_exists(self, repo_id: str, *, repo_type: str | None = None, token: str | bool | None = None) -> bool:
        with self._op("repo_exists", self._repo_url(repo_id, repo_type), repo_id=repo_id, repo_type=repo_type):
            # like HfApi.repo_exists: gated counts as existing, unreadable or missing does not
            try:
                self._check_access(repo_id, repo_type)
            except GatedRepoError:
                return True
            except RepositoryNotFoundError:
                return False
            return True

    def file_exists(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> bool:
        with self._op(
            "file_exists",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        ):
            return filename in self._tree(repo_id, repo_type, revision)

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        private: bool = False,
        exist_ok: bool = False,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> RepoUrl:
        url = self._repo_url(repo_id, repo_type)
        with self._op(
            "create_repo", url, repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok, **kwargs
        ):
            if repo_id in self.repos:
                if not exist_ok:
                    self._raise(409, url)
            else:
                # a new repo takes the name for itself
                self.redirects.pop(repo_id, None)
                self.add_repo(repo_id, repo_type=repo_type or "model", private=private)
            self.authorized.add(repo_id)
            return RepoUrl(f"{ENDPOINT}/{_prefix(self.repo_types[repo_id])}{repo_id}")

    def update_repo_settings(
        self,
        repo_id: str,
        *,
        gated: str | bool | None = None,
        private: bool | None = None,
        visibility: str | None = None,
        token: str | bool | None = None,
        repo_type: str | None = None,
    ) -> None:
        # like the library: one way to name the visibility, and something to change
        if private is not None and visibility is not None:
            raise ValueError("`private` and `visibility` cannot be passed together.")
        if visibility is not None:
            private = visibility != "public"
        if gated is None and private is None:
            raise ValueError("At least one setting must be updated.")
        with self._op(
            "update_repo_settings",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            gated=gated,
            private=private,
            repo_type=repo_type,
        ):
            repo_id = self.redirects.get(repo_id, repo_id)
            self._check_access(repo_id, repo_type)
            if repo_id in self.frozen:
                return
            self.settings.setdefault(repo_id, {}).update(
                {k: v for k, v in {"gated": gated, "private": private}.items() if v is not None}
            )
            if gated is not None:
                (self.gated.add if gated else self.gated.discard)(repo_id)
            if private is not None:
                (self.private.add if private else self.private.discard)(repo_id)

    def delete_repo(
        self,
        repo_id: str,
        *,
        token: str | bool | None = None,
        repo_type: str | None = None,
        missing_ok: bool = False,
    ) -> None:
        url = f"{ENDPOINT}/api/repos/delete"
        with self._op("delete_repo", url, repo_id=repo_id, repo_type=repo_type, missing_ok=missing_ok):
            if self.repo_types.get(repo_id) != (repo_type or "model"):
                if missing_ok:
                    return
                self._raise(404, url, {"X-Error-Code": "RepoNotFound"})
            for held in (self.repos, self.repo_types, self.heads, self.settings):
                held.pop(repo_id, None)
            for flagged in (self.gated, self.private, self.authorized):
                flagged.discard(repo_id)

    def move_repo(
        self, from_id: str, to_id: str, *, repo_type: str | None = None, token: str | bool | None = None
    ) -> None:
        url = f"{ENDPOINT}/api/repos/move"
        with self._op("move_repo", url, repo_id=from_id, to_id=to_id, repo_type=repo_type):
            if self.repo_types.get(from_id) != (repo_type or "model"):
                self._raise(404, url, {"X-Error-Code": "RepoNotFound"})
            if to_id in self.repos:
                self._raise(409, url)
            for held in (self.repos, self.repo_types, self.heads, self.settings):
                if from_id in held:
                    held[to_id] = held.pop(from_id)
            for flagged in (self.gated, self.private, self.authorized, self.frozen):
                if from_id in flagged:
                    flagged.discard(from_id)
                    flagged.add(to_id)
            self.redirects[from_id] = to_id

    def repo_info(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> ModelInfo | DatasetInfo:
        sized = bool(kwargs.get("files_metadata"))
        with self._op(
            "repo_info", self._repo_url(repo_id, repo_type), repo_id=repo_id, repo_type=repo_type, revision=revision
        ):
            repo_id = self.redirects.get(repo_id, repo_id)
            tree = self._tree(repo_id, repo_type, revision)
            payload = {
                "id": repo_id,
                "sha": self._sha(repo_id, revision),
                "private": repo_id in self.private,
                "gated": "auto" if repo_id in self.gated else False,
                # like the library: file sizes only when files_metadata is asked for
                "siblings": [
                    {"rfilename": path, **({"size": len(data)} if sized else {})} for path, data in sorted(tree.items())
                ],
            }
            return DatasetInfo(**payload) if (repo_type or "model") == "dataset" else ModelInfo(**payload)

    def model_info(
        self, repo_id: str, *, revision: str | None = None, token: str | bool | None = None, **kwargs: Any
    ) -> ModelInfo:
        return self.repo_info(repo_id, repo_type="model", revision=revision, **kwargs)

    def dataset_info(
        self, repo_id: str, *, revision: str | None = None, token: str | bool | None = None, **kwargs: Any
    ) -> DatasetInfo:
        return self.repo_info(repo_id, repo_type="dataset", revision=revision)

    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
        token: str | bool | None = None,
    ) -> list[str]:
        with self._op(
            "list_repo_files",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        ):
            return sorted(self._tree(repo_id, repo_type, revision))

    def list_repo_commits(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
        formatted: bool = False,
    ) -> list[GitCommitInfo]:
        with self._op(
            "list_repo_commits",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        ):
            self._tree(repo_id, repo_type, revision)
            # commits are kept in the order they were made; the Hub lists the newest first
            made = [oid for oid in self.repos[repo_id] if oid != "main"]
            upto = made[: made.index(self._sha(repo_id, revision)) + 1]
            return [
                GitCommitInfo(
                    commit_id=oid,
                    authors=[self.user],
                    created_at=datetime(2026, 11, 2, 10, 0, tzinfo=UTC),
                    title=self.messages[oid].partition("\n")[0],
                    message="",
                    formatted_title=None,
                    formatted_message=None,
                )
                for oid in reversed(upto)
            ]

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        expand: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
        token: str | bool | None = None,
    ) -> list[RepoFile | RepoFolder]:
        with self._op(
            "list_repo_tree",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            recursive=recursive,
            repo_type=repo_type,
            revision=revision,
        ):
            tree = self._tree(repo_id, repo_type, revision)
            prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ""
            inside = {path: data for path, data in tree.items() if path.startswith(prefix)}
            folders = {path.rsplit("/", depth)[0] for path in inside for depth in range(1, path.count("/") + 1)}
            folders = {folder for folder in folders if len(folder) >= len(prefix)}
            if not recursive:
                level = prefix.count("/")
                inside = {path: data for path, data in inside.items() if path.count("/") == level}
                folders = {folder for folder in folders if folder.count("/") == level}
            # like the Hub: folders are listed beside the files
            entries = [RepoFolder(path=folder, oid=hashlib.sha1(folder.encode()).hexdigest()) for folder in folders]
            entries += [self._repo_file(path, data) for path, data in inside.items()]
            return sorted(entries, key=lambda entry: entry.path)

    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str] | str,
        *,
        expand: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
        token: str | bool | None = None,
    ) -> list[RepoFile]:
        paths = [paths] if isinstance(paths, str) else list(paths)
        with self._op(
            "get_paths_info",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            paths=paths,
            repo_type=repo_type,
            revision=revision,
        ):
            tree = self._tree(repo_id, repo_type, revision)
            return [self._repo_file(path, tree[path]) for path in paths if path in tree]

    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> CommitInfo:
        with self._op(
            "upload_file",
            self._repo_url(repo_id, repo_type),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            **kwargs,
        ):
            self._writable(repo_id, repo_type, revision)
            update = {path_in_repo: _read(path_or_fileobj)}
            return self._commit(repo_id, commit_message or f"Upload {path_in_repo}", update)

    def create_commit(
        self,
        repo_id: str,
        operations: Any,
        *,
        commit_message: str,
        repo_type: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> CommitInfo:
        operations = list(operations)
        paths = [operation.path_in_repo for operation in operations]
        with self._op(
            "create_commit",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            paths=paths,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            **kwargs,
        ):
            self._writable(repo_id, repo_type, revision)
            update = {operation.path_in_repo: _read(operation.path_or_fileobj) for operation in operations}
            return self._commit(repo_id, commit_message, update)

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = None,
        repo_type: str | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        parent_commit: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> CommitInfo:
        with self._op(
            "upload_folder",
            self._repo_url(repo_id, repo_type),
            repo_id=repo_id,
            folder_path=str(folder_path),
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            parent_commit=parent_commit,
            **kwargs,
        ):
            repo_id = self.redirects.get(repo_id, repo_id)
            self._writable(repo_id, repo_type, revision)
            # like the Hub: a commit on another parent than the head is refused
            if parent_commit is not None and parent_commit != self.heads[repo_id]:
                self._raise(412, self._repo_url(repo_id, repo_type))
            folder = Path(folder_path)
            relative = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
            prefix = (path_in_repo or "").strip("/")
            update = {
                f"{prefix}/{rel}" if prefix else rel: (folder / rel).read_bytes()
                for rel in filter_repo_objects(relative, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
            }
            # like the library: patterns are matched under path_in_repo, .gitattributes stays, an added file is kept
            held = {
                path[len(prefix) + 1 if prefix else 0 :]: path
                for path in self.repos[repo_id]["main"]
                if not prefix or path.startswith(f"{prefix}/")
            }
            doomed = [] if delete_patterns is None else filter_repo_objects(held, allow_patterns=delete_patterns)
            delete = [held[rel] for rel in doomed if held[rel] != ".gitattributes" and held[rel] not in update]
            return self._commit(repo_id, commit_message or f"Upload folder {prefix or '/'}", update, delete)

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        local_dir: str | Path | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> str:
        target = Path(local_dir) if local_dir else self.root / repo_id
        # like the library: the lock directory appears even when the call fails
        (target / ".cache" / "huggingface" / "download").mkdir(parents=True, exist_ok=True)
        try:
            with self._op(
                "hf_hub_download",
                self._repo_url(repo_id, repo_type),
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                local_dir=_opt_str(local_dir),
            ):
                tree = self._tree(repo_id, repo_type, revision)
                if filename not in tree:
                    self._raise(
                        404,
                        f"{ENDPOINT}/{_prefix(repo_type)}{repo_id}/resolve/{revision or 'main'}/{filename}",
                        {"X-Error-Code": "EntryNotFound"},
                    )
                return str(self._write(target, filename, tree[filename]))
        except (RevisionNotFoundError, RemoteEntryNotFoundError):
            raise
        except HfHubHTTPError as error:
            # like the library: Hub unreachable, a file on disk is returned
            if (target / filename).is_file():
                return str(target / filename)
            _raise_unreachable(error)

    def snapshot_download(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        local_dir: str | Path | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> str:
        target = Path(local_dir) if local_dir else self.root / repo_id
        try:
            with self._op(
                "snapshot_download",
                self._repo_url(repo_id, repo_type),
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=_opt_str(local_dir),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            ):
                tree = self._tree(repo_id, repo_type, revision)
                for path in filter_repo_objects(
                    sorted(tree), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
                ):
                    self._write(target, path, tree[path])
                target.mkdir(parents=True, exist_ok=True)
                return str(target)
        except RevisionNotFoundError:
            raise
        except HfHubHTTPError as error:
            # like the library: with the Hub unreachable a non-empty directory is returned
            if target.is_dir() and any(target.iterdir()):
                return str(target)
            _raise_unreachable(error)

    def create_collection(
        self,
        title: str,
        *,
        namespace: str | None = None,
        description: str | None = None,
        private: bool = False,
        exists_ok: bool = False,
        token: str | bool | None = None,
    ) -> Collection:
        owner = namespace or self.user
        slug = f"{owner}/{title.lower().replace(' ', '-')}-{hashlib.sha1(title.encode()).hexdigest()[:24]}"
        with self._op(
            "create_collection",
            f"{ENDPOINT}/api/collections",
            title=title,
            namespace=namespace,
            description=description,
            private=private,
            exists_ok=exists_ok,
        ):
            if slug in self.collections and not exists_ok:
                self._raise(409, f"{ENDPOINT}/api/collections")
            self.collections.setdefault(
                slug,
                {
                    "slug": slug,
                    "title": title,
                    "owner": {"name": owner, "type": "user"},
                    "items": [],
                    "description": description,
                    "private": private,
                },
            )
            return self._collection(slug)

    def list_collections(
        self,
        *,
        owner: list[str] | str | None = None,
        item: list[str] | str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        token: str | bool | None = None,
    ) -> list[Collection]:
        owners = [owner] if isinstance(owner, str) else owner
        with self._op("list_collections", f"{ENDPOINT}/api/collections", owner=owner, item=item, limit=limit):
            found = [
                # like the Hub: a listing carries at most four items per collection
                self._collection(slug, max_items=4)
                for slug, record in self.collections.items()
                if owners is None or record["owner"]["name"] in owners
            ]
            return found[:limit]

    def get_collection(self, collection_slug: str, *, token: str | bool | None = None) -> Collection:
        url = f"{ENDPOINT}/api/collections/{collection_slug}"
        with self._op("get_collection", url, collection_slug=collection_slug):
            if collection_slug not in self.collections:
                self._raise(404, url)
            return self._collection(collection_slug)

    def add_collection_item(
        self,
        collection_slug: str,
        item_id: str,
        item_type: str,
        *,
        note: str | None = None,
        exists_ok: bool = False,
        token: str | bool | None = None,
    ) -> Collection:
        url = f"{ENDPOINT}/api/collections/{collection_slug}/items"
        with self._op(
            "add_collection_item",
            url,
            collection_slug=collection_slug,
            item_id=item_id,
            item_type=item_type,
            note=note,
            exists_ok=exists_ok,
        ):
            if collection_slug not in self.collections:
                self._raise(404, url)
            items = self.collections[collection_slug]["items"]
            if any(i["id"] == item_id and i["type"] == item_type for i in items):
                if not exists_ok:
                    self._raise(409, url)
            else:
                items.append(
                    {
                        "_id": f"i{len(items)}",
                        "id": item_id,
                        "type": item_type,
                        "position": len(items),
                        "note": {"text": note} if note else None,
                    }
                )
            return self._collection(collection_slug)

    @contextmanager
    def _op(self, method: str, url: str, **kwargs: Any):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.threads.add(threading.get_ident())
            self.calls.append((method, kwargs))
            failure = self._failures.popleft() if self._failures else None
        try:
            if failure is not None:
                self._raise(failure[0], url, failure[1])
            yield
        finally:
            with self._lock:
                self._in_flight -= 1

    def _raise(self, status: int, url: str, headers: dict[str, str] | None = None) -> None:
        response = httpx2.Response(status, headers=headers or {}, request=httpx2.Request("GET", url))
        hf_raise_for_status(response)
        raise AssertionError(f"hf_raise_for_status did not raise for {status}")

    def _repo_url(self, repo_id: str, repo_type: str | None) -> str:
        return f"{ENDPOINT}/api/{repo_type or 'model'}s/{repo_id}"

    def _check_access(self, repo_id: str, repo_type: str | None) -> None:
        url = self._repo_url(repo_id, repo_type)
        if repo_id not in self.repos:
            self._raise(404, url, {"X-Error-Code": "RepoNotFound"})
        if repo_id in self.gated and repo_id not in self.authorized:
            self._raise(403, url, {"X-Error-Code": "GatedRepo"})
        if repo_id in self.private and repo_id not in self.authorized:
            self._raise(401, url)

    def _tree(self, repo_id: str, repo_type: str | None, revision: str | None) -> dict[str, bytes]:
        self._check_access(repo_id, repo_type)
        trees = self.repos[repo_id]
        revision = revision or "main"
        if revision not in trees:
            self._raise(
                404, f"{self._repo_url(repo_id, repo_type)}/revision/{revision}", {"X-Error-Code": "RevisionNotFound"}
            )
        return trees[revision]

    def _writable(self, repo_id: str, repo_type: str | None, revision: str | None) -> None:
        self._tree(repo_id, repo_type, revision)
        if revision not in (None, "main"):
            self._raise(400, self._repo_url(repo_id, repo_type))

    def _sha(self, repo_id: str, revision: str | None) -> str:
        return self.heads[repo_id] if revision in (None, "main") else revision

    def _commit(
        self, repo_id: str, message: str, update: dict[str, bytes] | None = None, delete: list[str] | None = None
    ) -> CommitInfo:
        # one lock, so the head always matches the live main tree
        with self._lock:
            tree = self.repos[repo_id]["main"]
            # like HfApi.create_commit: an upload that changes nothing makes no commit
            if update is not None and not delete and all(tree.get(path) == data for path, data in update.items()):
                return self._commit_info(repo_id, message, self.heads[repo_id])
            self._commits += 1
            tree.update(update or {})
            for path in delete or ():
                del tree[path]
            snapshot = dict(tree)
            digest = hashlib.sha1(
                json.dumps(sorted((p, hashlib.sha256(b).hexdigest()) for p, b in snapshot.items())).encode()
            )
            digest.update(str(self._commits).encode())
            oid = digest.hexdigest()
            self.repos[repo_id][oid] = snapshot
            self.heads[repo_id] = oid
            self.messages[oid] = message
        return self._commit_info(repo_id, message, oid)

    def _commit_info(self, repo_id: str, message: str, oid: str) -> CommitInfo:
        url = f"{ENDPOINT}/{_prefix(self.repo_types[repo_id])}{repo_id}/commit/{oid}"
        return CommitInfo(commit_url=url, commit_message=message, commit_description="", oid=oid)

    def _repo_file(self, path: str, data: bytes) -> RepoFile:
        payload: dict[str, Any] = {"path": path, "size": len(data), "oid": hashlib.sha1(data).hexdigest()}
        if path.endswith(LFS_SUFFIXES):
            payload["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 134}
        return RepoFile(**payload)

    def _collection(self, slug: str, max_items: int | None = None) -> Collection:
        record = self.collections[slug]
        return Collection(
            slug=slug,
            title=record["title"],
            owner=record["owner"],
            items=[dict(item) for item in record["items"][:max_items]],
            lastUpdated=datetime(2026, 11, 2, 10, 0, tzinfo=UTC).isoformat().replace("+00:00", "Z"),
            position=0,
            private=record["private"],
            theme="blue",
            upvotes=0,
            description=record["description"],
        )

    @staticmethod
    def _write(target: Path, path: str, data: bytes) -> Path:
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination


def _raise_unreachable(error: HfHubHTTPError) -> NoReturn:
    # like the library: repo and auth errors raw, the rest wrapped
    if isinstance(error, RepositoryNotFoundError) or error.response.status_code == 401:
        raise error
    raise LocalEntryNotFoundError("An error happened while trying to locate the files on the Hub.") from error


def _prefix(repo_type: str | None) -> str:
    return "" if repo_type in (None, "model") else f"{repo_type}s/"


def _read(source: str | Path | bytes | Any) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    return data if isinstance(data, bytes) else data.encode()


def _opt_str(value: str | Path | None) -> str | None:
    return None if value is None else str(value)
