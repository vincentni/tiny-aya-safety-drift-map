"""Save what a job has produced: data files, then the marker, then the run record."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import CommitInfo
from pydantic import BaseModel

from driftmap.core.hub import Hub
from driftmap.core.records import PartialMarker, RunRecord
from driftmap.core.redact import mask
from driftmap.jobkit.deadline import Deadline
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import Push, Run, Status, record_path

logger = logging.getLogger(__name__)

MAX_NOTE = 300
RepoType = Literal["model", "dataset"]


@dataclass
class _Upload:
    target: str
    path_in_repo: str
    source: str | Path | bytes
    key: str | None
    repo_type: RepoType
    error: Exception | None = None


def _json(model: BaseModel) -> bytes:
    return (model.model_dump_json(by_alias=True, indent=2) + "\n").encode()


class Flusher:
    """Uploads a job's files and closes its run; every method can be called again after a failure."""

    def __init__(self, hub: Hub, run: Run, *, urgent: Hub | None = None) -> None:
        self._hub = hub
        # a Hub that does not retry, for when time has run out
        self._urgent = urgent or hub
        self.run = run
        self._pending: list[_Upload] = []
        self._aside: list[_Upload] = []
        self._noted: set[str] = set()
        self._pushes: dict[tuple[str, str], Push] = {}
        # keys of finished work units, in the order their uploads landed
        self.done: list[str] = []
        self.resume_from = "start"

    def add(
        self,
        target: str,
        path_in_repo: str,
        source: str | Path | bytes,
        *,
        key: str | None = None,
        repo_type: RepoType = "dataset",
    ) -> None:
        """Queue a file for the next upload or flush; ``key`` joins ``done`` once the file is on the Hub."""
        self._queue(target, path_in_repo, source, key, repo_type)

    def upload(
        self,
        target: str,
        path_in_repo: str,
        source: str | Path | bytes,
        *,
        key: str | None = None,
        repo_type: RepoType = "dataset",
    ) -> CommitInfo:
        """Upload a file now, after anything still queued; raises when this file fails, which sets it aside."""
        item = self._queue(target, path_in_repo, source, key, repo_type)
        self._push_pending()
        if item.error is not None:
            raise item.error
        return self._pushes[repo_type, target].commit

    def pushed(
        self, target: str, commit: CommitInfo, *, repo_type: RepoType = "model", sha256: str | None = None
    ) -> None:
        """Record a push made elsewhere, such as a model folder, as an output of the run."""
        self._pushes[repo_type, target] = Push(repo_type, target, commit, sha256)

    def mark_done(self, key: str) -> None:
        """Mark a work unit that has no file of its own as done."""
        if key not in self.done:
            self.done.append(key)

    def flush(self, *, data: bool = True) -> RunRecord:
        """Save a state a later run can resume from; ``data=False`` skips the queued files."""
        return self.finish("partial", data=data)

    def finish(
        self,
        status: Status,
        *,
        oom: str | None = None,
        data: bool = True,
        after_data: Callable[[], object] | None = None,
    ) -> RunRecord:
        """Push files, the marker unless ok, then the record; ``data=False`` skips the files and never retries."""
        if data:
            # files set aside earlier get one more try
            self._pending[:0], self._aside = self._aside, []
            self._push_pending()
            if after_data is not None:
                after_data()
            if self._aside and status == "ok":
                status = "failed"
        elif self._pending or self._aside:
            self.run.note(f"{len(self._pending) + len(self._aside)} file(s) not uploaded")
            if status == "ok":
                status = "partial"
        run_id = self.run.record.run_id
        marker = None
        if status != "ok":
            marker = PartialMarker(
                resume_from=self.resume_from,
                done=list(self.done),
                written_at=self.run.clock.now(),
                config_hash=self.run.record.config_hash,
            )
            single = self._single(data)
            try:
                self._put(marker_path(run_id), marker, f"marker for {run_id}", single)
            except Exception as error:
                if not single:
                    raise
                # no time for another round: the record still gets its attempt
                logger.warning("marker push failed: %s", error)
                self.run.note(f"marker push failed: {type(error).__name__}: {mask(str(error))}"[:MAX_NOTE])
        record = self.run.finish(status, self._pushes.values(), partial=marker, oom=oom)
        self._put(record_path(run_id), record, f"{status} record for {run_id}", self._single(data))
        return record

    def _single(self, data: bool) -> bool:
        # past the last call a retry could outlive the job
        return not data or self.run.past_last_call()

    def _put(self, path: str, model: BaseModel, message: str, single: bool) -> None:
        hub = self._urgent if single else self._hub
        hub.upload_file(self.run.results, path, _json(model), repo_type="dataset", commit_message=message)

    def _queue(
        self, target: str, path_in_repo: str, source: str | Path | bytes, key: str | None, repo_type: RepoType
    ) -> _Upload:
        item = _Upload(target, path_in_repo, source, key, repo_type)
        self._pending.append(item)
        return item

    def _push_pending(self) -> None:
        """Upload the queue in order; a file that fails is set aside and noted, the rest go on."""
        while self._pending:
            item = self._pending[0]
            try:
                commit = self._hub.upload_file(item.target, item.path_in_repo, item.source, repo_type=item.repo_type)
            except Exception as error:
                item.error = error
                self._aside.append(self._pending.pop(0))
                logger.warning("upload of %s failed: %s", item.path_in_repo, error)
                if item.path_in_repo not in self._noted:
                    self._noted.add(item.path_in_repo)
                    self.run.note(f"upload failed: {item.path_in_repo}: {mask(str(error))}"[:MAX_NOTE])
                continue
            self._pending.pop(0)
            self.pushed(item.target, commit, repo_type=item.repo_type)
            if item.key is not None:
                self.mark_done(item.key)


def __getattr__(name: str) -> Any:
    if name != "StepFlushCallback":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # transformers imports torch, so the class is built on first use
    from transformers import TrainerCallback

    class StepFlushCallback(TrainerCallback):
        """Stops training at the deadline, checked every ``every`` steps; set ``trainer`` after building it."""

        def __init__(self, deadline: Deadline, trainer: Any | None = None, every: int = 100) -> None:
            self.deadline = deadline
            self.trainer = trainer
            self.every = every

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step % self.every or not self.deadline.should_flush():
                return
            control.should_training_stop = True
            push = getattr(self.trainer, "push_in_progress", None)
            # the adapter must be on the Hub before a marker names it
            if push is not None:
                push.wait_until_done()

    globals()[name] = StepFlushCallback
    return StepFlushCallback
