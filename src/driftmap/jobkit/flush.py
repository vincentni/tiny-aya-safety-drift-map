"""Save what a job has produced: data files, then the marker, then the run record."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import CommitInfo, CommitOperationAdd
from pydantic import BaseModel

from driftmap.core.hub import Hub
from driftmap.core.records import PartialMarker, RunRecord
from driftmap.core.redact import mask
from driftmap.jobkit.deadline import Deadline
from driftmap.jobkit.resume import marker_path
from driftmap.jobkit.runrecord import Push, Run, Status, record_path

logger = logging.getLogger(__name__)

MAX_NOTE = 300
# between work units the marker and the record are saved when the last save is this old: each save costs two
# commits, and the Hub can take half a minute to answer one
PROGRESS_S = 300
RepoType = Literal["model", "dataset"]


def unit_message(run_id: str, key: str) -> str:
    """Title of the commits that carry a work unit's files; a resumed run reads the unit back from it."""
    return f"{run_id}: {key}"


@dataclass
class _Upload:
    target: str
    path_in_repo: str
    source: str | Path | bytes
    key: str | None
    repo_type: RepoType
    error: Exception | None = None
    # a file of a work unit: every file of the unit, and whether this one is on the Hub
    unit: "_Unit | None" = None
    landed: bool = False


@dataclass
class _Unit:
    key: str
    files: list[_Upload]


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
        # when this run last saved its marker and record, on the run's clock
        self._saved: float | None = None

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

    def commit(
        self, files: Sequence[tuple[str, str, str | Path | bytes]], *, key: str, repo_type: RepoType = "dataset"
    ) -> None:
        """Upload the files of one work unit now, after anything still queued: one commit per repo.

        ``files`` are (repo, path in it, source); the repos are committed to in the order they first appear, each
        commit titled with the run and ``key``, which joins ``done`` once the last one landed. Raises when a
        commit fails: its files, and those of the repos after it, are set aside for the closing save, and the
        key is done when the last of them is on the Hub.
        """
        self._push_pending()
        unit = _Unit(key, [])
        groups: dict[str, list[_Upload]] = {}
        for target, path_in_repo, source in files:
            item = _Upload(target, path_in_repo, source, None, repo_type, unit=unit)
            unit.files.append(item)
            groups.setdefault(target, []).append(item)
        left = list(groups.values())
        message = unit_message(self.run.record.run_id, key)
        while left:
            group = left[0]
            try:
                if len(group) == 1:
                    item = group[0]
                    landed = self._hub.upload_file(
                        item.target, item.path_in_repo, item.source, repo_type=repo_type, commit_message=message
                    )
                else:
                    added = [CommitOperationAdd(item.path_in_repo, item.source) for item in group]
                    landed = self._hub.create_commit(
                        group[0].target, added, repo_type=repo_type, commit_message=message
                    )
            except Exception as error:
                for item in (item for group in left for item in group):
                    item.error = error
                    self._aside.append(item)
                self._failed(group[0], error)
                raise
            for item in left.pop(0):
                item.landed = True
            self.pushed(group[0].target, landed, repo_type=repo_type)
        self.mark_done(key)

    def credit(self, target: str, run_id: str, *, repo_type: RepoType = "dataset") -> None:
        """Take as done the units of ``run_id`` whose commit reached ``target``, the repo its units commit to last.

        The marker a run resumes from can be minutes older than the last unit it put on the Hub.
        """
        prefix = unit_message(run_id, "")
        for commit in reversed(self._hub.list_repo_commits(target, repo_type=repo_type)):
            if commit.title.startswith(prefix):
                self.mark_done(commit.title.removeprefix(prefix))

    def progress(self) -> None:
        """Call between work units: save the marker and the record if this run has not yet, or not for PROGRESS_S.

        A save that fails is a note; the next call tries again, and so does the closing save.
        """
        if self._saved is not None and self.run.clock.monotonic() - self._saved < PROGRESS_S:
            return
        try:
            self.flush()
        except Exception as error:
            self.run.note(f"marker push failed: {type(error).__name__}: {mask(str(error))}"[:MAX_NOTE])

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
        self._saved = self.run.clock.monotonic()
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

    def _failed(self, item: _Upload, error: Exception) -> None:
        logger.warning("upload of %s failed: %s", item.path_in_repo, error)
        if item.path_in_repo not in self._noted:
            self._noted.add(item.path_in_repo)
            self.run.note(f"upload failed: {item.path_in_repo}: {mask(str(error))}"[:MAX_NOTE])

    def _push_pending(self) -> None:
        """Upload the queue in order; a file that fails is set aside and noted, the rest go on."""
        while self._pending:
            item = self._pending[0]
            try:
                commit = self._hub.upload_file(item.target, item.path_in_repo, item.source, repo_type=item.repo_type)
            except Exception as error:
                item.error = error
                self._aside.append(self._pending.pop(0))
                self._failed(item, error)
                continue
            self._pending.pop(0)
            self.pushed(item.target, commit, repo_type=item.repo_type)
            item.landed = True
            if item.key is not None:
                self.mark_done(item.key)
            # a unit is done with its last file, whichever of them that is
            if item.unit is not None and all(other.landed for other in item.unit.files):
                self.mark_done(item.unit.key)


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
