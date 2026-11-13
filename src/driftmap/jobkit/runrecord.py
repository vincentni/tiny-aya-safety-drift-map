"""The run record: kept in memory while the job works, pushed last."""

import logging
import math
import resource
import sys
from collections.abc import Iterable
from typing import Any, Literal, NamedTuple

from huggingface_hub import CommitInfo

from driftmap.core.records import JobKind, Mode, PartialMarker, Ref, RunRecord
from driftmap.core.redact import mask
from driftmap.core.retry import policy
from driftmap.core.schemas import SCHEMA_RUN
from driftmap.core.settings import Settings
from driftmap.jobkit.deadline import Clock, Deadline

logger = logging.getLogger(__name__)

# the label the launcher puts on every job it submits
RUN_LABEL = "driftmap_run"
NO_JOB = "job lookup failed: command, flavor and price unknown"
Status = Literal["ok", "partial", "oom", "failed"]


class Push(NamedTuple):
    """One commit that carried an output."""

    kind: Literal["model", "dataset"]
    repo: str
    commit: CommitInfo
    sha256: str | None = None


def record_path(run_id: str) -> str:
    """Where a run's record lives in the results repo."""
    return f"runs/{run_id}.json"


def find_job(client: Any, run_id: str) -> Any | None:
    """The job labelled with this run id, or None when it cannot be looked up."""
    try:
        jobs = policy()(client.list_jobs, labels={RUN_LABEL: run_id})
        return next((job for job in jobs if (job.labels or {}).get(RUN_LABEL) == run_id), None)
    except Exception as error:
        logger.warning("job lookup failed: %s: %s", type(error).__name__, error)
        return None


def peak_vram_gb() -> float | None:
    """Peak CUDA memory of this process; None without torch or a GPU."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def peak_rss_gb() -> float:
    """Peak resident memory of this process or its largest child."""
    # ru_maxrss is in kibibytes on Linux
    peaks = (resource.getrusage(who).ru_maxrss for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return round(max(peaks) / 2**20, 3)


class Run:
    """A job's record and the clocks that close it; the body updates ``record`` in place."""

    def __init__(self, record: RunRecord, results: str, deadline: Deadline, clock: Clock) -> None:
        self.record = record
        self.results = results
        self.clock = clock
        self._deadline = deadline

    def past_last_call(self) -> bool:
        """True when no time is left to retry a Hub call."""
        return self._deadline.last_call_s() <= 0

    def note(self, text: str) -> None:
        """Add a note once."""
        if text not in self.record.notes:
            self.record.notes.append(text)

    def finish(
        self, status: Status, outputs: Iterable[Push], *, partial: PartialMarker | None = None, oom: str | None = None
    ) -> RunRecord:
        """The record as it stands now, closed with ``status``; revisions come from the pushes' commits."""
        record = self.record
        wall_clock_s = self._deadline.running_s()
        minutes = math.ceil(self._deadline.elapsed_s() / 60)
        rate = record.tokens_per_s
        if rate is None and record.generated_tokens and wall_clock_s > 0:
            rate = round(record.generated_tokens / wall_clock_s, 3)
        closing = {
            "outputs": [Ref(kind=p.kind, repo=p.repo, revision=p.commit.oid, sha256=p.sha256) for p in outputs],
            "finished": self.clock.now(),
            "wall_clock_s": round(wall_clock_s, 3),
            "minutes": minutes,
            "cost_micro_usd": minutes * record.price_micro_usd_per_min if record.price_micro_usd_per_min else None,
            "tokens_per_s": rate,
            "status": status,
            "partial": partial,
            "oom": oom,
            "peak_vram_gb": peak_vram_gb(),
            "peak_rss_gb": peak_rss_gb(),
        }
        return RunRecord.model_validate(record.model_dump(by_alias=True) | closing)


def start(
    settings: Settings,
    *,
    run_id: str,
    kind: JobKind,
    mode: Mode,
    code_sha: str,
    deadline: Deadline,
    clock: Clock,
    job: Any | None = None,
    inputs: Iterable[Ref] = (),
) -> Run:
    """Open the record as ``partial``; ``job`` (from ``find_job``) supplies the command, flavor and price."""
    flavor = getattr(job, "flavor", None) or "unknown"
    record = RunRecord(
        schema=SCHEMA_RUN,
        run_id=run_id,
        job_id=getattr(job, "id", None),
        kind=kind,
        mode=mode,
        config_hash=settings.config_hash(),
        code_sha=code_sha,
        inputs=list(inputs),
        outputs=[],
        merged_sha256=None,
        # the job's environment and secrets are never copied
        command=[mask(part) for part in getattr(job, "command", None) or []],
        started=clock.now(),
        finished=None,
        wall_clock_s=None,
        flavor=str(flavor),
        price_micro_usd_per_min=settings.prices_micro_usd_per_min.get(str(flavor), 0),
        minutes=None,
        cost_micro_usd=None,
        prompts=0,
        prompt_tokens=0,
        generated_tokens=0,
        tokens_per_s=None,
        status="partial",
        partial=None,
        oom=None,
        peak_vram_gb=None,
        peak_rss_gb=None,
        install_minutes=round(deadline.waited_s / 60, 2),
        notes=[] if job is not None else [NO_JOB],
    )
    return Run(record, f"{settings.owner}/{settings.repos.results}", deadline, clock)
