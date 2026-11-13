"""Entry point shared by the job scripts."""

import argparse
import logging
import os
import signal
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import FrameType
from typing import Any, Self

from driftmap.core.clients import make_jobs_client
from driftmap.core.hub import Hub
from driftmap.core.records import JobKind, RunRecord
from driftmap.core.redact import configure_logging, mask
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings
from driftmap.jobkit.deadline import Clock, Deadline, DeadlineReached, SystemClock, parse_submitted_at
from driftmap.jobkit.flush import Flusher
from driftmap.jobkit.oom import oom_text
from driftmap.jobkit.resume import read_partial
from driftmap.jobkit.runrecord import Run, Status, find_job, start

logger = logging.getLogger(__name__)

TRAILER = "driftmap-job: finished status={status} exit={code}"
MAX_NOTE = 500
# after the last call the interrupt repeats until the body gives way
REPEAT_S = 2.0


@dataclass
class Context:
    """What a job body works with."""

    args: argparse.Namespace
    settings: Settings
    hub: Hub
    deadline: Deadline
    flusher: Flusher
    run: Run

    def checkpoint(self) -> None:
        """Call between work units: raises DeadlineReached at the deadline and runs the one forced flush."""
        if self.deadline.should_flush():
            raise DeadlineReached(f"deadline reached {self.deadline.elapsed_s():.0f} s after submission")
        if self.deadline.force_due():
            self.run.note(f"forced flush after {self.deadline.running_s():.0f} s")
            self.flusher.flush()


class _Watchdog:
    """While armed, raises DeadlineReached in the main thread: at the last call, again every REPEAT_S, on SIGTERM."""

    def __init__(self, deadline: Deadline, live: bool) -> None:
        self._deadline = deadline
        # signal handlers can only be set from the main thread
        self._live = live and threading.current_thread() is threading.main_thread()
        self.armed = False
        self._heed_term = True
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> Self:
        if self._live:
            self._previous = {number: signal.signal(number, self._fire) for number in (signal.SIGALRM, signal.SIGTERM)}
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stand_down()
        for number, handler in self._previous.items():
            # None: the earlier handler was not installed from Python
            signal.signal(number, signal.SIG_DFL if handler is None else handler)

    def arm(self) -> None:
        if self._live:
            self.armed = True
            # a zero delay would switch the timer off
            signal.setitimer(signal.ITIMER_REAL, max(self._deadline.last_call_s(), 1.0), REPEAT_S)

    def ignore_term(self) -> None:
        """Keep the timer, but let SIGTERM pass so the marker and record pushes can finish."""
        self._heed_term = False

    def stand_down(self) -> None:
        """Stop interrupting; SIGTERM is ignored too."""
        self.armed = False
        if self._live:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def _fire(self, number: int, frame: FrameType | None) -> None:
        # stays armed: a body that absorbs one interrupt gets the next
        if not self.armed:
            return
        if number == signal.SIGTERM:
            if not self._heed_term:
                return
            raise DeadlineReached("stopped by SIGTERM", urgent=True)
        after = self._deadline.elapsed_s()
        raise DeadlineReached(f"last call: interrupted {after:.0f} s after submission", urgent=True)


def _parse(argv: Sequence[str] | None, arguments: Callable[[argparse.ArgumentParser], object] | None) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-s", required=True, type=int)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--submitted-at", required=True, type=parse_submitted_at)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-flush-after-s", type=float)
    parser.add_argument("--resume", metavar="RUN_ID", help="continue from the marker this earlier run left")
    if arguments is not None:
        arguments(parser)
    return parser.parse_args(argv)


def _require_token() -> None:
    if not os.environ.get("HF_TOKEN", "").strip():
        raise DriftmapError(ExitCode.INTEGRITY, "HF_TOKEN is not set; the launcher passes it as a job secret")


def _no_wait(seconds: float) -> None:
    raise TimeoutError("no time left to retry a Hub call")


def _exit_code(error: BaseException) -> ExitCode:
    code = getattr(error, "exit_code", None)
    # 10 and 11 belong to a deadline or out-of-memory stop
    if isinstance(code, int) and code in ExitCode and code > ExitCode.OOM:
        return ExitCode(code)
    return ExitCode.UNEXPECTED


def _verdict(error: BaseException | None) -> tuple[Status, ExitCode, str | None, bool, str | None]:
    """Status, exit code, out-of-memory text, urgency and record note for what stopped the job."""
    if error is None:
        return "ok", ExitCode.SUCCESS, None, False, None
    if isinstance(error, DeadlineReached):
        logger.warning("%s; saving the run", error)
        return "partial", ExitCode.PARTIAL, None, error.urgent, str(error)
    try:
        oom = oom_text(error)
        status, code = ("oom", ExitCode.OOM) if oom else ("failed", _exit_code(error))
        note = f"{type(error).__name__}: {mask(str(error))}"[:MAX_NOTE]
    except Exception:
        # an error that cannot describe itself still gets its run saved
        oom, status, code, note = None, "failed", ExitCode.UNEXPECTED, type(error).__name__
    logger.error("job %s: %s", status, note, exc_info=error)
    return status, code, oom, False, note


def _caught(work: Callable[[], Any], watchdog: _Watchdog) -> tuple[Any, BaseException | None]:
    """Run ``work`` with the watchdog armed; returns its result, or None and whatever stopped it."""
    try:
        try:
            watchdog.arm()
            return work(), None
        finally:
            # a plain store: the smallest window for a late interrupt
            watchdog.armed = False
    except BaseException as error:
        return None, error
    finally:
        watchdog.stand_down()


def _save(
    flusher: Flusher, watchdog: _Watchdog, deadline: Deadline, status: Status, oom: str | None, urgent: bool
) -> RunRecord | None:
    """Close the run on the Hub; returns the record that got there, or None."""
    if not urgent and deadline.last_call_s() > 0:
        # the timer stays on; SIGTERM is ignored after the data files
        record, error = _caught(lambda: flusher.finish(status, oom=oom, after_data=watchdog.ignore_term), watchdog)
        if error is None:
            return record
        if isinstance(error, DeadlineReached):
            logger.warning("%s; one attempt left for the marker and record", error)
            flusher.run.note(str(error))
        else:
            logger.error("could not push the run record", exc_info=error)
            flusher.run.note(f"closing push failed: {type(error).__name__}: {mask(str(error))}"[:MAX_NOTE])
            # a run whose push failed did not end ok
            status = "failed" if status == "ok" else status
    try:
        return flusher.finish(status, oom=oom, data=False)
    except Exception:
        logger.exception("could not push the run record")
        return None


def run(
    body: Callable[[Context], object],
    argv: Sequence[str] | None = None,
    *,
    kind: JobKind,
    settings: Settings,
    arguments: Callable[[argparse.ArgumentParser], object] | None = None,
    client: Any | None = None,
    clock: Clock | None = None,
) -> int:
    """Run a job body, save the run however it ends, print the trailer and return the exit code."""
    args = _parse(argv, arguments)
    # an injected clock means a test: logging and signal handlers stay untouched
    live = clock is None
    clock = clock or SystemClock()
    if live:
        configure_logging()
    deadline = Deadline(args.timeout_s, args.submitted_at, clock, force_after_s=args.force_flush_after_s)
    opened: list[Flusher] = []

    def work() -> None:
        _require_token()
        api = make_jobs_client(settings) if client is None else client
        hub = Hub(api, gated_repos=settings.gated_repos, in_job=True)
        # every smoke plan carries the forced flush, a full plan never does
        mode = "full" if args.force_flush_after_s is None else "smoke"
        job = find_job(api, args.run_id)
        this = start(
            settings,
            run_id=args.run_id,
            kind=kind,
            mode=mode,
            code_sha=args.code_sha,
            deadline=deadline,
            clock=clock,
            job=job,
        )
        hub.create_repo(this.results, repo_type="dataset", private=True)
        last_hub = Hub(api, gated_repos=settings.gated_repos, in_job=True, sleep=_no_wait)
        flusher = Flusher(hub, this, urgent=last_hub)
        if args.resume:

            def note(text: str) -> None:
                logger.warning(text)
                this.note(text)

            marker = read_partial(hub, args.resume, this.record.config_hash, results=this.results, note=note)
            if marker is not None:
                flusher.done, flusher.resume_from = list(marker.done), marker.resume_from
        # earlier, a stop has nothing of this run to save
        opened.append(flusher)
        body(Context(args, settings, hub, deadline, flusher, this))

    status: Status = "failed"
    code = ExitCode.UNEXPECTED
    try:
        with _Watchdog(deadline, live) as watchdog:
            _, error = _caught(work, watchdog)
            status, code, oom, urgent, note = _verdict(error)
            record = None
            if opened:
                if note:
                    opened[0].run.note(note)
                record = _save(opened[0], watchdog, deadline, status, oom, urgent)
            saved = record.status if record else None
            # exits 0 and 10 promise a pushed record with that status
            if saved != status and (saved or status in ("ok", "partial")):
                status = saved or "failed"
                code = ExitCode.PARTIAL if status == "partial" else ExitCode.UNEXPECTED
    finally:
        print(TRAILER.format(status=status, code=int(code)), flush=True)
    return int(code)
