"""The job's deadline, counted from the moment it was submitted."""

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def now(self) -> datetime: ...


class SystemClock:
    monotonic = staticmethod(time.monotonic)

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)


class DeadlineReached(BaseException):
    """Time to save and stop; ``urgent`` when there is no time left for data uploads."""

    # not an Exception: a library's broad handler must not swallow it
    def __init__(self, reason: str, *, urgent: bool = False) -> None:
        super().__init__(reason)
        self.urgent = urgent


def parse_submitted_at(text: str) -> datetime:
    """An ISO timestamp carrying an offset or Z, as UTC; ValueError when it has neither."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"expected a UTC offset or Z in {text!r}")
    return moment.astimezone(UTC)


class Deadline:
    """Tells a job when to flush: ``reserve_s`` before the platform's timeout."""

    def __init__(
        self,
        timeout_s: float,
        submitted_at: datetime,
        clock: Clock,
        reserve_s: float | None = None,
        force_after_s: float | None = None,
    ) -> None:
        started = clock.now().astimezone(UTC)
        self.timeout_s = timeout_s
        self.reserve_s = max(timeout_s / 10, 180) if reserve_s is None else reserve_s
        # a launcher clock running ahead must not move the deadline later
        self.effective_submitted_at = min(submitted_at.astimezone(UTC), started)
        self.waited_s = (started - self.effective_submitted_at).total_seconds()
        self._clock = clock
        self._started = clock.monotonic()
        self._force_after_s = force_after_s

    def running_s(self) -> float:
        """Seconds since this process started."""
        return self._clock.monotonic() - self._started

    def elapsed_s(self) -> float:
        """Seconds since submission, queueing and install included."""
        return self.waited_s + self.running_s()

    def should_flush(self) -> bool:
        return self.elapsed_s() >= self.timeout_s - self.reserve_s

    def last_call_s(self) -> float:
        """Seconds until a blocked job must be interrupted: a third of the reserve before the timeout."""
        return self.timeout_s - self.reserve_s / 3 - self.elapsed_s()

    def force_due(self) -> bool:
        """True once, when the process has run for ``force_after_s``."""
        if self._force_after_s is None or self.running_s() < self._force_after_s:
            return False
        self._force_after_s = None
        return True
