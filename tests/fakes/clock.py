from datetime import UTC, datetime, timedelta


class FakeClock:
    """Clock that only moves on ``sleep()`` or ``advance()`` and never blocks."""

    def __init__(self, monotonic: float = 1000.0, now: datetime | None = None) -> None:
        self._monotonic = monotonic
        self._now = now or datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self._monotonic

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._now += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)
