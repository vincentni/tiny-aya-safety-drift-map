"""Retry policy for Hub calls."""

import logging
import time
from collections.abc import Callable

import httpx2
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from huggingface_hub.utils import parse_ratelimit_headers
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

ATTEMPTS = 3
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_RESET_WAIT_S = 120

_TRANSPORT_ERRORS = (httpx2.TimeoutException, httpx2.NetworkError, httpx2.RemoteProtocolError)
_jitter = wait_exponential_jitter(multiplier=1, max=30)


def root_cause(exc: BaseException) -> BaseException:
    """The Hub error behind a failed download, else the error itself."""
    if isinstance(exc, LocalEntryNotFoundError) and exc.__cause__ is not None:
        return exc.__cause__
    return exc


def is_transient(exc: BaseException) -> bool:
    """Worth another attempt; a 429 whose reset is past the cap is not."""
    exc = root_cause(exc)
    if isinstance(exc, HfHubHTTPError):
        return exc.response.status_code in TRANSIENT_STATUS and (_reset_s(exc) or 0) <= MAX_RESET_WAIT_S
    return isinstance(exc, _TRANSPORT_ERRORS)


def policy(sleep: Callable[[float], None] = time.sleep, *, retry_throttled: bool = True) -> Retrying:
    """Three attempts with jittered backoff; a 429 waits for the server's reset instead."""

    def retryable(exc: BaseException) -> bool:
        return is_transient(exc) and (retry_throttled or not _is_throttled(root_cause(exc)))

    return Retrying(
        stop=stop_after_attempt(ATTEMPTS),
        wait=_wait,
        retry=retry_if_exception(retryable),
        before_sleep=_log,
        reraise=True,
        sleep=sleep,
    )


def _is_throttled(exc: BaseException) -> bool:
    return isinstance(exc, HfHubHTTPError) and exc.response.status_code == 429


def _reset_s(exc: HfHubHTTPError) -> int | None:
    if not _is_throttled(exc):
        return None
    # reads the IETF ratelimit header only; Retry-After parses to None
    limit = parse_ratelimit_headers(exc.response.headers)
    return limit.reset_in_seconds if limit else None


def _wait(state: RetryCallState) -> float:
    exc = root_cause(state.outcome.exception())
    reset = _reset_s(exc) if isinstance(exc, HfHubHTTPError) else None
    return _jitter(state) if reset is None else reset


def _log(state: RetryCallState) -> None:
    exc = root_cause(state.outcome.exception())
    status = getattr(getattr(exc, "response", None), "status_code", "")
    logger.warning(
        "Hub call failed (%s %s), attempt %d of %d, retrying in %.1f s",
        type(exc).__name__,
        status,
        state.attempt_number,
        ATTEMPTS,
        state.upcoming_sleep,
    )
