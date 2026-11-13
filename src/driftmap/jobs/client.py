"""Hugging Face Jobs calls: submit, inspect, prices, logs and billing usage."""

import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx2
from huggingface_hub import HfApi, JobInfo
from huggingface_hub._jobs_api import TERMINAL_JOB_STAGES, JobHardwareInfo
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

from driftmap.core.clients import REQUEST_TIMEOUT, register_backend
from driftmap.core.retry import policy
from driftmap.core.schemas import DriftmapError, ExitCode

BILLING_PATH = "/api/settings/billing/usage/jobs"
BILLING_SCOPE = "Billing > Read billing usage and payment method status"
MIN_POLL_S = 60


class BillingUnavailable(DriftmapError):
    """Billing usage cannot be read, or its body carries no usable figure."""

    exit_code_default = ExitCode.READINESS


class HfJobsClient:
    """Jobs calls in the owner's namespace; reads are retried, a submit never is."""

    def __init__(
        self,
        settings: Any,
        api: HfApi,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api = api
        # unnamed, the library resolves it with the machine's stored token
        self._namespace = settings.owner
        self._retry = policy(sleep)
        self._sleep = sleep
        self._monotonic = monotonic

    def run_job(self, **kwargs: Any) -> JobInfo:
        return self.api.run_job(namespace=self._namespace, **kwargs)

    def run_uv_job(self, script: str, **kwargs: Any) -> JobInfo:
        return self.api.run_uv_job(script, namespace=self._namespace, **kwargs)

    def inspect_job(self, *, job_id: str) -> JobInfo:
        return self._retry(self.api.inspect_job, job_id=job_id, namespace=self._namespace)

    def list_jobs(self, *, status: list[str] | str | None = None) -> list[JobInfo]:
        # the library already backs off; a second layer multiplies requests
        return list(self.api.list_jobs(status=status, namespace=self._namespace))

    def list_jobs_hardware(self) -> list[JobHardwareInfo]:
        return self._retry(self.api.list_jobs_hardware)

    def wait_for_job(self, job_id: str, *, timeout: float, poll_interval: float = MIN_POLL_S) -> JobInfo:
        """Block until the job is terminal; TimeoutError after ``timeout`` seconds, the job keeps running."""
        if poll_interval < MIN_POLL_S:
            raise ValueError(f"poll_interval must be at least {MIN_POLL_S} s, got {poll_interval}")
        if timeout is None or timeout < 0:
            raise ValueError(f"timeout must be a number of seconds, got {timeout}")
        deadline = self._monotonic() + timeout
        while True:
            # the library's wait gives up on the first failed poll
            job = self.inspect_job(job_id=job_id)
            if job.status.stage in TERMINAL_JOB_STAGES:
                return job
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise TimeoutError(f"job {job_id} is still {job.status.stage} after {timeout} s")
            self._sleep(min(poll_interval, remaining))

    def fetch_job_logs(self, *, job_id: str, tail: int | None = None) -> list[str]:
        """Log lines available now; the stream is not followed."""
        return list(self.api.fetch_job_logs(job_id=job_id, namespace=self._namespace, tail=tail))

    def billing_usage(self) -> dict[str, Any]:
        """The usage body as the Hub returns it; BillingUnavailable on a 401 or 403."""
        try:
            return self._retry(self._billing_usage)
        except HfHubHTTPError as error:
            if error.response.status_code in (401, 403):
                raise BillingUnavailable(f"this token cannot read billing usage; it needs '{BILLING_SCOPE}'") from error
            raise

    def _billing_usage(self) -> dict[str, Any]:
        url = f"{self.api.endpoint}{BILLING_PATH}"
        response = get_session().get(url, headers=build_hf_headers(token=self.api.token))
        hf_raise_for_status(response)
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise BillingUnavailable("billing usage did not answer with a JSON object")
        return body


def register() -> None:
    """Make ``make_jobs_client`` return an HfJobsClient; call once at startup."""
    register_backend(HfJobsClient)


def used_micro_usd(usage: Any) -> int:
    """Spend so far from a billing usage body, in micro-USD; BillingUnavailable when it has none."""
    period = usage.get("usage") if isinstance(usage, dict) else None
    used = period.get("usedMicroUsd") if isinstance(period, dict) else None
    # bool is an int subclass
    if type(used) is not int or used < 0:
        raise BillingUnavailable(f"billing usage carries no whole usedMicroUsd, got {used!r}")
    return used


def minute_prices(rows: Iterable[JobHardwareInfo]) -> dict[str, int]:
    """Micro-USD per minute by flavor; a flavor billed in another unit is left out."""
    return {
        row.name: row.unit_cost_micro_usd
        for row in rows
        if row.unit_label == "minute" and type(row.unit_cost_micro_usd) is int
    }


def head_status(url: str, *, transport: httpx2.BaseTransport | None = None) -> int:
    """Status of a HEAD request once redirects are followed."""
    with httpx2.Client(transport=transport, follow_redirects=True, timeout=REQUEST_TIMEOUT) as session:
        return session.head(url).status_code
