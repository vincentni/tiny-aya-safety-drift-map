import dataclasses
import json
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from huggingface_hub._jobs_api import TERMINAL_JOB_STAGES, JobHardwareInfo, JobInfo
from huggingface_hub.utils import hf_raise_for_status

ENDPOINT = "https://huggingface.co"

# rows from the Jobs hardware endpoint, 2026-11-01
DEFAULT_HARDWARE: list[dict[str, Any]] = [
    {
        "name": "cpu-basic",
        "prettyName": "CPU Basic",
        "cpu": "2 vCPU",
        "ram": "16 GB",
        "ephemeralStorage": "50 GB",
        "accelerator": None,
        "unitCostMicroUSD": 167,
        "unitCostUSD": 0.000167,
        "unitLabel": "minute",
    },
    {
        "name": "cpu-upgrade",
        "prettyName": "CPU Upgrade",
        "cpu": "8 vCPU",
        "ram": "32 GB",
        "ephemeralStorage": "50 GB",
        "accelerator": None,
        "unitCostMicroUSD": 500,
        "unitCostUSD": 0.0005,
        "unitLabel": "minute",
    },
    {
        "name": "l4x1",
        "prettyName": "Nvidia L4",
        "cpu": "8 vCPU",
        "ram": "30 GB",
        "ephemeralStorage": "400 GB",
        "accelerator": {"type": "gpu", "model": "L4", "quantity": "1", "vram": "24 GB", "manufacturer": "Nvidia"},
        "unitCostMicroUSD": 13333,
        "unitCostUSD": 0.013333,
        "unitLabel": "minute",
    },
    {
        "name": "a10g-small",
        "prettyName": "Nvidia A10G - small",
        "cpu": "4 vCPU",
        "ram": "15 GB",
        "ephemeralStorage": "400 GB",
        "accelerator": {"type": "gpu", "model": "A10G", "quantity": "1", "vram": "24 GB", "manufacturer": "Nvidia"},
        "unitCostMicroUSD": 16667,
        "unitCostUSD": 0.016667,
        "unitLabel": "minute",
    },
]

SECRET_KEY = re.compile(r"TOKEN|SECRET|PASSWORD|KEY", re.IGNORECASE)
SECRET_VALUE = re.compile(r"hf_[A-Za-z0-9]{30,}")
MASK = "***"


class FakeJobsClient:
    """Scripted HF Jobs client that returns real ``JobInfo`` objects."""

    def __init__(
        self, *, user: str = "vincentnkq", hardware: list[dict[str, Any]] | None = None, now: datetime | None = None
    ) -> None:
        self.user = user
        self.hardware = [dict(row) for row in (hardware or DEFAULT_HARDWARE)]
        self.billing_used_micro_usd = 0
        self.billing_total_minutes = 0
        self.billing_status = 200
        self.logs: dict[str, list[str]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._now = now or datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
        self._scripts: dict[str, deque[dict[str, Any]]] = {}
        self._pending_ids: deque[str] = deque()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._minted = 0

    def script_job(self, job_id: str, statuses: list[dict[str, Any]]) -> None:
        """Queue the stages a job walks through; the last one repeats."""
        assert statuses, "a job script needs at least one status"
        self._scripts[job_id] = deque(statuses)
        self._pending_ids.append(job_id)

    def run_job(self, *, image: str, command: list[str], flavor: str | None = None, **kwargs: Any) -> JobInfo:
        self._record("run_job", image=image, command=command, flavor=flavor, **kwargs)
        return self._accept(dockerImage=image, command=list(command), flavor=flavor, labels=kwargs.get("labels"))

    def run_uv_job(
        self,
        script: str,
        *,
        script_args: list[str] | None = None,
        flavor: str | None = None,
        image: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        self._record("run_uv_job", script=script, script_args=script_args, flavor=flavor, image=image, **kwargs)
        command = ["uv", "run", script, *(script_args or [])]
        return self._accept(dockerImage=image, command=command, flavor=flavor, labels=kwargs.get("labels"))

    def inspect_job(self, *, job_id: str, namespace: str | None = None, token: str | bool | None = None) -> JobInfo:
        self._record("inspect_job", job_id=job_id, namespace=namespace)
        current = self._peek(job_id)
        self._advance(job_id)
        return self._job_info(job_id, current)

    def list_jobs(self, **kwargs: Any) -> list[JobInfo]:
        self._record("list_jobs", **kwargs)
        return [self._job_info(job_id, self._peek(job_id)) for job_id in self._jobs]

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
        stages: list[Any] | None = None,
        namespace: str | None = None,
        token: str | bool | None = None,
    ) -> JobInfo:
        self._record(
            "wait_for_job",
            job_id=job_id,
            timeout=timeout,
            poll_interval=poll_interval,
            stages=stages,
            namespace=namespace,
        )
        targets = set(stages or ()) | set(TERMINAL_JOB_STAGES)
        while True:
            info = self._job_info(job_id, self._peek(job_id))
            if info.status.stage in targets:
                return info
            if not self._advance(job_id):
                if timeout is None:
                    raise AssertionError(
                        f"job {job_id!r} script ended on non-terminal stage {info.status.stage!r} with no timeout"
                    )
                raise TimeoutError(f"Job '{job_id}' is still in stage '{info.status.stage}' after {timeout} seconds.")

    def cancel_job(self, *, job_id: str, namespace: str | None = None, token: str | bool | None = None) -> None:
        self._record("cancel_job", job_id=job_id, namespace=namespace)
        self._peek(job_id)
        self._scripts[job_id] = deque([{"stage": "CANCELED"}])

    def fetch_job_logs(
        self, *, job_id: str, namespace: str | None = None, token: str | bool | None = None, **kwargs: Any
    ) -> list[str]:
        self._record("fetch_job_logs", job_id=job_id, namespace=namespace, **kwargs)
        self._peek(job_id)
        return list(self.logs.get(job_id, []))

    def list_jobs_hardware(self, token: str | bool | None = None) -> list[JobHardwareInfo]:
        self._record("list_jobs_hardware")
        return [JobHardwareInfo(**row) for row in self.hardware]

    def billing_usage(self) -> dict[str, Any]:
        self._record("billing_usage")
        if self.billing_status != 200:
            request = httpx2.Request("GET", f"{ENDPOINT}/api/settings/billing/usage/jobs")
            hf_raise_for_status(httpx2.Response(self.billing_status, request=request))
        usage = {
            "totalMinutes": self.billing_total_minutes,
            "jobDetails": [],
            "usedMicroUsd": self.billing_used_micro_usd,
            "periodStart": "2026-11-01T00:00:00.000Z",
            "periodEnd": _iso(self._now),
        }
        return {"usage": usage, "hasAccess": True}

    def whoami(self, token: str | bool | None = None) -> dict[str, Any]:
        self._record("whoami")
        return {"name": self.user, "type": "user"}

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    def _accept(self, **base: Any) -> JobInfo:
        job_id = self._pending_ids.popleft() if self._pending_ids else self._mint()
        self._jobs[job_id] = {"createdAt": _iso(self._now), **base}
        return self._job_info(job_id, self._peek(job_id))

    def _mint(self) -> str:
        self._minted += 1
        job_id = f"fake{self._minted:020d}"
        self._scripts[job_id] = deque([{"stage": "COMPLETED"}])
        return job_id

    def _peek(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            request = httpx2.Request("GET", f"{ENDPOINT}/api/jobs/{self.user}/{job_id}")
            hf_raise_for_status(httpx2.Response(404, request=request))
        return self._scripts[job_id][0]

    def _advance(self, job_id: str) -> bool:
        script = self._scripts[job_id]
        if len(script) > 1:
            script.popleft()
            return True
        return False

    def _job_info(self, job_id: str, entry: dict[str, Any]) -> JobInfo:
        if "stage" in entry:
            status, extra = entry, {}
        else:
            status, extra = entry["status"], {k: v for k, v in entry.items() if k != "status"}
        base = self._jobs[job_id]
        payload = {
            "id": job_id,
            "owner": {"id": "u1", "name": self.user, "type": "user"},
            "createdAt": base["createdAt"],
            "dockerImage": base.get("dockerImage"),
            "command": base.get("command"),
            "flavor": base.get("flavor"),
            "labels": base.get("labels"),
            "durations": {"schedulingSecs": 10, "runningSecs": 0, "totalSecs": 10},
            **extra,
            "status": dict(status),
        }
        return JobInfo(**payload)


class FileBackedFakeClient:
    """``FakeJobsClient`` that logs every call to a JSONL file, secrets masked."""

    def __init__(self, path: str | Path, inner: FakeJobsClient | None = None) -> None:
        self.inner = inner or FakeJobsClient()
        self.path = Path(path)
        sidecar = self.path.with_name(self.path.name + ".script.json")
        if inner is None and sidecar.exists():
            self._apply_script(json.loads(sidecar.read_text()))

    def _apply_script(self, script: dict[str, Any]) -> None:
        for job_id, statuses in script.get("jobs", {}).items():
            self.inner.script_job(job_id, statuses)
        for key in ("billing_used_micro_usd", "billing_total_minutes", "billing_status"):
            if key in script:
                setattr(self.inner, key, script[key])

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.inner, name)
        if name.startswith("_") or not callable(target):
            return target

        def recorded(*args: Any, **kwargs: Any) -> Any:
            try:
                reply = target(*args, **kwargs)
            except Exception as exc:
                self._append(name, args, kwargs, error=f"{type(exc).__name__}: {exc}")
                raise
            self._append(name, args, kwargs, reply=_jsonable(reply))
            return reply

        return recorded

    def _append(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], **extra: Any) -> None:
        line = {
            "method": method,
            "args": _mask(list(args)),
            "kwargs": _mask(kwargs),
            **{k: _mask(v) for k, v in extra.items()},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, default=str) + "\n")


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _mask(value: Any, key: str | None = None) -> Any:
    if key is not None and SECRET_KEY.search(key):
        return {k: MASK for k in value} if isinstance(value, dict) else MASK
    if isinstance(value, dict):
        return {k: _mask(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(item) for item in value]
    if isinstance(value, str) and SECRET_VALUE.search(value):
        return MASK
    return value
