import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError, HFValidationError
from huggingface_hub.utils import hf_raise_for_status

from driftmap.core.retry import is_transient, policy
from tests.fakes import FakeClock

URL = "https://huggingface.co/api/models/vincentnkq/x"


def http_error(status: int, headers: dict[str, str] | None = None) -> HfHubHTTPError:
    response = httpx2.Response(status, headers=headers or {}, request=httpx2.Request("GET", URL))
    with pytest.raises(HfHubHTTPError) as raised:
        hf_raise_for_status(response)
    return raised.value


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retried_statuses(status):
    assert is_transient(http_error(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 501])
def test_statuses_that_fail_at_once(status):
    assert not is_transient(http_error(status))


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (httpx2.ReadTimeout("slow"), True),
        (httpx2.ConnectTimeout("slow"), True),
        (httpx2.ConnectError("refused"), True),
        (httpx2.ReadError("reset"), True),
        (httpx2.RemoteProtocolError("server disconnected"), True),
        (HFValidationError("bad repo id"), False),
        (TimeoutError("job still running"), False),
        (KeyboardInterrupt(), False),
    ],
)
def test_transport_errors(exc, transient):
    assert is_transient(exc) is transient


def test_a_network_blip_is_retried_and_the_result_returned():
    clock = FakeClock()
    outcomes = [httpx2.ConnectError("refused"), http_error(502), "done"]

    def flaky():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert policy(clock.sleep)(flaky) == "done"
    assert len(clock.sleeps) == 2 and outcomes == []


def test_a_fourth_attempt_never_happens():
    clock = FakeClock()
    calls = []

    def down():
        calls.append(1)
        raise httpx2.ReadTimeout("slow")

    with pytest.raises(httpx2.ReadTimeout):
        policy(clock.sleep)(down)
    assert len(calls) == 3 and len(clock.sleeps) == 2
    assert all(wait <= 30 for wait in clock.sleeps)
