import subprocess
import sys

import pytest
import torch
from auditkit.errors import ModelError

from driftmap.core.errors import Breaker, OutOfMemory, chain, classify_oom
from driftmap.core.schemas import DriftmapError, ExitCode

OOM_TEXT = "CUDA out of memory. Tried to allocate 2.00 GiB"


def chained(outer, inner, explicit):
    try:
        try:
            raise inner
        except Exception as error:
            if explicit:
                raise outer from error
            raise outer  # noqa: B904
    except Exception as error:
        return error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (torch.OutOfMemoryError("anything"), True),
        (RuntimeError(OOM_TEXT), True),
        (chained(ModelError("generate failed"), torch.OutOfMemoryError("x"), explicit=True), True),
        (chained(ModelError("generate failed"), torch.OutOfMemoryError("x"), explicit=False), True),
        (chained(ModelError("outer"), chained(ValueError("mid"), RuntimeError(OOM_TEXT), True), True), True),
        (RuntimeError("backend down"), False),
        (chained(ModelError("generate failed"), RuntimeError("backend down"), explicit=True), False),
        (MemoryError("host memory"), False),
    ],
)
def test_classify_oom_reads_the_whole_exception_chain(error, expected):
    assert classify_oom(error) is expected


def test_chain_yields_each_error_once_even_in_a_cycle():
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first
    second.__context__ = first
    assert list(chain(first)) == [first, second]
    assert classify_oom(first) is False


def test_classify_oom_works_before_torch_is_imported(child_env):
    code = (
        "import sys\n"
        "from driftmap.core.errors import classify_oom\n"
        "assert 'torch' not in sys.modules\n"
        "assert classify_oom(RuntimeError('CUDA out of memory')) is True\n"
        "assert classify_oom(RuntimeError('backend down')) is False\n"
    )
    done = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(("error", "code"), [(OutOfMemory, ExitCode.OOM), (Breaker, ExitCode.COLUMN_FAILED)])
def test_shared_errors_carry_their_exit_codes(error, code):
    raised = error("what happened")
    assert isinstance(raised, DriftmapError)
    assert raised.exit_code is code
    assert int(code) in (11, 13)
    assert str(raised) == "what happened"
    assert (int(OutOfMemory("x").exit_code), int(Breaker("x").exit_code)) == (11, 13)
