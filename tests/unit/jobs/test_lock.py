import subprocess
import sys

import pytest

from driftmap.core.schemas import ExitCode
from driftmap.jobs.lock import LockHeld, launch_lock

HOLDER = """
import sys
from driftmap.jobs.lock import launch_lock
with launch_lock(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.read()
"""


def test_second_launcher_is_refused_while_another_process_holds_the_lock(tmp_path, child_env):
    path = tmp_path / "state" / "launch.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path)],
        env=child_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline() == "held\n"
        with pytest.raises(LockHeld, match="another launch holds the lock") as caught, launch_lock(path):
            pytest.fail("entered a held lock")
        assert caught.value.exit_code is ExitCode.GUARD_REFUSAL == 4
        holder.kill()
        holder.wait(timeout=30)
        # the kernel drops a dead holder's lock
        with launch_lock(path):
            pass
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_lock_is_exclusive_and_released_on_exit(tmp_path):
    path = tmp_path / "launch.lock"
    with launch_lock(path):
        with pytest.raises(LockHeld), launch_lock(path):
            pytest.fail("entered a held lock")
    with pytest.raises(RuntimeError, match="boom"), launch_lock(path):
        raise RuntimeError("boom")
    with launch_lock(path):
        pass
