"""`ManagedComfyUI` against a real child process: the stub `tests/stub_comfyui/main.py` with the real health check.

Opt-in with `--run-process` or `RUN_SD_PROCESS=1`, since each test waits on a process starting and stopping.
"""
import signal
import socket
import sys
import threading
from pathlib import Path

import pytest

from sd_backend_client import BackendLaunchError, ManagedComfyUI, ManagedState

pytestmark = pytest.mark.process

STUB_DIR = Path(__file__).parent / 'stub_comfyui'
READY_TIMEOUT = 30.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _managed(*arguments: str, logs: list[str]) -> ManagedComfyUI:
    command = [sys.executable, str(STUB_DIR / 'main.py'), *arguments]
    return ManagedComfyUI(command, STUB_DIR, _free_port(), on_log=logs.append)


def test_start_ready_stop():
    """The stub starts, serves its own arguments, logs its output line by line, and stops on terminate."""
    logs: list[str] = []
    managed = _managed('--port', '1', logs=logs)
    managed.start()
    try:
        managed.wait_until_ready(timeout=READY_TIMEOUT, poll_interval=0.1)
        assert managed.state == ManagedState.READY
        assert managed.connect(live_progress=False).get_system_stats().system.argv[-1] == str(managed.command[-1])
    finally:
        managed.stop(grace_seconds=10)
    assert managed.state == ManagedState.STOPPED
    assert logs[:3] == ['[INFO] Starting stub server', 'loading 50%', 'loading 100%']


def test_crash_before_ready():
    """A stub that exits before serving fails the wait with its exit code and last output line."""
    logs: list[str] = []
    managed = _managed('--crash', logs=logs)
    managed.start()
    with pytest.raises(BackendLaunchError) as raised:
        managed.wait_until_ready(timeout=READY_TIMEOUT, poll_interval=0.1)
    assert raised.value.exit_code == 3
    assert raised.value.output_tail[-1] == 'Stub crash requested'
    assert managed.state == ManagedState.CRASHED


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows has no SIGTERM to ignore: terminate already kills')
def test_stop_kills_a_process_ignoring_terminate():
    """A stub that ignores SIGTERM is killed after the grace period."""
    logs: list[str] = []
    managed = _managed('--ignore-term', logs=logs)
    managed.start()
    managed.wait_until_ready(timeout=READY_TIMEOUT, poll_interval=0.1)
    managed.stop(grace_seconds=0.5)
    assert managed.state == ManagedState.STOPPED
    assert managed.exit_code == -signal.SIGKILL


def test_another_server_on_the_port_is_not_ready():
    """A second instance on a port the first holds fails to bind and crashes, and never counts as ready."""
    logs: list[str] = []
    first = _managed(logs=logs)
    second = ManagedComfyUI([sys.executable, str(STUB_DIR / 'main.py'), '--lowvram-stub'], STUB_DIR,
                            int(first.command[-1]), on_log=logs.append)
    first.start()
    try:
        first.wait_until_ready(timeout=READY_TIMEOUT, poll_interval=0.1)
        second.start()
        cancel = threading.Event()
        with pytest.raises(BackendLaunchError):
            second.wait_until_ready(timeout=READY_TIMEOUT, cancel=cancel, poll_interval=0.1)
        assert second.state == ManagedState.CRASHED
    finally:
        second.stop()
        first.stop()
