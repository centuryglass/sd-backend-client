"""`ManagedComfyUI` lifecycle, driven through a fake process, health check and clock with no server or waits."""
import threading
from pathlib import Path
from typing import Callable, Optional

import pytest

from sd_backend_client import BackendLaunchError, BackendTimeoutError, ComfyUiWebservice, LaunchCancelledError, \
    ManagedComfyUI, ManagedState
from sd_backend_client.launch import managed_comfyui
from sd_backend_client.launch.managed_comfyui import server_arguments

from .fake_process import KILLED_EXIT_CODE, TERMINATED_EXIT_CODE, FakeProcessFactory

PORT = 8189
LAUNCH_COMMAND = ['/sd/venv/bin/python', '/sd/ComfyUI/main.py', '--lowvram']
SERVED_ARGV = ['/sd/ComfyUI/main.py', '--lowvram', '--listen', '127.0.0.1', '--port', str(PORT)]
EXIT_STATES = (ManagedState.STOPPED, ManagedState.EXITED, ManagedState.CRASHED)


class Harness:
    """A `ManagedComfyUI` with fakes, recording its log lines and states.

    `answers` holds what each health check returns, in order; an empty queue answers `None`. Each sleep advances the
    fake clock by its interval.
    """

    def __init__(self, launch_command: Optional[list[str]] = None, ignores_terminate: bool = False,
                 launch_error: Optional[OSError] = None) -> None:
        self.factory = FakeProcessFactory(ignores_terminate, launch_error)
        self.answers: list[Optional[list[str]] | Callable[[], Optional[list[str]]]] = []
        self.checked_urls: list[str] = []
        self.now = 0.0
        self.logs: list[str] = []
        self.states: list[ManagedState] = []
        self.exited = threading.Event()
        self.managed = ManagedComfyUI(launch_command or LAUNCH_COMMAND, '/sd/ComfyUI', PORT, env={'KEEP': '1'},
                                      on_log=self.logs.append, on_state=self._record_state,
                                      process_factory=self.factory, health_check=self._health_check,
                                      clock=lambda: self.now, sleep=self._sleep)

    def _record_state(self, state: ManagedState) -> None:
        self.states.append(state)
        if state in EXIT_STATES:
            self.exited.set()

    def _health_check(self, url: str) -> Optional[list[str]]:
        self.checked_urls.append(url)
        if not self.answers:
            return None
        answer = self.answers.pop(0)
        return answer() if callable(answer) else answer

    def _sleep(self, seconds: float) -> None:
        self.now += seconds

    def exit_process(self, code: int) -> None:
        """End the fake process and wait for the reader thread to record the exit."""
        self.exited.clear()
        self.factory.last.exit(code)
        assert self.exited.wait(), 'the reader thread never recorded the exit'


def test_start_then_ready():
    """The wait polls the URL until the server reports this process's arguments, then moves to READY."""
    harness = Harness()
    harness.answers = [None, None, SERVED_ARGV]
    harness.managed.start()
    harness.managed.wait_until_ready(timeout=10, poll_interval=1)
    assert harness.managed.state == ManagedState.READY
    assert harness.states == [ManagedState.STARTING, ManagedState.READY]
    assert harness.checked_urls == ['http://127.0.0.1:8189'] * 3
    assert harness.now == 2
    harness.managed.stop()


def test_start_sets_command_cwd_and_environment():
    """The process gets the rewritten command, the install folder and an unbuffered UTF-8 environment."""
    harness = Harness(['/sd/venv/bin/python', 'main.py', '--port', '8188', '--listen', '0.0.0.0', '--cpu'])
    harness.managed.start()
    process = harness.factory.last
    assert process.command == ['/sd/venv/bin/python', 'main.py', '--cpu', '--listen', '127.0.0.1', '--port', '8189']
    assert process.cwd == '/sd/ComfyUI'
    assert process.env == {'KEEP': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
    harness.managed.stop()


@pytest.mark.parametrize(('arguments', 'expected'), [
    ([], ['--listen', 'h', '--port', '1']),
    (['--port=8188', '--listen=0.0.0.0', '-x'], ['-x', '--listen', 'h', '--port', '1']),
    (['--listen', '--cpu'], ['--cpu', '--listen', 'h', '--port', '1']),
    (['--listen'], ['--listen', 'h', '--port', '1']),
    (['--port'], ['--listen', 'h', '--port', '1']),
])
def test_server_arguments_replace_listen_and_port(arguments: list[str], expected: list[str]):
    """Every form of `--listen` and `--port` in the command is replaced by the managed ones."""
    assert server_arguments(arguments, 'h', 1) == expected


def test_paths_and_ipv6_host():
    """Path arguments become strings, and an IPv6 host is bracketed in the URL."""
    managed = ManagedComfyUI([Path('/py'), Path('/sd/main.py')], Path('/sd'), 9000, host='::1')
    assert managed.command == [str(Path('/py')), str(Path('/sd/main.py')), '--listen', '::1', '--port', '9000']
    assert managed.url == 'http://[::1]:9000'


@pytest.mark.parametrize(('command', 'port'), [([], 8188), (LAUNCH_COMMAND, 0), (LAUNCH_COMMAND, 65536)])
def test_invalid_arguments_raise_value_error(command: list[str], port: int):
    """An empty command or an out-of-range port is rejected at construction."""
    with pytest.raises(ValueError):
        ManagedComfyUI(command, '/sd', port)


def test_interpreter_options_before_the_script_are_not_compared():
    """The portable build runs `python -s main.py`; ComfyUI's argv starts at the script."""
    harness = Harness(['python.exe', '-s', 'ComfyUI\\main.py', '--windows-standalone-build'])
    harness.answers = [['ComfyUI\\main.py', '--windows-standalone-build', '--listen', '127.0.0.1', '--port', '8189']]
    harness.managed.start()
    harness.managed.wait_until_ready(timeout=10)
    assert harness.managed.state == ManagedState.READY
    harness.managed.stop()


def test_a_command_without_a_script_compares_listen_and_port():
    """Without a `.py` script to anchor on, only the appended `--listen` and `--port` must match."""
    harness = Harness(['comfyui-launcher'])
    harness.answers = [['whatever', '--listen', '127.0.0.1', '--port', '8189']]
    harness.managed.start()
    harness.managed.wait_until_ready(timeout=10)
    assert harness.managed.state == ManagedState.READY
    harness.managed.stop()


def test_another_comfyui_on_the_port_times_out():
    """A server reporting other arguments is someone else's ComfyUI, and never counts as ready."""
    harness = Harness()
    harness.answers = [['main.py', '--port', '8189']] * 20
    harness.managed.start()
    with pytest.raises(BackendTimeoutError):
        harness.managed.wait_until_ready(timeout=5, poll_interval=1)
    assert harness.managed.state == ManagedState.STARTING
    assert harness.factory.last.terminate_count == 0
    harness.managed.stop()


def test_timeout_leaves_the_process_running():
    """A timeout raises without stopping the process, and a later wait can still succeed."""
    harness = Harness()
    harness.managed.start()
    with pytest.raises(BackendTimeoutError):
        harness.managed.wait_until_ready(timeout=3, poll_interval=1)
    assert harness.now == 3
    assert harness.managed.is_running
    harness.answers = [SERVED_ARGV]
    harness.managed.wait_until_ready(timeout=3)
    assert harness.managed.state == ManagedState.READY
    harness.managed.stop()


def test_exit_during_the_wait_crashes_and_raises():
    """A non-zero exit during the wait raises with the exit code and output, even if a late answer arrives."""
    harness = Harness()

    def crash() -> list[str]:
        harness.factory.last.emit('Port 8189 is already in use\n')
        harness.exit_process(1)
        return SERVED_ARGV  # A late answer after the exit doesn't make it ready.

    harness.answers = [None, crash]
    harness.managed.start()
    with pytest.raises(BackendLaunchError) as raised:
        harness.managed.wait_until_ready(timeout=10)
    assert raised.value.exit_code == 1
    assert raised.value.output_tail == ['Port 8189 is already in use']
    assert 'Port 8189 is already in use' in str(raised.value)
    assert harness.managed.state == ManagedState.CRASHED
    assert harness.managed.exit_code == 1


def test_clean_exit_without_stop_is_exited():
    """An exit with code 0 that `stop` didn't request is EXITED, not CRASHED."""
    harness = Harness()
    harness.managed.start()
    harness.exit_process(0)
    assert harness.managed.state == ManagedState.EXITED
    with pytest.raises(BackendLaunchError):
        harness.managed.wait_until_ready()


def test_crash_after_ready_then_restart():
    """A crash after READY is reported, and `start` launches a fresh process with a cleared exit record."""
    harness = Harness()
    harness.answers = [SERVED_ARGV]
    harness.managed.start()
    harness.managed.wait_until_ready()
    harness.exit_process(-9)
    assert harness.managed.state == ManagedState.CRASHED
    harness.answers = [SERVED_ARGV]
    harness.managed.start()
    assert harness.managed.exit_code is None
    assert not harness.managed.output_tail
    harness.managed.wait_until_ready()
    assert harness.states == [ManagedState.STARTING, ManagedState.READY, ManagedState.CRASHED,
                              ManagedState.STARTING, ManagedState.READY]
    assert len(harness.factory.processes) == 2
    harness.managed.stop()


def test_failed_launch_crashes_and_raises():
    """A command that can't be launched raises from `start` and leaves the state CRASHED."""
    harness = Harness(launch_error=FileNotFoundError(2, 'No such file or directory'))
    with pytest.raises(BackendLaunchError) as raised:
        harness.managed.start()
    assert raised.value.exit_code is None
    assert harness.managed.state == ManagedState.CRASHED
    with pytest.raises(BackendLaunchError, match='failed to launch'):
        harness.managed.wait_until_ready()


def test_start_while_running_raises():
    """Starting a running instance raises and launches nothing."""
    harness = Harness()
    harness.managed.start()
    with pytest.raises(RuntimeError):
        harness.managed.start()
    assert len(harness.factory.processes) == 1
    harness.managed.stop()


def test_stop_when_never_started_does_nothing():
    """Stopping a never-started instance changes nothing, and waiting on it raises."""
    harness = Harness()
    harness.managed.stop()
    assert not harness.states
    with pytest.raises(BackendLaunchError, match='never started'):
        harness.managed.wait_until_ready()


def test_stop_terminates():
    """A process that honours terminate is not killed, and a second stop does nothing."""
    harness = Harness()
    harness.managed.start()
    harness.managed.stop(grace_seconds=1)
    process = harness.factory.last
    assert (process.terminate_count, process.kill_count) == (1, 0)
    assert harness.managed.state == ManagedState.STOPPED
    assert harness.managed.exit_code == TERMINATED_EXIT_CODE
    harness.managed.stop()
    assert process.terminate_count == 1


def test_stop_kills_after_the_grace_period():
    """A process still running after the grace period is killed."""
    harness = Harness(ignores_terminate=True)
    harness.managed.start()
    harness.managed.stop(grace_seconds=1)
    process = harness.factory.last
    assert (process.terminate_count, process.kill_count) == (1, 1)
    assert harness.managed.state == ManagedState.STOPPED
    assert harness.managed.exit_code == KILLED_EXIT_CODE


def test_stop_during_the_wait_cancels_it():
    """A `stop` from another call during the wait ends it with LaunchCancelledError."""
    harness = Harness()
    harness.answers = [lambda: (harness.managed.stop(), None)[1], SERVED_ARGV]
    harness.managed.start()
    with pytest.raises(LaunchCancelledError):
        harness.managed.wait_until_ready()
    assert harness.managed.state == ManagedState.STOPPED


def test_cancel_during_the_wait():
    """Setting the cancel event ends the wait and leaves the process running."""
    harness = Harness()
    cancel = threading.Event()
    harness.answers = [lambda: (cancel.set(), None)[1]]
    harness.managed.start()
    with pytest.raises(LaunchCancelledError):
        harness.managed.wait_until_ready(cancel=cancel)
    assert harness.managed.state == ManagedState.STARTING
    harness.managed.stop()


def test_log_lines_reach_on_log_in_order_without_escapes():
    """Output lines reach `on_log` in order, without escape codes, including a final unterminated line."""
    harness = Harness()
    harness.managed.start()
    harness.factory.last.emit('\x1b[32m[INFO]\x1b[0m Starting server\n', 'progress 50%\n', 'no newline at exit')
    harness.exit_process(1)
    assert harness.logs == ['[INFO] Starting server', 'progress 50%', 'no newline at exit']
    assert harness.managed.output_tail == harness.logs


def test_callback_errors_are_logged_and_ignored(caplog: pytest.LogCaptureFixture):
    """A raising callback is logged, and output reading and stopping carry on."""
    def fail(_value: object) -> None:
        raise RuntimeError('callback failed')

    factory = FakeProcessFactory()
    managed = ManagedComfyUI(LAUNCH_COMMAND, '/sd', PORT, on_log=fail, on_state=fail, process_factory=factory)
    managed.start()
    factory.last.emit('line\n')
    managed.stop()
    assert managed.state == ManagedState.STOPPED
    assert managed.output_tail == ['line']
    assert 'callback failed' in caplog.text


def test_default_wait_ends_early_on_exit():
    """Without an injected sleep, the wait between polls ends as soon as the process exits."""
    factory = FakeProcessFactory()
    managed = ManagedComfyUI(LAUNCH_COMMAND, '/sd', PORT, process_factory=factory,
                             health_check=lambda _url: factory.last.exit(1))
    managed.start()
    with pytest.raises(BackendLaunchError):
        managed.wait_until_ready(timeout=600, poll_interval=600)


def test_connect_returns_a_client_for_the_url():
    """`connect` returns a ComfyUI client for the managed URL, passing keyword arguments on."""
    service = ManagedComfyUI(LAUNCH_COMMAND, '/sd', PORT).connect(live_progress=False)
    assert isinstance(service, ComfyUiWebservice)
    assert service.server_url == 'http://127.0.0.1:8189'


def test_comfyui_argv_reads_system_stats(monkeypatch: pytest.MonkeyPatch):
    """The default health check returns the `argv` from `/system_stats`."""
    class Stats:
        """The one field of `SystemStatResponse` the health check reads."""

        class system:  # pylint: disable=invalid-name
            """`SystemObject` with only `argv`."""

            argv = ('main.py', '--port', '1')

    monkeypatch.setattr(ComfyUiWebservice, 'get_system_stats', lambda _self: Stats())
    assert managed_comfyui.comfyui_argv('http://127.0.0.1:1') == ['main.py', '--port', '1']


def test_comfyui_argv_is_none_when_nothing_answers(monkeypatch: pytest.MonkeyPatch):
    """The default health check returns None when the connection fails."""
    def refuse(_self: ComfyUiWebservice) -> None:
        raise ConnectionRefusedError()

    monkeypatch.setattr(ComfyUiWebservice, 'get_system_stats', refuse)
    assert managed_comfyui.comfyui_argv('http://127.0.0.1:1') is None
