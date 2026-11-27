"""Starts, health-checks, streams output from and stops a ComfyUI install on disk.

`ManagedComfyUI` runs the install with `subprocess` and owns only the process it started: it never stops or reports
on a ComfyUI that something else started. `start` refuses a port where a ComfyUI already answers, and readiness means
the server on the port reports this process's own arguments, so another ComfyUI on the port never counts as ready.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from enum import Enum
from typing import IO, Any, Callable, Mapping, Optional, Protocol, Sequence

from pydantic import ValidationError

from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.errors import BackendLaunchError, BackendTimeoutError, LaunchCancelledError, SDBackendError

logger = logging.getLogger(__name__)

DEFAULT_HOST = '127.0.0.1'
DEFAULT_READY_TIMEOUT = 300.0
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_STOP_GRACE = 10.0
HEALTH_CHECK_REQUEST_TIMEOUT = 5.0
OUTPUT_TAIL_LINES = 50

# ComfyUI arguments that `ManagedComfyUI` sets itself. `--listen`'s value is optional.
_PORT_ARG = '--port'
_LISTEN_ARG = '--listen'

# Terminal escape sequences, such as the color codes ComfyUI's log uses: CSI, OSC, and other two-character escapes.
_TERMINAL_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]')


class ManagedState(Enum):
    """The lifecycle state of a `ManagedComfyUI`'s process."""
    STOPPED = 'stopped'
    """Not started, or ended by `ManagedComfyUI.stop`."""
    STARTING = 'starting'
    """Running, but not yet confirmed serving by `ManagedComfyUI.wait_until_ready`."""
    READY = 'ready'
    """Running and serving at `ManagedComfyUI.url`."""
    EXITED = 'exited'
    """Exited with code 0 without `ManagedComfyUI.stop` being called."""
    CRASHED = 'crashed'
    """Exited with a non-zero code, or failed to launch, without `ManagedComfyUI.stop` being called."""


class ManagedProcess(Protocol):
    """The part of `subprocess.Popen` that `ManagedComfyUI` uses. Tests pass a fake through `process_factory`."""
    stdout: Optional[IO[str]]

    def poll(self) -> Optional[int]:
        """Return the exit code, or `None` while running."""

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for exit and return the exit code, raising `subprocess.TimeoutExpired` after `timeout`."""

    def terminate(self) -> None:
        """Ask the process to exit."""

    def kill(self) -> None:
        """End the process at once."""


# Takes the command, working directory and environment, and returns a started process with text-mode `stdout`.
ProcessFactory = Callable[[list[str], str, dict[str, str]], ManagedProcess]
# Takes a server URL and returns the `argv` the ComfyUI there reports, or `None` if no ComfyUI answers. Blocks.
HealthCheck = Callable[[str], Optional[list[str]]]
LogCallback = Callable[[str], None]
StateCallback = Callable[[ManagedState], None]


def comfyui_argv(url: str) -> Optional[list[str]]:
    """Return the `argv` the ComfyUI server at `url` reports, or `None` if no ComfyUI answers there. Blocks."""
    webservice = ComfyUiWebservice(url, request_timeout=HEALTH_CHECK_REQUEST_TIMEOUT, live_progress=False)
    try:
        return list(webservice.get_system_stats().system.argv)
    except (SDBackendError, ValidationError, ValueError, OSError):
        return None
    finally:
        webservice.disconnect()


def _popen(command: list[str], cwd: str, env: dict[str, str]) -> ManagedProcess:
    # Universal newlines make `\r` progress updates and `\r\n` line breaks each end a line.
    return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)


def server_arguments(arguments: Sequence[str], host: str, port: int) -> list[str]:
    """Return `arguments` with any `--port` or `--listen` replaced by ones serving on `host` and `port`."""
    kept: list[str] = []
    remaining = list(arguments)
    while remaining:
        arg = remaining.pop(0)
        if arg == _PORT_ARG:
            if remaining:
                remaining.pop(0)
        elif arg == _LISTEN_ARG:
            if remaining and not remaining[0].startswith('-'):
                remaining.pop(0)
        elif not arg.startswith(f'{_PORT_ARG}=') and not arg.startswith(f'{_LISTEN_ARG}='):
            kept.append(arg)
    return [*kept, _LISTEN_ARG, host, _PORT_ARG, str(port)]


def _script_arguments(arguments: Sequence[str]) -> list[str]:
    """Return the arguments after the first `.py` script, which are the `argv[1:]` ComfyUI reports.

    Interpreter options before the script, such as the portable build's `-s`, are not in ComfyUI's `argv`. Without a
    script, only the `--listen` and `--port` arguments `server_arguments` appends are returned.
    """
    for index, arg in enumerate(arguments):
        if arg.endswith('.py'):
            return list(arguments[index + 1:])
    return list(arguments[-4:])


class ManagedComfyUI:
    """One ComfyUI install on disk, run as a child process serving on `host` and `port`.

    `start` launches the process in the `STARTING` state, unless a ComfyUI already answers on the port, and
    `wait_until_ready` blocks until it serves, moving it to `READY`. `stop` ends it. A process that exits without
    `stop` moves to `EXITED` or `CRASHED`. Only the process this instance started is ever stopped. Destroying a
    running instance leaves its process running, so the owner calls `stop` before quitting.

    Callbacks run on the thread that caused the change, and must not block:

    - `on_log` runs on a reader thread this class starts, once per line of output.
    - `on_state` runs on the reader thread when the process exits, and on the caller's thread for `start`,
      `wait_until_ready` and `stop`.

    A GUI caller has to hand both over to its own thread. An exception a callback raises is logged and ignored.

    On Windows, `subprocess` terminates a process at once, so `stop` gives ComfyUI no chance to shut down.
    """

    def __init__(self, launch_command: Sequence[str | os.PathLike[str]], cwd: str | os.PathLike[str], port: int,
                 host: str = DEFAULT_HOST, env: Optional[Mapping[str, str]] = None,
                 on_log: Optional[LogCallback] = None, on_state: Optional[StateCallback] = None, *,
                 model_paths_config: str | os.PathLike[str] | None = None,
                 process_factory: ProcessFactory = _popen, health_check: HealthCheck = comfyui_argv,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Optional[Callable[[float], None]] = None) -> None:
        """Set up the install to run, without starting it.

        Parameters
        ----------
        launch_command: Sequence[str | os.PathLike[str]]
            The interpreter, the entry point and any arguments, such as `DetectedInstall.launch_command`. The
            interpreter always comes from here, never from `sys.executable`. Any `--port` or `--listen` in it is
            replaced.
        cwd: str | os.PathLike[str]
            The process's working directory, normally the install's folder.
        port: int
            The port the server listens on.
        host: str, default='127.0.0.1'
            The address the server listens on.
        env: Mapping[str, str], optional
            The process's environment. Defaults to this process's environment. `PYTHONUNBUFFERED` and
            `PYTHONIOENCODING` are always set, so output arrives line by line as UTF-8.
        on_log: Callable[[str], None], optional
            Called with each line of output, without the line break or terminal escape sequences.
        on_state: Callable[[ManagedState], None], optional
            Called with each new state.
        model_paths_config: str | os.PathLike[str], optional
            An additional, caller-managed extra model paths file. Existing ComfyUI configurations are still loaded.
        process_factory, health_check, clock, sleep
            Replace the process launcher, the readiness check, `time.monotonic` and the wait between readiness polls.
            Tests use these to run without a server. The default wait ends early when the process exits.

        Raises
        ------
        ValueError
            If `launch_command` is empty or `port` is out of range.
        """
        if len(launch_command) == 0:
            raise ValueError('launch_command is empty: it needs at least an interpreter or executable')
        if not 0 < port < 65536:
            raise ValueError(f'port must be between 1 and 65535, got {port}')
        self._program = os.fspath(launch_command[0])
        arguments = [os.fspath(arg) for arg in launch_command[1:]]
        if model_paths_config is not None:
            arguments.extend(('--extra-model-paths-config', os.fspath(model_paths_config)))
        self._arguments = server_arguments(arguments, host, port)
        self._expected_argv_tail = _script_arguments(self._arguments)
        self._cwd = os.fspath(cwd)
        self._env = dict(os.environ if env is None else env)
        self._env['PYTHONUNBUFFERED'] = '1'
        self._env['PYTHONIOENCODING'] = 'utf-8'
        self._host = host
        self._port = port
        self._on_log = on_log
        self._on_state = on_state
        self._process_factory = process_factory
        self._health_check = health_check
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._state = ManagedState.STOPPED
        self._process: Optional[ManagedProcess] = None
        self._reader: Optional[threading.Thread] = None
        self._exited = threading.Event()
        self._stopping = False
        self._exit_code: Optional[int] = None
        self._output_tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)

    @property
    def command(self) -> list[str]:
        """The full command `start` runs, with the `--listen` and `--port` arguments set."""
        return [self._program, *self._arguments]

    @property
    def state(self) -> ManagedState:
        """The process's current lifecycle state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether this instance's process is running, in the `STARTING` or `READY` state."""
        return self._state in (ManagedState.STARTING, ManagedState.READY)

    @property
    def exit_code(self) -> Optional[int]:
        """The last process's exit code, or `None` if it is running or never started."""
        return self._exit_code

    @property
    def output_tail(self) -> list[str]:
        """The last lines of the current or last process's output, oldest first."""
        with self._lock:
            return list(self._output_tail)

    @property
    def url(self) -> str:
        """The URL the server answers at once ready."""
        host = f'[{self._host}]' if ':' in self._host else self._host
        return f'http://{host}:{self._port}'

    def connect(self, **kwargs: Any) -> ComfyUiWebservice:
        """Return a `ComfyUiWebservice` for `url`. Keyword arguments go to its constructor."""
        return ComfyUiWebservice(self.url, **kwargs)

    def checkpoints_available(self) -> bool:
        """Check whether the running ComfyUI lists a checkpoint, without loading a model."""
        webservice = self.connect(live_progress=False, request_timeout=HEALTH_CHECK_REQUEST_TIMEOUT)
        try:
            return bool(webservice.list_checkpoints())
        finally:
            webservice.disconnect()

    def start(self) -> None:
        """Launch the process in the `STARTING` state and return without waiting for it to serve.

        Runs one health check first, which blocks for up to a few seconds. An instance whose process ended can start
        again.

        Raises
        ------
        RuntimeError
            If this instance's process is already running.
        BackendLaunchError
            If a ComfyUI already answers at `url`, with no process launched and the state unchanged. Or if the process
            could not be launched, such as when the interpreter is missing, with the state `CRASHED`.
        """
        if self.is_running:
            raise RuntimeError(f'ComfyUI at {self.url} is already running')
        # A ComfyUI started with the same arguments would pass `wait_until_ready` as this process.
        if self._health_check(self.url) is not None:
            raise BackendLaunchError(f'A ComfyUI server is already running at {self.url}. Connect to it, stop it, or '
                                     'start this one on another port.', None, [])
        # The last reader thread has set an exit state, so it is finishing and the join is short:
        self._join_reader()
        with self._lock:
            if self.is_running:
                raise RuntimeError(f'ComfyUI at {self.url} is already running')
            self._stopping = False
            self._exit_code = None
            self._output_tail.clear()
            self._exited.clear()
            command = self.command
            logger.info('Starting ComfyUI: %s', subprocess.list2cmdline(command))
            try:
                process = self._process_factory(command, self._cwd, self._env)
            except OSError as err:
                self._process = None
                self._reader = None
                self._exited.set()
                self._set_state(ManagedState.CRASHED)
                raise BackendLaunchError(f'Could not launch ComfyUI with {command[0]!r} in {self._cwd!r}: {err}',
                                         None, []) from err
            self._process = process
            self._reader = threading.Thread(target=self._read_output, args=(process,),
                                            name=f'ComfyUI output :{self._port}', daemon=True)
            self._set_state(ManagedState.STARTING)
            self._reader.start()

    def wait_until_ready(self, timeout: float = DEFAULT_READY_TIMEOUT, cancel: Optional[threading.Event] = None,
                         poll_interval: float = DEFAULT_POLL_INTERVAL) -> None:
        """Block until the server reports this process's arguments, then move to `READY`.

        Returns at once if already `READY`. A timeout leaves the process running in the `STARTING` state.

        Raises
        ------
        BackendLaunchError
            If the process exits before it serves, or was not running. Its `exit_code` and `output_tail` say why.
        BackendTimeoutError
            If `timeout` seconds pass first.
        LaunchCancelledError
            If `cancel` is set or `stop` is called first.
        """
        deadline = self._clock() + timeout
        while True:
            if cancel is not None and cancel.is_set():
                raise LaunchCancelledError(f'Stopped waiting for ComfyUI at {self.url}: cancelled')
            with self._lock:
                if self._state == ManagedState.READY:
                    return
                if self._stopping or (self._state == ManagedState.STOPPED and self._exit_code is not None):
                    raise LaunchCancelledError(f'Stopped waiting for ComfyUI at {self.url}: it was stopped')
                if self._state != ManagedState.STARTING:
                    raise self._launch_error()
            argv = self._health_check(self.url)
            with self._lock:
                if argv is not None and self._state == ManagedState.STARTING and not self._stopping \
                        and self._is_own_argv(argv):
                    self._set_state(ManagedState.READY)
                    return
            if self._clock() >= deadline:
                raise BackendTimeoutError(f'ComfyUI at {self.url} did not start serving within {timeout} seconds. '
                                          'It is still running.')
            if self._sleep is not None:
                self._sleep(poll_interval)
            else:
                self._exited.wait(poll_interval)

    def stop(self, grace_seconds: float = DEFAULT_STOP_GRACE) -> None:
        """Ask the process to exit, kill it if it hasn't after `grace_seconds`, and wait for it to end.

        Moves to `STOPPED`. Does nothing if this instance's process isn't running.
        """
        with self._lock:
            process = self._process
            if process is None or not self.is_running or self._stopping:
                return
            self._stopping = True
        logger.info('Stopping ComfyUI at %s', self.url)
        process.terminate()
        try:
            process.wait(grace_seconds)
        except subprocess.TimeoutExpired:
            logger.warning('ComfyUI did not exit within %s seconds, killing it', grace_seconds)
            process.kill()
            process.wait()
        self._join_reader()

    def _is_own_argv(self, argv: Sequence[str]) -> bool:
        tail = self._expected_argv_tail
        return len(argv) > len(tail) and list(argv[len(argv) - len(tail):]) == tail

    def _launch_error(self) -> BackendLaunchError:
        tail = self.output_tail
        if self._process is None:
            started = 'failed to launch' if self._state == ManagedState.CRASHED else 'was never started'
            message = f'ComfyUI at {self.url} {started}'
        else:
            message = f'ComfyUI exited with code {self._exit_code} before serving at {self.url}'
            if tail:
                message += f'. Last output: {tail[-1]}'
        return BackendLaunchError(message, self._exit_code, tail)

    def _read_output(self, process: ManagedProcess) -> None:
        """Pass each output line to `on_log` until the pipe closes, then record the exit. Runs on the reader thread."""
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = _TERMINAL_ESCAPE_PATTERN.sub('', raw_line.rstrip('\r\n'))
                with self._lock:
                    self._output_tail.append(line)
                if self._on_log is not None:
                    try:
                        self._on_log(line)
                    except Exception:  # pylint: disable=broad-exception-caught
                        logger.exception('ComfyUI log callback failed')
            process.stdout.close()
        exit_code = process.wait()
        with self._lock:
            self._exit_code = exit_code
            if self._stopping:
                logger.info('ComfyUI stopped (exit code %s)', exit_code)
                state = ManagedState.STOPPED
            elif exit_code == 0:
                logger.info('ComfyUI exited')
                state = ManagedState.EXITED
            else:
                logger.error('ComfyUI exited unexpectedly with code %s', exit_code)
                state = ManagedState.CRASHED
            self._stopping = False
            self._exited.set()
            self._set_state(state)

    def _join_reader(self) -> None:
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join()

    def _set_state(self, state: ManagedState) -> None:
        if state == self._state:
            return
        self._state = state
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception('ComfyUI state callback failed')
