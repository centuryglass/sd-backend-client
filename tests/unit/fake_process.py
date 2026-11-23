"""A fake child process for `ManagedComfyUI` tests: output and exits the test drives, with no real process or waits."""
import queue
import subprocess
import threading
from typing import Iterator, Optional

TERMINATED_EXIT_CODE = -15
KILLED_EXIT_CODE = -9


class _FakeStdout:
    """Yields the lines `FakeProcess.emit` queues until the process exits."""

    def __init__(self) -> None:
        self._lines: queue.Queue[Optional[str]] = queue.Queue()
        self.closed = False

    def put(self, text: Optional[str]) -> None:
        """Queue a line, or `None` to end the output."""
        self._lines.put(text)

    def __iter__(self) -> Iterator[str]:
        while (line := self._lines.get()) is not None:
            yield line

    def close(self) -> None:
        """Record that the reader closed the pipe."""
        self.closed = True


class FakeProcess:
    """Stands in for `subprocess.Popen`. The test emits output with `emit` and ends the process with `exit`.

    `wait` with a timeout never sleeps: it raises `subprocess.TimeoutExpired` at once if the process is still
    running. With `ignores_terminate`, `terminate` is recorded but the process keeps running until killed.
    """

    def __init__(self, command: list[str], cwd: str, env: dict[str, str], ignores_terminate: bool = False) -> None:
        self.command = command
        self.cwd = cwd
        self.env = env
        self.ignores_terminate = ignores_terminate
        self.stdout = _FakeStdout()
        self.returncode: Optional[int] = None
        self.terminate_count = 0
        self.kill_count = 0
        self._exited = threading.Event()

    def emit(self, *lines: str) -> None:
        """Queue output lines, each with its line break, as text-mode `stdout` returns them."""
        for line in lines:
            self.stdout.put(line)

    def exit(self, code: int) -> None:
        """End the process with exit code `code`, closing its output."""
        if self._exited.is_set():
            return
        self.returncode = code
        self._exited.set()
        self.stdout.put(None)

    def poll(self) -> Optional[int]:
        """Return the exit code, or `None` while running."""
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        """Return the exit code, raising at once if a timeout is given and the process is running."""
        if timeout is not None and not self._exited.is_set():
            raise subprocess.TimeoutExpired(self.command, timeout)
        self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        """Record the request, and exit unless `ignores_terminate` is set."""
        self.terminate_count += 1
        if not self.ignores_terminate:
            self.exit(TERMINATED_EXIT_CODE)

    def kill(self) -> None:
        """Record the kill and exit."""
        self.kill_count += 1
        self.exit(KILLED_EXIT_CODE)


class FakeProcessFactory:
    """A `ManagedComfyUI` `process_factory` that records each `FakeProcess` it creates, or raises `error`."""

    def __init__(self, ignores_terminate: bool = False, error: Optional[OSError] = None) -> None:
        self.ignores_terminate = ignores_terminate
        self.error = error
        self.processes: list[FakeProcess] = []

    @property
    def last(self) -> FakeProcess:
        """The most recently created process."""
        return self.processes[-1]

    def __call__(self, command: list[str], cwd: str, env: dict[str, str]) -> FakeProcess:
        if self.error is not None:
            raise self.error
        process = FakeProcess(command, cwd, env, self.ignores_terminate)
        self.processes.append(process)
        return process
