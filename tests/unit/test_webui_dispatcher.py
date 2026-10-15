"""Offline tests for WebUIDispatcher and WebUIGenerationHandle: job lifecycle, FIFO order, cancellation, idle-wait.

Jobs are fake generation calls that block on a `threading.Event`, so each test steps the worker thread through its
states explicitly. `Event.wait` timeouts below are safety bounds that only expire when a test is already failing.
"""
# pylint: disable=protected-access
import threading
from typing import Any, Optional

import pytest
from PIL import Image

from sd_backend_client.api.shared_data.generation_handle import GenerationError, GenerationStatus
from sd_backend_client.errors import BackendTimeoutError, SDBackendError, ServerError
from sd_backend_client.api.webui import webui_generation_handle
from sd_backend_client.api.webui.response_formats import ProgressResponseBody
from sd_backend_client.api.webui.webui_generation_handle import WebUIDispatcher, WebUIGenerationHandle, create_task_id

SAFETY_TIMEOUT_S = 5.0


def _progress(job_count: int = 0, progress: float = 0.0, textinfo: Optional[str] = None) -> ProgressResponseBody:
    """A /sdapi/v1/progress response body with the given queue length and progress fraction."""
    return ProgressResponseBody.model_validate({
        'progress': progress,
        'eta_relative': 1.5,
        'state': {'skipped': False, 'interrupted': False, 'stopping_generation': False, 'job': '',
                  'job_count': job_count, 'job_timestamp': '0', 'job_no': 0, 'sampling_step': 0,
                  'sampling_steps': 0},
        'current_image': None,
        'textinfo': textinfo,
    })


class _FakeService:
    """Stands in for A1111Webservice's progress_check and interrupt, replaying scripted progress results."""

    def __init__(self, progress_script: Optional[list[Any]] = None) -> None:
        self._script = list(progress_script) if progress_script is not None else [_progress()]
        self.progress_calls = 0
        self.interrupts = 0

    def progress_check(self) -> ProgressResponseBody:
        """Return the next scripted response, repeating the last; scripted exceptions are raised."""
        entry = self._script[min(self.progress_calls, len(self._script) - 1)]
        self.progress_calls += 1
        if isinstance(entry, BaseException):
            raise entry
        return entry

    def interrupt(self) -> dict:
        """Record the interrupt request."""
        self.interrupts += 1
        return {}


class _BlockingJob:
    """A generation call that signals when it starts, then blocks until released."""

    def __init__(self, name: str, run_log: list[str]) -> None:
        self.name = name
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._run_log = run_log

    def __call__(self) -> dict[str, Any]:
        self.calls += 1
        self._run_log.append(self.name)
        self.started.set()
        assert self.release.wait(SAFETY_TIMEOUT_S), f'job {self.name} was never released'
        return {'images': [Image.new('RGBA', (1, 1))], 'info': self.name}


def _dispatcher(service: Optional[_FakeService] = None) -> WebUIDispatcher:
    return WebUIDispatcher(service if service is not None else _FakeService(), wait_for_idle=False)


def _wait_done(handle: WebUIGenerationHandle) -> None:
    assert handle._done_event.wait(SAFETY_TIMEOUT_S), f'{handle.task_id} never reached a terminal state'


def test_create_task_id_matches_webui_format():
    """Task ids follow the server's task(<type>-XXXXXXX) shape and are distinct."""
    task_ids = {create_task_id('txt2img') for _ in range(20)}
    assert len(task_ids) == 20
    for task_id in task_ids:
        assert task_id.startswith('task(txt2img-') and task_id.endswith(')')
        assert len(task_id) == len('task(txt2img-)') + 7


def test_job_moves_pending_to_active_to_finished():
    """A job reports ACTIVE with server progress while running, then FINISHED with its images."""
    service = _FakeService([_progress(job_count=1, progress=0.25, textinfo='Sampling')])
    run_log: list[str] = []
    job = _BlockingJob('a', run_log)
    handle = _dispatcher(service).submit(job, 'task(a)')

    assert job.started.wait(SAFETY_TIMEOUT_S)
    active = handle.poll()
    assert active.status is GenerationStatus.ACTIVE
    assert active.progress == 0.25
    assert active.overall_progress == 0.25  # The WebUI's progress already spans the whole job.
    assert active.eta_seconds == 1.5
    assert active.text_info == 'Sampling'

    job.release.set()
    result = handle.wait(timeout=SAFETY_TIMEOUT_S)
    assert handle.poll().status is GenerationStatus.FINISHED
    assert result.raw_info == 'a'
    assert result.task_id == 'task(a)'
    assert len(result.images) == 1


def test_jobs_run_one_at_a_time_in_fifo_order():
    """Queued jobs stay PENDING with their queue index until the running job ends, then run in submit order."""
    run_log: list[str] = []
    jobs = [_BlockingJob(name, run_log) for name in 'abc']
    dispatcher = _dispatcher()
    handles = [dispatcher.submit(job, f'task({job.name})') for job in jobs]

    assert jobs[0].started.wait(SAFETY_TIMEOUT_S)
    pending = [handle.poll() for handle in handles[1:]]
    assert [progress.status for progress in pending] == [GenerationStatus.PENDING] * 2
    assert [progress.queue_index for progress in pending] == [0, 1]

    for job, handle in zip(jobs, handles):
        assert job.started.wait(SAFETY_TIMEOUT_S)
        job.release.set()
        _wait_done(handle)
    assert run_log == ['a', 'b', 'c']
    assert all(handle.poll().status is GenerationStatus.FINISHED for handle in handles)


def test_cancel_pending_job_never_runs_it():
    """Cancelling a client-queued job finalizes it CANCELLED at once and never sends its request."""
    service = _FakeService()
    run_log: list[str] = []
    first, second, third = (_BlockingJob(name, run_log) for name in 'abc')
    dispatcher = _dispatcher(service)
    first_handle = dispatcher.submit(first, 'task(a)')
    second_handle = dispatcher.submit(second, 'task(b)')
    third_handle = dispatcher.submit(third, 'task(c)')
    assert first.started.wait(SAFETY_TIMEOUT_S)

    assert second_handle.cancel() is True
    assert second_handle.poll().status is GenerationStatus.CANCELLED
    assert third_handle.poll().queue_index == 0
    with pytest.raises(GenerationError) as error:
        second_handle.wait(timeout=SAFETY_TIMEOUT_S)
    assert error.value.status is GenerationStatus.CANCELLED

    first.release.set()
    assert third.started.wait(SAFETY_TIMEOUT_S)
    third.release.set()
    _wait_done(third_handle)
    _wait_done(first_handle)
    assert second.calls == 0
    assert run_log == ['a', 'c']
    assert service.interrupts == 0


def test_cancel_active_job_interrupts_server_and_discards_result():
    """Cancelling the running job sends an interrupt, and the job ends CANCELLED even if the POST returns images."""
    service = _FakeService()
    job = _BlockingJob('a', [])
    handle = _dispatcher(service).submit(job, 'task(a)')
    assert job.started.wait(SAFETY_TIMEOUT_S)

    assert handle.cancel() is True
    assert service.interrupts == 1
    job.release.set()
    with pytest.raises(GenerationError) as error:
        handle.wait(timeout=SAFETY_TIMEOUT_S)
    assert error.value.status is GenerationStatus.CANCELLED


def test_cancel_active_uninterruptible_job_is_refused_and_job_finishes():
    """An uninterruptible job refuses cancel() once running, sends no interrupt, and still delivers its result."""
    service = _FakeService()
    job = _BlockingJob('a', [])
    handle = _dispatcher(service).submit(job, 'task(a)', interruptible=False)
    assert job.started.wait(SAFETY_TIMEOUT_S)

    assert handle.cancel() is False
    assert service.interrupts == 0
    job.release.set()
    assert handle.wait(timeout=SAFETY_TIMEOUT_S).raw_info == 'a'


def test_cancel_pending_uninterruptible_job_never_runs_it():
    """An uninterruptible job is still cancellable while it waits in the client-side queue."""
    run_log: list[str] = []
    dispatcher = _dispatcher()
    blocker = _BlockingJob('blocker', run_log)
    dispatcher.submit(blocker, 'task(blocker)')
    assert blocker.started.wait(SAFETY_TIMEOUT_S)
    queued = _BlockingJob('queued', run_log)
    handle = dispatcher.submit(queued, 'task(queued)', interruptible=False)

    assert handle.cancel() is True
    blocker.release.set()
    _wait_done(handle)
    assert handle.poll().status is GenerationStatus.CANCELLED
    assert queued.calls == 0


def test_cancel_after_finish_is_rejected():
    """cancel() returns False once the job is terminal, and sends nothing."""
    service = _FakeService()
    job = _BlockingJob('a', [])
    job.release.set()
    handle = _dispatcher(service).submit(job, 'task(a)')
    handle.wait(timeout=SAFETY_TIMEOUT_S)

    assert handle.cancel() is False
    assert handle.poll().status is GenerationStatus.FINISHED
    assert service.interrupts == 0


def test_failed_job_reports_error_and_next_job_still_runs():
    """An exception from the generation call ends the job FAILED with its message; the worker keeps going."""
    cause = ServerError(500, 'out of memory', '/sdapi/v1/txt2img', 'POST')

    def failing_call() -> dict[str, Any]:
        raise cause

    dispatcher = _dispatcher()
    failed = dispatcher.submit(failing_call, 'task(fail)')
    next_job = _BlockingJob('next', [])
    next_job.release.set()
    following = dispatcher.submit(next_job, 'task(next)')

    with pytest.raises(GenerationError, match='out of memory') as error:
        failed.wait(timeout=SAFETY_TIMEOUT_S)
    assert isinstance(error.value, SDBackendError)
    assert error.value.status is GenerationStatus.FAILED
    assert error.value.__cause__ is cause
    assert failed.poll().text_info == str(cause)
    assert following.wait(timeout=SAFETY_TIMEOUT_S).raw_info == 'next'


def test_wait_times_out_while_job_runs():
    """wait() raises BackendTimeoutError, which is also a TimeoutError, when the timeout elapses first."""
    job = _BlockingJob('a', [])
    handle = _dispatcher().submit(job, 'task(a)')
    try:
        with pytest.raises(BackendTimeoutError):
            handle.wait(timeout=0)
    finally:
        job.release.set()
    _wait_done(handle)


def test_wait_reports_progress_snapshots():
    """wait() passes the final FINISHED snapshot to on_progress."""
    job = _BlockingJob('a', [])
    job.release.set()
    handle = _dispatcher().submit(job, 'task(a)')
    snapshots = []
    handle.wait(timeout=SAFETY_TIMEOUT_S, poll_interval=0.01, on_progress=snapshots.append)
    assert snapshots[-1].status is GenerationStatus.FINISHED


# Idle-wait tests call WebUIDispatcher._run_job on the test thread with a fake clock, so they need no worker thread
# and no real sleeps.

class _FakeClock:
    """Replaces the `time` module inside webui_generation_handle: sleep advances monotonic time and is recorded."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        """Current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance fake time without blocking."""
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(name='clock')
def _clock_fixture(monkeypatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(webui_generation_handle, 'time', clock)
    return clock


def _idle_waiting_job(service: _FakeService) -> tuple[WebUIDispatcher, WebUIGenerationHandle, list[str]]:
    """A dispatcher with idle-wait on and one handle, built without starting the worker thread."""
    dispatcher = WebUIDispatcher(service, wait_for_idle=True)
    run_log: list[str] = []

    def run() -> dict[str, Any]:
        run_log.append('ran')
        return {'images': [], 'info': None}

    return dispatcher, WebUIGenerationHandle(dispatcher, run, 'task(a)'), run_log


def test_idle_wait_backs_off_until_server_is_idle(clock):
    """Dispatch waits for job_count 0, doubling the poll interval up to _IDLE_POLL_MAX_S."""
    service = _FakeService([_progress(job_count=1)] * 6 + [_progress(job_count=0)])
    dispatcher, handle, run_log = _idle_waiting_job(service)
    dispatcher._run_job(handle)

    assert clock.sleeps == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.0, 1.0])
    assert run_log == ['ran']
    assert handle.poll().status is GenerationStatus.FINISHED


def test_idle_wait_gives_up_after_repeated_progress_errors(clock):
    """_IDLE_MAX_ERRORS consecutive progress_check failures end the wait, and the job is dispatched."""
    service = _FakeService([RuntimeError('connection refused')] * webui_generation_handle._IDLE_MAX_ERRORS)
    dispatcher, handle, run_log = _idle_waiting_job(service)
    dispatcher._run_job(handle)

    assert service.progress_calls == webui_generation_handle._IDLE_MAX_ERRORS
    assert len(clock.sleeps) == webui_generation_handle._IDLE_MAX_ERRORS - 1
    assert run_log == ['ran']


def test_idle_wait_dispatches_anyway_at_the_cap(clock, monkeypatch):
    """A server that stays busy past _IDLE_WAIT_CAP_S still gets the job."""
    monkeypatch.setattr(webui_generation_handle, '_IDLE_WAIT_CAP_S', 3.0)
    service = _FakeService([_progress(job_count=2)])
    dispatcher, handle, run_log = _idle_waiting_job(service)
    dispatcher._run_job(handle)

    assert clock.now == pytest.approx(3.0)
    assert run_log == ['ran']


def test_cancel_during_idle_wait_never_dispatches(clock):
    """A cancel that lands while waiting for the server ends the job CANCELLED without sending it."""
    service = _FakeService([_progress(job_count=1)])
    dispatcher, handle, run_log = _idle_waiting_job(service)
    original_sleep = clock.sleep

    def cancel_on_first_sleep(seconds: float) -> None:
        original_sleep(seconds)
        handle._cancel_requested = True

    clock.sleep = cancel_on_first_sleep  # type: ignore[method-assign]
    dispatcher._run_job(handle)

    assert not run_log
    assert handle.poll().status is GenerationStatus.CANCELLED
    assert service.interrupts == 0
