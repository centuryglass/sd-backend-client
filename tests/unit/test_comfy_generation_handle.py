"""Offline unit tests for :class:`ComfyGenerationHandle`.

These pin the handle's server-status translation and cancellation logic without a live ComfyUI server,
using a scripted fake service. They cover the pieces that are easy to get subtly wrong:

- ``AsyncTaskStatus`` -> ``GenerationStatus`` mapping and PENDING queue-index passthrough;
- the full PENDING -> ACTIVE -> FINISHED ``wait()`` path, including result download;
- "cancel wins": once cancelled, a NOT_FOUND / FAILED server state reports CANCELLED;
- clean PENDING cancel (queue removal) vs. ACTIVE cancel (interrupt), and no-op cancel of a finished job;
- registration-lag smoothing: a NOT_FOUND seen before the job is ever registered reads as PENDING, until the
  registration grace period ends;
- failure detail and live progress passed through from the service.
"""
from typing import Optional

import pytest
from PIL import Image

from sd_backend_client.util.visual.inpaint_crop import PasteBack
from sd_backend_client.api.comfyui.comfyui_generation_handle import ComfyGenerationHandle
from sd_backend_client.api.comfyui.comfyui_progress_listener import LiveProgress, SamplerRunPlan
from sd_backend_client.api.comfyui.comfyui_types import (ImageFileReference, PromptExecOutputs,
                                                      QueueAdditionResponse)
from sd_backend_client.api.comfyui_webservice import AsyncTaskProgress, AsyncTaskStatus
from sd_backend_client.api.shared_data.generation_handle import (GenerationError, GenerationStatus)
from sd_backend_client.errors import BackendTimeoutError, WorkflowValidationError


class FakeComfyService:
    """A stand-in for ComfyUiWebservice that replays a scripted sequence of task-progress snapshots."""

    def __init__(self, script: list[AsyncTaskProgress], progress_listener=None):
        self._script = list(script)
        self.progress_listener = progress_listener
        self._calls = 0
        self.removed: list[str] = []       # task_ids passed to remove_from_queue
        self.interrupted: list = []         # task_ids (or None) passed to interrupt

    def check_queue_entry(self, prompt_id: str, number) -> AsyncTaskProgress:
        # Advance through the script, clamping on the last snapshot for any further polls.
        snapshot = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return snapshot

    def remove_from_queue(self, task_id: str) -> None:
        self.removed.append(task_id)

    def interrupt(self, task_id=None) -> None:
        self.interrupted.append(task_id)

    def download_images(self, image_refs):
        return [Image.new('RGBA', (4, 4), (i, i, i, 255)) for i, _ in enumerate(image_refs)]


class FakeListener:
    """A stand-in for ComfyProgressListener that records watched jobs and reports fixed progress."""

    def __init__(self, live: LiveProgress):
        self.live = live
        self.watched: set[str] = set()
        self.run_plans: dict[str, Optional[SamplerRunPlan]] = {}

    def watch(self, prompt_id: str, run_plan: Optional[SamplerRunPlan] = None) -> None:
        """Record the job as watched, with its run plan."""
        self.watched.add(prompt_id)
        self.run_plans[prompt_id] = run_plan

    def unwatch(self, prompt_id: str) -> None:
        """Stop recording the job as watched."""
        self.watched.discard(prompt_id)

    def get_progress(self, prompt_id: str):
        """Return the fixed progress for a watched job."""
        return self.live if prompt_id in self.watched else None


class FakeClock:
    """A settable time source."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _outputs(n: int = 1) -> PromptExecOutputs:
    return PromptExecOutputs(images=[ImageFileReference(filename=f'out_{i}.png', subfolder='')
                                     for i in range(n)])


def _handle(script, seed=None):
    return ComfyGenerationHandle(FakeComfyService(script), 'prompt-123', number=7, seed=seed)


# --------------------------------------------------------------------------- #
# from_queue_response
# --------------------------------------------------------------------------- #

def test_from_queue_response_populates_ids_and_seed():
    response = QueueAdditionResponse(prompt_id='abc', number=4, seed=99)
    handle = ComfyGenerationHandle.from_queue_response(FakeComfyService([]), response)
    assert handle.task_id == 'abc'
    assert handle._number == 4
    assert handle._seed == 99


def test_from_queue_response_rejects_missing_prompt_id():
    response = QueueAdditionResponse(prompt_id=None, error='node validation failed')
    with pytest.raises(WorkflowValidationError, match='rejected the workflow: node validation failed'):
        ComfyGenerationHandle.from_queue_response(FakeComfyService([]), response)


def test_from_queue_response_accepts_missing_number():
    """A response without a queue number still gives a handle, since jobs are found by prompt id."""
    response = QueueAdditionResponse(prompt_id='abc', number=None)
    handle = ComfyGenerationHandle.from_queue_response(FakeComfyService([]), response)
    assert handle.task_id == 'abc'


# --------------------------------------------------------------------------- #
# Status mapping
# --------------------------------------------------------------------------- #

def test_poll_maps_pending_with_queue_index():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=3)])
    progress = handle.poll()
    assert progress.status is GenerationStatus.PENDING
    assert progress.queue_index == 3


def test_poll_maps_active_and_clears_queue_index():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE, index=3)])
    progress = handle.poll()
    assert progress.status is GenerationStatus.ACTIVE
    assert progress.queue_index is None  # index only meaningful while PENDING


def test_poll_maps_failed():
    # Register the job first (ACTIVE) so the later FAILED isn't smoothed as a registration-lag case.
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
                      AsyncTaskProgress(status=AsyncTaskStatus.FAILED)])
    handle.poll()
    assert handle.poll().status is GenerationStatus.FAILED


def test_failed_job_wait_raises_with_server_reason():
    """wait() raises GenerationError carrying the error ComfyUI reported."""
    reason = 'ComfyUI execution failed in KSampler (node 3): RuntimeError: out of memory'
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.FAILED, error=reason)])
    with pytest.raises(GenerationError, match='out of memory') as error:
        handle.wait(poll_interval=0.0)
    assert error.value.status is GenerationStatus.FAILED


# --------------------------------------------------------------------------- #
# Lifecycle / wait
# --------------------------------------------------------------------------- #

def test_wait_runs_through_lifecycle_and_downloads(tmp_path):
    handle = _handle([
        AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=1),
        AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
        AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(2)),
    ], seed=42)
    seen = []
    result = handle.wait(poll_interval=0.0, on_progress=lambda p: seen.append(p.status))
    assert [s for s in (GenerationStatus.PENDING, GenerationStatus.ACTIVE, GenerationStatus.FINISHED)
            if s in seen] == [GenerationStatus.PENDING, GenerationStatus.ACTIVE, GenerationStatus.FINISHED]
    assert len(result.images) == 2
    assert result.seed == 42
    assert result.seeds == [42, 42]  # a ComfyUI batch shares one seed
    assert not result.control_maps
    assert result.task_id == 'prompt-123'
    assert isinstance(result.raw_info, PromptExecOutputs)


def test_build_result_uses_cached_outputs_from_finished_poll():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(1))])
    handle.poll()  # caches outputs
    result = handle._build_result()
    assert len(result.images) == 1
    assert not result.seeds and result.seed is None  # no seed, as for a basic upscale or a preview


# --------------------------------------------------------------------------- #
# Registration-lag smoothing
# --------------------------------------------------------------------------- #

def test_not_found_before_registration_reads_as_pending():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    progress = handle.poll()
    assert progress.status is GenerationStatus.PENDING  # smoothed: job not yet visible in the queue


def test_not_found_after_registration_is_terminal():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
                      AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    assert handle.poll().status is GenerationStatus.ACTIVE  # registers the job
    progress = handle.poll()
    assert progress.status is GenerationStatus.NOT_FOUND
    assert 'lost track' in progress.text_info


def test_not_found_is_terminal_after_registration_grace():
    """A job the server never lists stops reading as PENDING once the grace period ends."""
    clock = FakeClock()
    handle = ComfyGenerationHandle(FakeComfyService([AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)]),
                                   'prompt-123', registration_grace=5.0, clock=clock)
    clock.now += 4.9
    assert handle.poll().status is GenerationStatus.PENDING
    clock.now += 0.1
    progress = handle.poll()
    assert progress.status is GenerationStatus.NOT_FOUND
    assert 'no record' in progress.text_info


def test_wait_on_unknown_job_raises_instead_of_waiting_forever():
    """wait() with no timeout ends with GenerationError(NOT_FOUND) once the grace period passes."""
    clock = FakeClock()
    service = FakeComfyService([AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    handle = ComfyGenerationHandle(service, 'prompt-123', registration_grace=5.0, clock=clock)

    def advance(_progress) -> None:
        clock.now += 1.0

    with pytest.raises(GenerationError, match='no record') as error:
        handle.wait(poll_interval=0.0, on_progress=advance)
    assert error.value.status is GenerationStatus.NOT_FOUND


# --------------------------------------------------------------------------- #
# Live progress
# --------------------------------------------------------------------------- #

def test_active_poll_reports_live_progress_and_finish_unwatches():
    """An ACTIVE poll adds the listener's progress, and a terminal poll stops watching the job."""
    preview = Image.new('RGBA', (2, 2))
    listener = FakeListener(LiveProgress(progress=0.25, eta_seconds=3.0, preview=preview))
    service = FakeComfyService([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=0),
                                AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),
                                AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(1))],
                               progress_listener=listener)
    handle = ComfyGenerationHandle(service, 'prompt-123')

    pending = handle.poll()
    assert listener.watched == {'prompt-123'}
    assert pending.progress is None

    active = handle.poll()
    assert (active.progress, active.eta_seconds, active.preview) == (0.25, 3.0, preview)

    finished = handle.poll()
    assert finished.progress == 1.0
    assert not listener.watched


def test_run_plan_reaches_the_listener_and_overall_progress_the_snapshot():
    """A handle built with a run plan watches its job with that plan and reports the listener's overall progress."""
    plan = SamplerRunPlan(node='9', steps=20, total_runs=6)
    listener = FakeListener(LiveProgress(progress=0.5, overall_progress=0.25))
    service = FakeComfyService([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE)], progress_listener=listener)
    handle = ComfyGenerationHandle.from_queue_response(
        service, QueueAdditionResponse(prompt_id='prompt-123', number=1), run_plan=plan)

    active = handle.poll()
    assert listener.run_plans == {'prompt-123': plan}
    assert (active.progress, active.overall_progress) == (0.5, 0.25)


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #

def test_cancel_pending_removes_from_queue_without_interrupt():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=2),  # cancel() reads this
                      AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])         # subsequent poll
    assert handle.cancel() is True
    assert handle._service.removed == ['prompt-123']
    assert handle._service.interrupted == []           # a queued job must not interrupt the running one
    assert handle.poll().status is GenerationStatus.CANCELLED  # cancel wins over NOT_FOUND


def test_cancel_active_interrupts():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE),   # cancel() reads this
                      AsyncTaskProgress(status=AsyncTaskStatus.FAILED)])  # interrupted run errors out
    assert handle.cancel() is True
    assert handle._service.interrupted == ['prompt-123']  # the interrupt names the job
    assert handle._service.removed == []
    assert handle.poll().status is GenerationStatus.CANCELLED  # cancel wins over FAILED


def test_cancel_finished_job_is_noop():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs(1))])
    assert handle.cancel() is False
    assert handle._service.removed == []
    assert handle._service.interrupted == []


def test_cancelled_job_wait_raises():
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.PENDING, index=0),
                      AsyncTaskProgress(status=AsyncTaskStatus.NOT_FOUND)])
    handle.cancel()
    with pytest.raises(GenerationError):
        handle.wait(poll_interval=0.0)


def test_wait_timeout_raises_backend_timeout_error():
    """The polling wait() raises BackendTimeoutError, which is also a TimeoutError, when the timeout elapses."""
    handle = _handle([AsyncTaskProgress(status=AsyncTaskStatus.ACTIVE)] * 3)
    with pytest.raises(BackendTimeoutError) as error:
        handle.wait(timeout=0, poll_interval=0.0)
    assert isinstance(error.value, TimeoutError)


# --------------------------------------------------------------------------- #
# Inpainting paste-back
# --------------------------------------------------------------------------- #

def test_inpaint_result_is_composited_onto_the_source():
    """Outside the mask the result is the source, inside it is the generated image, and grey blends."""
    source = Image.new('RGBA', (4, 4), (200, 0, 0, 255))
    mask = Image.new('L', (4, 4), 0)
    mask.paste(255, (0, 0, 2, 4))      # left half white
    mask.putpixel((2, 0), 128)
    service = FakeComfyService([AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs())])
    handle = ComfyGenerationHandle(service, 'p', paste_back=PasteBack(source, mask))
    result = handle.wait()
    image = result.images[0]
    assert image.getpixel((0, 0)) == (0, 0, 0, 255)         # generated image (fake output is all zeros)
    assert image.getpixel((3, 3)) == (200, 0, 0, 255)       # source
    assert 0 < image.getpixel((2, 0))[0] < 200              # blended


def test_result_is_untouched_without_paste_back():
    """Jobs without a mask return the downloaded images as they are."""
    service = FakeComfyService([AsyncTaskProgress(status=AsyncTaskStatus.FINISHED, outputs=_outputs())])
    assert ComfyGenerationHandle(service, 'p').wait().images[0].getpixel((0, 0)) == (0, 0, 0, 255)
