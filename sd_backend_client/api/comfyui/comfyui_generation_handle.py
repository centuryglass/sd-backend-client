"""ComfyUI implementation of the backend-agnostic generation handle.

ComfyUI is natively asynchronous: submitting a workflow to ``/prompt`` returns immediately with a
``prompt_id`` (and a queue ``number``), and you poll ``/history`` + ``/queue`` — wrapped here by
``ComfyUiWebservice.check_queue_entry`` — until the job finishes, then download the resulting image
references. That maps almost one-to-one onto :class:`GenerationHandle`:

- ``poll()``          -> ``check_queue_entry(prompt_id, number)``, translating ``AsyncTaskStatus``.
- ``_build_result()`` -> ``download_images`` on the finished job's output references.
- ``cancel()``        -> drop a still-queued job from the queue, or interrupt the running one.

While the job is unfinished the handle watches it on the service's ``progress_listener``, which fills
``progress``, ``eta_seconds`` and ``preview`` from ComfyUI's websocket.

Because ComfyUI is already async there is no client-side dispatch queue here (unlike the WebUI handle);
the server *is* the queue. A handle just tracks one ``prompt_id`` and reports what the server says about
it, with two small adaptations layered on top of the raw server status:

* **Cancellation is inferred, not reported.** ComfyUI has no "cancelled" task state — a removed queue
  entry simply disappears (``NOT_FOUND``) and an interrupted run leaves an incomplete/absent history
  entry (``NOT_FOUND`` / ``FAILED``). So once ``cancel()`` has been requested, those terminal states are
  reported as ``CANCELLED`` (mirroring the WebUI handle's "cancel wins" behavior).
* **Post-submit registration lag is smoothed.** Until the job has been seen at least once, a ``NOT_FOUND``
  within ``registration_grace`` seconds of creating the handle is reported as ``PENDING``, so a
  ``poll()`` right after submission doesn't fail if the server is slow to list the job. After the grace
  period ``NOT_FOUND`` is terminal, so ``wait()`` ends on a job the server has lost (restarted, history
  cleared, wrong id).

No Qt / numpy / cv2 here: results are plain ``PIL.Image`` (per the repo's hard constraints).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

from sd_backend_client.api.comfyui_webservice import AsyncTaskStatus
from sd_backend_client.api.shared_data.generation_handle import (GenerationHandle, GenerationProgress,
                                                              GenerationResult, GenerationStatus)
from sd_backend_client.errors import WorkflowValidationError

if TYPE_CHECKING:
    from sd_backend_client.api.comfyui.comfyui_types import PromptExecOutputs, QueueAdditionResponse
    from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice

logger = logging.getLogger(__name__)

DEFAULT_REGISTRATION_GRACE = 5.0

# ComfyUI's queue-status enum maps onto the unified lifecycle almost verbatim; it has no CANCELLED (see
# the module docstring — cancellation is inferred from NOT_FOUND / FAILED after a cancel is requested).
_STATUS_MAP: dict[AsyncTaskStatus, GenerationStatus] = {
    AsyncTaskStatus.PENDING: GenerationStatus.PENDING,
    AsyncTaskStatus.ACTIVE: GenerationStatus.ACTIVE,
    AsyncTaskStatus.FINISHED: GenerationStatus.FINISHED,
    AsyncTaskStatus.FAILED: GenerationStatus.FAILED,
    AsyncTaskStatus.NOT_FOUND: GenerationStatus.NOT_FOUND,
}


class ComfyGenerationHandle(GenerationHandle):
    """Handle to one ComfyUI ``/prompt`` job, identified by its ``prompt_id`` (+ queue ``number``)."""

    def __init__(self, service: 'ComfyUiWebservice', prompt_id: str, number: Optional[int] = None,
                 seed: Optional[int] = None, registration_grace: float = DEFAULT_REGISTRATION_GRACE,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """
        Parameters
        ----------
        service: ComfyUiWebservice
            The client that queued the job.
        prompt_id: str
            The job's id from the queue response.
        number: int, optional
            The job's queue number from the queue response. Kept for reference; jobs are found by prompt_id.
        seed: int, optional
            The seed the queued workflow used, copied into the result.
        registration_grace: float, default=DEFAULT_REGISTRATION_GRACE
            Seconds after creation during which a job the server has never listed reads as PENDING, not NOT_FOUND.
        clock: Callable[[], float], default=time.monotonic
            Time source for registration_grace.
        """
        super().__init__(prompt_id)
        self._service = service
        self._number = number
        self._seed = seed
        self._clock = clock
        self._registration_deadline = clock() + registration_grace
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._ever_registered = False           # Has the server ever reported this job as real?
        self._outputs: Optional['PromptExecOutputs'] = None  # Cached from the FINISHED poll.

    @classmethod
    def from_queue_response(cls, service: 'ComfyUiWebservice',
                            response: 'QueueAdditionResponse') -> 'ComfyGenerationHandle':
        """Build a handle from any of the service's queue responses.

        Raises ``WorkflowValidationError`` if the response has no ``prompt_id``, which means the server rejected
        the workflow.
        """
        if response.prompt_id is None:
            raise WorkflowValidationError(response.error, response.node_errors)
        return cls(service, response.prompt_id, response.number, seed=response.seed)

    # --- GenerationHandle surface -----------------------------------------------------------------

    def poll(self) -> GenerationProgress:
        assert self._task_id is not None  # guaranteed by __init__
        progress = self._service.check_queue_entry(self._task_id, self._number)
        status = _STATUS_MAP[progress.status]

        with self._lock:
            if progress.outputs is not None:
                self._outputs = progress.outputs
            if status in (GenerationStatus.PENDING, GenerationStatus.ACTIVE, GenerationStatus.FINISHED):
                self._ever_registered = True
            cancel_requested = self._cancel_requested
            ever_registered = self._ever_registered

        # A removed/interrupted job disappears or errors out; honor the caller's cancel intent.
        if cancel_requested and status in (GenerationStatus.NOT_FOUND, GenerationStatus.FAILED):
            return self._finish(GenerationProgress(GenerationStatus.CANCELLED))
        if status is GenerationStatus.NOT_FOUND and not ever_registered:
            if self._clock() < self._registration_deadline:
                return self._unfinished(GenerationProgress(GenerationStatus.PENDING,
                                                           text_info='Queued (awaiting registration)'))
            return self._finish(GenerationProgress(
                status, text_info=f'ComfyUI has no record of job {self._task_id!r}. The server may have restarted '
                                  'or cleared its history, or the id is wrong.'))
        if status is GenerationStatus.NOT_FOUND:
            return self._finish(GenerationProgress(
                status, text_info=f'ComfyUI lost track of job {self._task_id!r}. The server may have restarted '
                                  'or cleared its history.'))
        if status is GenerationStatus.FAILED:
            return self._finish(GenerationProgress(status, text_info=progress.error))
        if status is GenerationStatus.FINISHED:
            return self._finish(GenerationProgress(status, progress=1.0))
        if status is GenerationStatus.PENDING:
            return self._unfinished(GenerationProgress(status, queue_index=progress.index))
        return self._unfinished(GenerationProgress(status), live=True)

    def _unfinished(self, snapshot: GenerationProgress, live: bool = False) -> GenerationProgress:
        """Watch the job for live progress, adding it to an ACTIVE snapshot when `live` is set."""
        listener = self._service.progress_listener
        assert self._task_id is not None
        if listener is None:
            return snapshot
        listener.watch(self._task_id)
        live_progress = listener.get_progress(self._task_id) if live else None
        if live_progress is not None:
            snapshot.progress = live_progress.progress
            snapshot.eta_seconds = live_progress.eta_seconds
            snapshot.preview = live_progress.preview
        return snapshot

    def _finish(self, snapshot: GenerationProgress) -> GenerationProgress:
        """Stop watching the job for live progress, since it has ended."""
        listener = self._service.progress_listener
        if listener is not None and self._task_id is not None:
            listener.unwatch(self._task_id)
        return snapshot

    def _build_result(self) -> GenerationResult:
        with self._lock:
            outputs = self._outputs
        if outputs is None:
            # Only reached if _build_result is called without a preceding FINISHED poll; fetch fresh.
            assert self._task_id is not None
            outputs = self._service.check_queue_entry(self._task_id, self._number).outputs
        image_refs = outputs.images if (outputs is not None and outputs.images is not None) else []
        images = self._service.download_images(image_refs)
        # A ComfyUI batch samples every image from one seed, so each image reports the job's seed.
        seeds = [] if self._seed is None else [self._seed] * len(images)
        return GenerationResult(images=images, seeds=seeds, seed=self._seed, task_id=self._task_id, raw_info=outputs)

    def cancel(self) -> bool:
        """Cancel the job: drop it from the queue if still PENDING, else interrupt the running job.

        Returns ``True`` if a cancellation was issued, ``False`` if the job had already finished. Cancelling a
        pending job leaves the running job untouched (see ``ComfyUiWebservice.remove_from_queue``). Cancelling an
        active job names it in the interrupt, so a job that finished in the meantime doesn't take another job down
        with it, except on ComfyUI versions without targeted interrupts (see ``ComfyUiWebservice.interrupt``).
        """
        assert self._task_id is not None
        status = _STATUS_MAP[self._service.check_queue_entry(self._task_id, self._number).status]
        if status is GenerationStatus.FINISHED:
            return False
        with self._lock:
            self._cancel_requested = True

        try:
            if status is GenerationStatus.PENDING:
                self._service.remove_from_queue(self._task_id)   # clean: doesn't touch the running job
            else:
                self._service.interrupt(self._task_id)           # ACTIVE (or settling): stop the run
        except Exception:  # pragma: no cover - cancellation is best-effort
            logger.exception('Failed to cancel ComfyUI job %s', self._task_id)
        return True
