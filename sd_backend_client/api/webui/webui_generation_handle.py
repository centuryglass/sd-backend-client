"""WebUI (A1111 / Forge) implementation of the backend-agnostic generation handle.

The WebUI's ``txt2img`` / ``img2img`` HTTP calls block until the image is ready, and the server has no
endpoint to drop a *specific* queued job — only to interrupt the one currently running. Naively firing
the blocking POST on a background thread therefore gives you a job you cannot cleanly cancel: killing the
client thread (by closing its socket) leaves the server diffusing the image anyway, because Starlette
runs the sync endpoint to completion regardless of client disconnect.

So instead of pushing work to the server immediately, this module keeps a **client-side single-slot
dispatch queue** (:class:`WebUIDispatcher`). A submitted job sits in that queue, unsent, until the
dispatcher's worker thread pulls it and fires the blocking POST — and only then, ideally once the server
is idle. The upshot:

- While a job is client-queued (``PENDING``) it hasn't touched the network, so cancelling it is instant
  and clean: drop it from the queue, done. No orphaned GPU work.
- Once dispatched (``ACTIVE``) ``cancel()`` maps to the server's ``/sdapi/v1/interrupt``.

This lifts the WebUI up to the same inspectable/cancellable-queue capability ComfyUI has natively,
instead of dragging the unified interface down to the WebUI's lowest common denominator. The only thing
given up is the server's own FIFO fairness when *multiple* clients share one instance — an accepted
trade-off for the single-client scripting use case this wrapper targets.

Every ``force_task_id`` is threaded into the request body so the dispatched job can later be correlated
with the server's ``/internal/progress`` view if finer-grained queue tracking is added.
"""
from __future__ import annotations

import logging
import random
import string
import threading
import time
from collections import deque
from typing import Callable, Optional, Protocol, TYPE_CHECKING

from PIL import Image  # type: ignore

from sd_backend_client.api.shared_data.generation_handle import (GenerationError, GenerationHandle,
                                                              GenerationProgress, GenerationResult,
                                                              GenerationStatus, ProgressCallback)
from sd_backend_client.api.webui.response_formats import GenerationInfoData
from sd_backend_client.errors import BackendTimeoutError
from sd_backend_client.util.visual.image_utils import image_from_base64
from sd_backend_client.util.visual.inpaint_crop import PasteBack

if TYPE_CHECKING:
    from sd_backend_client.api.a1111_webservice import ImageResponse

logger = logging.getLogger(__name__)

# A blocking generation call (A1111Webservice._post_generation bound to a prepared body) producing images.
GenerationCall = Callable[[], 'ImageResponse']

# Tuning for WebUIDispatcher._await_server_idle: poll /sdapi/v1/progress with exponential backoff between
# these bounds, waiting at most _IDLE_WAIT_CAP_S for the server to drain before dispatching anyway. A short
# run of consecutive progress-endpoint errors is tolerated (treated as "still busy") before giving up.
_IDLE_POLL_MIN_S = 0.1
_IDLE_POLL_MAX_S = 1.0
_IDLE_WAIT_CAP_S = 600.0
_IDLE_MAX_ERRORS = 3


class _ProgressInterruptService(Protocol):
    """The slice of ``A1111Webservice`` this module depends on (kept narrow to avoid a circular import)."""

    def progress_check(self): ...  # -> ProgressResponseBody

    def interrupt(self) -> dict: ...


def create_task_id(task_type: str) -> str:
    """Mint a WebUI-style task id, matching the server's own ``task(<type>-XXXXXXX)`` format.

    Using the same shape means the id is a valid ``force_task_id`` the server will accept verbatim.
    """
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    return f'task({task_type}-{suffix})'


def build_webui_result(images: list[Image.Image], info: Optional[object],
                       task_id: Optional[str]) -> GenerationResult:
    """Sort a WebUI response's images into a `GenerationResult`, using `info` to find the generated ones.

    The WebUI returns one flat image list: an optional batch grid at ``info.index_of_first_image``'s offset, then one
    image per entry of ``info.all_seeds``, then any images extensions appended (ControlNet detect maps, inpainting
    masks). Without a `GenerationInfoData` (basic upscaling, preprocessor previews), or when the counts don't fit
    that layout, every image counts as generated and `seeds` stays empty.
    """
    if not isinstance(info, GenerationInfoData):
        return GenerationResult(images=images, task_id=task_id, raw_info=info)
    first = info.index_of_first_image
    seed_count = len(info.all_seeds)
    if seed_count == 0 or first < 0 or first + seed_count > len(images):
        logger.warning('WebUI job %s returned %d images, which does not match index_of_first_image=%d and %d seeds; '
                       'treating every image as generated', task_id, len(images), first, seed_count)
        return GenerationResult(images=images, seed=info.seed, task_id=task_id, raw_info=info)
    extra = {'grid': images[0]} if first > 0 else {}
    return GenerationResult(images=images[first:first + seed_count], seeds=list(info.all_seeds),
                            control_maps=images[first + seed_count:], seed=info.all_seeds[0], task_id=task_id,
                            raw_info=info, extra=extra)


class WebUIGenerationHandle(GenerationHandle):
    """Handle to one WebUI generation job managed by a :class:`WebUIDispatcher`.

    Lifecycle: ``PENDING`` (client-queued, unsent) → ``ACTIVE`` (dispatched, blocking POST in flight) →
    ``FINISHED`` / ``FAILED`` / ``CANCELLED``. All state transitions are guarded by ``self._lock``; the
    dispatcher worker and the caller's cancel/poll threads all go through it.
    """

    def __init__(self, dispatcher: 'WebUIDispatcher', run: GenerationCall, task_id: str,
                 interruptible: bool = True, paste_back: Optional[PasteBack] = None) -> None:
        """Create a handle for `run`. When `paste_back` is set, each generated image is pasted into it."""
        super().__init__(task_id)
        self._paste_back = paste_back
        self._dispatcher = dispatcher
        self._run = run
        self._interruptible = interruptible
        self._lock = threading.Lock()
        self._status = GenerationStatus.PENDING
        self._cancel_requested = False
        self._result: Optional['ImageResponse'] = None
        self._error: Optional[BaseException] = None
        self._done_event = threading.Event()

    # --- GenerationHandle surface -----------------------------------------------------------------

    def poll(self) -> GenerationProgress:
        with self._lock:
            status = self._status
            error = self._error
        if status is GenerationStatus.PENDING:
            index = self._dispatcher.queue_index_of(self)
            return GenerationProgress(status, queue_index=index, text_info='Queued (client-side)')
        if status is GenerationStatus.ACTIVE:
            return self._active_progress()
        if status is GenerationStatus.FAILED:
            return GenerationProgress(status, text_info=str(error) if error is not None else None)
        return GenerationProgress(status)

    def _build_result(self) -> GenerationResult:
        with self._lock:
            result = self._result
        assert result is not None, '_build_result called before job reached FINISHED'
        built = build_webui_result(list(result['images']), result.get('info'), self._task_id)
        if self._paste_back is not None:
            built.images = [self._paste_back.apply(image) for image in built.images]
        return built

    def cancel(self) -> bool:
        """Cancel the job. Clean while ``PENDING`` (never dispatched); interrupt-based while ``ACTIVE``.

        Returns ``True`` if the cancellation was accepted. Returns ``False`` if the job had already reached a
        terminal state, or if it is ``ACTIVE`` on an endpoint the server's interrupt may not stop (see
        ``WebUIDispatcher.submit``), in which case the job runs on and finishes normally.
        """
        with self._lock:
            if self._status.is_terminal:
                return False
            status = self._status
            if status is GenerationStatus.ACTIVE and not self._interruptible:
                return False
            self._cancel_requested = True

        if status is GenerationStatus.PENDING:
            # Try to yank it from the client-side queue before the worker ever POSTs it.
            if self._dispatcher.remove_pending(self):
                self._finalize(GenerationStatus.CANCELLED)
                return True
            # Lost the race: the worker just claimed it. The _cancel_requested flag is already set, so
            # the worker will honor it either before dispatch or right after the blocking POST returns.
            return True

        # ACTIVE: the request is server-side and running; interrupt the running job.
        try:
            self._dispatcher.service.interrupt()
        except Exception:  # pragma: no cover - interrupt is best-effort
            logger.exception('Failed to interrupt active WebUI job %s', self._task_id)
        return True

    # --- Efficient blocking wait (overrides the poll-loop with the worker's completion event) ------

    def wait(self, timeout: Optional[float] = None, poll_interval: float = 0.5,
             on_progress: Optional[ProgressCallback] = None) -> GenerationResult:
        """Block until the job finishes, sleeping on the worker's completion event rather than busy-polling.

        Semantically identical to the base ``GenerationHandle.wait``: ``on_progress`` still fires with a
        fresh snapshot roughly every ``poll_interval`` seconds (for progress bars / live previews),
        ``BackendTimeoutError`` is raised if ``timeout`` elapses, and a non-FINISHED terminal state raises
        ``GenerationError``. The difference is that when nothing is happening we wait on the event, so an
        idle wait costs no polling.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._done_event.is_set():
            if on_progress is not None:
                on_progress(self.poll())
            if deadline is None:
                slice_s = poll_interval
            else:
                slice_s = min(poll_interval, max(0.0, deadline - time.monotonic()))
                if slice_s <= 0.0:
                    raise BackendTimeoutError(f'Generation {self._task_id!r} did not finish within {timeout}s')
            self._done_event.wait(slice_s)

        final = self.poll()
        if on_progress is not None:
            on_progress(final)
        if final.status is GenerationStatus.FINISHED:
            return self._build_result()
        with self._lock:
            error = self._error
        raise GenerationError(final.status, final.text_info) from error

    # --- Internal transitions (called by the dispatcher worker) -----------------------------------

    def _begin_dispatch(self) -> bool:
        """Move PENDING → ACTIVE unless cancellation was requested first. Returns False if cancelled."""
        with self._lock:
            if self._cancel_requested:
                cancelled = True
            else:
                self._status = GenerationStatus.ACTIVE
                cancelled = False
        if cancelled:
            self._finalize(GenerationStatus.CANCELLED)
            return False
        return True

    def _resolve(self, result: 'ImageResponse') -> None:
        """Store a successful result, unless a cancel landed while the POST was in flight."""
        with self._lock:
            if self._cancel_requested:
                terminal = GenerationStatus.CANCELLED
            else:
                self._result = result
                terminal = GenerationStatus.FINISHED
        self._finalize(terminal)

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            self._error = error
        self._finalize(GenerationStatus.FAILED)

    def _finalize(self, status: GenerationStatus) -> None:
        with self._lock:
            self._status = status
        self._done_event.set()

    def _active_progress(self) -> GenerationProgress:
        """Query the server's global progress endpoint for the currently-running (our) job."""
        try:
            body = self._dispatcher.service.progress_check()
        except Exception:  # pragma: no cover - progress is best-effort
            return GenerationProgress(GenerationStatus.ACTIVE)
        preview: Optional[Image.Image] = None
        if body.current_image:
            try:
                preview = image_from_base64(body.current_image)
            except Exception:
                preview = None
        return GenerationProgress(GenerationStatus.ACTIVE, progress=body.progress,
                                 eta_seconds=body.eta_relative, preview=preview, text_info=body.textinfo)


class WebUIDispatcher:
    """Owns a single-slot, client-side dispatch queue for one WebUI server.

    A background daemon worker pulls jobs FIFO and runs each blocking generation call to completion before
    starting the next, so only one request is ever in flight on the server at a time. Jobs waiting in the
    queue have not been sent anywhere and can be cancelled with no server involvement.
    """

    def __init__(self, service: _ProgressInterruptService, wait_for_idle: bool = True) -> None:
        self.service = service
        self._wait_for_idle = wait_for_idle
        self._queue: deque[WebUIGenerationHandle] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._worker: Optional[threading.Thread] = None

    def submit(self, run: GenerationCall, task_id: str, interruptible: bool = True,
               paste_back: Optional[PasteBack] = None) -> WebUIGenerationHandle:
        """Enqueue a blocking generation call and return its handle immediately (job starts PENDING).

        Pass ``interruptible=False`` for a call that ``/sdapi/v1/interrupt`` is not known to stop, such as basic
        upscaling or a ControlNet preprocessor preview. Its handle then refuses to cancel once dispatched.
        `paste_back` is passed to the handle.
        """
        handle = WebUIGenerationHandle(self, run, task_id, interruptible, paste_back)
        with self._lock:
            self._queue.append(handle)
            self._ensure_worker_locked()
            self._not_empty.notify()
        return handle

    def queue_index_of(self, handle: WebUIGenerationHandle) -> Optional[int]:
        """0-based position of a still-queued handle, or ``None`` if it is no longer pending."""
        with self._lock:
            try:
                return self._queue.index(handle)
            except ValueError:
                return None

    def remove_pending(self, handle: WebUIGenerationHandle) -> bool:
        """Remove a handle from the pending queue if still present. Returns whether it was removed."""
        with self._lock:
            try:
                self._queue.remove(handle)
                return True
            except ValueError:
                return False

    # --- Worker ----------------------------------------------------------------------------------

    def _ensure_worker_locked(self) -> None:
        """Start the daemon worker if it isn't already running. Caller must hold ``self._lock``."""
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, name='webui-dispatch', daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue:
                    self._not_empty.wait()
                handle = self._queue.popleft()
            self._run_job(handle)

    def _run_job(self, handle: WebUIGenerationHandle) -> None:
        # Wait for the server to drain *while the job is still PENDING* — i.e. before _begin_dispatch
        # flips it to ACTIVE. This keeps the not-yet-sent job cleanly cancellable during the wait (a
        # cancel drops it with no POST), and means we never fire our request into a server-side queue we
        # couldn't cancel out of. _await_server_idle also returns early if the job is cancelled meanwhile.
        if self._wait_for_idle:
            self._await_server_idle(handle)
        # Honor a cancel that landed before/during the idle wait; otherwise transition PENDING -> ACTIVE.
        if not handle._begin_dispatch():
            return
        try:
            result = handle._run()
        except BaseException as error:  # noqa: BLE001 - surface any failure through the handle
            logger.exception('WebUI generation %s failed', handle.task_id)
            handle._fail(error)
            return
        handle._resolve(result)

    def _await_server_idle(self, handle: WebUIGenerationHandle) -> None:
        """Block until the server reports no active/queued job, so our POST isn't server-queued.

        A specific *queued* WebUI job cannot be cancelled server-side, so if we POSTed while the server
        were busy our job would silently drop below the client-side cancellability guarantee. To avoid
        that, hold dispatch here until ``/sdapi/v1/progress`` reports ``state.job_count == 0``, polling
        with exponential backoff up to ``_IDLE_WAIT_CAP_S``.

        Best-effort by design: a short run of consecutive progress-endpoint errors is treated as "still
        busy" (transient blip) but eventually gives up rather than blocking forever; hitting the cap logs
        and dispatches anyway. Returns immediately if ``handle`` is cancelled while we wait, so the caller
        can finalize it CANCELLED without ever sending the request.
        """
        deadline = time.monotonic() + _IDLE_WAIT_CAP_S
        interval = _IDLE_POLL_MIN_S
        errors = 0
        while time.monotonic() < deadline:
            if handle._cancel_requested:
                return  # caller's _begin_dispatch will finalize CANCELLED; never POSTed.
            try:
                job_count = self.service.progress_check().state.job_count
                errors = 0
            except Exception:  # network blip: treat as busy for a few tries, then give up.
                errors += 1
                if errors >= _IDLE_MAX_ERRORS:
                    logger.debug('Giving up idle-wait for %s after %d progress errors', handle.task_id, errors)
                    return
                job_count = 1
            if job_count <= 0:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))
            interval = min(interval * 2, _IDLE_POLL_MAX_S)
        logger.warning('WebUI server still busy after %.0fs; dispatching %s anyway',
                       _IDLE_WAIT_CAP_S, handle.task_id)
