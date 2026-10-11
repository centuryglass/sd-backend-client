"""Backend-agnostic handle for a single in-flight image-generation job.

This is the seam that hides the sync/async discrepancy between the two backends. ComfyUI is natively
asynchronous (submit returns a queue ticket; you poll/stream for progress and download results later),
while the WebUI's HTTP call blocks until the image is ready. Both are modeled here as the *same* thing:
an async submission that yields a :class:`GenerationHandle`.

The design principle is **async is the primitive, blocking is derived**. A backend implements the three
abstract members below (``poll``, ``_build_result``, ``cancel``); the blocking ``wait()`` convenience is
provided here once, in terms of ``poll()``, so every backend gets a correct blocking path for free.

- ComfyUI: ``poll()`` wraps ``check_queue_entry``; ``_build_result()`` downloads the finished image refs.
- WebUI:   the blocking POST runs on a background thread (see ``webui_generation_handle``); ``poll()``
           reports thread state + server progress, and ``_build_result()`` returns the images the thread
           already produced.

No Qt / numpy / cv2 here: previews and results are plain ``PIL.Image`` (per the repo's hard constraints).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from PIL import Image  # type: ignore

from sd_backend_client.errors import BackendTimeoutError, GenerationError

__all__ = ['GenerationStatus', 'GenerationProgress', 'GenerationResult', 'ProgressCallback', 'GenerationError',
           'GenerationHandle']


class GenerationStatus(Enum):
    """Unified lifecycle status for a generation job across both backends.

    This is the superset of ComfyUI's ``AsyncTaskStatus`` and the states derivable from the WebUI's
    progress/thread signals. ``PENDING`` means accepted by the server but not yet running (queued behind
    other work); ``ACTIVE`` means it is the job currently being diffused.
    """
    PENDING = 'pending'      # Accepted/queued, not yet running.
    ACTIVE = 'active'        # Currently being generated.
    FINISHED = 'finished'    # Completed successfully; results are retrievable.
    FAILED = 'failed'        # Server or generation error.
    CANCELLED = 'cancelled'  # Interrupted/removed by the caller.
    NOT_FOUND = 'not_found'  # Server has no record of this task (expired history, bad id, ...).

    @property
    def is_terminal(self) -> bool:
        """Whether no further status changes are expected for this job."""
        return self in (GenerationStatus.FINISHED, GenerationStatus.FAILED,
                        GenerationStatus.CANCELLED, GenerationStatus.NOT_FOUND)


@dataclass
class GenerationProgress:
    """A single point-in-time snapshot of a job's state, backend-agnostic.

    All numeric fields are best-effort: a backend that cannot supply one leaves it ``None`` rather than
    guessing. ``queue_index`` is only meaningful while ``status is PENDING``.
    """
    status: GenerationStatus
    progress: Optional[float] = None       # Fraction complete in [0, 1], if the backend reports it.
    queue_index: Optional[int] = None      # 0-based position in the pending queue (PENDING only).
    eta_seconds: Optional[float] = None    # Estimated seconds remaining, if known.
    preview: Optional[Image.Image] = None  # Live-preview frame, if the backend/caller requested one.
    text_info: Optional[str] = None        # Human-readable status line straight from the server.

    @property
    def done(self) -> bool:
        """Whether this snapshot represents a terminal state."""
        return self.status.is_terminal


@dataclass
class GenerationResult:
    """The final output of a finished generation job. Every field except `raw_info` means the same on both backends.

    Attributes
    ----------
    images: list[Image.Image]
        The images the job generated, in order. Never includes ControlNet detect maps or a WebUI batch grid.
    seeds: list[int]
        The seed each image was generated with, parallel to `images`, or empty when the job used no seed (basic
        upscaling, preprocessor previews) or the backend did not report one per image. A WebUI batch uses
        consecutive seeds. A ComfyUI batch shares one seed, and each image's batch position also shapes its noise.
    control_maps: list[Image.Image]
        Images the backend returned after the generated ones: ControlNet detect maps, and inpainting masks when the
        WebUI's return-mask settings are on. Only the WebUI returns these, and only when its ControlNet extension's
        "do not append detectmap" setting is off.
    seed: int, optional
        The job's base seed: the first entry of `seeds`, or the seed used when no images came back.
    task_id: str, optional
        The server-side id of the job.
    raw_info: object, optional
        The backend's own metadata, whose type depends on the backend: `GenerationInfoData` on the WebUI,
        `PromptExecOutputs` on ComfyUI. Code that reads it is backend-specific.
    extra: dict
        Backend-specific extras. Holds the WebUI's batch grid image under ``'grid'`` when the server returned one.
    """
    images: list[Image.Image]
    seeds: list[int] = field(default_factory=list)
    control_maps: list[Image.Image] = field(default_factory=list)
    seed: Optional[int] = None
    task_id: Optional[str] = None
    raw_info: Optional[object] = None
    extra: dict = field(default_factory=dict)


ProgressCallback = Callable[[GenerationProgress], None]


class GenerationHandle(ABC):
    """Backend-agnostic handle to one submitted generation job.

    Obtain one from a backend's ``submit_*`` method. Then either poll it yourself for progress/UI, or
    call :meth:`wait` to block until it finishes. A handle refers to exactly one job and is not reusable.
    """

    def __init__(self, task_id: Optional[str]) -> None:
        self._task_id = task_id

    @property
    def task_id(self) -> Optional[str]:
        """Server-side identifier for this job, if the backend exposes one."""
        return self._task_id

    # --- Abstract, backend-specific surface -------------------------------------------------------

    @abstractmethod
    def poll(self) -> GenerationProgress:
        """Return a fresh, non-blocking snapshot of the job's current state.

        Must be cheap and side-effect-free enough to call in a UI loop. Implementations should never
        block on generation completing here — that is what :meth:`wait` is for.
        """

    @abstractmethod
    def _build_result(self) -> GenerationResult:
        """Assemble the final :class:`GenerationResult`. Only called after ``poll()`` reports FINISHED.

        For ComfyUI this downloads the finished image references; for WebUI it returns the images the
        background request thread already produced.
        """

    @abstractmethod
    def cancel(self) -> bool:
        """Attempt to cancel the job. Returns ``True`` if the server accepted the cancellation.

        Capability note: ComfyUI can drop a still-queued task *or* interrupt the running one; the WebUI
        can only interrupt the job that is *currently running* — a specific queued WebUI job cannot be
        cancelled server-side, so this returns ``False`` in that case. The WebUI's interrupt is not known to stop a
        running basic upscale or preprocessor preview, so cancelling one of those once it is running returns
        ``False`` and the job finishes normally.
        """

    # --- Concrete, shared conveniences ------------------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether the job has reached a terminal state (cheap poll)."""
        return self.poll().done

    def wait(self, timeout: Optional[float] = None, poll_interval: float = 0.5,
             on_progress: Optional[ProgressCallback] = None) -> GenerationResult:
        """Block until the job finishes, then return its result. This is the derived "blocking" path.

        Parameters
        ----------
        timeout: Optional[float]
            Max seconds to wait before raising ``BackendTimeoutError``. ``None`` waits indefinitely.
        poll_interval: float
            Seconds between ``poll()`` calls.
        on_progress: Optional[ProgressCallback]
            Invoked with each :class:`GenerationProgress` snapshot, for progress bars / live previews.

        Raises
        ------
        BackendTimeoutError
            If ``timeout`` elapses before the job reaches a terminal state.
        GenerationError
            If the job ends FAILED / CANCELLED / NOT_FOUND.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            progress = self.poll()
            if on_progress is not None:
                on_progress(progress)
            if progress.status is GenerationStatus.FINISHED:
                return self._build_result()
            if progress.status.is_terminal:
                raise GenerationError(progress.status, progress.text_info)
            if deadline is not None and time.monotonic() >= deadline:
                raise BackendTimeoutError(f'Generation {self._task_id!r} did not finish within {timeout}s')
            time.sleep(poll_interval)
