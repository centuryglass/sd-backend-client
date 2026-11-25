"""Exceptions raised by sd_backend_client.

Every failure that comes from talking to a backend raises a subclass of `SDBackendError`, so callers can catch the
package's errors as one group or pick out the cases they handle. Invalid caller input raises the builtin `ValueError`
and is not part of this tree.

The connection and timeout errors also inherit the builtin `ConnectionError` and `TimeoutError`, so code that already
catches those keeps working.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sd_backend_client.api.comfyui.comfyui_types import ErrorEntry, NodeErrorEntry
    from sd_backend_client.api.shared_data.generation_handle import GenerationStatus


class SDBackendError(Exception):
    """Base class for errors raised while talking to a Stable Diffusion backend."""


class BackendConnectionError(SDBackendError, ConnectionError):
    """The backend could not be reached, or the connection failed before a response arrived.

    Check that the server is running and that the URL is correct.
    """


class BackendTimeoutError(SDBackendError, TimeoutError):
    """The backend did not respond, or a job did not finish, within the allowed time.

    Raise the client's `request_timeout` or `generation_timeout`, or the `timeout` passed to the call.
    """


class AuthError(SDBackendError):
    """The server rejected authentication, and retrying with the available credentials did not fix it."""


class ServerError(SDBackendError):
    """The server answered a request with a failure status.

    Attributes
    ----------
    status_code: int
        The HTTP status code.
    body: str
        The response body as text, which usually holds the server's own error message.
    endpoint: str
        The endpoint the request was sent to.
    method: str
        The HTTP method used.
    """

    def __init__(self, status_code: int, body: str, endpoint: str, method: str) -> None:
        self.status_code = status_code
        self.body = body
        self.endpoint = endpoint
        self.method = method
        super().__init__(f'{method} {endpoint} failed with status {status_code}: {body}')


class UnexpectedResponseError(SDBackendError):
    """The server answered with success, but the response was not in the expected format.

    This usually means the server runs a version or fork this package does not support.
    """


class WorkflowValidationError(SDBackendError):
    """ComfyUI rejected a queued workflow before running it.

    Attributes
    ----------
    error: str | ErrorEntry | None
        The top-level error ComfyUI reported.
    node_errors: dict[str, NodeErrorEntry]
        Per-node errors, keyed by the workflow's node id. Empty when ComfyUI reported none.
    """

    def __init__(self, error: Optional[str | ErrorEntry], node_errors: dict[str, NodeErrorEntry]) -> None:
        self.error = error
        self.node_errors = node_errors
        super().__init__(self._describe(error, node_errors))

    @staticmethod
    def _describe(error: Optional[str | ErrorEntry], node_errors: dict[str, NodeErrorEntry]) -> str:
        parts: list[str] = []
        if isinstance(error, str):
            parts.append(error)
        elif error is not None:
            parts.append(error.message + (f': {error.details}' if error.details else ''))
        for node_id, node_error in node_errors.items():
            messages = '; '.join(entry.message + (f': {entry.details}' if entry.details else '')
                                 for entry in node_error.errors)
            parts.append(f'node {node_id} ({node_error.class_type}): {messages}')
        if not parts:
            parts.append('no error details returned')
        return 'ComfyUI rejected the workflow: ' + ' | '.join(parts)


class GenerationError(SDBackendError):
    """A generation job ended in a terminal state other than FINISHED.

    Raised by `GenerationHandle.wait`. `status` holds the job's final `GenerationStatus`. When the job failed because
    of another exception, that exception is the `__cause__`.
    """

    def __init__(self, status: GenerationStatus, message: Optional[str] = None) -> None:
        self.status = status
        super().__init__(message or f'Generation ended with status {status.value}')


class BackendLaunchError(SDBackendError):
    """A backend process this package started exited, or failed to launch, before it served.

    Also raised without launching when a server of that backend already answers on the port. Otherwise the last output
    lines usually say why, such as a missing dependency or a port already in use.

    Attributes
    ----------
    exit_code: int, optional
        The process's exit code, or None if it failed to launch or never started.
    output_tail: list[str]
        The last lines of the process's output, oldest first.
    """

    def __init__(self, message: str, exit_code: Optional[int], output_tail: list[str]) -> None:
        self.exit_code = exit_code
        self.output_tail = output_tail
        super().__init__(message)


class LaunchCancelledError(SDBackendError):
    """A wait for a backend process to start serving ended because it was cancelled or the process was stopped."""
