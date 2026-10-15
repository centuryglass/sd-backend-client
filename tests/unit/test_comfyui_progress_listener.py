"""Offline unit tests for ComfyProgressListener: websocket message parsing and the background thread's lifecycle.

Message shapes follow what ComfyUI's server.py sends: JSON text messages with `type` and `data`, and binary preview
frames that start with a big-endian event type.
"""
# pylint: disable=protected-access
import json
import struct
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import websocket
from PIL import Image

from sd_backend_client.api.comfyui.comfyui_progress_listener import (ComfyProgressListener, LiveProgress,
                                                                     SamplerRunPlan, _JobState)
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.errors import BackendConnectionError
from sd_backend_client.util.visual.image_utils import image_to_png_bytes

PROMPT_ID = 'prompt-a'
OTHER_ID = 'prompt-b'
JOIN_TIMEOUT = 10.0  # Only bounds a hung test; no assertion depends on timing.


class FakeClock:
    """A settable time source."""

    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


def _text(message_type: str, **data: Any) -> str:
    return json.dumps({'type': message_type, 'data': data})


def _preview_png() -> bytes:
    return image_to_png_bytes(Image.new('RGB', (3, 2), (255, 0, 0)))


def _listener(clock: FakeClock | None = None) -> ComfyProgressListener:
    def unused_websocket():
        raise AssertionError('the websocket should not be opened')
    return ComfyProgressListener(unused_websocket, clock=clock or FakeClock())


def _watched(listener: ComfyProgressListener, *prompt_ids: str) -> ComfyProgressListener:
    """Register jobs as watch() does, without starting the background thread."""
    for prompt_id in prompt_ids:
        listener._jobs[prompt_id] = _JobState(LiveProgress())
    return listener


# Text messages

def test_progress_message_sets_fraction_and_eta():
    """Progress is value/max, and the ETA extrapolates the step rate since the node's first message."""
    clock = FakeClock()
    listener = _watched(_listener(clock), PROMPT_ID)
    listener.handle_message(_text('progress', value=2, max=20, prompt_id=PROMPT_ID, node='3'))
    first = listener.get_progress(PROMPT_ID)
    assert first is not None and first.progress == 0.1 and first.eta_seconds is None

    clock.now += 4.0  # 2 steps in 4 s, 12 steps left
    listener.handle_message(_text('progress', value=4, max=20, prompt_id=PROMPT_ID, node='3'))
    second = listener.get_progress(PROMPT_ID)
    assert second is not None
    assert second.progress == 0.2
    assert second.eta_seconds == pytest.approx(32.0)


def test_new_node_restarts_the_eta():
    """A second sampler node's steps don't reuse the first node's rate."""
    clock = FakeClock()
    listener = _watched(_listener(clock), PROMPT_ID)
    listener.handle_message(_text('progress', value=1, max=10, prompt_id=PROMPT_ID, node='3'))
    clock.now += 1.0
    listener.handle_message(_text('progress', value=10, max=10, prompt_id=PROMPT_ID, node='3'))
    clock.now += 1.0
    listener.handle_message(_text('progress', value=1, max=5, prompt_id=PROMPT_ID, node='12'))
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.progress == 0.2 and progress.eta_seconds is None


def test_restarted_sampler_run_restarts_the_eta():
    """A node that samples again from step 1 starts a new run, so the ETA ignores the time spent on earlier runs."""
    clock = FakeClock()
    listener = _watched(_listener(clock), PROMPT_ID)
    for run in range(3):
        clock.now = 100.0 + run * 10.0
        listener.handle_message(_text('progress', value=1, max=10, prompt_id=PROMPT_ID, node='9'))
        clock.now += 4.0  # 1 s per step
        listener.handle_message(_text('progress', value=5, max=10, prompt_id=PROMPT_ID, node='9'))
        progress = listener.get_progress(PROMPT_ID)
        assert progress is not None and progress.progress == 0.5
        assert progress.eta_seconds == pytest.approx(5.0)
        clock.now += 5.0
        listener.handle_message(_text('progress', value=10, max=10, prompt_id=PROMPT_ID, node='9'))


def test_change_of_max_on_one_node_starts_a_new_run():
    """A run with a different max under the same node, like an upscale model's, doesn't share the sampler's rate."""
    clock = FakeClock()
    listener = _watched(_listener(clock), PROMPT_ID)
    listener.handle_message(_text('progress', value=1, max=4, prompt_id=PROMPT_ID, node='9'))
    clock.now += 30.0
    listener.handle_message(_text('progress', value=2, max=10, prompt_id=PROMPT_ID, node='9'))
    clock.now += 2.0
    listener.handle_message(_text('progress', value=4, max=10, prompt_id=PROMPT_ID, node='9'))
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.eta_seconds == pytest.approx(6.0)


def _tiled_job(plan: SamplerRunPlan | None) -> ComfyProgressListener:
    """A listener watching PROMPT_ID with a run plan, registered as watch() does without starting the thread."""
    listener = _listener()
    listener._jobs[PROMPT_ID] = _JobState(LiveProgress(), plan)
    return listener


def _sampler_run(listener: ComfyProgressListener, steps: int, node: str = '9') -> list[float | None]:
    """Send one sampler run as ComfyUI throttles it (first, middle and last step), returning overall progress after
    each message."""
    overall = []
    for value in (1, steps // 2, steps):
        listener.handle_message(_text('progress', value=value, max=steps, prompt_id=PROMPT_ID, node=node))
        progress = listener.get_progress(PROMPT_ID)
        assert progress is not None
        overall.append(progress.overall_progress)
    return overall


def test_run_plan_reports_progress_across_sampler_runs():
    """overall_progress counts finished runs plus the current run's fraction, while progress stays per run."""
    listener = _tiled_job(SamplerRunPlan(node='9', steps=8, total_runs=4))
    listener.handle_message(_text('progress', value=1, max=1, prompt_id=PROMPT_ID, node='9'))  # The upscale model.
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.overall_progress is None

    overall = [value for _ in range(4) for value in _sampler_run(listener, 8)]
    assert overall == pytest.approx([1 / 32, 4 / 32, 8 / 32, 9 / 32, 12 / 32, 16 / 32,
                                     17 / 32, 20 / 32, 24 / 32, 25 / 32, 28 / 32, 1.0])
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.progress == 1.0


def test_run_plan_ignores_other_nodes():
    """Progress from a node other than the planned one doesn't count as a sampler run."""
    listener = _tiled_job(SamplerRunPlan(node='9', steps=8, total_runs=2))
    _sampler_run(listener, 8, node='5')
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.overall_progress is None
    assert _sampler_run(listener, 8)[-1] == pytest.approx(0.5)


def test_run_plan_eta_covers_the_whole_job():
    """With a run plan, the ETA extrapolates overall progress since sampling began, decode time included."""
    clock = FakeClock()
    listener = _listener(clock)
    listener._jobs[PROMPT_ID] = _JobState(LiveProgress(), SamplerRunPlan(node='9', steps=10, total_runs=4,
                                                                          runs_before=1, run_after_each=True))

    def send(value: int, maximum: int, at: float):
        clock.now = at
        listener.handle_message(_text('progress', value=value, max=maximum, prompt_id=PROMPT_ID, node='9'))
        progress = listener.get_progress(PROMPT_ID)
        assert progress is not None
        return progress.eta_seconds

    assert send(1, 10, at=-5.0) is None  # The upscale model's run says nothing about sampling speed.
    assert send(1, 10, at=0.0) is None   # Sampling starts: steps finish at 1 s each.
    assert send(5, 10, at=4.0) == pytest.approx(35.0)
    send(10, 10, at=9.0)
    send(1, 3, at=9.5)  # The first tile's decode run.
    send(3, 3, at=10.0)
    send(1, 10, at=10.0)
    assert send(5, 10, at=14.0) == pytest.approx(25.0)


# The max of each progress run a live Ultimate SD Upscale job reported (3x3 tiles, Half Tile seam fix, 7 steps, tiled
# decode): one upscale model run, then each of the 21 sampler runs followed by its tile's VAE decode run.
_LIVE_RUN_MAXES = [4, 7, 10, 7, 10, 7, 7, 7, 10, 7, 10, 7, 7, 7, 7, 7, 7, 7, 5, 7, 10, 7, 10, 7, 7, 7, 10, 7, 10, 7,
                   7, 7, 10, 7, 10, 7, 10, 7, 10, 7, 7, 7, 7]


def test_run_plan_skips_upscale_and_decode_runs_whose_max_matches_the_steps():
    """Runs in the plan's non-sampler positions never count, even when their max equals the step count."""
    listener = _tiled_job(SamplerRunPlan(node='9', steps=7, total_runs=21, runs_before=1, run_after_each=True))
    overall = []
    for maximum in _LIVE_RUN_MAXES:
        overall.extend(_sampler_run(listener, maximum))
    sampler_finals = [overall[index * 3 + 2] for index in range(1, len(_LIVE_RUN_MAXES), 2)]
    assert sampler_finals == pytest.approx([run / 21 for run in range(1, 22)])
    decode_finals = [overall[index * 3 + 2] for index in range(2, len(_LIVE_RUN_MAXES), 2)]
    assert decode_finals == sampler_finals  # A decode run leaves overall progress where its sampler run ended.


def test_run_plan_caps_unplanned_runs_at_complete():
    """A sampler-like run the plan doesn't predict is counted, but overall progress stays monotonic and never passes 1.
    """
    listener = _tiled_job(SamplerRunPlan(node='9', steps=4, total_runs=2))
    overall = [value or 0.0 for _ in range(3) for value in _sampler_run(listener, 4)]
    assert overall == sorted(overall)
    assert overall[-4:] == [1.0] * 4


@pytest.mark.parametrize('plan', [None, SamplerRunPlan(node='9', steps=8, total_runs=0)])
def test_no_overall_progress_without_a_usable_plan(plan: SamplerRunPlan | None):
    """Jobs without a run plan, or whose plan expects no sampler runs, report no overall progress."""
    listener = _tiled_job(plan)
    assert _sampler_run(listener, 8) == [None, None, None]


def test_progress_without_prompt_id_goes_to_the_executing_job():
    """Older ComfyUI versions omit prompt_id from progress; it belongs to the job named by 'executing'."""
    listener = _watched(_listener(), PROMPT_ID, OTHER_ID)
    listener.handle_message(_text('executing', node='3', prompt_id=OTHER_ID))
    listener.handle_message(_text('progress', value=5, max=10))
    other = listener.get_progress(OTHER_ID)
    assert other is not None and other.progress == 0.5
    own = listener.get_progress(PROMPT_ID)
    assert own is not None and own.progress is None


@pytest.mark.parametrize('message', [
    _text('progress', value=5, max=10, prompt_id='unwatched'),
    _text('progress', value=5, max=0, prompt_id=PROMPT_ID),
    _text('progress', value='5', max=10, prompt_id=PROMPT_ID),
    _text('status', status={'exec_info': {'queue_remaining': 1}}),
    '{"type": "progress"}',
    '[1, 2]',
    'not json',
])
def test_irrelevant_or_malformed_messages_change_nothing(message: str):
    """Messages for other jobs, unknown types and malformed bodies leave watched jobs untouched."""
    listener = _watched(_listener(), PROMPT_ID)
    listener.handle_message(message)
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.progress is None


@pytest.mark.parametrize('message', [
    _text('executing', node=None, prompt_id=PROMPT_ID),
    _text('execution_success', prompt_id=PROMPT_ID, timestamp=1),
    _text('execution_error', prompt_id=PROMPT_ID, node_id='3', exception_message='boom'),
    _text('execution_interrupted', prompt_id=PROMPT_ID, node_id='3'),
])
def test_job_end_messages_stop_watching_the_job(message: str):
    """A job the server reports ended is no longer watched, even if its handle is never polled again."""
    listener = _watched(_listener(), PROMPT_ID, OTHER_ID)
    listener.handle_message(message)
    assert listener.get_progress(PROMPT_ID) is None
    assert listener.get_progress(OTHER_ID) is not None


# Binary preview frames

def test_preview_frame_goes_to_the_executing_job():
    """A PREVIEW_IMAGE frame has no prompt id, so it belongs to the job named by 'executing'."""
    listener = _watched(_listener(), PROMPT_ID)
    listener.handle_message(_text('executing', node='3', prompt_id=PROMPT_ID))
    listener.handle_message(struct.pack('>II', 1, 2) + _preview_png())
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.preview is not None
    assert progress.preview.size == (3, 2)
    assert progress.preview.mode == 'RGBA'


def test_preview_frame_with_metadata_uses_its_prompt_id():
    """A PREVIEW_IMAGE_WITH_METADATA frame names its job in a JSON header."""
    listener = _watched(_listener(), PROMPT_ID, OTHER_ID)
    listener.handle_message(_text('executing', node='3', prompt_id=PROMPT_ID))
    metadata = json.dumps({'prompt_id': OTHER_ID, 'node_id': '3', 'image_type': 'image/png'}).encode()
    listener.handle_message(struct.pack('>II', 4, len(metadata)) + metadata + _preview_png())
    other = listener.get_progress(OTHER_ID)
    assert other is not None and other.preview is not None
    own = listener.get_progress(PROMPT_ID)
    assert own is not None and own.preview is None


@pytest.mark.parametrize('message', [
    b'\x00\x00',
    struct.pack('>II', 1, 2) + b'not an image',
    struct.pack('>II', 4, 5) + b'{bad}' + b'',
    struct.pack('>II', 3, 0) + b'text frame',
])
def test_bad_or_unknown_binary_frames_are_ignored(message: bytes):
    """Short, undecodable and unknown binary frames set no preview."""
    listener = _watched(_listener(), PROMPT_ID)
    listener.handle_message(_text('executing', node='3', prompt_id=PROMPT_ID))
    listener.handle_message(message)
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.preview is None


# Background thread

class _FakeConnection:
    """Replays scripted messages, then reports read timeouts until `release` is set."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        self.drained = threading.Event()
        self.release = threading.Event()
        self.timeout: float | None = None

    def settimeout(self, timeout: float) -> None:
        """Record the read timeout."""
        self.timeout = timeout

    def recv(self) -> str:
        """Return the next scripted message, or time out once they run out."""
        if self.messages:
            return self.messages.pop(0)
        self.drained.set()
        self.release.wait(JOIN_TIMEOUT)
        raise websocket.WebSocketTimeoutException('timed out')


def test_thread_reads_messages_and_exits_once_nothing_is_watched():
    """watch() starts one thread that applies messages and stops after the last job is unwatched."""
    connection = _FakeConnection([_text('progress', value=3, max=4, prompt_id=PROMPT_ID)])
    opened: list[int] = []

    @contextmanager
    def open_websocket() -> Iterator[_FakeConnection]:
        opened.append(1)
        yield connection

    listener = ComfyProgressListener(open_websocket, receive_timeout=0.25)
    listener.watch(PROMPT_ID)
    thread = listener._thread
    assert thread is not None
    listener.watch(OTHER_ID)  # a running thread serves every watched job
    assert connection.drained.wait(JOIN_TIMEOUT)
    progress = listener.get_progress(PROMPT_ID)
    assert progress is not None and progress.progress == 0.75
    assert connection.timeout == 0.25

    listener.unwatch(PROMPT_ID)
    listener.unwatch(OTHER_ID)
    connection.release.set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert listener._thread is None
    assert opened == [1]


def test_failed_connection_waits_for_the_retry_interval():
    """After the websocket fails to open, watch() doesn't reconnect until retry_interval has passed."""
    clock = FakeClock()
    attempts: list[int] = []

    def open_websocket():
        attempts.append(1)
        raise BackendConnectionError('refused')

    listener = ComfyProgressListener(open_websocket, clock=clock, retry_interval=5.0)
    listener.watch(PROMPT_ID)
    thread = listener._thread
    assert thread is not None
    thread.join(JOIN_TIMEOUT)
    assert attempts == [1] and listener._thread is None

    clock.now += 4.0
    listener.watch(PROMPT_ID)
    assert listener._thread is None

    clock.now += 1.0
    listener.watch(PROMPT_ID)
    thread = listener._thread
    assert thread is not None
    thread.join(JOIN_TIMEOUT)
    assert attempts == [1, 1]


def test_service_creates_a_listener_unless_live_progress_is_off():
    """ComfyUiWebservice has a progress listener by default, and none with live_progress=False."""
    assert isinstance(ComfyUiWebservice('http://127.0.0.1:8188').progress_listener, ComfyProgressListener)
    assert ComfyUiWebservice('http://127.0.0.1:8188', live_progress=False).progress_listener is None
