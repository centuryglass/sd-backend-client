"""Offline tests for ComfyUiWebservice's HTTP requests and response parsing, against a fake `requests.Session`.

The response bodies follow the shapes ComfyUI's /history, /queue, /object_info and /upload endpoints return.
"""
# pylint: disable=protected-access
import io
import json
import re
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit
from unittest.mock import MagicMock

import pytest
import websocket
from PIL import Image

from sd_backend_client.api.comfyui.comfyui_progress_listener import SamplerRunPlan
from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui_webservice import (AsyncTaskStatus, ComfyEndpoints, ComfyModelType,
                                                      ComfyUiWebservice, EXTENDED_TIMEOUT)
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from sd_backend_client.errors import BackendConnectionError, BackendTimeoutError, ServerError, \
    UnexpectedResponseError, WorkflowValidationError
from sd_backend_client.util.visual.image_utils import image_to_png_bytes

from .fake_session import FakeSession as _FakeSession, CannedStatus as _Status

PROMPT_ID = '7d4c0f0e-6a3b-4c1e-9f6e-0a1b2c3d4e5f'
OTHER_ID = '11111111-2222-3333-4444-555555555555'


def _service(routes: Optional[dict[tuple[str, str], Any]] = None) -> tuple[ComfyUiWebservice, _FakeSession]:
    service = ComfyUiWebservice('http://127.0.0.1:8188')
    session = _FakeSession(routes)
    service._session = session  # type: ignore[assignment]
    return service, session


def _history_entry(status_str: str = 'success', completed: bool = True,
                   outputs: Optional[dict[str, Any]] = None,
                   messages: Optional[list[list[Any]]] = None) -> dict[str, Any]:
    """One /history/<prompt_id> entry, keyed by prompt id."""
    if outputs is None:
        outputs = {'9': {'images': [{'filename': 'ComfyUI_00001_.png', 'subfolder': '', 'type': 'output'}]}}
    return {PROMPT_ID: {
        'prompt': [7, PROMPT_ID, {}, {'client_id': 'abc'}, ['9']],
        'outputs': outputs,
        'status': {'status_str': status_str, 'completed': completed,
                   'messages': [['execution_start', {'prompt_id': PROMPT_ID, 'timestamp': 1700000000000}]]
                   + (messages or [])},
    }}


def _queue(running: list[tuple[int, str]], pending: list[tuple[int, str]]) -> dict[str, Any]:
    """A /queue response holding (number, prompt_id) entries."""
    return {'queue_running': [[number, prompt_id, {}, {}, ['9']] for number, prompt_id in running],
            'queue_pending': [[number, prompt_id, {}, {}, ['9']] for number, prompt_id in pending]}


# check_queue_entry

def test_finished_history_entry_collects_images_from_every_output_node():
    """A completed entry is FINISHED, with the image references of all output nodes."""
    outputs = {'9': {'images': [{'filename': 'a.png', 'subfolder': '', 'type': 'output'}]},
               '12': {'images': [{'filename': 'b.png', 'subfolder': 'sub', 'type': 'output'}]}}
    service, session = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry(outputs=outputs)})
    progress = service.check_queue_entry(PROMPT_ID, 7)

    assert progress.status is AsyncTaskStatus.FINISHED
    assert progress.outputs is not None
    assert [(ref.filename, ref.subfolder) for ref in progress.outputs.images] == [('a.png', ''), ('b.png', 'sub')]
    assert not session.of('GET', ComfyEndpoints.QUEUE)


def test_error_history_entry_is_failed():
    """An entry whose status_str is 'error' is FAILED."""
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry('error', False, outputs={})})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.FAILED


def test_error_history_entry_carries_the_execution_error():
    """A FAILED status reports the node and exception from ComfyUI's execution_error message."""
    error = ['execution_error', {'prompt_id': PROMPT_ID, 'timestamp': 1700000000001, 'node_id': '3',
                                 'node_type': 'KSampler', 'executed': [], 'exception_type': 'RuntimeError',
                                 'exception_message': 'CUDA out of memory', 'traceback': ['...'],
                                 'current_inputs': {}, 'current_outputs': {}}]
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry('error', False, outputs={},
                                                                           messages=[error])})
    progress = service.check_queue_entry(PROMPT_ID)
    assert progress.status is AsyncTaskStatus.FAILED
    assert progress.error == ('ComfyUI execution failed in KSampler (node 3): RuntimeError: CUDA out of memory')


def test_interrupted_history_entry_says_where_it_stopped():
    """An interrupted run is FAILED, with a reason naming the node it stopped in."""
    interrupted = ['execution_interrupted', {'prompt_id': PROMPT_ID, 'timestamp': 1700000000001, 'node_id': '3',
                                             'node_type': 'KSampler', 'executed': []}]
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry('error', False, outputs={},
                                                                           messages=[interrupted])})
    assert service.check_queue_entry(PROMPT_ID).error == 'ComfyUI execution was interrupted in KSampler (node 3)'


def test_output_nodes_without_images_still_finish():
    """An output node that saves no images doesn't stop the entry from reading as FINISHED."""
    outputs = {'9': {'images': [{'filename': 'a.png', 'subfolder': '', 'type': 'output'}]},
               '15': {'text': ['a caption']}}
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry(outputs=outputs)})
    progress = service.check_queue_entry(PROMPT_ID)
    assert progress.status is AsyncTaskStatus.FINISHED
    assert progress.outputs is not None
    assert [ref.filename for ref in progress.outputs.images] == ['a.png']


def test_malformed_history_entry_raises_unexpected_response():
    """A history entry that doesn't match ComfyUI's format raises UnexpectedResponseError."""
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): {PROMPT_ID: {'status': 'done'}}})
    with pytest.raises(UnexpectedResponseError, match='unexpected format'):
        service.check_queue_entry(PROMPT_ID)


def test_entry_finishing_between_history_and_queue_reads_is_finished():
    """A job that leaves the queue after the first /history read is found by a second /history read."""
    history_reads: list[int] = []

    def history() -> dict[str, Any]:
        history_reads.append(1)
        return {} if len(history_reads) == 1 else _history_entry()

    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): history,
                           ('GET', ComfyEndpoints.QUEUE): _queue([], [])})
    assert service.check_queue_entry(PROMPT_ID).status is AsyncTaskStatus.FINISHED
    assert len(history_reads) == 2


def test_running_entry_is_active():
    """A prompt missing from history but in queue_running is ACTIVE."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue([(7, PROMPT_ID)], [])})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.ACTIVE


def test_incomplete_history_entry_falls_through_to_queue():
    """A history entry not yet completed is resolved from the queue."""
    service, _ = _service({('GET', f'/history/{PROMPT_ID}'): _history_entry(completed=False, outputs={}),
                           ('GET', ComfyEndpoints.QUEUE): _queue([(7, PROMPT_ID)], [])})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.ACTIVE


def test_pending_entry_index_counts_lower_numbers():
    """A pending prompt's index is the count of pending entries with a lower number, in any listed order."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue(
        [(3, OTHER_ID)], [(9, 'later'), (7, PROMPT_ID), (5, 'earlier-a'), (4, 'earlier-b')])})
    progress = service.check_queue_entry(PROMPT_ID, 7)
    assert progress.status is AsyncTaskStatus.PENDING
    assert progress.index == 2


def test_pending_entry_is_found_without_a_task_number():
    """A pending prompt is matched by its id, so the queue number is optional."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue([], [(9, 'later'), (7, PROMPT_ID), (5, 'a')])})
    progress = service.check_queue_entry(PROMPT_ID, None)
    assert progress.status is AsyncTaskStatus.PENDING
    assert progress.index == 1


def test_unknown_entry_is_not_found():
    """A prompt in neither history nor queue is NOT_FOUND."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): _queue([(3, OTHER_ID)], [(9, 'later')])})
    assert service.check_queue_entry(PROMPT_ID, 7).status is AsyncTaskStatus.NOT_FOUND


# Queue control

def test_interrupt_with_task_deletes_it_then_interrupts():
    """interrupt(task_id) deletes the task from the queue and then posts an /interrupt naming it."""
    service, session = _service()
    service.interrupt(PROMPT_ID)
    assert [request['path'] for request in session.requests] == [ComfyEndpoints.QUEUE, ComfyEndpoints.INTERRUPT]
    assert session.requests[0]['json'] == {'clear': None, 'delete': [PROMPT_ID]}
    assert session.requests[1]['json'] == {'prompt_id': PROMPT_ID}


def test_interrupt_without_task_stops_the_running_job():
    """interrupt() posts a bare /interrupt and touches no queue entry."""
    service, session = _service()
    service.interrupt()
    assert [request['path'] for request in session.requests] == [ComfyEndpoints.INTERRUPT]
    assert session.requests[0].get('json') is None


def test_remove_from_queue_does_not_interrupt():
    """remove_from_queue only posts the queue deletion."""
    service, session = _service()
    service.remove_from_queue(PROMPT_ID)
    assert [request['path'] for request in session.requests] == [ComfyEndpoints.QUEUE]
    assert session.requests[0]['json']['delete'] == [PROMPT_ID]


# Option discovery

KSAMPLER_INFO = {'KSampler': {
    'input': {'required': {
        'model': ['MODEL'],
        'seed': ['INT', {'default': 0, 'min': 0, 'max': 18446744073709551615}],
        'sampler_name': [['euler', 'dpmpp_2m'], {}],
        'scheduler': [['normal', 'karras'], {}],
    }},
    'input_order': {'required': ['model', 'seed', 'sampler_name', 'scheduler']},
    'output': ['LATENT'], 'output_is_list': [False], 'output_name': ['LATENT'], 'name': 'KSampler',
    'display_name': 'KSampler', 'description': '', 'python_module': 'nodes', 'category': 'sampling',
    'output_node': False,
}}


def test_sampler_and_scheduler_names_come_from_ksampler_info_fetched_once():
    """Sampler and scheduler options are read from /object_info/KSampler, which is cached."""
    service, session = _service({('GET', '/object_info/KSampler'): KSAMPLER_INFO})
    assert service.get_sampler_names() == ['euler', 'dpmpp_2m']
    assert service.get_scheduler_names() == ['normal', 'karras']
    assert len(session.of('GET', '/object_info/KSampler')) == 1


def test_is_node_available_checks_for_node_key():
    """is_node_available is true only when /object_info/<name> returns that node."""
    service, _ = _service({('GET', '/object_info/KSampler'): KSAMPLER_INFO})
    assert service.is_node_available('KSampler')
    assert not service.is_node_available('UltimateSDUpscale')


def test_get_models_requests_the_model_type_folder():
    """get_models lists one model folder."""
    service, session = _service({('GET', '/models/loras'): ['detail.safetensors']})
    assert service.get_models(ComfyModelType.LORA) == ['detail.safetensors']
    assert session.requests[0]['timeout'] is not None


def test_preprocessor_discovery_skips_nodes_that_fail_validation():
    """Nodes in /object_info that don't match NodeInfoResponse are skipped, and the result is cached."""
    canny = {
        'input': {'required': {'image': ['IMAGE'],
                               'low_threshold': ['INT', {'default': 100, 'min': 1, 'max': 255}]}},
        'input_order': {'required': ['image', 'low_threshold']},
        'output': ['IMAGE'], 'output_is_list': [False], 'output_name': ['IMAGE'], 'name': 'CannyEdgePreprocessor',
        'display_name': 'Canny Edge', 'description': '', 'python_module': 'custom_nodes.controlnet_aux',
        'category': 'ControlNet Preprocessors/Line Extractors', 'output_node': False,
    }
    service, session = _service({('GET', ComfyEndpoints.OBJECT_INFO): {'CannyEdgePreprocessor': canny,
                                                                      'BrokenNode': {'name': 'BrokenNode'}}})
    names = [preprocessor.name for preprocessor in service.get_controlnet_preprocessors()]
    assert names == ['CannyEdgePreprocessor']
    service.get_controlnet_preprocessors()
    assert len(session.of('GET', ComfyEndpoints.OBJECT_INFO)) == 1


# Uploads and downloads

def _upload_response(name: str = 'uploaded.png') -> dict[str, str]:
    return {'name': name, 'subfolder': 'sd_backend_client', 'type': 'input'}


def _uploaded_names(session: _FakeSession) -> list[str]:
    return [upload['files']['image'][0] for upload in session.of('POST', ComfyEndpoints.IMG_UPLOAD)]


def test_upload_image_sends_png_multipart_named_by_content():
    """upload_image posts a PNG as multipart form data under a content-hash name, without overwriting."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response()})
    image = Image.new('RGBA', (3, 2), (1, 2, 3, 255))
    reference = service.upload_image(image)

    assert reference == ImageFileReference(filename='uploaded.png', subfolder='sd_backend_client', type='input')
    uploads = session.of('POST', ComfyEndpoints.IMG_UPLOAD)
    assert len(uploads) == 1
    assert uploads[0]['data'] == {'type': 'input', 'subfolder': 'sd_backend_client'}
    name, data, content_type = uploads[0]['files']['image']
    assert re.fullmatch(r'[0-9a-f]{32}\.png', name)
    assert content_type == 'image/png'
    assert Image.open(io.BytesIO(data)).size == (3, 2)


def test_upload_image_names_differ_for_different_images():
    """Two different images never share an upload name, so a queued job can't load a later job's image."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response()})
    service.upload_image(Image.new('RGBA', (3, 2), (1, 2, 3, 255)))
    service.upload_image(Image.new('RGBA', (3, 2), (4, 5, 6, 255)))
    service.upload_image(Image.new('RGBA', (2, 3), (1, 2, 3, 255)))
    service.upload_image(Image.new('RGBA', (3, 2), (1, 2, 3, 255)))

    names = _uploaded_names(session)
    assert len(set(names[:3])) == 3
    assert names[3] == names[0]


def test_upload_image_uses_server_returned_reference():
    """The returned reference is the server's, which may rename a file whose name is taken."""
    service, _ = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response('control (1).png')})
    reference = service.upload_image(Image.new('RGBA', (1, 1)), 'control')
    assert reference.filename == 'control (1).png'


def test_upload_image_adds_png_extension_to_name():
    """A name without .png gets one."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response('control_0.png')})
    service.upload_image(Image.new('RGBA', (1, 1)), 'control_0')
    assert _uploaded_names(session) == ['control_0.png']


def test_upload_image_overwrite_sends_overwrite_flag():
    """overwrite=True asks the server to replace a same-named file."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response()})
    service.upload_image(Image.new('RGBA', (1, 1)), 'fixed', overwrite=True)
    assert session.requests[0]['data']['overwrite'] == '1'


def test_upload_image_rejects_non_base64_string():
    """A string that is neither a file path nor base64 raises ValueError."""
    service, _ = _service()
    with pytest.raises(ValueError, match='base64'):
        service.upload_image('not base64!')


def test_download_images_sends_reference_as_query():
    """Each reference becomes /view query parameters, and each downloaded image is returned in order."""
    png = image_to_png_bytes(Image.new('RGB', (2, 2), (255, 0, 0)))
    service, session = _service({('GET', ComfyEndpoints.VIEW_IMAGE): png})
    refs = [ImageFileReference(filename='a.png', subfolder='', type='output'),
            ImageFileReference(filename='b.png', subfolder='sub')]
    images = service.download_images(refs)

    assert len(images) == 2
    assert images[0].mode == 'RGBA'
    assert [request['query'] for request in session.requests] == [
        {'filename': 'a.png', 'subfolder': '', 'type': 'output'},
        {'filename': 'b.png', 'subfolder': 'sub'}]


def test_download_images_raises_when_an_image_is_missing():
    """A failed download raises instead of returning fewer images than were generated."""
    service, _ = _service({('GET', ComfyEndpoints.VIEW_IMAGE): _Status(404, 'not found')})
    with pytest.raises(ServerError, match='404: not found'):
        service.download_images([ImageFileReference(filename='a.png', subfolder='')])


def test_download_images_raises_when_data_is_not_an_image():
    """Data that does not decode as an image raises UnexpectedResponseError naming the file."""
    service, _ = _service({('GET', ComfyEndpoints.VIEW_IMAGE): b'not an image'})
    with pytest.raises(UnexpectedResponseError, match='a.png'):
        service.download_images([ImageFileReference(filename='a.png', subfolder='')])


def test_download_images_encodes_reserved_characters_in_query():
    """Filenames and subfolders containing '&', '#', '?' or spaces reach the server intact."""
    png = image_to_png_bytes(Image.new('RGB', (1, 1)))
    service, session = _service({('GET', ComfyEndpoints.VIEW_IMAGE): png})
    service.download_images([ImageFileReference(filename='a&b #1?.png', subfolder='x y/z', type='output')])
    assert session.requests[0]['query'] == {'filename': 'a&b #1?.png', 'subfolder': 'x y/z', 'type': 'output'}


def test_is_node_available_encodes_node_name_as_one_path_segment():
    """A node name containing '/' or '#' stays one /object_info path segment."""
    service, session = _service({('GET', f'{ComfyEndpoints.OBJECT_INFO}/a%2Fb%23c'): {'a/b#c': {}}})
    assert service.is_node_available('a/b#c')
    assert session.requests[0]['path'] == f'{ComfyEndpoints.OBJECT_INFO}/a%2Fb%23c'


def test_requests_use_the_constructor_timeout_unless_they_set_their_own():
    """Plain requests get request_timeout; image downloads keep their longer fixed timeout."""
    service = ComfyUiWebservice('http://127.0.0.1:8188', request_timeout=7)
    session = _FakeSession({('GET', ComfyEndpoints.VIEW_IMAGE): image_to_png_bytes(Image.new('RGB', (1, 1)))})
    service._session = session  # type: ignore[assignment]
    service.get_embeddings()
    service.download_images([ImageFileReference(filename='a.png', subfolder='')])
    assert [request['timeout'] for request in session.requests] == [7, EXTENDED_TIMEOUT]


# Generation requests

def test_txt2img_posts_workflow_with_client_id_and_returns_seed():
    """txt2img queues the built workflow under this client's id and reports the seed it used."""
    service, session = _service({('GET', '/models/configs'): [],
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 7,
                                                                   'node_errors': {}}})
    params = ComfyUIDiffusionParams(prompt='a cat', sd_model_name='model.safetensors', seed=1234,
                                    denoising_strength=0.4)
    response = service.txt2img(params)

    assert (response.prompt_id, response.number, response.seed) == (PROMPT_ID, 7, 1234)
    body = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']
    assert body['client_id'] == service._client_id
    samplers = [node for node in body['prompt'].values() if node['class_type'] == 'KSampler']
    assert len(samplers) == 1
    assert samplers[0]['inputs']['seed'] == 1234
    assert samplers[0]['inputs']['denoise'] == 1.0


# ComfyUI's /prompt rejection body: an `error` plus `node_errors` keyed by node id.
_REJECTION = {
    'error': {'type': 'prompt_outputs_failed_validation', 'message': 'Prompt outputs failed validation',
              'details': '', 'extra_info': {}},
    'node_errors': {'4': {
        'errors': [{'type': 'value_not_in_list', 'message': 'Value not in list',
                    'details': "ckpt_name: 'missing.safetensors' not in ['model.safetensors']",
                    'extra_info': {'input_name': 'ckpt_name', 'input_config': [['model.safetensors']],
                                   'received_value': 'missing.safetensors'}}],
        'dependent_outputs': ['9'], 'class_type': 'CheckpointLoaderSimple'}},
}


def test_rejected_workflow_raises_workflow_validation_error():
    """A 400 from /prompt with ComfyUI's error body raises WorkflowValidationError carrying the node errors."""
    service, _ = _service({('GET', '/models/configs'): [],
                           ('POST', ComfyEndpoints.PROMPT): _Status(400, json.dumps(_REJECTION))})
    expected = re.escape("node 4 (CheckpointLoaderSimple): Value not in list: ckpt_name: 'missing.safetensors'")
    with pytest.raises(WorkflowValidationError, match=expected) as error:
        service.txt2img(ComfyUIDiffusionParams(sd_model_name='missing.safetensors'))
    assert error.value.node_errors['4'].errors[0].extra_info['input_name'] == 'ckpt_name'
    assert isinstance(error.value.__cause__, ServerError)


def test_prompt_400_without_error_body_stays_a_server_error():
    """A 400 whose body is not ComfyUI's rejection format is reported as a plain ServerError."""
    service, _ = _service({('GET', '/models/configs'): [],
                           ('POST', ComfyEndpoints.PROMPT): _Status(400, 'Bad Request')})
    with pytest.raises(ServerError, match='400: Bad Request') as error:
        service.txt2img(ComfyUIDiffusionParams(sd_model_name='model.safetensors'))
    assert not isinstance(error.value, WorkflowValidationError)


def test_queue_info_without_queue_lists_raises_unexpected_response():
    """A /queue response missing its queue lists raises UnexpectedResponseError."""
    service, _ = _service({('GET', ComfyEndpoints.QUEUE): {'queue_running': []}})
    with pytest.raises(UnexpectedResponseError, match='queue_pending'):
        service.get_queue_info()


def test_basic_upscale_without_an_installed_model_raises_value_error():
    """Basic upscaling with a model the server does not have is a caller error, raised before uploading the image."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                           ('GET', '/models/upscale_models'): ['4x.pth']})
    params = DiffusionUpscalingParams(upscaling_mode='missing.pth', use_stable_diffusion_upscaling=False)
    with pytest.raises(ValueError, match='missing.pth'):
        service.upscale(Image.new('RGBA', (8, 8)), 16, 16, params)
    assert not session.of('POST', ComfyEndpoints.IMG_UPLOAD)


def test_basic_upscale_without_a_model_name_uses_the_first_installed_model():
    """A basic upscale with no upscaling_mode loads the server's first upscaling model and reports no seed."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                                 ('GET', '/models/upscale_models'): ['4x.pth', '2x.pth'],
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    response = service.upscale(Image.new('RGBA', (8, 8)), 16, 16)

    prompt = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']['prompt']
    [loader] = [node for node in prompt.values() if node['class_type'] == 'UpscaleModelLoader']
    assert loader['inputs']['model_name'] == '4x.pth'
    assert response.seed is None


def test_diffusion_upscale_reports_its_seed():
    """A Stable Diffusion upscale's queue response carries the seed its sampler uses."""
    service, _ = _service({('GET', '/models/configs'): [],
                           ('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                           ('GET', '/models/upscale_models'): [],
                           ('GET', ComfyEndpoints.OBJECT_INFO): {},
                           ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    params = DiffusionUpscalingParams(use_stable_diffusion_upscaling=True, use_ultimate_upscale_script=False,
                                      diffusion_params=ComfyUIDiffusionParams(sd_model_name='model.safetensors',
                                                                              seed=4321))
    assert service.upscale(Image.new('RGBA', (8, 8)), 16, 16, params).seed == 4321


def test_ultimate_upscale_handle_carries_a_sampler_run_plan():
    """An Ultimate SD Upscale job's handle plans the tile node's sampler runs, so it can report overall progress."""
    service, session = _service({('GET', '/models/configs'): [],
                                 ('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                                 ('GET', '/models/upscale_models'): ['4x.pth'],
                                 ('GET', f'{ComfyEndpoints.OBJECT_INFO}/UltimateSDUpscale'): {'UltimateSDUpscale': {}},
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    params = DiffusionUpscalingParams(use_stable_diffusion_upscaling=True, upscaling_mode='4x.pth', step_count=12,
                                      tile_width=512, tile_height=512, seam_fix_mode='Band Pass',
                                      diffusion_params=ComfyUIDiffusionParams(sd_model_name='model.safetensors'))
    handle = service.submit_upscale(Image.new('RGBA', (512, 384)), 1024, 768, params)

    prompt = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']['prompt']
    [node_key] = [key for key, node in prompt.items() if node['class_type'] == 'UltimateSDUpscale']
    assert handle._run_plan == SamplerRunPlan(node=node_key, steps=12, total_runs=2 * 2 + (1 + 1), runs_before=1,
                                              run_after_each=True)


def test_basic_upscale_handle_has_no_sampler_run_plan():
    """A basic upscale has no tiled sampler, so its handle reports no overall progress."""
    service, _ = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                           ('GET', '/models/upscale_models'): ['4x.pth'],
                           ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    assert service.submit_upscale(Image.new('RGBA', (8, 8)), 16, 16)._run_plan is None


@pytest.mark.parametrize('mask', [None, Image.new('L', (8, 8), 255)])
@pytest.mark.parametrize('has_mask_input', [False, True])
def test_preprocessor_preview_uploads_a_mask_only_when_given_and_accepted(mask: Optional[Image.Image],
                                                                          has_mask_input: bool):
    """The preview uploads the source image, and the mask only when one is given and the preprocessor takes it."""
    service, session = _service({('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                                 ('POST', ComfyEndpoints.MASK_UPLOAD): _upload_response('mask.png'),
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    preprocessor = ControlNetPreprocessor(name='InpaintPreprocessor', has_mask_input=has_mask_input)
    service.controlnet_preprocessor_preview(Image.new('RGBA', (8, 8)), mask, preprocessor)

    assert len(session.of('POST', ComfyEndpoints.IMG_UPLOAD)) == 1
    mask_sent = mask is not None and has_mask_input
    assert len(session.of('POST', ComfyEndpoints.MASK_UPLOAD)) == int(mask_sent)
    prompt = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']['prompt']
    assert any(node['class_type'] == 'LoadImageMask' for node in prompt.values()) is mask_sent


def test_img2img_uploads_init_image_and_wires_it_into_the_workflow():
    """img2img uploads the init image and loads the uploaded reference in the graph."""
    service, session = _service({('GET', '/models/configs'): [],
                                 ('POST', ComfyEndpoints.IMG_UPLOAD): _upload_response(),
                                 ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1}})
    params = ComfyUIDiffusionParams(sd_model_name='model.safetensors', init_images=[Image.new('RGBA', (8, 8))])
    service.img2img(params)

    assert len(session.of('POST', ComfyEndpoints.IMG_UPLOAD)) == 1
    prompt = session.of('POST', ComfyEndpoints.PROMPT)[0]['json']['prompt']
    load_nodes = [node for node in prompt.values() if node['class_type'] == 'LoadImage']
    assert [node['inputs']['image'] for node in load_nodes] == ['sd_backend_client/uploaded.png']


def test_img2img_and_inpaint_require_their_inputs():
    """img2img needs an init image; inpaint also needs a mask. Neither contacts the server when they are missing."""
    service, session = _service()
    with pytest.raises(ValueError, match='init image'):
        service.img2img(ComfyUIDiffusionParams())
    with pytest.raises(ValueError, match='mask'):
        service.inpaint(ComfyUIDiffusionParams(init_images=[Image.new('RGBA', (8, 8))]))
    assert not session.requests


# Websocket progress parsing

@pytest.mark.parametrize('message, expected', [
    ('{"type": "progress", "data": {"value": 5, "max": 20}}', 25.0),
    ('{"type": "status", "data": {"status": {}}}', None),
    ('{"type": "progress", "data": {}}', None),
    ('not json', None),
])
def test_parse_percentage_from_websocket_message(message: str, expected: Optional[float]):
    """Only progress messages with value and max produce a percentage."""
    assert ComfyUiWebservice.parse_percentage_from_websocket_message(message) == expected


@pytest.mark.parametrize('server_url, expected', [
    ('http://127.0.0.1:8188', 'ws://127.0.0.1:8188/ws'),
    ('http://127.0.0.1:8188/', 'ws://127.0.0.1:8188/ws'),
    ('https://comfy.example.com/proxy/comfy/', 'wss://comfy.example.com/proxy/comfy/ws'),
])
def test_open_websocket_maps_scheme_and_keeps_base_path(monkeypatch, server_url: str, expected: str):
    """https servers connect over wss, and a base path in the server URL is kept."""
    socket = MagicMock()
    monkeypatch.setattr(websocket, 'WebSocket', lambda: socket)
    service = ComfyUiWebservice(server_url)
    with service.open_websocket():
        pass
    url = urlsplit(socket.connect.call_args.args[0])
    assert f'{url.scheme}://{url.netloc}{url.path}' == expected
    assert dict(parse_qsl(url.query)) == {'clientId': service._client_id}
    socket.close.assert_called_once()


@pytest.mark.parametrize('failure, expected', [
    (websocket.WebSocketTimeoutException('timed out'), BackendTimeoutError),
    (ConnectionRefusedError('refused'), BackendConnectionError),
    (websocket.WebSocketException('Handshake status 404 Not Found'), BackendConnectionError),
])
def test_open_websocket_wraps_connection_failures(monkeypatch, failure: Exception, expected: type):
    """A websocket that fails to open raises the package's connection or timeout error."""
    socket = MagicMock()
    socket.connect.side_effect = failure
    monkeypatch.setattr(websocket, 'WebSocket', lambda: socket)
    with pytest.raises(expected) as error:
        with ComfyUiWebservice('http://127.0.0.1:8188').open_websocket():
            pass
    assert error.value.__cause__ is failure
