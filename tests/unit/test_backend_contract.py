"""Contract test for `Backend`: both clients run the same submit-and-wait lifecycle through the shared interface.

Each client gets a `FakeSession` that answers the endpoints one generation touches, so the tests run
offline. Every test is parametrized over both backends; a test that needs a backend-specific branch is a gap in the
interface.
"""
# pylint: disable=protected-access
from typing import Any, Callable

import pytest
from PIL import Image

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui_webservice import ComfyEndpoints, ComfyUiWebservice
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.backend import Backend
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    ParameterDef, PreprocessorParams
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.generation_handle import GenerationHandle, GenerationStatus
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody
from sd_backend_client.util.visual.image_utils import image_to_base64, image_to_png_bytes

from .fake_session import FakeSession
from .soft_inpainting_responses import COMFYUI_DIFFERENTIAL_DIFFUSION_INFO, WEBUI_SOFT_INPAINTING_INFO
from .webui_responses import generation_info, generation_response

PROMPT_ID = '7d4c0f0e-6a3b-4c1e-9f6e-0a1b2c3d4e5f'
RESULT_PIXEL = (12, 34, 56, 255)
DETECT_MAP_PIXEL = (255, 255, 255, 255)
SEED = 1234
UPSCALER_NAME = 'upscaler.pth'
WAIT_TIMEOUT_S = 5.0
DIFFERENTIAL_DIFFUSION_INFO = f'{ComfyEndpoints.OBJECT_INFO}/DifferentialDiffusion'


def _result_image() -> Image.Image:
    return Image.new('RGBA', (8, 8), RESULT_PIXEL)


class _Harness:
    """A client on a fake session, plus how to read the prompt and image count out of what it sent."""

    def __init__(self, service: Backend, session: FakeSession,
                 generation_requests: Callable[[FakeSession], list[dict[str, Any]]],
                 sent_prompts: Callable[[dict[str, Any]], list[str]],
                 sent_image_count: Callable[[FakeSession], int],
                 soft_inpainting_routes: list[tuple[str, str]]) -> None:
        self.service = service
        self.session = session
        self.generation_requests = lambda: generation_requests(session)
        self.sent_prompts = sent_prompts
        self.sent_image_count = lambda: sent_image_count(session)
        self.soft_inpainting_routes = soft_inpainting_routes

    def remove_soft_inpainting(self) -> None:
        """Make the fake server one without soft inpainting."""
        for route in self.soft_inpainting_routes:
            self.session.routes[route] = {'txt2img': [], 'img2img': []} if route[1].endswith('/scripts') else {}


def _webui() -> _Harness:
    service = A1111Webservice('http://webui.invalid')
    image_response = {'images': [image_to_base64(_result_image())]}
    # The WebUI's ControlNet extension appends its detect map after the generated image.
    generation = generation_response([image_to_base64(_result_image()),
                                      image_to_base64(Image.new('RGBA', (8, 8), DETECT_MAP_PIXEL))],
                                     generation_info([SEED]))
    session = FakeSession({
        ('POST', A1111Webservice.Endpoints.TXT2IMG): generation,
        ('POST', A1111Webservice.Endpoints.IMG2IMG): generation,
        ('POST', A1111Webservice.Endpoints.UPSCALE): {'image': image_to_base64(_result_image())},
        ('POST', A1111Webservice.Endpoints.CONTROLNET_PREVIEW): image_response,
        ('GET', A1111Webservice.Endpoints.UPSCALERS): [
            {'name': name, 'model_name': None, 'model_path': None, 'model_url': None, 'scale': 4.0}
            for name in ('None', UPSCALER_NAME)],
        ('GET', A1111Webservice.Endpoints.SCRIPTS): {'txt2img': [], 'img2img': ['soft inpainting']},
        ('GET', A1111Webservice.Endpoints.SCRIPT_INFO): [WEBUI_SOFT_INPAINTING_INFO],
        ('GET', A1111Webservice.Endpoints.PROGRESS): {
            'progress': 0.5, 'eta_relative': 1.0, 'current_image': None, 'textinfo': None,
            'state': {'skipped': False, 'interrupted': False, 'stopping_generation': False, 'job': '',
                      'job_count': 0, 'job_timestamp': '0', 'job_no': 0, 'sampling_step': 0,
                      'sampling_steps': 0}},
    })
    service._session = session  # type: ignore[assignment]

    def generation_requests(fake: FakeSession) -> list[dict[str, Any]]:
        return fake.of('POST', A1111Webservice.Endpoints.TXT2IMG) + fake.of('POST', A1111Webservice.Endpoints.IMG2IMG)

    def sent_image_count(fake: FakeSession) -> int:
        body = generation_requests(fake)[0]['json']
        return len(body.get('init_images', [])) + ('mask' in body)

    return _Harness(service, session, generation_requests, lambda request: [request['json']['prompt']],
                    sent_image_count, [('GET', A1111Webservice.Endpoints.SCRIPTS)])


def _comfyui() -> _Harness:
    service = ComfyUiWebservice('http://comfyui.invalid', live_progress=False)
    upload = {'name': 'uploaded.png', 'subfolder': 'sd_backend_client', 'type': 'input'}
    history = {PROMPT_ID: {
        'prompt': [1, PROMPT_ID, {}, {}, ['9']],
        'outputs': {'9': {'images': [{'filename': 'out.png', 'subfolder': '', 'type': 'output'}]}},
        'status': {'status_str': 'success', 'completed': True, 'messages': []},
    }}
    session = FakeSession({
        ('GET', '/models/configs'): [],
        ('GET', f'{ComfyEndpoints.MODELS}/upscale_models'): [UPSCALER_NAME],
        ('POST', ComfyEndpoints.IMG_UPLOAD): upload,
        ('POST', ComfyEndpoints.MASK_UPLOAD): upload,
        ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1, 'node_errors': {}},
        ('GET', f'{ComfyEndpoints.HISTORY}/{PROMPT_ID}'): history,
        ('GET', ComfyEndpoints.VIEW_IMAGE): image_to_png_bytes(_result_image()),
        ('GET', DIFFERENTIAL_DIFFUSION_INFO): COMFYUI_DIFFERENTIAL_DIFFUSION_INFO,
    })
    service._session = session  # type: ignore[assignment]

    def sent_prompts(request: dict[str, Any]) -> list[str]:
        nodes = request['json']['prompt'].values()
        return [node['inputs']['text'] for node in nodes if node['class_type'] == 'CLIPTextEncode']

    def sent_image_count(fake: FakeSession) -> int:
        return len(fake.of('POST', ComfyEndpoints.IMG_UPLOAD)) + len(fake.of('POST', ComfyEndpoints.MASK_UPLOAD))

    return _Harness(service, session, lambda fake: fake.of('POST', ComfyEndpoints.PROMPT), sent_prompts,
                    sent_image_count, [('GET', DIFFERENTIAL_DIFFUSION_INFO)])


BACKENDS = {'webui': _webui, 'comfyui': _comfyui}


@pytest.fixture(name='harness', params=sorted(BACKENDS))
def _harness_fixture(request) -> _Harness:
    """Each backend's client on a fake session."""
    return BACKENDS[request.param]()


def _params(params_type: type[DiffusionParams] = DiffusionParams, init_image: bool = False, mask: bool = False,
            **fields: Any) -> DiffusionParams:
    return params_type(prompt='a red apple', sd_model_name='model.safetensors', seed=SEED,
                       init_images=[Image.new('RGBA', (8, 8), (200, 0, 0, 255))] if init_image else None,
                       mask=Image.new('L', (8, 8), 255) if mask else None, **fields)


OPERATIONS: dict[str, tuple[Callable[[Backend, DiffusionParams], GenerationHandle], int]] = {
    'txt2img': (lambda service, params: service.submit_txt2img(params), 0),
    'img2img': (lambda service, params: service.submit_img2img(params), 1),
    'inpaint': (lambda service, params: service.submit_inpaint(params), 2),
}


def test_both_clients_implement_backend(harness: _Harness):
    """Each client is a `Backend`."""
    assert isinstance(harness.service, Backend)


@pytest.mark.parametrize('operation', sorted(OPERATIONS))
@pytest.mark.parametrize('params_type', [DiffusionParams, DiffusionRequestBody, ComfyUIDiffusionParams])
def test_submit_then_wait_returns_the_generated_images(harness: _Harness, operation: str,
                                                       params_type: type[DiffusionParams]):
    """Every submit method accepts any `DiffusionParams` subclass and yields a handle that finishes with the images.

    txt2img sends no source image, img2img sends one and inpainting sends the image and its mask.
    """
    submit, expected_image_count = OPERATIONS[operation]
    params = _params(params_type, init_image=True, mask=operation == 'inpaint')

    handle = submit(harness.service, params)
    statuses: list[GenerationStatus] = []
    result = handle.wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01,
                         on_progress=lambda progress: statuses.append(progress.status))

    assert isinstance(handle, GenerationHandle)
    assert handle.task_id is not None and result.task_id == handle.task_id
    assert statuses[-1] is GenerationStatus.FINISHED
    assert [image.getpixel((0, 0)) for image in result.images] == [RESULT_PIXEL]
    assert result.seeds == [SEED] and result.seed == SEED
    assert result.raw_info is not None
    [request] = harness.generation_requests()
    assert 'a red apple' in harness.sent_prompts(request)
    assert harness.sent_image_count() == expected_image_count


@pytest.mark.parametrize('operation', sorted(OPERATIONS))
def test_job_uses_params_as_they_were_at_submit_time(harness: _Harness, operation: str):
    """Changing the caller's params after submit changes neither the job nor leaves job state on the params."""
    submit, _ = OPERATIONS[operation]
    params = _params(init_image=True, mask=True)
    original = params.model_copy()

    handle = submit(harness.service, params)
    params.prompt = 'changed after submit'
    handle.wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01)

    [request] = harness.generation_requests()
    assert 'a red apple' in harness.sent_prompts(request)
    assert 'changed after submit' not in harness.sent_prompts(request)
    params.prompt = original.prompt
    assert params == original


@pytest.mark.parametrize('operation, params, message', [
    ('img2img', _params(), 'init image'),
    ('inpaint', _params(mask=True), 'init image'),
    ('inpaint', _params(init_image=True), 'mask'),
])
def test_missing_inputs_raise_before_contacting_the_server(harness: _Harness, operation: str,
                                                           params: DiffusionParams, message: str):
    """A missing init image or mask raises ValueError from the submit call, without any request."""
    submit, _ = OPERATIONS[operation]
    with pytest.raises(ValueError, match=message):
        submit(harness.service, params)
    assert not harness.session.requests


@pytest.mark.parametrize('operation', ['img2img', 'inpaint'])
def test_soft_inpainting_submits_with_the_server_defaults(harness: _Harness, operation: str):
    """An inpainting or masked img2img job with soft inpainting on and every setting at its default finishes."""
    submit, _ = OPERATIONS[operation]
    defaults = {parameter.key: parameter.default_value
                for parameter in harness.service.get_soft_inpainting_parameters()}
    params = _params(init_image=True, mask=True, soft_inpainting=True, soft_inpainting_params=defaults)
    result = submit(harness.service, params).wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01)
    assert [image.getpixel((0, 0)) for image in result.images] == [RESULT_PIXEL]


def test_txt2img_ignores_soft_inpainting(harness: _Harness):
    """Text-to-image runs with soft inpainting on, though it has no mask."""
    harness.service.submit_txt2img(_params(soft_inpainting=True)).wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01)
    assert len(harness.generation_requests()) == 1


def test_soft_inpainting_without_a_mask_raises_before_contacting_the_server(harness: _Harness):
    """Both backends silently ignore soft inpainting without a mask, so the client raises instead."""
    with pytest.raises(ValueError, match='mask'):
        harness.service.submit_img2img(_params(init_image=True, soft_inpainting=True))
    assert not harness.session.requests


@pytest.mark.parametrize('operation', ['img2img', 'inpaint'])
def test_soft_inpainting_on_a_server_without_it_raises(harness: _Harness, operation: str):
    """Soft inpainting on a server that lacks it raises ValueError naming the capability, and queues nothing."""
    harness.remove_soft_inpainting()
    submit, _ = OPERATIONS[operation]
    with pytest.raises(ValueError, match='BackendCapabilities.soft_inpainting'):
        submit(harness.service, _params(init_image=True, mask=True, soft_inpainting=True))
    assert not harness.generation_requests()


def test_soft_inpainting_settings_are_validated_against_the_server(harness: _Harness):
    """An unknown setting or a value above the server's maximum raises ValueError, and queues nothing."""
    parameter = harness.service.get_soft_inpainting_parameters()[0]
    assert parameter.max_val is not None
    for values in ({'no_such_setting': 1.0}, {parameter.key: parameter.max_val + 1}):
        with pytest.raises(ValueError, match='Soft inpainting'):
            harness.service.submit_inpaint(_params(init_image=True, mask=True, soft_inpainting=True,
                                                   soft_inpainting_params=values))
    assert not harness.generation_requests()


def test_soft_inpainting_settings_are_ignored_while_it_is_off(harness: _Harness):
    """Settings left in `soft_inpainting_params` are not validated or sent while soft inpainting is off."""
    params = _params(init_image=True, mask=True, soft_inpainting_params={'no_such_setting': 1.0})
    harness.service.submit_inpaint(params).wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01)
    assert len(harness.generation_requests()) == 1


PREPROCESSOR = ControlNetPreprocessor(name='canny', parameters=[ParameterDef(key='low_threshold', default_value=100)])


@pytest.mark.parametrize('upscale_params', [None, DiffusionUpscalingParams(upscaling_mode=UPSCALER_NAME)])
def test_submit_upscale_then_wait_returns_the_upscaled_image(harness: _Harness,
                                                             upscale_params: DiffusionUpscalingParams | None):
    """`submit_upscale` yields a handle that finishes with the one upscaled image."""
    handle = harness.service.submit_upscale(Image.new('RGBA', (8, 8)), 16, 16, upscale_params)
    result = handle.wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01)

    assert isinstance(handle, GenerationHandle)
    assert handle.task_id is not None and result.task_id == handle.task_id
    assert [image.getpixel((0, 0)) for image in result.images] == [RESULT_PIXEL]
    assert not result.seeds and not result.control_maps


@pytest.mark.parametrize('size', [(8, 8), (4, 8), (0, 16)])
def test_upscale_to_a_size_no_larger_raises_before_contacting_the_server(harness: _Harness, size: tuple[int, int]):
    """A requested size that does not exceed the source in some dimension raises ValueError, without any request."""
    with pytest.raises(ValueError, match='must exceed the source size'):
        harness.service.submit_upscale(Image.new('RGBA', (8, 8)), *size)
    assert not harness.session.requests


@pytest.mark.parametrize('preprocessor', [PREPROCESSOR,
                                          PreprocessorParams(typedef=PREPROCESSOR,
                                                             parameter_values={'low_threshold': 50})])
@pytest.mark.parametrize('mask', [False, True])
def test_submit_preprocessor_preview_then_wait_returns_the_control_image(
        harness: _Harness, preprocessor: ControlNetPreprocessor | PreprocessorParams, mask: bool):
    """`submit_preprocessor_preview` takes either preprocessor form, with or without a mask, and yields one image."""
    handle = harness.service.submit_preprocessor_preview(Image.new('RGBA', (8, 8)), preprocessor,
                                                         Image.new('L', (8, 8), 255) if mask else None)
    result = handle.wait(timeout=WAIT_TIMEOUT_S, poll_interval=0.01)

    assert isinstance(handle, GenerationHandle)
    assert handle.task_id is not None and result.task_id == handle.task_id
    assert [image.getpixel((0, 0)) for image in result.images] == [RESULT_PIXEL]
    assert not result.seeds and not result.control_maps
