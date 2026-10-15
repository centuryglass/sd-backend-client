"""Unit tests for the request bodies A1111Webservice builds inline, with `post` replaced so no server is needed."""
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from sd_backend_client.api.a1111_webservice import A1111Webservice, DEFAULT_GENERATION_TIMEOUT
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    ParameterDef, PreprocessorParams
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.webui.controlnet_webui_constants import FIRST_GENERIC_PARAMETER_KEY
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody
from sd_backend_client.api.webservice import DEFAULT_REQUEST_TIMEOUT
from sd_backend_client.errors import SDBackendError, ServerError, UnexpectedResponseError
from sd_backend_client.util.visual.image_utils import image_from_base64, image_to_base64, image_to_png_bytes


class _FakeResponse:
    """Minimal stand-in for a successful `requests.Response` carrying one image."""
    status_code = 200

    def json(self) -> dict[str, Any]:
        """Return a response body holding a single 1x1 image."""
        return {'images': [image_to_base64(Image.new('RGBA', (1, 1)))]}


def test_preprocessor_preview_sends_mask_as_opaque_grayscale(monkeypatch):
    """The preview request's mask is converted with mask_to_grayscale, like the img2img mask."""
    service = A1111Webservice('http://unused.invalid')
    sent_bodies: list[dict[str, Any]] = []

    def fake_post(_endpoint: str, body: dict[str, Any], *_args: Any, **_kwargs: Any) -> _FakeResponse:
        sent_bodies.append(body)
        return _FakeResponse()

    monkeypatch.setattr(service, 'post', fake_post)
    image = Image.new('RGBA', (2, 1), (10, 20, 30, 255))
    mask = Image.new('RGBA', (2, 1), (0, 0, 0, 0))
    mask.putpixel((0, 0), (255, 0, 0, 255))
    service.controlnet_preprocessor_preview(image, mask, ControlNetPreprocessor(name='inpaint_only+lama'))

    input_images = sent_bodies[0]['controlnet_input_images']
    assert len(input_images) == 2
    emitted_mask = image_from_base64(input_images[1])
    assert emitted_mask.getpixel((0, 0)) == (255, 255, 255, 255)
    assert emitted_mask.getpixel((1, 0)) == (0, 0, 0, 255)


def _service_with_stubbed_post(monkeypatch) -> tuple[A1111Webservice, list[tuple[str, dict[str, Any]]]]:
    """A service whose `post` records (endpoint, body) and whose dispatcher skips the server idle wait."""
    service = A1111Webservice('http://unused.invalid')
    sent: list[tuple[str, dict[str, Any]]] = []

    def fake_post(endpoint: str, body: dict[str, Any], *_args: Any, **_kwargs: Any) -> _FakeResponse:
        sent.append((endpoint, body))
        return _FakeResponse()

    monkeypatch.setattr(service, 'post', fake_post)
    service._generation_dispatcher._wait_for_idle = False  # pylint: disable=protected-access
    return service, sent


def test_submit_txt2img_reuses_body_with_distinct_task_ids(monkeypatch):
    """Submitting one body repeatedly gives each job its own task id and leaves the caller's body unchanged."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody()
    handles = [service.submit_txt2img(body) for _ in range(3)]
    for handle in handles:
        handle.wait()
    assert body.force_task_id is None
    task_ids = [handle.task_id for handle in handles]
    assert len(set(task_ids)) == 3
    assert sorted(sent_body['force_task_id'] for _, sent_body in sent) == sorted(task_ids)


def test_submit_txt2img_snapshots_body_at_submit_time(monkeypatch):
    """Changing the caller's body after submit does not change the job's request."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody(steps=7)
    handle = service.submit_txt2img(body)
    body.steps = 99
    handle.wait()
    assert sent[0][1]['steps'] == 7


def test_submit_img2img_reuse_with_different_images(monkeypatch):
    """Reusing one body for several img2img submissions sends the image each call had and leaves the body unchanged."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody()
    colors = [(255, 0, 0, 255), (0, 255, 0, 255)]
    handles = []
    for color in colors:
        body.init_images = [Image.new('RGBA', (2, 2), color)]
        handles.append(service.submit_img2img(body))
    for handle in handles:
        handle.wait()
    assert body.mask is None
    assert body.force_task_id is None
    emitted = sorted(image_from_base64(sent_body['init_images'][0]).getpixel((0, 0)) for _, sent_body in sent)
    assert emitted == sorted(colors)
    assert len({sent_body['force_task_id'] for _, sent_body in sent}) == 2


def test_img2img_does_not_modify_request_body(monkeypatch):
    """The blocking img2img leaves the caller's body without the image and mask it was called with."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    body = DiffusionRequestBody()
    service.img2img(Image.new('RGBA', (2, 2)), Image.new('RGBA', (2, 2)), body)
    assert body.init_images is None
    assert body.mask is None
    assert len(sent[0][1]['init_images']) == 1
    assert 'mask' in sent[0][1]


_IDLE_PROGRESS = {'progress': 0.0, 'eta_relative': 0.0, 'current_image': None, 'textinfo': None,
                  'state': {'skipped': False, 'interrupted': False, 'stopping_generation': False, 'job': '',
                            'job_count': 0, 'job_timestamp': '0', 'job_no': 0, 'sampling_step': 0,
                            'sampling_steps': 0}}

UPSCALERS = [{'name': name, 'model_name': None, 'model_path': None, 'model_url': None, 'scale': 4.0}
             for name in ('None', 'Lanczos', 'R-ESRGAN 4x+')]


class _RoutedResponse:
    """A `requests.Response` stand-in carrying a JSON body and a status."""

    def __init__(self, body: Any, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> Any:
        """Return the canned body."""
        return self._body


def _service_with_routes(monkeypatch, get_routes: dict[str, Any],
                         post_body: Any) -> tuple[A1111Webservice, list[tuple[str, Any]]]:
    """A service whose `get` answers from `get_routes` (an exception value is raised) and whose `post` records
    (endpoint, body) and returns `post_body`."""
    service = A1111Webservice('http://unused.invalid')
    sent: list[tuple[str, Any]] = []

    def fake_get(endpoint: str, *_args: Any, **_kwargs: Any) -> _RoutedResponse:
        body = get_routes[endpoint]
        if isinstance(body, BaseException):
            raise body
        return _RoutedResponse(body)

    def fake_post(endpoint: str, body: Any, *_args: Any, **_kwargs: Any) -> _RoutedResponse:
        sent.append((endpoint, body))
        return _RoutedResponse(post_body)

    monkeypatch.setattr(service, 'get', fake_get)
    monkeypatch.setattr(service, 'post', fake_post)
    return service, sent


def test_basic_upscale_defaults_to_first_real_upscaler(monkeypatch):
    """Without a requested upscaler, the first one that isn't 'None' is used, and the single image is decoded."""
    result_image = image_to_base64(Image.new('RGBA', (8, 4)))
    service, sent = _service_with_routes(monkeypatch, {A1111Webservice.Endpoints.UPSCALERS: UPSCALERS},
                                         {'image': result_image})
    result = service.upscale(Image.new('RGBA', (4, 2)), 8, 4)

    endpoint, body = sent[0]
    assert endpoint == A1111Webservice.Endpoints.UPSCALE
    assert (body['upscaler_1'], body['upscaling_resize_w'], body['upscaling_resize_h']) == ('Lanczos', 8, 4)
    assert result['images'][0].size == (8, 4)
    assert result['info'] is None


def test_ultimate_upscale_sends_script_args_through_img2img(monkeypatch):
    """SD upscaling with the script posts img2img with the script's positional args and the requested size."""
    service, sent = _service_with_routes(monkeypatch, {A1111Webservice.Endpoints.UPSCALERS: UPSCALERS},
                                         {'images': [image_to_base64(Image.new('RGBA', (1, 1)))]})
    params = DiffusionUpscalingParams(upscaling_mode='R-ESRGAN 4x+', use_stable_diffusion_upscaling=True,
                                      tile_width=640, tile_height=512,
                                      diffusion_params=DiffusionParams(soft_inpainting=True))
    service.upscale(Image.new('RGBA', (4, 2)), 16, 8, params)

    endpoint, body = sent[0]
    assert endpoint == A1111Webservice.Endpoints.IMG2IMG
    assert body['script_name'] == 'ultimate sd upscale'
    assert 'soft inpainting' not in body['alwayson_scripts']  # The upscale pass has no mask.
    args = body['script_args']
    assert len(args) == 18
    assert (args[1], args[2]) == (640, 512)
    assert args[8] == 2  # index of 'R-ESRGAN 4x+' in /sdapi/v1/upscalers
    assert (args[14], args[15], args[16]) == (1, 16, 8)
    assert (body['width'], body['height'], body['batch_size'], body['n_iter']) == (16, 8, 1, 1)


def test_image_response_with_invalid_info_keeps_images(monkeypatch):
    """An info string that isn't JSON leaves info None without dropping the images."""
    service, _ = _service_with_routes(monkeypatch, {}, {'images': [image_to_base64(Image.new('RGBA', (1, 1)))],
                                                        'info': 'not json'})
    result = service.txt2img(DiffusionRequestBody())
    assert len(result['images']) == 1
    assert result['info'] is None


def test_get_vae_falls_back_to_forge_endpoint(monkeypatch):
    """When /sdapi/v1/sd-vae fails, Forge's /sdapi/v1/sd-modules is used."""
    service, _ = _service_with_routes(monkeypatch, {
        A1111Webservice.Endpoints.VAE_MODELS: ServerError(404, 'Not Found', '/sdapi/v1/sd-vae', 'GET'),
        A1111Webservice.ForgeEndpoints.SD_MODULES: [{'model_name': 'ae', 'filename': '/models/VAE/ae.safetensors'}],
    }, None)
    assert [vae.model_name for vae in service.get_vae()] == ['ae']


def test_get_samplers_accepts_forge_boolean_options(monkeypatch):
    """Forge's sampler options mix strings with boolean flags."""
    service, _ = _service_with_routes(monkeypatch, {A1111Webservice.Endpoints.SAMPLERS: [
        {'name': 'DPM++ SDE', 'aliases': ['k_dpmpp_sde'],
         'options': {'scheduler': 'karras', 'second_order': True, 'brownian_noise': True}},
    ]}, None)
    [sampler] = service.get_samplers()
    assert sampler.options == {'scheduler': 'karras', 'second_order': True, 'brownian_noise': True}


def test_get_styles_accepts_forge_divider_entries(monkeypatch):
    """Forge lists section dividers as styles with null prompts."""
    service, _ = _service_with_routes(monkeypatch, {A1111Webservice.Endpoints.STYLES: [
        {'name': '---------------- STYLES ----------------', 'prompt': None, 'negative_prompt': None},
        {'name': 'cinematic', 'prompt': '{prompt}, film grain', 'negative_prompt': ''},
    ]}, None)
    styles = service.get_styles()
    assert (styles[0].prompt, styles[0].negative_prompt) == (None, None)
    assert styles[1].prompt == '{prompt}, film grain'


def test_controlnet_preprocessors_are_fetched_once_and_returned_as_copies(monkeypatch):
    """The parsed module list is cached, and callers get copies they can modify."""
    calls: list[str] = []
    service, _ = _service_with_routes(monkeypatch, {}, None)

    def fake_get(endpoint: str, *_args: Any, **_kwargs: Any) -> _RoutedResponse:
        calls.append(endpoint)
        return _RoutedResponse({'module_list': ['none', 'canny']})

    monkeypatch.setattr(service, 'get', fake_get)
    first = service.get_controlnet_preprocessors()
    first[0].name = 'changed'
    second = service.get_controlnet_preprocessors()
    assert [preprocessor.name for preprocessor in second] == ['none', 'canny']
    assert calls == [A1111Webservice.Endpoints.CONTROLNET_MODULES]


@pytest.mark.parametrize('response, caption', [({'caption': 'a cat'}, 'a cat'), ('a dog', 'a dog')])
def test_interrogate_accepts_object_or_string_response(monkeypatch, response: Any, caption: str):
    """Interrogation captions are read from either response form."""
    service, sent = _service_with_routes(monkeypatch, {}, response)
    assert service.interrogate(Image.new('RGBA', (1, 1))) == caption
    assert sent[0][1]['model'] == 'clip'


def _service_with_mock_session(**kwargs: Any) -> tuple[A1111Webservice, MagicMock]:
    """A service whose `requests.Session` is a mock answering every request with a one-image response."""
    service = A1111Webservice('http://unused.invalid/', **kwargs)
    session = MagicMock()
    session.get.return_value = session.post.return_value = MagicMock(status_code=200, ok=True,
                                                                     json=_FakeResponse().json)
    service._session = session  # pylint: disable=protected-access
    return service, session


def test_generation_requests_use_generation_timeout():
    """Requests that block until images are ready use generation_timeout, other requests use request_timeout."""
    service, session = _service_with_mock_session(request_timeout=5, generation_timeout=900)
    image = Image.new('RGBA', (1, 1))
    service.txt2img(DiffusionRequestBody())
    service.img2img(image, None, DiffusionRequestBody())
    service.controlnet_preprocessor_preview(image, None, ControlNetPreprocessor(name='canny'))
    service.interrupt()
    assert [call.kwargs['timeout'] for call in session.post.call_args_list] == [900, 900, 900, 5]
    assert [call.args[0] for call in session.post.call_args_list] == [
        'http://unused.invalid/sdapi/v1/txt2img', 'http://unused.invalid/sdapi/v1/img2img',
        'http://unused.invalid/controlnet/detect', 'http://unused.invalid/sdapi/v1/interrupt']


def test_default_timeouts_are_bounded():
    """Without arguments neither kind of request waits indefinitely."""
    service, session = _service_with_mock_session()
    service.txt2img(DiffusionRequestBody())
    session.get.return_value.json = lambda: []
    service.get_styles()
    assert session.post.call_args.kwargs['timeout'] == DEFAULT_GENERATION_TIMEOUT
    assert session.get.call_args.kwargs['timeout'] == DEFAULT_REQUEST_TIMEOUT


def test_get_thumbnail_sends_file_path_as_query_param():
    """The thumbnail path goes to requests as a parameter, so reserved characters in it are encoded."""
    service, session = _service_with_mock_session()
    session.get.return_value.content = image_to_png_bytes(Image.new('RGBA', (1, 1)))
    assert service.get_thumbnail('models/Lora/a&b #1.png') is not None
    assert session.get.call_args.args[0] == 'http://unused.invalid/sd_extra_networks/thumb'
    assert session.get.call_args.kwargs['params'] == {'filename': 'models/Lora/a&b #1.png'}


class _HtmlResponse:
    """A successful response whose body is an HTML page rather than JSON."""
    status_code = 200
    url = 'http://unused.invalid/sdapi/v1/txt2img'
    text = '<html>proxy error</html>'

    def json(self) -> Any:
        """Fail the way `requests` does on a non-JSON body."""
        raise ValueError('Expecting value: line 1 column 1 (char 0)')


def test_non_json_image_response_raises_unexpected_response(monkeypatch):
    """A generation response that is not JSON raises UnexpectedResponseError quoting the body."""
    service = A1111Webservice('http://unused.invalid')
    monkeypatch.setattr(service, 'post', lambda *_args, **_kwargs: _HtmlResponse())
    with pytest.raises(UnexpectedResponseError, match='proxy error') as error:
        service.txt2img(DiffusionRequestBody())
    assert isinstance(error.value, SDBackendError)


def test_interrogate_with_unexpected_response_raises(monkeypatch):
    """An interrogate response that is neither a caption string nor an object raises UnexpectedResponseError."""
    service, _ = _service_with_routes(monkeypatch, {}, ['not', 'a', 'caption'])
    with pytest.raises(UnexpectedResponseError, match='interrogate'):
        service.interrogate(Image.new('RGBA', (1, 1)))


def test_preprocessor_preview_without_images_raises(monkeypatch):
    """A /controlnet/detect response with no images raises UnexpectedResponseError."""
    service, _ = _service_with_routes(monkeypatch, {}, {'images': []})
    preprocessor = ControlNetPreprocessor(name='canny')
    with pytest.raises(UnexpectedResponseError, match='no preview image'):
        service.controlnet_preprocessor_preview(Image.new('RGBA', (1, 1)), None, preprocessor)


def test_txt2img_sends_controlnet_units_of_plain_params_as_alwayson_script(monkeypatch):
    """Blocking txt2img converts a plain DiffusionParams, so its ControlNet units go into alwayson_scripts."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    unit = ControlNetUnit(model=ControlNetModel('control_canny.safetensors'))
    service.txt2img(DiffusionParams(prompt='x', controlnet_units=[unit]))

    body = sent[0][1]
    assert 'controlnet_units' not in body
    [unit_args] = body['alwayson_scripts']['controlNet']['args']
    assert unit_args['model'] == 'control_canny.safetensors'


def test_submit_txt2img_sends_no_source_image_or_mask(monkeypatch):
    """submit_txt2img leaves out init images and the mask, which only image-to-image requests use."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    params = DiffusionParams(init_images=[Image.new('RGBA', (2, 2))], mask=Image.new('L', (2, 2)))
    service.submit_txt2img(params).wait()
    assert 'init_images' not in sent[0][1]
    assert 'mask' not in sent[0][1]
    assert params.init_images is not None and params.mask is not None


def test_txt2img_sends_no_soft_inpainting(monkeypatch):
    """Both txt2img paths leave out the Soft Inpainting script, which only runs on img2img."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    params = DiffusionRequestBody(soft_inpainting=True)
    service.submit_txt2img(params).wait()
    service.txt2img(params)
    assert [('soft inpainting' in body['alwayson_scripts']) for _, body in sent] == [False, False]
    assert params.soft_inpainting


def test_submit_upscale_uses_image_and_params_as_they_were_at_submit_time(monkeypatch):
    """Changing the image or upscale params after submit_upscale changes neither the request nor the result."""
    service, sent = _service_with_routes(monkeypatch, {A1111Webservice.Endpoints.UPSCALERS: UPSCALERS,
                                                       A1111Webservice.Endpoints.PROGRESS: _IDLE_PROGRESS},
                                         {'image': image_to_base64(Image.new('RGBA', (8, 4)))})
    image = Image.new('RGBA', (4, 2), (10, 20, 30, 255))
    params = DiffusionUpscalingParams(upscaling_mode='R-ESRGAN 4x+')
    handle = service.submit_upscale(image, 8, 4, params)
    image.putpixel((0, 0), (0, 0, 0, 0))
    params.upscaling_mode = 'Lanczos'
    result = handle.wait(timeout=5.0, poll_interval=0.01)

    endpoint, body = sent[0]
    assert endpoint == A1111Webservice.Endpoints.UPSCALE
    assert body['upscaler_1'] == 'R-ESRGAN 4x+'
    assert image_from_base64(body['image']).getpixel((0, 0)) == (10, 20, 30, 255)
    assert result.images[0].size == (8, 4)


@pytest.mark.parametrize('params, interruptible', [
    (None, False),
    (DiffusionUpscalingParams(use_stable_diffusion_upscaling=False), False),
    (DiffusionUpscalingParams(use_stable_diffusion_upscaling=True), True),
])
def test_submit_upscale_is_interruptible_only_for_diffusion_upscaling(params: DiffusionUpscalingParams | None,
                                                                      interruptible: bool):
    """Only a Stable Diffusion upscale runs through img2img, which the server's interrupt can stop."""
    service = A1111Webservice('http://unused.invalid')
    dispatcher = MagicMock()
    service._dispatcher = dispatcher  # pylint: disable=protected-access
    service.submit_upscale(Image.new('RGBA', (4, 4)), 8, 8, params)
    assert dispatcher.submit.call_args.kwargs['interruptible'] is interruptible


def test_submit_preprocessor_preview_sends_overrides_and_mask_and_cannot_be_interrupted(monkeypatch):
    """The preview job posts to /controlnet/detect with the image, mask and parameter overrides, uninterruptibly."""
    service, sent = _service_with_stubbed_post(monkeypatch)
    submit = MagicMock(wraps=service._generation_dispatcher.submit)  # pylint: disable=protected-access
    monkeypatch.setattr(service._generation_dispatcher, 'submit', submit)  # pylint: disable=protected-access
    preprocessor = PreprocessorParams(
        typedef=ControlNetPreprocessor(name='canny', parameters=[
            ParameterDef(key=FIRST_GENERIC_PARAMETER_KEY, default_value=100)]),
        parameter_values={FIRST_GENERIC_PARAMETER_KEY: 50})
    result = service.submit_preprocessor_preview(Image.new('RGBA', (2, 2)), preprocessor,
                                                 Image.new('L', (2, 2))).wait(timeout=5.0, poll_interval=0.01)

    endpoint, body = sent[0]
    assert endpoint == A1111Webservice.Endpoints.CONTROLNET_PREVIEW
    assert body['controlnet_module'] == 'canny'
    assert body['controlnet_threshold_a'] == 50
    assert len(body['controlnet_input_images']) == 2
    assert len(result.images) == 1 and result.raw_info is None
    assert submit.call_args.kwargs['interruptible'] is False
