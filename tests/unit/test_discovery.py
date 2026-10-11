"""Offline tests for `Backend`'s discovery methods and `connect_to_backend`.

Each backend's fake server holds the same models and options in its own wire format, so the shared listings can be
checked against one expectation where the backends agree, and each name can be checked to round-trip into a request.
"""
# pylint: disable=protected-access
from typing import Any, Callable

import pytest
from PIL import Image

from sd_backend_client.api import detect
from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui.nodes.controlnet.apply_controlnet_node import ApplyControlNetNode
from sd_backend_client.api.comfyui.nodes.ultimate_upscale_node import ULTIMATE_UPSCALE_NODE_NAME
from sd_backend_client.api.comfyui_webservice import ComfyEndpoints, ComfyUiWebservice
from sd_backend_client.api.detect import connect_to_backend
from sd_backend_client.api.shared_data.backend import Backend
from sd_backend_client.api.shared_data.backend_options import BackendCapabilities, BackendOption
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.errors import AuthError, ServerError, UnexpectedResponseError
from sd_backend_client.util.visual.image_utils import image_to_base64

from .fake_session import CannedStatus, FakeSession

CHECKPOINT = 'sd15/model.safetensors'
NOT_FOUND = CannedStatus(404, '{"detail": "Not Found"}')
PROMPT_ID = '7d4c0f0e-6a3b-4c1e-9f6e-0a1b2c3d4e5f'


def _webui_routes(full: bool) -> dict[tuple[str, str], Any]:
    """WebUI routes; `full` adds ControlNet, Ultimate SD Upscale and the scheduler list (A1111 1.9+)."""
    endpoints = A1111Webservice.Endpoints
    routes: dict[tuple[str, str], Any] = {
        ('GET', endpoints.SD_MODELS): [{'title': f'{CHECKPOINT} [abc1234567]', 'model_name': 'sd15_model',
                                         'hash': 'abc1234567', 'sha256': None, 'filename': f'/models/{CHECKPOINT}',
                                         'config': None}],
        ('GET', endpoints.VAE_MODELS): [{'model_name': 'vae.safetensors', 'filename': '/models/vae.safetensors'}],
        ('GET', endpoints.LORA_MODELS): [{'name': 'detail', 'alias': 'detail', 'path': '/l/detail.safetensors'},
                                         {'name': 'style_v2', 'alias': 'style', 'path': '/l/style_v2.safetensors'}],
        ('GET', endpoints.HYPERNETWORKS): [{'name': 'hn', 'path': '/h/hn.pt'}],
        ('GET', endpoints.SAMPLERS): [{'name': name, 'aliases': [], 'options': {}}
                                      for name in ('Euler a', 'DPM++ 2M', 'Restart')],
        ('GET', endpoints.UPSCALERS): [{'name': name, 'model_name': None, 'model_path': None, 'model_url': None,
                                        'scale': 4.0} for name in ('None', 'R-ESRGAN 4x+')],
        ('GET', endpoints.SCRIPTS): {'txt2img': [], 'img2img': ['loopback']},
        ('GET', endpoints.SCHEDULERS): NOT_FOUND,
        ('GET', endpoints.CONTROLNET_VERSION): NOT_FOUND,
        ('GET', endpoints.CONTROLNET_MODELS): NOT_FOUND,
        ('GET', endpoints.INTERROGATE): CannedStatus(405, '{"detail": "Method Not Allowed"}'),
        ('POST', endpoints.TXT2IMG): {'images': [image_to_base64(Image.new('RGBA', (8, 8)))]},
    }
    if full:
        routes.update({
            ('GET', endpoints.SCRIPTS): {'txt2img': [], 'img2img': ['loopback', 'ultimate sd upscale']},
            ('GET', endpoints.SCHEDULERS): [{'name': 'karras', 'label': 'Karras'}, {'name': 'ddim', 'label': 'DDIM'}],
            ('GET', endpoints.CONTROLNET_VERSION): {'version': 3},
            ('GET', endpoints.CONTROLNET_MODELS): {'model_list': ['control_v11p_sd15_canny [d14c016b]']},
        })
    return routes


def _comfyui_routes(full: bool) -> dict[tuple[str, str], Any]:
    """ComfyUI routes; `full` adds the Ultimate SD Upscale node. ControlNet nodes are part of ComfyUI itself."""
    models = ComfyEndpoints.MODELS
    info = ComfyEndpoints.OBJECT_INFO
    return {
        ('GET', f'{models}/checkpoints'): [CHECKPOINT],
        ('GET', f'{models}/configs'): [],
        ('GET', f'{models}/vae'): ['vae.safetensors'],
        ('GET', f'{models}/loras'): ['detail.safetensors', 'style_v2.safetensors'],
        ('GET', f'{models}/hypernetworks'): ['hn.pt'],
        ('GET', f'{models}/upscale_models'): ['4x.pth'],
        ('GET', f'{models}/controlnet'): ['control_v11p_sd15_canny.pth'],
        ('GET', f'{info}/KSampler'): {'KSampler': {
            'input': {'required': {'sampler_name': [['euler_ancestral', 'dpmpp_2m', 'res_multistep']],
                                   'scheduler': [['karras', 'ddim_uniform']]}},
            'input_order': {'required': ['sampler_name', 'scheduler']},
            'output': ['LATENT'], 'output_is_list': [False], 'output_name': ['LATENT'], 'name': 'KSampler',
            'display_name': 'KSampler', 'description': '', 'python_module': 'nodes', 'category': 'sampling',
            'output_node': False}},
        ('GET', f'{info}/{ApplyControlNetNode.CLASS_TYPE}'): {ApplyControlNetNode.CLASS_TYPE: {}},
        ('GET', f'{info}/{ULTIMATE_UPSCALE_NODE_NAME}'): {ULTIMATE_UPSCALE_NODE_NAME: {}} if full else {},
        ('POST', ComfyEndpoints.PROMPT): {'prompt_id': PROMPT_ID, 'number': 1, 'node_errors': {}},
    }


def _webui(full: bool = True) -> tuple[A1111Webservice, FakeSession]:
    service = A1111Webservice('http://webui.invalid')
    session = FakeSession(_webui_routes(full))
    service._session = session  # type: ignore[assignment]
    return service, session


def _comfyui(full: bool = True) -> tuple[ComfyUiWebservice, FakeSession]:
    service = ComfyUiWebservice('http://comfyui.invalid', live_progress=False)
    session = FakeSession(_comfyui_routes(full))
    service._session = session  # type: ignore[assignment]
    return service, session


BACKENDS: dict[str, Callable[..., tuple[Backend, FakeSession]]] = {'webui': _webui, 'comfyui': _comfyui}


@pytest.fixture(name='backend', params=sorted(BACKENDS))
def _backend_fixture(request) -> Backend:
    return BACKENDS[request.param]()[0]


def _names(options: list[BackendOption]) -> list[str]:
    return [option.name for option in options]


# Shared listings: one expectation on both backends.

def test_checkpoints_list_the_relative_file_name(backend: Backend):
    """A checkpoint is named by its file name relative to the models directory on both backends."""
    assert _names(backend.list_checkpoints()) == [CHECKPOINT]


def test_vaes_list_the_file_name(backend: Backend):
    """VAEs are named by file name on both backends."""
    assert backend.list_vaes() == [BackendOption(name='vae.safetensors')]


def test_samplers_use_shared_names_with_webui_labels(backend: Backend):
    """Samplers in `SAMPLER_WEBUI_NAMES` list under their shared name, labelled with the WebUI name."""
    samplers = backend.list_samplers()
    assert samplers[:2] == [BackendOption(name='euler_ancestral', display_name='Euler a'),
                            BackendOption(name='dpmpp_2m', display_name='DPM++ 2M')]
    assert len(samplers) == 3


def test_schedulers_use_shared_names(backend: Backend):
    """WebUI's scheduler names map to the shared names ComfyUI uses."""
    assert _names(backend.list_schedulers()) == ['karras', 'ddim_uniform']


def test_controlnet_models_are_controlnet_model_instances(backend: Backend):
    """ControlNet models list as `ControlNetModel`, ready for `ControlNetUnit.model`."""
    [model] = backend.list_controlnet_models()
    assert isinstance(model, ControlNetModel)
    assert model.full_model_name.startswith('control_v11p_sd15_canny')


def test_comfyui_model_configs_list_the_values_sd_model_config_accepts():
    """`list_model_configs` lists the config files ComfyUI reports, by file name."""
    service, session = _comfyui()
    session.routes[('GET', f'{ComfyEndpoints.MODELS}/configs')] = ['v1-inference.yaml', 'v2-inference.yaml']
    assert service.list_model_configs() == [BackendOption(name='v1-inference.yaml'),
                                            BackendOption(name='v2-inference.yaml')]


def test_full_server_reports_every_optional_generation_feature(backend: Backend):
    """A server with ControlNet, Ultimate SD Upscale and schedulers reports all three."""
    capabilities = backend.get_capabilities()
    assert capabilities.controlnet and capabilities.ultimate_upscale and capabilities.scheduler


def test_checkpoint_name_round_trips_into_a_generation_request():
    """A listed checkpoint name, set as `sd_model_name`, selects that checkpoint in each backend's request."""
    webui, webui_session = _webui()
    webui.txt2img(DiffusionParams(sd_model_name=webui.list_checkpoints()[0].name))
    [request] = webui_session.of('POST', A1111Webservice.Endpoints.TXT2IMG)
    assert request['json']['override_settings']['sd_model_checkpoint'] == CHECKPOINT

    comfyui, comfyui_session = _comfyui()
    comfyui.txt2img(DiffusionParams(sd_model_name=comfyui.list_checkpoints()[0].name))
    [request] = comfyui_session.of('POST', ComfyEndpoints.PROMPT)
    loaders = [node for node in request['json']['prompt'].values()
               if node['class_type'] == 'CheckpointLoaderSimple']
    assert [loader['inputs']['ckpt_name'] for loader in loaders] == [CHECKPOINT]


def test_lora_name_round_trips_into_a_comfyui_prompt_tag():
    """A listed ComfyUI LoRA name in a `<lora:name:weight>` tag becomes a LoRA loader for that file."""
    service, session = _comfyui()
    name = service.list_loras()[1].name
    service.txt2img(DiffusionParams(sd_model_name=CHECKPOINT, prompt=f'a cat <lora:{name}:0.5>'))
    [request] = session.of('POST', ComfyEndpoints.PROMPT)
    loaders = [node for node in request['json']['prompt'].values() if node['class_type'] == 'LoraLoader']
    assert [loader['inputs']['lora_name'] for loader in loaders] == [name]


# Backend-specific details.

def test_webui_checkpoint_display_name_is_the_title():
    """The WebUI title, hash suffix included, is the display name."""
    service, _ = _webui()
    assert service.list_checkpoints()[0].display_name == f'{CHECKPOINT} [abc1234567]'


def test_webui_loras_show_an_alias_only_when_it_differs():
    """A LoRA's alias is its display name unless it repeats the name."""
    service, _ = _webui()
    assert service.list_loras() == [BackendOption(name='detail'), BackendOption(name='style_v2', display_name='style')]


def test_webui_upscalers_leave_out_the_none_upscaler():
    """WebUI's no-op 'None' upscaler is not listed."""
    service, _ = _webui()
    assert _names(service.list_upscalers()) == ['R-ESRGAN 4x+']


def test_comfyui_lists_upscale_models_and_hypernetworks_by_file_name():
    """ComfyUI options are model file names, with no display name."""
    service, _ = _comfyui()
    assert service.list_upscalers() == [BackendOption(name='4x.pth')]
    assert service.list_hypernetworks() == [BackendOption(name='hn.pt')]


def test_webui_lists_hypernetworks_by_name():
    """WebUI hypernetworks are listed by the name its prompt tags use."""
    service, _ = _webui()
    assert service.list_hypernetworks() == [BackendOption(name='hn')]


def test_webui_without_a_hypernetwork_endpoint_lists_none():
    """Forge Neo has no /sdapi/v1/hypernetworks, so it lists no hypernetworks."""
    service, session = _webui()
    session.routes[('GET', A1111Webservice.Endpoints.HYPERNETWORKS)] = NOT_FOUND
    assert not service.list_hypernetworks()


def test_webui_without_an_interrogate_endpoint_reports_it_missing():
    """Forge Neo has no /sdapi/v1/interrogate, which a GET probe sees as a 404 rather than a 405."""
    service, session = _webui()
    session.routes[('GET', A1111Webservice.Endpoints.INTERROGATE)] = NOT_FOUND
    assert not service.get_capabilities().interrogate


def test_webui_without_extensions_lists_nothing_and_reports_missing_features():
    """A WebUI without ControlNet, Ultimate SD Upscale or a scheduler list reports each as missing."""
    service, _ = _webui(full=False)
    assert not service.list_controlnet_models()
    assert not service.list_schedulers()
    assert service.get_capabilities() == BackendCapabilities(controlnet=False, ultimate_upscale=False,
                                                             scheduler=False, interrogate=True, free_memory=False)


def test_webui_reports_forge_builtin_controlnet_without_a_version_endpoint():
    """Forge's built-in ControlNet serves /controlnet/model_list but not /controlnet/version."""
    service, session = _webui()
    session.routes[('GET', A1111Webservice.Endpoints.CONTROLNET_VERSION)] = NOT_FOUND
    assert service.get_capabilities().controlnet


def test_webui_discovery_raises_server_errors_other_than_not_found():
    """Only a 404 means a missing feature; other failures still raise."""
    service, session = _webui()
    session.routes[('GET', A1111Webservice.Endpoints.CONTROLNET_MODELS)] = CannedStatus(500, 'boom')
    with pytest.raises(ServerError, match='500'):
        service.list_controlnet_models()


def test_comfyui_without_ultimate_upscale_node_reports_it_missing():
    """ComfyUI reports Ultimate SD Upscale only when its node is installed."""
    service, _ = _comfyui(full=False)
    assert service.get_capabilities() == BackendCapabilities(controlnet=True, ultimate_upscale=False,
                                                             scheduler=True, interrogate=False, free_memory=True)


# connect_to_backend

@pytest.fixture(name='server')
def _server_fixture(monkeypatch) -> FakeSession:
    """One fake server that every client `connect_to_backend` creates talks to."""
    session = FakeSession({('GET', ComfyEndpoints.SYSTEM_STATS): NOT_FOUND,
                           ('GET', A1111Webservice.Endpoints.OPTIONS): NOT_FOUND})
    monkeypatch.setattr(detect.ComfyUiWebservice, '__init__', _with_session(ComfyUiWebservice.__init__, session))
    monkeypatch.setattr(detect.A1111Webservice, '__init__', _with_session(A1111Webservice.__init__, session))
    return session


def _with_session(init: Callable[..., None], session: FakeSession) -> Callable[..., None]:
    def patched(self, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        self._session = session
    return patched


def test_connect_returns_comfyui_for_a_server_with_system_stats(server: FakeSession):
    """A server answering /system_stats with system info is ComfyUI."""
    server.routes[('GET', ComfyEndpoints.SYSTEM_STATS)] = {'system': {'os': 'posix'}, 'devices': []}
    service = connect_to_backend('http://server.invalid/', request_timeout=7)
    assert isinstance(service, ComfyUiWebservice)
    assert service.server_url == 'http://server.invalid' and service.request_timeout == 7
    assert not server.of('GET', A1111Webservice.Endpoints.OPTIONS)


def test_connect_returns_webui_for_a_server_with_sdapi_options(server: FakeSession):
    """A server without /system_stats that answers /sdapi/v1/options is WebUI."""
    server.routes[('GET', A1111Webservice.Endpoints.OPTIONS)] = {'sd_model_checkpoint': CHECKPOINT}
    service = connect_to_backend('http://server.invalid')
    assert isinstance(service, A1111Webservice)


def test_connect_passes_credentials_to_a_webui_that_requires_them(server: FakeSession):
    """The credentials provider answers WebUI's 401, and the returned client keeps the accepted credentials."""
    credentials = ('user', 'password')

    def authenticated(body: Any) -> Callable[[], Any]:
        return lambda: body if server.auth == credentials else CannedStatus(401, 'Unauthorized')

    server.routes[('GET', A1111Webservice.Endpoints.OPTIONS)] = authenticated({'sd_model_checkpoint': CHECKPOINT})
    server.routes[('GET', A1111Webservice.Endpoints.PROGRESS)] = authenticated({})
    service = connect_to_backend('http://server.invalid', credentials_provider=lambda: credentials)
    assert isinstance(service, A1111Webservice)
    assert server.auth == credentials


def test_connect_without_credentials_for_a_webui_that_requires_them_raises(server: FakeSession):
    """A WebUI answering 401 with no credentials provider raises AuthError."""
    server.routes[('GET', A1111Webservice.Endpoints.OPTIONS)] = CannedStatus(401, 'Unauthorized')
    with pytest.raises(AuthError):
        connect_to_backend('http://server.invalid')


@pytest.mark.parametrize('options', [NOT_FOUND, ['not', 'an', 'object']])
def test_connect_to_an_unrecognized_server_raises(server: FakeSession, options: Any):
    """A server that answers neither probe with a JSON object raises UnexpectedResponseError naming both probes."""
    server.routes[('GET', A1111Webservice.Endpoints.OPTIONS)] = options
    with pytest.raises(UnexpectedResponseError, match='--api'):
        connect_to_backend('http://server.invalid')
