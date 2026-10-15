"""Contract tests: real ComfyUI and WebUI responses, replayed offline through the clients' public methods.

Each `fixtures/recorded/<label>.json` was written by `scripts/capture_fixtures.py` against a live server. The tests
replace the client's `get` with a lookup into that recording, so a response shape the parsing code no longer accepts
fails here rather than only against a live server. When a test fails after a client change requests an endpoint the
recording lacks, re-run the capture script and commit the new recording.
"""
# pylint: disable=protected-access
from typing import Any

import pytest
from pydantic import ValidationError

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui.comfyui_types import CONTROLNET_PREPROCESSOR_CATEGORY, NodeInfoResponse
from sd_backend_client.api.comfyui.controlnet_comfyui_utils import COMBO_INPUT_TYPE, INVALID_PREPROCESSOR_NODES
from sd_backend_client.api.comfyui_webservice import AsyncTaskStatus, ComfyEndpoints, ComfyUiWebservice
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor
from sd_backend_client.api.shared_data.parameter_def import ParameterDef
from sd_backend_client.api.webui.soft_inpainting import SOFT_INPAINTING_PARAMETERS

from .recorded_replay import RECORDED_DIR, recordings, replay

# ComfyUI input types get_all_preprocessors reads as parameters; any other type name is a node connection.
PARAMETER_INPUT_TYPES = {'INT', 'FLOAT', 'BOOLEAN', 'STRING', COMBO_INPUT_TYPE}

COMFYUI_RECORDINGS = recordings('comfyui')
WEBUI_RECORDINGS = recordings('webui')
# Recordings made before soft inpainting discovery existed hold no script-info, and replaying them fails.
WEBUI_SCRIPT_INFO_RECORDINGS = [recording for recording in WEBUI_RECORDINGS
                                if A1111Webservice.Endpoints.SCRIPT_INFO in recording.values[0]['responses']]


def _assert_parameters_consistent(preprocessor: ControlNetPreprocessor) -> None:
    """Each parameter's default is one of its options and inside its numeric range, and keys are unique."""
    _assert_parameter_defs_consistent(preprocessor.name, preprocessor.parameters)


def _assert_parameter_defs_consistent(owner: str, parameters: list[ParameterDef]) -> None:
    """Each parameter's default is one of its options and inside its numeric range, and keys are unique."""
    keys = [parameter.key for parameter in parameters]
    assert len(keys) == len(set(keys)), f'{owner} repeats a parameter key: {keys}'
    for parameter in parameters:
        label = f'{owner}.{parameter.key}'
        if parameter.option_list is not None:
            assert parameter.default_value in parameter.option_list, label
        if isinstance(parameter.default_value, (int, float)) and not isinstance(parameter.default_value, bool):
            if parameter.min_val is not None:
                assert parameter.default_value >= parameter.min_val, label
            if parameter.max_val is not None:
                assert parameter.default_value <= parameter.max_val, label


def test_each_backend_has_a_recording():
    """At least one recording per backend exists, so the parametrized tests below never all vanish."""
    assert COMFYUI_RECORDINGS, f'No ComfyUI recording in {RECORDED_DIR}; run scripts/capture_fixtures.py'
    assert WEBUI_RECORDINGS, f'No WebUI recording in {RECORDED_DIR}; run scripts/capture_fixtures.py'


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS + WEBUI_RECORDINGS)
def test_capabilities_match_the_live_server(recording):
    """Every replayed discovery listing parses, and capabilities match what the client reported against the server.

    The recorded capabilities pin what each WebUI fork serves: Forge Neo has no interrogation, for example."""
    service = _comfy(recording) if recording['meta']['backend'] == 'comfyui' else _webui(recording)
    assert service.get_capabilities().model_dump() == recording['meta']['capabilities']
    for list_options in (service.list_checkpoints, service.list_vaes, service.list_loras, service.list_hypernetworks,
                         service.list_samplers, service.list_schedulers, service.list_upscalers,
                         service.list_controlnet_models):
        list_options()


# ComfyUI

def _comfy(recording: dict[str, Any]) -> ComfyUiWebservice:
    service = ComfyUiWebservice('http://127.0.0.1:8188')
    replay(service, recording)
    return service


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_system_stats_parse(recording):
    """/system_stats validates as SystemStatResponse and reports the recorded version."""
    stats = _comfy(recording).get_system_stats()
    assert stats.system.comfyui_version == recording['meta']['version']
    assert stats.devices


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_sampler_and_scheduler_names(recording):
    """KSampler's node info yields non-empty sampler and scheduler name lists."""
    service = _comfy(recording)
    for names in (service.get_sampler_names(), service.get_scheduler_names()):
        assert names and all(isinstance(name, str) for name in names)


def _requires_other_connection(node: NodeInfoResponse) -> bool:
    """Whether a required input, other than `image` and `mask`, takes a node connection rather than a parameter.

    get_all_preprocessors skips such nodes, since DynamicPreprocessorNode wires only the image and mask."""
    for input_name, input_tuple in node.input.required.items():
        if input_name in ('image', 'mask'):
            continue
        input_type = input_tuple[0]
        if isinstance(input_type, str) and input_type not in PARAMETER_INPUT_TYPES:
            return True
    return False


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_discovers_every_recorded_preprocessor(recording):
    """Every usable node in the preprocessor category becomes a preprocessor, with consistent parameters."""
    object_info = recording['responses'][ComfyEndpoints.OBJECT_INFO]
    expected = set()
    for name, info in object_info.items():
        try:
            node = NodeInfoResponse.model_validate(info)
        except ValidationError:
            continue
        if CONTROLNET_PREPROCESSOR_CATEGORY in node.category and name not in INVALID_PREPROCESSOR_NODES \
                and not _requires_other_connection(node):
            expected.add(name)
    assert expected, 'The recording has no usable preprocessor nodes to test discovery against'

    service = _comfy(recording)
    preprocessors = service.get_controlnet_preprocessors()
    assert {preprocessor.name for preprocessor in preprocessors} == expected
    for preprocessor in preprocessors:
        _assert_parameters_consistent(preprocessor)
    assert isinstance(service.get_controlnet_type_categories(preprocessors), dict)


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_discovery_lists_recorded_models(recording):
    """Checkpoints, ControlNet models and samplers list under the names ComfyUI reported."""
    responses = recording['responses']
    service = _comfy(recording)
    assert [option.name for option in service.list_checkpoints()] == responses['/models/checkpoints']
    assert [model.full_model_name for model in service.list_controlnet_models()] == responses['/models/controlnet']
    assert [option.name for option in service.list_samplers()] == service.get_sampler_names()


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_finished_history_entry(recording):
    """The recorded job's /history entry reads as FINISHED, with the image references it produced."""
    meta = recording['meta']
    progress = _comfy(recording).check_queue_entry(meta['prompt_id'], meta['number'])
    assert progress.status == AsyncTaskStatus.FINISHED
    assert progress.outputs is not None and progress.outputs.images
    for image in progress.outputs.images:
        assert image.filename and image.type == 'output'


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_queue_info_parses(recording):
    """/queue validates as QueueInfoResponse."""
    queue_info = _comfy(recording).get_queue_info()
    assert isinstance(queue_info.queue_running, list) and isinstance(queue_info.queue_pending, list)


@pytest.mark.parametrize('recording', COMFYUI_RECORDINGS)
def test_comfyui_soft_inpainting_parameters(recording):
    """Differential Diffusion's settings parse from the recorded node info, and never offer a strength of 0."""
    parameters = _comfy(recording).get_soft_inpainting_parameters()
    _assert_parameter_defs_consistent('DifferentialDiffusion', parameters)
    for parameter in parameters:
        assert parameter.min_val is not None and parameter.min_val > 0


# WebUI

def _webui(recording: dict[str, Any]) -> A1111Webservice:
    service = A1111Webservice('http://127.0.0.1:7860')
    replay(service, recording)
    return service


@pytest.mark.parametrize('recording', WEBUI_RECORDINGS)
def test_webui_option_lists(recording):
    """Samplers, latent upscale modes and scripts parse, and the name lists are non-empty."""
    service = _webui(recording)
    assert service.get_samplers()
    assert service.get_latent_upscale_modes()
    scripts = service.get_scripts()
    assert isinstance(scripts.txt2img, list) and isinstance(scripts.img2img, list)


@pytest.mark.parametrize('recording', WEBUI_RECORDINGS)
def test_webui_discovery_lists_recorded_samplers_and_controlnet_models(recording):
    """Samplers list under shared names labelled with WebUI's names, and ControlNet models match model_list.

    Forge's model_list starts with a 'None' entry, which discovery leaves out."""
    service = _webui(recording)
    samplers = service.list_samplers()
    assert [option.display_name for option in samplers] == [sampler.name for sampler in service.get_samplers()]
    assert any(option.name == 'euler_ancestral' for option in samplers)
    assert [model.full_model_name for model in service.list_controlnet_models()] == \
        [name for name in service.get_controlnet_models().model_list if name.lower() != 'none']


@pytest.mark.parametrize('recording', WEBUI_RECORDINGS)
def test_webui_discovers_every_recorded_module(recording):
    """Every /controlnet/module_list module becomes a preprocessor, with consistent parameters."""
    service = _webui(recording)
    module_names = service.get_controlnet_modules().module_list
    preprocessors = service.get_controlnet_preprocessors()
    assert [preprocessor.name for preprocessor in preprocessors] == module_names
    for preprocessor in preprocessors:
        _assert_parameters_consistent(preprocessor)
    assert isinstance(service.get_controlnet_type_categories(), dict)


@pytest.mark.parametrize('recording', WEBUI_SCRIPT_INFO_RECORDINGS)
def test_webui_soft_inpainting_parameters_match_the_known_definitions(recording):
    """The recorded script-info gives the six known settings, with the defaults and limits the package hardcodes."""
    parameters = _webui(recording).get_soft_inpainting_parameters()
    _assert_parameter_defs_consistent('Soft inpainting', parameters)
    known = [(parameter.key, parameter.display_name, parameter.default_value, parameter.min_val, parameter.max_val,
              parameter.step_val) for parameter in SOFT_INPAINTING_PARAMETERS]
    assert [(parameter.key, parameter.display_name, parameter.default_value, parameter.min_val, parameter.max_val,
             parameter.step_val) for parameter in parameters] == known
