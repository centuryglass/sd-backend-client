"""Offline tests for ControlNet option discovery: preprocessor parsing on both backends, and control type categories.

The inputs are hand-written in the shapes ComfyUI's /object_info and the WebUI's /controlnet/module_list return.
`fixtures/a1111_controlnet_module_list.json` is a subset of the A1111 sd-webui-controlnet extension's
/controlnet/module_list response, written from the extension's source (`get_modules_detail` and
`PreprocessorParameter.api_json`). To refresh it from a live server, run
`curl "$SD_API_URL/controlnet/module_list?alias_names=false"` and keep the entries the tests use.
"""
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sd_backend_client.api.comfyui.comfyui_types import NodeInfoResponse
from sd_backend_client.api.comfyui.controlnet_comfyui_utils import get_all_preprocessors as comfy_preprocessors
from sd_backend_client.api.shared_data.controlnet.controlnet_category_builder import ControlNetCategoryBuilder
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import (ControlNetPreprocessor,
                                                                                PreprocessorParams)
from sd_backend_client.api.webui.controlnet_webui_constants import ControlNetModuleResponse, ModuleDetail
from sd_backend_client.api.webui.controlnet_webui_utils import get_all_preprocessors as webui_preprocessors
from sd_backend_client.errors import UnexpectedResponseError

PREPROCESSOR_CATEGORY = 'ControlNet Preprocessors/Line Extractors'
A1111_MODULE_LIST = Path(__file__).parent / 'fixtures' / 'a1111_controlnet_module_list.json'


def _node(name: str, required: dict[str, Any], optional: dict[str, Any] | None = None,
          category: str = PREPROCESSOR_CATEGORY, display_name: str | None = None) -> NodeInfoResponse:
    """A NodeInfoResponse in /object_info form, with input_order following the dicts' order."""
    input_types: dict[str, Any] = {'required': required}
    input_order = {'required': list(required)}
    if optional is not None:
        input_types['optional'] = optional
        input_order['optional'] = list(optional)
    return NodeInfoResponse.model_validate({
        'input': input_types, 'input_order': input_order, 'output': ['IMAGE'], 'output_is_list': [False],
        'output_name': ['IMAGE'], 'name': name, 'display_name': display_name, 'description': None,
        'python_module': 'custom_nodes.comfyui_controlnet_aux', 'category': category, 'output_node': False})


def _by_name(preprocessors: list[ControlNetPreprocessor]) -> dict[str, ControlNetPreprocessor]:
    return {preprocessor.name: preprocessor for preprocessor in preprocessors}


# ComfyUI

def test_comfyui_parses_each_parameter_type():
    """INT, FLOAT, BOOLEAN, STRING and combo inputs become ParameterDefs with defaults and ranges."""
    node = _node('AnyLinePreprocessor', {
        'image': ['IMAGE'],
        'resolution': ['INT', {'default': 512, 'min': 64, 'max': 16384, 'step': 64, 'tooltip': 'Output size'}],
        'strength': ['FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}],
        'safe': ['BOOLEAN', {'default': True}],
        'merge_with_lineart': [['lineart_standard', 'lineart_realistic'], {}],
    }, optional={'mask': ['MASK'], 'label': ['STRING', {'multiline': False}]}, display_name='AnyLine Lineart')
    preprocessor = comfy_preprocessors({'AnyLinePreprocessor': node})[0]

    params = {param.key: param for param in preprocessor.parameters}
    assert list(params) == ['resolution', 'strength', 'safe', 'merge_with_lineart', 'label']
    resolution = params['resolution']
    assert (resolution.default_value, resolution.min_val, resolution.max_val, resolution.step_val) == \
        (512, 64, 16384, 64)
    assert resolution.description == 'Output size'
    assert (params['strength'].default_value, params['strength'].step_val) == (0.5, 0.01)
    assert params['safe'].default_value is True
    assert params['merge_with_lineart'].default_value == 'lineart_standard'
    assert params['label'].default_value == ''
    assert params['label'].required is False and resolution.required is True

    assert preprocessor.has_image_input and preprocessor.has_mask_input
    assert preprocessor.description == 'AnyLine Lineart'
    assert preprocessor.category_name == PREPROCESSOR_CATEGORY


def test_comfyui_combo_inputs_keep_their_options():
    """A combo input's choices become the parameter's option_list."""
    node = _node('ModePreprocessor', {'image': ['IMAGE'], 'mode': [['a', 'b'], {}]})
    assert comfy_preprocessors({'ModePreprocessor': node})[0].parameters[0].option_list == ['a', 'b']


def test_comfyui_reads_combo_type_inputs():
    """The newer `["COMBO", {"options": [...]}]` form parses like a list combo, keeping a listed default."""
    node = _node('ModePreprocessor', {
        'image': ['IMAGE'],
        'mode': ['COMBO', {'options': ['a', 'b', 'c'], 'default': 'b', 'multiselect': False, 'tooltip': 'Mode'}],
    })
    param = comfy_preprocessors({'ModePreprocessor': node})[0].parameters[0]
    assert (param.option_list, param.default_value, param.description) == (['a', 'b', 'c'], 'b', 'Mode')


def test_comfyui_combo_default_falls_back_to_first_option():
    """A combo default that isn't one of the options is replaced by the first option."""
    node = _node('ModePreprocessor', {'image': ['IMAGE'], 'mode': [['a', 'b'], {'default': 'z'}]})
    assert comfy_preprocessors({'ModePreprocessor': node})[0].parameters[0].default_value == 'a'


@pytest.mark.parametrize('combo', [
    [[], {}],
    ['COMBO', {'options': []}],
    ['COMBO', {'options': ['a', 'b'], 'multiselect': True}],
], ids=['empty-list', 'empty-combo', 'multiselect'])
def test_comfyui_unusable_combos_are_treated_as_unknown_inputs(combo: list[Any]):
    """An empty or multiselect combo skips the node when required, and is dropped when optional."""
    required = _node('RequiredPreprocessor', {'image': ['IMAGE'], 'mode': combo})
    optional = _node('OptionalPreprocessor', {'image': ['IMAGE']}, optional={'mode': combo})
    preprocessors = _by_name(comfy_preprocessors({'Required': required, 'Optional': optional}))
    assert list(preprocessors) == ['OptionalPreprocessor']
    assert not preprocessors['OptionalPreprocessor'].parameters


def test_comfyui_combo_options_validate_preprocessor_params():
    """PreprocessorParams accepts a listed combo value and rejects an unlisted one, as the server does."""
    node = _node('ModePreprocessor', {'image': ['IMAGE'], 'mode': ['COMBO', {'options': ['a', 'b']}]})
    preprocessor = comfy_preprocessors({'ModePreprocessor': node})[0]
    assert PreprocessorParams(typedef=preprocessor, parameter_values={'mode': 'b'}).parameter_values == {'mode': 'b'}
    with pytest.raises(ValidationError, match='invalid option'):
        PreprocessorParams(typedef=preprocessor, parameter_values={'mode': 'z'})


def test_comfyui_filters_non_preprocessor_and_unusable_nodes():
    """Only usable nodes in the preprocessor category are kept."""
    nodes = {
        'CannyEdgePreprocessor': _node('CannyEdgePreprocessor', {'image': ['IMAGE']}),
        'KSampler': _node('KSampler', {'model': ['MODEL']}, category='sampling'),
        'ExecuteAllControlNetPreprocessors': _node('ExecuteAllControlNetPreprocessors', {'image': ['IMAGE']}),
        'OddInputPreprocessor': _node('OddInputPreprocessor', {'image': ['IMAGE'], 'model': ['DEPTH_MODEL']}),
    }
    preprocessors = _by_name(comfy_preprocessors(nodes))
    assert list(preprocessors) == ['CannyEdgePreprocessor']
    assert preprocessors['CannyEdgePreprocessor'].description == 'CannyEdge'
    assert preprocessors['CannyEdgePreprocessor'].has_mask_input is False


def test_comfyui_ignores_unhandled_optional_input():
    """An optional input of an unknown type is dropped without discarding the node."""
    node = _node('DepthPreprocessor', {'image': ['IMAGE']}, optional={'model': ['DEPTH_MODEL']})
    preprocessors = comfy_preprocessors({'DepthPreprocessor': node})
    assert [preprocessor.name for preprocessor in preprocessors] == ['DepthPreprocessor']
    assert not preprocessors[0].parameters


# WebUI

def test_webui_preset_parameters_without_module_details():
    """Without API details, parameters come from the preset tables in controlnet_webui_constants."""
    preprocessors = _by_name(webui_preprocessors(['canny', 'inpaint_only', 'depth_marigold', 'reference_only']))

    canny = preprocessors['canny']
    assert [param.key for param in canny.parameters] == \
        ['control_mode', 'resize_mode', 'processor_res', 'threshold_a', 'threshold_b']
    assert canny.parameters[3].display_name == 'Low Threshold'
    assert canny.model_free is False
    assert [param.key for param in preprocessors['inpaint_only'].parameters] == ['control_mode']
    assert preprocessors['depth_marigold'].parameters[2].default_value == 768
    assert preprocessors['reference_only'].model_free is True


def test_webui_module_details_map_resolution_and_thresholds():
    """API sliders map to processor_res by name and to threshold_a/threshold_b by order."""
    details = {'canny': ModuleDetail.model_validate({'model_free': False, 'sliders': [
        {'name': 'Resolution', 'value': 512, 'min': 64, 'max': 2048, 'step': 8},
        {'name': 'Low Threshold', 'value': 100, 'min': 1, 'max': 255, 'step': 1},
        {'name': 'High Threshold', 'value': 200, 'min': 1, 'max': 255, 'step': 1},
    ]})}
    canny = webui_preprocessors(['canny'], details)[0]

    params = {param.key: param for param in canny.parameters}
    assert list(params) == ['control_mode', 'resize_mode', 'processor_res', 'threshold_a', 'threshold_b']
    assert (params['processor_res'].min_val, params['processor_res'].max_val) == (64, 2048)
    assert params['threshold_a'].display_name == 'Low Threshold'
    assert params['threshold_b'].default_value == 200


def test_webui_module_details_reject_a_third_threshold():
    """More than two non-resolution sliders is an error."""
    slider = {'name': 'Value', 'value': 0, 'min': 0, 'max': 1, 'step': 1}
    details = {'odd': ModuleDetail.model_validate({'model_free': True, 'sliders': [slider] * 3})}
    with pytest.raises(UnexpectedResponseError, match='odd'):
        webui_preprocessors(['odd'], details)


def test_webui_reads_a1111_module_list_response():
    """The A1111 extension's module_list response yields API-defined parameters, not the presets."""
    response = ControlNetModuleResponse.model_validate(json.loads(A1111_MODULE_LIST.read_text(encoding='utf-8')))
    assert response.module_details is not None
    preprocessors = _by_name(webui_preprocessors(response.module_list, response.module_details))
    assert list(preprocessors) == response.module_list

    mlsd = {param.key: param for param in preprocessors['mlsd'].parameters}
    assert list(mlsd) == ['control_mode', 'resize_mode', 'processor_res', 'threshold_a', 'threshold_b']
    assert (mlsd['processor_res'].default_value, mlsd['processor_res'].step_val) == (512, 8)
    assert mlsd['threshold_a'].display_name == 'MLSD Value Threshold'
    assert (mlsd['threshold_b'].default_value, mlsd['threshold_b'].max_val) == (0.1, 20.0)

    # The preset table names this slider 'Style Fidelity', so the label shows the API definition was used.
    reference = preprocessors['reference_only']
    assert reference.parameters[-1].display_name == 'Style Fidelity (only for Balanced mode)'
    assert reference.model_free is True
    assert [param.key for param in preprocessors['inpaint_only'].parameters] == ['control_mode']


def test_webui_module_slider_step_is_optional():
    """A slider without a step parses, leaving the parameter's step unset."""
    details = {'blur': ModuleDetail.model_validate({'model_free': True, 'sliders': [
        {'name': 'Sigma', 'value': 9.0, 'min': 0.01, 'max': 64.0}]})}
    sigma = webui_preprocessors(['blur'], details)[0].parameters[-1]
    assert (sigma.key, sigma.default_value, sigma.step_val) == ('threshold_a', 9.0, None)


def test_webui_forge_module_list_has_no_details():
    """Forge's response has only module_list, so the presets are used."""
    response = ControlNetModuleResponse.model_validate({'module_list': ['canny']})
    assert response.module_details is None


# Control type categories

def test_category_builder_keeps_types_with_available_options():
    """Types keep listed and pattern-matched options that exist, and types with nothing available are dropped."""
    builder = ControlNetCategoryBuilder(['canny', 'CannyEdgePreprocessor', 'depth_midas'],
                                        ['control_v11p_sd15_canny [d14c016b]', 'my_canny_model'])
    control_types = builder.get_control_types()

    canny = control_types['Canny']
    assert canny['module_list'] == ['None', 'canny', 'CannyEdgePreprocessor']
    assert canny['model_list'] == ['None', 'control_v11p_sd15_canny [d14c016b]', 'my_canny_model']
    assert canny['default_option'] == 'canny'
    assert canny['default_model'] == 'control_v11p_sd15_canny [d14c016b]'
    assert 'Depth' not in control_types


def test_category_builder_matches_preprocessors_by_comfyui_category():
    """A ComfyUI preprocessor whose name doesn't match a type's pattern is matched through its category."""
    builder = ControlNetCategoryBuilder(['Zoe-DepthMapPreprocessor', 'MiDaS-NormalMapPreprocessor'],
                                        ['control_depth.safetensors'],
                                        {'MiDaS-NormalMapPreprocessor': 'ControlNet Preprocessors/Depth'})
    assert builder.get_control_types()['Depth']['module_list'] == \
        ['None', 'Zoe-DepthMapPreprocessor', 'MiDaS-NormalMapPreprocessor']


def test_category_builder_prefers_api_type_definitions():
    """API control type definitions are applied before the static ones."""
    api_types = {'control_types': {'Canny': {'module_list': ['none', 'canny'], 'model_list': ['None', 'api_model'],
                                             'default_option': 'canny', 'default_model': 'api_model'}}}
    builder = ControlNetCategoryBuilder(['canny'], ['api_model', 'control_v11p_sd15_canny [d14c016b]'],
                                        None, api_types)  # type: ignore[arg-type]
    assert builder.get_control_types()['Canny']['default_model'] == 'api_model'
