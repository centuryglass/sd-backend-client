"""Helper functions for processing ControlNet configuration data for use with the WebUI API."""
import logging
from typing import Optional

from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, ParameterDef
from sd_backend_client.api.webui.controlnet_webui_constants import RESIZE_MODE_PARAM_KEY, RESIZE_MODE_LABEL, \
    RESIZE_MODE_DEFAULT, \
    RESIZE_MODE_OPTIONS, ModuleDetail, PREPROCESSOR_NO_CONTROL_MODE, CONTROL_MODE_PARAM_KEY, CONTROL_MODE_LABEL, \
    CONTROL_MODE_DEFAULT, CONTROL_MODE_OPTIONS, FIRST_GENERIC_PARAMETER_KEY, SECOND_GENERIC_PARAMETER_KEY, \
    PREPROCESSOR_RES_PARAM_NAME, PREPROCESSOR_RES_PARAM_KEY, PREPROCESSOR_NO_RESOLUTION, PREPROCESSOR_RES_DEFAULT, \
    PREPROCESSOR_RES_DEFAULTS, PREPROCESSOR_RES_LABEL, PREPROCESSOR_RES_MIN, PREPROCESSOR_RES_MAX, \
    PREPROCESSOR_RES_STEP, THRESHOLD_A_PARAMETER_NAMES, THRESHOLD_B_PARAMETER_NAMES, PREPROCESSOR_MODEL_FREE
from sd_backend_client.errors import UnexpectedResponseError

logger = logging.getLogger(__name__)


def _resize_mode_parameter() -> ParameterDef:
    """Returns the Resize mode preprocessor parameter, which is always identical, but only used for certain
       preprocessors."""
    return ParameterDef(key=RESIZE_MODE_PARAM_KEY,
                        default_value=RESIZE_MODE_DEFAULT,
                        display_name=RESIZE_MODE_LABEL,
                        required=True,
                        option_list=list(RESIZE_MODE_OPTIONS))


def get_all_preprocessors(preprocessor_names: list[str],
                          preprocessor_details: Optional[dict[str, ModuleDetail]] = None
                          ) -> list[ControlNetPreprocessor]:
    """Returns the full set of available preprocessors, loading preprocessor details from API definitions if possible,
       or predefined definitions otherwise."""
    preprocessors: list[ControlNetPreprocessor] = []
    for preprocessor_name in preprocessor_names:
        parameters: list[ParameterDef] = []

        # Start with standard "control mode " parameter:
        if preprocessor_name not in PREPROCESSOR_NO_CONTROL_MODE:
            control_type_param = ParameterDef(key=CONTROL_MODE_PARAM_KEY,
                                              default_value=CONTROL_MODE_DEFAULT,
                                              display_name=CONTROL_MODE_LABEL,
                                              required=True,
                                              option_list=list(CONTROL_MODE_OPTIONS))
            parameters.append(control_type_param)

        # Load preprocessor parameters from API details if possible:
        if preprocessor_details is not None and preprocessor_name in preprocessor_details:
            preprocessor_dict = preprocessor_details[preprocessor_name]
            next_threshold_key: Optional[str] = FIRST_GENERIC_PARAMETER_KEY
            resize_insert_index = len(parameters)

            # Iterate through parameter definitions.  Resolution parameter is identified by name, threshold_a and
            # threshold_b are identified by their order.
            for parameter_definition in preprocessor_dict.sliders:
                name = parameter_definition.name
                if name.lower() == PREPROCESSOR_RES_PARAM_NAME.lower():
                    parameters.insert(resize_insert_index, _resize_mode_parameter())
                    res_param = ParameterDef(key=PREPROCESSOR_RES_PARAM_KEY,
                                             display_name=PREPROCESSOR_RES_LABEL,
                                             default_value=parameter_definition.value,
                                             required=True,
                                             min_val=parameter_definition.min,
                                             max_val=parameter_definition.max,
                                             step_val=parameter_definition.step)
                    parameters.insert(resize_insert_index + 1, res_param)
                else:
                    if next_threshold_key is None:
                        raise UnexpectedResponseError(f'Unexpected extra parameter in "{preprocessor_name}" details')
                    key = next_threshold_key
                    if next_threshold_key == FIRST_GENERIC_PARAMETER_KEY:
                        next_threshold_key = SECOND_GENERIC_PARAMETER_KEY
                    else:
                        next_threshold_key = None
                    parameter = ParameterDef(key=key,
                                             display_name=name,
                                             default_value=parameter_definition.value,
                                             required=True,
                                             min_val=parameter_definition.min,
                                             max_val=parameter_definition.max,
                                             step_val=parameter_definition.step)
                    parameters.append(parameter)

        else:  # No API preprocessor definition, use predefined constants:
            if preprocessor_name not in PREPROCESSOR_NO_RESOLUTION:
                parameters.append(_resize_mode_parameter())
                resolution_default = PREPROCESSOR_RES_DEFAULT if preprocessor_name not in PREPROCESSOR_RES_DEFAULTS \
                    else PREPROCESSOR_RES_DEFAULTS[preprocessor_name]
                parameters.append(ParameterDef(key=PREPROCESSOR_RES_PARAM_KEY,
                                               display_name=PREPROCESSOR_RES_LABEL,
                                               default_value=resolution_default,
                                               required=True,
                                               min_val=PREPROCESSOR_RES_MIN,
                                               max_val=PREPROCESSOR_RES_MAX,
                                               step_val=PREPROCESSOR_RES_STEP))
            for param_key, preset_list in ((FIRST_GENERIC_PARAMETER_KEY, THRESHOLD_A_PARAMETER_NAMES),
                                           (SECOND_GENERIC_PARAMETER_KEY, THRESHOLD_B_PARAMETER_NAMES)):
                if preprocessor_name not in preset_list:
                    break  # "threshold_b" will never be used if "threshold_a" isn't
                threshold_def = preset_list[preprocessor_name]
                parameters.append(ParameterDef(key=param_key,
                                               display_name=threshold_def.name,
                                               default_value=threshold_def.default,
                                               required=True,
                                               min_val=threshold_def.min,
                                               max_val=threshold_def.max,
                                               step_val=threshold_def.step))
        # create preprocessor:
        preprocessor = ControlNetPreprocessor(name=preprocessor_name, parameters=parameters)
        if preprocessor_details is not None and preprocessor_name in preprocessor_details:
            preprocessor.model_free = preprocessor_details[preprocessor_name].model_free
        else:
            preprocessor.model_free = preprocessor_name in PREPROCESSOR_MODEL_FREE
        preprocessors.append(preprocessor)
    return preprocessors