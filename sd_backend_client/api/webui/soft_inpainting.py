"""The WebUI Soft Inpainting script's settings, and their order in its `alwayson_scripts` arguments.

A1111 1.8+, Forge, reForge and Forge Neo all ship the script under the same name, with the same arguments. Its
arguments are an enable flag followed by the six settings in `SOFT_INPAINTING_PARAMETERS`, in that order.
"""
from typing import Any, Mapping, Optional

from sd_backend_client.api.shared_data.parameter_def import ParameterDef
from sd_backend_client.api.webui.script_info_types import ScriptInfo

SOFT_INPAINTING_SCRIPT_KEY = 'soft inpainting'
"""The script's name in `/sdapi/v1/scripts`, `/sdapi/v1/script-info` and `alwayson_scripts`."""

SOFT_INPAINTING_PARAMETERS: tuple[ParameterDef, ...] = (
    ParameterDef(key='schedule_bias', display_name='Schedule bias', default_value=1.0, min_val=0.0, max_val=8.0,
                 step_val=0.1,
                 description='Shifts when preservation of original content occurs during denoising. Below 1 '
                             'preserves earlier in denoising, above 1 later.'),
    ParameterDef(key='preservation_strength', display_name='Preservation strength', default_value=0.5, min_val=0.0,
                 max_val=8.0, step_val=0.05,
                 description='How strongly partially masked content is preserved. Low values favor generated '
                             'content, high values favor the original.'),
    ParameterDef(key='transition_contrast_boost', display_name='Transition contrast boost', default_value=4.0,
                 min_val=1.0, max_val=32.0, step_val=0.5,
                 description='Amplifies the contrast that may be lost in partially masked regions. Low values soften '
                             'blending and can fade detail; high values can oversaturate.'),
    ParameterDef(key='mask_influence', display_name='Mask influence', default_value=0.0, min_val=0.0, max_val=1.0,
                 step_val=0.05,
                 description='How strongly the original mask biases the difference threshold. 0 ignores the mask, '
                             '1 follows it closely.'),
    ParameterDef(key='difference_threshold', display_name='Difference threshold', default_value=0.5, min_val=0.0,
                 max_val=8.0, step_val=0.25,
                 description='How much a region can change before the original pixels stop being blended in. Original '
                             'pixels fall below 50% opacity at this difference.'),
    ParameterDef(key='difference_contrast', display_name='Difference contrast', default_value=2.0, min_val=0.0,
                 max_val=8.0, step_val=0.25,
                 description='Sharpness of the transition between blended and replaced pixels. Low values give gradual '
                             'blends that can ghost; high values make transitions sudden.'),
)
"""The script's settings after its enable flag, in argument order, with the values every known fork reports.

The first three act on the latent blend during sampling. The last three shape the mask the server composites the
result through, which it builds from how much each region changed.
"""


def soft_inpainting_parameters(script_info: Optional[ScriptInfo]) -> list[ParameterDef]:
    """Return `SOFT_INPAINTING_PARAMETERS`, updated with the labels, defaults and limits in the server's script info.

    `script_info` is the script's `/sdapi/v1/script-info` entry. Descriptions always come from
    `SOFT_INPAINTING_PARAMETERS`, since the server provides none. When `script_info` is None or its argument count
    differs, the hardcoded definitions are returned unchanged.
    """
    parameters = [parameter.model_copy() for parameter in SOFT_INPAINTING_PARAMETERS]
    if script_info is None or len(script_info.args) != len(parameters) + 1:
        return parameters
    for parameter, arg in zip(parameters, script_info.args[1:]):
        if not isinstance(arg.value, (int, float)) or isinstance(arg.value, bool):
            continue
        parameter.display_name = arg.label
        parameter.default_value = float(arg.value)
        parameter.min_val = arg.minimum
        parameter.max_val = arg.maximum
        parameter.step_val = arg.step
    return parameters


def soft_inpainting_script_args(values: Mapping[str, Any]) -> list[Any]:
    """Return the script's `alwayson_scripts` arguments: the enable flag, then each setting in argument order.

    Settings missing from `values` take their `SOFT_INPAINTING_PARAMETERS` default, so the request doesn't depend on
    the server's defaults. Keys that name no setting are left out.
    """
    return [True] + [values.get(parameter.key, parameter.default_value) for parameter in SOFT_INPAINTING_PARAMETERS]
