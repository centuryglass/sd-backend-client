"""Data format definitions for the WebUI API's ControlNet endpoints."""
from typing import Dict, Optional, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.util.visual.image_utils import image_to_base64

CONTROLNET_SCRIPT_KEY = 'controlNet'


class ControlNetModelResponse(BaseModel):
    """Response format from the WebUI API when loading ControlNet model options."""
    model_config = ConfigDict(protected_namespaces=('model_validate', 'model_dump'))

    model_list: list[str]


class ControlNetSliderDef(BaseModel):
    """Defines a ControlNet preprocessor's use of the resolution, 'threshold_a' or 'threshold_b' parameters."""
    name: str
    min: float | int
    max: float | int
    default: float | int
    step: float | int


class ModuleSliderDef(BaseModel):
    """One preprocessor slider as the sd-webui-controlnet extension's /controlnet/module_list endpoint returns it."""
    name: str
    value: float | int  # The slider's default.
    min: float | int
    max: float | int
    step: Optional[float | int] = None  # Omitted by older extension versions.


class ModuleDetail(BaseModel):
    """Defines a ControlNet preprocessor's parameters as returned by the preprocessor module list endpoint."""
    model_config = ConfigDict(protected_namespaces=('model_validate', 'model_dump'))

    model_free: bool  # Whether the preprocessor module can be used without a model.
    sliders: list[ModuleSliderDef]


class ControlNetModuleResponse(BaseModel):
    """Response format when loading ControlNet preprocessor options.

    The A1111 extension sends the details under the `module_detail` key, which `module_details` reads. Forge omits it,
    so it defaults to None.
    """
    module_list: list[str]
    module_details: Optional[Dict[str, ModuleDetail]] = Field(default=None, validation_alias='module_detail')


class ControlNetUnitDict(BaseModel):
    """Data format used with the ControlNet script's values within the alwayson_scripts section in WebUI generation
       requests. In the script values section, The ControlNet value parameter is a list holding up to three of these
       objects.  Optional parameters that only affect the UI are omitted."""
    use_preview_as_input: bool = False
    enabled: bool = True
    pixel_perfect: Optional[bool] = True
    low_vram: Optional[bool] = False
    module: str = 'None' # AKA preprocessor
    model: str = 'None'
    weight: float= 1.0
    image: Optional[str] = None  # base64 image, usually necessary.
    resize_mode: Literal['Just Resize', 'Crop and Resize', 'Resize and Fill'] = 'Crop and Resize'
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    control_mode: Optional[Literal['Balanced', 'My prompt is more important', 'ControlNet is more important']] = 'Balanced'

    # preprocessor-specific values:
    processor_res: Optional[int] = None  # Some use this, some don't. Set to -1 or omit if it's not used.
    # Effects of threshold values (if any) vary based on preprocessor.
    threshold_a: Optional[float] = None
    threshold_b: Optional[float] = None

    @classmethod
    def from_unit(cls, unit: ControlNetUnit) -> 'ControlNetUnitDict':
        """Build the WebUI wire format from a shared ControlNetUnit.

        This is the single ControlNetUnit -> WebUI serialization point, so the shared data class stays free of
        WebUI-specific details. Preprocessor parameter keys (`processor_res`, `threshold_a/b`, `control_mode`,
        `resize_mode`) already match this model's field names, so any that correspond to a real field are copied
        across; anything else is ignored.
        """
        preprocessor = unit.preprocessor
        unit_dict = cls(
            module=preprocessor.typedef.name if preprocessor is not None else 'None',
            model=unit.model.full_model_name if unit.model is not None else 'None',
            weight=float(unit.control_strength),
            guidance_start=float(unit.control_start),
            guidance_end=float(unit.control_end),
            pixel_perfect=unit.pixel_perfect,
            low_vram=unit.low_vram,
        )
        if unit.image is not None:
            unit_dict.image = image_to_base64(unit.image, include_prefix=True)
        if preprocessor is not None:
            for key, value in preprocessor.parameter_values.items():
                if key in cls.model_fields:
                    setattr(unit_dict, key, value)
        return unit_dict


# Constants defining the "Control mode" option shared by most preprocessors:
CONTROL_MODE_PARAM_KEY = 'control_mode'
"""`ParameterDef.key` of the "Control mode" setting that `Backend.get_controlnet_preprocessors` adds to WebUI
preprocessors. It is a unit-level ControlNet setting, not an option of the preprocessor itself."""
CONTROL_MODE_LABEL = 'Control mode'
CONTROL_MODE_OPTIONS = ['Balanced', 'My prompt is more important', 'ControlNet is more important']
CONTROL_MODE_DEFAULT = CONTROL_MODE_OPTIONS[0]

# Constants defining the "Resolution" option shared by most preprocessors:
PREPROCESSOR_RES_PARAM_NAME = 'Resolution'  # For identifying it in an API "sliders" list
PREPROCESSOR_RES_PARAM_KEY = 'processor_res'
"""`ParameterDef.key` of the preprocessor "Resolution" setting that `Backend.get_controlnet_preprocessors` adds to WebUI
preprocessors. It is a unit-level ControlNet setting, not an option of the preprocessor itself."""
PREPROCESSOR_RES_LABEL = 'Resolution'
PREPROCESSOR_RES_MIN = 128
PREPROCESSOR_RES_MAX = 2048
PREPROCESSOR_RES_DEFAULT = 512
PREPROCESSOR_RES_STEP = 8

# Constants defining the "Resize mode" option shared by all preprocessors that also accept a resolution:
RESIZE_MODE_PARAM_KEY = 'resize_mode'
"""`ParameterDef.key` of the "Resize mode" setting that `Backend.get_controlnet_preprocessors` adds to WebUI
preprocessors. It is a unit-level ControlNet setting, not an option of the preprocessor itself."""
RESIZE_MODE_LABEL = 'Resize mode'
RESIZE_MODE_OPTIONS = ['Just Resize', 'Crop and Resize', 'Resize and Fill']
RESIZE_MODE_DEFAULT = RESIZE_MODE_OPTIONS[1]

CONTROL_WEIGHT_KEY = 'weight'
START_STEP_KEY = 'guidance_start'
END_STEP_KEY = 'guidance_end'

# Generic keys used for setting preprocessor-specific values.
FIRST_GENERIC_PARAMETER_KEY = 'threshold_a'
SECOND_GENERIC_PARAMETER_KEY = 'threshold_b'


class WebUIControlTypeDef(TypedDict):
    """One ControlNet category as sent by the WebUI."""

    module_list: list[str]
    model_list: list[str]
    default_option: str
    default_model: str


class ControlTypeResponse(TypedDict):
    """Response format for the /controlnet/control_types endpoint, containing one entry per available control type."""
    control_types: dict[str, WebUIControlTypeDef]


# PREPROCESSOR PARAMETER PRESETS:
# -------------------------------
#  The WebUI API stores all ControlNet preprocessor parameters under the "processor_res", "threshold_a", "threshold_b",
# "control_mode", and "resize_mode" keys. The Automatic1111 WebUI API sends parameter definitions we can use to handle
# differences in how these parameters are used, but the Forge WebUI does not. Instead, follow these steps when
# creating Parameter objects for Forge preprocessors and displaying UI controls:
#
# "processor_res":
#   If the preprocessor name is in the PREPROCESSOR_NO_RESOLUTION set below, omit this parameter. Otherwise, use the
#   PREPROCESSOR_RES_PARAM_NAME, PREPROCESSOR_RES_PARAM_KEY, PREPROCESSOR_RES_MIN, PREPROCESSOR_RES_MAX, and
#   PREPROCESSOR_RES_DEFAULT constants above to define this as a TYPE_INT Parameter. If the preprocessor name has an
#   entry in the PREPROCESSOR_RES_DEFAULTS dict below, that value should replace PREPROCESSOR_RES_DEFAULT.
#
# "control_mode":
#   If the preprocessor name is in the PREPROCESSOR_NO_CONTROL_MODE set below, omit this parameter. Otherwise, add it
#   as a TYPE_STR Parameter with valid options set to the CONTROL_MODE_OPTIONS string list defined above.
#
# "resize_mode":
#   This should be included in any preprocessor that uses "processor_res".  It should also be a TYPE_STR Parameter, with
#   valid options set to the RESIZE_MODE_OPTIONS string list defined above.
#
# "threshold_a", "threshold_b":
#   The name, default, and range values for these parameters are stored under the preprocessor name in the
#   THRESHOLD_A_PARAMETER_NAMES and THRESHOLD_B_PARAMETER_NAMES dicts below.  They're always either TYPE_INT or
#   TYPE_FLOAT, check the type of the 'min', 'max', 'default', or 'step' values to determine which one.  If the
#   preprocessor name isn't present in one of the THRESHOLD_*_PARAMETER_NAMES dicts, then the preprocessor doesn't use
#   that value.
#
#  When parameterizing values, make sure "threshold_a" always comes before "threshold_b", because the order is used to
#  determine which is which.

# Defines the set of preprocessors that have alternate default resolutions.
PREPROCESSOR_RES_DEFAULTS: dict[str, int] = {
    'depth_marigold': 768
}

# Defines the set of preprocessors that explicitly exclude resolution.
PREPROCESSOR_NO_RESOLUTION: set[str] = {
    'inpaint_only',
    'invert (from white bg & black line)',
    'shuffle',
    'ip-adapter_pulid',
    'inpaint_only+lama',
    'inpaint_global_harmonious',
    'facexlib',
    'None',
    'none'
}

# Defines the set of preprocessors that don't have the control_mode option.
PREPROCESSOR_NO_CONTROL_MODE: set[str] = {
    'ip-adapter-auto',
    'revision_clipvision',
    'ip-adapter_clip_h',
    't2ia_style_clipvision',
    'revision_ignore_prompt',
    'ip-adapter_face_id_plus',
    'ip-adapter_face_id',
    'ip-adapter_clip_sdxl_plus_vith',
    'ip-adapter_clip_g',
    'None',
    'none'

}

# Defines preprocessors that don't need a corresponding model.
PREPROCESSOR_MODEL_FREE: set[str] = {
    'revision_clipvision',
    'revision_ignore_prompt',
    'reference_only',
    'reference_adain+attn',
    'reference_adain'
}

THRESHOLD_A_PARAMETER_NAMES: dict[str, ControlNetSliderDef] = {
    'reference_only': ControlNetSliderDef(name='Style Fidelity', min=0.0, max=1.0, default=0.5, step=0.01),
    'canny': ControlNetSliderDef(name='Low Threshold', min=0, max=256, default=100, step=1),
    'CLIP-G (Revision)': ControlNetSliderDef(name='Noise Augmentation', min=0.0, max=1.0, default=0.0, step=0.01),
    'CLIP-G (Revision ignore prompt)': ControlNetSliderDef(name='Noise Augmentation', min=0.0, max=1.0, default=0.0, step=0.01),
    'revision_clipvision': ControlNetSliderDef(name='Noise Augmentation', min=0.0, max=1.0, default=0.0, step=0.01),
    'revision_ignore_prompt': ControlNetSliderDef(name='Noise Augmentation', min=0.0, max=1.0, default=0.0, step=0.01),
    'tile_resample': ControlNetSliderDef(name='Downsampling Rate', min=1.0, max=8.0, default=1.0, step=0.01),
    'tile_colorfix+sharp': ControlNetSliderDef(name='Variation', min=3, max=32, default=8, step=1),
    'tile_colorfix': ControlNetSliderDef(name='Variation', min=3, max=32, default=8, step=1),
    'reference_adain+attn': ControlNetSliderDef(name='Style Fidelity', min=0.0, max=1.0, default=0.5, step=0.01),
    'reference_adain': ControlNetSliderDef(name='Style Fidelity', min=0.0, max=1.0, default=0.5, step=0.01),
    'recolor_luminance': ControlNetSliderDef(name='Gamma Correction', min=0.1, max=2.0, default=1.0, step=0.001),
    'recolor_intensity': ControlNetSliderDef(name='Gamma Correction', min=0.1, max=2.0, default=1.0, step=0.001),
    'mlsd': ControlNetSliderDef(name='MLSD Value Threshold', min=0.01, max=2.0, default=0.1, step=0.01),
    'threshold': ControlNetSliderDef(name='Binarization Threshold', min=0.0, max=255.0, default=127.0, step=0.01),
    'softedge_teed': ControlNetSliderDef(name='Safe Steps', min=0, max=10, default=2, step=1),
    'softedge_anyline': ControlNetSliderDef(name='Safe Steps', min=0, max=10, default=2, step=1),
    'scribble_xdog': ControlNetSliderDef(name='XDoG Threshold', min=1.0, max=64.0, default=32.0, step=0.01),
    'normal_midas': ControlNetSliderDef(name='Normal Backgroud Threshold', min=0.0, max=1.0, default=0.4, step=0.01),
    'normal_dsine': ControlNetSliderDef(name='Fov', min=0.0, max=360.0, default=60, step=0.1),
    'mediapipe_face': ControlNetSliderDef(name='Max Faces', min=1, max=10, default=1, step=1),
    'depth_leres++': ControlNetSliderDef(name='Remove Near %', min=0.0, max=100.0, default=0.0, step=0.1),
    'depth_leres': ControlNetSliderDef(name='Remove Near %', min=0.0, max=100.0, default=0.0, step=0.1),
    'blur_gaussian': ControlNetSliderDef(name='Sigma', min=0.01, max=64.0, default=9.0, step=0.01)
}

THRESHOLD_B_PARAMETER_NAMES: dict[str, ControlNetSliderDef] = {
    'canny': ControlNetSliderDef(name='High Threshold', min=0, max=256, default=200, step=1),
    'tile_colorfix+sharp': ControlNetSliderDef(name='Sharpness', min=0.0, max=2.0, default=1.0, step=0.01),
    'mlsd': ControlNetSliderDef(name='MLSD Distance Threshold', min=0.01, max=20.0, default=0.1, step=0.01),
    'normal_dsine': ControlNetSliderDef(name='Iterations', min=1, max=20, default=5, step=1),
    'mediapipe_face': ControlNetSliderDef(name='Min Face Confidence', min=0.01, max=1.0, default=0.5, step=0.01),
    'depth_leres++': ControlNetSliderDef(name='Remove Background %', min=0.0, max=100.0, default=0.0, step=0.1),
    'depth_leres': ControlNetSliderDef(name='Remove Background %', min=0.0, max=100.0, default=0.0, step=0.1)
}
