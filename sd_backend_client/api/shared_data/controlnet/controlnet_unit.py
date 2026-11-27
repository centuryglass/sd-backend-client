"""Represents everything needed to define a ControlNet layer, including a preprocessor module and/or a ControlNet model
 (usually both, but not always), and all associated settings.  Values can be serialized and deserialized in both WebUI
 and ComfyUI formats."""
from typing import Optional, Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import PreprocessorParams

class ControlNetUnit(BaseModel):
    """Represents everything needed to define a ControlNet layer, including a preprocessor module and/or a ControlNet
       model (usually both, but not always), and all associated settings. Values can be serialized and deserialized in
       both WebUI and ComfyUI formats.

       Pass `image` as a Pillow image from Python. This field is omitted from the JSON schema.
    """
    # use_attribute_docstrings copies each field's docstring into its FieldInfo.description.
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra='forbid',
                              use_attribute_docstrings=True)

    image: SkipJsonSchema[Optional[Image.Image]] = None
    """Control image the preprocessor reads, or the model reads directly when there is no preprocessor.

    None uses the generation's source image (`DiffusionParams.init_images`), as the WebUI ControlNet extension does.
    On ComfyUI, a unit with no image raises `ValueError` when there is no source image either.
    """

    preprocessor: Optional[PreprocessorParams] = None
    """Preprocessor that turns `image` into a control map, with its parameter values.

    None passes `image` to the model unchanged, for an image that is already a control map.
    """

    model: Optional[ControlNetModel] = None
    """ControlNet model that applies the control map to generation, as listed by `Backend.list_controlnet_models`.

    None works only on WebUI, with a model-free preprocessor (see `ControlNetPreprocessor.model_free`). ComfyUI skips
    and logs a unit with no model.
    """

    control_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    """ControlNet strength, from 0.0 to 2.0."""

    control_start: float = Field(default=0.0, ge=0.0, le=1.0)
    """The point where the ControlNet is enabled, as a fraction of total diffusion step count.

    Must not exceed `control_end`. Assignment is checked against the current `control_end`, so when moving both
    bounds, assign them in an order that keeps start <= end at each step.
    """

    control_end: float = Field(default=1.0, ge=0.0, le=1.0)
    """The point where the ControlNet is disabled, as a fraction of total diffusion step count.

    Must not be less than `control_start`.
    """

    low_vram: bool = False
    """Whether the WebUI ControlNet extension trades speed for lower VRAM use on this unit. ComfyUI ignores it."""

    pixel_perfect: bool = True
    """Whether the WebUI ControlNet extension picks the preprocessor resolution from the control image and output sizes.

    When on, it overrides the preprocessor's resolution parameter (`PREPROCESSOR_RES_PARAM_KEY`). ComfyUI ignores it.
    """

    @model_validator(mode='after')
    def _check_control_range(self) -> Self:
        if self.control_start > self.control_end:
            raise ValueError(f'control_start ({self.control_start}) must not exceed control_end ({self.control_end})')
        return self
