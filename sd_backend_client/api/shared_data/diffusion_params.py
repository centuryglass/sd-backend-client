"""Shared image diffusion operation parameter set.

Each field here has one meaning on both backends. Fields only one backend supports live on its subclass
(`DiffusionRequestBody` for WebUI, `ComfyUIDiffusionParams` for ComfyUI).
"""
import logging
from typing import Any, Optional

from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STRENGTH = 0.75
"""Denoising strength both backends use for img2img and inpainting when `DiffusionParams.denoising_strength` is None.

It matches WebUI's own img2img default.
"""


class DiffusionParams(BaseModel):
    """Request body format for image generation (all types)"""
    # use_enum_values: store/serialize enum fields (e.g. WebUI ResizeMode, InpaintFillOption) as their underlying
    # values, so model_dump() produces the JSON-serializable ints the APIs expect. validate_default ensures enum
    # *defaults* (e.g. inpainting_fill=InpaintFillOption.ORIGINAL) are converted too, not just explicitly-set values.
    # validate_assignment applies the field constraints to attribute assignment too, so `params.steps = 0` raises.
    # extra='forbid' makes a misspelled field raise instead of being dropped. Converting between subclasses must pass
    # only fields the target defines (see `DiffusionRequestBody.from_params`).
    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True, validate_default=True,
                              validate_assignment=True, extra='forbid')

    ### Basic image generation:
    sd_model_name: str = ''
    """Stable Diffusion model name."""

    sampler: str = Field(default='euler_ancestral', validation_alias=AliasChoices('sampler', 'sampler_name'))
    """Sampling algorithm that iteratively denoises the image, as a shared name from `sampler_names`.

    Shared names are ComfyUI's KSampler names (`'euler_ancestral'`, `'dpmpp_2m'`); each backend translates them (see
    `sampler_names.SAMPLER_WEBUI_NAMES`). WebUI names (`'Euler a'`) are accepted on both backends. Other names are
    sent unchanged, so they work only on a backend that defines them. The constructor also accepts the field as
    `sampler_name`, the WebUI request key.
    """

    scheduler: str = 'normal'
    """Noise schedule that sets the size of each denoising step, as a shared name from `sampler_names`.

    Shared names are ComfyUI's KSampler names (`'normal'`, `'karras'`); see `sampler_names.SCHEDULER_WEBUI_NAMES` for
    the WebUI versions that accept them.
    """

    batch_size: int = Field(default=1, ge=1)
    """Number of images generated in parallel by one request.

    ComfyUI accepts 1 to `diffusion_workflow_builder.MAX_BATCH_SIZE`. WebUI can also repeat the batch with
    `DiffusionRequestBody.n_iter`.
    """

    steps: int = Field(default=30, ge=1)
    """Number of denoising steps per image generation."""

    cfg_scale: float = Field(default=7.0, ge=0.0)  # guidance scale
    """
    How strongly generation follows the text prompt. Typical useful range is model-specific, usually 6-10 for typical
    models, 1-2 for LCM and Turbo models.
    """

    width: int = Field(default=512, gt=0)
    """Generated image width in pixels.

    Stable Diffusion works on a latent grid 1/8 the image size, so pass a multiple of 8 for a predictable output size.
    Other sizes are accepted, and each backend rounds them its own way.

    For img2img and inpainting, the source image and mask are resized to `width` x `height` before generation. On WebUI
    `DiffusionRequestBody.resize_mode` chooses how; ComfyUI always stretches, like WebUI's default `JUST_RESIZE`.
    """

    height: int = Field(default=512, gt=0)
    """Generated image height in pixels. See `width` for size rounding and how img2img sources are resized."""


    ### Prompt:
    prompt: str = ''
    """Text description guiding what the generated image should contain."""

    negative_prompt: str = ''
    """Text description guiding what the generated image should avoid containing."""

    seed: int = -1
    """Random seed controlling generation. Any value less than zero selects a new random seed."""


    ### Img2img and inpainting only:
    init_images: Optional[list[Image.Image]] = None
    """
    List of Pil images for inpainting or image to image.
    """

    denoising_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Amount the source image changes in img2img and inpainting, from 0.0 (no change) to 1.0 (complete replacement).

    None selects `DEFAULT_DENOISING_STRENGTH`. Text-to-image generation ignores it on ComfyUI and uses it only for
    the high-res fix pass on WebUI.
    """


    ### Inpainting only:
    mask: Optional[Image.Image] = None
    """Inpainting mask image on both backends: opaque or white pixels mark the region to change.

    Either an alpha mask or an opaque grayscale mask is accepted (see `mask_to_grayscale`). Each backend converts to
    its own wire format, so callers never invert the mask for ComfyUI.
    """

    mask_blur: Optional[int] = Field(default=None, ge=0)
    """Blur applied to the inpainting mask's edges in both directions, as a Gaussian standard deviation in pixels
    (0 disables it).

    The blurred edge extends about 2.5 times this value from the original boundary. When set, `mask_blur_x` and
    `mask_blur_y` are ignored. When all three are None, both backends use `MASK_BLUR_DEFAULT`. On ComfyUI the client
    blurs the mask before upload and returns the blurred mask as `GenerationResult.mask`.
    """

    mask_blur_x: Optional[int] = Field(default=None, ge=0)
    """Horizontal standard deviation (in pixels) of the mask edge blur. Ignored when `mask_blur` is set.

    When only one of `mask_blur_x` and `mask_blur_y` is set, the other axis uses `MASK_BLUR_DEFAULT`.
    """

    mask_blur_y: Optional[int] = Field(default=None, ge=0)
    """Vertical standard deviation (in pixels) of the mask edge blur. Ignored when `mask_blur` is set.

    When only one of `mask_blur_x` and `mask_blur_y` is set, the other axis uses `MASK_BLUR_DEFAULT`.
    """

    controlnet_units: list[ControlNetUnit] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert the request body to a dict, removing unused optional parameters."""
        return self.model_dump(exclude_none=True)
