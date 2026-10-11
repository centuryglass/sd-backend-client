"""Typedefs for WebUI API data."""
import logging
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from sd_backend_client.api.shared_data.diffusion_params import DEFAULT_DENOISING_STRENGTH, DiffusionParams
from sd_backend_client.api.shared_data.sampler_names import webui_sampler_name, webui_scheduler_name
from sd_backend_client.api.webui.controlnet_webui_constants import CONTROLNET_SCRIPT_KEY, ControlNetUnitDict
from sd_backend_client.api.webui.script_info_types import ScriptRequestData
from sd_backend_client.util.visual.image_utils import image_to_base64, mask_to_grayscale

logger = logging.getLogger(__name__)

class ResizeMode(Enum):
    """Controls how source images are resized when source resolution doesn't match generated image size."""
    JUST_RESIZE = 0
    CROP_AND_RESIZE = 1
    RESIZE_AND_FILL = 2
    RESIZE_LATENT_UPSCALE = 3

class InpaintFillOption(Enum):
    """Controls how source image data is processed before inpainting. (ORIGINAL is almost always the right choice.)"""
    FILL = 0
    ORIGINAL = 1
    LATENT_NOISE = 2
    LATENT_NOTHING = 3


class DiffusionRequestBody(DiffusionParams):
    """Request body format for WebUI image generation (all types)"""

    ### Basic image generation:
    n_iter: int = Field(default=1, ge=1)  # number of batches
    """Number of image batches to generate."""


    resize_mode: Optional[ResizeMode] = None
    """Controls how source images are resized when source resolution doesn't match generated image size.

    None (omitted) selects the server default, `JUST_RESIZE`, which matches ComfyUI's resizing.
    """

    image_cfg_scale: Optional[float] = None
    """How strongly generation follows the image prompt (InstructPix2Pix models only)."""

    include_init_images: Optional[bool] = None
    """Whether init_images should be sent back with the response."""


    ### Inpainting only:
    inpainting_fill: Optional[InpaintFillOption] = InpaintFillOption.ORIGINAL
    """Controls how source image data is processed before inpainting. ORIGINAL is almost always the right choice.
   """

    inpaint_full_res: Optional[bool] = None
    """Whether inpainting renders only the masked region at full resolution instead of the whole image"""

    inpaint_full_res_padding: Optional[int] = None
    """Padding (in pixels) added around the masked region when inpaint_full_res is enabled."""

    inpainting_mask_invert: Optional[int] = None  # 0=don't invert, 1=invert
    """Whether to invert the inpainting mask (0=don't invert, 1=invert)."""

    mask_round: Optional[bool] = None
    """Whether mask values are rounded to hard edges instead of being treated as a gradient."""

    initial_noise_multiplier: Optional[float] = None
    """Scaling factor applied to the extra noise added to masked areas before inpainting."""


    ### Prompt:
    styles: Optional[list[str]] = None
    """Names of saved prompt styles to apply to the prompt and negative prompt."""

    subseed: Optional[int] = None
    """Secondary "variation" seed, blended with the main seed to add variance across generations."""

    subseed_strength: Optional[float] = None
    """How strongly the subseed influences generation, from 0.0 (main seed only) to 1.0 (subseed only)."""


    ### Seed resize options: for getting similar results at different resolutions:
    seed_resize_from_h: Optional[int] = None
    """Source height used to reproduce a seed's results at a different resolution."""

    seed_resize_from_w: Optional[int] = None
    """Source width used to reproduce a seed's results at a different resolution."""


    ### minor extra features:
    restore_faces: bool = False
    """Whether to run a face restoration model as a post-processing step."""

    tiling: bool = False
    """Whether to generate an image that tiles seamlessly when repeated."""

    refiner_checkpoint: Optional[str] = None
    """Name of the refiner model used for the final generation steps."""

    refiner_switch_at: Optional[int] = None  # step count
    """Step count at which generation switches from the base model to the refiner model."""


    ### settings and misc. server behavior
    infotext: Optional[str] = None
    """Generation metadata string. If provided, it overrides other parameters."""

    override_settings: Optional[dict[str, Any]] = None
    """Server settings to temporarily override for this request.

    `to_dict` adds `sd_model_checkpoint` from `sd_model_name` unless this dict already sets it. The server matches the
    name against each checkpoint's file name (relative to its models directory, e.g. `model.safetensors`), its title
    (`model.safetensors [abc1234567]`), its stem or its hash. A name that matches nothing is dropped without an error,
    and the request uses the server's configured checkpoint.
    """

    override_settings_restore_afterwards: Optional[bool] = None
    """Whether overridden settings are restored to their previous values after this request.

    The server treats `None` (omitted) as `True`. Restoring resets the server's checkpoint setting but leaves the
    requested checkpoint loaded, so later requests for the same checkpoint don't reload it.
    """

    do_not_save_samples: Optional[bool] = None
    """Whether the server should skip saving individual generated images to disk."""

    do_not_save_grid: Optional[bool] = None
    """Whether the server should skip saving the combined image grid to disk."""

    disable_extra_networks: Optional[bool] = None
    """Whether to disable extra networks such as LoRAs and Hypernetworks."""

    send_images: bool = True
    """Whether generated images are included in the response."""

    save_images: bool = False
    """Whether generated images are saved to disk on the server."""

    comments: Optional[dict[str, Any]] = None
    """Arbitrary extra information to embed in the generated image metadata."""

    force_task_id: Optional[str] = None
    """Assigns this ID to the job instead of using a randomly generated one."""


    ### High-res fix:
    enable_hr: Optional[bool] = None
    """Whether to enable the high-resolution fix, a second upscaling pass after initial generation."""

    firstpass_image: Optional[str] = None
    """Alternate initial image (base64) used for the first pass of a high-res fix generation."""

    firstphase_width: Optional[int] = None
    """Width of the first (pre-upscale) high-res fix pass."""

    firstphase_height: Optional[int] = None
    """Height of the first (pre-upscale) high-res fix pass."""

    hr_scale: Optional[float] = None
    """Factor by which the image is upscaled during the high-res fix pass."""

    hr_upscaler: Optional[str] = None
    """Name of the upscaler model used during the high-res fix pass."""

    hr_second_pass_steps: Optional[int] = None
    """Number of denoising steps in the high-res fix second pass."""

    hr_resize_x: Optional[int] = None
    """Target width for the high-res fix pass, as an alternative to hr_scale."""

    hr_resize_y: Optional[int] = None
    """Target height for the high-res fix pass, as an alternative to hr_scale."""

    hr_sampler_name: Optional[str] = None
    """Sampling algorithm used for the high-res fix second pass."""

    hr_prompt: Optional[str] = None
    """Alternate prompt used for the high-res fix second pass."""

    hr_negative_prompt: Optional[str] = None
    """Alternate negative prompt used for the high-res fix second pass."""


    ### custom scripts:
    script_name: Optional[str] = None
    """Name of a custom script to run for this generation."""

    script_args: Optional[list[Any]] = None
    """Positional arguments passed to the script named by script_name."""

    alwayson_scripts: Optional[dict[str, ScriptRequestData]] = None
    """Data for "always-on" scripts (e.g. ControlNet) applied to the generation.

    `to_dict` replaces any `alwayson_scripts['controlNet']` entry with args built from `controlnet_units`, so set
    ControlNet through `controlnet_units`. `to_dict` does not modify this field.
    """


    ### Karras(?) sampler parameters (probably don't need to use these)
    eta: Optional[float] = None
    """Amount of noise added at each sampling step for ancestral/stochastic samplers."""

    s_min_uncond: Optional[float] = None
    """Sigma threshold below which the negative prompt is skipped to speed up sampling."""

    s_churn: Optional[float] = None
    """Amount of extra stochastic noise mixed in during sampling."""

    s_tmax: Optional[float] = None
    """Maximum sigma value at which s_churn noise is applied."""

    s_tmin: Optional[float] = None
    """Minimum sigma value at which s_churn noise is applied."""

    s_noise: Optional[float] = None
    """Scaling factor for the extra noise added by s_churn."""

    # Probably deprecated, present for compatibility reasons:
    sampler_index: Optional[str] = None
    """Deprecated WebUI alias for the `sampler_name` request key. The server ignores it, since `to_dict` always sends
    `sampler_name` from `sampler`."""

    @classmethod
    def from_params(cls, params: Optional[DiffusionParams] = None) -> 'DiffusionRequestBody':
        """Return a deep copy of `params` as a request body, or a default body when `params` is None.

        A `DiffusionRequestBody` keeps its WebUI-only fields. Any other `DiffusionParams` subclass keeps only the
        shared fields, so fields specific to another backend are dropped.
        """
        if params is None:
            return cls()
        copied = params.model_copy(deep=True)
        if isinstance(copied, cls):
            return copied
        return cls(**{name: getattr(copied, name) for name in DiffusionParams.model_fields.keys()})

    def to_dict(self) -> dict[str, Any]:
        """Convert the request body to a dict, removing unused optional parameters."""
        data = super().to_dict()
        data['sampler_name'] = webui_sampler_name(data.pop('sampler'))
        data['scheduler'] = webui_scheduler_name(data.pop('scheduler'))
        if self.init_images and self.denoising_strength is None:
            data['denoising_strength'] = DEFAULT_DENOISING_STRENGTH
        # The WebUI request schema has no sd_model_name field; the checkpoint is selected through override_settings.
        sd_model_name = data.pop('sd_model_name', '')
        if sd_model_name:
            data.setdefault('override_settings', {}).setdefault('sd_model_checkpoint', sd_model_name)
        # Build ControlNet request parameters into the output only, so self is left unchanged.
        scripts = data.setdefault('alwayson_scripts', {})
        if CONTROLNET_SCRIPT_KEY in scripts:
            scripts[CONTROLNET_SCRIPT_KEY] = {'args': []}  # controlnet_units replaces any caller-supplied entry
        for control_unit in self.controlnet_units:
            control_unit_dict = ControlNetUnitDict.from_unit(control_unit)
            scripts.setdefault(CONTROLNET_SCRIPT_KEY, {'args': []})
            # exclude_none: omit unset optionals (processor_res / threshold_a / threshold_b) rather than sending them
            # as null, which the ControlNet extension chokes on (e.g. `unit.processor_res < 0`). The server defaults them.
            scripts[CONTROLNET_SCRIPT_KEY]['args'].append(control_unit_dict.model_dump(exclude_none=True))
        if 'controlnet_units' in data:
            del data['controlnet_units']  # Include only under scripts
        # Ensure images are prefixed base64:
        if 'init_images' in data:
            images = data['init_images']
            assert isinstance(images, list)
            for i in range(len(images)):
                images[i] = image_to_base64(images[i], include_prefix=True)
        if 'mask' in data:
            data['mask'] = image_to_base64(mask_to_grayscale(data['mask']), include_prefix=True)
        return data
