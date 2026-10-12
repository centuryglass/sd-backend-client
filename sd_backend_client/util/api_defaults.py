"""Defaults shared by the backend-agnostic parameter models."""

GENERATION_SIZE_DEFAULT = 512

UPSCALING_DENOISING_STRENGTH_DEFAULT = 0.3
UPSCALING_STEP_COUNT_DEFAULT = 30

MASK_BLUR_DEFAULT = 4
"""Gaussian standard deviation (in pixels) of the inpainting mask blur when a request leaves `mask_blur` unset. Matches
the WebUI's server default."""

INPAINT_FULL_RES_PADDING_DEFAULT = 32
"""Padding (in pixels) around the masked region when `inpaint_full_res_padding` is unset. Matches the WebUI's server
default."""
