"""Defaults shared by the backend-agnostic parameter models."""

GENERATION_SIZE_DEFAULT = 512

UPSCALING_DENOISING_STRENGTH_DEFAULT = 0.3
UPSCALING_STEP_COUNT_DEFAULT = 30

MASK_BLUR_DEFAULT = 4
"""Gaussian standard deviation (in pixels) of the inpainting mask blur when a request leaves `mask_blur` unset. Matches
the WebUI's server default."""
