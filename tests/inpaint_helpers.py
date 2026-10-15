"""Shared setup for the live inpainting tests that run the same job on both backends."""
import pytest
from PIL import Image

from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

from .comfy_helpers import build_comfy_params
from .helpers import make_structured_image


def inpaint_params(backend: str, request: pytest.FixtureRequest, prompt: str, size: int, steps: int, seed: int,
                   mask: Image.Image) -> DiffusionParams:
    """Inpainting params for `backend` at 0.85 denoising, with a `size` x `size` structured source and `mask`.

    `backend` is 'comfyui' or 'webui'. ComfyUI params use the `comfy_checkpoint` fixture's checkpoint.
    """
    if backend == 'comfyui':
        params: DiffusionParams = build_comfy_params(request.getfixturevalue('comfy_checkpoint'), prompt=prompt)
    else:
        params = DiffusionRequestBody(prompt=prompt, save_images=False, send_images=True)
    params.width = params.height = size
    params.steps = steps
    params.seed = seed
    params.denoising_strength = 0.85
    params.init_images = [make_structured_image(size, size)]
    params.mask = mask
    return params
