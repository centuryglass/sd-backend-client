"""ComfyUI-specific helpers for the integration tests.

ComfyUI's API is asynchronous and workflow-based: generation calls *queue* a job and
return a ``prompt_id``; you then poll ``check_queue_entry`` until it finishes and
download the resulting image references. These helpers wrap that lifecycle and the
cache setup the workflow builders read from.
"""
import time
from typing import Optional

import pytest
from PIL import Image

from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui_webservice import AsyncTaskStatus, ComfyModelType, ComfyUiWebservice
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit

from .helpers import SD_UPSCALE_DENOISE, SD_UPSCALE_STEPS, UPSCALE_SOURCE_PROMPT, pick_upscaler_model

# ComfyUI SD1.5 defaults kept small/cheap but coherent enough to compare.
COMFY_SIZE = 256
COMFY_STEPS = 8


def build_comfy_params(checkpoint: str, sampler: str = 'euler', scheduler: str = 'karras',
                          prompt: str = 'a red apple on a wooden table') -> ComfyUIDiffusionParams:
    """Create small, valid ComfyUI generation parameter set."""
    return ComfyUIDiffusionParams(sd_model_name=checkpoint,
                                    prompt=prompt,
                                    steps=COMFY_STEPS,
                                    sampler=sampler,
                                    scheduler=scheduler,
                                    cfg_scale=7.0,
                                    batch_size=1,
                                    width=COMFY_SIZE,
                                    height=COMFY_SIZE,
                                    seed=1)



def comfy_sd_upscale_params(service: ComfyUiWebservice, checkpoint: str, use_ultimate_upscale_script: bool,
                            use_upscaler_model: bool = True,
                            tile_controlnet: Optional[ControlNetUnit] = None) -> DiffusionUpscalingParams:
    """Diffusion upscale parameters at the realistic `SD_UPSCALE_STEPS` / `SD_UPSCALE_DENOISE` settings.

    With `use_upscaler_model`, the image goes through the server's `pick_upscaler_model` choice before diffusion, and
    the test skips if the server has none.
    """
    upscaler = pick_upscaler_model(service.get_models(ComfyModelType.UPSCALING)) if use_upscaler_model else None
    if use_upscaler_model and upscaler is None:
        pytest.skip('No upscale models installed on the ComfyUI server.')
    return DiffusionUpscalingParams(upscaling_mode=upscaler or '',
                                    use_stable_diffusion_upscaling=True,
                                    use_ultimate_upscale_script=use_ultimate_upscale_script,
                                    diffusion_params=build_comfy_params(checkpoint, prompt=UPSCALE_SOURCE_PROMPT),
                                    step_count=SD_UPSCALE_STEPS,
                                    denoising_strength=SD_UPSCALE_DENOISE,
                                    tile_controlnet=tile_controlnet)


def make_comfy_mask(editable_box: tuple[int, int, int, int], size: int = COMFY_SIZE) -> Image.Image:
    """Build an inpainting mask in the shared `DiffusionParams.mask` convention: white (opaque) is changed.

    `ComfyUiWebservice.upload_mask` converts this to ComfyUI's alpha polarity, so the same mask works on WebUI.
    """
    mask = Image.new('RGBA', (size, size), (0, 0, 0, 255))
    editable = Image.new('RGBA', (editable_box[2] - editable_box[0], editable_box[3] - editable_box[1]),
                         (255, 255, 255, 255))
    mask.paste(editable, (editable_box[0], editable_box[1]))
    return mask


def find_canny_preprocessor(service: ComfyUiWebservice):
    """Return a Canny-style preprocessor object, or None if none is installed."""
    preprocessors = service.get_controlnet_preprocessors()
    exact = next((p for p in preprocessors if p.name == 'CannyEdgePreprocessor'), None)
    if exact is not None:
        return exact
    return next((p for p in preprocessors if 'canny' in p.name.lower()), None)


def find_canny_model(service: ComfyUiWebservice):
    """Return a Canny ControlNet model filename, or None if none is installed."""
    models = service.get_controlnet_models()
    return next((m for m in models if 'canny' in m.lower()), None)


def find_tile_preprocessor(service: ComfyUiWebservice):
    """Return a tile-style preprocessor object, or None if none is installed."""
    preprocessors = service.get_controlnet_preprocessors()
    return next((p for p in preprocessors if 'tile' in p.name.lower()), None)


def find_tile_model(service: ComfyUiWebservice):
    """Return a tile ControlNet model filename, or None if none is installed."""
    models = service.get_controlnet_models()
    return next((m for m in models if 'tile' in m.lower()), None)


def wait_for_comfy_images(service: ComfyUiWebservice, response, timeout: float = 240.0, poll: float = 1.0):
    """Block until a queued ComfyUI job finishes, returning its downloaded PIL images.

    Raises on submission errors, task failure, or timeout.
    """
    if response.prompt_id is None:
        raise RuntimeError(f'ComfyUI rejected the workflow: {response.node_errors or response}')
    prompt_id = response.prompt_id
    task_number = response.number
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = service.check_queue_entry(prompt_id, task_number)
        status = progress.status
        if status == AsyncTaskStatus.FINISHED:
            if progress.outputs is None or progress.outputs.images is None:
                images = []
            else:
                images = progress.outputs.images
            return service.download_images(images)
        if status == AsyncTaskStatus.FAILED:
            raise RuntimeError(f'ComfyUI task {prompt_id} failed during execution: {progress.error}')
        time.sleep(poll)
    raise TimeoutError(f'ComfyUI task {prompt_id} did not finish within {timeout}s')
