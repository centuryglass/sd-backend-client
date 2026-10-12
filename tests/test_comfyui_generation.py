"""Generation integration tests: real diffusion round-trips against ComfyUI.

ComfyUI jobs are asynchronous — each call queues a workflow and returns a ``prompt_id``
that we poll to completion, then download the images (see ``comfy_helpers``). Opt-in
(needs a checkpoint + GPU): run with ``--run-generation`` / ``RUN_SD_GENERATION=1``.
Results are saved under the output dir for visual fidelity inspection.
"""
import pytest
from PIL import Image

from sd_backend_client.api.comfyui_webservice import ComfyModelType
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import PreprocessorParams
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit

from .comfy_helpers import (COMFY_SIZE, build_comfy_params, find_tile_model, find_tile_preprocessor,
                            make_comfy_mask, wait_for_comfy_images)
from .helpers import (BASIC_UPSCALE_SOURCE_SIZE, UPSCALE_SOURCE_PROMPT, UPSCALE_SOURCE_SIZE, images_differ,
                      make_structured_image, region_mean_diff, save_output, shrink)

pytestmark = [pytest.mark.integration, pytest.mark.generation]


def _assert_valid_image(image, expected_size: int = COMFY_SIZE):
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGBA'  # decoded images are normalized to RGBA
    assert abs(image.width - expected_size) <= 8
    assert abs(image.height - expected_size) <= 8


def test_txt2img_produces_image(comfy_service, comfy_checkpoint, output_dir):
    params = build_comfy_params(comfy_checkpoint)
    response = comfy_service.txt2img(params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) == 1
    _assert_valid_image(images[0])
    save_output(output_dir, 'comfy_txt2img', images[0])


def test_txt2img_seed_is_reported(comfy_service, comfy_checkpoint):
    params = build_comfy_params(comfy_checkpoint)
    params.seed = 98765
    response = comfy_service.txt2img(params)
    # The client echoes back the seed it actually queued.
    assert response.seed == 98765
    wait_for_comfy_images(comfy_service, response)  # drain the job so it doesn't linger


def test_img2img_round_trip(comfy_service, comfy_checkpoint, output_dir):
    params = build_comfy_params(comfy_checkpoint, prompt='a vivid landscape painting')
    params.denoising_strength = 0.75  # enough to visibly transform the textured source
    source = make_structured_image(COMFY_SIZE, COMFY_SIZE)
    params.init_images = [source]
    params.seed = 7
    response = comfy_service.img2img(params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    result = images[0]
    _assert_valid_image(result)
    save_output(output_dir, 'comfy_img2img_source', source)
    save_output(output_dir, 'comfy_img2img_result', result)
    assert images_differ(source, result) >= 10.0, 'img2img returned an image nearly identical to its source'


def test_inpaint_respects_mask(comfy_service, comfy_checkpoint, output_dir):
    """Inpainting must change the masked (editable) region while preserving the rest.

    Uses a textured source and full denoising so the change is unmistakable, and a mask
    whose white center is the editable region (the shared `DiffusionParams.mask` convention).
    The assertion is region-aware: the center should change far more than the border,
    which proves the mask is actually being followed rather than the whole image being
    regenerated (or nothing happening at all).
    """
    params = build_comfy_params(comfy_checkpoint, prompt='a photograph of a flower')
    params.denoising_strength = 1.0  # flat/low denoise barely changes inpainted pixels

    source = make_structured_image(COMFY_SIZE, COMFY_SIZE)
    center_box = (COMFY_SIZE // 4, COMFY_SIZE // 4, COMFY_SIZE * 3 // 4, COMFY_SIZE * 3 // 4)
    border_box = (0, 0, COMFY_SIZE, COMFY_SIZE // 8)
    mask = make_comfy_mask(center_box)
    params.init_images = [source]
    params.mask = mask
    params.seed = 7

    response = comfy_service.inpaint(params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    result = images[0]
    _assert_valid_image(result)
    save_output(output_dir, 'comfy_inpaint_source', source)
    save_output(output_dir, 'comfy_inpaint_mask', mask)
    save_output(output_dir, 'comfy_inpaint_result', result)

    edited_diff = region_mean_diff(source, result, center_box)
    preserved_diff = region_mean_diff(source, result, border_box)
    assert edited_diff >= 20.0, f'masked region barely changed (diff {edited_diff:.2f}); inpainting may be a no-op'
    assert edited_diff >= 3 * preserved_diff, (
        f'inpainting did not respect the mask: edited-region diff {edited_diff:.2f} is not clearly greater than '
        f'preserved-region diff {preserved_diff:.2f} (VAE round-trip noise floor). '
        f'Inspect comfy_inpaint_source.png / _mask.png / _result.png in {output_dir}.'
    )


def _submit_inpaint(comfy_service, comfy_checkpoint, **overrides):
    """Run `submit_inpaint` on the standard textured source with a white center square, returning (source, image)."""
    params = build_comfy_params(comfy_checkpoint, prompt='a photograph of a flower')
    params.denoising_strength = 1.0
    params.seed = 7
    params.init_images = [make_structured_image(COMFY_SIZE, COMFY_SIZE)]
    params.mask = make_comfy_mask((COMFY_SIZE // 4, COMFY_SIZE // 4, COMFY_SIZE * 3 // 4, COMFY_SIZE * 3 // 4))
    for name, value in overrides.items():
        setattr(params, name, value)
    result = comfy_service.submit_inpaint(params).wait(timeout=240)
    return params.init_images[0], result.images[0]


def test_submit_inpaint_pastes_the_result_onto_the_source(comfy_service, comfy_checkpoint, output_dir):
    """`submit_inpaint` returns the source unchanged away from the mask, unlike raw `inpaint()`.

    Raw `inpaint()` output has been through the VAE, so even preserved pixels drift. `submit_inpaint` composites the
    result onto the source, so a crisp-mask run must match the source outside the mask.
    """
    source, result = _submit_inpaint(comfy_service, comfy_checkpoint, mask_blur=0)
    save_output(output_dir, 'comfy_submit_inpaint_source', source)
    save_output(output_dir, 'comfy_submit_inpaint_result', result)
    _assert_valid_image(result)
    border_box = (0, 0, COMFY_SIZE, COMFY_SIZE // 8)
    center_box = (COMFY_SIZE // 4 + 8, COMFY_SIZE // 4 + 8, COMFY_SIZE * 3 // 4 - 8, COMFY_SIZE * 3 // 4 - 8)
    assert region_mean_diff(source, result, border_box) <= 1.0, 'the preserved region was not pasted back unchanged'
    assert region_mean_diff(source, result, center_box) >= 20.0, 'the masked region barely changed'


def test_mask_blur_feathers_the_inpainting_boundary(comfy_service, comfy_checkpoint, output_dir):
    """A large `mask_blur` blends generated pixels into the source just outside the mask; `mask_blur=0` does not."""
    source, crisp = _submit_inpaint(comfy_service, comfy_checkpoint, mask_blur=0)
    _, feathered = _submit_inpaint(comfy_service, comfy_checkpoint, mask_blur=12)
    save_output(output_dir, 'comfy_mask_blur_0', crisp)
    save_output(output_dir, 'comfy_mask_blur_12', feathered)
    # A strip just outside the mask's left edge (x=64), clear of the corners.
    strip = (COMFY_SIZE // 4 - 8, COMFY_SIZE * 3 // 8, COMFY_SIZE // 4 - 1, COMFY_SIZE * 5 // 8)
    crisp_diff = region_mean_diff(source, crisp, strip)
    feathered_diff = region_mean_diff(source, feathered, strip)
    assert crisp_diff <= 1.0, f'a crisp mask changed pixels outside it (diff {crisp_diff:.2f})'
    assert feathered_diff >= 2.0 and feathered_diff > crisp_diff, \
        f'mask_blur=12 left the area outside the mask unblended (diff {feathered_diff:.2f})'


def test_upscale(comfy_service, comfy_upscale_source, output_dir):
    upscale_models = comfy_service.get_models(ComfyModelType.UPSCALING)
    if not upscale_models:
        pytest.skip('No upscale models installed on the ComfyUI server.')
    # Basic (non-SD) upscaling path, driven by an explicit upscaler model.
    upscale_params = DiffusionUpscalingParams(upscaling_mode=upscale_models[0])

    source = shrink(comfy_upscale_source, BASIC_UPSCALE_SOURCE_SIZE)
    response = comfy_service.upscale(source, UPSCALE_SOURCE_SIZE, UPSCALE_SOURCE_SIZE, upscale_params)
    images = wait_for_comfy_images(comfy_service, response)
    assert len(images) >= 1
    result = images[0]
    _assert_valid_image(result, expected_size=UPSCALE_SOURCE_SIZE)
    save_output(output_dir, 'comfy_upscale_source', source)
    save_output(output_dir, 'comfy_upscale_result', result)


def test_upscale_with_ultimate_sd_upscale(comfy_service, comfy_checkpoint, comfy_upscale_source, output_dir):
    """Ultimate SD Upscale: a tiled diffusion pass refines a 4x upscale, with an optional tile ControlNet unit.

    512x512 -> 2048x2048 so the tiled pass actually spans multiple tiles at the default 512px tile size.
    """
    if not comfy_service.get_capabilities().ultimate_upscale:
        pytest.skip('Ultimate SD Upscale is not available on this ComfyUI server.')

    tile_controlnet = None
    tile_model = find_tile_model(comfy_service)
    tile_preprocessor = find_tile_preprocessor(comfy_service)
    if tile_model and tile_preprocessor:
        tile_controlnet = ControlNetUnit(model=ControlNetModel(tile_model),
                                         preprocessor=PreprocessorParams(typedef=tile_preprocessor))

    upscale_params = DiffusionUpscalingParams(use_stable_diffusion_upscaling=True,
                                              use_ultimate_upscale_script=True,
                                              diffusion_params=build_comfy_params(comfy_checkpoint,
                                                                                  prompt=UPSCALE_SOURCE_PROMPT),
                                              step_count=8,
                                              tile_controlnet=tile_controlnet)
    source = comfy_upscale_source
    response = comfy_service.upscale(source, 2048, 2048, upscale_params)
    images = wait_for_comfy_images(comfy_service, response, timeout=300.0)
    assert len(images) == 1
    result = images[0]
    _assert_valid_image(result, expected_size=2048)
    save_output(output_dir, 'comfy_upscale_sd_source', source)
    save_output(output_dir, 'comfy_upscale_sd_result', result)


def test_upscale_fallback_tiled_vae(comfy_service, comfy_checkpoint, comfy_upscale_source, output_dir):
    """Without the Ultimate SD Upscale node, the diffusion upscale path falls back to tiled VAE encode/decode."""
    upscale_params = DiffusionUpscalingParams(use_stable_diffusion_upscaling=True,
                                              use_ultimate_upscale_script=False,
                                              diffusion_params=build_comfy_params(comfy_checkpoint,
                                                                                  prompt=UPSCALE_SOURCE_PROMPT),
                                              step_count=8)
    source = comfy_upscale_source
    response = comfy_service.upscale(source, 2048, 2048, upscale_params)
    images = wait_for_comfy_images(comfy_service, response, timeout=300.0)
    assert len(images) == 1
    result = images[0]
    _assert_valid_image(result, expected_size=2048)
    save_output(output_dir, 'comfy_upscale_fallback_result', result)


def test_interrupt_is_accepted(comfy_service):
    # With no active job this is a no-op, but the endpoint should respond without raising.
    comfy_service.interrupt()
