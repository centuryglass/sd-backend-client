"""Live soft inpainting tests on both backends, driven through `submit_inpaint`.

Each test inpaints an ellipse in the middle of a 512x512 source at 0.85 denoising, through a mask blurred with sigma
32, once plainly and once with `soft_inpainting`. Soft inpainting must change the output, or it silently did nothing,
as the ControlNet generation tests check for control units. On the WebUI the result's infotext also proves the
Soft Inpainting script ran. How much the output changes varies a lot by checkpoint, so the threshold is low. Whether
the change blends better is left to the saved PNGs (see `tests/review.html`).
"""
from typing import Any

import pytest
from PIL import Image, ImageDraw

from sd_backend_client.api.shared_data.generation_handle import GenerationResult
from sd_backend_client.api.webui.response_formats import GenerationInfoData

from .helpers import images_differ, region_mean_diff, save_output
from .inpaint_helpers import inpaint_params

pytestmark = [pytest.mark.integration, pytest.mark.generation]

SIZE = 512
STEPS = 20
MASK_BOX = (160, 160, 352, 352)
MASK_BLUR = 32                          # the blurred edge reaches about 80 px past MASK_BOX
FAR_CORNER = (0, 0, 64, 64)             # outside the blurred edge
PROMPT = 'a bowl of red apples, detailed still life photograph'
MIN_SOFT_INPAINTING_DIFF = 0.5
"""Smallest mean pixel difference between plain and soft inpainting that counts as soft inpainting having run."""


def _mask() -> Image.Image:
    mask = Image.new('L', (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).ellipse(MASK_BOX, fill=255)
    return mask


def _service(backend: str, request: pytest.FixtureRequest) -> Any:
    """The backend's client, skipping the test when the server lacks soft inpainting."""
    service = request.getfixturevalue('comfy_service' if backend == 'comfyui' else 'service')
    if not service.get_capabilities().soft_inpainting:
        pytest.skip(f'The {backend} server has no soft inpainting (BackendCapabilities.soft_inpainting).')
    return service


def _inpaint(backend: str, request: pytest.FixtureRequest, soft_inpainting: bool) -> tuple[Image.Image,
                                                                                         GenerationResult]:
    params = inpaint_params(backend, request, PROMPT, SIZE, STEPS, 21, _mask())
    params.mask_blur = MASK_BLUR
    params.soft_inpainting = soft_inpainting
    result = _service(backend, request).submit_inpaint(params).wait(timeout=300)
    return params.init_images[0], result


@pytest.mark.parametrize('backend', ['webui', 'comfyui'])
def test_soft_inpainting_changes_the_result(backend, request, output_dir):
    """Soft inpainting gives a different result from plain inpainting with the same seed, and keeps the far source."""
    source, plain = _inpaint(backend, request, soft_inpainting=False)
    _, soft = _inpaint(backend, request, soft_inpainting=True)
    for name, image in (('source', source), ('mask', _mask()), ('plain', plain.images[0]), ('soft', soft.images[0])):
        save_output(output_dir, f'{backend}_soft_inpainting_{name}', image)

    assert soft.images[0].size == source.size
    difference = images_differ(plain.images[0], soft.images[0])
    assert difference >= MIN_SOFT_INPAINTING_DIFF, \
        f'soft inpainting changed the result by only {difference:.2f}; it may have silently done nothing'
    assert region_mean_diff(source, soft.images[0], FAR_CORNER) <= 2.0, \
        'pixels far outside the mask changed; the result was not composited onto the source'
    if backend == 'webui':
        assert isinstance(soft.raw_info, GenerationInfoData)
        assert 'Soft inpainting enabled' in soft.raw_info.extra_generation_params
        assert isinstance(plain.raw_info, GenerationInfoData)
        assert 'Soft inpainting enabled' not in plain.raw_info.extra_generation_params
