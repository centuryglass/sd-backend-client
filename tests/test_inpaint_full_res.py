"""Live "inpaint only masked" (`inpaint_full_res`) tests on both backends, driven through `submit_inpaint`.

Each test inpaints a ~14% patch of a 512x512 source at 0.6 denoising with a prompt full of fine detail, once over the
whole image and once with `inpaint_full_res`. The full-res run crops a 256x256 region around the mask and renders it
at 512x512, so the patch gets about twice the pixels and should come out visibly finer. The saved PNGs make that easy
to confirm by eye (see `tests/README.md`); the assertions check what is deterministic: pixels outside the mask are
untouched, the patch changed, and the full-res patch carries more fine detail than the whole-image one.
"""
import pytest
from PIL import Image, ImageFilter, ImageStat

from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

from .comfy_helpers import build_comfy_params, make_comfy_mask
from .helpers import images_differ, make_structured_image, region_mean_diff, save_output

pytestmark = [pytest.mark.integration, pytest.mark.generation]

SIZE = 512
STEPS = 20
MASK_BOX = (240, 96, 432, 288)         # 192x192 = 14% of the image, off-center so the crop has to shift
PADDING = 32                           # crop is then 256x256, rendered at 512x512
FAR_POINT = (40, 470)                  # context far from the mask, to force a larger crop
PROMPT = ('an ornate clockwork mechanism, intricate brass gears, tiny screws, fine engraved filigree, '
          'highly detailed macro photograph')
OUTSIDE_MARGIN = 24                    # the blurred mask edge reaches about 10 px past MASK_BOX


def _params(backend: str, request: pytest.FixtureRequest) -> DiffusionParams:
    """Shared inpainting params for `backend`, with the source and mask attached."""
    if backend == 'comfyui':
        params: DiffusionParams = build_comfy_params(request.getfixturevalue('comfy_checkpoint'), prompt=PROMPT)
    else:
        params = DiffusionRequestBody(prompt=PROMPT, save_images=False, send_images=True)
    params.width = params.height = SIZE
    params.steps = STEPS
    params.seed = 11
    params.denoising_strength = 0.6
    params.init_images = [make_structured_image(SIZE, SIZE)]
    params.mask = make_comfy_mask(MASK_BOX, SIZE)
    return params


def _service(backend: str, request: pytest.FixtureRequest):
    return request.getfixturevalue('comfy_service' if backend == 'comfyui' else 'service')


def _inpaint(backend: str, request: pytest.FixtureRequest, **overrides) -> tuple[DiffusionParams, Image.Image]:
    params = _params(backend, request)
    for name, value in overrides.items():
        setattr(params, name, value)
    result = _service(backend, request).submit_inpaint(params).wait(timeout=300)
    return params, result.images[0]


def _detail(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Mean edge energy inside `box`: higher means finer detail."""
    return ImageStat.Stat(image.convert('L').filter(ImageFilter.FIND_EDGES).crop(box)).mean[0]


def _assert_only_the_patch_changed(source: Image.Image, result: Image.Image) -> None:
    assert result.size == source.size
    left, top, right, bottom = MASK_BOX
    outside = [(0, 0, SIZE, top - OUTSIDE_MARGIN), (0, bottom + OUTSIDE_MARGIN, SIZE, SIZE),
               (0, 0, left - OUTSIDE_MARGIN, SIZE), (right + OUTSIDE_MARGIN, 0, SIZE, SIZE)]
    for box in outside:
        assert region_mean_diff(source, result, box) <= 2.0, \
            f'pixels outside the mask changed in {box}; the result was not pasted back into the source'
    inside = (left + 16, top + 16, right - 16, bottom - 16)
    assert region_mean_diff(source, result, inside) >= 15.0, 'the masked patch barely changed'


@pytest.mark.parametrize('backend', ['webui', 'comfyui'])
def test_inpaint_full_res_adds_detail_to_a_small_patch(backend, request, output_dir):
    """Rendering the crop at full size gives the patch more fine detail than inpainting the whole image."""
    _, whole = _inpaint(backend, request)
    params, cropped = _inpaint(backend, request, inpaint_full_res=True, inpaint_full_res_padding=PADDING)
    source = params.init_images[0]
    for name, image in (('source', source), ('mask', params.mask), ('whole', whole), ('full_res', cropped)):
        save_output(output_dir, f'{backend}_full_res_{name}', image)

    _assert_only_the_patch_changed(source, whole)
    _assert_only_the_patch_changed(source, cropped)
    assert images_differ(whole, cropped) >= 2.0, 'inpaint_full_res produced the same image as whole-image inpainting'
    assert _detail(cropped, MASK_BOX) > _detail(whole, MASK_BOX), \
        f'the full-res patch is not finer than the whole-image patch; compare the {backend}_full_res_* PNGs'


@pytest.mark.parametrize('backend', ['webui', 'comfyui'])
def test_inpaint_full_res_crop_points_widen_the_crop(backend, request, output_dir):
    """A far-away crop point enlarges the crop, so the patch gets fewer pixels but the result still pastes back."""
    _, cropped = _inpaint(backend, request, inpaint_full_res=True, inpaint_full_res_padding=PADDING)
    params, with_point = _inpaint(backend, request, inpaint_full_res=True, inpaint_full_res_padding=PADDING,
                                  inpaint_crop_points=[FAR_POINT])
    save_output(output_dir, f'{backend}_full_res_far_point', with_point)

    _assert_only_the_patch_changed(params.init_images[0], with_point)
    assert images_differ(cropped, with_point) >= 2.0, 'the crop point did not change the crop'
    assert _detail(with_point, MASK_BOX) < _detail(cropped, MASK_BOX), \
        f'the wider crop should render the patch coarser; compare {backend}_full_res_far_point.png to ' \
        f'{backend}_full_res_full_res.png'
