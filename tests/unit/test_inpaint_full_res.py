"""Offline unit tests for "inpaint only masked": crop bounds, paste-back, and the ComfyUI and WebUI request paths."""
from typing import Any, Optional
from unittest.mock import patch

import pytest
from PIL import Image

from sd_backend_client.api.a1111_webservice import _crop_for_client_inpainting
from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference, QueueAdditionResponse
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody
from sd_backend_client.util.visual.inpaint_crop import PasteBack, compute_crop_region, expand_crop_region, \
    soft_inpainting_compositing_mask


def _mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new('L', size, 0)
    mask.paste(255, box)
    return mask


# --- compute_crop_region ----------------------------------------------------------------------------

def test_crop_is_mask_bounds_plus_padding():
    """A square mask under a square output crops to its bounds grown by the padding."""
    assert compute_crop_region(_mask((100, 100), (40, 40, 60, 60)), 10, (64, 64)) == (30, 30, 70, 70)


def test_crop_padding_clamps_to_the_image():
    """Padding never reaches past the image edge."""
    assert compute_crop_region(_mask((100, 100), (2, 2, 20, 20)), 10, (64, 64)) == (0, 0, 30, 30)


def test_empty_mask_has_no_crop():
    """An empty mask returns None so callers inpaint the whole image."""
    assert compute_crop_region(Image.new('L', (50, 50), 0), 8, (64, 64)) is None


def test_extra_points_grow_the_crop():
    """Points outside the padded bounds are contained in the crop."""
    region = compute_crop_region(_mask((200, 200), (90, 90, 110, 110)), 0, (64, 64), [(20, 100), (150, 160)])
    left, top, right, bottom = region
    assert left <= 20 and right > 150 and top <= 90 and bottom > 160


def test_extra_points_outside_the_image_are_clamped():
    """A point off the image extends the crop to the image edge, not beyond."""
    region = compute_crop_region(_mask((100, 100), (40, 40, 60, 60)), 0, (64, 64), [(-50, 500)])
    assert region[0] == 0 and region[3] == 100
    assert all(0 <= v <= 100 for v in region)


def test_wide_crop_expands_vertically_to_the_output_ratio():
    """A wide region gains height until it matches a square output, keeping its width."""
    assert expand_crop_region((20, 40, 80, 60), (64, 64), (100, 100)) == (20, 20, 80, 80)


def test_tall_crop_expands_horizontally_to_the_output_ratio():
    """A tall region gains width until it matches a square output, keeping its height."""
    assert expand_crop_region((40, 20, 60, 80), (64, 64), (100, 100)) == (20, 20, 80, 80)


def test_expansion_shifts_back_inside_the_image():
    """Expansion past an edge shifts the region instead of leaving the image."""
    assert expand_crop_region((0, 0, 60, 20), (64, 64), (100, 100)) == (0, 0, 60, 60)
    assert expand_crop_region((40, 80, 100, 100), (64, 64), (100, 100)) == (40, 40, 100, 100)


def test_expansion_matches_a_non_square_output():
    """The expanded region has the output's aspect ratio."""
    left, top, right, bottom = expand_crop_region((40, 40, 60, 60), (128, 64), (200, 200))
    assert (right - left) / (bottom - top) == 2


# --- PasteBack --------------------------------------------------------------------------------------

def test_paste_back_places_the_crop_and_keeps_source_outside_the_mask():
    """The generated crop lands at the region; only masked pixels change."""
    source = Image.new('RGBA', (20, 20), (200, 0, 0, 255))
    mask = _mask((20, 20), (6, 6, 10, 10))
    generated = Image.new('RGBA', (8, 8), (0, 0, 200, 255))       # crop (4, 4, 12, 12), resized from 8x8
    result = PasteBack(source, mask, (4, 4, 12, 12)).apply(generated)
    assert result.size == (20, 20)
    assert result.getpixel((7, 7)) == (0, 0, 200, 255)
    assert result.getpixel((5, 5)) == (200, 0, 0, 255)            # inside crop, outside mask
    assert result.getpixel((15, 15)) == (200, 0, 0, 255)          # outside crop


def test_paste_back_resizes_the_result_to_the_region():
    """A result larger than the region is scaled down to it before pasting."""
    source = Image.new('RGBA', (20, 20), (200, 0, 0, 255))
    full = Image.new('L', (20, 20), 255)
    result = PasteBack(source, full, (0, 0, 10, 10)).apply(Image.new('RGBA', (64, 64), (0, 0, 200, 255)))
    assert result.getpixel((9, 9)) == (0, 0, 200, 255)
    assert result.getpixel((10, 10)) == (200, 0, 0, 255)


# --- ComfyUI request path ---------------------------------------------------------------------------

class _Upload:
    def __init__(self, params: DiffusionParams) -> None:
        self.image: Optional[Image.Image] = None
        self.mask: Optional[Image.Image] = None
        service = ComfyUiWebservice('http://localhost:8188')

        def fake_upload_image(image: Image.Image, *_a: Any, **_k: Any) -> ImageFileReference:
            self.image = image
            return ImageFileReference(filename='src.png', subfolder='', type='input')

        def fake_upload_mask(mask: Image.Image, _ref: Any, *_a: Any, **_k: Any) -> ImageFileReference:
            self.mask = mask
            return ImageFileReference(filename='mask.png', subfolder='', type='input')

        with patch.object(service, 'get_models', return_value=[]), \
                patch.object(service, 'upload_image', fake_upload_image), \
                patch.object(service, 'upload_mask', fake_upload_mask), \
                patch.object(service, '_queue_prompt', return_value=QueueAdditionResponse(prompt_id='x')):
            _, self.paste_back = service._generate(params)


def _params(**kwargs: Any) -> DiffusionParams:
    return DiffusionParams(prompt='p', width=64, height=64, init_images=[Image.new('RGBA', (200, 200))],
                           mask=_mask((200, 200), (90, 90, 110, 110)), mask_blur=0, **kwargs)


def test_comfyui_uploads_the_crop_resized_to_the_output_size():
    """The crop of the image and mask upload at the output size."""
    upload = _Upload(_params(inpaint_full_res=True, inpaint_full_res_padding=10))
    assert upload.image.size == (64, 64) and upload.mask.size == (64, 64)
    assert upload.paste_back.region == (80, 80, 120, 120)


def test_comfyui_default_padding_applies_when_unset():
    """An unset padding uses INPAINT_FULL_RES_PADDING_DEFAULT."""
    upload = _Upload(_params(inpaint_full_res=True))
    assert upload.paste_back.region == (58, 58, 142, 142)


def test_comfyui_without_full_res_uploads_the_whole_image():
    """Without the flag the whole image uploads, with no crop region."""
    upload = _Upload(_params())
    assert upload.image.size == (200, 200) and upload.paste_back.region is None


def test_comfyui_empty_mask_falls_back_to_the_whole_image():
    """An empty mask uploads the whole image, as the WebUI does."""
    params = _params(inpaint_full_res=True)
    params.mask = Image.new('L', (200, 200), 0)
    upload = _Upload(params)
    assert upload.image.size == (200, 200) and upload.paste_back.region is None


def test_comfyui_extra_points_extend_the_crop():
    """Extra points enlarge the uploaded crop."""
    upload = _Upload(_params(inpaint_full_res=True, inpaint_full_res_padding=0, inpaint_crop_points=[(20, 100)]))
    left, _, right, _ = upload.paste_back.region
    assert left <= 20 and right >= 110


# --- WebUI request path -----------------------------------------------------------------------------

def _body(**kwargs: Any) -> DiffusionRequestBody:
    return DiffusionRequestBody(width=64, height=64, init_images=[Image.new('RGBA', (200, 200))],
                                mask=_mask((200, 200), (90, 90, 110, 110)), mask_blur=0, **kwargs)


def test_webui_keeps_the_server_flag_without_points():
    """Without extra points the WebUI body keeps the server-side flag and padding."""
    body = _body(inpaint_full_res=True, inpaint_full_res_padding=10)
    assert _crop_for_client_inpainting(body) is None
    data = body.to_dict()
    assert data['inpaint_full_res'] is True and data['inpaint_full_res_padding'] == 10
    assert data['init_images'][0] and 'inpaint_crop_points' not in data


def test_webui_with_points_crops_client_side_and_clears_the_flag():
    """Extra points make the client crop, and the server flag goes out off."""
    body = _body(inpaint_full_res=True, inpaint_full_res_padding=10, inpaint_crop_points=[(20, 100)])
    paste_back = _crop_for_client_inpainting(body)
    assert paste_back is not None and paste_back.region is not None
    left, top, right, bottom = paste_back.region
    assert left <= 20
    assert body.init_images[0].size == (right - left, bottom - top)
    assert body.mask.size == body.init_images[0].size
    data = body.to_dict()
    assert data['inpaint_full_res'] is False and 'inpaint_crop_points' not in data
    assert paste_back.source.size == (200, 200)


def test_webui_points_ignored_when_full_res_is_off():
    """Points do nothing unless inpaint_full_res is enabled."""
    body = _body(inpaint_crop_points=[(20, 100)])
    assert _crop_for_client_inpainting(body) is None
    assert body.init_images[0].size == (200, 200)


@pytest.mark.parametrize('padding', [-1])
def test_padding_rejects_negative_values(padding):
    """A negative padding fails validation."""
    with pytest.raises(ValueError):
        DiffusionParams(inpaint_full_res_padding=padding)


def test_soft_inpainting_compositing_mask_ramps_across_the_outer_edge_of_the_blur():
    """Levels up to 10% keep the source, levels from 20% take the result, and levels between ramp linearly."""
    gradient = Image.new('L', (256, 1))
    gradient.putdata(list(range(256)))
    levels = [soft_inpainting_compositing_mask(gradient).getpixel((x, 0)) for x in range(256)]
    assert levels[:26] == [0] * 26
    assert levels[51:] == [255] * 205
    assert levels[26:51] == sorted(levels[26:51]) and 0 < levels[38] < 255
