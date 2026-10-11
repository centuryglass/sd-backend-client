"""Offline unit tests for the inpainting mask blur: `blur_mask`, the ComfyUI request path and the WebUI body."""
from typing import Any, Optional
from unittest.mock import patch

import pytest
from PIL import Image

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference, QueueAdditionResponse
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody
from sd_backend_client.util.api_defaults import MASK_BLUR_DEFAULT
from sd_backend_client.util.visual.image_utils import blur_mask


def _half_mask(size: int = 64) -> Image.Image:
    """Opaque grayscale mask whose left half marks the change region."""
    mask = Image.new('RGBA', (size, size), (0, 0, 0, 255))
    mask.paste(Image.new('RGBA', (size // 2, size), (255, 255, 255, 255)), (0, 0))
    return mask


def test_blur_mask_zero_only_converts():
    """A blur of 0 leaves the mask's values unchanged."""
    result = blur_mask(_half_mask(), 0)
    assert result.mode == 'L'
    assert result.getpixel((0, 0)) == 255 and result.getpixel((63, 0)) == 0
    assert result.getpixel((31, 0)) == 255 and result.getpixel((32, 0)) == 0


def test_blur_mask_feathers_the_edge_within_about_2_5_sigma():
    """The blurred edge is grey near the boundary and unchanged beyond roughly 2.5 sigma of it."""
    sigma = 4
    row = blur_mask(_half_mask(), sigma)
    assert 0 < row.getpixel((31, 0)) < 255
    assert 0 < row.getpixel((32, 0)) < 255
    reach = int(2.5 * sigma + 0.5)
    assert row.getpixel((31 - 2 * reach, 0)) == 255
    assert row.getpixel((32 + 2 * reach, 0)) == 0
    assert row.getpixel((31 - reach, 0)) > 250
    assert row.getpixel((32 + reach, 0)) < 5


class _Capture:
    """Records the mask `_generate` uploads, with the network calls stubbed out."""

    def __init__(self, params: DiffusionParams) -> None:
        self.uploaded: Optional[Image.Image] = None
        service = ComfyUiWebservice('http://localhost:8188')
        ref = ImageFileReference(filename='src.png', subfolder='', type='input')

        def fake_upload_mask(mask: Image.Image, _ref: ImageFileReference, *_a: Any, **_k: Any) -> ImageFileReference:
            self.uploaded = mask
            return ImageFileReference(filename='mask.png', subfolder='', type='input')

        with patch.object(service, 'get_models', return_value=[]), \
                patch.object(service, 'upload_image', return_value=ref), \
                patch.object(service, 'upload_mask', fake_upload_mask), \
                patch.object(service, '_queue_prompt', return_value=QueueAdditionResponse(prompt_id='x')):
            service._generate(params)


def _params(**kwargs: Any) -> DiffusionParams:
    return DiffusionParams(prompt='p', init_images=[Image.new('RGBA', (64, 64))], mask=_half_mask(), **kwargs)


def test_comfyui_blurs_uploaded_mask():
    """An explicit mask_blur blurs the uploaded mask."""
    capture = _Capture(_params(mask_blur=6))
    assert capture.uploaded is not None
    assert 0 < capture.uploaded.getpixel((31, 0)) < 255


def test_comfyui_defaults_to_the_webui_blur():
    """An unset mask_blur uses MASK_BLUR_DEFAULT."""
    capture = _Capture(_params())
    expected = blur_mask(_half_mask(), MASK_BLUR_DEFAULT)
    assert capture.uploaded is not None
    assert capture.uploaded.tobytes() == expected.tobytes()


def test_comfyui_zero_blur_uploads_a_hard_edge():
    """mask_blur=0 uploads the mask unblurred."""
    capture = _Capture(_params(mask_blur=0))
    assert capture.uploaded is not None
    assert capture.uploaded.getpixel((31, 0)) == 255 and capture.uploaded.getpixel((32, 0)) == 0


def test_comfyui_mask_blur_wins_over_x_and_y():
    """`mask_blur` overrides `mask_blur_x` and `mask_blur_y`, as on the WebUI."""
    capture = _Capture(_params(mask_blur=6, mask_blur_x=0, mask_blur_y=0))
    assert capture.uploaded is not None
    assert capture.uploaded.tobytes() == blur_mask(_half_mask(), 6).tobytes()


def test_comfyui_blurs_each_axis_separately():
    """With only x and y set, a vertical-only blur leaves a vertical edge sharp, and a horizontal blur softens it."""
    vertical_only = _Capture(_params(mask_blur_x=0, mask_blur_y=6)).uploaded
    horizontal_only = _Capture(_params(mask_blur_x=6, mask_blur_y=0)).uploaded
    assert vertical_only is not None and horizontal_only is not None
    assert vertical_only.getpixel((31, 32)) == 255 and vertical_only.getpixel((32, 32)) == 0
    assert 0 < horizontal_only.getpixel((31, 32)) < 255


def test_comfyui_unset_axis_uses_the_default():
    """When only one axis is set, the other uses MASK_BLUR_DEFAULT."""
    capture = _Capture(_params(mask_blur_x=0))
    assert capture.uploaded is not None
    assert capture.uploaded.tobytes() == blur_mask(_half_mask(), 0, MASK_BLUR_DEFAULT).tobytes()


@pytest.mark.parametrize('field', ['mask_blur', 'mask_blur_x', 'mask_blur_y'])
def test_mask_blur_rejects_negative_values(field):
    """A negative blur is invalid."""
    with pytest.raises(ValueError):
        DiffusionParams(**{field: -1})


def test_webui_body_sends_mask_blur_unchanged():
    """The WebUI wire format keeps `mask_blur`, and omits it when unset."""
    assert DiffusionRequestBody(mask_blur=7).to_dict()['mask_blur'] == 7
    assert 'mask_blur' not in DiffusionRequestBody().to_dict()


def test_webui_body_sends_x_and_y_unchanged():
    """The WebUI wire format keeps `mask_blur_x` and `mask_blur_y`."""
    data = DiffusionRequestBody(mask_blur_x=2, mask_blur_y=5).to_dict()
    assert data['mask_blur_x'] == 2 and data['mask_blur_y'] == 5
    assert 'mask_blur' not in data
