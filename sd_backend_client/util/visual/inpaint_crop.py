"""Client-side "inpaint only masked" cropping, following A1111's `StableDiffusionProcessingImg2Img.init`.

The crop is computed from the blurred mask, grown to contain any extra points, and expanded to the output's aspect
ratio. The generated crop is resized back and pasted into the source image (`PasteBack`).
"""
from dataclasses import dataclass
from typing import Optional, Sequence, TypeAlias

from PIL import Image

CropBox: TypeAlias = tuple[int, int, int, int]
"""A crop region as (left, top, right, bottom) pixel bounds, right and bottom exclusive."""


def compute_crop_region(blurred_mask: Image.Image, padding: int, output_size: tuple[int, int],
                        extra_points: Optional[Sequence[tuple[int, int]]] = None) -> Optional[CropBox]:
    """Return the region of the image to inpaint at full resolution, or None when the mask is empty.

    The region is the bounds of the nonzero pixels of `blurred_mask`, plus `padding` on every side, grown to contain
    every point in `extra_points`, expanded to the aspect ratio of `output_size` (see `expand_crop_region`) and
    clamped to the image. An empty mask returns None, and callers then inpaint the whole image, as the WebUI does.
    """
    gray = blurred_mask.convert('L')
    bounds = gray.getbbox()
    if bounds is None:
        return None
    width, height = gray.size
    left, top, right, bottom = bounds
    left, top = max(left - padding, 0), max(top - padding, 0)
    right, bottom = min(right + padding, width), min(bottom + padding, height)
    for x, y in extra_points or ():
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        left, top = min(left, x), min(top, y)
        right, bottom = max(right, x + 1), max(bottom, y + 1)
    return expand_crop_region((left, top, right, bottom), output_size, gray.size)


def expand_crop_region(region: CropBox, output_size: tuple[int, int], image_size: tuple[int, int]) -> CropBox:
    """Grow `region` to the aspect ratio of `output_size`, shifting it to stay inside an image of `image_size`.

    Matches A1111's `masking.expand_crop_region`. When the image is too small to reach the ratio, the region is
    clamped to the image and keeps a different ratio.
    """
    left, top, right, bottom = region
    image_width, image_height = image_size
    output_ratio = output_size[0] / output_size[1]
    if (right - left) / (bottom - top) > output_ratio:
        diff = int((right - left) / output_ratio - (bottom - top))
        top -= diff // 2
        bottom += diff - diff // 2
        if bottom >= image_height:
            shift = bottom - image_height
            bottom -= shift
            top -= shift
        if top < 0:
            bottom -= top
            top = 0
        bottom = min(bottom, image_height)
    else:
        diff = int((bottom - top) * output_ratio - (right - left))
        left -= diff // 2
        right += diff - diff // 2
        if right >= image_width:
            shift = right - image_width
            right -= shift
            left -= shift
        if left < 0:
            right -= left
            left = 0
        right = min(right, image_width)
    return left, top, right, bottom


@dataclass
class PasteBack:
    """What a finished inpainting job needs to put its results onto the source image.

    `source` and `mask` are full size: `mask` is white where the result replaces the source. It is the blurred mask,
    or `soft_inpainting_compositing_mask` of it for a soft inpainting job. When
    `region` is set, results are crops of that region of `source` and are resized and pasted in before the mask is
    applied.
    """
    source: Image.Image
    mask: Image.Image
    region: Optional[CropBox] = None
    return_layers: bool = False
    """Whether the handle reports `layers` for each result. Set only when the mask is the one the backend composites
    through; the WebUI's own blend inside a crop is unknown, so its paste-backs leave this off."""

    def layers(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Return `image` as a full-size RGBA image, and the 'L' mask that composites it onto the source.

        When `region` is set, `image` is resized and pasted into a copy of the source first. Both returned images
        have the size of that result, so `Image.composite(uncomposited, source, mask)` equals `apply(image)` when the
        sizes match the source's.
        """
        image = image.convert('RGBA')
        if self.region is not None:
            left, top, right, bottom = self.region
            uncropped = self.source.convert('RGBA')
            uncropped.paste(image.resize((right - left, bottom - top), Image.Resampling.LANCZOS), (left, top))
            image = uncropped
        mask = self.mask.convert('L')
        if mask.size != image.size:
            mask = mask.resize(image.size)
        return image, mask

    def apply(self, image: Image.Image) -> Image.Image:
        """Return `image` composited onto the source: result where the mask is white, source where it is black."""
        image, mask = self.layers(image)
        source = self.source.convert('RGBA')
        if image.size != source.size:
            source = source.resize(image.size)
        return Image.composite(image, source, mask)


SOFT_INPAINTING_COMPOSITE_OFFSET = 0.1
"""Mask level at and below which a soft-inpainted result is not composited in.

See `soft_inpainting_compositing_mask`.
"""

SOFT_INPAINTING_COMPOSITE_THRESHOLD = 0.2
"""Mask level at and above which a soft-inpainted result fully replaces the source.

See `soft_inpainting_compositing_mask`.
"""


def soft_inpainting_compositing_mask(blurred_mask: Image.Image) -> Image.Image:
    """Return the mask to composite a Differential Diffusion result through, from the blurred mask it sampled with.

    Differential Diffusion already fades the change out across the blurred mask's grey band, so compositing through
    that mask again would fade it twice. The returned mask ramps linearly from 0 at `SOFT_INPAINTING_COMPOSITE_OFFSET`
    to 1 at `SOFT_INPAINTING_COMPOSITE_THRESHOLD`, as the comfyui-inpaint-nodes "Denoise to Compositing Mask" node does
    with its defaults. Pixels outside the band keep the source.
    """
    offset = SOFT_INPAINTING_COMPOSITE_OFFSET * 255
    ramp = (SOFT_INPAINTING_COMPOSITE_THRESHOLD - SOFT_INPAINTING_COMPOSITE_OFFSET) * 255
    lookup = [min(max(round((value - offset) / ramp * 255), 0), 255) for value in range(256)]
    return blurred_mask.convert('L').point(lookup)


def crop_to_region(image: Image.Image, region: CropBox, output_size: tuple[int, int]) -> Image.Image:
    """Return the `region` of `image` resized to `output_size`."""
    return image.crop(region).resize(output_size, Image.Resampling.LANCZOS)
