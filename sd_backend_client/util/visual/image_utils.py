"""Helpers for converting between PIL images and the byte/base64 formats image generation backends expect.

This is a slimmed, Qt-free replacement for IntraPaint's much larger image_utils module: the standalone API only
needs to encode images for upload and decode them from responses.
"""
import base64
import hashlib
import io
from typing import Optional, TypeAlias, Union

from PIL import Image, ImageChops, ImageFilter

BASE_64_PREFIX = 'data:image/png;base64,'


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG byte data."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL image from raw encoded image bytes (PNG, JPEG, etc.), normalized to RGBA."""
    image: Image.Image = Image.open(io.BytesIO(data))
    image.load()
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    return image


def image_to_base64(image: Union[Image.Image, str], include_prefix: bool = False) -> str:
    """Convert a PIL image or an image file path to a base64 string.

    When given a file path the raw file bytes are encoded as-is; when given a PIL image it is re-encoded as PNG.
    """
    if isinstance(image, str):
        with open(image, 'rb') as file:
            image_str = base64.b64encode(file.read()).decode('utf-8')
    else:
        image_str = base64.b64encode(image_to_png_bytes(image)).decode('utf-8')
    if include_prefix:
        return BASE_64_PREFIX + image_str
    return image_str


def image_from_base64(image_str: str) -> Image.Image:
    """Return a PIL image (RGBA) from base64-encoded string data, tolerating an optional data-URI prefix."""
    if image_str.startswith(BASE_64_PREFIX):
        image_str = image_str[len(BASE_64_PREFIX):]
    data = base64.b64decode(image_str)
    return image_from_bytes(data)


def mask_to_grayscale(mask: Image.Image) -> Image.Image:
    """Convert an inpainting mask to an opaque `L`-mode image: white where masked, black where not.

    The source channel depends on the mask's content:
    - Opaque and grayscale (R == G == B everywhere, including any `L`-mode image): brightness is the source luminance.
    - Anything else: brightness is the source alpha. A fully opaque colored image becomes all white.

    Hazard: decoded images are normalized to RGBA, so an opaque grayscale mask arrives with alpha 255 everywhere.
    Reading alpha without the opaque-grayscale check would turn it into an all-white mask.

    Forge's ControlNet extension flattens RGBA input over white before reading it, which loses an alpha-based mask.
    IntraPaint's `mask_to_grayscale` reads alpha only; this version also accepts pre-converted grayscale masks.
    """
    rgba = mask if mask.mode == 'RGBA' else mask.convert('RGBA')
    red, green, blue, alpha = rgba.split()
    is_opaque = alpha.getextrema() == (255, 255)
    is_grayscale = ImageChops.difference(red, green).getbbox() is None \
        and ImageChops.difference(green, blue).getbbox() is None
    if is_opaque and is_grayscale:
        return red
    return alpha


def blur_mask(mask: Image.Image, blur_x: float, blur_y: Optional[float] = None) -> Image.Image:
    """Return `mask` as an `L`-mode grayscale mask (white = change) with its edges feathered by a Gaussian blur.

    `blur_x` and `blur_y` are the horizontal and vertical Gaussian standard deviations in pixels, the same meaning as
    the WebUI's `mask_blur_x` and `mask_blur_y`; `blur_y` defaults to `blur_x`. The feathered edge extends roughly
    2.5 times the blur from the original boundary. A blur of 0 on both axes only converts the mask.
    """
    gray = mask_to_grayscale(mask)
    blur_y = blur_x if blur_y is None else blur_y
    if blur_x <= 0 and blur_y <= 0:
        return gray.copy()
    return gray.filter(ImageFilter.GaussianBlur((max(blur_x, 0), max(blur_y, 0))))


ImageKey: TypeAlias = tuple[str, tuple[int, int], bytes]


def get_image_key(image: Image.Image) -> ImageKey:
    """Creates a key usable for hashing image values."""
    return image.mode, image.size, hashlib.blake2b(image.tobytes(), digest_size=16).digest()
