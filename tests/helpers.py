"""Small shared helpers for the integration tests."""
import os
from typing import Iterable, Optional

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import (ControlNetPreprocessor,
                                                                              PreprocessorParams)
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody

# Keep generation cheap: tiny canvas, minimal steps, single image.
TEST_SIZE = 256
TEST_STEPS = 4
UPSCALE_SOURCE_PROMPT = ('a weathered stone cottage covered in ivy beside a flower garden, highly detailed, '
                         'sharp focus, intricate textures')
"""Prompt for the generated upscale-test source: fine foliage and stone texture make upscaling quality visible."""
UPSCALE_SOURCE_SIZE = 512
BASIC_UPSCALE_SOURCE_SIZE = 128
"""The basic upscale tests shrink the generated source to this size, then upscale it back to `UPSCALE_SOURCE_SIZE`."""
SD_UPSCALE_SIZE = 2048
"""Target size for the diffusion upscale tests: 4x the source, so the tiled pass spans many 512px tiles."""
SD_UPSCALE_STEPS = 20
SD_UPSCALE_DENOISE = 0.35
"""Diffusion upscale step count and denoising strength, in the range real Ultimate SD Upscale runs use. The WebUI
   runs about `steps * denoise` sampling steps per tile unless its `img2img_fix_steps` option is set."""
MIN_UPSCALE_DETAIL_RATIO = 0.35
"""`upscale_detail_ratio` floor for diffusion upscales. A blurred result scores below 0.1; a usable one above 0.45."""

PREFERRED_CHECKPOINTS = ('deliberate_v3', 'cyberrealistic', 'dreamshaper', 'v1-5', 'sd15', 'sd-v1')
"""Checkpoint name fragments `pick_test_checkpoint` prefers, in order. All are SD1.5 models."""

_PREFERRED_UPSCALERS = ('ultrasharp', 'remacri', 'r-esrgan 4x+', 'esrgan_4x', 'dat x4', 'superscale')
_RESAMPLER_UPSCALERS = ('none', 'lanczos', 'nearest')


def pick_test_checkpoint(names: Iterable[str]) -> Optional[str]:
    """The checkpoint generation tests use: the first `PREFERRED_CHECKPOINTS` match that isn't an inpainting model.

    Falls back to the first non-inpainting checkpoint, then to the first checkpoint, or None if `names` is empty.
    Inpainting checkpoints expect extra mask channels, so they distort img2img and upscale results.
    """
    names = list(names)
    usable = [name for name in names if 'inpaint' not in name.lower()]
    for hint in PREFERRED_CHECKPOINTS:
        match = next((name for name in usable if hint in name.lower()), None)
        if match:
            return match
    if usable:
        return usable[0]
    return names[0] if names else None


def pick_upscaler_model(names: Iterable[str]) -> Optional[str]:
    """A model-based 4x upscaler from `names` (ESRGAN family, DAT, ...), or None if there are only resamplers."""
    models = [name for name in names if name.lower() not in _RESAMPLER_UPSCALERS]
    for hint in _PREFERRED_UPSCALERS:
        match = next((name for name in models if hint in name.lower()), None)
        if match:
            return match
    return models[0] if models else None


def _edge_energy(image: Image.Image, size: tuple[int, int]) -> float:
    return ImageStat.Stat(image.convert('L').resize(size, Image.Resampling.LANCZOS)
                          .filter(ImageFilter.FIND_EDGES)).mean[0]


def upscale_detail_ratio(source: Image.Image, result: Image.Image) -> float:
    """Edge energy of `result` scaled down to `source`'s size, relative to `source`'s own.

    Scaling down first averages away fine noise and dithering, so a blurred upscale scores low even when its full-size
    pixels are noisy.
    """
    return _edge_energy(result, source.size) / _edge_energy(source, source.size)


def assert_valid_image(image: Image.Image, expected_size: int) -> None:
    """Fail unless `image` is an RGBA PIL image within 8 px of `expected_size` on each side.

    Results are normalized to RGBA, and servers may round dimensions to a multiple of 8.
    """
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGBA'
    assert abs(image.width - expected_size) <= 8
    assert abs(image.height - expected_size) <= 8


def assert_upscale_kept_detail(source: Image.Image, result: Image.Image) -> None:
    """Fail if a diffusion upscale's `upscale_detail_ratio` is below `MIN_UPSCALE_DETAIL_RATIO`."""
    ratio = upscale_detail_ratio(source, result)
    assert ratio >= MIN_UPSCALE_DETAIL_RATIO, f'the upscale lost most of the source detail (ratio {ratio:.2f})'


def make_test_image(width: int = TEST_SIZE, height: int = TEST_SIZE,
                    color: tuple[int, int, int, int] = (128, 64, 200, 255)) -> Image.Image:
    """A solid-color RGBA image usable as an img2img / upscale / interrogate source."""
    return Image.new('RGBA', (width, height), color)


def make_mask(width: int = TEST_SIZE, height: int = TEST_SIZE) -> Image.Image:
    """A mask with a white square in the middle (region to inpaint) on black."""
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    quarter_w, quarter_h = width // 4, height // 4
    white = Image.new('RGBA', (width - 2 * quarter_w, height - 2 * quarter_h), (255, 255, 255, 255))
    mask.paste(white, (quarter_w, quarter_h))
    return mask


def make_edge_image(width: int = TEST_SIZE, height: int = TEST_SIZE) -> Image.Image:
    """A high-contrast image with clear edges, so a Canny/edge preprocessor has something to detect.

    (A solid color has no edges and would make ControlNet a silent no-op regardless of wiring.)
    """
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([width // 6, height // 6, width * 5 // 6, height * 5 // 6], outline=(0, 0, 0, 255), width=6)
    draw.ellipse([width // 3, height // 3, width * 2 // 3, height * 2 // 3], outline=(0, 0, 0, 255), width=6)
    draw.line([0, 0, width, height], fill=(0, 0, 0, 255), width=4)
    draw.line([0, height, width, 0], fill=(0, 0, 0, 255), width=4)
    return image


def make_structured_image(width: int = TEST_SIZE, height: int = TEST_SIZE) -> Image.Image:
    """A textured, non-flat RGBA source for inpainting tests.

    A flat color barely changes under inpainting unless denoising is cranked, and gives
    nothing to visually compare. This combines a diagonal color gradient with geometric
    shapes so inpainted regions are obvious and per-region diffs are meaningful.
    """
    image = Image.new('RGBA', (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (int(255 * x / width), int(255 * y / height),
                            int(255 * (x + y) / (width + height)), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([width // 6, height // 6, width * 5 // 6, height * 5 // 6],
                   outline=(255, 255, 255, 255), width=5)
    draw.ellipse([width // 3, height // 3, width * 2 // 3, height * 2 // 3],
                 outline=(0, 0, 0, 255), width=5)
    return image


def region_mean_diff(a: Image.Image, b: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Mean per-pixel difference between two images within a crop box (left, top, right, bottom)."""
    return images_differ(a.crop(box), b.crop(box))


def images_differ(a: Image.Image, b: Image.Image) -> float:
    """Mean per-pixel absolute difference between two images (0.0 == identical).

    Used to catch ControlNet silently doing nothing: same seed + prompt with vs. without
    a control unit should produce visibly different images when ControlNet actually applies.
    """
    a_rgb = a.convert('RGB')
    b_rgb = b.convert('RGB').resize(a_rgb.size)
    diff = ImageChops.difference(a_rgb, b_rgb)
    histogram = diff.histogram()
    # Weighted mean across the three channel histograms.
    total = sum(value * (index % 256) for index, value in enumerate(histogram))
    pixel_count = a_rgb.width * a_rgb.height * 3
    return total / pixel_count if pixel_count else 0.0


def make_controlnet_unit(module: str, model: str, control_image: Image.Image,
                         weight: float = 1.0) -> ControlNetUnit:
    """A minimal ControlNetUnit for the controlnet_units interface (serialized to WebUI by DiffusionRequestBody)."""
    return ControlNetUnit(
        image=control_image,
        model=ControlNetModel(model),
        preprocessor=PreprocessorParams(typedef=ControlNetPreprocessor(name=module)),
        control_strength=weight,
        control_start=0.0,
        control_end=1.0,
    )


def find_tile_controlnet_pairing(service) -> Optional[tuple[str, str]]:
    """A (module, model) tile ControlNet pairing from /controlnet/control_types, or None if unavailable.

    Mirrors conftest.controlnet_pairing's usability check, but looks for a category whose name
    contains "tile" instead of preferring Canny.
    """
    control_types = service.get_controlnet_control_types()['control_types']
    for name, entry in control_types.items():
        if 'tile' not in name.lower():
            continue
        module, model = entry.get('default_option'), entry.get('default_model')
        if module and model and module.lower() != 'none' and model.lower() != 'none':
            return module, model
    return None


def add_controlnet_unit(body: DiffusionRequestBody, unit: ControlNetUnit) -> None:
    """Attach a ControlNet unit to a request body via the controlnet_units interface, which DiffusionRequestBody
       serializes into alwayson_scripts on send."""
    body.controlnet_units.append(unit)


def save_output(output_dir: str, name: str, image: Image.Image) -> str:
    """Save an image under output_dir as a PNG and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{name}.png')
    image.convert('RGBA').save(path)
    return path


def fast_request_body(prompt: str = 'a red apple on a table') -> DiffusionRequestBody:
    """A minimal, fast request body that does not depend on the shared config defaults."""
    body = DiffusionRequestBody()
    body.prompt = prompt
    body.negative_prompt = ''
    body.steps = TEST_STEPS
    body.cfg_scale = 4.0
    body.width = TEST_SIZE
    body.height = TEST_SIZE
    body.batch_size = 1
    body.n_iter = 1
    body.seed = 1
    body.save_images = False
    body.send_images = True
    return body


def shrink(image: Image.Image, size: int) -> Image.Image:
    """Return a `size` x `size` LANCZOS downscale of `image`."""
    return image.resize((size, size), Image.Resampling.LANCZOS)
