"""Unit tests for the A1111/Forge request body — pure dataclass -> wire-dict logic.

``DiffusionRequestBody`` is already a dataclass, and its ``to_dict()`` is what actually
shapes the JSON sent to the WebUI. These tests pin that transform without a server. They
construct the body directly (never ``load_data`` / ``Cache``), so they survive the config
refactor; ``DiffusionRequestBody`` is essentially the target params dataclass for A1111.
"""
import base64
import io

from PIL import Image

from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.shared_data.diffusion_params import DEFAULT_DENOISING_STRENGTH, DiffusionParams
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody
from sd_backend_client.util.visual.image_utils import BASE_64_PREFIX, image_from_base64, image_to_base64


def test_to_dict_strips_unset_optional_fields():
    body = DiffusionRequestBody()
    data = body.to_dict()
    # None-valued optionals must not be serialized (the server would reject/misread them).
    assert 'denoising_strength' not in data
    assert 'mask' not in data
    assert 'init_images' not in data
    # Non-None defaults are kept.
    assert data['steps'] == body.steps
    assert data['cfg_scale'] == body.cfg_scale
    assert data['width'] == body.width and data['height'] == body.height


def test_to_dict_keeps_explicitly_set_optionals():
    body = DiffusionRequestBody()
    body.denoising_strength = 0.55
    body.mask = Image.new('RGBA', (2, 2), (255, 0, 0, 255))
    data = body.to_dict()
    assert data['denoising_strength'] == 0.55
    assert data['mask'].startswith(BASE_64_PREFIX)


def test_to_dict_fills_default_denoising_strength_only_for_img2img():
    """An img2img body without denoising_strength sends DEFAULT_DENOISING_STRENGTH."""
    body = DiffusionRequestBody(init_images=[Image.new('RGB', (2, 2))])
    assert body.to_dict()['denoising_strength'] == DEFAULT_DENOISING_STRENGTH
    assert body.denoising_strength is None  # to_dict leaves the body unchanged


def _emitted_mask_pixels(mask: Image.Image) -> list[int]:
    """Serialize `mask` through to_dict() and return the grayscale values of the emitted mask, row by row."""
    body = DiffusionRequestBody()
    body.mask = mask
    emitted = image_from_base64(body.to_dict()['mask'])
    pixels = [emitted.getpixel((x, y)) for y in range(emitted.height) for x in range(emitted.width)]
    # image_from_base64 normalizes to RGBA; an opaque L-mode source decodes with R == G == B and alpha 255.
    assert all(pixel[3] == 255 and pixel[0] == pixel[1] == pixel[2] for pixel in pixels)
    return [pixel[0] for pixel in pixels]


def test_to_dict_sends_alpha_mask_as_opaque_grayscale():
    """A colored alpha mask is sent with brightness equal to its alpha."""
    # Opaque red where selected, transparent elsewhere, one half-selected pixel.
    mask = Image.new('RGBA', (3, 1), (0, 0, 0, 0))
    mask.putpixel((0, 0), (255, 0, 0, 255))
    mask.putpixel((1, 0), (255, 0, 0, 128))
    assert _emitted_mask_pixels(mask) == [255, 128, 0]


def test_to_dict_sends_fully_opaque_colored_mask_as_all_white():
    """An opaque colored mask reads as fully masked."""
    # A selection covering the whole image is opaque everywhere; brightness comes from alpha, not the highlight color.
    mask = Image.new('RGBA', (2, 1), (255, 0, 0, 255))
    assert _emitted_mask_pixels(mask) == [255, 255]


def test_to_dict_keeps_grayscale_mask_luminance():
    """An L-mode mask is sent with its own luminance."""
    grayscale = Image.new('L', (3, 1), 0)
    grayscale.putpixel((0, 0), 255)
    grayscale.putpixel((1, 0), 100)
    assert _emitted_mask_pixels(grayscale) == [255, 100, 0]


def test_to_dict_keeps_grayscale_mask_luminance_after_rgba_normalization():
    """An opaque grayscale mask keeps its luminance after RGBA normalization; reading alpha would make it all white."""
    # An L-mode mask that went through image_from_base64 is RGBA with alpha 255 everywhere.
    grayscale = Image.new('L', (3, 1), 0)
    grayscale.putpixel((0, 0), 255)
    grayscale.putpixel((1, 0), 100)
    normalized = image_from_base64(image_to_base64(grayscale))
    assert normalized.mode == 'RGBA'
    assert _emitted_mask_pixels(normalized) == [255, 100, 0]


def test_to_dict_emits_single_channel_mask():
    """The mask is encoded as a single-channel L-mode PNG."""
    body = DiffusionRequestBody()
    body.mask = Image.new('RGBA', (2, 2), (255, 0, 0, 255))
    encoded = body.to_dict()['mask'][len(BASE_64_PREFIX):]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as emitted:
        assert emitted.mode == 'L'


def test_to_dict_passes_through_core_generation_params():
    body = DiffusionRequestBody()
    body.prompt = 'a red apple'
    body.negative_prompt = 'blurry'
    body.seed = 99
    body.steps = 7
    body.sampler = 'euler_ancestral'
    body.scheduler = 'ddim_uniform'
    data = body.to_dict()
    assert data['prompt'] == 'a red apple'
    assert data['negative_prompt'] == 'blurry'
    assert data['seed'] == 99
    assert data['steps'] == 7
    assert data['sampler_name'] == 'Euler a'  # shared names are translated to WebUI's
    assert data['scheduler'] == 'ddim'
    assert 'sampler' not in data


def test_to_dict_does_not_modify_the_body():
    """to_dict builds ControlNet args into its output only, so repeated calls are identical and self is unchanged."""
    body = DiffusionRequestBody(controlnet_units=[ControlNetUnit()])
    body.alwayson_scripts = {'controlNet': {'args': [{'stale': True}]}, 'other': {'args': [1]}}
    before = body.model_dump()
    first = body.to_dict()
    second = body.to_dict()
    assert body.model_dump() == before
    assert first == second
    assert len(first['alwayson_scripts']['controlNet']['args']) == 1
    assert 'stale' not in first['alwayson_scripts']['controlNet']['args'][0]
    assert first['alwayson_scripts']['other'] == {'args': [1]}


def test_to_dict_leaves_alwayson_scripts_unset():
    """A body with no scripts keeps alwayson_scripts None after to_dict."""
    body = DiffusionRequestBody()
    body.to_dict()
    assert body.alwayson_scripts is None



def test_to_dict_selects_checkpoint_through_override_settings():
    """sd_model_name becomes override_settings['sd_model_checkpoint'], the only checkpoint field the WebUI API reads."""
    data = DiffusionRequestBody(sd_model_name='model.safetensors').to_dict()
    assert data['override_settings'] == {'sd_model_checkpoint': 'model.safetensors'}
    assert 'sd_model_name' not in data
    assert 'override_settings_restore_afterwards' not in data


def test_to_dict_omits_checkpoint_override_when_sd_model_name_is_empty():
    """An empty sd_model_name leaves the server's configured checkpoint in place."""
    data = DiffusionRequestBody().to_dict()
    assert 'sd_model_name' not in data
    assert 'override_settings' not in data


def test_to_dict_keeps_caller_checkpoint_override_and_other_settings():
    """A caller-supplied sd_model_checkpoint wins over sd_model_name, and other overrides pass through."""
    overrides = {'sd_model_checkpoint': 'b.safetensors', 'CLIP_stop_at_last_layers': 2}
    body = DiffusionRequestBody(sd_model_name='a.safetensors', override_settings=overrides)
    assert body.to_dict()['override_settings'] == {'sd_model_checkpoint': 'b.safetensors',
                                                   'CLIP_stop_at_last_layers': 2}
    body = DiffusionRequestBody(sd_model_name='a.safetensors', override_settings={'CLIP_stop_at_last_layers': 2})
    assert body.to_dict()['override_settings'] == {'CLIP_stop_at_last_layers': 2,
                                                   'sd_model_checkpoint': 'a.safetensors'}
    assert body.override_settings == {'CLIP_stop_at_last_layers': 2}

# def test_add_init_image_appends_base64_data_uri():
# TODO: test to_dict serializes correctly with base64 images

def test_from_params_converts_shared_fields_and_drops_other_backends_fields():
    """from_params copies every shared field of another DiffusionParams subclass and ignores its own fields."""
    image = Image.new('RGBA', (4, 4), (1, 2, 3, 255))
    params = ComfyUIDiffusionParams(prompt='a cat', sampler='dpmpp_2m', seed=5, clip_skip=2, init_images=[image],
                                    controlnet_units=[ControlNetUnit()])
    body = DiffusionRequestBody.from_params(params)

    assert type(body) is DiffusionRequestBody  # pylint: disable=unidiomatic-typecheck
    shared_fields = set(DiffusionParams.model_fields.keys()) - {'init_images', 'controlnet_units'}
    assert {name: getattr(body, name) for name in shared_fields} == \
        {name: getattr(params, name) for name in shared_fields}
    assert not hasattr(body, 'clip_skip')
    assert body.init_images is not None and body.init_images[0] is not image
    assert body.init_images[0].tobytes() == image.tobytes()
    assert body.controlnet_units[0] is not params.controlnet_units[0]


def test_from_params_deep_copies_a_request_body():
    """from_params keeps a DiffusionRequestBody's WebUI-only fields in a copy that shares no state with it."""
    body = DiffusionRequestBody(n_iter=3, override_settings={'CLIP_stop_at_last_layers': 2})
    copied = DiffusionRequestBody.from_params(body)
    assert copied == body
    copied.override_settings['CLIP_stop_at_last_layers'] = 1  # type: ignore[index]
    assert body.override_settings == {'CLIP_stop_at_last_layers': 2}


def test_from_params_without_params_is_a_default_body():
    """from_params(None) is a default request body."""
    assert DiffusionRequestBody.from_params(None) == DiffusionRequestBody()


def test_to_dict_sends_soft_inpainting_args_in_script_order():
    """Soft inpainting becomes the script's enable flag and all six settings, set or defaulted, in argument order."""
    body = DiffusionRequestBody(soft_inpainting=True,
                                soft_inpainting_params={'mask_influence': 0.25, 'schedule_bias': 2.0})
    data = body.to_dict()
    assert data['alwayson_scripts']['soft inpainting'] == {'args': [True, 2.0, 0.5, 4.0, 0.25, 0.5, 2.0]}
    assert 'soft_inpainting' not in data and 'soft_inpainting_params' not in data


def test_to_dict_sends_no_soft_inpainting_while_it_is_off():
    """With soft inpainting off, neither the script nor the package's own fields are sent."""
    data = DiffusionRequestBody(soft_inpainting_params={'schedule_bias': 2.0}).to_dict()
    assert 'soft inpainting' not in data['alwayson_scripts']
    assert 'soft_inpainting' not in data and 'soft_inpainting_params' not in data


def test_to_dict_replaces_a_caller_soft_inpainting_entry_under_any_spelling():
    """The server matches script names case-insensitively, so a caller's entry is dropped whatever its case."""
    body = DiffusionRequestBody(soft_inpainting=True, alwayson_scripts={'Soft Inpainting': {'args': [False]},
                                                                        'other': {'args': [1]}})
    scripts = body.to_dict()['alwayson_scripts']
    assert set(scripts) == {'soft inpainting', 'other'}
    assert scripts['soft inpainting']['args'][0] is True
    assert body.alwayson_scripts == {'Soft Inpainting': {'args': [False]}, 'other': {'args': [1]}}
