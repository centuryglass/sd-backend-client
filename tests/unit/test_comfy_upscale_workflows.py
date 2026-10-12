"""Unit tests for the ComfyUI upscaling workflows: graph shape and output sizing, no server needed."""
from typing import Optional

import pytest
from PIL import Image

from sd_backend_client.api.comfyui.basic_upscale_workflow_builder import build_basic_upscaling_workflow
from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.diffusion_workflow_builder import ExtensionModelType
from sd_backend_client.api.comfyui.latent_upscale_workflow_builder import LatentUpscaleWorkflowBuilder
from sd_backend_client.api.comfyui.nodes.ultimate_upscale_node import ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_tiled_node import TILE_MAX, TILE_MIN
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.api.comfyui.preprocessor_preview_workflow_builder import PreprocessorPreviewWorkflowBuilder
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import (ControlNetPreprocessor,
                                                                              ParameterDef, PreprocessorParams)
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.util.geometry import Size

SOURCE = ImageFileReference(filename='source.png', subfolder='')
FINAL_SIZE = Size(1536, 1024)


def nodes_of_type(workflow: dict, class_type: str) -> list[dict]:
    """All nodes in a workflow dict with the given class_type."""
    return [node for node in workflow.values() if node['class_type'] == class_type]


def single_node(workflow: dict, class_type: str) -> dict:
    """The one node of a given class_type."""
    matches = nodes_of_type(workflow, class_type)
    assert len(matches) == 1, f'expected exactly one {class_type} node, found {len(matches)}'
    return matches[0]


def source_class(workflow: dict, connection: list) -> str:
    """The class_type of the node a connection points at."""
    return workflow[connection[0]]['class_type']


def latent_builder(ultimate: bool, upscale_model: Optional[str],
                   tile_unit: Optional[ControlNetUnit] = None) -> LatentUpscaleWorkflowBuilder:
    """A latent upscale builder for a 3x upscale to FINAL_SIZE."""
    builder = LatentUpscaleWorkflowBuilder(SOURCE, 3.0, FINAL_SIZE, Size(512, 512), ultimate, upscale_model, tile_unit)
    builder.sd_model = 'model.safetensors'
    return builder


def build_latent_workflow(ultimate: bool, upscale_model: Optional[str]) -> dict:
    """Build the latent upscale graph for a 3x upscale to FINAL_SIZE."""
    return latent_builder(ultimate, upscale_model).build_workflow().get_workflow_dict()


def test_basic_workflow_resizes_model_output_to_target():
    """The basic graph resizes the model output to the exact target size."""
    workflow = build_basic_upscaling_workflow(SOURCE, 'esrgan.pth', FINAL_SIZE).get_workflow_dict()

    resize = single_node(workflow, 'ImageScale')
    assert resize['inputs']['width'] == 1536
    assert resize['inputs']['height'] == 1024
    assert resize['inputs']['crop'] == 'disabled'
    apply_model = single_node(workflow, 'ImageUpscaleWithModel')
    assert source_class(workflow, resize['inputs']['image']) == 'ImageUpscaleWithModel'
    assert source_class(workflow, apply_model['inputs']['image']) == 'LoadImage'
    assert source_class(workflow, apply_model['inputs']['upscale_model']) == 'UpscaleModelLoader'
    save = single_node(workflow, 'SaveImage')
    assert source_class(workflow, save['inputs']['images']) == 'ImageScale'


def test_basic_workflow_without_size_skips_resize():
    """With no final size, the model output goes straight to SaveImage."""
    workflow = build_basic_upscaling_workflow(SOURCE, 'esrgan.pth').get_workflow_dict()

    assert not nodes_of_type(workflow, 'ImageScale')
    save = single_node(workflow, 'SaveImage')
    assert source_class(workflow, save['inputs']['images']) == 'ImageUpscaleWithModel'


@pytest.mark.parametrize('workflow', [
    build_basic_upscaling_workflow(SOURCE, 'esrgan.pth').get_workflow_dict(),
    build_latent_workflow(True, 'esrgan.pth'),
    build_latent_workflow(False, None),
], ids=['basic', 'latent_ultimate', 'latent_plain'])
def test_upscale_workflows_save_with_visible_prefix(workflow: dict):
    """An empty prefix makes ComfyUI save hidden `._00001_.png` files."""
    assert single_node(workflow, 'SaveImage')['inputs']['filename_prefix'] == 'sd_backend_client'


def test_latent_workflow_without_ultimate_applies_upscale_model_then_resizes():
    """Without Ultimate SD Upscale, the upscale model runs before the resize and VAE encode."""
    workflow = build_latent_workflow(False, 'esrgan.pth')

    resize = single_node(workflow, 'ImageScale')
    assert (resize['inputs']['width'], resize['inputs']['height']) == (1536, 1024)
    apply_model = single_node(workflow, 'ImageUpscaleWithModel')
    assert source_class(workflow, resize['inputs']['image']) == 'ImageUpscaleWithModel'
    assert source_class(workflow, apply_model['inputs']['image']) == 'LoadImage'
    assert source_class(workflow, apply_model['inputs']['upscale_model']) == 'UpscaleModelLoader'
    encode = single_node(workflow, 'VAEEncodeTiled')
    assert source_class(workflow, encode['inputs']['pixels']) == 'ImageScale'
    assert source_class(workflow, single_node(workflow, 'KSampler')['inputs']['latent_image']) == 'VAEEncodeTiled'
    assert not nodes_of_type(workflow, 'LatentUpscale')


def test_latent_workflow_without_ultimate_or_model_resizes_pixels_before_encode():
    """With no upscale model, the image is resized in pixel space; a latent resize decodes to a blurred image."""
    workflow = build_latent_workflow(False, None)

    assert not nodes_of_type(workflow, 'ImageUpscaleWithModel')
    assert not nodes_of_type(workflow, 'UpscaleModelLoader')
    assert not nodes_of_type(workflow, 'LatentUpscale')
    resize = single_node(workflow, 'ImageScale')
    assert (resize['inputs']['width'], resize['inputs']['height']) == (1536, 1024)
    assert resize['inputs']['upscale_method'] == 'lanczos'
    assert source_class(workflow, resize['inputs']['image']) == 'LoadImage'
    encode = single_node(workflow, 'VAEEncodeTiled')
    assert source_class(workflow, encode['inputs']['pixels']) == 'ImageScale'
    assert source_class(workflow, single_node(workflow, 'KSampler')['inputs']['latent_image']) == 'VAEEncodeTiled'


def test_ultimate_workflow_with_model_wires_model_and_uses_multiplier():
    """The Ultimate SD Upscale node receives the model and the multiplier."""
    workflow = build_latent_workflow(True, 'esrgan.pth')

    ultimate = single_node(workflow, 'UltimateSDUpscale')
    assert ultimate['inputs']['upscale_by'] == 3.0
    assert source_class(workflow, ultimate['inputs']['upscale_model']) == 'UpscaleModelLoader'
    assert source_class(workflow, ultimate['inputs']['image']) == 'LoadImage'
    assert not nodes_of_type(workflow, 'ImageScale')


def test_ultimate_workflow_without_model_prescales_by_multiplier():
    """With no model, the Ultimate SD Upscale input is pre-scaled by the multiplier."""
    workflow = build_latent_workflow(True, None)

    scale_by = single_node(workflow, 'ImageScaleBy')
    assert scale_by['inputs']['scale_by'] == 3.0
    assert not nodes_of_type(workflow, 'UpscaleModelLoader')


@pytest.mark.parametrize('size', [(64, 64), (32, 64), (64, 32)])
def test_upscale_rejects_non_enlarging_size(size: tuple[int, int]):
    """upscale() raises ValueError when the target does not exceed the source."""
    # The size check runs before any request, so no server is contacted.
    service = ComfyUiWebservice('http://127.0.0.1:1')
    with pytest.raises(ValueError, match='must exceed'):
        service.upscale(Image.new('RGBA', (64, 64)), *size)


def _tile_unit() -> ControlNetUnit:
    """A tile ControlNet unit with an image-input preprocessor."""
    typedef = ControlNetPreprocessor(name='TilePreprocessor', has_image_input=True, has_mask_input=False)
    return ControlNetUnit(model=ControlNetModel('control_tile.safetensors'),
                          preprocessor=PreprocessorParams(typedef=typedef), control_strength=0.6)


@pytest.mark.parametrize('ultimate', [True, False])
def test_latent_workflow_chains_extension_models(ultimate: bool):
    """LoRAs chain model and CLIP, hypernetworks chain the model only, and the sampler gets the last model."""
    builder = latent_builder(ultimate, None)
    builder.add_extension_model('detail.safetensors', 0.8, 0.5, ExtensionModelType.LORA)
    builder.add_extension_model('style.pt', 0.3, 0.3, ExtensionModelType.HYPERNETWORK)
    workflow = builder.build_workflow().get_workflow_dict()

    lora = single_node(workflow, 'LoraLoader')
    hypernet = single_node(workflow, 'HypernetworkLoader')
    assert source_class(workflow, lora['inputs']['model']) == 'CheckpointLoaderSimple'
    assert source_class(workflow, lora['inputs']['clip']) == 'CheckpointLoaderSimple'
    assert source_class(workflow, hypernet['inputs']['model']) == 'LoraLoader'
    for encoder in nodes_of_type(workflow, 'CLIPTextEncode'):
        assert source_class(workflow, encoder['inputs']['clip']) == 'LoraLoader'
    sampler = single_node(workflow, ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL if ultimate else 'KSampler')
    assert source_class(workflow, sampler['inputs']['model']) == 'HypernetworkLoader'


@pytest.mark.parametrize('ultimate', [True, False])
def test_latent_workflow_applies_clip_skip(ultimate: bool):
    """CLIP skip routes the checkpoint's CLIP output through CLIPSetLastLayer before both prompt encoders."""
    builder = latent_builder(ultimate, None)
    builder.clip_skip = 2
    workflow = builder.build_workflow().get_workflow_dict()

    skip = single_node(workflow, 'CLIPSetLastLayer')
    assert skip['inputs']['stop_at_clip_layer'] == -2
    assert source_class(workflow, skip['inputs']['clip']) == 'CheckpointLoaderSimple'
    encoders = nodes_of_type(workflow, 'CLIPTextEncode')
    assert len(encoders) == 2
    assert all(source_class(workflow, node['inputs']['clip']) == 'CLIPSetLastLayer' for node in encoders)


@pytest.mark.parametrize('ultimate', [True, False])
def test_latent_workflow_without_clip_skip_omits_skip_node(ultimate: bool):
    """The default CLIP skip adds no CLIPSetLastLayer node."""
    workflow = latent_builder(ultimate, None).build_workflow().get_workflow_dict()
    assert not nodes_of_type(workflow, 'CLIPSetLastLayer')


def test_latent_workflow_uses_config_loader_when_config_is_set():
    """A model config path switches the checkpoint loader to CheckpointLoader with that config."""
    builder = latent_builder(True, None)
    builder.model_config_path = 'v1-inference.yaml'
    workflow = builder.build_workflow().get_workflow_dict()

    loader = single_node(workflow, 'CheckpointLoader')
    assert loader['inputs']['config_name'] == 'v1-inference.yaml'
    assert not nodes_of_type(workflow, 'CheckpointLoaderSimple')


@pytest.mark.parametrize('ultimate', [True, False])
def test_latent_workflow_applies_tile_controlnet_to_conditioning(ultimate: bool):
    """The tile unit preprocesses the source image and its conditioning feeds the sampler."""
    workflow = latent_builder(ultimate, None, _tile_unit()).build_workflow().get_workflow_dict()

    preprocessor = single_node(workflow, 'TilePreprocessor')
    assert source_class(workflow, preprocessor['inputs']['image']) == 'LoadImage'
    apply_node = single_node(workflow, 'ControlNetApplyAdvanced')
    assert apply_node['inputs']['strength'] == 0.6
    assert source_class(workflow, apply_node['inputs']['image']) == 'TilePreprocessor'
    assert source_class(workflow, apply_node['inputs']['control_net']) == 'ControlNetLoader'
    apply_key = next(key for key, node in workflow.items() if node['class_type'] == 'ControlNetApplyAdvanced')
    sampler = single_node(workflow, ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL if ultimate else 'KSampler')
    assert tuple(sampler['inputs']['positive']) == (apply_key, 0)
    assert tuple(sampler['inputs']['negative']) == (apply_key, 1)


def test_latent_workflow_builds_repeatably():
    """Building twice gives the same graph, so connections on reused extension and ControlNet nodes don't leak."""
    builder = latent_builder(True, 'esrgan.pth', _tile_unit())
    builder.add_extension_model('detail.safetensors', 0.8, 0.5, ExtensionModelType.LORA)
    assert builder.build_workflow().get_workflow_dict() == builder.build_workflow().get_workflow_dict()


@pytest.mark.parametrize('requested, emitted', [
    (100, TILE_MIN),
    (700, 640),
    (TILE_MAX + 512, TILE_MAX),
])
def test_latent_workflow_snaps_vae_tile_size_into_range(requested: int, emitted: int):
    """Without Ultimate SD Upscale, the VAE tile size is rounded down to a TILE_STEP multiple and clamped."""
    builder = latent_builder(False, None)
    builder.vae_tile_size = requested
    workflow = builder.build_workflow().get_workflow_dict()

    assert single_node(workflow, 'VAEEncodeTiled')['inputs']['tile_size'] == emitted
    assert single_node(workflow, 'VAEDecodeTiled')['inputs']['tile_size'] == emitted


def test_latent_workflow_passes_ultimate_tunables():
    """Tunables set on the builder reach the UltimateSDUpscale node."""
    builder = latent_builder(True, 'esrgan.pth')
    builder.tile_size = Size(768, 640)
    builder.seam_fix_mode = 'Band Pass'
    builder.mask_blur = 4
    ultimate = single_node(builder.build_workflow().get_workflow_dict(), 'UltimateSDUpscale')

    assert (ultimate['inputs']['tile_width'], ultimate['inputs']['tile_height']) == (768, 640)
    assert ultimate['inputs']['seam_fix_mode'] == 'Band Pass'
    assert ultimate['inputs']['mask_blur'] == 4


def test_latent_workflow_without_source_image_raises():
    """A builder whose source image was cleared refuses to build."""
    builder = latent_builder(True, None)
    builder.source_image = None
    with pytest.raises(ValueError, match='No image'):
        builder.build_workflow()


def test_preprocessor_preview_without_a_required_image_raises_value_error():
    """A preview of a preprocessor that takes an image needs a source image."""
    builder = PreprocessorPreviewWorkflowBuilder(ControlNetPreprocessor(name='CannyEdgePreprocessor'))
    with pytest.raises(ValueError, match='CannyEdgePreprocessor'):
        builder.build_workflow(None)


_CANNY = ControlNetPreprocessor(name='CannyEdgePreprocessor', has_mask_input=False, parameters=[
    ParameterDef(key='low_threshold', default_value=100), ParameterDef(key='high_threshold', default_value=200)])


@pytest.mark.parametrize('preprocessor, expected', [
    (_CANNY, {'low_threshold': 100, 'high_threshold': 200}),
    (PreprocessorParams(typedef=_CANNY, parameter_values={'low_threshold': 50}),
     {'low_threshold': 50, 'high_threshold': 200}),
])
def test_preprocessor_preview_applies_parameter_overrides_over_defaults(
        preprocessor: ControlNetPreprocessor | PreprocessorParams, expected: dict[str, int]):
    """Preview inputs take each parameter's default unless a PreprocessorParams sets it."""
    builder = PreprocessorPreviewWorkflowBuilder(preprocessor)
    workflow = builder.build_workflow(SOURCE).get_workflow_dict()

    node = single_node(workflow, 'CannyEdgePreprocessor')
    assert {key: node['inputs'][key] for key in expected} == expected
    assert source_class(workflow, node['inputs']['image']) == 'LoadImage'
