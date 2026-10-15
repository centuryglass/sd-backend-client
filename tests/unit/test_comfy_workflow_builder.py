"""Unit tests for the ComfyUI workflow builder — pure params -> node-graph logic.

No server, no GPU, fully deterministic. They drive ``DiffusionWorkflowBuilder`` by setting
its attributes directly (never ``load_cached_settings`` / ``Cache``) and assert the shape of
the graph ``build_workflow()`` produces. This is the 'functional core' surface discussed for
the config refactor: it does not touch the backend, so it survives that refactor. When the
builder's inputs become a params dataclass, only the *setup* here should change — the
structural assertions on the built graph stay put, and that's precisely what protects the
refactor from silently changing the emitted workflow.
"""
import pytest

from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.diffusion_workflow_builder import DiffusionWorkflowBuilder
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import (
    ControlNetPreprocessor, ParameterDef, PreprocessorParams)
from sd_backend_client.util.geometry import Size


def nodes_of_type(workflow: dict, class_type: str) -> list[dict]:
    """All nodes in a workflow dict with the given ComfyUI class_type."""
    return [node for node in workflow.values() if node['class_type'] == class_type]


def single_node(workflow: dict, class_type: str) -> dict:
    """The one node of a given class_type, asserting there is exactly one."""
    matches = nodes_of_type(workflow, class_type)
    assert len(matches) == 1, f'expected exactly one {class_type} node, found {len(matches)}'
    return matches[0]


def make_txt2img_builder() -> DiffusionWorkflowBuilder:
    builder = DiffusionWorkflowBuilder()
    builder.prompt = 'a corgi astronaut, detailed'
    builder.negative_prompt = 'blurry, low quality'
    builder.sd_model = 'deliberate_v3.safetensors'
    builder.steps = 12
    builder.cfg_scale = 6.5
    builder.sampler = 'euler'
    builder.scheduler = 'karras'
    builder.seed = 4242
    builder.batch_size = 1
    builder.image_size = Size(384, 256)
    return builder


def test_txt2img_graph_has_expected_backbone():
    workflow = make_txt2img_builder().build_workflow().get_workflow_dict()

    checkpoint = single_node(workflow, 'CheckpointLoaderSimple')
    assert checkpoint['inputs']['ckpt_name'] == 'deliberate_v3.safetensors'

    sampler = single_node(workflow, 'KSampler')
    assert sampler['inputs']['steps'] == 12
    assert sampler['inputs']['cfg'] == 6.5
    assert sampler['inputs']['sampler_name'] == 'euler'
    assert sampler['inputs']['scheduler'] == 'karras'
    assert sampler['inputs']['seed'] == 4242
    assert sampler['inputs']['denoise'] == 1.0  # txt2img is always full denoise

    single_node(workflow, 'VAEDecode')
    single_node(workflow, 'SaveImage')


def test_save_image_prefix_is_visible_by_default():
    """An empty prefix makes ComfyUI save hidden `._00001_.png` files."""
    workflow = make_txt2img_builder().build_workflow().get_workflow_dict()
    assert single_node(workflow, 'SaveImage')['inputs']['filename_prefix'] == 'sd_backend_client'


def test_txt2img_encodes_both_prompts():
    workflow = make_txt2img_builder().build_workflow().get_workflow_dict()
    encoded = {node['inputs']['text'] for node in nodes_of_type(workflow, 'CLIPTextEncode')}
    assert encoded == {'a corgi astronaut, detailed', 'blurry, low quality'}


def test_txt2img_uses_empty_latent_with_size_and_batch():
    builder = make_txt2img_builder()
    builder.batch_size = 3
    workflow = builder.build_workflow().get_workflow_dict()

    latent = single_node(workflow, 'EmptyLatentImage')
    assert latent['inputs']['width'] == 384
    assert latent['inputs']['height'] == 256
    assert latent['inputs']['batch_size'] == 3
    # A pure txt2img graph never encodes an input image.
    assert nodes_of_type(workflow, 'VAEEncode') == []
    assert nodes_of_type(workflow, 'LoadImage') == []


def test_ksampler_is_wired_to_model_conditioning_and_latent():
    workflow = make_txt2img_builder().build_workflow().get_workflow_dict()
    sampler_inputs = single_node(workflow, 'KSampler')['inputs']
    # Connections serialize as [node_key, output_slot] pairs; assert they are wired, not literal.
    for key in ('model', 'positive', 'negative', 'latent_image'):
        assert isinstance(sampler_inputs[key], (list, tuple)) and len(sampler_inputs[key]) == 2, \
            f'KSampler.{key} is not connected to another node'


def test_img2img_uses_vae_encode_instead_of_empty_latent():
    builder = make_txt2img_builder()
    builder.source_image = 'IntraPaint/src_image.png [input]'
    builder.denoising_strength = 0.6
    workflow = builder.build_workflow().get_workflow_dict()

    assert nodes_of_type(workflow, 'EmptyLatentImage') == []
    encode = single_node(workflow, 'VAEEncode')
    load_image = single_node(workflow, 'LoadImage')
    assert load_image['inputs']['image'] == 'IntraPaint/src_image.png [input]'
    # The source is stretched to image_size before encoding, as WebUI does:
    scale = single_node(workflow, 'ImageScale')
    assert (scale['inputs']['width'], scale['inputs']['height'], scale['inputs']['crop']) == (384, 256, 'disabled')
    assert workflow[str(scale['inputs']['image'][0])] is load_image
    assert workflow[str(encode['inputs']['pixels'][0])] is scale
    # Denoising strength should flow through to the sampler for img2img.
    assert single_node(workflow, 'KSampler')['inputs']['denoise'] == 0.6


def test_clip_skip_inserts_set_last_layer_node():
    baseline = make_txt2img_builder().build_workflow().get_workflow_dict()
    assert nodes_of_type(baseline, 'CLIPSetLastLayer') == []  # default clip_skip=1 => no node

    builder = make_txt2img_builder()
    builder.clip_skip = 2
    workflow = builder.build_workflow().get_workflow_dict()
    skip_node = single_node(workflow, 'CLIPSetLastLayer')
    # ComfyUI encodes "skip N" as a negative last-layer index.
    assert skip_node['inputs']['stop_at_clip_layer'] == -2


def test_batch_size_is_validated():
    builder = DiffusionWorkflowBuilder()
    for bad in (0, 65):
        try:
            builder.batch_size = bad
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_size={bad} should have been rejected')
    builder.batch_size = 4  # in range, no raise
    assert builder.batch_size == 4


def _canny_params() -> PreprocessorParams:
    """A fresh but equal set of Canny preprocessor params on each call."""
    typedef = ControlNetPreprocessor(name='Canny', has_mask_input=False, parameters=[
        ParameterDef(key='low_threshold', default_value=0.4, required=True)])
    return PreprocessorParams(typedef=typedef)


def test_model_only_controlnet_unit_feeds_control_image_directly():
    """A model with no preprocessor, used for an already-preprocessed control image, applies the image as-is."""
    builder = make_txt2img_builder()
    control_image = ImageFileReference(filename='control.png', subfolder='IntraPaint')
    builder.add_controlnet_unit('control_canny.safetensors', None, control_image, 1.0, 0.0, 1.0)
    workflow = builder.build_workflow().get_workflow_dict()

    apply_node = single_node(workflow, 'ControlNetApplyAdvanced')
    load_image_key = next(key for key, node in workflow.items() if node['class_type'] == 'LoadImage')
    assert apply_node['inputs']['image'][0] == load_image_key
    assert workflow[load_image_key]['inputs']['image'] == 'IntraPaint/control.png'


def test_identical_preprocessor_and_image_share_one_preprocessor_node():
    """Units with equal preprocessor params and the same control image reuse one preprocessor node."""
    builder = make_txt2img_builder()
    control_image = ImageFileReference(filename='control.png', subfolder='IntraPaint')
    builder.add_controlnet_unit('model_a.safetensors', _canny_params(), control_image, 1.0, 0.0, 1.0)
    builder.add_controlnet_unit('model_b.safetensors', _canny_params(), control_image, 0.5, 0.0, 1.0)
    workflow = builder.build_workflow().get_workflow_dict()

    single_node(workflow, 'Canny')
    assert len(nodes_of_type(workflow, 'ControlNetApplyAdvanced')) == 2


def test_controlnet_unit_without_image_reuses_source_image():
    """A unit with no control image is fed the source image, matching the WebUI extension."""
    builder = make_txt2img_builder()
    builder.set_source_image_from_reference(ImageFileReference(filename='source.png', subfolder='IntraPaint'))
    builder.add_controlnet_unit('control_canny.safetensors', _canny_params(), None, 1.0, 0.0, 1.0)
    workflow = builder.build_workflow().get_workflow_dict()

    load_image = single_node(workflow, 'LoadImage')
    assert load_image['inputs']['image'] == 'IntraPaint/source.png'
    assert all(node['inputs'].get('image') != '' for node in nodes_of_type(workflow, 'LoadImage'))
    load_image_key = next(key for key, node in workflow.items() if node['class_type'] == 'LoadImage')
    assert single_node(workflow, 'Canny')['inputs']['image'][0] == load_image_key


def test_controlnet_unit_without_any_image_raises_naming_the_unit():
    """With neither a control image nor a source image, building fails instead of emitting LoadImage ''."""
    builder = make_txt2img_builder()
    builder.add_controlnet_unit('control_canny.safetensors', None, None, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match='control_canny.safetensors'):
        builder.build_workflow()


# LoRA and hypernetwork prompt tags

LORAS = ['detail.safetensors', 'styles/ink.safetensors']
HYPERNETWORKS = ['anime.pt']


def _load_tagged(prompt: str, negative: str = '') -> DiffusionWorkflowBuilder:
    builder = DiffusionWorkflowBuilder()
    builder.load_diffusion_parameters(
        ComfyUIDiffusionParams(prompt=prompt, negative_prompt=negative, sd_model_name='m.safetensors', seed=1),
        LORAS, HYPERNETWORKS)
    return builder


def _source_class(workflow: dict, connection: list) -> str:
    return workflow[connection[0]]['class_type']


def test_lora_tag_becomes_lora_loader_wired_into_model_and_clip():
    """A LoRA tag resolves to the server's file name, is stripped from the prompt, and feeds sampler and encoders."""
    builder = _load_tagged('a cat <lora:detail:0.8> sitting')
    assert builder.prompt == 'a cat  sitting'
    workflow = builder.build_workflow().get_workflow_dict()

    lora = single_node(workflow, 'LoraLoader')
    assert lora['inputs']['lora_name'] == 'detail.safetensors'
    assert lora['inputs']['strength_model'] == 0.8
    assert lora['inputs']['strength_clip'] == 0.8
    assert _source_class(workflow, lora['inputs']['model']) == 'CheckpointLoaderSimple'
    assert _source_class(workflow, single_node(workflow, 'KSampler')['inputs']['model']) == 'LoraLoader'
    for encoder in nodes_of_type(workflow, 'CLIPTextEncode'):
        assert _source_class(workflow, encoder['inputs']['clip']) == 'LoraLoader'
    assert {encoder['inputs']['text'] for encoder in nodes_of_type(workflow, 'CLIPTextEncode')} == {'a cat  sitting',
                                                                                                    ''}


def test_lora_tag_with_clip_weight_and_lyco_alias():
    """A second weight sets CLIP strength, and <lyco:...> is treated as a LoRA; full file names also match."""
    builder = _load_tagged('<lyco:styles/ink:0.5:0.25> <lora:detail.safetensors:1>')
    loras = nodes_of_type(builder.build_workflow().get_workflow_dict(), 'LoraLoader')
    assert [(n['inputs']['lora_name'], n['inputs']['strength_model'], n['inputs']['strength_clip']) for n in loras] \
        == [('styles/ink.safetensors', 0.5, 0.25), ('detail.safetensors', 1.0, 1.0)]


def test_hypernetwork_tag_chains_model_only():
    """A hypernetwork tag adds a HypernetworkLoader on the model path."""
    builder = _load_tagged('<hypernet:anime:0.6> a fox')
    workflow = builder.build_workflow().get_workflow_dict()
    hypernet = single_node(workflow, 'HypernetworkLoader')
    assert hypernet['inputs']['hypernetwork_name'] == 'anime.pt'
    assert hypernet['inputs']['strength'] == 0.6
    assert _source_class(workflow, single_node(workflow, 'KSampler')['inputs']['model']) == 'HypernetworkLoader'


def test_negative_prompt_tag_negates_strength():
    """A tag in the negative prompt applies with negated strength and is stripped from it."""
    builder = _load_tagged('a cat', 'blurry <lora:detail:0.5>')
    assert builder.negative_prompt == 'blurry '
    lora = single_node(builder.build_workflow().get_workflow_dict(), 'LoraLoader')
    assert lora['inputs']['strength_model'] == -0.5


@pytest.mark.parametrize('prompt, match', [
    ('a cat <lora:missing:0.8>', 'missing.*detail.safetensors'),
    ('a cat <hypernet:detail:0.8>', 'hypernetwork "detail"'),
    ('a cat <lora:detail:strong>', 'not a number'),
])
def test_unresolvable_tag_raises(prompt: str, match: str):
    """An unknown model or a non-numeric weight raises instead of silently dropping the tag."""
    with pytest.raises(ValueError, match=match):
        _load_tagged(prompt)


def test_tag_without_model_lists_raises():
    """Without the server's model lists, a tag cannot resolve and raises."""
    builder = DiffusionWorkflowBuilder()
    with pytest.raises(ValueError, match='Available LoRA models: none'):
        builder.load_diffusion_parameters(ComfyUIDiffusionParams(prompt='a cat <lora:foo:0.8>', sd_model_name='m'))


def _soft_inpainting_builder(mask: bool = True, **soft_inpainting_params: float) -> DiffusionWorkflowBuilder:
    builder = DiffusionWorkflowBuilder()
    builder.load_diffusion_parameters(
        ComfyUIDiffusionParams(prompt='a cat <lora:detail:0.8>', sd_model_name='m.safetensors', seed=1,
                               soft_inpainting=True, soft_inpainting_params=soft_inpainting_params),
        LORAS, HYPERNETWORKS)
    builder.source_image = 'src.png [input]'
    if mask:
        builder.mask = 'mask.png [input]'
    return builder


def test_soft_inpainting_samples_through_differential_diffusion_after_the_loras():
    """Soft inpainting patches the model after the LoRA loaders, and the sampler reads the patched model."""
    workflow = _soft_inpainting_builder().build_workflow().get_workflow_dict()
    patch = single_node(workflow, 'DifferentialDiffusion')
    assert _source_class(workflow, patch['inputs']['model']) == 'LoraLoader'
    assert _source_class(workflow, single_node(workflow, 'KSampler')['inputs']['model']) == 'DifferentialDiffusion'
    assert 'strength' not in patch['inputs']  # Older ComfyUI builds reject inputs they don't list.
    # The sampler's latent carries the noise mask that Differential Diffusion reads:
    sampled_latent = single_node(workflow, 'KSampler')['inputs']['latent_image']
    assert _source_class(workflow, workflow[sampled_latent[0]]['inputs']['samples']) == 'SetLatentNoiseMask'


def test_soft_inpainting_sends_a_set_strength():
    """A `strength` setting becomes the node's input."""
    workflow = _soft_inpainting_builder(strength=0.5).build_workflow().get_workflow_dict()
    assert single_node(workflow, 'DifferentialDiffusion')['inputs']['strength'] == 0.5


@pytest.mark.parametrize('strength', [0.0, 1.5])
def test_soft_inpainting_rejects_a_strength_comfyui_would_misread_or_reject(strength: float):
    """ComfyUI treats a strength of 0 as 1, and rejects one above 1."""
    with pytest.raises(ValueError, match='strength'):
        _soft_inpainting_builder(strength=strength)


def test_soft_inpainting_without_a_mask_leaves_the_model_unpatched():
    """Without a mask there is no noise mask to schedule, so the model is not patched."""
    workflow = _soft_inpainting_builder(mask=False).build_workflow().get_workflow_dict()
    assert nodes_of_type(workflow, 'DifferentialDiffusion') == []
