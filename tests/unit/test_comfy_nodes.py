"""Unit tests for the ComfyUI node models and ComfyNodeGraph: input fields, output slots, keys and wiring."""
from copy import deepcopy

import pytest

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.nodes.clip_skip_node import CLIPSkipNode
from sd_backend_client.api.comfyui.nodes.comfy_node import NodeOutput, Output
from sd_backend_client.api.comfyui.nodes.comfy_node_graph import ComfyNodeGraph
from sd_backend_client.api.comfyui.nodes.controlnet.dynamic_preprocessor_node import DynamicPreprocessorNode
from sd_backend_client.api.comfyui.nodes.input.clip_text_encode_node import ClipTextEncodeNode
from sd_backend_client.api.comfyui.nodes.input.load_image_node import LoadImageNode
from sd_backend_client.api.comfyui.nodes.input.simple_checkpoint_loader_node import SimpleCheckpointLoaderNode
from sd_backend_client.api.comfyui.nodes.save_image_node import SaveImageNode
from sd_backend_client.api.comfyui.nodes.ultimate_upscale_node import (ULTIMATE_UPSCALE_NODE_NAME,
                                                                       ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL,
                                                                       UltimateUpscaleNode)
from sd_backend_client.api.comfyui.nodes.vae.vae_decode_node import VAEDecodeNode
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_tiled_node import TILE_MAX, TILE_MIN, VAEEncodeTiledNode
from sd_backend_client.api.comfyui.preprocessor_preview_workflow_builder import PreprocessorPreviewWorkflowBuilder
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor


def test_output_reads_as_node_output_on_instances():
    """An `Output` read from a node is that node's `NodeOutput` for the slot."""
    loader = SimpleCheckpointLoaderNode(ckpt_name='m.safetensors')
    assert isinstance(SimpleCheckpointLoaderNode.vae_out, Output)
    assert loader.vae_out == NodeOutput(loader, 2)
    assert loader.vae_out != SimpleCheckpointLoaderNode(ckpt_name='m.safetensors').vae_out


def test_unknown_inputs_are_rejected():
    """A misspelled input fails at construction or assignment."""
    with pytest.raises(ValueError):
        ClipTextEncodeNode(text='a cat', clipp=None)  # type: ignore[call-arg]
    node = ClipTextEncodeNode(text='a cat')
    with pytest.raises(ValueError):
        node.not_an_input = LoadImageNode(image='x.png').image_out


def test_connection_inputs_reject_non_outputs():
    """Connection inputs take only a `NodeOutput`, never a raw `(key, slot)` pair."""
    with pytest.raises(ValueError):
        ClipTextEncodeNode(text='a cat', clip=('3', 1))  # type: ignore[arg-type]


def test_workflow_dict_numbers_upstream_nodes_first_and_includes_shared_nodes_once():
    """Keys count up from 3 in dependency order, and connected nodes join the graph once each."""
    loader = SimpleCheckpointLoaderNode(ckpt_name='m.safetensors')
    positive = ClipTextEncodeNode(text='p', clip=loader.clip_out)
    negative = ClipTextEncodeNode(text='n', clip=loader.clip_out)
    decode = VAEDecodeNode(vae=loader.vae_out, samples=positive.conditioning_out)
    graph = ComfyNodeGraph()
    graph.add_node(SaveImageNode(images=decode.image_out))
    graph.add_node(negative)

    assert graph.get_workflow_dict() == {
        '3': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'm.safetensors'}},
        '4': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'p', 'clip': ('3', 1)}},
        '5': {'class_type': 'VAEDecode', 'inputs': {'samples': ('4', 0), 'vae': ('3', 2)}},
        '6': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'sd_backend_client', 'images': ('5', 0)}},
        '7': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'n', 'clip': ('3', 1)}},
    }


def test_unconnected_inputs_are_left_out():
    """An unset connection input is omitted from the node dict."""
    assert ClipTextEncodeNode(text='p').get_dict({}) == {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'p'}}


def test_connection_cycle_raises():
    """A connection cycle is an error, not infinite recursion."""
    first = VAEDecodeNode()
    second = VAEDecodeNode(samples=first.image_out)
    first.samples = second.image_out
    graph = ComfyNodeGraph()
    graph.add_node(second)
    with pytest.raises(ValueError, match='cycle'):
        graph.get_workflow_dict()


def test_graph_deepcopy_is_independent_and_keeps_shared_nodes_shared():
    """A copied graph ignores later edits to the original nodes and keeps its shape."""
    loader = SimpleCheckpointLoaderNode(ckpt_name='m.safetensors')
    text = ClipTextEncodeNode(text='p', clip=loader.clip_out)
    decode = VAEDecodeNode(vae=loader.vae_out, samples=text.conditioning_out)
    graph = ComfyNodeGraph()
    graph.add_node(decode)
    expected = graph.get_workflow_dict()

    graph_copy = deepcopy(graph)
    text.clear_connections()
    loader.ckpt_name = 'other.safetensors'

    assert text.clip is None
    assert graph_copy.get_workflow_dict() == expected
    assert len(graph_copy.get_workflow_dict()) == 3


def test_clip_skip_is_sent_as_a_negative_layer_index():
    """A positive CLIP skip becomes the negative layer index the node expects."""
    assert CLIPSkipNode(stop_at_clip_layer=2).stop_at_clip_layer == -2
    assert CLIPSkipNode(stop_at_clip_layer=-3).stop_at_clip_layer == -3


@pytest.mark.parametrize('tile_size', [TILE_MIN - 64, TILE_MAX + 64, TILE_MIN + 1])
def test_vae_tile_size_must_be_in_range_and_on_step(tile_size: int):
    """Tiled VAE nodes reject tile sizes out of range or off the step."""
    with pytest.raises(ValueError):
        VAEEncodeTiledNode(tile_size=tile_size)


def test_dynamic_preprocessor_sends_its_name_parameters_and_connections():
    """The preprocessor node sends its name as the class type and its parameters as inputs."""
    image = LoadImageNode(image='x.png')
    node = DynamicPreprocessorNode(node_name='Canny', parameters={'low_threshold': 0.4}, image=image.image_out)
    assert node.get_dict({image: '3'}) == {'class_type': 'Canny',
                                           'inputs': {'low_threshold': 0.4, 'image': ('3', 0)}}


def test_dynamic_preprocessor_rejects_inputs_it_does_not_take():
    """Connecting an image or mask the preprocessor does not take fails."""
    image = LoadImageNode(image='x.png').image_out
    with pytest.raises(ValueError):
        DynamicPreprocessorNode(node_name='MaskOnly', parameters={}, has_image_input=False, image=image)
    node = DynamicPreprocessorNode(node_name='Canny', parameters={})
    with pytest.raises(ValueError):
        node.mask = image


def test_ultimate_upscale_variant_without_model_renames_image_and_drops_upscale_by():
    """The no-upscaler variant changes the class type and image input name and drops `upscale_by`."""
    image = LoadImageNode(image='x.png')
    with_model = UltimateUpscaleNode(upscale_by=2.0, image=image.image_out)
    without_model = UltimateUpscaleNode(use_upscaler=False, upscale_by=2.0, image=image.image_out)

    assert with_model.class_type == ULTIMATE_UPSCALE_NODE_NAME
    assert with_model.get_dict({image: '3'})['inputs']['image'] == ('3', 0)
    assert with_model.get_dict({image: '3'})['inputs']['upscale_by'] == 2.0
    inputs = without_model.get_dict({image: '3'})['inputs']
    assert without_model.class_type == ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL
    assert inputs['upscaled_image'] == ('3', 0)
    assert 'image' not in inputs and 'upscale_by' not in inputs and 'use_upscaler' not in inputs


def test_preprocessor_preview_rebuild_without_mask_drops_the_mask():
    """Each preview build wires its own mask, so a later build without one sends none."""
    builder = PreprocessorPreviewWorkflowBuilder(ControlNetPreprocessor(name='Inpaint', has_mask_input=True))
    source = ImageFileReference(filename='source.png', subfolder='')
    mask = ImageFileReference(filename='mask.png', subfolder='')

    with_mask = builder.build_workflow(source, mask).get_workflow_dict()
    without_mask = builder.build_workflow(source).get_workflow_dict()

    assert 'LoadImageMask' in {node['class_type'] for node in with_mask.values()}
    assert 'LoadImageMask' not in {node['class_type'] for node in without_mask.values()}
    preprocessor = next(node for node in without_mask.values() if node['class_type'] == 'Inpaint')
    assert 'mask' not in preprocessor['inputs']


def test_preprocessor_preview_output_node_follows_save_output():
    """A preview goes to the temp folder by default and to the output folder, under its own prefix, when saving."""
    source = ImageFileReference(filename='source.png', subfolder='')
    preprocessor = ControlNetPreprocessor(name='Canny')

    temp = PreprocessorPreviewWorkflowBuilder(preprocessor).build_workflow(source).get_workflow_dict()
    saved_builder = PreprocessorPreviewWorkflowBuilder(preprocessor, save_output=True)
    saved = saved_builder.build_workflow(source).get_workflow_dict()

    assert {node['class_type'] for node in temp.values()} >= {'PreviewImage'}
    assert 'SaveImage' not in {node['class_type'] for node in temp.values()}
    save = next(node for node in saved.values() if node['class_type'] == 'SaveImage')
    assert save['inputs']['filename_prefix'] == 'Canny_preview'
    assert 'PreviewImage' not in {node['class_type'] for node in saved.values()}
