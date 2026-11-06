"""Creates a ComfyUI workflow used to apply an upscaling model."""

from typing import Optional

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.nodes.apply_upscaler_node import ApplyUpscalerNode
from sd_backend_client.api.comfyui.nodes.comfy_node_graph import ComfyNodeGraph
from sd_backend_client.api.comfyui.nodes.image_scale_node import ImageScaleNode
from sd_backend_client.api.comfyui.nodes.input.load_image_node import LoadImageNode
from sd_backend_client.api.comfyui.nodes.input.load_upscaler_node import LoadUpscalerNode
from sd_backend_client.api.comfyui.nodes.save_image_node import output_image_node
from sd_backend_client.api.comfyui.workflow_builder_utils import image_ref_to_str
from sd_backend_client.util.geometry import Size


def build_basic_upscaling_workflow(source_image: ImageFileReference, upscale_model_name: str,
                                  final_image_size: Optional[Size] = None, save_output: bool = False) -> ComfyNodeGraph:
    """Creates a ComfyUI workflow that applies an upscaling model, then resizes the result to `final_image_size`.

    The model's native scale factor rarely matches the requested size, so the resize step sets the output size.
    When `final_image_size` is None the resize is skipped and the output has the model's native scale.

    The result goes to the server's temp folder unless `save_output` is True, which saves it to the output folder."""
    image = LoadImageNode(image=image_ref_to_str(source_image)).image_out
    upscale_model = LoadUpscalerNode(model_name=upscale_model_name).upscale_model_out
    image = ApplyUpscalerNode(upscale_model=upscale_model, image=image).image_out
    if final_image_size is not None:
        image = ImageScaleNode(width=final_image_size.width(), height=final_image_size.height(), image=image).image_out
    workflow = ComfyNodeGraph()
    workflow.add_node(output_image_node(image, save_output))
    return workflow
