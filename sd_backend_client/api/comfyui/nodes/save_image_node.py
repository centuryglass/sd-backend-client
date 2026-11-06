"""ComfyUI nodes that terminate a workflow by outputting its images."""
from typing import ClassVar

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection

# Prefix for images the package's workflows save on the server. ComfyUI names files `<prefix>_00001_.png`, so an
# empty prefix produces hidden dotfiles.
DEFAULT_FILENAME_PREFIX = 'sd_backend_client'


class SaveImageNode(ComfyNode):
    """A ComfyUI node used to save image data permanently to the server's output folder. It has no outputs."""
    CLASS_TYPE: ClassVar[str] = 'SaveImage'

    filename_prefix: str = DEFAULT_FILENAME_PREFIX
    images: Connection = None  # Image source, e.g. VAE decoder


class PreviewImageNode(ComfyNode):
    """A ComfyUI node that writes images to the server's temp folder, which ComfyUI clears on restart. It has no
    outputs, and the images stay downloadable through `/view` with type `temp` until then."""
    CLASS_TYPE: ClassVar[str] = 'PreviewImage'

    images: Connection = None  # Image source, e.g. VAE decoder


def output_image_node(images: Connection, save_output: bool,
                      filename_prefix: str = DEFAULT_FILENAME_PREFIX) -> SaveImageNode | PreviewImageNode:
    """Return the node that outputs `images`: a `SaveImageNode` when `save_output` is set, else a `PreviewImageNode`.

    `filename_prefix` only applies to saved images."""
    if save_output:
        return SaveImageNode(filename_prefix=filename_prefix, images=images)
    return PreviewImageNode(images=images)
