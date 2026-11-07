"""Utility functions for ComfyUI workflow builders."""
import random

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference

random.seed()


MAX_SEED = 0xffffffffffffffff


def random_seed() -> int:
    """Gets a random seed within the expected range valid seeds accepted by ComfyUI."""
    return random.randrange(0, MAX_SEED)


def image_ref_to_str(image_ref: ImageFileReference) -> str:
    """Converts an uploaded image/mask file reference to the string ComfyUI's load nodes take.

    A reference to a temp-folder upload gets the ` [temp]` annotation, which tells the load nodes to read the temp
    folder instead of the input folder."""
    name = image_ref.filename
    if image_ref.subfolder:
        name = f'{image_ref.subfolder}/{name}'
    if image_ref.type == 'temp':
        name = f'{name} [temp]'
    return name
