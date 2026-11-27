"""`DiffusionParams` with the fields only the ComfyUI client supports."""
import logging
from typing import Optional

from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams

logger = logging.getLogger(__name__)


class ComfyUIDiffusionParams(DiffusionParams):
    """`DiffusionParams` plus the fields only ComfyUI supports, for txt2img, img2img and inpainting.

    `ComfyUiWebservice` reads these fields when it builds the generation workflow. `A1111Webservice` accepts an
    instance and ignores them.
    """

    load_as_inpainting_model: bool = False
    """Whether inpainting conditions the checkpoint as a dedicated inpainting model.

    When on and a mask is set, the workflow encodes the source and mask with ComfyUI's `InpaintModelConditioning`
    node, which inpainting checkpoints need. Normal checkpoints should leave it off.
    """

    vae_tiling_enabled: bool = False
    """Whether the VAE encodes and decodes images in tiles of `vae_tile_size`, to lower VRAM use on large images."""

    vae_tile_size: int = 512
    """Tile width and height in pixels when `vae_tiling_enabled` is on.

    ComfyUI's tiled VAE nodes take a multiple of 64 from 320 to 4096. Another value raises `ValueError` when the
    workflow is built.
    """

    clip_skip: int = 1
    """Which CLIP text encoder layer, counting back from the last, encodes the prompts. 1 uses the last layer.

    It matches WebUI's "Clip skip" setting: 2 stops at the second-to-last layer. Values above 1 add a
    `CLIPSetLastLayer` node. Some checkpoints, often anime-style SD 1.x models, expect 2.
    """

    sd_model_config: Optional[str] = None
    """Model config file to load the checkpoint with, from `ComfyUiWebservice.list_model_configs`.

    None, or a name the server lacks, selects a config whose name matches the checkpoint's, or the server's
    automatic detection when none matches.
    """

