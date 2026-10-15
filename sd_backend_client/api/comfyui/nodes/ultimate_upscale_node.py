"""A ComfyUI node used to apply the 'Ultimate SD Upscale' script."""
import math
from typing import Any, ClassVar, Optional

from pydantic import Field

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output
from sd_backend_client.api.shared_data.api_datatypes import RedrawMode, SeamFixMode

ULTIMATE_UPSCALE_NODE_NAME = 'UltimateSDUpscale'
ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL = 'UltimateSDUpscaleNoUpscale'


def _round_length(length: float) -> int:
    """Round to the nearest multiple of 8 as the node's `usdu_patch.round_length` does, halves to even."""
    return round(length / 8) * 8


def ultimate_upscale_sampler_runs(image_width: int, image_height: int, upscale_by: float, tile_width: int,
                                  tile_height: int, redraw_mode: RedrawMode, seam_fix_mode: SeamFixMode) -> int:
    """Return how many times the node runs its sampler: once per redraw tile and once per seam-fix tile.

    `image_width` and `image_height` are the size of the node's image input. The tile grid covers that size times
    `upscale_by`, rounded to a multiple of 8, as the node's patched `USDUpscaler.__init__` computes it. The node's own
    console tile total skips that rounding and can disagree.

    This counts one tile per sampler run, which holds while `UltimateUpscaleNode` sends no `batch_size`. Batched
    redraw tiles share a sampler run.
    """
    rows = math.ceil(_round_length(image_height * upscale_by) / tile_height)
    cols = math.ceil(_round_length(image_width * upscale_by) / tile_width)
    redraw_runs = 0 if redraw_mode == 'None' else rows * cols
    half_tile_runs = rows * (cols - 1) + (rows - 1) * cols
    seam_fix_runs = {
        'None': 0,
        'Band Pass': (rows - 1) + (cols - 1),
        'Half Tile': half_tile_runs,
        'Half Tile + Intersections': half_tile_runs + (rows - 1) * (cols - 1),
    }[seam_fix_mode]
    return redraw_runs + seam_fix_runs


def _upscale_factor(num: int) -> int:
    """The script's `USDUpscaler.get_factor`: the upscale-model pass that `num` is divisible by, or 0 for none."""
    if num == 1:
        return 2
    for factor in (4, 3, 2):
        if num % factor == 0:
            return factor
    return 0


def ultimate_upscale_model_runs(image_width: int, image_height: int, upscale_by: float) -> int:
    """Return how many times the upscaler variant runs its upscale model, each before any sampling.

    The script's `USDUpscaler.get_factors` splits the whole-number scale factor into passes of 2x, 3x or 4x, each one
    call to `ImageUpscaleWithModel`. The variant without an upscale model resizes with PIL and makes no calls.
    """
    canvas_side = max(_round_length(image_width * upscale_by), _round_length(image_height * upscale_by))
    scale_factor = math.ceil(canvas_side / max(image_width, image_height))
    while _upscale_factor(scale_factor) == 0:
        scale_factor += 1
    runs, current_scale = 0, 1
    while current_scale < scale_factor:
        factor = _upscale_factor(scale_factor // current_scale)
        runs += 1  # The script still runs a pass for a factor of 0, then stops.
        if factor == 0:
            break
        current_scale *= factor
    return runs


class UltimateUpscaleNode(ComfyNode):
    """A ComfyUI node used to apply the 'Ultimate SD Upscale' script.

    `use_upscaler` picks the node variant. The variant without an upscale model leaves out `upscale_by`, takes no
    `upscale_model` connection, and names its image input `upscaled_image`.
    """
    CLASS_TYPE: ClassVar[str] = ULTIMATE_UPSCALE_NODE_NAME

    use_upscaler: bool = Field(True, exclude=True)

    upscale_by: Optional[float] = None  # Upscaler variant only.
    seed: int = 0
    steps: int = 20
    cfg: float = 8.0
    sampler_name: str = 'euler'
    scheduler: str = 'normal'
    denoise: float = 0.35
    mode_type: RedrawMode = 'Linear'
    tile_width: int = 512
    tile_height: int = 512
    mask_blur: int = 8
    tile_padding: int = 32
    force_uniform_tiles: bool = False
    tiled_decode: bool = True

    seam_fix_mode: SeamFixMode = 'None'
    seam_fix_denoise: float = 0.35
    seam_fix_width: int = 64
    seam_fix_mask_blur: int = 8
    seam_fix_padding: int = 16

    image: Connection = None  # Image source node
    model: Connection = None  # Usually CheckpointLoaderSimple
    positive: Connection = None  # Usually CLIPTextEncode
    negative: Connection = None  # Usually CLIPTextEncode
    vae: Connection = None
    upscale_model: Connection = None  # Upscaler variant only.

    image_out = Output(0)

    @property
    def class_type(self) -> str:
        return ULTIMATE_UPSCALE_NODE_NAME if self.use_upscaler else ULTIMATE_UPSCALE_NODE_WITHOUT_UPSCALE_MODEL

    def inputs(self) -> dict[str, Any]:
        inputs = super().inputs()
        if not self.use_upscaler:
            inputs.pop('upscale_by', None)
            if 'image' in inputs:
                inputs['upscaled_image'] = inputs.pop('image')
        return inputs
