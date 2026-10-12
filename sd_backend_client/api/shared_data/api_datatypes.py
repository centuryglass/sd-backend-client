"""Shared definitions for generic API data types."""
from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict, Field

from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.util.api_defaults import UPSCALING_DENOISING_STRENGTH_DEFAULT, UPSCALING_STEP_COUNT_DEFAULT, \
    GENERATION_SIZE_DEFAULT

# Tiled-redraw modes, in the order both backends expect. ComfyUI's UltimateSDUpscale node takes the string directly;
# the WebUI "ultimate sd upscale" script takes the index into this list.
REDRAW_MODES: tuple[str, ...] = ('Linear', 'Chess', 'None')
RedrawMode = Literal['Linear', 'Chess', 'None']

# Seam-fix modes, in the order both backends expect. ComfyUI takes the string; the WebUI script takes the index. The
# WebUI script labels these slightly differently ('Band pass', 'Half tile offset pass', ...) but the ordering matches.
SEAM_FIX_MODES: tuple[str, ...] = ('None', 'Band Pass', 'Half Tile', 'Half Tile + Intersections')
SeamFixMode = Literal['None', 'Band Pass', 'Half Tile', 'Half Tile + Intersections']


class DiffusionUpscalingParams(BaseModel):
    """Dataclass defining parameters for Stable Diffusion upscaling.

    Covers the "Ultimate SD Upscale" workflow on both backends (ComfyUI's UltimateSDUpscale node and the WebUI script of
    the same name) plus the optional ControlNet tile pass. The diffusion pass itself (prompt, seed, cfg, sampler, model,
    ...) is carried by `diffusion_params`; the fields here are the upscale-specific tunables layered on top of it.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra='forbid')

    upscaling_mode: str = ""
    """Optional basic upscaler to use for basic upscaling or preprocessing before enhancing details via diffusion."""

    use_stable_diffusion_upscaling: bool = False
    """Whether to use tiled Stable Diffusion upscaling to refine details."""

    use_ultimate_upscale_script: bool = True
    """
    Whether to use the "Ultimate SD Upscale" script/node when use_stable_diffusion_upscaling=True. When False (or when
    the script/node is unavailable), backends fall back to resizing the image and refining it with one img2img pass
    at the full target size. That pass is not tiled, so it needs the VRAM for the whole image, and higher
    `denoising_strength` values can make the model repeat content across the larger canvas.
    """

    diffusion_params: DiffusionParams = Field(default_factory=DiffusionParams)
    """
    Core diffusion parameters (prompt, negative prompt, seed, cfg scale, sampler, scheduler, checkpoint, ...) for the
    detail-enhancing diffusion pass. `denoising_strength`/`step_count` below take precedence over the matching fields
    here, since upscaling typically uses a much lower denoising strength than a from-scratch generation.
    """

    denoising_strength: float = Field(default=UPSCALING_DENOISING_STRENGTH_DEFAULT, ge=0.0, le=1.0)
    """
    Denoising strength (range: 0.0, 1.0) to use when enhancing details via diffusion when
    use_stable_diffusion_upscaling=True
    """

    step_count: int = Field(default=UPSCALING_STEP_COUNT_DEFAULT, ge=1)
    """Number of detail-enhancing diffusion steps to apply per image tile when use_stable_diffusion_upscaling=True"""

    tile_width: int = Field(default=GENERATION_SIZE_DEFAULT, gt=0)
    """Width (pixels) for each tiled diffusion step when use_stable_diffusion_upscaling=True"""

    tile_height: int = Field(default=GENERATION_SIZE_DEFAULT, gt=0)
    """Height (pixels) for each tiled diffusion step when use_stable_diffusion_upscaling=True"""

    mask_blur: int = Field(default=8, ge=0)
    """Blur radius (pixels) applied to tile edges before compositing."""

    tile_padding: int = Field(default=32, ge=0)
    """Padding (pixels) added around each tile before diffusion."""

    redraw_mode: RedrawMode = 'Linear'
    """Order in which tiles are redrawn ('Linear', 'Chess', or 'None' to skip the redraw pass)."""

    force_uniform_tiles: bool = False
    """Force all tiles to be the same size, padding the image as needed. ComfyUI only."""

    tiled_decode: bool = True
    """Decode the latent in tiles to reduce VRAM use. ComfyUI only."""

    ### Seam-fix pass (applied after the tiled redraw to hide tile boundaries):
    seam_fix_mode: SeamFixMode = 'None'
    """Seam-fix strategy, or 'None' to skip the seam-fix pass."""

    seam_fix_denoise: float = Field(default=0.35, ge=0.0, le=1.0)
    """Denoising strength for the seam-fix pass, from 0.0 to 1.0."""

    seam_fix_width: int = Field(default=64, ge=0)
    """Width (pixels) of the band redrawn along seams during the seam-fix pass."""

    seam_fix_mask_blur: int = Field(default=8, ge=0)
    """Blur radius (pixels) applied to seam-fix masks."""

    seam_fix_padding: int = Field(default=16, ge=0)
    """Padding (pixels) added around seam-fix regions."""

    ### WebUI-only output options:
    save_upscaled_image: bool = False
    """Whether the WebUI script should separately save the upscaled image. WebUI only."""

    save_seams_fix_image: bool = False
    """Whether the WebUI script should separately save the seam-fix image. WebUI only."""

    tile_controlnet: Optional[ControlNetUnit] = None
    """Optional ControlNet to apply during tiled diffusion when use_stable_diffusion_upscaling=True"""
