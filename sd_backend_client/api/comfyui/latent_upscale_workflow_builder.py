"""Creates a ComfyUI workflow for tiled latent upscaling, using the Ultimate SD Upscale script, the ControlNet tile
   model, and a secondary upscaling model when possible."""
from copy import deepcopy
from typing import Optional

from sd_backend_client.api.shared_data.api_datatypes import RedrawMode, SeamFixMode
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.util.geometry import Size

from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.diffusion_workflow_builder import DiffusionWorkflowBuilder
from sd_backend_client.api.comfyui.nodes.apply_upscaler_node import ApplyUpscalerNode
from sd_backend_client.api.comfyui.nodes.basic_scaling_node import BasicScalingNode
from sd_backend_client.api.comfyui.nodes.comfy_node_graph import ComfyNodeGraph
from sd_backend_client.api.comfyui.nodes.image_scale_node import ImageScaleNode
from sd_backend_client.api.comfyui.nodes.input.load_image_node import LoadImageNode
from sd_backend_client.api.comfyui.nodes.input.load_upscaler_node import LoadUpscalerNode
from sd_backend_client.api.comfyui.nodes.ksampler_node import KSamplerNode
from sd_backend_client.api.comfyui.nodes.save_image_node import SaveImageNode
from sd_backend_client.api.comfyui.nodes.ultimate_upscale_node import UltimateUpscaleNode
from sd_backend_client.api.comfyui.nodes.vae.vae_decode_tiled_node import VAEDecodeTiledNode
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_tiled_node import (TILE_MIN, TILE_STEP, TILE_MAX,
                                                                           VAEEncodeTiledNode)
from sd_backend_client.api.comfyui.workflow_builder_utils import image_ref_to_str

class LatentUpscaleWorkflowBuilder(DiffusionWorkflowBuilder):
    """Creates a ComfyUI workflow for tiled latent upscaling, using the Ultimate SD Upscale script, the ControlNet tile
       model, and a secondary upscaling model when possible."""

    def __init__(self,
                 source_image: ImageFileReference,
                 upscale_by: float,
                 final_image_size: Size,
                 tile_size: Size,
                 ultimate_upscale_script_available: bool,
                 upscale_model: Optional[str] = None,
                 controlnet_tile_unit: Optional[ControlNetUnit] = None) -> None:
        super().__init__()
        self.source_image = image_ref_to_str(source_image)
        self._upscale_multiplier = upscale_by
        self._final_image_size = final_image_size
        self._ultimate_sd_upscale = ultimate_upscale_script_available
        self._upscale_model_name = upscale_model
        self._tile_size = Size(tile_size)

        # "Ultimate SD Upscale" tunables, applied to the UltimateSDUpscale node in build_workflow(). Callers override
        # these directly before building; defaults match the node's own sensible defaults.
        self.mask_blur = 8
        self.tile_padding = 32
        self.redraw_mode: RedrawMode = 'Linear'
        self.force_uniform_tiles = False
        self.tiled_decode = True
        self.seam_fix_mode: SeamFixMode = 'None'
        self.seam_fix_denoise = 0.35
        self.seam_fix_width = 64
        self.seam_fix_mask_blur = 8
        self.seam_fix_padding = 16

        if controlnet_tile_unit is not None and controlnet_tile_unit.model is not None:
            self.add_controlnet_unit(controlnet_tile_unit.model.full_model_name,
                                     controlnet_tile_unit.preprocessor, source_image,
                                     float(controlnet_tile_unit.control_strength),
                                     float(controlnet_tile_unit.control_start),
                                     float(controlnet_tile_unit.control_end))

    @property
    def tile_size(self) -> Size:
        """Access the tile size used for latent upscaling."""
        return self._tile_size

    @tile_size.setter
    def tile_size(self, size: Size) -> None:
        self._tile_size = Size(size)

    def build_workflow(self) -> ComfyNodeGraph:
        """Use the provided parameters to build a complete workflow graph.

        With Ultimate SD Upscale, the output is the source size times `upscale_by` on both axes, where `upscale_by`
        is the larger of the requested width and height ratios. A requested aspect ratio that differs from the
        source's is not honored on this path. Without it, the image goes through the upscale model (if any), is
        resized to the exact final size in pixel space, then refined by one img2img pass over the whole image. Only
        the VAE encode and decode are tiled on that path, so the sampler sees the full-size latent at once.
        """
        # Load model(s):
        sd_model, clip, vae = self._load_checkpoint()
        sd_model, clip = self._apply_extension_models(sd_model, clip)

        # Load starting image:
        source_image_str = self.source_image
        if source_image_str is None:
            raise ValueError('No image provided for upscaling')
        image = LoadImageNode(image=source_image_str).image_out

        # Load prompt conditioning:
        positive, negative = self._encode_prompts(clip)

        # Load tile ControlNet unit, if available:
        controlnet_units = self.controlnet_unit_nodes
        assert len(controlnet_units) <= 1, f'Expected at most one controlNet unit, found {len(controlnet_units)}'
        if len(controlnet_units) > 0:
            positive, negative = self._wire_controlnet_unit(controlnet_units[0], image, None, positive, negative,
                                                            vae)

        # Load upscale model node, if available:
        upscale_model_node: Optional[LoadUpscalerNode] = None
        if self._upscale_model_name is not None and self._upscale_model_name != '':
            upscale_model_node = LoadUpscalerNode(model_name=self._upscale_model_name)
        elif self._ultimate_sd_upscale:
            # Use a basic scaling node to get the image to the right size instead.
            image = BasicScalingNode(scale_by=self._upscale_multiplier, image=image).image_out

        # Set up ultimate upscale script, if available:
        if self._ultimate_sd_upscale:
            ultimate_upscale_node = UltimateUpscaleNode(
                use_upscaler=upscale_model_node is not None,
                upscale_by=self._upscale_multiplier if upscale_model_node is not None else None,
                seed=self.seed,
                steps=self.steps,
                cfg=self.cfg_scale,
                sampler_name=self.sampler,
                scheduler=self.scheduler,
                denoise=self.denoising_strength,
                mode_type=self.redraw_mode,
                tile_width=self.tile_size.width(),
                tile_height=self.tile_size.height(),
                mask_blur=self.mask_blur,
                tile_padding=self.tile_padding,
                force_uniform_tiles=self.force_uniform_tiles,
                tiled_decode=self.tiled_decode,
                seam_fix_mode=self.seam_fix_mode,
                seam_fix_denoise=self.seam_fix_denoise,
                seam_fix_width=self.seam_fix_width,
                seam_fix_mask_blur=self.seam_fix_mask_blur,
                seam_fix_padding=self.seam_fix_padding,
                image=image,
                model=sd_model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_model=upscale_model_node.upscale_model_out if upscale_model_node is not None else None)
            image = ultimate_upscale_node.image_out

        else:  # No ultimate SD upscale, we'll try to get by with img2img with tiled VAE encoding/decoding.
            if upscale_model_node is not None:
                image = ApplyUpscalerNode(upscale_model=upscale_model_node.upscale_model_out, image=image).image_out
            # Resize in pixel space: a latent-space resize decodes to a blurred, mottled image that a low-denoise
            # img2img pass can't recover.
            image = ImageScaleNode(width=self._final_image_size.width(), height=self._final_image_size.height(),
                                   image=image).image_out

            vae_tile_size = self.vae_tile_size
            vae_tile_size -= (vae_tile_size % TILE_STEP)
            if vae_tile_size < TILE_MIN:
                vae_tile_size = TILE_MIN
            elif vae_tile_size > TILE_MAX:
                vae_tile_size = TILE_MAX
            latent = VAEEncodeTiledNode(tile_size=vae_tile_size, pixels=image, vae=vae).latent_out
            sampler_node = KSamplerNode(cfg=self.cfg_scale, steps=self.steps, sampler_name=self.sampler,
                                        denoise=self.denoising_strength, scheduler=self.scheduler, seed=self.seed,
                                        latent_image=latent, model=sd_model, positive=positive, negative=negative)
            image = VAEDecodeTiledNode(tile_size=vae_tile_size, vae=vae, samples=sampler_node.latent_out).image_out

        workflow = ComfyNodeGraph()
        workflow.add_node(SaveImageNode(filename_prefix=self.filename_prefix, images=image))
        # Changes to the returned graph shouldn't affect the workflow builder, so create a deep copy to return:
        final_workflow = deepcopy(workflow)
        self._clear_saved_node_connections()
        return final_workflow
