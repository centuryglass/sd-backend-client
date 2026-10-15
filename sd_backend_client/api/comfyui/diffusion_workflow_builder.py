"""Unified class for building text to image, image to image, and inpainting ComfyUI workflows."""
import os
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui.comfyui_types import ImageFileReference
from sd_backend_client.api.comfyui.nodes.clip_skip_node import CLIPSkipNode
from sd_backend_client.api.comfyui.nodes.comfy_node import NodeOutput
from sd_backend_client.api.comfyui.nodes.comfy_node_graph import ComfyNodeGraph
from sd_backend_client.api.comfyui.nodes.controlnet.apply_controlnet_node import ApplyControlNetNode
from sd_backend_client.api.comfyui.nodes.controlnet.dynamic_preprocessor_node import DynamicPreprocessorNode
from sd_backend_client.api.comfyui.nodes.controlnet.load_controlnet_node import LoadControlNetNode
from sd_backend_client.api.comfyui.nodes.image_scale_node import ImageScaleNode
from sd_backend_client.api.comfyui.nodes.inpaint_model_conditioning_node import InpaintModelConditioningNode
from sd_backend_client.api.comfyui.nodes.input.checkpoint_loader_node import CheckpointLoaderNode
from sd_backend_client.api.comfyui.nodes.input.clip_text_encode_node import ClipTextEncodeNode
from sd_backend_client.api.comfyui.nodes.input.empty_latent_image_node import EmptyLatentNode
from sd_backend_client.api.comfyui.nodes.input.load_image_mask_node import LoadImageMaskNode
from sd_backend_client.api.comfyui.nodes.input.load_image_node import LoadImageNode
from sd_backend_client.api.comfyui.nodes.input.simple_checkpoint_loader_node import SimpleCheckpointLoaderNode
from sd_backend_client.api.comfyui.nodes.ksampler_node import KSamplerNode
from sd_backend_client.api.comfyui.nodes.latent_mask_node import LatentMaskNode
from sd_backend_client.api.comfyui.nodes.model_extensions.differential_diffusion_node import (
    STRENGTH_KEY, DifferentialDiffusionNode)
from sd_backend_client.api.comfyui.nodes.model_extensions.hypernet_loader_node import HypernetLoaderNode
from sd_backend_client.api.comfyui.nodes.model_extensions.lora_loader_node import LoraLoaderNode
from sd_backend_client.api.comfyui.nodes.repeat_latent_node import RepeatLatentNode
from sd_backend_client.api.comfyui.nodes.save_image_node import DEFAULT_FILENAME_PREFIX, SaveImageNode
from sd_backend_client.api.comfyui.nodes.vae.vae_decode_node import VAEDecodeNode
from sd_backend_client.api.comfyui.nodes.vae.vae_decode_tiled_node import VAEDecodeTiledNode
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_node import VAEEncodeNode
from sd_backend_client.api.comfyui.nodes.vae.vae_encode_tiled_node import VAEEncodeTiledNode
from sd_backend_client.api.comfyui.workflow_builder_utils import random_seed, image_ref_to_str
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import PreprocessorParams
from sd_backend_client.api.shared_data.diffusion_params import DEFAULT_DENOISING_STRENGTH, DiffusionParams
from sd_backend_client.api.shared_data.sampler_names import comfyui_sampler_name, comfyui_scheduler_name
from sd_backend_client.util.geometry import Size

DEFAULT_STEP_COUNT = 30
DEFAULT_CFG = 8.0
DEFAULT_SIZE = 512
DEFAULT_SAMPLER = 'euler'
DEFAULT_SCHEDULER = 'normal'
MAX_BATCH_SIZE = 64

# Prompt tag syntax WebUI uses for extra networks: <lora|lyco|hypernet:name:weight[:clip_weight]>
EXTENSION_MODEL_PATTERN = r'<(lora|lyco|hypernet):([^:><]+):([^:>]+)(?::([^>]+))?>'


class ExtensionModelType(Enum):
    """Distinguishes between the two types of accepted extension model."""
    LORA = 0
    HYPERNETWORK = 1


@dataclass
class ControlNetNodeData:
    """All nodes associated with a single ControlNet input."""
    model_node: Optional[LoadControlNetNode]
    preprocessor_node: Optional[DynamicPreprocessorNode]
    control_apply_node: ApplyControlNetNode
    preprocessor: Optional[PreprocessorParams]
    control_image: str


class DiffusionWorkflowBuilder:
    """Unified class for building text to image, image to image, and inpainting ComfyUI workflows."""

    def __init__(self) -> None:
        self._batch_size = 1
        self._prompt = ''
        self._negative = ''
        self._steps = DEFAULT_STEP_COUNT
        self._cfg_scale = DEFAULT_CFG
        self._size = Size(DEFAULT_SIZE, DEFAULT_SIZE)
        self._sd_model = ''
        self._denoising = 1.0
        self._sampler: str = DEFAULT_SAMPLER
        self._scheduler: str = DEFAULT_SCHEDULER
        self._seed = random_seed()
        self._filename_prefix = DEFAULT_FILENAME_PREFIX

        # Optional params that can be set to alter diffusion behavior:
        self._load_as_inpainting_model = False
        self._clip_skip = 1
        self._vae_tiling = False
        self._vae_tile_size = DEFAULT_SIZE
        self._model_config: Optional[str] = None
        self._source_image: Optional[str] = None
        self._mask: Optional[str] = None
        self._differential_diffusion = False
        self._differential_diffusion_strength: Optional[float] = None

        self._extension_model_nodes: list[LoraLoaderNode | HypernetLoaderNode] = []
        self._controlnet_units: list[ControlNetNodeData] = []

    @property
    def batch_size(self) -> int:
        """Access the number of images to generate in the diffusion batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int) -> None:
        if size < 1 or size > MAX_BATCH_SIZE:
            raise ValueError(f'Batch size {size} not in range 1-{MAX_BATCH_SIZE}')
        self._batch_size = size

    @property
    def cfg_scale(self) -> float:
        """Accesses the prompt adherence scale value."""
        return self._cfg_scale

    @cfg_scale.setter
    def cfg_scale(self, cfg_scale: float) -> None:
        if cfg_scale < 0.0:
            raise ValueError('cfg_scale must be positive.')
        self._cfg_scale = cfg_scale

    @property
    def clip_skip(self) -> int:
        """Accesses the 'CLIP skip' value: the image generation step (counting from the last step backwards) where the
           CLIP model is disabled."""
        return self._clip_skip

    @clip_skip.setter
    def clip_skip(self, clip_skip: int) -> None:
        self._clip_skip = clip_skip

    @property
    def denoising_strength(self) -> float:
        """Accesses the denoising fraction (0.0 to 1.0). Should only be set below 1.0 when using a source image."""
        return self._denoising

    @denoising_strength.setter
    def denoising_strength(self, denoising: float) -> None:
        if denoising < 0.0 or denoising > 1.0:
            raise ValueError(f'Denoising strength {denoising} out of range 0.0-1.0')
        self._denoising = denoising

    @property
    def sd_model(self) -> str:
        """Accesses the Stable Diffusion model used for image generation."""
        return self._sd_model

    @sd_model.setter
    def sd_model(self, model: str) -> None:
        self._sd_model = model

    @property
    def load_as_inpainting_model(self) -> bool:
        """Accesses whether the workflow should be configured for a dedicated inpainting model."""
        return self._load_as_inpainting_model

    @load_as_inpainting_model.setter
    def load_as_inpainting_model(self, is_inpainting_model: bool) -> None:
        self._load_as_inpainting_model = is_inpainting_model

    @property
    def model_config_path(self) -> Optional[str]:
        """Accesses the config path used for advanced model loading."""
        return self._model_config

    @model_config_path.setter
    def model_config_path(self, path: Optional[str]) -> None:
        self._model_config = path

    @property
    def prompt(self) -> str:
        """Accesses the prompt used to guide image generation."""
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str) -> None:
        self._prompt = prompt

    @property
    def negative_prompt(self) -> str:
        """Accesses the negative prompt used to guide image generation."""
        return self._negative

    @negative_prompt.setter
    def negative_prompt(self, prompt: str) -> None:
        self._negative = prompt

    @property
    def sampler(self) -> str:
        """Accesses the sampling algorithm used for the diffusion process."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: str) -> None:
        self._sampler = sampler

    @property
    def scheduler(self) -> str:
        """Accesses the scheduler used to control the magnitude of individual diffusion steps."""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: str) -> None:
        self._scheduler = scheduler

    @property
    def seed(self) -> int:
        """Accesses the seed used to control diffusion randomization.  Setting any value less than zero selects a new
           random seed."""
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        if seed < 0:
            seed = random_seed()
        self._seed = seed

    @property
    def steps(self) -> int:
        """Accesses the number of diffusion steps to take."""
        return self._steps

    @steps.setter
    def steps(self, steps: int) -> None:
        if steps < 1:
            raise ValueError('Step count must be 1 or greater.')
        self._steps = steps

    @property
    def source_image(self) -> Optional[str]:
        """Accesses the initial image to use as initial latent data. If None, empty latent data is used.  Otherwise,
           this needs to be an image that was already uploaded to ComfyUI."""
        return self._source_image

    @source_image.setter
    def source_image(self, source_image: Optional[str]) -> None:
        self._source_image = source_image

    def set_source_image_from_reference(self, source_image: ImageFileReference) -> None:
        """Converts a ComfyUI API image reference to an appropriate string format and assigns it to source_image."""
        self.source_image = image_ref_to_str(source_image)

    @property
    def mask(self) -> Optional[str]:
        """Accesses the inpainting mask to apply to the source image.  If None, no mask is used. Otherwise, this needs
           to be a mask that was already uploaded to ComfyUI, and source_image should be set to the matching image
           specified when the mask was uploaded."""
        return self._mask

    @mask.setter
    def mask(self, mask: Optional[str]) -> None:
        self._mask = mask

    def set_mask_from_reference(self, mask_reference: ImageFileReference) -> None:
        """Converts a ComfyUI API image reference to an appropriate string format and assigns it to the mask."""
        self.mask = image_ref_to_str(mask_reference)

    @property
    def differential_diffusion(self) -> bool:
        """Accesses whether a masked workflow samples through a `DifferentialDiffusion` model patch.

        The patch makes the grey levels of the mask a per-pixel denoising schedule. Workflows without both a source
        image and a mask leave it out.
        """
        return self._differential_diffusion

    @differential_diffusion.setter
    def differential_diffusion(self, enabled: bool) -> None:
        self._differential_diffusion = enabled

    @property
    def differential_diffusion_strength(self) -> Optional[float]:
        """Accesses the `DifferentialDiffusion` node's `strength` input, or None to leave the input out."""
        return self._differential_diffusion_strength

    @differential_diffusion_strength.setter
    def differential_diffusion_strength(self, strength: Optional[float]) -> None:
        if strength is not None and not 0.0 < strength <= 1.0:
            raise ValueError(f'Differential Diffusion strength {strength} out of range: expected 0 < strength <= 1')
        self._differential_diffusion_strength = strength

    @property
    def image_size(self) -> Size:
        """Accesses the generated image size. A source image is stretched to this size before encoding."""
        return self._size

    @image_size.setter
    def image_size(self, size: Size) -> None:
        self._size = Size(size)

    @property
    def filename_prefix(self) -> str:
        """Accesses the prefix used when saving image files."""
        return self._filename_prefix

    @filename_prefix.setter
    def filename_prefix(self, prefix: str) -> None:
        self._filename_prefix = prefix

    @property
    def vae_tiling_enabled(self) -> bool:
        """Accesses whether tiled VAE encoding/decoding will be used."""
        return self._vae_tiling

    @vae_tiling_enabled.setter
    def vae_tiling_enabled(self, use_tiling: bool) -> None:
        self._vae_tiling = use_tiling

    @property
    def vae_tile_size(self) -> int:
        """Accesses the VAE tile resolution used if VAE tiling is enabled."""
        return self._vae_tile_size

    @vae_tile_size.setter
    def vae_tile_size(self, tile_size: int) -> None:
        self._vae_tile_size = tile_size

    def add_extension_model(self, model_name: str, model_strength: float, clip_strength: float,
                            model_type: ExtensionModelType) -> None:
        """Adds a LoRA or Hypernetwork model to the workflow. Models are applied in the order that they're added."""
        if model_type == ExtensionModelType.LORA:
            self._extension_model_nodes.append(LoraLoaderNode(lora_name=model_name, strength_model=model_strength,
                                                              strength_clip=clip_strength))
        else:
            self._extension_model_nodes.append(HypernetLoaderNode(hypernetwork_name=model_name,
                                                                  strength=model_strength))

    @property
    def extension_model_nodes(self) -> list[LoraLoaderNode | HypernetLoaderNode]:
        """Returns the list of extension model nodes, in the order that they should be applied."""
        return [*self._extension_model_nodes]

    @property
    def controlnet_unit_nodes(self) -> list[ControlNetNodeData]:
        """Returns the list of ControlNet unit nodes, with associated data."""
        return [*self._controlnet_units]

    def add_controlnet_unit(self,
                            model_name: Optional[str],
                            preprocessor: Optional[PreprocessorParams],
                            control_image_ref: Optional[ImageFileReference],
                            strength: float, start_step: float, end_step: float) -> None:
        """Adds a new ControlNet unit to the workflow."""
        control_image_str = '' if control_image_ref is None else image_ref_to_str(control_image_ref)

        model_node: Optional[LoadControlNetNode] = None
        preprocessor_node: Optional[DynamicPreprocessorNode] = None

        # Check for and reuse identical preprocessor nodes or control models:
        for control_unit_data in self._controlnet_units:
            if control_unit_data.preprocessor == preprocessor \
                    and control_unit_data.control_image == control_image_str:
                preprocessor_node = control_unit_data.preprocessor_node
            if control_unit_data.model_node is not None and control_unit_data.model_node.control_net_name == model_name:
                model_node = control_unit_data.model_node
        if model_node is None and model_name is not None:
            model_node = LoadControlNetNode(control_net_name=model_name)
        if preprocessor_node is None and preprocessor is not None:
            control_inputs = dict(preprocessor.parameter_values)
            preprocessor_node = DynamicPreprocessorNode(node_name=preprocessor.typedef.name, parameters=control_inputs,
                                                        has_image_input=preprocessor.typedef.has_image_input,
                                                        has_mask_input=preprocessor.typedef.has_mask_input)
        control_apply_node = ApplyControlNetNode(strength=strength, start_percent=start_step, end_percent=end_step)
        new_control_unit = ControlNetNodeData(model_node, preprocessor_node, control_apply_node, preprocessor,
                                               control_image_str)
        self._controlnet_units.append(new_control_unit)

    def build_workflow(self) -> ComfyNodeGraph:
        """Use the provided parameters to build a complete workflow graph."""
        # Load model(s):
        sd_model, clip, vae = self._load_checkpoint()
        sd_model, clip = self._apply_extension_models(sd_model, clip)
        if self.differential_diffusion and self.source_image is not None and self.mask is not None:
            sd_model = DifferentialDiffusionNode(model=sd_model,
                                                 strength=self.differential_diffusion_strength).model_out

        # Load prompt conditioning:
        positive, negative = self._encode_prompts(clip)

        # Load image source:
        mask_load_node: Optional[LoadImageMaskNode] = None
        mask = self.mask
        if mask is not None:
            mask_load_node = LoadImageMaskNode(image=mask)

        source_image = self.source_image
        if source_image is None:
            image_loading_node: Optional[LoadImageNode] = None
            latent = EmptyLatentNode(batch_size=self.batch_size, width=self.image_size.width(),
                                     height=self.image_size.height()).latent_out
        else:
            image_loading_node = LoadImageNode(image=source_image)
            # Stretch the source to the requested size, matching WebUI's default resize mode:
            scaled_image = ImageScaleNode(width=self.image_size.width(), height=self.image_size.height(),
                                          image=image_loading_node.image_out).image_out

            if mask_load_node is not None and self.load_as_inpainting_model:
                inpaint_conditioning_node = InpaintModelConditioningNode(positive=positive, negative=negative, vae=vae,
                                                                         pixels=scaled_image,
                                                                         mask=mask_load_node.mask_out)
                positive = inpaint_conditioning_node.positive_out
                negative = inpaint_conditioning_node.negative_out
                latent = inpaint_conditioning_node.latent_out
            else:
                if self.vae_tiling_enabled:
                    latent = VAEEncodeTiledNode(tile_size=self._vae_tile_size, pixels=scaled_image,
                                                vae=vae).latent_out
                else:
                    latent = VAEEncodeNode(pixels=scaled_image, vae=vae).latent_out
                if mask_load_node is not None:
                    latent = LatentMaskNode(samples=latent, mask=mask_load_node.mask_out).latent_out
                latent = RepeatLatentNode(amount=self.batch_size, samples=latent).latent_out

        # Load ControlNet Units:
        loaded_images: dict[str, LoadImageNode] = {}
        if image_loading_node is not None:
            source_image = self.source_image
            assert source_image is not None
            loaded_images[source_image] = image_loading_node
        for controlnet_unit in self._controlnet_units:
            control_img_str = controlnet_unit.control_image
            if control_img_str == '':
                # No control image: reuse the source image, as the WebUI ControlNet extension does.
                if image_loading_node is None:
                    unit_name = controlnet_unit.model_node.control_net_name if controlnet_unit.model_node is not None \
                        else controlnet_unit.preprocessor.typedef.name if controlnet_unit.preprocessor is not None \
                        else 'unnamed'
                    raise ValueError(f'ControlNet unit "{unit_name}" has no control image and there is no source'
                                     ' image to use instead. Set an image on the unit or provide an init image.')
                control_image_node = image_loading_node
            elif control_img_str in loaded_images:
                control_image_node = loaded_images[control_img_str]
            else:
                control_image_node = LoadImageNode(image=control_img_str)
                loaded_images[control_img_str] = control_image_node
            positive, negative = self._wire_controlnet_unit(
                controlnet_unit, control_image_node.image_out,
                mask_load_node.mask_out if mask_load_node is not None else None, positive, negative, vae)

        # Core diffusion process in KSamplerNode:
        sampling_node = KSamplerNode(cfg=self.cfg_scale, steps=self.steps, sampler_name=self.sampler,
                                     denoise=self.denoising_strength, scheduler=self.scheduler, seed=self.seed,
                                     model=sd_model, positive=positive, negative=negative, latent_image=latent)

        # Decode and save images:
        if self.vae_tiling_enabled:
            image = VAEDecodeTiledNode(tile_size=self._vae_tile_size, vae=vae,
                                       samples=sampling_node.latent_out).image_out
        else:
            image = VAEDecodeNode(vae=vae, samples=sampling_node.latent_out).image_out

        workflow = ComfyNodeGraph()
        workflow.add_node(SaveImageNode(filename_prefix=self.filename_prefix, images=image))
        # Changes to the returned graph shouldn't affect the workflow builder, so create a deep copy to return:
        final_workflow = deepcopy(workflow)
        self._clear_saved_node_connections()
        return final_workflow

    def _load_checkpoint(self) -> tuple[NodeOutput, NodeOutput, NodeOutput]:
        """Returns the model, CLIP and VAE outputs of a new loader node for `sd_model`."""
        model_loading_node: SimpleCheckpointLoaderNode | CheckpointLoaderNode
        if self.model_config_path is None:
            model_loading_node = SimpleCheckpointLoaderNode(ckpt_name=self.sd_model)
        else:
            model_loading_node = CheckpointLoaderNode(ckpt_name=self.sd_model, config_name=self.model_config_path)
        return model_loading_node.model_out, model_loading_node.clip_out, model_loading_node.vae_out

    @staticmethod
    def _wire_controlnet_unit(unit: ControlNetNodeData, control_image: NodeOutput, mask: Optional[NodeOutput],
                              positive: NodeOutput, negative: NodeOutput,
                              vae: NodeOutput) -> tuple[NodeOutput, NodeOutput]:
        """Connects one ControlNet unit's nodes, and returns the conditioning outputs with the unit applied.

        The preprocessor, when present, takes `control_image` (and `mask`, if it has a mask input), and the apply
        node uses its output. Otherwise the apply node uses `control_image` directly.
        """
        apply_node = unit.control_apply_node
        apply_node.image = control_image
        if unit.preprocessor_node is not None:
            if unit.preprocessor_node.has_image_input:
                unit.preprocessor_node.image = control_image
            if unit.preprocessor_node.has_mask_input and mask is not None:
                unit.preprocessor_node.mask = mask
            apply_node.image = unit.preprocessor_node.image_out
        apply_node.positive = positive
        apply_node.negative = negative
        if unit.model_node is not None:
            apply_node.control_net = unit.model_node.controlnet_out
        apply_node.vae = vae
        return apply_node.positive_out, apply_node.negative_out

    def _encode_prompts(self, clip: NodeOutput) -> tuple[NodeOutput, NodeOutput]:
        """Applies CLIP skip to `clip`, and returns the positive and negative prompt conditioning outputs."""
        if self.clip_skip > 1:
            clip = CLIPSkipNode(stop_at_clip_layer=self.clip_skip, clip=clip).clip_out
        positive = ClipTextEncodeNode(text=self.prompt, clip=clip).conditioning_out
        negative = ClipTextEncodeNode(text=self.negative_prompt, clip=clip).conditioning_out
        return positive, negative

    def _apply_extension_models(self, sd_model: NodeOutput, clip: NodeOutput) -> tuple[NodeOutput, NodeOutput]:
        """Chains the extension model nodes onto the model and CLIP outputs, and returns the final outputs."""
        for extension_node in self._extension_model_nodes:
            extension_node.model = sd_model
            sd_model = extension_node.model_out
            if isinstance(extension_node, LoraLoaderNode):
                extension_node.clip = clip
                clip = extension_node.clip_out
        return sd_model, clip

    def _clear_saved_node_connections(self) -> None:
        """Disconnects the nodes the builder keeps between builds, so one build's wiring can't leak into the next."""
        for node in self._extension_model_nodes:
            node.clear_connections()
        for controlnet_unit in self._controlnet_units:
            if controlnet_unit.preprocessor_node is not None:
                controlnet_unit.preprocessor_node.clear_connections()
            if controlnet_unit.model_node is not None:
                controlnet_unit.model_node.clear_connections()
            controlnet_unit.control_apply_node.clear_connections()

    def _load_extension_models(self, available_loras: Sequence[str], available_hypernetworks: Sequence[str]) -> None:
        """Replaces `<lora:name:weight>`, `<lyco:...>` and `<hypernet:...>` prompt tags with extension model nodes.

        A tag's name matches a model file by its full name or its name without the final extension. An optional
        second weight sets the LoRA CLIP strength separately. Tags in the negative prompt apply with negated strength.
        Matched tags are removed from both prompts.
        """
        name_maps: dict[ExtensionModelType, dict[str, str]] = {}
        for model_type, model_list in ((ExtensionModelType.LORA, available_loras),
                                       (ExtensionModelType.HYPERNETWORK, available_hypernetworks)):
            name_map: dict[str, str] = {}
            for model_option in model_list:
                name_map[os.path.splitext(model_option)[0]] = model_option
            for model_option in model_list:
                name_map[model_option] = model_option
            name_maps[model_type] = name_map

        for prompt, strength_multiplier in ((self.prompt, 1.0),
                                            (self.negative_prompt, -1.0)):
            for match in re.finditer(EXTENSION_MODEL_PATTERN, prompt):
                model_type = ExtensionModelType.HYPERNETWORK if match.group(1) == 'hypernet' \
                    else ExtensionModelType.LORA
                type_label = 'hypernetwork' if model_type == ExtensionModelType.HYPERNETWORK else 'LoRA'
                model_name = match.group(2)
                if model_name not in name_maps[model_type]:
                    available = ', '.join(sorted(set(name_maps[model_type].values()))) or 'none'
                    raise ValueError(f'Prompt tag {match.group(0)} names {type_label} "{model_name}", which the'
                                     f' server does not list. Available {type_label} models: {available}')
                model_name = name_maps[model_type][model_name]
                model_strength_str = match.group(3)
                clip_strength_str = match.group(4) if match.group(4) is not None else model_strength_str
                try:
                    model_strength = float(model_strength_str) * strength_multiplier
                    clip_strength = float(clip_strength_str) * strength_multiplier
                except ValueError:
                    raise ValueError(f'Prompt tag {match.group(0)} has a strength that is not a number') from None
                self.add_extension_model(model_name, model_strength, clip_strength, model_type)

            if strength_multiplier > 0:
                self.prompt = re.sub(EXTENSION_MODEL_PATTERN, '', prompt)
            else:
                self.negative_prompt = re.sub(EXTENSION_MODEL_PATTERN, '', prompt)

    def load_diffusion_parameters(self, diffusion_params: DiffusionParams,
                                  available_loras: Sequence[str] = (),
                                  available_hypernetworks: Sequence[str] = ()) -> None:
        """Loads diffusion parameters, turning LoRA and hypernetwork prompt tags into loader nodes.

        Tags are resolved against `available_loras` and `available_hypernetworks`, the server's model file lists
        (`ComfyUiWebservice.get_lora_models` and `get_hypernetwork_models`). See `_load_extension_models`.

        Raises
        ------
        ValueError
            If a prompt tag names a model not in the matching list, or has a non-numeric strength.
        """
        if not isinstance(diffusion_params, ComfyUIDiffusionParams):
            diffusion_params = ComfyUIDiffusionParams(**{name: getattr(diffusion_params, name)
                                                       for name in DiffusionParams.model_fields.keys()})
        self.sd_model = diffusion_params.sd_model_name
        self.batch_size = diffusion_params.batch_size
        self.prompt = diffusion_params.prompt
        self.negative_prompt = diffusion_params.negative_prompt
        self.steps = diffusion_params.steps
        self.cfg_scale = diffusion_params.cfg_scale
        self.image_size = Size(diffusion_params.width, diffusion_params.height)
        self.sampler = comfyui_sampler_name(diffusion_params.sampler)
        self.scheduler = comfyui_scheduler_name(diffusion_params.scheduler)
        seed = diffusion_params.seed
        if seed < 0:
            seed = random_seed()
        self.seed = seed

        self.load_as_inpainting_model = diffusion_params.load_as_inpainting_model
        self.vae_tiling_enabled = diffusion_params.vae_tiling_enabled
        self.vae_tile_size = diffusion_params.vae_tile_size
        self.clip_skip = diffusion_params.clip_skip
        self.differential_diffusion = diffusion_params.soft_inpainting
        self.differential_diffusion_strength = diffusion_params.soft_inpainting_params.get(STRENGTH_KEY) \
            if diffusion_params.soft_inpainting else None

        self._load_extension_models(available_loras, available_hypernetworks)

        if diffusion_params.init_images:
            self.denoising_strength = DEFAULT_DENOISING_STRENGTH if diffusion_params.denoising_strength is None \
                else diffusion_params.denoising_strength
        self.model_config_path = diffusion_params.sd_model_config
