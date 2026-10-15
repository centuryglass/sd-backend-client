"""The interface both backend clients implement, so callers can drive either one the same way.

`A1111Webservice` and `ComfyUiWebservice` subclass `Backend`. Each `submit_*` method copies what it needs from its
arguments before returning, and returns a `GenerationHandle` for the queued job. Fields that only the other backend
supports are ignored. The `list_*` methods and `get_capabilities` report what the server offers, in the same types on
both backends. Methods outside this class (blocking `txt2img` and `upscale`, `get_*` listings in each server's own
format, `interrupt`) are backend-specific and differ in signature and return type.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional

from PIL import Image

from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.backend_options import BackendCapabilities, BackendOption
from sd_backend_client.api.shared_data.controlnet.controlnet_constants import ControlTypeDef
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    PreprocessorParams
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.generation_handle import GenerationHandle
from sd_backend_client.api.shared_data.parameter_def import ParameterDef, validate_parameter_values

__all__ = ['Backend', 'require_init_image', 'require_mask', 'require_soft_inpainting', 'require_upscale_size']


class Backend(ABC):
    """A Stable Diffusion server client that queues generation jobs and returns a handle for each.

    Every `submit_*` method validates its inputs before contacting the server, raising `ValueError` for a missing
    image or mask, or for an upscale size that is not larger than the image. A returned handle's job runs with its
    arguments as they were at submit time, so the caller may change or reuse them afterwards.
    """

    @abstractmethod
    def submit_txt2img(self, diffusion_params: DiffusionParams) -> GenerationHandle:
        """Queue a text-to-image job and return its handle. Init images, the mask and soft inpainting are ignored."""

    @abstractmethod
    def submit_img2img(self, diffusion_params: DiffusionParams) -> GenerationHandle:
        """Queue an image-to-image job on `diffusion_params.init_images[0]` and return its handle.

        A `mask`, if set, limits changes to the masked region, as in `submit_inpaint`.

        Raises
        ------
        ValueError
            If `diffusion_params.init_images` is empty.
        """

    @abstractmethod
    def submit_inpaint(self, diffusion_params: DiffusionParams) -> GenerationHandle:
        """Queue an inpainting job that changes the masked region of `diffusion_params.init_images[0]`.

        Raises
        ------
        ValueError
            If `diffusion_params.init_images` is empty or `diffusion_params.mask` is None.
        """

    @abstractmethod
    def submit_upscale(self, image: Image.Image, width: int, height: int,
                       upscale_params: Optional[DiffusionUpscalingParams] = None) -> GenerationHandle:
        """Queue a job that upscales `image` to `width` x `height` and return its handle.

        The result holds the one upscaled image. The job applies the upscaling model named by
        `upscale_params.upscaling_mode` and resizes to the requested size; with
        `upscale_params.use_stable_diffusion_upscaling` set it also runs a tiled diffusion pass. A basic upscale with
        `upscale_params` None or no `upscaling_mode` uses the server's first upscaling model. A named model the server
        lacks falls back to that first model on WebUI, and raises `ValueError` for a basic upscale on ComfyUI.

        Raises
        ------
        ValueError
            If `width` x `height` does not exceed the size of `image` in at least one dimension.
        """

    @abstractmethod
    def submit_preprocessor_preview(self, image: Image.Image,
                                    preprocessor: ControlNetPreprocessor | PreprocessorParams,
                                    mask: Optional[Image.Image] = None) -> GenerationHandle:
        """Queue a job that runs a ControlNet preprocessor on `image` and return its handle.

        The result holds the one preprocessed control image. A bare `ControlNetPreprocessor` runs with the server's
        parameter defaults, and a `PreprocessorParams` overrides them with its `parameter_values`. `mask` is passed
        only to preprocessors that take one, such as inpainting preprocessors.
        """

    @abstractmethod
    def list_checkpoints(self) -> list[BackendOption]:
        """List the server's Stable Diffusion checkpoints. Each `name` is a valid `DiffusionParams.sd_model_name`."""

    @abstractmethod
    def list_vaes(self) -> list[BackendOption]:
        """List the server's VAE models by file name. No `DiffusionParams` field selects one."""

    @abstractmethod
    def list_loras(self) -> list[BackendOption]:
        """List the server's LoRA models. Each `name` works in a `<lora:name:weight>` prompt tag."""

    @abstractmethod
    def list_hypernetworks(self) -> list[BackendOption]:
        """List the server's hypernetworks. Each `name` works in a `<hypernet:name:weight>` prompt tag."""

    @abstractmethod
    def list_samplers(self) -> list[BackendOption]:
        """List the server's samplers. Each `name` is a valid `DiffusionParams.sampler`.

        Names are shared names from `sampler_names` where one exists, so the same sampler has the same name on both
        backends. `display_name` holds the WebUI name for a sampler that has one.
        """

    @abstractmethod
    def list_schedulers(self) -> list[BackendOption]:
        """List the server's noise schedulers. Each `name` is a valid `DiffusionParams.scheduler`.

        Names are shared names from `sampler_names` where one exists. A server that ignores the scheduler (see
        `BackendCapabilities.scheduler`) lists none.
        """

    @abstractmethod
    def list_upscalers(self) -> list[BackendOption]:
        """List the server's upscaling models. Each `name` is a valid `DiffusionUpscalingParams.upscaling_mode`."""

    @abstractmethod
    def list_controlnet_models(self) -> list[ControlNetModel]:
        """List the server's ControlNet models, for `ControlNetUnit.model`. Empty when the server lacks ControlNet."""

    @abstractmethod
    def get_controlnet_preprocessors(self, update_cache: bool = False) -> list[ControlNetPreprocessor]:
        """List the server's ControlNet preprocessors with their parameters.

        The list is cached after the first call; `update_cache` reloads it. Callers may change the returned list.
        """

    @abstractmethod
    def get_controlnet_type_categories(self) -> dict[str, ControlTypeDef]:
        """Group the server's ControlNet preprocessors and models into control types, keyed by type name."""

    @abstractmethod
    def get_soft_inpainting_parameters(self) -> list[ParameterDef]:
        """List the server's soft inpainting settings, for `DiffusionParams.soft_inpainting_params`.

        The settings differ between backends: six for the WebUI's Soft Inpainting script, and Differential Diffusion's
        `strength` on ComfyUI builds that have it. Empty when the server lacks soft inpainting (see
        `BackendCapabilities.soft_inpainting`). Each call queries the server again.
        """

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Report which optional features the server has. Each call queries the server again."""


def require_init_image(diffusion_params: DiffusionParams, operation: str) -> None:
    """Raise `ValueError` unless `diffusion_params` holds an init image for `operation`."""
    if not diffusion_params.init_images:
        raise ValueError(f'Must set an init image in diffusion_params for {operation}')


def require_mask(diffusion_params: DiffusionParams, operation: str) -> None:
    """Raise `ValueError` unless `diffusion_params` holds a mask for `operation`."""
    if diffusion_params.mask is None:
        raise ValueError(f'Must set a mask in diffusion_params for {operation}')


def require_soft_inpainting(diffusion_params: DiffusionParams,
                            get_parameters: Callable[[], Optional[list[ParameterDef]]]) -> None:
    """Raise `ValueError` unless soft inpainting is off, or the job has a mask and valid settings for the server.

    `get_parameters` returns the server's soft inpainting settings, or None when the server lacks soft inpainting. It
    is called only after the mask check passes, so a missing mask raises without contacting the server. Both backends
    ignore soft inpainting without a mask, so a request without one raises here instead of silently doing nothing.
    """
    if not diffusion_params.soft_inpainting:
        return
    require_mask(diffusion_params, 'soft inpainting')
    parameters = get_parameters()
    if parameters is None:
        raise ValueError('Soft inpainting is on, but the server does not support it. Check '
                         'BackendCapabilities.soft_inpainting, or turn DiffusionParams.soft_inpainting off.')
    validate_parameter_values(parameters, diffusion_params.soft_inpainting_params, 'Soft inpainting')


def require_upscale_size(image: Image.Image, width: int, height: int) -> None:
    """Raise `ValueError` unless `width` x `height` exceeds the size of `image` in at least one dimension."""
    if width <= 0 or height <= 0 or (width <= image.width and height <= image.height):
        raise ValueError(f'Requested size {width}x{height} must exceed the source size {image.width}x{image.height} '
                         'in at least one dimension')
