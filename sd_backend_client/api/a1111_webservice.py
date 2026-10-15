"""
Accesses the A1111/stable-diffusion-webui through its REST API, providing access to image generation and editing
through Stable Diffusion.
"""
import json
import logging
import re
from copy import deepcopy
from typing import Optional, Any, Callable, cast, TYPE_CHECKING
from typing_extensions import TypedDict

import requests  # type: ignore
from PIL import Image  # type: ignore
from requests import Response

from sd_backend_client.api.shared_data.controlnet.controlnet_category_builder import ControlNetCategoryBuilder
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    PreprocessorParams
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams, REDRAW_MODES, SEAM_FIX_MODES
from sd_backend_client.api.shared_data.backend import Backend, require_init_image, require_mask, \
    require_soft_inpainting, require_upscale_size
from sd_backend_client.api.shared_data.backend_options import BackendCapabilities, BackendOption
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.parameter_def import ParameterDef
from sd_backend_client.api.shared_data.sampler_names import comfyui_sampler_name, comfyui_scheduler_name
from sd_backend_client.api.webservice import WebService, DEFAULT_REQUEST_TIMEOUT
from sd_backend_client.api.webui.controlnet_webui_constants import (ControlNetModelResponse, ControlNetModuleResponse,
                                                      ControlTypeDef, ControlTypeResponse,
                                                      FIRST_GENERIC_PARAMETER_KEY, SECOND_GENERIC_PARAMETER_KEY,
                                                      PREPROCESSOR_RES_PARAM_KEY)
from sd_backend_client.api.webui.controlnet_webui_utils import get_all_preprocessors
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody
from sd_backend_client.api.webui.request_formats import UpscalingRequestBody
from sd_backend_client.api.webui.response_formats import GenerationInfoData, ProgressResponseBody, \
    InterrogateResponse, PromptStyleData, SamplerInfo, UpscalerInfo, ModelInfo, VaeInfo, LoraInfo
from sd_backend_client.api.webui.script_info_types import ScriptResponseData, ScriptInfo
from sd_backend_client.api.webui.soft_inpainting import SOFT_INPAINTING_SCRIPT_KEY, soft_inpainting_parameters
from sd_backend_client.errors import AuthError, SDBackendError, ServerError, UnexpectedResponseError
from sd_backend_client.util.api_defaults import INPAINT_FULL_RES_PADDING_DEFAULT
from sd_backend_client.util.visual.image_utils import (image_to_base64, image_from_base64, image_from_bytes,
                                                    mask_to_grayscale, blur_mask)
from sd_backend_client.util.visual.inpaint_crop import PasteBack, compute_crop_region

if TYPE_CHECKING:
    from sd_backend_client.api.webui.webui_generation_handle import WebUIDispatcher, WebUIGenerationHandle

logger = logging.getLogger(__name__)


class ImageResponse(TypedDict):
    """Defines image generation response format."""
    images: list[Image.Image]
    info: Optional[GenerationInfoData]


ULTIMATE_UPSCALE_SCRIPT = 'ultimate sd upscale'
INTERROGATE_DEFAULT_MODEL = 'clip'
DEFAULT_GENERATION_TIMEOUT = 600.0
MAX_LOGIN_ATTEMPTS = 3
SETTINGS_UPDATE_TIMEOUT = 90
# The " [hash]" suffix WebUI appends to a checkpoint's relative file name to form its title, once it has hashed it.
CHECKPOINT_TITLE_HASH_PATTERN = r' \[[0-9a-fA-F]+\]$'


def _crop_for_client_inpainting(body: DiffusionRequestBody) -> Optional[PasteBack]:
    """Replace `body`'s image and mask with their crop when `inpaint_crop_points` changes the crop the server would use.

    The server flag is turned off, since the crop is already made. Returns what the results need to be pasted back
    into the source, or None and `body` unchanged when the server can crop alone or the mask is empty.
    """
    if not body.inpaint_full_res or not body.inpaint_crop_points or not body.init_images or body.mask is None:
        return None
    source = body.init_images[0]
    blurred_mask = blur_mask(body.mask, *body.mask_blur_extents())
    padding = body.inpaint_full_res_padding
    region = compute_crop_region(blurred_mask, INPAINT_FULL_RES_PADDING_DEFAULT if padding is None else padding,
                                 (body.width, body.height), body.inpaint_crop_points)
    if region is None:
        return None
    body.init_images = [source.crop(region)] + body.init_images[1:]
    body.mask = mask_to_grayscale(body.mask).crop(region)  # The server blurs the crop's mask itself.
    body.inpaint_full_res = False
    return PasteBack(source, Image.new('L', source.size, 255), region)


class A1111Webservice(WebService, Backend):
    """
    A1111Webservice provides access to the a1111/stable-diffusion-webui through the REST API.

    Its `submit_*`, `list_*` and `get_capabilities` methods implement `Backend`. The blocking `txt2img`, `img2img`,
    `upscale` and `controlnet_preprocessor_preview` methods are WebUI-specific.
    """

    # noinspection SpellCheckingInspection
    class Endpoints:
        """REST API endpoint constants."""
        OPTIONS = '/sdapi/v1/options'
        REFRESH_CKPT = '/sdapi/v1/refresh-checkpoints'
        REFRESH_VAE = '/sdapi/v1/refresh-vae'
        REFRESH_LORA = '/sdapi/v1/refresh-loras'
        PROGRESS = '/sdapi/v1/progress'
        IMG2IMG = '/sdapi/v1/img2img'
        TXT2IMG = '/sdapi/v1/txt2img'
        UPSCALE = '/sdapi/v1/extra-single-image'
        INTERROGATE = '/sdapi/v1/interrogate'
        INTERRUPT = '/sdapi/v1/interrupt'
        STYLES = '/sdapi/v1/prompt-styles'
        SCRIPTS = '/sdapi/v1/scripts'
        SCRIPT_INFO = '/sdapi/v1/script-info'
        SAMPLERS = '/sdapi/v1/samplers'
        SCHEDULERS = '/sdapi/v1/schedulers'
        UPSCALERS = '/sdapi/v1/upscalers'
        LATENT_UPSCALE_MODES = '/sdapi/v1/latent-upscale-modes'
        HYPERNETWORKS = '/sdapi/v1/hypernetworks'
        SD_MODELS = '/sdapi/v1/sd-models'
        VAE_MODELS = '/sdapi/v1/sd-vae'
        LORA_MODELS = '/sdapi/v1/loras'
        CONTROLNET_VERSION = '/controlnet/version'
        CONTROLNET_MODELS = '/controlnet/model_list'
        CONTROLNET_MODULES = '/controlnet/module_list'
        CONTROLNET_CONTROL_TYPES = '/controlnet/control_types'
        CONTROLNET_SETTINGS = '/controlnet/settings'
        CONTROLNET_PREVIEW = '/controlnet/detect'
        LOGIN = '/login'
        EXTRA_NW_DATA = '/sd_extra_networks/metadata'
        EXTRA_NW_THUMB = '/sd_extra_networks/thumb'
        EXTRA_NW_CARD = '/sd_extra_networks/card'

    class ForgeEndpoints:
        """REST API endpoint constants (Forge WebUI alternates)"""
        SD_MODULES = '/sdapi/v1/sd-modules'

    def __init__(self, url: str,
                 credentials_provider: Optional[Callable[[], Optional[tuple[str, str]]]] = None,
                 request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
                 generation_timeout: Optional[float] = DEFAULT_GENERATION_TIMEOUT) -> None:
        """Create the webservice client.

        Parameters
        ----------
        url: str
            Base URL of the A1111 / Forge WebUI server.
        credentials_provider: optional callable
            Invoked with no arguments when the server requires authentication. It should return a
            (username, password) pair to attempt, or None to abort (raising AuthError). IntraPaint prompted for
            these through a Qt login dialog; a standalone caller injects its own prompt or fixed credentials here.
        request_timeout: float, optional, default=DEFAULT_REQUEST_TIMEOUT
            Timeout in seconds for metadata, settings and other non-generation requests. None waits indefinitely.
        generation_timeout: float, optional, default=DEFAULT_GENERATION_TIMEOUT
            Timeout in seconds for requests that block until images are ready: txt2img, img2img, upscale and
            ControlNet preprocessor previews. None waits indefinitely.
        """
        super().__init__(url, request_timeout)
        self._generation_timeout = generation_timeout
        self._preprocessor_cache: Optional[list[ControlNetPreprocessor]] = None
        self._credentials_provider = credentials_provider
        self._dispatcher: Optional['WebUIDispatcher'] = None

    @property
    def generation_timeout(self) -> Optional[float]:
        """Timeout in seconds for requests that block until images are ready, or None to wait indefinitely."""
        return self._generation_timeout

    @property
    def _generation_dispatcher(self) -> 'WebUIDispatcher':
        """Lazily-created client-side dispatch queue backing the async ``submit_*`` methods."""
        if self._dispatcher is None:
            from sd_backend_client.api.webui.webui_generation_handle import WebUIDispatcher
            self._dispatcher = WebUIDispatcher(self)
        return self._dispatcher

    def _submit(self, endpoint: str, task_type: str, snapshot: DiffusionRequestBody,
                paste_back: Optional[PasteBack] = None) -> 'WebUIGenerationHandle':
        """Enqueue a POST of `snapshot` to `endpoint` on the client-side dispatcher, returning its handle.

        `snapshot` must be a copy the caller no longer changes. Its `force_task_id` is set to a new task id unless it
        already has one. `paste_back` is applied to each result image.
        """
        task_id = snapshot.force_task_id or self._new_task_id(task_type)
        snapshot.force_task_id = task_id
        return self._generation_dispatcher.submit(lambda: self._post_generation(endpoint, snapshot), task_id,
                                                  paste_back=paste_back)

    def _submit_call(self, run: Callable[[], ImageResponse], task_type: str,
                     interruptible: bool) -> 'WebUIGenerationHandle':
        """Enqueue a blocking call on the client-side dispatcher under a new task id, returning its handle."""
        return self._generation_dispatcher.submit(run, self._new_task_id(task_type), interruptible=interruptible)

    @staticmethod
    def _new_task_id(task_type: str) -> str:
        """Mint a WebUI task id for a job of `task_type` (see `create_task_id`)."""
        from sd_backend_client.api.webui.webui_generation_handle import create_task_id
        return create_task_id(task_type)

    def submit_txt2img(self, diffusion_params: Optional[DiffusionParams] = None) -> 'WebUIGenerationHandle':
        """Enqueue a txt2img job and return a handle immediately, implementing `Backend.submit_txt2img`.

        The blocking POST is deferred to the client-side dispatcher (see ``webui_generation_handle``), so
        the returned handle starts ``PENDING`` and is cleanly cancellable until it is actually dispatched.
        Call ``handle.wait()`` for the blocking result, or poll ``handle.poll()`` for progress.
        Any `DiffusionParams` is accepted and converted with `DiffusionRequestBody.from_params`, so the job uses a
        copy taken at submit time. None submits a default request body.
        """
        snapshot = DiffusionRequestBody.from_params(diffusion_params)
        snapshot.init_images = None
        snapshot.mask = None
        snapshot.soft_inpainting = False
        return self._submit(A1111Webservice.Endpoints.TXT2IMG, 'txt2img', snapshot)

    def submit_img2img(self, diffusion_params: DiffusionParams) -> 'WebUIGenerationHandle':
        """Enqueue an img2img job and return a handle immediately, implementing `Backend.submit_img2img`.

        See `submit_txt2img` for the dispatch, cancel and copy semantics.
        """
        require_init_image(diffusion_params, 'img2img')
        require_soft_inpainting(diffusion_params, self._soft_inpainting_parameters)
        return self._submit(A1111Webservice.Endpoints.IMG2IMG, 'img2img',
                            DiffusionRequestBody.from_params(diffusion_params))

    def submit_inpaint(self, diffusion_params: DiffusionParams) -> 'WebUIGenerationHandle':
        """Enqueue an inpainting job and return a handle immediately, implementing `Backend.submit_inpaint`.

        WebUI inpaints through its img2img endpoint. See `submit_txt2img` for the dispatch, cancel and copy semantics.
        With `inpaint_full_res`, the server crops around the mask. When `inpaint_crop_points` is also set, the client
        crops instead, sends the crop with the server flag off, and pastes each result back into the source.
        """
        require_init_image(diffusion_params, 'inpainting')
        require_mask(diffusion_params, 'inpainting')
        require_soft_inpainting(diffusion_params, self._soft_inpainting_parameters)
        body = DiffusionRequestBody.from_params(diffusion_params)
        paste_back = _crop_for_client_inpainting(body)
        return self._submit(A1111Webservice.Endpoints.IMG2IMG, 'img2img', body, paste_back)

    def submit_upscale(self, image: Image.Image, width: int, height: int,
                       upscale_params: Optional[DiffusionUpscalingParams] = None) -> 'WebUIGenerationHandle':
        """Enqueue an `upscale` call and return a handle immediately, implementing `Backend.submit_upscale`.

        The job uses copies of `image` and `upscale_params` taken at submit time. Once dispatched, a basic upscale
        cannot be cancelled, since the server's interrupt is not known to stop it; a Stable Diffusion upscale can.
        See `submit_txt2img` for the dispatch semantics.
        """
        require_upscale_size(image, width, height)
        image = image.copy()
        upscale_params = None if upscale_params is None else upscale_params.model_copy(deep=True)
        interruptible = upscale_params is not None and upscale_params.use_stable_diffusion_upscaling
        return self._submit_call(lambda: self.upscale(image, width, height, upscale_params), 'upscale', interruptible)

    def submit_preprocessor_preview(self, image: Image.Image,
                                    preprocessor: ControlNetPreprocessor | PreprocessorParams,
                                    mask: Optional[Image.Image] = None) -> 'WebUIGenerationHandle':
        """Enqueue a `controlnet_preprocessor_preview` call and return a handle immediately.

        Implements `Backend.submit_preprocessor_preview`. The job uses copies of its arguments taken at submit time.
        Once dispatched it cannot be cancelled, since the server's interrupt is not known to stop `/controlnet/detect`.
        See `submit_txt2img` for the dispatch semantics.
        """
        image = image.copy()
        mask = None if mask is None else mask.copy()
        preprocessor = preprocessor.model_copy(deep=True)

        def run() -> ImageResponse:
            return {'images': [self.controlnet_preprocessor_preview(image, mask, preprocessor)], 'info': None}
        return self._submit_call(run, 'preview', interruptible=False)

    # General utility:
    def login_check(self):
        """Calls the login check endpoint, returning a status 401 response if a login is required."""
        return self.get('/login_check')

    def set_config(self, config_updates: dict) -> None:
        """
        Updates the Stable Diffusion WebUI configuration.

        Parameters
        ----------
        config_updates: dict
            Maps settings that should change to their updated values. Use the get_settings method's response body
            to check available options.
        """
        self.post(A1111Webservice.Endpoints.OPTIONS, config_updates, timeout=SETTINGS_UPDATE_TIMEOUT).json()

    def refresh_checkpoints(self) -> requests.Response:
        """Requests an updated list of available Stable Diffusion models.

        Returns
        -------
        response
            HTTP response with the list of updated Stable Diffusion models.
        """
        return self.post(A1111Webservice.Endpoints.REFRESH_CKPT, body={})

    def refresh_vae(self) -> requests.Response:
        """Requests an updated list of available Stable Diffusion VAE models.

        VAE models handle the conversion between images and the latent image space. Different VAE models can be used
        to adjust performance and final image quality.


        Returns
        -------
        response
            HTTP response with the list of updated Stable Diffusion VAE models.
        """
        return self.post(A1111Webservice.Endpoints.REFRESH_VAE, body={})

    def refresh_loras(self) -> requests.Response:
        """Requests an updated list of available Stable Diffusion LoRA models.

        LoRA models augment existing Stable Diffusion models, usually to provide support for new concepts, characters,
        or art styles.

        Returns
        -------
        response
            HTTP response with the list of updated Stable Diffusion LoRA models.
        """
        return self.post(A1111Webservice.Endpoints.REFRESH_LORA, body={})

    def progress_check(self) -> ProgressResponseBody:
        """Checks the progress of an ongoing image operation."""
        return ProgressResponseBody.model_validate(
            self.get(A1111Webservice.Endpoints.PROGRESS).json())

    # Image manipulation:
    def img2img(self, image: Image.Image, mask: Optional[Image.Image] = None,
                request_body: Optional[DiffusionParams] = None) -> ImageResponse:
        """Sends a request to alter an image section using selected parameters.

        Parameters
        ----------
        image: Image.Image
            Source image to transform. If request_body is not None, and it already has an image, this parameter is
            ignored.
        mask: Optional[Image.Image] = None
            Optional inpainting mask.  This will also be ignored if request_body is not None, and it already has a mask.
        request_body : Optional[DiffusionParams] = None
            Optional initial request body to use. If None, a default DiffusionRequestBody is used. Any other
            `DiffusionParams` is converted with `DiffusionRequestBody.from_params`. The caller's body is not modified.
        Returns
        -------
        ImageResponse
            All generated images, plus accompanying image generation data if available.
        """
        if isinstance(request_body, DiffusionRequestBody):
            request_body = request_body.model_copy()
        else:
            request_body = DiffusionRequestBody.from_params(request_body)
        if not request_body.init_images:
            request_body.init_images = [image]
        else:
            request_body.init_images = list(request_body.init_images)
        if request_body.mask is None and mask is not None:
            request_body.mask = mask
        require_soft_inpainting(request_body, self._soft_inpainting_parameters)
        return self._post_generation(A1111Webservice.Endpoints.IMG2IMG, request_body)

    def txt2img(self, request_body: Optional[DiffusionParams] = None) -> ImageResponse:
        """Sends a request to generate new images using selected parameter.

        Parameters
        ----------
        request_body : Optional[DiffusionParams] = None
            Optional initial request body to use. If None, a new one will be constructed from default parameters.
            Any `DiffusionParams` other than a `DiffusionRequestBody` is converted with
            `DiffusionRequestBody.from_params`.
        Returns
        -------
        ImageResponse
            All generated images, plus accompanying image generation data if available.
        """
        if not isinstance(request_body, DiffusionRequestBody):
            request_body = DiffusionRequestBody.from_params(request_body)
        if request_body.soft_inpainting:
            request_body = request_body.model_copy(update={'soft_inpainting': False})
        return self._post_generation(A1111Webservice.Endpoints.TXT2IMG, request_body)

    def _post_generation(self, endpoint: str, request_body: DiffusionRequestBody) -> ImageResponse:
        """POSTs a generation request body, blocking until its images are ready."""
        res = self.post(endpoint, request_body.to_dict(), timeout=self._generation_timeout)
        return self._handle_image_response(res)

    def controlnet_preprocessor_preview(self, image: Image.Image, mask: Optional[Image.Image],
                                        preprocessor: ControlNetPreprocessor | PreprocessorParams) -> Image.Image:
        """Gets a preview image for a ControlNet preprocessor.

        `preprocessor` may be a bare `ControlNetPreprocessor`, in which case the server applies its own
        parameter defaults, or a `PreprocessorParams` to override threshold_a/threshold_b/processor_res.
        """
        input_images: list[str] = [image_to_base64(image, True)]
        if mask is not None:
            input_images.append(image_to_base64(mask_to_grayscale(mask), True))
        typedef = preprocessor.typedef if isinstance(preprocessor, PreprocessorParams) else preprocessor
        body: dict[str, int | float | str | list[str]] = {
            'controlnet_module': typedef.name,
            'controlnet_input_images': input_images
        }
        if isinstance(preprocessor, PreprocessorParams):
            detect_param_keys = {
                FIRST_GENERIC_PARAMETER_KEY: 'controlnet_threshold_a',
                SECOND_GENERIC_PARAMETER_KEY: 'controlnet_threshold_b',
                PREPROCESSOR_RES_PARAM_KEY: 'controlnet_processor_res',
            }
            for param_key, detect_key in detect_param_keys.items():
                if param_key in preprocessor.parameter_values:
                    body[detect_key] = preprocessor.parameter_values[param_key]
        res = self.post(A1111Webservice.Endpoints.CONTROLNET_PREVIEW, body, timeout=self._generation_timeout)
        images = self._handle_image_response(res)['images']
        if not images:
            raise UnexpectedResponseError(f'{A1111Webservice.Endpoints.CONTROLNET_PREVIEW} returned no preview image')
        return images[0]

    def _validate_tile_controlnet(self, tile_control_unit: Optional[ControlNetUnit]) -> Optional[ControlNetUnit]:
        """Return the tile ControlNet unit only if it's fully specified and installed on the server, else None."""
        if tile_control_unit is None:
            return None
        if (tile_control_unit.model is None
                or tile_control_unit.preprocessor is None
                or float(tile_control_unit.control_strength) == 0.0
                or float(tile_control_unit.control_start) >= float(tile_control_unit.control_end)):
            return None
        models = self.get_controlnet_models().model_list
        preprocessors = [preprocessor.name for preprocessor in self.get_controlnet_preprocessors()]
        if (tile_control_unit.model.full_model_name not in models
                or tile_control_unit.preprocessor.typedef.name not in preprocessors):
            return None
        return tile_control_unit

    def upscale(self,
                image: Image.Image,
                width: int,
                height: int,
                sd_upscale_params: Optional[DiffusionUpscalingParams] = None) -> ImageResponse:
        """Sends a request to upscale an image.

        Parameters
        ----------
        image : Image.Image
            Source image to upscale.
        width : int
            New image width in pixels requested.
        height : int
            New image height in pixels requested.
        sd_upscale_params : Optional[DiffusionUpscalingParams]
            Parameters for stable diffusion upscaling using the "Ultimate SD Upscale" workflow. If None, only basic
            upscaling will be used.
        Returns
        -------
        ImageResponse
            The generated image, plus accompanying image generation data if available.
        """
        upscale_options = [upscaler.name for upscaler in self.get_upscalers()]
        # Default to the first real upscaler rather than upscale_options[0], which is the no-op 'None' upscaler that
        # would leave the image unchanged.
        upscaler: str = next((name for name in upscale_options if name.lower() != 'none'), upscale_options[0])
        if sd_upscale_params is not None and sd_upscale_params.upscaling_mode in upscale_options:
            upscaler = sd_upscale_params.upscaling_mode

        if sd_upscale_params is not None and sd_upscale_params.use_stable_diffusion_upscaling:
            # Populate the core diffusion pass (prompt, seed, cfg, sampler, checkpoint, ...) from diffusion_params,
            # then override the upscale-specific bits, which take precedence:
            request_body = DiffusionRequestBody.from_params(sd_upscale_params.diffusion_params)
            request_body.init_images = [image]
            request_body.soft_inpainting = False
            request_body.denoising_strength = sd_upscale_params.denoising_strength
            request_body.steps = sd_upscale_params.step_count
            request_body.width = width
            request_body.height = height
            request_body.batch_size = 1
            request_body.n_iter = 1

            # Attach the optional tile ControlNet via controlnet_units; DiffusionRequestBody.to_dict() serializes these
            # into alwayson_scripts['controlnet']. (Setting alwayson_scripts directly here would be wiped by to_dict.)
            tile_control_unit = self._validate_tile_controlnet(sd_upscale_params.tile_controlnet)
            request_body.controlnet_units = [tile_control_unit] if tile_control_unit is not None else []

            if sd_upscale_params.use_ultimate_upscale_script:
                request_body.script_name = ULTIMATE_UPSCALE_SCRIPT
                # Positional args for the "ultimate sd upscale" script (order confirmed via /sdapi/v1/script-info):
                request_body.script_args = [
                    None,  # [0] not used
                    sd_upscale_params.tile_width,        # [1] tile width
                    sd_upscale_params.tile_height,       # [2] tile height
                    sd_upscale_params.mask_blur,         # [3] mask_blur
                    sd_upscale_params.tile_padding,      # [4] tile padding
                    sd_upscale_params.seam_fix_width,    # [5] seams_fix_width
                    sd_upscale_params.seam_fix_denoise,  # [6] seams_fix_denoise
                    sd_upscale_params.seam_fix_padding,  # [7] seams_fix_padding
                    upscale_options.index(upscaler),     # [8] upscaler_index (into /sdapi/v1/upscalers)
                    sd_upscale_params.save_upscaled_image,   # [9] save_upscaled_image
                    REDRAW_MODES.index(sd_upscale_params.redraw_mode),  # [10] redraw mode (0=Linear,1=Chess,2=None)
                    sd_upscale_params.save_seams_fix_image,   # [11] save_seams_fix_image
                    sd_upscale_params.seam_fix_mask_blur,     # [12] seams_fix_mask_blur
                    SEAM_FIX_MODES.index(sd_upscale_params.seam_fix_mode),  # [13] seams_fix_type
                    1,       # [14] target_size_type (1 = use custom width/height below)
                    width,   # [15] custom_width
                    height,  # [16] custom_height
                    None     # [17] custom_scale (ignored when target_size_type=1)
                ]
            return self.img2img(image, None, request_body)
        # otherwise, normal upscaling without controlNet:
        body: UpscalingRequestBody = {
            'resize_mode': 1,
            'upscaling_resize_w': width,
            'upscaling_resize_h': height,
            'upscaler_1': upscaler,
            'image': image_to_base64(image, include_prefix=True)
        }
        res = self.post(A1111Webservice.Endpoints.UPSCALE, body, timeout=self._generation_timeout)
        return self._handle_image_response(res)

    def interrogate(self, image: Image.Image, interrogate_model: Optional[str] = None) -> str:
        """Requests text describing an image.

        Parameters
        ----------
        image : PIL Image
            The image to describe.
        interrogate_model : Optional[str]
            Specific image interrogation model to use. Must be supported by the backend. Defaults to
            INTERROGATE_DEFAULT_MODEL.
        Returns
        -------
        str
            A brief description of the image.
        """
        if interrogate_model is None:
            interrogate_model = INTERROGATE_DEFAULT_MODEL
        body = {
            'model': interrogate_model,
            'image': image_to_base64(image, include_prefix=True)
        }
        res = self._response_json(self.post(A1111Webservice.Endpoints.INTERROGATE, body, timeout=60))
        if isinstance(res, dict):
            return InterrogateResponse.model_validate(res).caption
        if not isinstance(res, str):
            raise UnexpectedResponseError(f'{A1111Webservice.Endpoints.INTERROGATE} returned {res!r}, expected a '
                                          'caption')
        return res

    def interrupt(self) -> dict:
        """
        Attempts to interrupt an ongoing image operation, returning a dict from the response body indicating the
        result.
        """
        res = self.post(A1111Webservice.Endpoints.INTERRUPT, body={})
        return res.json()

    @staticmethod
    def _response_json(res: Response) -> Any:
        """Parses a successful response's JSON body, raising UnexpectedResponseError if it is not JSON."""
        try:
            return res.json()
        except ValueError as err:
            raise UnexpectedResponseError(f'Expected a JSON response from {res.url}, got: {res.text[:200]!r}') from err

    @staticmethod
    def _handle_image_response(res: Response) -> ImageResponse:
        """Decodes the images in a successful generation response.

        Raises UnexpectedResponseError if the body is not JSON or an image cannot be decoded."""
        res_body = A1111Webservice._response_json(res)
        if not isinstance(res_body, dict):
            raise UnexpectedResponseError(f'Expected a JSON object from {res.url}, got {type(res_body).__name__}')
        try:
            return A1111Webservice._decode_image_response(res_body)
        except (OSError, ValueError) as err:
            raise UnexpectedResponseError(f'Images returned by {res.url} could not be decoded: {err}') from err

    @staticmethod
    def _decode_image_response(res_body: dict[str, Any]) -> ImageResponse:
        images = []
        info_data: Optional[GenerationInfoData] = None
        if 'images' in res_body:
            for image in res_body['images']:
                images.append(image_from_base64(image))
            info = res_body.get('info')  # absent on some endpoints (e.g. /controlnet/detect)
            if isinstance(info, str):
                try:
                    info_data = GenerationInfoData.model_validate(json.loads(info))
                except json.JSONDecodeError:
                    logger.error(f'Image response info not valid JSON, got {info}')
                    info_data = None
            elif info is not None:
                info_data = GenerationInfoData.model_validate(info)
        elif 'image' in res_body:  # basic upscaling result
            images = [image_from_base64(res_body['image'])]
            info_data = None
        image_response: ImageResponse = {
            'images': images,
            'info': info_data
        }
        return image_response

    # Load misc. service info:
    def get_config(self) -> dict[str, Any]:
        """Returns a dict containing the current Stable Diffusion WebUI configuration."""
        return self.get('/sdapi/v1/options').json()

    def get_styles(self) -> list[PromptStyleData]:
        """Returns a list of image generation style objects saved by the Stable Diffusion WebUI."""
        res_body = self.get(A1111Webservice.Endpoints.STYLES).json()
        all_styles: list[PromptStyleData] = []
        for serialized_style in res_body:
            all_styles.append(PromptStyleData.model_validate(serialized_style))
        return all_styles

    def get_scripts(self) -> ScriptResponseData:
        """Returns available scripts installed to the Stable Diffusion WebUI.
        Returns
        -------
        dict
            Response will have 'txt2img' and 'img2img' keys, each holding a list of scripts available for that mode.
        """
        return ScriptResponseData.model_validate(self.get(A1111Webservice.Endpoints.SCRIPTS).json())

    def get_script_info(self) -> list[ScriptInfo]:
        """Returns information on expected script parameters
        Returns
        -------
        list of dict
            Objects defining all parameters required by each script.
        """
        return [ScriptInfo.model_validate(item) for item in self.get(A1111Webservice.Endpoints.SCRIPT_INFO).json()]

    def _get_name_list(self, endpoint: str) -> list[str]:
        res_body = self.get(endpoint).json()
        return [obj['name'] for obj in res_body]

    def get_samplers(self) -> list[SamplerInfo]:
        """Returns the list of image sampler algorithms available for image generation."""
        return [SamplerInfo.model_validate(item)
                for item in self.get(A1111Webservice.Endpoints.SAMPLERS).json()]

    def get_upscalers(self) -> list[UpscalerInfo]:
        """Returns the list of image upscalers available."""
        return [UpscalerInfo.model_validate(item)
                for item in self.get(A1111Webservice.Endpoints.UPSCALERS).json()]

    def get_latent_upscale_modes(self) -> list[str]:
        """Returns the list of Stable Diffusion enhanced upscaling modes."""
        return self._get_name_list(A1111Webservice.Endpoints.LATENT_UPSCALE_MODES)

    def get_hypernetworks(self) -> list[str]:
        """Returns the list of hypernetworks available.

        Hypernetworks are a simpler form of model for augmenting full Stable Diffusion models. Each hypernetwork
        introduces a single style or concept.
        """
        return self._get_name_list(A1111Webservice.Endpoints.HYPERNETWORKS)

    def get_models(self) -> list[ModelInfo]:
        """Returns the list of available Stable Diffusion models cached by the webui.

        If available models may have changed, instead consider using the slower refresh_checkpoints method.
        """
        return [ModelInfo.model_validate(item)
                for item in self.get(A1111Webservice.Endpoints.SD_MODELS).json()]

    def get_vae(self) -> list[VaeInfo]:
        """Returns the list of available Stable Diffusion VAE models cached by the webui.

        If available models may have changed, instead consider using the slower refresh_vae method.
        """
        try:
            vae_models = self.get(A1111Webservice.Endpoints.VAE_MODELS).json()
        except ServerError:
            vae_models = self.get(A1111Webservice.ForgeEndpoints.SD_MODULES).json()
        return [VaeInfo.model_validate(item) for item in vae_models]

    def get_controlnet_version(self) -> int:
        """
        Returns the installed version of the Stable Diffusion ControlNet extension, or raises ServerError if the
        extension is not installed.

        Forge's built-in ControlNet has no version endpoint, so this raises there even though ControlNet works; use
        `get_capabilities` to check for ControlNet support.
        """
        return self.get(A1111Webservice.Endpoints.CONTROLNET_VERSION).json()['version']

    def get_controlnet_models(self) -> ControlNetModelResponse:
        """Returns a dict defining the models available to the Stable Diffusion ControlNet extension."""
        return ControlNetModelResponse.model_validate(
                    self.get(A1111Webservice.Endpoints.CONTROLNET_MODELS).json())

    def get_controlnet_modules(self) -> ControlNetModuleResponse:
        """Returns a dict defining the modules available to the Stable Diffusion ControlNet extension."""
        return ControlNetModuleResponse.model_validate(
                    self.get(A1111Webservice.Endpoints.CONTROLNET_MODULES).json())

    def get_controlnet_control_types(self) -> ControlTypeResponse:
        """Returns a dict defining the control types available to the Stable Diffusion ControlNet extension."""
        return cast(ControlTypeResponse, self.get(A1111Webservice.Endpoints.CONTROLNET_CONTROL_TYPES).json())

    def get_controlnet_settings(self) -> dict[str, Any]:
        """Returns the current settings applied to the Stable Diffusion ControlNet extension."""
        return self.get(A1111Webservice.Endpoints.CONTROLNET_SETTINGS).json()

    def get_controlnet_preprocessors(self, update_cache=False) -> list[ControlNetPreprocessor]:
        """Queries the API for ControlNet preprocessor modules, and parameterizes and returns all options."""
        if update_cache or self._preprocessor_cache is None:
            modules = self.get_controlnet_modules()
            module_names = modules.module_list
            module_details = modules.module_details
            self._preprocessor_cache = get_all_preprocessors(module_names, module_details)
        assert self._preprocessor_cache is not None
        return deepcopy(self._preprocessor_cache)

    def get_controlnet_type_categories(self) -> dict[str, ControlTypeDef]:
        """Gets the set of valid ControlNet preprocessor/model categories, taking into account available options and
           API category definitions if possible."""
        modules = self.get_controlnet_modules()
        models = self.get_controlnet_models()
        try:
            control_type_defs = self.get_controlnet_control_types()
        except (KeyError, ServerError):
            control_type_defs = None
        preprocessor_names = modules.module_list
        model_names = models.model_list
        control_type_builder = ControlNetCategoryBuilder(preprocessor_names, model_names, None, control_type_defs)
        return control_type_builder.get_control_types()

    def get_loras(self) -> list[LoraInfo]:
        """Returns the list of available Stable Diffusion LoRA models cached by the webui.

        If available models may have changed, instead consider using the slower refresh_loras method.
        """
        return [LoraInfo.model_validate(item)
                for item in self.get(A1111Webservice.Endpoints.LORA_MODELS).json()]

    def get_thumbnail(self, file_path: str) -> Optional[Image.Image]:
        """Attempts to load one of the extra model thumbnails given a path parameter."""
        try:
            res = self.get(A1111Webservice.Endpoints.EXTRA_NW_THUMB,
                           url_params={'filename': file_path})
            if not res.ok:
                return None
            return image_from_bytes(res.content)
        except (SDBackendError, OSError, ValueError) as err:
            logger.error(f'Failed to load thumbnail "{file_path}": {err}')
            return None

    def _get_if_present(self, endpoint: str) -> Optional[Any]:
        """GET `endpoint` and return its JSON body, or None if the server answers 404 (an extension or feature it
        lacks)."""
        try:
            return self.get(endpoint).json()
        except ServerError as err:
            if err.status_code == 404:
                return None
            raise

    def _has_post_route(self, endpoint: str) -> bool:
        """Whether the server serves the POST-only `endpoint`, probed with a GET that FastAPI answers with 405 if the
        route exists and 404 if not."""
        try:
            self.get(endpoint)
        except ServerError as err:
            return err.status_code != 404
        return True

    # Backend discovery methods:

    def list_checkpoints(self) -> list[BackendOption]:
        """List checkpoints, implementing `Backend.list_checkpoints`.

        Each `name` is the checkpoint's file name relative to the models directory, which stays the same once the
        server hashes the file. `display_name` is its title as WebUI shows it.
        """
        return [BackendOption(name=re.sub(CHECKPOINT_TITLE_HASH_PATTERN, '', model.title), display_name=model.title)
                for model in self.get_models()]

    def list_vaes(self) -> list[BackendOption]:
        """List VAE models, implementing `Backend.list_vaes`."""
        return [BackendOption(name=vae.model_name) for vae in self.get_vae()]

    def list_loras(self) -> list[BackendOption]:
        """List LoRA models, implementing `Backend.list_loras`. `display_name` is the LoRA's alias, if it differs."""
        return [BackendOption(name=lora.name, display_name=lora.alias if lora.alias != lora.name else None)
                for lora in self.get_loras()]

    def list_hypernetworks(self) -> list[BackendOption]:
        """List hypernetworks, implementing `Backend.list_hypernetworks`.

        Forge Neo has no hypernetwork support or `/sdapi/v1/hypernetworks`, and lists none.
        """
        hypernetworks = self._get_if_present(A1111Webservice.Endpoints.HYPERNETWORKS)
        if hypernetworks is None:
            return []
        return [BackendOption(name=hypernetwork['name']) for hypernetwork in hypernetworks]

    def list_samplers(self) -> list[BackendOption]:
        """List samplers by shared name, implementing `Backend.list_samplers`. `display_name` is the WebUI name."""
        return [BackendOption(name=comfyui_sampler_name(sampler.name), display_name=sampler.name)
                for sampler in self.get_samplers()]

    def list_schedulers(self) -> list[BackendOption]:
        """List schedulers by shared name, implementing `Backend.list_schedulers`. `display_name` is the WebUI label.

        Servers older than A1111 1.9 have no scheduler list and return an empty one.
        """
        schedulers = self._get_if_present(A1111Webservice.Endpoints.SCHEDULERS)
        if schedulers is None:
            return []
        return [BackendOption(name=comfyui_scheduler_name(scheduler['name']), display_name=scheduler.get('label'))
                for scheduler in schedulers]

    def list_upscalers(self) -> list[BackendOption]:
        """List upscalers, implementing `Backend.list_upscalers`. WebUI's no-op 'None' upscaler is left out."""
        return [BackendOption(name=upscaler.name) for upscaler in self.get_upscalers()
                if upscaler.name.lower() != 'none']

    def list_controlnet_models(self) -> list[ControlNetModel]:
        """List ControlNet models, implementing `Backend.list_controlnet_models`."""
        models = self._get_if_present(A1111Webservice.Endpoints.CONTROLNET_MODELS)
        if models is None:
            return []
        return [ControlNetModel(name) for name in ControlNetModelResponse.model_validate(models).model_list
                if name.lower() != 'none']

    def _soft_inpainting_parameters(self) -> Optional[list[ParameterDef]]:
        """Return the Soft Inpainting script's settings, or None when the img2img script list lacks it."""
        if SOFT_INPAINTING_SCRIPT_KEY not in (script.lower() for script in self.get_scripts().img2img):
            return None
        script_info = next((script for script in self.get_script_info()
                            if script.is_img2img and script.name.lower() == SOFT_INPAINTING_SCRIPT_KEY), None)
        return soft_inpainting_parameters(script_info)

    def get_soft_inpainting_parameters(self) -> list[ParameterDef]:
        """List the Soft Inpainting script's settings, implementing `Backend.get_soft_inpainting_parameters`.

        Labels, defaults and limits come from `/sdapi/v1/script-info`. Descriptions are the package's own, since the
        server has none.
        """
        return self._soft_inpainting_parameters() or []

    def get_capabilities(self) -> BackendCapabilities:
        """Report optional features, implementing `Backend.get_capabilities`.

        ControlNet support is detected from `/controlnet/model_list`, which both the sd-webui-controlnet extension and
        Forge's built-in ControlNet serve; Forge has no `/controlnet/version`. Ultimate SD Upscale and Soft Inpainting
        are detected from the img2img script list, scheduler support from `/sdapi/v1/schedulers`, and interrogation
        from whether
        `/sdapi/v1/interrogate` exists, which Forge Neo drops.
        """
        img2img_scripts = [script.lower() for script in self.get_scripts().img2img]
        return BackendCapabilities(
            controlnet=self._get_if_present(A1111Webservice.Endpoints.CONTROLNET_MODELS) is not None,
            ultimate_upscale=ULTIMATE_UPSCALE_SCRIPT in img2img_scripts,
            soft_inpainting=SOFT_INPAINTING_SCRIPT_KEY in img2img_scripts,
            scheduler=self._get_if_present(A1111Webservice.Endpoints.SCHEDULERS) is not None,
            interrogate=self._has_post_route(A1111Webservice.Endpoints.INTERROGATE),
            free_memory=False)

    def login(self, username: str, password: str) -> requests.Response:
        """Attempt to log in with a username and password."""
        body = {'username': username, 'password': password}
        return self.post(A1111Webservice.Endpoints.LOGIN, body, 'x-www-form-urlencoded',
                         throw_on_failure=False)

    def _handle_auth_error(self):
        """Asks the credentials provider for credentials, up to MAX_LOGIN_ATTEMPTS times.

        Credentials are checked with an authenticated GET to an /sdapi/v1/ endpoint, which is what --api-auth
        protects. They are installed on the session only once accepted. Raises AuthError if no provider is
        configured, the provider returns None, or every attempt is rejected."""
        if self._credentials_provider is None:
            raise AuthError('Authentication required, but no credentials_provider was configured.')
        for attempt in range(1, MAX_LOGIN_ATTEMPTS + 1):
            credentials = self._credentials_provider()
            if credentials is None:
                logger.info('Login aborted')
                raise AuthError('Login aborted: the credentials provider returned no credentials.')
            previous_auth = self._session.auth
            self.set_auth(credentials)
            try:
                response = self.get(A1111Webservice.Endpoints.PROGRESS,
                                    url_params={'skip_current_image': 'true'},
                                    fail_on_auth_error=True, throw_on_failure=False)
            except BaseException:
                self.set_auth(previous_auth)
                raise
            if response.status_code != 401:
                if not response.ok:
                    logger.warning(f'Credential check returned status {response.status_code}')
                return
            self.set_auth(previous_auth)
            logger.warning(f'Login attempt {attempt} of {MAX_LOGIN_ATTEMPTS} was rejected (status 401).')
        raise AuthError(f'Authentication failed after {MAX_LOGIN_ATTEMPTS} attempts: the server returned status 401.'
                        ' Check the credentials and the server\'s --api-auth setting.')
