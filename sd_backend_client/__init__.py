"""A headless client for the ComfyUI and Forge/A1111 WebUI Stable Diffusion APIs.

The names in `__all__` are the supported public API: import them from `sd_backend_client` itself. Every other module
and name, including the ComfyUI node graph and workflow builders, the HTTP layer and the wire formats under
`sd_backend_client.api`, stays importable but is internal and may change or move in any release.

A package class that a `Backend` or `GenerationHandle` method takes or returns, or that a field of an exported model
holds, belongs in `__all__` too; `tests/unit/test_public_api.py` enforces this. The clients' backend-specific methods
outside `Backend` may use internal types.

Each name in `__all__` needs a directive on a `docs/reference/` page, or the docs build fails (AGENTS.md,
"Documentation site").
"""
from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.api.detect import connect_to_backend
from sd_backend_client.api.shared_data.api_datatypes import DiffusionUpscalingParams
from sd_backend_client.api.shared_data.backend import Backend
from sd_backend_client.api.shared_data.backend_options import BackendCapabilities, BackendOption
from sd_backend_client.api.shared_data.controlnet.controlnet_category_builder import CONTROLNET_MODEL_NONE, \
    PREPROCESSOR_NONE
from sd_backend_client.api.shared_data.controlnet.controlnet_constants import ControlTypeDef
from sd_backend_client.api.shared_data.controlnet.controlnet_model import ControlNetModel
from sd_backend_client.api.shared_data.controlnet.controlnet_preprocessor import ControlNetPreprocessor, \
    ParameterDef, PreprocessorParams
from sd_backend_client.api.shared_data.controlnet.controlnet_unit import ControlNetUnit
from sd_backend_client.api.shared_data.diffusion_params import DiffusionParams
from sd_backend_client.api.shared_data.generation_handle import GenerationHandle, GenerationProgress, \
    GenerationResult, GenerationStatus, ProgressCallback
from sd_backend_client.api.webui.controlnet_webui_constants import CONTROL_MODE_PARAM_KEY, \
    PREPROCESSOR_RES_PARAM_KEY, RESIZE_MODE_PARAM_KEY
from sd_backend_client.api.webui.diffusion_request_body import DiffusionRequestBody, InpaintFillOption, ResizeMode
from sd_backend_client.api.webui.script_info_types import ScriptRequestData
from sd_backend_client.discovery.backend_detector import SearchLocations, default_search_locations, detect
from sd_backend_client.discovery.detected_backend import BackendKind, DetectedInstall, DetectedServer, \
    DetectionResult, Finding, FindingCode, FindingSeverity, InstallOwner
from sd_backend_client.errors import AuthError, BackendConnectionError, BackendLaunchError, BackendTimeoutError, \
    GenerationError, LaunchCancelledError, SDBackendError, ServerError, UnexpectedResponseError, \
    WorkflowValidationError
from sd_backend_client.launch.managed_comfyui import ManagedComfyUI, ManagedState

__all__ = [
    # Clients
    'Backend',
    'connect_to_backend',
    'A1111Webservice',
    'ComfyUiWebservice',
    # Generation parameters
    'DiffusionParams',
    'DiffusionRequestBody',
    'ResizeMode',
    'InpaintFillOption',
    'ScriptRequestData',
    'ComfyUIDiffusionParams',
    'DiffusionUpscalingParams',
    # ControlNet
    'ControlNetUnit',
    'ControlNetModel',
    'ControlNetPreprocessor',
    'PreprocessorParams',
    'ParameterDef',
    'ControlTypeDef',
    'PREPROCESSOR_NONE',
    'CONTROLNET_MODEL_NONE',
    'CONTROL_MODE_PARAM_KEY',
    'RESIZE_MODE_PARAM_KEY',
    'PREPROCESSOR_RES_PARAM_KEY',
    # Discovery
    'BackendOption',
    'BackendCapabilities',
    # Local detection
    'detect',
    'SearchLocations',
    'default_search_locations',
    'DetectionResult',
    'DetectedInstall',
    'DetectedServer',
    'Finding',
    'FindingCode',
    'FindingSeverity',
    'BackendKind',
    'InstallOwner',
    # Running a local install
    'ManagedComfyUI',
    'ManagedState',
    # Generation handles
    'GenerationHandle',
    'GenerationStatus',
    'GenerationProgress',
    'GenerationResult',
    'ProgressCallback',
    # Errors
    'SDBackendError',
    'BackendConnectionError',
    'BackendTimeoutError',
    'AuthError',
    'ServerError',
    'UnexpectedResponseError',
    'WorkflowValidationError',
    'GenerationError',
    'BackendLaunchError',
    'LaunchCancelledError',
]
