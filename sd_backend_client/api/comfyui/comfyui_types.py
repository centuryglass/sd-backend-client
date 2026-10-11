"""Typedefs for ComfyUI API data."""
from typing import TypeAlias, Literal, Optional, Any

from pydantic import BaseModel

# Misc. types used within multiple request/response objects:
FileType: TypeAlias = Literal['input', 'temp', 'output']

# I/O parameter definitions:
ComplexTypes: TypeAlias = Literal[
    'CLIP',
    'CONDITIONING',
    'IMAGE',
]

ComplexInputType: TypeAlias = Literal[
                                  'CLIP',
                                  'CLIP_VISION',
                                  'CLIP_VISION_OUTPUT',
                                  'CONDITIONING',
                                  'CONTROL_NET',
                                  'GLIGEN',
                                  'LATENT',
                                  'MASK',
                                  'MODEL',
                                  'STYLE_MODEL',
                                  'VAE'
                              ] | ComplexTypes

NodeReturnType: TypeAlias = Literal[
                                'AUDIO',
                                'LATENT',
                                'LATENT_OPERATION',
                            ] | ComplexTypes

DisplayType: TypeAlias = Literal['color', 'number', 'slider']

CONTROLNET_PREPROCESSOR_CATEGORY = 'ControlNet Preprocessors'
CONTROLNET_PREPROCESSOR_OUTPUT_NAME = ['IMAGE']
CONTROLNET_PREPROCESSOR_REQUIRED_INPUT = 'image'


class ParamDef(BaseModel):
    """Input value parameters accepted across all types."""
    tooltip: Optional[str] = None


class IntParamDef(ParamDef):
    """Defines int parameter ranges, step size, and default value."""
    # min/max/step are omitted by some ComfyUI nodes (e.g. KSampler's `steps` and `seed` carry no `step`).
    default: int
    min: Optional[int] = None
    max: Optional[int] = None
    step: Optional[int] = None
    display: Optional[DisplayType] = None


class FloatParamDef(ParamDef):
    """Defines float parameter ranges, step size, and default value."""
    default: float
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    round: Optional[float] = None


class StrParamDef(ParamDef):
    """Defines string parameter requirements."""
    default: Optional[str] = None
    multiline: bool = False
    dynamicPrompts: bool = False


class BoolParamDef(ParamDef):
    """Defines a default boolean parameter value."""
    default: bool


# A node input is a tuple of (type-name-or-option-list, optional param-def dict). The exact per-type shape varies
# widely across core and custom nodes, and consumers only index [0]/[1] (re-validating [1] as the appropriate
# ParamDef when they need it), so keep this permissive rather than modeling every variant as a strict union.
InputParam: TypeAlias = tuple[Any, ...]


class InputTypeDef(BaseModel):
    """Defines inputs for a ComfyUI node."""
    required: dict[str, InputParam]
    optional: Optional[dict[str, InputParam]] = None


# API request/parameter/response types, by endpoint:

IMAGE_UPLOAD_FILE_NAME = 'image'


class ImageUploadParams(BaseModel):
    """Image upload body."""
    type: Optional[FileType] = None
    subfolder: Optional[str] = None
    overwrite: Optional[str] = None  # 'true' or '1' to overwrite


class ImageUploadResponse(BaseModel):
    """Image upload response body."""
    name: str
    subfolder: str
    type: FileType


class ImageFileReference(BaseModel):
    """References an uploaded or generated image file."""
    filename: str
    subfolder: str
    type: Optional[FileType] = None  # used when output_dir from filename is None, default = 'output'


class MaskUploadParams(ImageUploadParams):
    """Mask upload body key constants."""
    original_ref: str  # when parsed as JSON, should be ImageFileReference


class SystemObject(BaseModel):
    """System data object used in SYSTEM_STATS responses."""
    os: str
    ram_total: int
    ram_free: int
    comfyui_version: str
    python_version: str
    pytorch_version: str
    embedded_python: bool
    argv: list[str]


class DeviceObject(BaseModel):
    """Torch device object used in SYSTEM_STATS responses."""
    name: str
    type: str
    index: int
    vram_total: int
    vram_free: int
    torch_vram_total: int
    torch_vram_free: int


class SystemStatResponse(BaseModel):
    """Response body format for the SYSTEM_STATS endpoint."""
    system: SystemObject
    devices: list[DeviceObject]


class NodeInfoResponse(BaseModel):
    """Response defining a ComfyUI node."""
    input: InputTypeDef
    input_order: dict[str, list[str]]  # keys are 'required' / 'optional' / 'hidden'
    # One entry per output; each is a type-name string, or a list of names for a COMBO output. Kept permissive
    # because get_controlnet_preprocessors validates every installed node, including arbitrary custom ones.
    output: list[Any]
    output_is_list: list[bool]  # length should match output
    output_name: list[str]  # Length should match output
    name: str
    display_name: Optional[str] = None  # get_all_preprocessors falls back to `name` when absent
    description: Optional[str] = None
    python_module: str
    category: str
    output_node: bool
    deprecated: Optional[bool] = None
    experimental: Optional[bool] = None


# QUEUED PROMPT/TASK DATA:

# Entry structure: (task_number, UUID, workflow, optional_extra_data)
# TODO: this is ugly, analyze the queue structure and make a pydantic class.
# A queue entry is (number, prompt_id, prompt, extra_data, outputs_to_execute); ComfyUI's exact arity varies by
# version, and we only read [0] (number) and [1] (prompt_id), so keep it a permissive variable-length tuple.
QueueEntry: TypeAlias = tuple[Any, ...]

ACTIVE_QUEUE_KEY = 'queue_running'
PENDING_QUEUE_KEY = 'queue_pending'


class QueueInfoResponse(BaseModel):
    """Response structure used when getting queued task info."""
    queue_running: list[QueueEntry]
    queue_pending: list[QueueEntry]


class QueueAdditionRequest(BaseModel):
    """Body structure to use when adding to the ComfyUI queue."""
    prompt: dict[str, Any]
    number: Optional[int] = None  # Sets priority
    front: Optional[bool] = None  # Pushes this job ahead of others
    client_id: Optional[str] = None  # Optional extra identifier
    extra_data: Optional[dict[str, Any]] = None  # Associate some arbitrary extra data


class ErrorEntry(BaseModel):
    """A single queue error."""
    type: str
    message: str
    details: str = ''
    extra_info: dict[str, Any] = {}  # Holds input names, input configs and received values of any JSON type.


class NodeErrorEntry(BaseModel):
    """Defines error data for a single node."""
    errors: list[ErrorEntry]
    dependent_outputs: list[str]  # connected node ids
    class_type: str


class QueueAdditionResponse(BaseModel):
    """Response structure used when a new job is queued."""
    prompt_id: Optional[str] = None  # UUID, omitted on error
    number: Optional[int] = None  # Queue number/priority, omitted on error
    error: Optional[str | ErrorEntry] = None
    node_errors: dict[str, NodeErrorEntry] = {}  # keyed by node id; empty on success

    # Set by ComfyUiWebservice, not by ComfyUI: the seed the queued workflow used.
    seed: Optional[int] = None


class QueueDeletionRequest(BaseModel):
    """Request structure used to delete queue items."""
    clear: Optional[bool] = None  # If true, the whole queue is wiped.
    delete: Optional[list[str]] = None  # Set specific queued items to delete.


class PromptStatusMessageData(BaseModel):
    """Extra data bundled with queued task messages.

    Fields past `timestamp` are set only by the message types that carry them: `nodes` by `execution_cached`, the
    node fields by `execution_error` and `execution_interrupted`, and the exception fields by `execution_error`.
    """
    prompt_id: str  # UUID
    timestamp: int
    nodes: Optional[list[str]] = None
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None


PromptStatusMessage: TypeAlias = tuple[str, PromptStatusMessageData]


class PromptExecStatus(BaseModel):
    """Status data associated with a task in the history, directly from ComfyUI/execution.py."""
    status_str: Literal['success', 'error']
    completed: bool
    messages: list[PromptStatusMessage]


class PromptExecOutputs(BaseModel):
    """Returns generated file info for a completed task.

    `images` is empty for output nodes that save no images, such as text or custom output nodes.
    """
    images: list[ImageFileReference] = []
    # TODO: track down format for other possible output types


class PromptHistory(BaseModel):
    """Prompt execution data from the /history endpoint."""
    prompt: QueueEntry
    outputs: dict[str, PromptExecOutputs]  # keys are output node ids
    status: PromptExecStatus


QueueHistoryResponse: TypeAlias = dict[str, PromptHistory]  # key is prompt_id


class FreeMemoryRequest(BaseModel):
    """Request format used when requesting cached models and memory to be cleared."""
    unload_models: bool
    free_memory: bool
