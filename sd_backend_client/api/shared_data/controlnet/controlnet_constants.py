"""Types and constants associated with ControlNet.  All values are either not specific to a single API, or are reused
   because they don't have a useful equivalent in both APIs."""
from typing import NotRequired
from typing_extensions import TypedDict



# FORGE/COMFYUI CONTROLNET CONSTANTS:
# The WebUI's /controlnet/control_types endpoint sorts preprocessors and ControlNet models into control types.
# ComfyUI has only its preprocessor node categories, so these definitions sort its options. On the WebUI they add to
# the endpoint's lists, and replace them on a server without it.

NONE_VALUE = 'None'

class ControlTypeDef(TypedDict):
    """A ControlNet category and its available options and defaults, on either backend."""
    module_list: list[str]
    model_list: list[str]
    default_option: str
    default_model: str


class StaticControlTypeDef(TypedDict):
    """A built-in category definition with optional patterns for additional available options."""

    module_list: list[str]
    model_list: list[str]
    default_option: str
    default_model: str
    preprocessor_pattern: NotRequired[str]
    model_pattern: NotRequired[str]


STATIC_CONTROL_TYPE_DEFS: dict[str, StaticControlTypeDef] = {
    'All': {
        'module_list': [],
        'model_list': [],
        'default_option': '',
        'default_model': '',
        'preprocessor_pattern': r'.*',
        'model_pattern': r'.*'
    },
    'Canny': {
        'module_list': [NONE_VALUE, 'canny', 'invert (from white bg & black line)'],
        'model_list': [NONE_VALUE],
        'default_option': 'canny',
        'default_model': 'control_v11p_sd15_canny [d14c016b]',
        'preprocessor_pattern': r'(?i)canny',
        'model_pattern': r'(?i)canny',
    },
    'Depth': {
        'module_list': [NONE_VALUE],
        'model_list': [NONE_VALUE],
        'default_option': 'depth_marigold',
        'default_model': 'control_v11f1p_sd15_depth [cfd03158]',
        'preprocessor_pattern': r'(?i)depth',
        'model_pattern': r'(?i)depth',
    },
    'IP-Adapter': {
        'module_list': [NONE_VALUE],
        'model_list': [NONE_VALUE],
        'default_option': 'InsightFace+CLIP-H (IPAdapter)',
        'default_model': 'ip-adapter-plus-face_sd15 [71693645]',
        'preprocessor_pattern': r'(?i)IPAdapter',
        'model_pattern': r'(?i)ip-adapter',
    },
    'Inpaint': {
        'module_list': [NONE_VALUE],
        'model_list': [NONE_VALUE],
        'default_option': 'inpaint_only+lama',
        'default_model': 'control_v11p_sd15_inpaint [ebff9138]',
        'preprocessor_pattern': r'(?i)inpaint',
        'model_pattern': r'(?i)inpaint',
    },
    'Instant-ID': {
        'module_list': [NONE_VALUE, 'InsightFace (InstantID)'],
        'model_list': [NONE_VALUE, 'diffusion_pytorch_model.safetensors'],
        'default_option': 'InsightFace (InstantID)',
        'default_model': NONE_VALUE,
        'preprocessor_pattern': r'(?i).*instant_id.*',
        'model_pattern': r'(?i).*instant_?id.*',
    },
    'InstructP2P': {
        'module_list': [NONE_VALUE],
        'model_list': [NONE_VALUE, 'control_v11e_sd15_ip2p [c4bb465c]'],
        'default_option': NONE_VALUE,
        'default_model': 'control_v11e_sd15_ip2p [c4bb465c]'
    },
    'NormalMap': {
        'module_list': [NONE_VALUE, 'normalbae'],
        'model_list': [NONE_VALUE],
        'default_option': 'normalbae',
        'default_model': 'control_v11p_sd15_normalbae [316696f1]',
        'preprocessor_pattern': r'(?i)normal',
        'model_pattern': r'(?i)normal',
    },
    'Lineart': {
        'module_list': [NONE_VALUE, 'lineart_standard (from white bg & black line)',
                        'invert (from white bg & black line)'],
        'model_list': [NONE_VALUE, 'control_v11p_sd15_lineart [43d4be0d]'],
        'default_option': 'lineart_standard (from white bg & black line)',
        'default_model': 'control_v11p_sd15_lineart [43d4be0d]',
        'preprocessor_pattern': r'(?i)line',
        'model_pattern': r'(?i).*lineart.*',
    },
    'MLSD': {
        'module_list': [NONE_VALUE, 'mlsd', 'invert (from white bg & black line)'],
        'model_list': [NONE_VALUE, 'control_v11p_sd15_mlsd [aca30ff0]'],
        'default_option': 'mlsd',
        'default_model': 'control_v11p_sd15_mlsd [aca30ff0]',
        'preprocessor_pattern': r'(?i).*mlsd.*',
        'model_pattern': r'(?i).*mlsd.*',
    },
    'OpenPose': {
        'module_list': [NONE_VALUE, 'openpose'],
        'model_list': [NONE_VALUE, 'control_v11p_sd15_openpose [cab727d4]'],
        'default_option': 'openpose_full',
        'default_model': 'control_v11p_sd15_openpose [cab727d4]',
        # DWPreprocessor and AnimalPosePreprocessor are ComfyUI's DWPose and animal pose nodes.
        'preprocessor_pattern': r'(?i)openpose|densepose|animal_?pose|^DWPreprocessor$',
        'model_pattern': r'(?i).*openpose.*',
    },
    'PhotoMaker': {
        'module_list': [NONE_VALUE, 'ClipVision (Photomaker)'],
        'model_list': [NONE_VALUE, 'photomaker-v1.bin'],
        'default_option': 'ClipVision (Photomaker)',
        'default_model': 'photomaker-v1.bin',
    },
    'Recolor': {
        'module_list': [NONE_VALUE, 'recolor_luminance', 'recolor_intensity'],
        'model_list': [NONE_VALUE, 't2iadapter_color-fp16 [743b5c62]'],
        'default_option': 'recolor_luminance',
        'default_model': 't2iadapter_color-fp16 [743b5c62]',
        'preprocessor_pattern': r'(?i).*recolor.*',
        'model_pattern': r'(?i).*color.*',
    },
    # NOTE: Reference and revision are special cases, they use only a pre-processor and no model.
    'Reference': {
        'module_list': [NONE_VALUE, 'reference_only', 'reference_adain+attn', 'reference_adain'],
        'model_list': [NONE_VALUE],
        'default_option': 'reference_only',
        'default_model': NONE_VALUE
    },
    'Revision': {
        'module_list': [NONE_VALUE, 'CLIP-G (Revision)', 'CLIP-G (Revision ignore prompt)'],
        'model_list': [NONE_VALUE],
        'default_option': 'CLIP-G (Revision)',
        'default_model': NONE_VALUE
    },
    'Scribble': {
        'module_list': [NONE_VALUE, 'invert (from white bg & black line)', 'scribble_pidinet'],
        'model_list': [NONE_VALUE],
        'default_option': 'scribble_pidinet',
        'default_model': 'control_v11p_sd15_scribble [d4ba51ff]',
        'preprocessor_pattern': r'(?i).*scribble.*',
        'model_pattern': r'(?i).*scribble.*',
    },
    'Segmentation': {
        'module_list': [NONE_VALUE, 'seg_ofade20k', 'mobile_sam'],
        'model_list': ['control_v11p_sd15_seg [e1f51eb9]'],
        'default_option': 'seg_ofade20k',
        'default_model': 'control_v11p_sd15_seg [e1f51eb9]',
        'preprocessor_pattern': r'(?i).*seg_.*',
        'model_pattern': r'(?i).*_seg.*',
    },
    'Shuffle': {
        'module_list': [NONE_VALUE, 'shuffle'],
        'model_list': ['control_v11e_sd15_shuffle [526bfdae]'],
        'default_option': 'shuffle',
        'default_model': 'control_v11e_sd15_shuffle [526bfdae]',
        'model_pattern': r'(?i).*shuffle.*',
    },
    'SoftEdge': {
        'module_list': [NONE_VALUE, 'softedge_pidinet'],
        'model_list': [NONE_VALUE, 'control_v11p_sd15_softedge [a8575a2a]'],
        'default_option': 'softedge_pidinet',
        'default_model': 'control_v11p_sd15_softedge [a8575a2a]',
        # ComfyUI's soft edge nodes and models are named for their method (HED, PiDiNet, TEED), not "softedge".
        # The lookahead keeps scribble and sketch variants of those methods in Scribble and T2I-Adapter.
        'preprocessor_pattern': r'(?i)softedge|^(?!.*(?:scribble|sketch)).*(?<![a-z])(?:hed|pidi|teed)',
        'model_pattern': r'(?i)softedge|^(?!.*scribble).*(?<![a-z])(?:hed|pidi|teed)',
    },
    'T2I-Adapter': {
        'module_list': [NONE_VALUE, 't2ia_style_clipvision'],
        'model_list': ['t2iadapter_canny-fp16 [f2e7f7cd]'],
        'default_option': 't2ia_style_clipvision',
        'default_model': 't2iadapter_canny-fp16 [f2e7f7cd]',
        'preprocessor_pattern': r'(?i).*t2ia_.*',
        'model_pattern': r'(?i).*t2iadapter.*',
    },
    'Tile': {
        'module_list': [NONE_VALUE, 'tile_resample', 'tile_colorfix+sharp', 'tile_colorfix'],
        'model_list': [NONE_VALUE, 'control_v11f1e_sd15_tile [a371b31b]'],
        'default_option': 'tile_resample',
        'default_model': 'control_v11f1e_sd15_tile [a371b31b]',
        'preprocessor_pattern': r'(?i).*tile_.*',
        'model_pattern': r'(?i).*tile.*',
    },
}
