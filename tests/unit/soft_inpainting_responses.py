"""Canned soft inpainting discovery responses for each backend, copied from live servers."""

WEBUI_SOFT_INPAINTING_INFO = {
    'name': 'soft inpainting', 'is_alwayson': True, 'is_img2img': True,
    'args': [{'label': 'Soft inpainting', 'value': False}] + [
        {'label': label, 'value': value, 'minimum': minimum, 'maximum': maximum, 'step': step}
        for label, value, minimum, maximum, step in (
            ('Schedule bias', 1, 0, 8, 0.1), ('Preservation strength', 0.5, 0, 8, 0.05),
            ('Transition contrast boost', 4, 1, 32, 0.5), ('Mask influence', 0, 0, 1, 0.05),
            ('Difference threshold', 0.5, 0, 8, 0.25), ('Difference contrast', 2, 0, 8, 0.25))]}
"""The Soft Inpainting entry of `/sdapi/v1/script-info`, as A1111 1.10 and every Forge variant return it."""

COMFYUI_DIFFERENTIAL_DIFFUSION_INFO = {'DifferentialDiffusion': {
    'input': {'required': {'model': ['MODEL', {}]},
              'optional': {'strength': ['FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}]}},
    'input_order': {'required': ['model'], 'optional': ['strength']}}}
"""`/object_info/DifferentialDiffusion` on a ComfyUI build with the optional `strength` input, trimmed."""
