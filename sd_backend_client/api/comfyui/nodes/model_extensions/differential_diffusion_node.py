"""A ComfyUI node that patches a model to treat the latent noise mask as a per-pixel denoising schedule."""
from typing import ClassVar, Optional

from pydantic import Field

from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, Connection, Output

STRENGTH_KEY = 'strength'
"""The node's optional `strength` input, absent on older ComfyUI builds, which reject inputs they don't list."""


class DifferentialDiffusionNode(ComfyNode):
    """A ComfyUI node that patches a model to treat the latent noise mask as a per-pixel denoising schedule.

    A pixel whose mask value is 0.3 starts changing about 30% of the way through sampling. Without a noise mask on the
    sampled latent, the patch has no effect.
    """
    CLASS_TYPE: ClassVar[str] = 'DifferentialDiffusion'

    model: Connection = None
    strength: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    """Blend between the binary schedule mask (1.0) and the soft mask (toward 0). None omits the input.

    ComfyUI treats 0 the same as 1.0, so the field rejects 0.
    """

    model_out = Output(0)
