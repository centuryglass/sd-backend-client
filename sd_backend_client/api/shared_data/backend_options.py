"""The types `Backend`'s discovery methods return, so callers can list a server's options without knowing its backend.

Each `BackendOption.name` is the value a caller passes back to the same server, so it differs between backends for
the same file. `BackendCapabilities` reports the optional features a server has.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict

__all__ = ['BackendOption', 'BackendCapabilities']


class BackendOption(BaseModel):
    """One option a server offers, such as a checkpoint or a sampler."""
    model_config = ConfigDict(frozen=True)

    name: str
    """The value to pass back to the server that listed it, in the parameter field the listing method names."""

    display_name: Optional[str] = None
    """A label the server gives the option, or None when it gives none and `name` is the label."""


class BackendCapabilities(BaseModel):
    """Optional features a server has, so callers can check before submitting a job that needs one."""
    model_config = ConfigDict(frozen=True)

    controlnet: bool
    """Whether `DiffusionParams.controlnet_units` and ControlNet preprocessor previews work.

    On WebUI this needs the ControlNet extension (or Forge's built-in equivalent).
    """

    ultimate_upscale: bool
    """Whether `DiffusionUpscalingParams.use_ultimate_upscale_script` takes effect.

    It needs the Ultimate SD Upscale script on WebUI or the `UltimateSDUpscale` node on ComfyUI. Without it, ComfyUI
    runs a plain diffusion pass and WebUI fails the request.
    """

    soft_inpainting: bool
    """Whether `DiffusionParams.soft_inpainting` takes effect.

    It needs the Soft Inpainting script on WebUI (built into A1111 1.8+ and Forge) or the `DifferentialDiffusion` node
    on ComfyUI. Without it, submitting with soft inpainting on raises `ValueError`.
    """

    scheduler: bool
    """Whether `DiffusionParams.scheduler` takes effect. WebUI accepts a separate scheduler from A1111 1.9 on."""

    interrogate: bool
    """Whether the client has a working `interrogate` method (`A1111Webservice.interrogate`)."""

    free_memory: bool
    """Whether the client has a working `free_memory` method (`ComfyUiWebservice.free_memory`)."""
