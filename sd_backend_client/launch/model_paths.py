"""Write a separate ComfyUI configuration for models already on disk."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from sd_backend_client.discovery.detected_backend import BackendKind, DetectionResult, InstallOwner


class ModelLayout(Enum):
    """How model category folders are named inside a model source."""

    COMFYUI = 'comfyui'
    WEBUI = 'webui'
    STABILITY_MATRIX = 'stability_matrix'


@dataclass(frozen=True)
class ModelSource:
    """A model folder and its directory naming convention."""

    path: Path
    layout: ModelLayout


_LAYOUTS = {
    ModelLayout.COMFYUI: {
        'checkpoints': 'checkpoints', 'diffusion_models': 'diffusion_models', 'unet': 'unet',
        'loras': 'loras', 'vae': 'vae', 'controlnet': 'controlnet', 'embeddings': 'embeddings',
    },
    ModelLayout.WEBUI: {
        'checkpoints': 'Stable-diffusion', 'loras': 'Lora', 'vae': 'VAE',
        'controlnet': 'ControlNet', 'upscale_models': 'ESRGAN',
    },
    ModelLayout.STABILITY_MATRIX: {
        'checkpoints': 'StableDiffusion', 'diffusion_models': 'DiffusionModels',
        'loras': 'Lora', 'vae': 'VAE', 'controlnet': 'ControlNet', 'embeddings': 'Embeddings',
    },
}


def model_sources_from_detection(detection: DetectionResult) -> list[ModelSource]:
    """Return existing model folders from detected installs, in discovery order.

    A Stability Matrix package under `Packages/` shares the library's `Models/` folder with its siblings.
    If there is no shared folder, use the package's own model folder.
    """
    sources: dict[Path, ModelSource] = {}
    for install in detection.installs:
        if install.kind == BackendKind.COMFYUI:
            path, layout = install.entry_point.parent / 'models', ModelLayout.COMFYUI
        else:
            path, layout = install.path / 'models', ModelLayout.WEBUI
        if install.owner == InstallOwner.STABILITY_MATRIX and install.path.parent.name.casefold() == 'packages':
            shared = install.path.parent.parent / 'Models'
            if shared.is_dir():
                path, layout = shared, ModelLayout.STABILITY_MATRIX
        if path.is_dir():
            sources.setdefault(path.resolve(), ModelSource(path, layout))
    return list(sources.values())


def _quote(value: str) -> str:
    # A quoted YAML scalar preserves drive letters, backslashes, apostrophes, spaces and '#'.
    if '\n' in value or '\r' in value:
        raise ValueError(f'Model paths cannot contain line breaks: {value!r}')
    return "'" + value.replace("'", "''") + "'"


def render_model_paths(sources: Iterable[ModelSource]) -> str:
    """Return extra-model-paths YAML; use absolute source folders when writing outside the install."""
    sections = []
    for index, source in enumerate(sources, start=1):
        lines = [f'shared_models_{index}:', f'  base_path: {_quote(str(source.path))}']
        lines.extend(f'  {key}: {_quote(folder)}' for key, folder in _LAYOUTS[source.layout].items())
        sections.append('\n'.join(lines))
    return '\n\n'.join(sections) + '\n' if sections else '{}\n'


def write_model_paths(destination: Path, sources: Iterable[ModelSource]) -> Path:
    """Write to a new file in a caller-owned folder; never overwrite an existing configuration.

    Resolve source folders first, since ComfyUI resolves relative paths against the configuration's location.
    Returns the absolute file path to pass to `ManagedComfyUI`.
    """
    config = render_model_paths(ModelSource(source.path.absolute(), source.layout) for source in sources)
    with destination.open('x', encoding='utf-8') as handle:
        handle.write(config)
    return destination.absolute()
