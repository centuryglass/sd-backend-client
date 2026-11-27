"""Extra model paths for a managed ComfyUI, without running a model."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from sd_backend_client import BackendKind, DetectedInstall, DetectionResult, InstallOwner, ManagedComfyUI
from sd_backend_client.discovery.backend_detector import parse_extra_model_paths
from sd_backend_client.launch.model_paths import ModelLayout, ModelSource, model_sources_from_detection, \
    render_model_paths, write_model_paths


@pytest.mark.parametrize(('layout', 'dirs'), [
    (ModelLayout.WEBUI, ('Stable-diffusion', 'Lora', 'VAE', 'ControlNet')),
    (ModelLayout.COMFYUI, ('checkpoints', 'loras', 'vae', 'controlnet')),
    (ModelLayout.STABILITY_MATRIX, ('StableDiffusion', 'Lora', 'VAE', 'ControlNet')),
])
def test_model_paths_round_trip(layout, dirs, tmp_path):
    """Each layout produces correctly quoted paths and a readable checkpoint folder."""
    root = tmp_path / "Sam's # GPU models"
    text = render_model_paths([ModelSource(root, layout)])
    for key, folder in zip(('checkpoints', 'loras', 'vae', 'controlnet'), dirs, strict=True):
        assert f"  {key}: '{folder}'\n" in text
    expected = [root / dirs[0]]
    if layout == ModelLayout.COMFYUI:
        expected += [root / 'diffusion_models', root / 'unet']
    elif layout == ModelLayout.STABILITY_MATRIX:
        expected.append(root / 'DiffusionModels')
    assert parse_extra_model_paths(text, tmp_path) == expected


def test_model_paths_preserve_windows_spelling(tmp_path):
    """A Windows path keeps its drive, spaces, backslashes and punctuation."""
    root = Path(r"C:\My Models\Alice's #1")
    text = render_model_paths([ModelSource(root, ModelLayout.WEBUI)])
    assert parse_extra_model_paths(text, tmp_path) == [tmp_path / root / 'Stable-diffusion']


def test_sources_from_detected_installs(tmp_path):
    """Capture portable and shared locations and avoid listing the same library twice."""
    webui = tmp_path / 'webui'
    portable = tmp_path / 'portable' / 'ComfyUI'
    matrix = tmp_path / 'matrix'
    package = matrix / 'Packages' / 'ComfyUI'
    for folder in (webui / 'models', portable / 'models', matrix / 'Models'):
        folder.mkdir(parents=True)
    installs = [
        DetectedInstall(BackendKind.WEBUI, InstallOwner.NONE, webui, webui / 'launch.py'),
        DetectedInstall(BackendKind.COMFYUI, InstallOwner.NONE, portable.parent, portable / 'main.py'),
        DetectedInstall(BackendKind.COMFYUI, InstallOwner.STABILITY_MATRIX, package, package / 'main.py'),
        DetectedInstall(BackendKind.WEBUI, InstallOwner.STABILITY_MATRIX, package, package / 'launch.py'),
        DetectedInstall(BackendKind.COMFYUI, InstallOwner.CALLER, tmp_path / 'absent', tmp_path / 'absent/main.py'),
    ]
    assert model_sources_from_detection(DetectionResult(installs)) == [
        ModelSource(webui / 'models', ModelLayout.WEBUI),
        ModelSource(portable / 'models', ModelLayout.COMFYUI),
        ModelSource(matrix / 'Models', ModelLayout.STABILITY_MATRIX),
    ]


def test_generated_config_does_not_replace_existing_file(tmp_path):
    """The app must provide a fresh filename, leaving existing config bytes intact."""
    old = tmp_path / 'extra_model_paths.yaml'
    old.write_text('# user-written config\n', encoding='utf-8')
    sources = [ModelSource(tmp_path / 'models', ModelLayout.COMFYUI)]
    with pytest.raises(FileExistsError):
        write_model_paths(old, sources)
    generated = write_model_paths(tmp_path / 'app model paths.yaml', sources)
    assert old.read_text(encoding='utf-8') == '# user-written config\n'
    assert generated.read_text(encoding='utf-8') == render_model_paths(sources)


def test_launch_preserves_existing_flags(tmp_path):
    """The generated path is a single argument added alongside the user config."""
    config = tmp_path / 'app model paths.yaml'
    managed = ManagedComfyUI(['python', 'main.py', '--extra-model-paths-config', 'user.yaml'], tmp_path, 8189,
                             model_paths_config=config)
    assert managed.command == [
        'python', 'main.py', '--extra-model-paths-config', 'user.yaml',
        '--extra-model-paths-config', str(config), '--listen', '127.0.0.1', '--port', '8189',
    ]


@pytest.mark.parametrize('available', [False, True])
def test_checkpoints_available_uses_existing_client(monkeypatch, tmp_path, available):
    """Use discovery alone to report checkpoints and disconnect the temporary client."""
    managed = ManagedComfyUI(['python', 'main.py'], tmp_path, 8189)
    client = Mock()
    client.list_checkpoints.return_value = [object()] if available else []
    connect = Mock(return_value=client)
    monkeypatch.setattr(managed, 'connect', connect)
    assert managed.checkpoints_available() is available
    client.list_checkpoints.assert_called_once_with()
    client.disconnect.assert_called_once_with()


def test_no_sources_produces_empty_mapping(tmp_path):
    """A choice of no model sources leaves the default ComfyUI layout unchanged."""
    text = render_model_paths([])
    assert text == '{}\n'
    assert parse_extra_model_paths(text, tmp_path) == []


def test_line_break_in_folder_is_rejected(tmp_path):
    """A source folder must be representable as one configuration line."""
    with pytest.raises(ValueError, match='line breaks'):
        render_model_paths([ModelSource(tmp_path / 'nested\nname', ModelLayout.COMFYUI)])
