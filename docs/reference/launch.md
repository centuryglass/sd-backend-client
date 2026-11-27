# Running a local install

`ManagedComfyUI` runs a ComfyUI install on disk as a child process, such as one `detect` found, and gives back a
connected `ComfyUiWebservice` once it serves. It only ever stops the process it started.

::: sd_backend_client.ManagedComfyUI

::: sd_backend_client.ManagedState

## Reusing an existing model library

Use `model_sources_from_detection(result)` to get existing ComfyUI, Forge/Automatic1111 or Stability Matrix model
roots. The caller can let users choose a subset or supply its own `ModelSource` entries. Allocate a new path in an
application-owned directory for each generated config. The writer will not overwrite an existing file.

```python
from pathlib import Path
from sd_backend_client import ManagedComfyUI, model_sources_from_detection, write_model_paths

def prepare_comfyui(detection, app_config_dir, launch_command, install_dir):
    sources = model_sources_from_detection(detection)
    extra = write_model_paths(Path(app_config_dir) / "shared-models.yaml", sources)
    return ManagedComfyUI(launch_command, install_dir, 8188, model_paths_config=extra)
```

The generated file only points at existing folders. It does not copy models, create symlinks, or modify
`extra_model_paths.yaml` in the ComfyUI installation. `checkpoints_available()` checks ComfyUI's list rather than
attempting to load a checkpoint. Call it after `start()` and `wait_until_ready()`. As with other managed
launches, call `stop()` when the app no longer needs the backend.

::: sd_backend_client.ModelLayout

::: sd_backend_client.ModelSource

::: sd_backend_client.model_sources_from_detection

::: sd_backend_client.render_model_paths

::: sd_backend_client.write_model_paths
