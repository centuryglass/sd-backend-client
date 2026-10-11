# ComfyUI

`ComfyUiWebservice` drives [ComfyUI](https://github.com/comfyanonymous/ComfyUI). The client builds a ComfyUI node graph
from the generation parameters, queues it, and downloads the finished images. ComfyUI has no authentication.

## Connect

```python
from sd_backend_client import ComfyUiWebservice

backend = ComfyUiWebservice('http://127.0.0.1:8188')
```

`connect_to_backend(url)` returns the same client when the URL answers as ComfyUI.

## Generate

ComfyUI needs a checkpoint name. `list_checkpoints()` returns the names the server accepts.

`ComfyUIDiffusionParams` adds ComfyUI's own fields to `DiffusionParams`: `clip_skip`, VAE tiling
(`vae_tiling_enabled`, `vae_tile_size`), `load_as_inpainting_model` and `sd_model_config`. A plain `DiffusionParams`
works too.

```python
from sd_backend_client import ComfyUIDiffusionParams

checkpoint = backend.list_checkpoints()[0].name
params = ComfyUIDiffusionParams(sd_model_name=checkpoint, prompt='a corgi astronaut, detailed',
                                sampler='euler', scheduler='karras', steps=20, batch_size=2, clip_skip=2)
result = backend.submit_txt2img(params).wait()
for index, image in enumerate(result.images):
    image.save(f'corgi-{index}.png')
```

LoRA and hypernetwork tags in the prompt (`<lora:name:0.8>`) work as they do on the WebUI, with the names that
`list_loras()` and `list_hypernetworks()` return.

## Progress and live previews

Generation is queued on the server, so `submit_*` returns as soon as the job is accepted. While a job runs, the client
reads step progress and preview frames from the server's websocket, and `poll()` reports them.

```python
def show_progress(progress):
    if progress.queue_index is not None:
        print(f'queued at position {progress.queue_index}')
    elif progress.progress is not None:
        print(f'{progress.progress:.0%}')
    if progress.preview is not None:
        progress.preview.save('preview.png')

result = backend.submit_txt2img(params).wait(on_progress=show_progress)
```

Pass `live_progress=False` to the constructor to skip the websocket. Handles then report only the queue state.

`handle.cancel()` removes a queued job from the server queue, or interrupts it if it is running.

## Beyond the shared interface

`ComfyUiWebservice` also has ComfyUI-only methods, such as `free_memory`, `list_model_configs` (the valid values for `sd_model_config`), `get_queue_info`, `is_node_available`, and
`txt2img` and friends that return the raw queue response. Their return types are ComfyUI's own formats, which are
internal and may change. See [Clients](../reference/clients.md#sd_backend_client.ComfyUiWebservice).
