# sd-backend-client

A headless Python client for the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and
[Forge / Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) Stable Diffusion APIs, including
their ControlNet support. It builds the request bodies and node graphs each backend expects, so one set of parameters
drives either server.

Images go in and come out as Pillow `PIL.Image` objects. The package has no GUI or Qt dependency, and every generation
parameter is passed explicitly as a [pydantic](https://docs.pydantic.dev/) model.

## Installation

```bash
pip install sd-backend-client
```

Python 3.11 or newer is required.

## Generate an image on either backend

`connect_to_backend` probes a server URL and returns a `ComfyUiWebservice` or an `A1111Webservice`. Both implement
`Backend`, so the code below works unchanged against either server.

```python
from sd_backend_client import DiffusionParams, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:8188')
checkpoint = backend.list_checkpoints()[0].name

params = DiffusionParams(sd_model_name=checkpoint, prompt='a corgi astronaut, detailed',
                         steps=20, width=512, height=512, seed=42)
handle = backend.submit_txt2img(params)   # queues the job and returns at once
result = handle.wait(timeout=300)         # blocks until the image is ready
result.images[0].save('corgi.png')
```

Every `submit_*` method returns a `GenerationHandle`. `wait()` blocks until the job finishes, while `poll()` returns
a `GenerationProgress` snapshot without blocking, for progress bars and live previews. A job runs with its parameters
as they were at submit time, so the same `DiffusionParams` can be changed and submitted again.

- **img2img:** set `init_images=[image]` and call `submit_img2img`.
- **Inpainting:** also set `mask`, where white or opaque pixels mark the region to change, and call `submit_inpaint`.
  Set `soft_inpainting=True` with a large `mask_blur` to blend the change into its surroundings.
- **Upscaling:** `submit_upscale(image, width, height)`.

## Choose options the server offers

The `list_*` methods return `BackendOption`s whose `name` is the value to pass back to the same server, and
`get_capabilities()` reports optional features such as ControlNet.

```python
samplers = [option.name for option in backend.list_samplers()]
if backend.get_capabilities().controlnet:
    controlnet_models = backend.list_controlnet_models()
```

## ControlNet

A `ControlNetUnit` pairs a control image with a model and an optional preprocessor. Add units to
`DiffusionParams.controlnet_units`, and each backend serializes them into its own form.

```python
from PIL import Image

from sd_backend_client import ControlNetUnit, PreprocessorParams

canny = next(p for p in backend.get_controlnet_preprocessors() if 'canny' in p.name.lower())
model = next(m for m in backend.list_controlnet_models() if 'canny' in m.full_model_name.lower())
params.controlnet_units = [ControlNetUnit(image=Image.open('sketch.png'), model=model,
                                          preprocessor=PreprocessorParams(typedef=canny))]
result = backend.submit_txt2img(params).wait()
```

## Errors

A failure talking to a server raises a subclass of `SDBackendError`, such as `BackendConnectionError`, `AuthError`
or `GenerationError`. Invalid input, such as an img2img job with no image, raises `ValueError` before anything is sent.

## Next steps

- The [WebUI](quickstart/webui.md) and [ComfyUI](quickstart/comfyui.md) quickstarts cover each server's own options.
- The [API reference](reference/index.md) documents every public name.
- The source, issue tracker and changelog are on [GitHub](https://github.com/centuryglass/sd-backend-client).
