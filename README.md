# sd-backend-client

[![CI](https://github.com/centuryglass/sd-backend-client/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/centuryglass/sd-backend-client/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](.github/workflows/ci.yml)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue)](LICENSE)

A headless Python client for Stable Diffusion servers. It drives **ComfyUI** and **Forge / Automatic1111 WebUI**
(with their **ControlNet** extensions) through one interface: build a `DiffusionParams`, submit it, and get
`PIL.Image` results back, without caring which backend is running.

The package was extracted from [IntraPaint](https://github.com/centuryglass/IntraPaint)'s `src/api` so other projects
can use it without the rest of the editor. It has no Qt or GUI dependency.

## Status

Under active refactoring, and the API may still change between minor versions. txt2img, img2img, inpainting,
upscaling (including tiled Ultimate SD Upscale) and ControlNet work end-to-end on both backends.

## Installation

```bash
pip install sd-backend-client
```

To install the latest unreleased code instead:

```bash
pip install git+https://github.com/centuryglass/sd-backend-client
```

Requires Python 3.11 or newer. Runtime dependencies: `pillow`, `requests`, `websocket-client`, `pydantic` and
`typing-extensions`.

## Quick start

```python
from sd_backend_client import DiffusionParams, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:7860')  # a ComfyUI or WebUI server
params = DiffusionParams(prompt='a corgi astronaut, detailed', steps=20, width=512, height=512, seed=42)
result = backend.submit_txt2img(params).wait(timeout=300)
result.images[0].save('corgi.png')
```

`connect_to_backend` probes the URL and returns an `A1111Webservice` or a `ComfyUiWebservice`. Both implement
`Backend`, so the rest of this README works the same on either. To skip the probe, construct the client directly:
`ComfyUiWebservice('http://127.0.0.1:8188')` or `A1111Webservice('http://127.0.0.1:7860')`.

Import everything from the package root. The names in `sd_backend_client.__all__` are the supported API. Modules
under `sd_backend_client.api` are internal and may change in any release.

## Generation jobs

Every `submit_*` method queues a job and returns a `GenerationHandle` at once. The job uses its arguments as they were
at submit time, so you can change and reuse a params object afterwards.

| Method | Job |
|---|---|
| `submit_txt2img(params)` | Text to image. |
| `submit_img2img(params)` | Image to image on `params.init_images[0]`. |
| `submit_inpaint(params)` | Inpaint the region of `params.init_images[0]` that `params.mask` marks. |
| `submit_upscale(image, width, height, upscale_params=None)` | Upscale one image. |
| `submit_preprocessor_preview(image, preprocessor, mask=None)` | Run a ControlNet preprocessor and return its control image. |

On ComfyUI, `submit_upscale`'s Ultimate SD Upscale path scales both sides by the larger of the requested
width/height ratios, so a target aspect ratio different from the source's isn't honored.

A handle offers:

- `wait(timeout=None, poll_interval=0.5, on_progress=None)` blocks until the job ends and returns a `GenerationResult`.
  It raises `GenerationError` if the job failed or was cancelled, and `BackendTimeoutError` if `timeout` passes first.
- `poll()` returns a `GenerationProgress` snapshot without blocking: `status` (a `GenerationStatus`), and where the
  backend reports them, `progress` (0 to 1), `overall_progress`, `queue_index`, `eta_seconds`, a `preview` image and
  `text_info`. On ComfyUI, `progress` covers one sampler run, so a tiled upscale's restarts for every tile, while
  `overall_progress` covers the whole job.
- `cancel()` tries to stop the job and returns whether the server accepted.
- `done` reports whether the job has ended, and `task_id` holds the server's id for it.

A `GenerationResult` holds `images`, the `seeds` each one used, any ControlNet `control_maps` the server returned,
the base `seed`, and the backend's own metadata in `raw_info`.

Poll a handle to drive a progress bar:

```python
from sd_backend_client import DiffusionParams, GenerationProgress, connect_to_backend

def show_progress(progress: GenerationProgress) -> None:
    if progress.progress is not None:
        print(f'{progress.status.value}: {progress.progress:.0%}')

backend = connect_to_backend('http://127.0.0.1:8188')
handle = backend.submit_txt2img(DiffusionParams(prompt='a lighthouse at dusk', batch_size=2))
result = handle.wait(on_progress=show_progress)
for image, seed in zip(result.images, result.seeds):
    image.save(f'lighthouse_{seed}.png')
```

**How each backend queues jobs.** ComfyUI queues jobs on the server: `cancel()` removes a queued job or interrupts the
running one. The WebUI API has no queue, so `A1111Webservice` keeps one on the client. It sends one job at a time,
once the server has finished whatever it is running. Until a job is sent, `cancel()` drops it without contacting the
server. After that, `cancel()` interrupts the server's current job. This queue assumes your client is the server's
only user: it does not coordinate with other clients, or with other `A1111Webservice` objects for the same server.

## Images and inpainting

Images in and out are `PIL.Image` objects. For img2img and inpainting, set `init_images` and, for inpainting,
`mask`. The source image and mask are resized to `width` x `height`.

The mask means the same on both backends: **white or opaque pixels mark the region to change.** Either a grayscale
mask or an alpha mask works.

```python
from PIL import Image, ImageDraw

from sd_backend_client import DiffusionParams, connect_to_backend

source = Image.open('photo.png')
mask = Image.new('L', source.size, 0)                           # black: keep
ImageDraw.Draw(mask).rectangle((128, 128, 384, 384), fill=255)  # white: repaint

backend = connect_to_backend('http://127.0.0.1:7860')
params = DiffusionParams(prompt='a bowl of fruit on a table', init_images=[source], mask=mask,
                         denoising_strength=0.8, width=source.width, height=source.height)
backend.submit_inpaint(params).wait().images[0].save('inpainted.png')
```

## ControlNet

A `ControlNetUnit` pairs a control image with a preprocessor, a ControlNet model, or both. Add units to
`params.controlnet_units`, and each backend converts them into its own request format. Preprocessor and model names
differ between backends, so look them up from the server. `get_controlnet_type_categories()` groups them by control
type:

```python
from PIL import Image

from sd_backend_client import ControlNetModel, ControlNetUnit, DiffusionParams, PreprocessorParams, \
    connect_to_backend

backend = connect_to_backend('http://127.0.0.1:7860')
canny = backend.get_controlnet_type_categories()['Canny']
preprocessor_name = next(name for name in canny['module_list'] if name.lower() != 'none')
model_name = next(name for name in canny['model_list'] if name.lower() != 'none')
preprocessor = next(p for p in backend.get_controlnet_preprocessors() if p.name == preprocessor_name)

unit = ControlNetUnit(image=Image.open('reference.png'),
                      preprocessor=PreprocessorParams(typedef=preprocessor, parameter_values={}),
                      model=ControlNetModel(model_name),
                      control_strength=0.8)
params = DiffusionParams(prompt='a robot with the same outline', controlnet_units=[unit])
backend.submit_txt2img(params).wait().images[0].save('controlled.png')
```

`PreprocessorParams` fills in the preprocessor's required parameters from their defaults. Set others by key in
`parameter_values`; `preprocessor.parameters` lists each key with its type, default and range.
[`examples/preprocess.py`](examples/preprocess.py) shows a complete script.

## Discovering what a server offers

```python
from sd_backend_client import DiffusionParams, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:8188')
checkpoints = [option.name for option in backend.list_checkpoints()]
samplers = [option.name for option in backend.list_samplers()]
capabilities = backend.get_capabilities()
print(checkpoints, samplers, capabilities.controlnet, capabilities.ultimate_upscale)

params = DiffusionParams(sd_model_name=checkpoints[0], sampler='dpmpp_2m', scheduler='karras', prompt='a fox')
```

The `list_*` methods return `BackendOption`s whose `name` goes straight into the matching parameter field. Sampler
and scheduler names are ComfyUI's (`'euler_ancestral'`, `'dpmpp_2m'`, `'karras'`) on both backends; WebUI names
such as `'Euler a'` are also accepted. `get_capabilities()` reports optional features such as ControlNet, Ultimate SD
Upscale and scheduler support.

## Backend-specific parameters

`DiffusionParams` holds the parameters both backends support. Each backend has a subclass with extra fields, and the
other backend ignores those fields:

- `DiffusionRequestBody` (WebUI): batch repeats (`n_iter`), `resize_mode`, inpainting fill and blur, styles,
  high-res fix, refiner and `override_settings`.
- `ComfyUIDiffusionParams` (ComfyUI): `clip_skip`, VAE tiling, an explicit model config, and loading the checkpoint
  as an inpainting model.

```python
from sd_backend_client import DiffusionRequestBody, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:7860')
params = DiffusionRequestBody(prompt='a castle on a hill', batch_size=2, n_iter=3, enable_hr=True, hr_scale=1.5)
images = backend.submit_txt2img(params).wait().images  # six images on WebUI, two on ComfyUI
```

## Authentication (WebUI)

For a WebUI started with `--api-auth`, pass a `credentials_provider` callback. The client calls it when the server
asks for credentials; it returns a `(username, password)` pair, or `None` to give up.

```python
from sd_backend_client import connect_to_backend

backend = connect_to_backend('http://127.0.0.1:7860', credentials_provider=lambda: ('user', 'password'))
```

ComfyUI has no authentication and ignores the callback.

## Errors

Every failure talking to a server raises a subclass of `SDBackendError`, so you can catch them as one group:

| Exception | Raised when |
|---|---|
| `BackendConnectionError` | The server can't be reached. Also a `ConnectionError`. |
| `BackendTimeoutError` | A request or `wait()` ran out of time. Also a `TimeoutError`. |
| `AuthError` | The WebUI rejected authentication. |
| `ServerError` | The server answered with an error status; `status_code` and `body` hold its reply. |
| `UnexpectedResponseError` | The response was not in the expected format, usually an unsupported server version. |
| `WorkflowValidationError` | ComfyUI rejected the generated workflow; `node_errors` holds per-node detail. |
| `GenerationError` | A job ended failed, cancelled or unknown to the server; `status` holds its final state. |

Invalid input, such as inpainting without a mask, raises `ValueError` before anything is sent.

```python
from sd_backend_client import DiffusionParams, GenerationError, SDBackendError, connect_to_backend

try:
    backend = connect_to_backend('http://127.0.0.1:8188')
    result = backend.submit_txt2img(DiffusionParams(prompt='a teapot')).wait(timeout=120)
except GenerationError as err:
    print(f'Generation ended with status {err.status.value}: {err}')
except SDBackendError as err:
    print(f'Backend error: {err}')
```

## Compatibility

Responses from these servers, recorded in October 2026, are replayed by the unit tests
(`tests/unit/fixtures/recorded/`):

- ComfyUI 0.39.0, with [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) and
  [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale).
- Automatic1111 WebUI with [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) (API version 3).
- Forge, reForge and Forge Neo, each with its built-in ControlNet.

Other versions may work but haven't been checked. The integration tests in `tests/` run the full generation paths
against a live server; [`tests/README.md`](tests/README.md) explains how to point them at yours.

## Differences from IntraPaint's client

- **No Qt.** Images are `PIL.Image` (decoded images are normalized to RGBA) and sizes use a small `Size` class.
  There are no UI strings, and no `cv2` or `numpy`.
- **No config layer.** IntraPaint read generation parameters from its `Cache` / `AppConfig` settings. Here they are
  passed explicitly as pydantic models, which validate them.
- **Authentication** uses the `credentials_provider` callback in place of IntraPaint's login dialog.

## Development

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, checks and pull requests. CI (`.github/workflows/ci.yml`) runs
`pytest tests/` on Python 3.11-3.14 and with the oldest allowed dependencies, plus pylint, mypy and a packaging check,
on every pull request into `main`. No backend runs in CI, so the integration tests skip themselves there and only the
offline unit tests run. Releases are cut by [release-please](https://github.com/googleapis/release-please) from
Conventional Commits pull request titles; see [`CHANGELOG.md`](CHANGELOG.md). Rules for coding agents are in
[`AGENTS.md`](AGENTS.md).

## License

Public domain, dedicated under [The Unlicense](https://unlicense.org); see [`LICENSE`](LICENSE). Use it for anything,
with no attribution or other obligations.
