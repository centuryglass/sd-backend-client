# sd-backend-client

[![CI](https://github.com/centuryglass/sd-backend-client/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/centuryglass/sd-backend-client/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](.github/workflows/ci.yml)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue)](LICENSE)

A headless Python client for Stable Diffusion servers. It drives **ComfyUI** and **Forge / Automatic1111 WebUI**
(with their **ControlNet** extensions) through one interface: build a [`DiffusionParams`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams), submit it, and get
`PIL.Image` results back, without caring which backend is running. The client builds the request each backend expects,
including ComfyUI's node graphs, so you never write a workflow by hand.

For applications that set up a backend for their users, it can also find ComfyUI and WebUI installs on the local
machine and run a ComfyUI install as a child process. Scripts that connect to a server you already run can ignore
those features.

The package is [IntraPaint](https://github.com/centuryglass/IntraPaint)'s backend client, published so other projects
can use it. It has no Qt or GUI dependency.

**Documentation:** <https://centuryglass.github.io/sd-backend-client/> has a quick start for each backend and the full
API reference, with every parameter's meaning, default and range.

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

You need a running server with at least one checkpoint: ComfyUI (default port 8188), or a Forge / A1111 WebUI started
with `--api` (default port 7860).

```python
from sd_backend_client import DiffusionParams, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:8188')  # or a WebUI at http://127.0.0.1:7860
checkpoint = backend.list_checkpoints()[0].name
params = DiffusionParams(sd_model_name=checkpoint, prompt='a corgi astronaut, detailed',
                         steps=20, width=512, height=512, seed=42)
result = backend.submit_txt2img(params).wait(timeout=300)
result.images[0].save('corgi.png')
```

[`connect_to_backend`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.connect_to_backend) probes the URL and returns an [`A1111Webservice`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.A1111Webservice) or a [`ComfyUiWebservice`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.ComfyUiWebservice). Both implement
[`Backend`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend), so the rest of this README works the same on either. To skip the probe, construct the client directly:
`ComfyUiWebservice('http://127.0.0.1:8188')` or `A1111Webservice('http://127.0.0.1:7860')`.

[`DiffusionParams`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams) lists every generation parameter with its default and allowed range.

Import everything from the package root. The names in `sd_backend_client.__all__` are the supported API. Modules
under `sd_backend_client.api` are internal and may change in any release.

### Authentication (WebUI)

For a WebUI started with `--api-auth`, pass a `credentials_provider` callback. The client calls it when the server
asks for credentials; it returns a `(username, password)` pair, or `None` to give up.

```python
from sd_backend_client import connect_to_backend

backend = connect_to_backend('http://127.0.0.1:7860', credentials_provider=lambda: ('user', 'password'))
```

ComfyUI has no authentication and ignores the callback.

## Examples

[`examples/`](examples/) holds runnable command-line scripts that work on either backend unless noted.
[`examples/README.md`](examples/README.md) covers their shared flags.

| Script | What it does |
|---|---|
| [`txt2img.py`](examples/txt2img.py) | Text to image, saving PNGs with A1111-style `parameters` metadata. |
| [`img2img.py`](examples/img2img.py) | Image to image, or inpainting when `--mask` is given. |
| [`inpaint.py`](examples/inpaint.py) | Inpainting. |
| [`rerun.py`](examples/rerun.py) | Regenerates an image from an earlier PNG's metadata, with overrides such as `--seed +1`. |
| [`backend_info.py`](examples/backend_info.py) | Lists a server's checkpoints, samplers, upscalers, ControlNet models and more. |
| [`upscale.py`](examples/upscale.py) | Upscales an image by a factor or to a size. |
| [`preprocess.py`](examples/preprocess.py) | Runs a ControlNet preprocessor and saves the control map. |
| [`xy_grid.py`](examples/xy_grid.py) | Sweeps one or two parameters into a labeled contact sheet. |
| [`infinite_zoom.py`](examples/infinite_zoom.py) | Repeatedly shrinks the frame and inpaints the border into a zoom-out animation. |
| [`telephone.py`](examples/telephone.py) | Interrogates an image, then runs img2img on it, for N rounds. WebUI only. |

## Generation jobs

Every `submit_*` method queues a job and returns a [`GenerationHandle`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationHandle) at once. The job uses its arguments as they
were at submit time, so you can change and reuse a params object afterwards.

| Method | Job |
|---|---|
| [`submit_txt2img(params)`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.submit_txt2img) | Text to image. |
| [`submit_img2img(params)`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.submit_img2img) | Image to image on `params.init_images[0]`. |
| [`submit_inpaint(params)`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.submit_inpaint) | Inpaint the region of `params.init_images[0]` that `params.mask` marks. |
| [`submit_upscale(image, width, height, upscale_params=None)`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.submit_upscale) | Upscale one image, with optional [`DiffusionUpscalingParams`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionUpscalingParams). |
| [`submit_preprocessor_preview(image, preprocessor, mask=None)`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.submit_preprocessor_preview) | Run a ControlNet preprocessor and return its control image. |

A handle offers:

- [`wait(timeout=None, poll_interval=0.5, on_progress=None)`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationHandle.wait) blocks until the job ends and
  returns a [`GenerationResult`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationResult). It raises [`GenerationError`](https://centuryglass.github.io/sd-backend-client/reference/errors/#sd_backend_client.GenerationError) if the job failed or was cancelled, and
  [`BackendTimeoutError`](https://centuryglass.github.io/sd-backend-client/reference/errors/#sd_backend_client.BackendTimeoutError) if `timeout` passes first.
- [`poll()`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationHandle.poll) returns a [`GenerationProgress`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationProgress) snapshot without blocking: a
  [`GenerationStatus`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationStatus), and where the backend reports them, step progress, queue position, ETA and a preview image.
- [`cancel()`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationHandle.cancel) tries to stop the job and returns whether the server accepted.
- `done` reports whether the job has ended, and `task_id` holds the server's id for it.

A [`GenerationResult`](https://centuryglass.github.io/sd-backend-client/reference/generation/#sd_backend_client.GenerationResult) holds `images`, the `seeds` each one used, any ControlNet `control_maps` the server returned,
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

Images in and out are `PIL.Image` objects. For img2img and inpainting, set
[`init_images`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.init_images) and, for inpainting, [`mask`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.mask). The source image and
mask are resized to [`width`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.width) x [`height`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.height).

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

### Soft inpainting

With [`soft_inpainting=True`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.soft_inpainting), the grey edge of a blurred mask becomes gradual,
per-pixel change, so the repainted region blends into its surroundings even at high denoising strength and with normal
checkpoints. It needs a [`mask_blur`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.mask_blur) well above the default of 4. The WebUI runs its Soft
Inpainting script and ComfyUI its Differential Diffusion node, and results come back composited onto the source on
both. Submitting with it on raises `ValueError` when the server lacks it or there is no mask.

```python
from PIL import Image, ImageDraw

from sd_backend_client import DiffusionParams, connect_to_backend

source = Image.open('photo.png')
mask = Image.new('L', source.size, 0)
ImageDraw.Draw(mask).ellipse((128, 128, 384, 384), fill=255)

backend = connect_to_backend('http://127.0.0.1:8188')
params = DiffusionParams(prompt='a bowl of fruit on a table', init_images=[source], mask=mask, mask_blur=32,
                         denoising_strength=0.9, width=source.width, height=source.height,
                         soft_inpainting=backend.get_capabilities().soft_inpainting)
backend.submit_inpaint(params).wait().images[0].save('soft_inpainted.png')
```

Each backend's algorithm takes its own settings. [`get_soft_inpainting_parameters()`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.get_soft_inpainting_parameters)
lists the server's as [`ParameterDef`](https://centuryglass.github.io/sd-backend-client/reference/discovery/#sd_backend_client.ParameterDef)s; set values by key in
[`soft_inpainting_params`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.soft_inpainting_params), and leave out any to use its default.

## ControlNet

A [`ControlNetUnit`](https://centuryglass.github.io/sd-backend-client/reference/controlnet/#sd_backend_client.ControlNetUnit) pairs a control image with a preprocessor, a ControlNet model, or both. Add units to
[`controlnet_units`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams.controlnet_units), and each backend converts them into its own request format. Preprocessor and model names
differ between backends, so look them up from the server.

The example below picks a Canny preprocessor and model:

1. [`get_controlnet_type_categories()`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.get_controlnet_type_categories) maps each control type to the names of
   its preprocessors (`module_list`) and models (`model_list`), including a `'none'` placeholder.
2. [`get_controlnet_preprocessors()`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.get_controlnet_preprocessors) returns the full
   [`ControlNetPreprocessor`](https://centuryglass.github.io/sd-backend-client/reference/controlnet/#sd_backend_client.ControlNetPreprocessor) definitions, which [`PreprocessorParams`](https://centuryglass.github.io/sd-backend-client/reference/controlnet/#sd_backend_client.PreprocessorParams) needs.

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

[`PreprocessorParams`](https://centuryglass.github.io/sd-backend-client/reference/controlnet/#sd_backend_client.PreprocessorParams) fills in the preprocessor's required parameters from their defaults. Set others by key in
`parameter_values`; [`preprocessor.parameters`](https://centuryglass.github.io/sd-backend-client/reference/controlnet/#sd_backend_client.ControlNetPreprocessor.parameters) lists each key with its type,
default and range. [`examples/preprocess.py`](examples/preprocess.py) shows a complete script.

## Discovering what a server offers

What these methods return:

- **[`BackendOption`](https://centuryglass.github.io/sd-backend-client/reference/discovery/#sd_backend_client.BackendOption)**, from every `list_*` method ([`list_checkpoints`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_checkpoints),
  [`list_samplers`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_samplers), [`list_schedulers`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_schedulers),
  [`list_upscalers`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_upscalers), [`list_vaes`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_vaes), [`list_loras`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_loras),
  [`list_hypernetworks`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_hypernetworks), [`list_controlnet_models`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.list_controlnet_models)).
  Its `name` goes straight into the matching parameter field.
- **[`BackendCapabilities`](https://centuryglass.github.io/sd-backend-client/reference/discovery/#sd_backend_client.BackendCapabilities)**, from [`get_capabilities()`](https://centuryglass.github.io/sd-backend-client/reference/clients/#sd_backend_client.Backend.get_capabilities): which optional features the
  server supports, such as ControlNet, Ultimate SD Upscale, soft inpainting and scheduler selection.
- **[`ParameterDef`](https://centuryglass.github.io/sd-backend-client/reference/discovery/#sd_backend_client.ParameterDef)**, for each setting of a ControlNet preprocessor or of soft inpainting: label, type, default and
  range.

```python
from sd_backend_client import DiffusionParams, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:8188')
checkpoints = [option.name for option in backend.list_checkpoints()]
samplers = [option.name for option in backend.list_samplers()]
capabilities = backend.get_capabilities()
print(checkpoints, samplers, capabilities.controlnet, capabilities.ultimate_upscale)

params = DiffusionParams(sd_model_name=checkpoints[0], sampler='dpmpp_2m', scheduler='karras', prompt='a fox')
```

Sampler and scheduler names are ComfyUI's (`'euler_ancestral'`, `'dpmpp_2m'`, `'karras'`) on both backends; WebUI
names such as `'Euler a'` are also accepted.

## Backend-specific parameters

[`DiffusionParams`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionParams) holds the parameters both backends support. Each backend has a subclass with extra fields, and the
other backend ignores those fields:

- [`DiffusionRequestBody`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.DiffusionRequestBody) (WebUI): batch repeats (`n_iter`), `resize_mode`, inpainting fill and blur, styles,
  high-res fix, refiner and `override_settings`.
- [`ComfyUIDiffusionParams`](https://centuryglass.github.io/sd-backend-client/reference/parameters/#sd_backend_client.ComfyUIDiffusionParams) (ComfyUI): `clip_skip`, VAE tiling, an explicit model config, and loading the checkpoint
  as an inpainting model.

```python
from sd_backend_client import DiffusionRequestBody, connect_to_backend

backend = connect_to_backend('http://127.0.0.1:7860')
params = DiffusionRequestBody(prompt='a castle on a hill', batch_size=2, n_iter=3, enable_hr=True, hr_scale=1.5)
images = backend.submit_txt2img(params).wait().images  # six images on WebUI, two on ComfyUI
```

## Errors

Every failure talking to a server raises a subclass of [`SDBackendError`](https://centuryglass.github.io/sd-backend-client/reference/errors/#sd_backend_client.SDBackendError), so you can catch them as one group:

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

## Setting up a backend for your users

The features in this section are for applications that find or start a backend themselves, such as an editor that
offers to use a ComfyUI the user already installed. If you connect to a server you run yourself, you can skip it.

### Finding backends on this machine

```python
from sd_backend_client import FindingSeverity, connect_to_backend, detect

result = detect()  # Blocks on disk reads and port probes: run it off a UI thread.
for install in result.installs:
    blockers = [finding.code.value for finding in install.findings if finding.severity == FindingSeverity.BLOCKER]
    print(install.kind.value, install.owner.value, install.path, blockers)
    if install.server is not None:
        print('  running at', install.server.url)

running = [install.server for install in result.installs if install.server is not None] + result.servers
url = next((server.url for server in running if server.kind is not None), None)
if url is not None:
    backend = connect_to_backend(url)
```

[`detect`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.detect) looks for ComfyUI, ComfyUI portable, A1111 and Forge installs in the home folder and its `Desktop`,
`Documents` and `Downloads`, in `/opt` on Linux, and at each fixed drive's root on Windows. It also reads the records
of Stability Matrix, Krita AI Diffusion and ComfyUI Desktop, and probes the default ports and any port an install is
configured for. It only reads files and never runs anything. It returns a [`DetectionResult`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.DetectionResult) of
[`DetectedInstall`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.DetectedInstall)s and [`DetectedServer`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.DetectedServer)s. Each carries [`Finding`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.Finding)s, such as a missing interpreter, no
checkpoints or a WebUI started without `--api`, as a [`FindingCode`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.FindingCode) with a details dict for your app to word.
[`SearchLocations`](https://centuryglass.github.io/sd-backend-client/reference/detection/#sd_backend_client.SearchLocations), `extra_roots`, `extra_urls` and `own_install` change where it looks.
`python scripts/detect_backends.py` prints what it finds on your machine.

### Running a local ComfyUI install

```python
from sd_backend_client import ManagedComfyUI, detect

install = next(install for install in detect().installs if install.launch_command is not None)
comfyui = ManagedComfyUI(install.launch_command, install.path, port=8189, on_log=print)
comfyui.start()
comfyui.wait_until_ready(timeout=300)  # Blocks: run it off a UI thread.
backend = comfyui.connect()
...
comfyui.stop()
```

[`ManagedComfyUI`](https://centuryglass.github.io/sd-backend-client/reference/launch/#sd_backend_client.ManagedComfyUI) starts the install with `--listen` and `--port` set, and counts it as ready only when the server on
that port reports its own arguments. `start` raises [`BackendLaunchError`](https://centuryglass.github.io/sd-backend-client/reference/errors/#sd_backend_client.BackendLaunchError) if a ComfyUI already answers on the port, so
check `detect().servers` or pick a free port first. `wait_until_ready` raises it, with the exit code and last output
lines, if the process exits before serving. `on_log` and `on_state` run on a reader thread, so a GUI app hands them
over to its own thread. `stop` only ever ends the process this instance started.

## Compatibility

Responses from these servers, recorded in October 2026, are replayed by the unit tests
(`tests/unit/fixtures/recorded/`):

- ComfyUI 0.39.0, with [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) and
  [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale).
- Automatic1111 WebUI with [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) (API version 3).
- Forge, reForge and Forge Neo, each with its built-in ControlNet.

Other versions may work but haven't been checked. The integration tests in `tests/` run the full generation paths
against a live server; [`tests/README.md`](tests/README.md) explains how to point them at yours.

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
