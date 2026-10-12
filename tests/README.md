# Backend integration tests

Live-API integration tests for the two backend clients:

- `sd_backend_client.api.a1111_webservice.A1111Webservice` (A1111 / Forge / reForge / Forge Neo)
- `sd_backend_client.api.comfyui_webservice.ComfyUiWebservice` (ComfyUI)

They talk to a **real, running** server rather than mocking HTTP. Point the suite at
whichever backend is up — the other backend's tests skip themselves automatically, so
`pytest tests/` is safe with only one (or neither) running.

## Requirements

- A running A1111/Forge/reForge/Forge Neo WebUI (`--api` enabled) **and/or** a running ComfyUI.
- `pip install -r requirements.txt` plus `pytest`.

## Environment

| Variable         | Default                  | Purpose                                        |
|------------------|--------------------------|------------------------------------------------|
| `SD_API_URL`     | `http://127.0.0.1:7860`  | A1111 base URL (overrides host+port).          |
| `SD_API_PORT`    | `7860`                   | A1111 port only, when `SD_API_URL` is unset.   |
| `SD_UNAME`       | —                        | A1111 username, if the server uses `--api-auth`.|
| `SD_PASS`        | —                        | A1111 password, if the server uses `--api-auth`.|
| `COMFYUI_API_URL`| `http://127.0.0.1:8188`  | ComfyUI base URL (overrides host+port).        |
| `COMFYUI_API_PORT`| `8188`                  | ComfyUI port only, when `COMFYUI_API_URL` unset.|
| `SD_TEST_OUTPUT_DIR` | `tests/output/`      | Where tests save PNGs and `results.js` for `review.html`.|

(ComfyUI has no authentication, so there is no ComfyUI equivalent of `SD_UNAME`/`SD_PASS`.)

## Running

```bash
# Pure-logic unit tests only (no server, no GPU — always runnable, e.g. in CI):
pytest tests/unit/

# Fast, read-only metadata + auth tests for whichever backend(s) are up (+ unit tests):
pytest tests/

# Include the slow generation tests (real diffusion; needs a checkpoint loaded):
pytest tests/ --run-generation
# or
RUN_SD_GENERATION=1 pytest tests/
```

Generation tests run on `deliberate_v3` when the server has it, or on the next SD1.5 checkpoint
in `helpers.PREFERRED_CHECKPOINTS`, and never on an inpainting checkpoint unless the server has
nothing else. On the WebUI, the `service` fixture loads that checkpoint for the session and
restores the server's original checkpoint afterward. The diffusion upscale tests also need a
model-based upscaler (ESRGAN family, DAT, ...) and skip without one.

## Unit tests (`tests/unit/`)

Backend-independent tests of the pure "functional core": the code that turns parameters
into request bodies / node graphs, plus serialization and the image/geometry helpers. They
need **no running server** and no GPU, are deterministic, and drive the builders by setting
attributes directly.

- **`test_comfy_workflow_builder.py`** — `DiffusionWorkflowBuilder.build_workflow()` graph
  structure (checkpoint/KSampler params, prompt encoding, txt2img vs img2img latent source,
  CLIP-skip insertion, batch validation).
- **`test_diffusion_request_body.py`** — `DiffusionRequestBody.to_dict()` None-stripping and
  field passthrough.
- **`test_controlnet_serialization.py`** — `ControlNetUnit` / preprocessor serialize↔deserialize
  round-trips (both WebUI and ComfyUI key formats).
- **`test_image_and_geometry.py`** — base64/PNG round-trips, RGBA normalization, `Size`.
- **`test_param_parity.py`** - the same `DiffusionParams` serialized for both backends asks for the same
  generation: sampler and scheduler names, size, batch size, denoising strength and checkpoint.
- **`test_example_scripts.py`** - runs each script in `examples/` against a stand-in client and checks the
  requests it builds and the files it writes.
- **`test_readme_examples.py`** - runs every Python example in the top-level `README.md` and the
  docs site (`docs/`), and the scripts in `examples/`, against each recorded server (see below),
  with job submission stubbed out, and checks they import only the public API.

A regression that silently changes an emitted request fails here (fast, offline) rather than
only showing up in a live generation.

### Recorded responses (`tests/unit/fixtures/recorded/`)

`test_recorded_responses.py` replays real server responses through the clients' public
methods: ComfyUI's `/object_info`, `/history`, `/queue` and `/system_stats`, and the WebUI's
ControlNet and option-list endpoints. `recorded_replay.py` holds the replay helper it shares
with `test_readme_examples.py`. Each `<label>.json` there is written by
`scripts/capture_fixtures.py`, which runs the clients against live servers (the same
`COMFYUI_API_URL`, `SD_API_URL`, `SD_UNAME`/`SD_PASS` variables as the integration tests) and
records every response they read:

```bash
python scripts/capture_fixtures.py                          # every reachable backend
python scripts/capture_fixtures.py --only webui --label a1111
```

There is one WebUI recording per fork: `a1111`, `forge`, `reforge` and `forge_neo`. Each
recording's `meta.capabilities` holds what `get_capabilities()` reported live, and the replay
checks the client derives the same from the recorded responses. The forks differ in which
routes they serve: the Forge forks have no `/controlnet/version` or `/controlnet/settings`,
and Forge Neo also drops `/sdapi/v1/hypernetworks` and `/sdapi/v1/interrogate`.

The ComfyUI capture queues one small txt2img job, so it needs a checkpoint. Recordings list
the server's model names, so review them before committing. When a client change requests an
endpoint a recording lacks, the replay fails and names it; re-run the capture to refresh.

## What's covered (integration)

### A1111 / Forge / reForge / Forge Neo

- **`test_a1111_metadata.py`** — read-only `get_*` accessors (config, samplers,
  upscalers, models, VAE with Forge fallback, LoRAs, hypernetworks, styles, scripts,
  progress, checkpoint refresh).
- **`test_a1111_auth.py`** — `/login` flow and the `credentials_provider` callback.
  Auto-detects whether the server enforces `--api-auth` (`auth_enforced` fixture) and
  asserts the right outcome either way.
- **`test_a1111_controlnet.py`** — ControlNet extension endpoints. Gated on
  `/controlnet/model_list` (the reliable "installed" signal); the fork-specific optional
  routes `/controlnet/version` and `/controlnet/settings` skip individually if a build
  (e.g. reForge) drops them.
- **`test_a1111_generation.py`** *(opt-in)* — real round-trips: txt2img (single/batch,
  seed echo), img2img, inpaint-with-mask, basic upscale, Ultimate SD Upscale (with an
  optional tile ControlNet unit, skipped when no tile model is installed), interrogate
  (skipped where the server has no `/sdapi/v1/interrogate`), interrupt.
- **`test_a1111_controlnet_generation.py`** *(opt-in)* — ControlNet diffusion. Includes a
  **silent-failure detector**: the same seed + prompt is generated with and without a
  control unit, and the test fails unless the two images actually differ.

### ComfyUI

ComfyUI's API is asynchronous and workflow-based (jobs are queued by `prompt_id` and
polled to completion); `comfy_helpers.py` wraps that lifecycle and the cache setup the
workflow builders read from.

- **`test_comfyui_metadata.py`** — read-only accessors (system stats, model types,
  checkpoints, VAE/LoRA/ControlNet/hypernetwork/upscale models, embeddings, extensions,
  sampler & scheduler names, `is_node_available`, queue info).
- **`test_comfyui_generation.py`** *(opt-in)* — real round-trips: txt2img (seed echo),
  img2img, inpaint-with-mask, upscale, Ultimate SD Upscale and its tiled-VAE fallback
  (with an optional tile ControlNet unit, skipped when no tile model is installed),
  interrupt.
- **`test_comfyui_controlnet_generation.py`** *(opt-in)* — Canny preprocessor preview
  plus the same **silent-failure detector** A/B (with vs. without a control unit).

### Both backends

- **`test_shared_params.py`** - each server lists the names in `sampler_names`' tables (unlisted ones are
  warnings); *(opt-in)* img2img resizes a source to `width` x `height` on both backends, and WebUI applies the
  requested scheduler.

## Reviewing saved outputs

No test checks that an image matches its prompt. After a `--run-generation` run, open
[`review.html`](review.html) in a browser, straight from disk, and work through it. For each test that saves images,
it shows:

- the images, which zoom on click
- the test's result on each backend, with failure details or the skip reason
- what each image should show, as a checklist
- what the test already asserts automatically

It also lists the results of the integration tests that save no images, and says when an image or a result is
missing and why.

The page reads two things from the output dir (default `tests/output/`, git-ignored):

- **Images.** The generation tests save them there as PNGs.
- **`results.js`.** At the end of every pytest session that runs integration tests, `review_results.py` merges each
  test's outcome into this file. A later skip doesn't erase an earlier result: the page shows the last run where the
  test actually ran and notes the skip.

When `SD_TEST_OUTPUT_DIR` points elsewhere, open `review.html?output=<that directory>/`. A test that saves a new
image, or a renamed one, needs a matching entry in the `GROUPS` list in `review.html`.

## Graceful degradation

The suite is designed to stay green across environments: it **skips** (never errors)
when the server is unreachable, when auth is disabled, when ControlNet is absent, or
when the generation flag is off.
