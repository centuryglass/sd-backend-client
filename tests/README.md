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
| `SD_TEST_OUTPUT_DIR` | `tests/output/`      | Where generation tests save PNGs for inspection.|

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

The generation tests save their images to the output dir (default `tests/output/`, git-ignored). No test checks
that an image matches its prompt, so after a `--run-generation` run, review the images against this checklist.

- Each test overwrites its own files. A file older than the rest came from a skipped test or from a test that no
  longer exists.
- Files with the `comfy_` prefix come from ComfyUI. All others come from WebUI.
- Generated images are small and use few steps, so soft, low-detail output is expected. Judge whether each image
  follows its prompt and inputs, not its quality.
- Inputs (`*_source`, `*_mask`, `*_input`) are drawn by `helpers.py` and are the same on every run. A change in one
  points to the test helpers, not the backend.

Every generated image:

- [ ] Is not solid black, solid gray or pure noise. Any of these points to a VAE, sampler or decode failure.
- [ ] Has no transparent or checkerboard areas.
- [ ] Has natural colors, not inverted or channel-swapped (for example, a red apple that comes out blue).

### WebUI

From `test_a1111_generation.py`, `test_a1111_async.py` and `test_shared_params.py`. Except where noted, images are
256x256 at 4 steps.

- [ ] `txt2img.png`: a red apple on a wooden table.
- [ ] `txt2img_batch_0.png`, `txt2img_batch_1.png`: two apple-on-table images that differ from each other. Two
  identical images mean the batch reused one seed.
- [ ] `img2img_result.png`: a landscape painting, clearly changed from `img2img_source.png` (a diagonal color
  gradient with a white square outline and a black circle outline). Traces of the source's colors or framing may
  remain at 0.75 denoise.
- [ ] `inpaint_result.png`: a flower fills the center square, which `inpaint_mask.png` marks white.
- [ ] Outside the center square of `inpaint_result.png`, the gradient and white square outline match
  `inpaint_source.png` pixel for pixel, with a crisp boundary (`mask_blur=0`). A changed border around an untouched
  center means the mask was inverted.
- [ ] `webui_full_res_whole.png` and `webui_full_res_full_res.png` (512x512, 20 steps, 0.85 denoise): both repaint
  the 96x96 patch near the top right of `webui_full_res_source.png` as clockwork gears and leave the rest of the
  gradient untouched. The `full_res` patch is visibly finer, with sharper gears and screws.
  `webui_full_res_far_point.png` is coarser than `full_res` again, because the far point at the bottom left widened
  the crop. Seeds vary in how much detail they draw, so judge the rendering scale rather than the amount of content.
- [ ] `upscale_result.png`: `upscale_source.png` at 256x256, with the same composition and no added content.
- [ ] `upscale_sd_result.png`: `upscale_sd_source.png` at 2048x2048, refined by a tiled Ultimate SD Upscale pass
  (visibly sharper than a plain resize, same composition). Missing when the server has no Ultimate SD Upscale
  script.
- [ ] `async_upscale_sd.png` (2048x2048): the async-handle counterpart of `upscale_sd_result.png`.
- [ ] `async_txt2img.png` (512x512, 20 steps, as are the other `async_*` images): a red apple on a wooden table.
- [ ] `async_img2img.png`: an oil-painting look in blue-dominant colors, from a flat blue source.
- [ ] `async_serialize_1.png` is a blue teapot and `async_serialize_2.png` is a green frog. Swapped or identical
  subjects mean the two queued jobs' results were mixed up.
- [ ] `async_idle_wait.png`: a small red cube. A fantasy landscape here is the other client's job, not ours.
- [ ] `shared_params_webui_img2img.png`: a 256x320 portrait watercolor from a 192x128 landscape source, filling the
  frame without stretching or letterbox bars.

### ComfyUI

From `test_comfyui_generation.py`, `test_comfyui_async.py` and `test_shared_params.py`. Except where noted, images
are 256x256 at 8 steps.

- [ ] `comfy_txt2img.png`: a red apple on a wooden table.
- [ ] `comfy_img2img_result.png`: a vivid landscape painting, clearly changed from `comfy_img2img_source.png`.
- [ ] `comfy_inpaint_result.png`: flower-like content replaces the center square that `comfy_inpaint_mask.png`
  marks white.
- [ ] Outside the center square of `comfy_inpaint_result.png`, the gradient and white square outline survive. Slight
  softening is normal here, because this test calls raw `inpaint()` and ComfyUI passes the whole image through the
  VAE. An edited border around an untouched center means `ComfyUiWebservice.upload_mask` converted the mask polarity
  wrong.
- [ ] `comfy_submit_inpaint_result.png`: a flower in the center square of `comfy_submit_inpaint_source.png`. Unlike
  the raw result above, everything outside the square matches the source pixel for pixel, with a crisp boundary
  (`mask_blur=0`), because `submit_inpaint` composites the result onto the source.
- [ ] `comfy_mask_blur_0.png` and `comfy_mask_blur_12.png`: the same flower. The first has a hard edge around the
  square. In the second the flower fades into the gradient over roughly 30 px beyond the original edge.
- [ ] `comfyui_full_res_whole.png`, `comfyui_full_res_full_res.png` and `comfyui_full_res_far_point.png`: the same
  three checks as the WebUI images above. Outside the patch the pixels match the source, because the client
  composites the result through the mask. The generated content reaches the patch edge; a blurry, half-generated band
  just inside the edge means the sampler got a soft noise mask instead of the rounded one.
- [ ] `comfy_upscale_result.png`: `comfy_upscale_source.png` at 256x256, sharper than a plain resize, with no new
  content. It is missing when the server has no upscale model installed.
- [ ] `comfy_upscale_sd_result.png`: `comfy_upscale_sd_source.png` at 2048x2048, refined by Ultimate SD Upscale
  (visibly sharper, same composition). Missing when the server has no Ultimate SD Upscale node.
- [ ] `comfy_upscale_fallback_result.png`: the same 2048x2048 upscale via the tiled VAE encode/decode fallback
  (`use_ultimate_upscale_script=False`), run regardless of whether the node is installed.
- [ ] `comfy_async_upscale_sd.png` (2048x2048): the async-handle counterpart of `comfy_upscale_sd_result.png`.
- [ ] `comfy_async_txt2img.png`: identical to `comfy_txt2img.png`, since both use the same prompt, seed, size and
  steps. A visible difference means the async path builds a different workflow.
- [ ] `comfy_async_img2img.png`: an oil-painting look in blue-dominant colors, from a flat blue source.
- [ ] `comfy_async_queue_1.png` is a blue teapot and `comfy_async_queue_2.png` is a green frog. Swapped or identical
  subjects mean a handle downloaded another prompt's outputs.
- [ ] `comfy_async_running_survived_cancel.png`: a fully rendered 512x512 fantasy landscape. A half-finished, noisy
  or missing image means cancelling the queued job interrupted the running one. A red cube is the cancelled job.
- [ ] `comfy_async_targeted_interrupt.png`: a fully rendered 512x512 fantasy landscape. A half-finished or noisy
  image means the server ignored the prompt id in `/interrupt` and stopped the wrong job.
- [ ] `shared_params_comfyui_img2img.png`: a 256x320 portrait watercolor filling the frame without stretching.
  Expect a similar style to the WebUI version, not the same image.

### ControlNet

ControlNet can fail silently: the server returns a normal image and ignores the control unit. Each backend's suite
generates "a photograph of a city street, detailed" with seed 42, once without ControlNet and once with Canny.

- [ ] `controlnet_input.png` and `comfy_controlnet_input.png`: identical black lines on white, a square outline, a
  centered circle and both corner-to-corner diagonals.
- [ ] `controlnet_preview_canny.png` and `comfy_controlnet_preview_CannyEdgePreprocessor.png`: thin white edge lines
  on black tracing the same shapes, with each thick input line doubled. They are 512x512, the preprocessors'
  default resolution, though the input is 256x256.
- [ ] `controlnet_baseline_no_cn.png` and `comfy_controlnet_baseline_no_cn.png`: a loose street scene with no
  visible square or circle structure.
- [ ] `controlnet_with_canny.png` and `comfy_controlnet_with_CannyEdgePreprocessor.png`: the composition follows
  the edge map, with the square frame, central circle and diagonals plainly visible, typically as a corridor or
  room seen head-on. The test only asserts a mean difference of 2/255 from the baseline, which a weak or partly
  applied unit can pass.

### What the tests already assert

| Image group | Asserted automatically | Left to the reviewer |
|-------------|------------------------|----------------------|
| All generated images | RGBA, size within 8 px of target | Prompt adherence; black, noise or color-swap output |
| img2img | Mean difference from source >= 10/255 | Style matches the prompt |
| `inpaint_full_res` | Pixels outside the patch match the source; the patch changed; full-res and far-crop-point results differ from the results they are compared with | The full-res patch is finer and the far-point patch coarser; the seam is clean |
| Inpainting | Center changes >= 20/255. WebUI border <= 5/255; ComfyUI center change >= 3x border change. ComfyUI `submit_inpaint` border <= 1/255; `mask_blur=12` blends pixels just outside the mask, `mask_blur=0` does not | Seam quality; the fill is a flower |
| Upscale | At least 200 px on each side | No added content or artifacts |
| ControlNet pair | Mean difference >= 2/255 | The edge structure shapes the image |
| Preprocessor preview | Not a single flat color | Edges match the input shapes |
| Async queue and cancel | Status sequence; both images returned | Each image matches its own prompt |
| Shared params img2img | 256x320 size; WebUI reports DPM++ 2M with Karras | No stretching or distortion |

## Graceful degradation

The suite is designed to stay green across environments: it **skips** (never errors)
when the server is unreachable, when auth is disabled, when ControlNet is absent, or
when the generation flag is off.
