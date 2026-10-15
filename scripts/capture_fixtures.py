"""Record real ComfyUI and WebUI responses into tests/unit/fixtures/recorded/, for the offline contract tests.

Each run writes one `<label>.json` per reachable backend. Its `responses` map every GET endpoint the client requested
to the JSON body the server returned, `errors` map endpoints that failed to their HTTP status, and `meta` names the
backend and its version and holds the `BackendCapabilities` the client reported. The client's own `get` is wrapped
while its public methods run, so the recorded endpoints are the ones the client requests and
`tests/unit/test_recorded_responses.py` can replay them through the same methods.

Usage (servers from the integration-test env vars, see tests/README.md):
    python scripts/capture_fixtures.py                     # every reachable backend
    python scripts/capture_fixtures.py --only webui --label forge
    python scripts/capture_fixtures.py --checkpoint my_model.safetensors

The ComfyUI capture queues one small txt2img job, so ComfyUI needs a checkpoint (the first one it lists, unless
--checkpoint names one). The output names the server's models and the ComfyUI checkpoint, so review it before
committing. ComfyUI's /object_info and the WebUI's /sdapi/v1/script-info are trimmed to the entries the tests read, and
system_stats' argv is emptied, since it holds local paths.
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import requests

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui.comfyui_diffusion_params import ComfyUIDiffusionParams
from sd_backend_client.api.comfyui.comfyui_types import CONTROLNET_PREPROCESSOR_CATEGORY, NodeInfoResponse
from sd_backend_client.api.comfyui_webservice import AsyncTaskStatus, ComfyEndpoints, ComfyUiWebservice
from sd_backend_client.api.shared_data.backend import Backend
from sd_backend_client.api.webservice import WebService
from sd_backend_client.api.webui.soft_inpainting import SOFT_INPAINTING_SCRIPT_KEY
from sd_backend_client.errors import ServerError

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / 'tests' / 'unit' / 'fixtures' / 'recorded'
GENERATION_TIMEOUT = 300.0

# Non-preprocessor nodes kept in the trimmed /object_info, so the tests see preprocessor discovery ignore them.
KEPT_COMFY_NODES = ('KSampler', 'CheckpointLoaderSimple', 'CLIPTextEncode', 'ControlNetApplyAdvanced')
# Nodes that fail NodeInfoResponse validation, kept up to this many, so the tests see discovery skip them.
MAX_INVALID_NODES_KEPT = 5


def endpoint_key(endpoint: str, url_params: Optional[dict[str, str]] = None) -> str:
    """The key a recorded response is stored under: the endpoint, plus its sorted query string when it has one.

    tests/unit/recorded_replay.py replays responses under the same keys."""
    if not url_params:
        return endpoint
    return f'{endpoint}?{urlencode(sorted(url_params.items()))}'


def _record_gets(service: WebService, responses: dict[str, Any], errors: dict[str, int]) -> None:
    """Wrap service.get so every JSON response is stored in responses and every ServerError's status in errors.

    The latest response per key wins."""
    real_get = service.get

    def recording_get(endpoint: str, *args: Any, **kwargs: Any) -> requests.Response:
        url_params = kwargs.get('url_params', args[1] if len(args) > 1 else None)
        try:
            res = real_get(endpoint, *args, **kwargs)
        except ServerError as err:
            errors[endpoint_key(endpoint, url_params)] = err.status_code
            raise
        if res.ok:
            try:
                responses[endpoint_key(endpoint, url_params)] = res.json()
            except ValueError:
                pass  # not JSON: images, thumbnails
        return res

    service.get = recording_get  # type: ignore[method-assign]


def _trim_object_info(object_info: dict[str, Any]) -> dict[str, Any]:
    """Keep the ControlNet preprocessor nodes, KEPT_COMFY_NODES, and a few nodes NodeInfoResponse rejects."""
    trimmed: dict[str, Any] = {}
    invalid_kept = 0
    for name, info in object_info.items():
        category = info.get('category', '') if isinstance(info, dict) else ''
        if name in KEPT_COMFY_NODES or (isinstance(category, str) and CONTROLNET_PREPROCESSOR_CATEGORY in category):
            trimmed[name] = info
            continue
        if invalid_kept < MAX_INVALID_NODES_KEPT:
            try:
                NodeInfoResponse.model_validate(info)
            except ValueError:
                trimmed[name] = info
                invalid_kept += 1
    return trimmed


def _wait_for_finish(service: ComfyUiWebservice, prompt_id: str, number: int) -> None:
    """Poll until the job finishes, fails or the deadline passes."""
    deadline = time.monotonic() + GENERATION_TIMEOUT
    while time.monotonic() < deadline:
        progress = service.check_queue_entry(prompt_id, number)
        status = progress.status
        if status == AsyncTaskStatus.FINISHED:
            return
        if status == AsyncTaskStatus.FAILED:
            sys.exit(f'ComfyUI job {prompt_id} failed: {progress.error}')
        time.sleep(1.0)
    sys.exit(f'ComfyUI job {prompt_id} did not finish within {GENERATION_TIMEOUT}s')


def _run_discovery(service: Backend) -> dict[str, bool]:
    """Call every `Backend` discovery method, so the recording holds the endpoints each one reads, and return the
    capabilities the server reported."""
    for list_options in (service.list_checkpoints, service.list_vaes, service.list_loras, service.list_hypernetworks,
                         service.list_samplers, service.list_schedulers, service.list_upscalers,
                         service.list_controlnet_models, service.get_soft_inpainting_parameters):
        list_options()
    return service.get_capabilities().model_dump()


def capture_comfyui(url: str, checkpoint: Optional[str]) -> dict[str, Any]:
    """Run the ComfyUI client's read methods and one small txt2img job, recording every response."""
    service = ComfyUiWebservice(url)
    responses: dict[str, Any] = {}
    errors: dict[str, int] = {}
    _record_gets(service, responses, errors)

    stats = service.get_system_stats()
    service.get_sampler_names()
    service.get_scheduler_names()
    service.get_model_types()
    preprocessors = service.get_controlnet_preprocessors()
    service.get_controlnet_type_categories(preprocessors)
    capabilities = _run_discovery(service)
    checkpoints = service.get_sd_checkpoints()
    if checkpoint is None:
        if not checkpoints:
            sys.exit('ComfyUI lists no checkpoints; the history capture needs one to run a txt2img job')
        checkpoint = checkpoints[0]

    params = ComfyUIDiffusionParams(sd_model_name=checkpoint, prompt='a red apple on a wooden table', steps=4,
                                    width=256, height=256, seed=1)
    queued = service.txt2img(params)
    if queued.prompt_id is None or queued.number is None:
        sys.exit(f'ComfyUI rejected the capture workflow: {queued.error or queued.node_errors}')
    _wait_for_finish(service, queued.prompt_id, queued.number)
    service.get_queue_info()

    responses[ComfyEndpoints.OBJECT_INFO] = _trim_object_info(responses[ComfyEndpoints.OBJECT_INFO])
    responses[ComfyEndpoints.SYSTEM_STATS]['system']['argv'] = []
    return {'meta': {'backend': 'comfyui', 'version': stats.system.comfyui_version,
                     'prompt_id': queued.prompt_id, 'number': queued.number, 'capabilities': capabilities},
            'responses': responses, 'errors': errors}


def capture_webui(url: str, credentials: Optional[tuple[str, str]]) -> dict[str, Any]:
    """Run the WebUI client's ControlNet, option-list and discovery read methods, recording every response."""
    service = A1111Webservice(url, credentials_provider=lambda: credentials)
    responses: dict[str, Any] = {}
    errors: dict[str, int] = {}
    _record_gets(service, responses, errors)

    service.get_samplers()
    service.get_latent_upscale_modes()
    service.get_scripts()
    if not service.get_capabilities().controlnet:
        sys.exit(f'{url} has no ControlNet API; the WebUI capture needs one')
    # Forge's built-in ControlNet has no version endpoint; the 404 lands in `errors`.
    version: Optional[int]
    try:
        version = service.get_controlnet_version()
    except ServerError as err:
        if err.status_code != 404:
            raise
        version = None
    service.get_controlnet_preprocessors()
    service.get_controlnet_type_categories()
    capabilities = _run_discovery(service)
    if A1111Webservice.Endpoints.SCRIPT_INFO in responses:
        responses[A1111Webservice.Endpoints.SCRIPT_INFO] = [
            script for script in responses[A1111Webservice.Endpoints.SCRIPT_INFO]
            if script['name'].lower() == SOFT_INPAINTING_SCRIPT_KEY]
    return {'meta': {'backend': 'webui', 'controlnet_version': version, 'capabilities': capabilities},
            'responses': responses, 'errors': errors}


def _reachable(url: str) -> bool:
    try:
        requests.get(url, timeout=5)
    except requests.RequestException:
        return False
    return True


def _write(out_dir: Path, label: str, capture: dict[str, Any]) -> None:
    capture['meta']['captured'] = datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{label}.json'
    path.write_text(json.dumps(capture, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Wrote {path} ({path.stat().st_size // 1024} KiB, {len(capture["responses"])} endpoints)')


def main() -> None:
    """Parse arguments and capture each requested, reachable backend."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--only', choices=('comfyui', 'webui'), help='capture one backend')
    parser.add_argument('--label', help='output file name without .json (default: the backend name); '
                                        'name a variant such as "forge" or "a1111"')
    parser.add_argument('--checkpoint', help='ComfyUI checkpoint for the history capture (default: the first listed)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT_DIR,
                        help=f'output directory (default: {DEFAULT_OUT_DIR})')
    args = parser.parse_args()
    if args.label is not None and args.only is None:
        parser.error('--label names one capture, so it needs --only')

    comfy_url = os.environ.get('COMFYUI_API_URL', 'http://127.0.0.1:8188')
    webui_url = os.environ.get('SD_API_URL', 'http://127.0.0.1:7860')
    username, password = os.environ.get('SD_UNAME'), os.environ.get('SD_PASS')
    credentials = (username, password) if username and password else None

    captures: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
        ('comfyui', comfy_url, lambda: capture_comfyui(comfy_url, args.checkpoint)),
        ('webui', webui_url, lambda: capture_webui(webui_url, credentials)),
    ]
    captured_any = False
    for backend, url, capture in captures:
        if args.only not in (None, backend):
            continue
        if not _reachable(url):
            print(f'Skipping {backend}: nothing answered at {url}')
            continue
        _write(args.out, args.label or backend, capture())
        captured_any = True
    if not captured_any:
        sys.exit('No backend was reachable; set COMFYUI_API_URL or SD_API_URL (tests/README.md)')


if __name__ == '__main__':
    main()
