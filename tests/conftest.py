"""Shared fixtures for the A1111 / Forge / reForge / Forge Neo live-API integration tests.

These tests talk to a **real** WebUI instance. They assume:

* The server is running and reachable (default ``http://127.0.0.1:7860``; override
  with the ``SD_API_URL`` environment variable, or ``SD_API_PORT`` to change only
  the port).
* If the server has ``--api-auth`` enabled, credentials are supplied through the
  ``SD_UNAME`` / ``SD_PASS`` environment variables.

Running
-------
Fast, read-only metadata tests run by default::

    pytest tests/

The generation tests (txt2img / img2img / upscale / interrogate) actually invoke
diffusion and need a checkpoint loaded. They are gated behind an opt-in flag::

    pytest tests/ --run-generation
    # or
    RUN_SD_GENERATION=1 pytest tests/
"""
import os

import pytest
from PIL import Image
from pydantic import ValidationError

from sd_backend_client.api.a1111_webservice import A1111Webservice
from sd_backend_client.api.comfyui_webservice import ComfyUiWebservice
from sd_backend_client.errors import AuthError, SDBackendError

from .comfy_helpers import build_comfy_params
from .helpers import UPSCALE_SOURCE_PROMPT, UPSCALE_SOURCE_SIZE, fast_request_body

DEFAULT_PORT = 7860
COMFY_DEFAULT_PORT = 8188


def pytest_addoption(parser):
    parser.addoption(
        '--run-generation',
        action='store_true',
        default=False,
        help='Run the slow generation tests that actually invoke diffusion (needs a loaded checkpoint).',
    )


def pytest_configure(config):
    config.addinivalue_line('markers', 'integration: talks to a live WebUI instance.')
    config.addinivalue_line('markers', 'generation: invokes real diffusion; slow, needs a loaded checkpoint.')
    config.addinivalue_line('markers', 'controlnet: requires the ControlNet extension to be installed.')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-generation') or os.environ.get('RUN_SD_GENERATION'):
        return
    skip_gen = pytest.mark.skip(reason='generation tests are opt-in: use --run-generation or RUN_SD_GENERATION=1')
    for item in items:
        if 'generation' in item.keywords:
            item.add_marker(skip_gen)


@pytest.fixture(scope='session')
def api_url() -> str:
    """Base URL of the WebUI server under test."""
    if os.environ.get('SD_API_URL'):
        return os.environ['SD_API_URL'].rstrip('/')
    port = os.environ.get('SD_API_PORT', str(DEFAULT_PORT))
    return f'http://127.0.0.1:{port}'


@pytest.fixture(scope='session')
def credentials():
    """(username, password) from SD_UNAME / SD_PASS, or None if unset."""
    uname = os.environ.get('SD_UNAME')
    password = os.environ.get('SD_PASS')
    if uname is None or password is None:
        return None
    return uname, password


# --------------------------------------------------------------------------- #
# ComfyUI backend
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='session')
def comfy_url() -> str:
    """Base URL of the ComfyUI server under test."""
    if os.environ.get('COMFYUI_API_URL'):
        return os.environ['COMFYUI_API_URL'].rstrip('/')
    port = os.environ.get('COMFYUI_API_PORT', str(COMFY_DEFAULT_PORT))
    return f'http://127.0.0.1:{port}'


@pytest.fixture(scope='session')
def comfy_service(comfy_url) -> ComfyUiWebservice:
    """A live ComfyUiWebservice, skipping the session if the server is unreachable.

    ComfyUI has no authentication, so (unlike the A1111 client) there are no credentials
    or login flow to wire up.
    """
    client = ComfyUiWebservice(comfy_url)
    try:
        client.get_system_stats()
    except SDBackendError as err:
        pytest.skip(f'No ComfyUI server reachable at {comfy_url}: {err}')
    yield client
    client.disconnect()


@pytest.fixture(scope='session')
def comfy_checkpoint(comfy_service) -> str:
    """Name of a checkpoint to generate with, preferring a small SD1.5 model.

    Skips dependent tests if the server has no checkpoints installed.
    """
    checkpoints = comfy_service.get_sd_checkpoints()
    if not checkpoints:
        pytest.skip('No checkpoints installed on the ComfyUI server.')
    # Prefer a non-inpainting SD1.5 model when we can identify one; fall back to the first.
    sd15_hints = ('deliberate_v3.safetensors', 'cyberrealistic', 'dreamshaper', 'v1-5', 'sd15', 'sd-v1')
    for hint in sd15_hints:
        match = next((c for c in checkpoints if hint in c.lower() and 'inpaint' not in c.lower()), None)
        if match:
            return match
    return checkpoints[0]


@pytest.fixture(scope='session')
def controlnet_available(service):
    """Skip dependent tests unless the ControlNet extension is installed and responding.

    Gated on ``/controlnet/model_list`` (the reliable "extension present" signal) rather
    than ``/controlnet/version``, which some forks (e.g. ReForge) drop.
    """
    try:
        models = service.get_controlnet_models()
    except (SDBackendError, ValidationError) as err:
        pytest.skip(f'ControlNet extension not installed or returned an unexpected response: {err}')
    if not models.model_list:
        pytest.skip('ControlNet is installed but /controlnet/model_list lists no models')
    return service


@pytest.fixture(scope='session')
def controlnet_pairing(controlnet_available):
    """A valid (module, model) ControlNet pairing discovered from /controlnet/control_types.

    Prefers Canny (edge control works well with the synthetic high-contrast test image),
    then falls back to any category exposing a non-"None" default module and model.
    """
    control_types = controlnet_available.get_controlnet_control_types()['control_types']

    def usable(type_entry):
        module = type_entry.get('default_option')
        model = type_entry.get('default_model')
        return module and model and module.lower() != 'none' and model.lower() != 'none'

    if 'Canny' in control_types and usable(control_types['Canny']):
        entry = control_types['Canny']
    else:
        entry = next((e for e in control_types.values() if usable(e)), None)
    if entry is None:
        pytest.skip('No ControlNet category exposes a usable default module + model pairing.')
    return entry['default_option'], entry['default_model']


@pytest.fixture(scope='session')
def output_dir() -> str:
    """Directory where generation tests save PNGs for visual fidelity inspection.

    Defaults to ``tests/output/`` next to this file; override with ``SD_TEST_OUTPUT_DIR``.
    """
    path = os.environ.get('SD_TEST_OUTPUT_DIR') or os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope='session')
def auth_enforced(api_url) -> bool:
    """Whether the server actually enforces API auth (was launched with ``--api-auth``).

    Detected by hitting a protected endpoint with no credentials and checking for a
    401. A WebUI started without ``--api-auth`` still serves a ``/login`` page that
    returns 200 for any input, so this probe (not the login response) is the reliable
    signal. Skips the session if the server is unreachable.
    """
    probe = A1111Webservice(api_url, credentials_provider=None)
    try:
        probe.get('/sdapi/v1/samplers', fail_on_auth_error=True)
        return False
    except AuthError:
        return True
    except SDBackendError as err:
        pytest.skip(f'No WebUI server reachable at {api_url}: {err}')
    finally:
        probe.disconnect()


@pytest.fixture(scope='session')
def service(api_url, credentials) -> A1111Webservice:
    """A live, authenticated A1111Webservice.

    Skips the entire session if the server is unreachable, and fails clearly if it
    requires auth but no SD_UNAME / SD_PASS were provided.
    """
    client = A1111Webservice(api_url, credentials_provider=lambda: credentials)
    try:
        # A cheap call that also transparently drives the login flow on a 401.
        client.get_samplers()
    except AuthError as err:
        pytest.fail(
            f'Server at {api_url} requires authentication but SD_UNAME / SD_PASS were not usable: {err}'
        )
    except SDBackendError as err:
        pytest.skip(f'No WebUI server reachable at {api_url}: {err}')
    yield client
    client.disconnect()


@pytest.fixture(scope='session')
def upscale_source(service) -> Image.Image:
    """A WebUI txt2img render of `UPSCALE_SOURCE_PROMPT`, generated once per session as the upscale tests' source."""
    body = fast_request_body(UPSCALE_SOURCE_PROMPT)
    body.width = body.height = UPSCALE_SOURCE_SIZE
    body.steps = 20
    body.cfg_scale = 7.0
    return service.txt2img(body)['images'][0]


@pytest.fixture(scope='session')
def comfy_upscale_source(comfy_service, comfy_checkpoint) -> Image.Image:
    """A ComfyUI txt2img render of `UPSCALE_SOURCE_PROMPT`, generated once per session as the upscale tests' source."""
    params = build_comfy_params(comfy_checkpoint, prompt=UPSCALE_SOURCE_PROMPT)
    params.width = params.height = UPSCALE_SOURCE_SIZE
    params.steps = 20
    return comfy_service.submit_txt2img(params).wait(timeout=300).images[0]
