"""The Python examples in the README and the docs site, and the scripts in `examples/`, run against recorded servers.

Each example runs offline with `connect_to_backend` and the client constructors patched to return a real client whose
`get` replays a recording (see `recorded_replay`), so discovery calls parse real responses. `detect` is patched to
return a fixed result, so no example reads this machine's files or probes its ports. `ManagedComfyUI` runs a fake
process whose server is ready at once, and its `connect` returns the offline client. The `submit_*` methods are
replaced by autospecced stand-ins that check the call signature and return an already-finished handle. A renamed
export, method or parameter fails here.
"""
import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional
from unittest.mock import create_autospec

import pytest
from PIL import Image

import sd_backend_client
from sd_backend_client import A1111Webservice, Backend, BackendKind, ComfyUiWebservice, DetectedInstall, \
    DetectedServer, DetectionResult, Finding, FindingCode, GenerationHandle, GenerationProgress, GenerationResult, \
    GenerationStatus, InstallOwner, ManagedComfyUI, ManagedState, PreprocessorParams

from .fake_process import FakeProcessFactory
from .recorded_replay import recordings, replay

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / 'README.md'
DOCS = REPO_ROOT / 'docs'
EXAMPLE_SCRIPTS = sorted((REPO_ROOT / 'examples').glob('*.py'))
SUBMIT_METHODS = ('submit_txt2img', 'submit_img2img', 'submit_inpaint', 'submit_upscale',
                  'submit_preprocessor_preview')
RECORDINGS = recordings('comfyui') + recordings('webui')


def _python_blocks(path: Path) -> list[str]:
    """Every fenced ```python block in a markdown file, in order."""
    return re.findall(r'^```python\n(.*?)^```', path.read_text(encoding='utf-8'), re.MULTILINE | re.DOTALL)


README_BLOCKS = _python_blocks(README)

# Docs pages whose examples are written for one backend; the others run against every recording.
DOC_PAGE_BACKENDS = {'quickstart/webui.md': 'webui', 'quickstart/comfyui.md': 'comfyui'}
DOC_PAGES = [page for page in sorted(DOCS.rglob('*.md')) if _python_blocks(page)]


def _doc_page_cases() -> list[Any]:
    """pytest params pairing each docs page that has examples with each recording its examples target."""
    cases = []
    for page in DOC_PAGES:
        name = page.relative_to(DOCS).as_posix()
        backend = DOC_PAGE_BACKENDS.get(name)
        page_recordings = recordings(backend) if backend else RECORDINGS
        cases += [pytest.param(page, param.values[0], id=f'{name}-{param.id}') for param in page_recordings]
    return cases


class _FinishedHandle(GenerationHandle):
    """A handle whose job has already finished with one small image."""

    def __init__(self) -> None:
        super().__init__('finished-task')
        self.cancelled = False

    def poll(self) -> GenerationProgress:
        return GenerationProgress(GenerationStatus.FINISHED, progress=1.0)

    def _build_result(self) -> GenerationResult:
        return GenerationResult(images=[Image.new('RGB', (8, 8))], seeds=[1], seed=1, task_id=self.task_id)

    def cancel(self) -> bool:
        self.cancelled = True
        return True


def _detection_result() -> DetectionResult:
    """A result with a running ComfyUI install, a WebUI install with a finding, and an unlinked server."""
    comfyui = DetectedInstall(kind=BackendKind.COMFYUI, owner=InstallOwner.NONE, path=Path('/sd/ComfyUI'),
                              entry_point=Path('/sd/ComfyUI/main.py'),
                              launch_command=['/sd/ComfyUI/venv/bin/python', '/sd/ComfyUI/main.py'],
                              server=DetectedServer(url='http://localhost:8188', kind=BackendKind.COMFYUI))
    webui = DetectedInstall(kind=BackendKind.WEBUI, owner=InstallOwner.STABILITY_MATRIX, path=Path('/sd/forge'),
                            entry_point=Path('/sd/forge/launch.py'), webui_variant='forge',
                            findings=[Finding(FindingCode.API_DISABLED, {'arguments': ''})])
    server = DetectedServer(url='http://localhost:7861', kind=None,
                            findings=[Finding(FindingCode.AUTH_REQUIRED, {'url': 'http://localhost:7861'})])
    return DetectionResult(installs=[comfyui, webui], servers=[server])


def _offline_backend(recording: dict[str, Any]) -> Backend:
    """A real client for the recording's backend that replays its responses and finishes every job at once."""
    service: A1111Webservice | ComfyUiWebservice
    if recording['meta']['backend'] == 'comfyui':
        service = ComfyUiWebservice('http://127.0.0.1:8188', live_progress=False)
    else:
        service = A1111Webservice('http://127.0.0.1:7860')
    replay(service, recording)
    for name in SUBMIT_METHODS:
        stand_in = create_autospec(getattr(service, name), side_effect=lambda *_args, **_kwargs: _FinishedHandle())
        setattr(service, name, stand_in)
    return service


def _offline_managed_comfyui(backend: Backend) -> type[ManagedComfyUI]:
    """A `ManagedComfyUI` that runs a fake process, is ready once started and connects to `backend`."""

    class OfflineManagedComfyUI(ManagedComfyUI):
        """Runs a fake process instead of the install, and connects to the recording's client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, process_factory=FakeProcessFactory(),
                             health_check=self._answer, **kwargs)

        def _answer(self, _url: str) -> Optional[list[str]]:
            return self.command[1:] if self.state == ManagedState.STARTING else None

        def connect(self, **kwargs: Any) -> Any:
            return backend

    return OfflineManagedComfyUI


@pytest.fixture(name='connect_offline')
def fixture_connect_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run in an empty directory, and return a function that points the package's clients at a recording.

    `connect_to_backend`, `ManagedComfyUI.connect` and the constructor of the recording's client class return the
    offline client. `detect` returns `_detection_result()` from the start.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sd_backend_client, 'detect',
                        create_autospec(sd_backend_client.detect, return_value=_detection_result()))

    def connect(recording: dict[str, Any]) -> Backend:
        backend = _offline_backend(recording)
        connect_stand_in = create_autospec(sd_backend_client.connect_to_backend, return_value=backend)
        monkeypatch.setattr(sd_backend_client, 'connect_to_backend', connect_stand_in)
        client_class = type(backend)
        monkeypatch.setattr(sd_backend_client, client_class.__name__,
                            create_autospec(client_class, return_value=backend))
        monkeypatch.setattr(sd_backend_client, 'ManagedComfyUI', _offline_managed_comfyui(backend))
        return backend

    return connect


def _create_opened_images(source: str, directory: Path) -> None:
    """Write a small placeholder for each file the code passes to `Image.open`."""
    for name in re.findall(r"Image\.open\('([^']+)'\)", source):
        Image.new('RGB', (512, 512), 'gray').save(directory / name)


def _load_example(path: Path) -> ModuleType:
    """Import an example script as a module, without running its `__main__` block."""
    spec = importlib.util.spec_from_file_location(f'example_{path.stem}', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_imports(source: str) -> list[tuple[str, list[str]]]:
    """The (module, imported names) of every import of the package or one of its modules in `source`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'sd_backend_client':
            found.append((node.module or '', [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names if alias.name.split('.')[0] == 'sd_backend_client']
    return found


def test_readme_has_python_examples():
    """The block extraction still finds the README's examples, so the tests below never silently run none."""
    assert len(README_BLOCKS) >= 5


def test_docs_have_python_examples():
    """The block extraction still finds the docs site's examples, so the docs tests below never silently run none."""
    assert {page.relative_to(DOCS).as_posix() for page in DOC_PAGES} >= {'index.md', *DOC_PAGE_BACKENDS}


@pytest.mark.parametrize('path', [README, *DOC_PAGES, *EXAMPLE_SCRIPTS], ids=lambda path: path.name)
def test_examples_import_only_the_public_api(path: Path):
    """Examples import from the package root, and only names in `__all__`, so they show the supported API."""
    sources = _python_blocks(path) if path.suffix == '.md' else [path.read_text(encoding='utf-8')]
    imports = [found for code in sources for found in _package_imports(code)]
    # Scripts built on `examples/_common.py` may take everything they need from it.
    assert imports or 'from _common import' in sources[0], f'{path.name} imports nothing from sd_backend_client'
    for module, names in imports:
        assert module == 'sd_backend_client', f'{path.name} imports from internal module {module}'
        for name in names:
            assert name in sd_backend_client.__all__, f'{path.name} imports {name}, which is not in __all__'


@pytest.mark.parametrize('recording', RECORDINGS)
@pytest.mark.parametrize('block_index', range(len(README_BLOCKS)), ids=lambda index: f'block{index}')
def test_readme_example_runs(block_index: int, recording: dict[str, Any], connect_offline, tmp_path: Path,
                             capsys: pytest.CaptureFixture[str]):
    """Each README example runs to completion against each recorded server."""
    del capsys  # Silences the examples' prints.
    source = README_BLOCKS[block_index]
    backend = connect_offline(recording)
    _create_opened_images(source, tmp_path)
    exec(compile(source, f'README.md block {block_index}', 'exec'), {'__name__': '__readme__'})  # pylint: disable=exec-used
    if 'submit_' in source and 'connect_to_backend' in source:
        assert any(getattr(backend, name).called for name in SUBMIT_METHODS), 'the example submitted no job'


@pytest.mark.parametrize(('page', 'recording'), _doc_page_cases())
def test_docs_page_examples_run(page: Path, recording: dict[str, Any], connect_offline, tmp_path: Path,
                                capsys: pytest.CaptureFixture[str]):
    """Each docs page's examples run to completion, in order and sharing names, against each recorded server.

    A page's later examples build on its earlier ones, such as the `backend` and `params` they define.
    """
    del capsys  # Silences the examples' prints.
    backend = connect_offline(recording)
    namespace: dict[str, Any] = {'__name__': '__docs__'}
    for index, source in enumerate(_python_blocks(page)):
        _create_opened_images(source, tmp_path)
        exec(compile(source, f'{page.relative_to(DOCS)} block {index}', 'exec'), namespace)  # pylint: disable=exec-used
    if 'submit_' in page.read_text(encoding='utf-8'):
        assert any(getattr(backend, name).called for name in SUBMIT_METHODS), 'the page submitted no job'


@pytest.mark.parametrize('recording', RECORDINGS)
def test_preprocess_example_sets_parameters(recording: dict[str, Any], connect_offline, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """The preprocess script finds a recorded preprocessor and passes it the requested parameter value."""
    # The value is the parameter's default, which is always in range.
    del capsys
    backend = connect_offline(recording)
    _create_opened_images("Image.open('in.png')", tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT / 'examples'))
    example = _load_example(REPO_ROOT / 'examples' / 'preprocess.py')
    numeric = next((p, d) for p in backend.get_controlnet_preprocessors() for d in p.parameters
                   if isinstance(d.default_value, int) and not isinstance(d.default_value, bool))
    preprocessor, parameter = numeric
    monkeypatch.setattr(sys, 'argv', ['preprocess', 'in.png', 'out.png', '--preprocessor', preprocessor.name,
                                      '--param', f'{parameter.key}={parameter.default_value}'])
    monkeypatch.setattr(example, 'connect', lambda _args: backend)

    example.main()

    assert (tmp_path / 'out.png').exists()
    submit: Any = backend.submit_preprocessor_preview
    params: Optional[PreprocessorParams] = submit.call_args.args[1]
    assert isinstance(params, PreprocessorParams)
    assert params.typedef.name == preprocessor.name
    assert params.parameter_values[parameter.key] == parameter.default_value
