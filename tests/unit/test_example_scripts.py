"""The scripts in `examples/` build the requests they should, against a stand-in backend.

Each script's `main()` runs with a patched `connect` that returns an autospecced client whose `submit_*` methods
return an already-finished handle, so no server is needed. The tests assert the parameters the script submitted and the
files it wrote. `test_readme_examples.py` covers the scripts' imports and `preprocess.py` against recorded servers.
"""
# pylint: disable=redefined-outer-name
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from unittest.mock import create_autospec

import pytest
from PIL import Image

from sd_backend_client import A1111Webservice, Backend, BackendCapabilities, BackendOption, ComfyUiWebservice, \
    GenerationHandle, GenerationProgress, GenerationResult, GenerationStatus

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'
SCRIPTS = sorted(path.stem for path in EXAMPLES.glob('*.py') if not path.name.startswith('_'))
SIZE = (64, 64)


class FinishedHandle(GenerationHandle):
    """A handle whose job has finished with one gradient image, so crops of it differ from each other."""

    def __init__(self) -> None:
        super().__init__('task')

    def poll(self) -> GenerationProgress:
        return GenerationProgress(GenerationStatus.FINISHED, progress=1.0)

    def _build_result(self) -> GenerationResult:
        return GenerationResult(images=[Image.linear_gradient('L').resize(SIZE).convert('RGB')], seeds=[77], seed=77)

    def cancel(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def examples_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Import scripts by name, as running them does, and work inside an empty directory."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.chdir(tmp_path)


def make_backend(cls: type = ComfyUiWebservice) -> Any:
    """An autospecced client of `cls` whose jobs finish at once and whose discovery lists are small."""
    backend = create_autospec(cls, instance=True)
    for name in ('submit_txt2img', 'submit_img2img', 'submit_inpaint', 'submit_upscale',
                 'submit_preprocessor_preview'):
        getattr(backend, name).side_effect = lambda *_a, **_k: FinishedHandle()
    backend.list_samplers.return_value = [BackendOption(name='euler_ancestral', display_name='Euler a')]
    backend.list_checkpoints.return_value = [BackendOption(name='model.safetensors')]
    for name in ('list_vaes', 'list_loras', 'list_hypernetworks', 'list_schedulers', 'list_upscalers'):
        getattr(backend, name).return_value = []
    backend.list_controlnet_models.return_value = []
    backend.get_controlnet_preprocessors.return_value = []
    backend.get_capabilities.return_value = BackendCapabilities(controlnet=False, ultimate_upscale=False,
                                                                soft_inpainting=False, scheduler=True,
                                                                interrogate=False, free_memory=True)
    return backend


@pytest.fixture
def run_script(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """Run a script's `main()` with `argv` against `backend`, returning the (reloaded) script module."""

    def run(name: str, argv: list[str], backend: Backend) -> ModuleType:
        for module_name in ('_common', 'img2img', name):
            sys.modules.pop(module_name, None)
        module = importlib.import_module(name)
        monkeypatch.setattr(module, 'connect', lambda _args: backend, raising=False)
        monkeypatch.setattr(importlib.import_module('img2img'), 'connect', lambda _args: backend)
        monkeypatch.setattr(sys, 'argv', [name, *argv])
        module.main()
        return module

    return run


def png_text(path: Path) -> str:
    """The `parameters` text chunk of the PNG at `path`."""
    with Image.open(path) as image:
        return str(image.info['parameters'])


@pytest.mark.parametrize('name', SCRIPTS)
def test_help_exits_cleanly(name: str, run_script: Callable[..., Any], capsys: pytest.CaptureFixture[str]):
    """Every script prints usage for `--help` and exits with status 0."""
    with pytest.raises(SystemExit) as exit_info:
        run_script(name, ['--help'], make_backend())
    assert exit_info.value.code == 0
    assert 'usage' in capsys.readouterr().out.lower()


def test_txt2img_submits_params_and_embeds_settings(run_script: Callable[..., Any], tmp_path: Path):
    """Flags reach `DiffusionParams`, and the saved PNG carries them in an A1111-style `parameters` chunk."""
    backend = make_backend()
    run_script('txt2img', ['a fox', '-n', 'blurry', '--width', '256', '--steps', '12', '--cfg', '5.5', '--seed', '9',
                           '--sampler', 'euler_ancestral', '--checkpoint', 'model.safetensors', '--batch-size', '2',
                           '-o', str(tmp_path / 'out')], backend)

    params = backend.submit_txt2img.call_args.args[0]
    assert (params.prompt, params.negative_prompt, params.width, params.steps) == ('a fox', 'blurry', 256, 12)
    assert (params.cfg_scale, params.seed, params.batch_size) == (5.5, 9, 2)
    saved = sorted((tmp_path / 'out').glob('*.png'))
    assert [path.name for path in saved] == ['00001-77.png']
    text = png_text(saved[0])
    assert text.startswith('a fox\nNegative prompt: blurry\nSteps: 12, Sampler: Euler a')
    assert 'Seed: 77' in text and 'Size: 256x512' in text and 'Model: model.safetensors' in text


def test_txt2img_does_not_overwrite_earlier_outputs(run_script: Callable[..., Any], tmp_path: Path):
    """A second run into the same directory continues the numbering."""
    for _ in range(2):
        run_script('txt2img', ['x', '-o', str(tmp_path / 'out')], make_backend())
    assert [path.name for path in sorted((tmp_path / 'out').glob('*.png'))] == ['00001-77.png', '00002-77.png']


def test_txt2img_rejects_invalid_params(run_script: Callable[..., Any]):
    """An out-of-range value exits with a message instead of a traceback."""
    with pytest.raises(SystemExit, match='Invalid generation parameters'):
        run_script('txt2img', ['x', '--steps', '0'], make_backend())


def test_parameters_text_round_trips():
    """`parse_parameters` recovers what `format_parameters` wrote, including commas and multi-line negatives."""
    common = importlib.import_module('_common')
    params = common.build_params({'prompt': 'a cat\nsitting', 'negative_prompt': 'bad\nugly', 'steps': 20,
                                  'cfg_scale': 6.5, 'width': 320, 'height': 192, 'sampler': 'dpmpp_2m',
                                  'scheduler': 'karras', 'sd_model_name': 'my, model.ckpt', 'batch_size': 3})
    parsed = common.parse_parameters(common.format_parameters(params, 99))
    assert parsed == {'prompt': 'a cat\nsitting', 'negative_prompt': 'bad\nugly', 'steps': 20, 'cfg_scale': 6.5,
                      'width': 320, 'height': 192, 'sampler': 'dpmpp_2m', 'scheduler': 'karras', 'seed': 99,
                      'sd_model_name': 'my, model.ckpt', 'batch_size': 3}


def test_parse_parameters_requires_settings_line():
    """Text without a `Steps:` line is not generation metadata."""
    with pytest.raises(ValueError):
        importlib.import_module('_common').parse_parameters('just a caption')


def test_img2img_and_inpaint_choose_the_job(run_script: Callable[..., Any]):
    """`--mask` switches img2img to inpainting, and `inpaint.py` requires it."""
    Image.new('RGB', (100, 70), 'blue').save('in.png')
    Image.new('L', (100, 70), 255).save('mask.png')

    backend = make_backend()
    run_script('img2img', ['p', '--image', 'in.png', '--strength', '0.5', '-o', 'o1'], backend)
    params = backend.submit_img2img.call_args.args[0]
    assert (params.width, params.height, params.denoising_strength) == (96, 64, 0.5)
    assert params.init_images and params.mask is None
    backend.submit_inpaint.assert_not_called()

    backend = make_backend()
    run_script('inpaint', ['p', '--image', 'in.png', '--mask', 'mask.png', '-o', 'o2'], backend)
    assert backend.submit_inpaint.call_args.args[0].mask is not None
    backend.submit_img2img.assert_not_called()

    with pytest.raises(SystemExit) as exit_info:
        run_script('inpaint', ['p', '--image', 'in.png'], make_backend())
    assert exit_info.value.code == 2


def test_rerun_reads_saved_settings_and_offsets_seed(run_script: Callable[..., Any]):
    """`rerun.py` regenerates from a txt2img output, with `--seed +1` and `--steps` applied."""
    run_script('txt2img', ['a fox', '--seed', '10', '--steps', '15', '-o', 'first'], make_backend())
    saved = next(Path('first').glob('*.png'))  # The stand-in reports seed 77 for the image.

    backend = make_backend()
    run_script('rerun', [str(saved), '--seed', '+1', '--steps', '40', '-o', 'second'], backend)

    params = backend.submit_txt2img.call_args.args[0]
    assert (params.prompt, params.seed, params.steps) == ('a fox', 78, 40)
    assert params.sampler == 'Euler a'


def test_rerun_rejects_png_without_settings(run_script: Callable[..., Any]):
    """A PNG with no `parameters` chunk exits with an explanation."""
    Image.new('RGB', SIZE).save('plain.png')
    with pytest.raises(SystemExit, match='no "parameters" text chunk'):
        run_script('rerun', ['plain.png'], make_backend())


def test_backend_info_json(run_script: Callable[..., Any], capsys: pytest.CaptureFixture[str]):
    """`--json` prints every listing and the capabilities as one JSON object."""
    run_script('backend_info', ['--json'], make_backend())
    info = json.loads(capsys.readouterr().out)
    assert info['checkpoints'] == ['model.safetensors']
    assert info['samplers'] == ['euler_ancestral']
    assert info['capabilities']['scheduler'] is True


def test_backend_info_text(run_script: Callable[..., Any], capsys: pytest.CaptureFixture[str]):
    """The text report names the backend and lists the checkpoints."""
    run_script('backend_info', [], make_backend())
    out = capsys.readouterr().out
    assert 'model.safetensors' in out and 'checkpoints (1)' in out


def test_upscale_size_and_upscaler(run_script: Callable[..., Any]):
    """`--scale` multiplies the source size and `--upscaler` selects the model."""
    Image.new('RGB', (40, 30)).save('in.png')
    backend = make_backend()
    run_script('upscale', ['in.png', 'out.png', '--scale', '2.5', '--upscaler', 'ESRGAN'], backend)
    args = backend.submit_upscale.call_args.args
    assert args[1:3] == (100, 75)
    assert args[3].upscaling_mode == 'ESRGAN'
    assert Image.open('out.png').size == SIZE


@pytest.mark.parametrize('argv', [['in.png', 'out.png'], ['in.png', 'out.png', '--scale', '2', '--size', '9x9'],
                                  ['in.png', 'out.png', '--size', 'big']])
def test_upscale_rejects_bad_size_flags(run_script: Callable[..., Any], argv: list[str]):
    """Zero or two size flags, or a malformed size, exit with a message."""
    Image.new('RGB', (40, 30)).save('in.png')
    with pytest.raises(SystemExit):
        run_script('upscale', argv, make_backend())


def test_xy_grid_submits_each_cell(run_script: Callable[..., Any]):
    """A 3x2 grid queues six jobs with the swept values and writes a labeled sheet taller than one row."""
    backend = make_backend()
    run_script('xy_grid', ['a fox', '--x', 'cfg=4,7,10', '--y', 'steps=10,20', '--seed', '3'], backend)

    submitted = [call.args[0] for call in backend.submit_txt2img.call_args_list]
    assert [(p.cfg_scale, p.steps) for p in submitted] == [(c, s) for s in (10, 20) for c in (4.0, 7.0, 10.0)]
    assert {p.seed for p in submitted} == {3}
    sheet = Image.open('grid.png')
    assert sheet.width > 3 * SIZE[0] and sheet.height > 2 * SIZE[1]


def test_xy_grid_prompt_search_and_replace(run_script: Callable[..., Any]):
    """`prompt_sr` replaces its first value in the prompt with each value."""
    backend = make_backend()
    run_script('xy_grid', ['a red car', '--x', 'prompt_sr=red,blue'], backend)
    assert [call.args[0].prompt for call in backend.submit_txt2img.call_args_list] == ['a red car', 'a blue car']


@pytest.mark.parametrize('axis', ['nonsense=1,2', 'cfg=', 'cfg'])
def test_xy_grid_rejects_bad_axis(run_script: Callable[..., Any], axis: str):
    """An unknown name or an empty value list exits with a message."""
    with pytest.raises(SystemExit, match='Bad axis'):
        run_script('xy_grid', ['x', '--x', axis], make_backend())


def test_infinite_zoom_masks_the_border_and_writes_an_animation(run_script: Callable[..., Any]):
    """Each round inpaints a bordered canvas (centre kept), and the animation holds every zoom step."""
    backend = make_backend()
    run_script('infinite_zoom', ['a path', '--width', '64', '--height', '64', '--rounds', '2',
                                 '--frames-per-round', '3', '--shrink', '0.5'], backend)

    backend.submit_txt2img.assert_called_once()
    assert backend.submit_inpaint.call_count == 2
    params = backend.submit_inpaint.call_args.args[0]
    assert params.mask.getpixel((0, 0)) == 255 and params.mask.getpixel((32, 32)) == 0
    animation = Image.open('zoom.gif')
    assert animation.n_frames == 2 * 3 + 1


def test_telephone_describes_then_redraws(run_script: Callable[..., Any], capsys: pytest.CaptureFixture[str]):
    """Each round interrogates the current image and uses the caption as the img2img prompt."""
    Image.new('RGB', SIZE).save('start.png')
    backend = make_backend(A1111Webservice)
    backend.interrogate.side_effect = ['first caption', 'second caption', 'final caption']
    run_script('telephone', ['start.png', '--rounds', '2', '-o', 'phone'], backend)

    assert [call.args[0].prompt for call in backend.submit_img2img.call_args_list] == \
        ['first caption', 'second caption']
    assert Path('phone/round-02.png').exists()
    assert Path('phone/captions.txt').read_text(encoding='utf-8').splitlines() == \
        ['first caption', 'second caption', 'final caption']
    assert 'final: final caption' in capsys.readouterr().out


def test_telephone_needs_webui(run_script: Callable[..., Any]):
    """A ComfyUI backend exits with a message, since only WebUI can interrogate."""
    Image.new('RGB', SIZE).save('start.png')
    with pytest.raises(SystemExit, match='needs a WebUI'):
        run_script('telephone', ['start.png'], make_backend(ComfyUiWebservice))
