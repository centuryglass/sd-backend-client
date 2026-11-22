"""The ControlNet reference data in `data/controlnet/` loads, links up and covers every name the recorded servers use.

`scripts/controlnet_data.py` holds the schema and the cross-record checks; this file runs them, and checks the data
against `fixtures/recorded/` so a new recording with an unknown preprocessor or model fails here.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from .recorded_replay import RECORDED_DIR

SCRIPT_PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'controlnet_data.py'

# Names the recorded servers list among preprocessors that aren't preprocessors, so no record covers them.
NOT_PREPROCESSORS = {
    # The WebUI's "no preprocessor" option.
    'none', 'None',
    # comfyui_controlnet_aux helpers that resize, route, annotate or batch-run other preprocessors.
    'AIO_Preprocessor', 'ControlNetAuxSimpleAddText', 'ControlNetPreprocessorSelector',
    'ExecuteAllControlNetPreprocessors', 'HintImageEnchance', 'ImageGenResolutionFromImage',
    'ImageGenResolutionFromLatent', 'PixelPerfectResolution',
    # comfyui_controlnet_aux pose-keypoint postprocessing, which takes POSE_KEYPOINT data, not an image.
    'FacialPartColoringFromPoseKps', 'RenderAnimalKps', 'RenderPeopleKps', 'SavePoseKpsAsJsonFile',
    'UpperBodyTrackingFromPoseKps',
    # Optical flow for video models, outside the SD1.5/SDXL image scope.
    'MaskOptFlow', 'Unimatch_OptFlowPreprocessor',
}

# Model files in the recordings that no record claims yet, each tracked by an open question in the named record.
UNIDENTIFIED_MODEL_FILES = {
    'controlnetxlCNXL_destitechInpaint-inpainting': 'destitech_inpaint_dreamer_sdxl',
}

# The WebUI ControlNet extensions append ` [<webui_hash>]` to model names; Forge Neo doesn't.
WEBUI_MODEL_NAME = re.compile(r'^(?P<stem>.*?)(?: \[(?P<hash>[0-9a-f]{8})\])?$')
WEIGHT_EXTENSION = re.compile(r'\.(safetensors|pth|ckpt|bin)$')


def _load_script() -> ModuleType:
    """Import `scripts/controlnet_data.py`, which isn't on the import path."""
    spec = importlib.util.spec_from_file_location('controlnet_data', SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


controlnet_data = _load_script()


@pytest.fixture(scope='module', name='data')
def fixture_data():
    """Every record, loaded once."""
    return controlnet_data.load_all()


def _recordings() -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text(encoding='utf-8'))['responses']
            for path in sorted(RECORDED_DIR.glob('*.json'))}


def _recorded_preprocessor_names() -> list[tuple[str, str]]:
    """(backend, name) for every preprocessor the recorded servers list."""
    names = []
    for label, responses in _recordings().items():
        if '/controlnet/module_list' in responses:
            module_list = responses['/controlnet/module_list']
            modules = module_list.get('module_list') or list(module_list.get('module_detail', {}))
            backend = 'webui' if label == 'a1111' else 'forge'
            names.extend((backend, name) for name in modules)
        if '/object_info' in responses:
            names.extend(('comfyui', name) for name, node in responses['/object_info'].items()
                         if 'ControlNet Preprocessors' in node['category'])
    return names


def _recorded_model_names() -> list[tuple[str, str | None]]:
    """(file name stem, webui hash or None) for every ControlNet model the recorded servers list."""
    names: list[tuple[str, str | None]] = []
    for responses in _recordings().values():
        listed = responses.get('/controlnet/model_list', {}).get('model_list', []) \
            + responses.get('/models/controlnet', [])
        for name in listed:
            if name == 'None':
                continue
            match = WEBUI_MODEL_NAME.match(name)
            assert match is not None
            names.append((WEIGHT_EXTENSION.sub('', match['stem']), match['hash']))
    return names


def test_every_file_loads(data):
    """Every file parses into its model, and the data covers each record kind."""
    assert data.control_types and data.preprocessors and data.models and data.sources


def test_references_resolve(data):
    """Every id a record names is a record of the right kind."""
    assert controlnet_data.reference_errors(data) == []


def test_verified_records_are_sourced(data):
    """A verified record has a non-tutorial source and no open questions."""
    assert controlnet_data.status_errors(data) == []


def test_text_uses_ascii_hyphens(data):
    """No record contains an em or en dash."""
    assert controlnet_data.text_errors(data) == []


def test_shared_names_are_disambiguated(data):
    """Records sharing a backend name differ by name_parameters."""
    assert controlnet_data.name_collision_errors(data) == []


def test_recorded_preprocessors_have_records(data):
    """Every preprocessor name in the recorded fixtures belongs to a record, apart from NOT_PREPROCESSORS."""
    known = {(backend, name) for record in data.preprocessors.values()
             for backend in controlnet_data.BACKENDS for name in getattr(record.names, backend)}
    missing = sorted({(backend, name) for backend, name in _recorded_preprocessor_names()
                      if (backend, name) not in known and name not in NOT_PREPROCESSORS})
    assert missing == []


def test_recorded_models_have_records(data):
    """Every ControlNet model in the recorded fixtures matches a record file, by name and by webui hash."""
    files = [model_file for model in data.models.values() for model_file in model.files]
    by_stem: dict[str, list] = {}
    for model_file in files:
        for name in (model_file.filename, *model_file.also_seen_as):
            by_stem.setdefault(WEIGHT_EXTENSION.sub('', name), []).append(model_file)
    problems = []
    for stem, webui_hash in _recorded_model_names():
        if stem in UNIDENTIFIED_MODEL_FILES:
            continue
        matches = by_stem.get(stem, [])
        if not matches:
            problems.append(f'{stem}: no record file')
        elif webui_hash is not None and all(model_file.webui_hash != webui_hash for model_file in matches):
            problems.append(f'{stem} [{webui_hash}]: matching record files have a different webui_hash')
    assert not problems, '\n'.join(problems)


def test_unidentified_model_files_are_tracked(data):
    """Each UNIDENTIFIED_MODEL_FILES entry names a record that still has an open question about it."""
    for stem, record_id in UNIDENTIFIED_MODEL_FILES.items():
        assert any(stem in question for question in data.models[record_id].open_questions), stem


def test_schema_rejects_unknown_keys():
    """A misspelled field fails validation instead of being dropped."""
    with pytest.raises(controlnet_data.ValidationError):
        controlnet_data.Preprocessor.model_validate({'id': 'x', 'label': 'X', 'contrl_types': []})


@pytest.mark.parametrize('label, summary', [('A' * (controlnet_data.LABEL_MAX_CHARS + 1), ''),
                                            ('Label.', ''),
                                            ('Label', 'S' * (controlnet_data.SUMMARY_MAX_CHARS + 1)),
                                            ('Label', 'two\nlines')])
def test_schema_enforces_text_tiers(label, summary):
    """Labels and summaries stay within their tier limits."""
    with pytest.raises(controlnet_data.ValidationError):
        controlnet_data.ControlType.model_validate({'id': 'x', 'label': label, 'summary': summary})


def test_schema_requires_prose_past_stub():
    """A drafted record needs its summary, description and how_it_works."""
    with pytest.raises(controlnet_data.ValidationError):
        controlnet_data.ControlType.model_validate({'id': 'x', 'label': 'X', 'summary': 'S.',
                                                    'research_status': 'drafted'})


def test_local_code_sources_use_project_paths():
    """A local_code source names its project and commit, and never a local filesystem path."""
    base = {'title': 'T', 'kind': 'local_code', 'accessed': '2026-11-22', 'project': 'p', 'commit': 'c'}
    controlnet_data.Source.model_validate({**base, 'path': 'scripts/x.py'})
    for bad in ({**base, 'path': '/home/user/x.py'}, {**base, 'path': ''}):
        with pytest.raises(controlnet_data.ValidationError):
            controlnet_data.Source.model_validate(bad)
