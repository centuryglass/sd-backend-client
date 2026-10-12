"""Tests for `tests/review_results.py`, the plugin that records integration-test outcomes for `tests/review.html`."""
from typing import Literal

import pytest

from ..review_results import (RESULTS_FILENAME, ResultsRecorder, merge_results, read_results, result_key,
                              write_results)

GEN = 'tests/test_a1111_generation.py::test_txt2img_produces_image'


def _report(when: Literal['setup', 'call', 'teardown'], outcome: Literal['passed', 'failed', 'skipped'],
            longrepr=None, nodeid: str = GEN) -> pytest.TestReport:
    return pytest.TestReport(nodeid, ('tests/test_a1111_generation.py', 0, 'test'), {}, outcome, longrepr, when,
                             duration=0.5)


def _record(tmp_path, *reports: pytest.TestReport) -> dict:
    recorder = ResultsRecorder(str(tmp_path), generation_enabled=True)
    for report in reports:
        recorder.pytest_runtest_logreport(report)
    recorder.pytest_sessionfinish()
    data = read_results(str(tmp_path / RESULTS_FILENAME))
    assert data is not None
    return data


def test_result_key_drops_the_directory_and_ignores_unit_tests():
    """Keys are `<module file>::<test>`, and only modules directly in `tests/` get one."""
    assert result_key(GEN) == 'test_a1111_generation.py::test_txt2img_produces_image'
    assert result_key('test_inpaint_full_res.py::test_x[webui]') is None
    assert result_key('tests/test_inpaint_full_res.py::test_x[webui]') == 'test_inpaint_full_res.py::test_x[webui]'
    assert result_key('tests/unit/test_public_api.py::test_all') is None
    assert result_key('tests/test_a1111_generation.py') is None


def test_merge_keeps_an_earlier_result_when_the_new_one_is_a_skip():
    """A skip notes itself on an earlier pass or failure; any other outcome replaces it."""
    passed = {'outcome': 'passed', 'time': 't1', 'duration': 1.0, 'details': ''}
    skipped = {'outcome': 'skipped', 'time': 't2', 'duration': 0.0, 'details': 'opt-in'}
    failed = {'outcome': 'failed', 'time': 't3', 'duration': 1.0, 'details': 'boom'}

    merged = merge_results({'a': passed, 'b': passed}, {'a': skipped, 'c': skipped})
    assert merged['a'] == {**passed, 'skipped_later': {'time': 't2', 'reason': 'opt-in'}}
    assert merged['b'] == passed
    assert merged['c'] == skipped
    assert merge_results(merged, {'a': failed})['a'] == failed


def test_results_round_trip_and_unreadable_files_read_as_none(tmp_path):
    """`read_results` parses what `write_results` wrote, and returns None for a missing, garbled or foreign file."""
    path = str(tmp_path / RESULTS_FILENAME)
    assert read_results(path) is None
    data = {'format': 1, 'tests': {'k': {'outcome': 'passed'}}}
    write_results(path, data)
    assert read_results(path) == data
    (tmp_path / RESULTS_FILENAME).write_text('not a results file', encoding='utf-8')
    assert read_results(path) is None
    write_results(path, {'format': 999, 'tests': {}})
    assert read_results(path) is None


def test_recorder_combines_phases_into_one_outcome(tmp_path):
    """A call failure plus a teardown error is one failed entry carrying both details and the summed duration."""
    data = _record(tmp_path, _report('setup', 'passed'), _report('call', 'failed', 'AssertionError: blank'),
                   _report('teardown', 'failed', 'cleanup broke'))
    entry = data['tests']['test_a1111_generation.py::test_txt2img_produces_image']
    assert entry['outcome'] == 'failed'
    assert entry['details'] == 'AssertionError: blank\n\n[error during teardown]\ncleanup broke'
    assert entry['duration'] == 1.5
    assert data['generation_enabled'] is True


def test_recorder_reports_setup_errors_and_skip_reasons(tmp_path):
    """A setup failure is an error, and a skip keeps its reason without the "Skipped: " prefix."""
    other = 'tests/test_comfyui_generation.py::test_upscale'
    data = _record(tmp_path, _report('setup', 'failed', 'ConnectionError'),
                   _report('setup', 'skipped', ('x.py', 1, 'Skipped: no upscale model'), nodeid=other))
    tests = data['tests']
    assert tests['test_a1111_generation.py::test_txt2img_produces_image']['outcome'] == 'error'
    assert tests['test_comfyui_generation.py::test_upscale'] == {
        'outcome': 'skipped', 'time': tests['test_comfyui_generation.py::test_upscale']['time'],
        'duration': 0.5, 'details': 'no upscale model'}


def test_recorder_lists_saved_images_and_skips_writing_without_integration_tests(tmp_path):
    """The file lists the PNGs in the output dir, and a unit-test-only session writes nothing."""
    (tmp_path / 'txt2img.png').write_bytes(b'')
    unit_only = ResultsRecorder(str(tmp_path), generation_enabled=False)
    unit_only.pytest_runtest_logreport(_report('call', 'passed', nodeid='tests/unit/test_x.py::test_y'))
    unit_only.pytest_sessionfinish()
    assert not (tmp_path / RESULTS_FILENAME).exists()

    data = _record(tmp_path, _report('call', 'passed'))
    assert list(data['images']) == ['txt2img.png']
