"""Records integration-test outcomes to ``results.js`` in the output dir, for ``tests/review.html`` to display.

The file is a script assigning ``window.SD_TEST_RESULTS``, not plain JSON, because browsers block ``fetch`` from a
page opened as a local file but still load a ``<script src>``. Each session merges into the existing file, so a run
of one test module keeps the other modules' results, matching how each test overwrites only its own PNGs.

A skip doesn't replace an earlier pass or failure: the earlier result stays and gains a ``skipped_later`` note, so a
quick ``pytest tests/`` after a ``--run-generation`` run doesn't hide the generation results the images came from.

Unit tests (``tests/unit/``) are not recorded.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Optional

import pytest

RESULTS_FILENAME = 'results.js'
RESULTS_FORMAT = 1
_ASSIGNMENT = 'window.SD_TEST_RESULTS ='
_HEADER = '// Written by tests/review_results.py and read by tests/review.html. Generated: do not edit.\n'


def result_key(nodeid: str) -> Optional[str]:
    """The key results are stored under: ``<module file name>::<test name>``, or None for a test outside ``tests/``.

    Dropping the directory keeps keys the same whatever directory pytest was invoked from. Only modules directly in
    ``tests/`` get a key, so unit tests are left out.
    """
    path, sep, name = nodeid.partition('::')
    if not sep:
        return None
    module = PurePosixPath(path)
    if module.parent.name != 'tests':
        return None
    return f'{module.name}::{name}'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _skip_reason(report: pytest.TestReport) -> str:
    if isinstance(report.longrepr, tuple):
        reason = str(report.longrepr[2])
        return reason.removeprefix('Skipped: ')
    return report.longreprtext


def merge_results(previous: dict[str, dict[str, Any]], session: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge one session's test entries into the previously saved ones.

    A session entry replaces the saved one, except that a skip keeps a saved non-skip result and records itself
    there as ``skipped_later``.
    """
    merged = dict(previous)
    for key, entry in session.items():
        old = merged.get(key)
        if entry['outcome'] == 'skipped' and old is not None and old['outcome'] != 'skipped':
            merged[key] = {**old, 'skipped_later': {'time': entry['time'], 'reason': entry['details']}}
        else:
            merged[key] = entry
    return merged


def read_results(path: str) -> Optional[dict[str, Any]]:
    """Parse a results file written by `write_results`, or return None if it is missing or unreadable."""
    try:
        with open(path, encoding='utf-8') as file:
            text = file.read()
        data = json.loads(text.split(_ASSIGNMENT, 1)[1].strip().removesuffix(';'))
    except (OSError, IndexError, ValueError):
        return None
    if not isinstance(data, dict) or data.get('format') != RESULTS_FORMAT:
        return None
    return data


def write_results(path: str, data: dict[str, Any]) -> None:
    """Write ``data`` as a results script, replacing the file atomically."""
    temp_path = f'{path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as file:
        file.write(_HEADER)
        file.write(f'{_ASSIGNMENT}\n{json.dumps(data, indent=1, sort_keys=True)};\n')
    os.replace(temp_path, path)


class ResultsRecorder:
    """Pytest plugin collecting each integration test's combined setup/call/teardown outcome."""

    def __init__(self, output_dir: str, generation_enabled: bool) -> None:
        self._output_dir = output_dir
        self._generation_enabled = generation_enabled
        self._session: dict[str, dict[str, Any]] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Fold one phase's report into the test's entry: a failed phase wins over a skip or a pass."""
        key = result_key(report.nodeid)
        if key is None:
            return
        entry = self._session.setdefault(key, {'outcome': 'passed', 'time': _now(), 'duration': 0.0, 'details': ''})
        entry['duration'] = round(entry['duration'] + report.duration, 2)
        if report.failed:
            if entry['outcome'] not in ('failed', 'error'):
                entry['outcome'] = 'failed' if report.when == 'call' else 'error'
                entry['details'] = ''
            prefix = '' if report.when == 'call' else f'[error during {report.when}]\n'
            entry['details'] = '\n\n'.join(part for part in (entry['details'], prefix + report.longreprtext) if part)
        elif report.skipped and entry['outcome'] == 'passed':
            entry['outcome'] = 'skipped'
            entry['details'] = _skip_reason(report)

    def pytest_sessionfinish(self) -> None:
        """Merge this session's entries into ``results.js``, with the current PNG timestamps."""
        if not self._session:
            return
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, RESULTS_FILENAME)
        previous = read_results(path) or {}
        images = {}
        for name in sorted(os.listdir(self._output_dir)):
            if name.endswith('.png'):
                mtime = os.path.getmtime(os.path.join(self._output_dir, name))
                images[name] = datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec='seconds')
        write_results(path, {
            'format': RESULTS_FORMAT,
            'updated': _now(),
            'generation_enabled': self._generation_enabled,
            'images': images,
            'tests': merge_results(previous.get('tests', {}), self._session),
        })
