"""The guide's generated ControlNet pages cover every record and every name, and record links resolve.

`scripts/guide_pages.py` builds the pages; the MkDocs build (`guide/mkdocs.yml`) renders them. These tests run the
generator without mkdocs, so CI's test job checks coverage of the data even though only the docs job builds the site.
"""
import importlib
import posixpath
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts'


# guide_pages imports controlnet_data by name, so scripts/ goes on the import path for both.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
guide_pages = importlib.import_module('guide_pages')


@pytest.fixture(scope='module', name='data')
def fixture_data():
    """Every record, loaded and checked once."""
    return guide_pages.load_checked()


@pytest.fixture(scope='module', name='pages')
def fixture_pages(data):
    """Every generated page, with record links expanded."""
    return {path: guide_pages.expand_record_links(markdown, path, data)
            for path, markdown in guide_pages.generate_pages(data).items()}


def test_every_record_has_a_page(data, pages):
    """Every control type, preprocessor and model gets a page."""
    expected = ({guide_pages.type_path(record_id) for record_id in data.control_types}
                | {guide_pages.preprocessor_path(record_id) for record_id in data.preprocessors}
                | {guide_pages.model_path(record_id) for record_id in data.models})
    assert not expected - set(pages)


def test_every_record_page_is_in_the_nav(data):
    """The generated nav sections list each record page once, plus each directory's index."""
    nav_paths = set()
    for entries in guide_pages.nav_sections(data).values():
        for entry in entries:
            nav_paths.update(entry.values() if isinstance(entry, dict) else [entry])
    record_count = len(data.control_types) + len(data.preprocessors) + len(data.models)
    assert len(nav_paths) == record_count + 3


def test_name_lookup_lists_every_name_file_and_hash(data, pages):
    """The name lookup holds every backend name, file name, other file name and WebUI hash."""
    lookup = pages[guide_pages.NAMES_PAGE]
    expected = []
    for records in (data.control_types, data.preprocessors):
        for record in records.values():
            for backend in guide_pages.BACKEND_LABELS:
                expected.extend(getattr(record.names, backend))
    for model in data.models.values():
        for file in model.files:
            expected.extend([file.filename, *file.also_seen_as])
            if file.webui_hash:
                expected.append(file.webui_hash)
    missing = sorted({name for name in expected if name not in lookup})
    assert not missing


def test_record_links_are_relative_and_resolve(pages):
    """Every token expands, and links are relative to the page holding them."""
    for path, markdown in pages.items():
        assert '[[' not in markdown, path
    canny = pages[guide_pages.model_path('control_v11p_sd15_canny')]
    assert '[Canny](../types/canny.md)' in canny
    assert '../sources.md#cn11_readme' in canny
    for path in pages:
        assert posixpath.normpath(path) == path


def test_unknown_record_link_raises(data):
    """A token naming an unknown record fails, naming the token."""
    with pytest.raises(guide_pages.RecordLinkError, match=r'\[\[model:no_such_model\]\]'):
        guide_pages.expand_record_links('[[type:canny]] and [[model:no_such_model]]', 'controlnet/index.md', data)


def test_link_text_override(data):
    """`[[kind:id|text]]` uses its own link text."""
    expanded = guide_pages.expand_record_links('[[type:canny|Canny maps]]', 'controlnet/choosing.md', data)
    assert expanded == '[Canny maps](types/canny.md)'


def test_drafted_record_lists_its_open_questions(data, pages):
    """A drafted record's page warns and lists each open question."""
    drafted = next(model for model in data.models.values() if model.research_status == 'drafted')
    page = pages[guide_pages.model_path(drafted.id)]
    assert '!!! warning "Drafted record"' in page
    for question in drafted.open_questions:
        assert ' '.join(question.split()) in page


def test_example_image_shows_only_when_present(data, tmp_path):
    """A record page shows an example image only when the guide's docs directory has one."""
    record_id = 'control_v11p_sd15_canny'
    without = guide_pages.model_page(data.models[record_id], data, tmp_path)
    assert '## Example' not in without
    image = tmp_path / guide_pages.EXAMPLES_DIR / 'models' / f'{record_id}.webp'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'')
    with_image = guide_pages.model_page(data.models[record_id], data, tmp_path)
    assert f'![ControlNet 1.1 Canny example](../examples/models/{record_id}.webp)' in with_image
