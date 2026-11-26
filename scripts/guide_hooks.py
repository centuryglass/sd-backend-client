"""MkDocs hooks for the guide site (`guide/mkdocs.yml`): add the generated ControlNet pages and expand record links.

The pages come from `scripts/guide_pages.py`, which builds them from `data/controlnet/`. They exist only in the build,
so the TOML stays the only copy of each fact. The build fails when the records fail to load or a cross-record check
fails, and when any page names an unknown record in a `[[kind:id]]` link.

Nav entries in `guide/mkdocs.yml` that name a generated directory with a trailing slash (`controlnet/models/`) are
replaced with that directory's index and one entry per record.
"""
import sys
from pathlib import Path
from typing import Any

# mkdocs comes from requirements-docs.txt, which the lint job does not install.
from mkdocs.config.defaults import MkDocsConfig  # pylint: disable=import-error
from mkdocs.exceptions import PluginError  # pylint: disable=import-error
from mkdocs.structure.files import File, Files  # pylint: disable=import-error
from mkdocs.structure.pages import Page  # pylint: disable=import-error

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position,import-error
from controlnet_data import ControlNetData, DataLoadError  # noqa: E402
from guide_pages import RecordLinkError, expand_record_links, generate_pages, load_checked, nav_sections  # noqa: E402

_data: list[ControlNetData] = []


def _loaded() -> ControlNetData:
    """The records `on_config` loaded for this build."""
    return _data[-1]


def _expand_nav(entries: list, sections: dict[str, list]) -> list:
    """`entries` with each generated-directory placeholder replaced by that directory's pages."""
    expanded = []
    for entry in entries:
        if isinstance(entry, dict):
            entry = {title: (sections.get(value, value) if isinstance(value, str)
                             else _expand_nav(value, sections))
                     for title, value in entry.items()}
        expanded.append(entry)
    return expanded


def on_config(config: MkDocsConfig, **_kwargs: Any) -> MkDocsConfig:
    """Load and check the records, then fill in the nav's generated sections."""
    try:
        data = load_checked()
    except (DataLoadError, ValueError) as err:
        raise PluginError(str(err)) from err
    _data.clear()
    _data.append(data)
    if config.nav:
        config.nav = _expand_nav(config.nav, nav_sections(data))
    return config


def on_files(files: Files, config: MkDocsConfig, **_kwargs: Any) -> Files:
    """Add a virtual file for every generated page."""
    docs_dir = Path(config.docs_dir)
    for path, markdown in generate_pages(_loaded(), docs_dir).items():
        if files.get_file_from_path(path) is not None:
            raise PluginError(f'{path} is generated from data/controlnet/; remove it from {docs_dir}')
        files.append(File.generated(config, path, content=markdown))
    return files


def on_page_markdown(markdown: str, page: Page, **_kwargs: Any) -> str:
    """Expand `[[kind:id]]` record links on every page, generated or hand-written."""
    try:
        return expand_record_links(markdown, page.file.src_uri, _loaded())
    except RecordLinkError as err:
        raise PluginError(str(err)) from err
