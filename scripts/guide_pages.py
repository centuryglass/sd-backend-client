"""Markdown for the guide site's generated ControlNet pages, built from the records in `data/controlnet/`.

The guide (`guide/mkdocs.yml`) gets one page per record plus index, name lookup and sources pages, all generated at
build time so the TOML stays the only copy of each fact. This module is pure: it turns loaded records into
`{page path: markdown}` and never imports mkdocs, so unit tests can run it. `scripts/guide_hooks.py` is the MkDocs
glue.

Pages link to records with `[[kind:id]]` or `[[kind:id|link text]]` tokens, kind being `type`, `pre`, `model` or
`source`. `expand_record_links` turns them into relative links and raises `RecordLinkError` on an unknown id, so
hand-written pages use the same tokens and a renamed record fails the build.

Usage: python scripts/guide_pages.py OUT_DIR    # write every generated page under OUT_DIR, for reading the markdown
"""
import posixpath
import re
import sys
from pathlib import Path
from typing import Callable

# scripts/ isn't a package; the guide hook and the unit tests put it on sys.path before importing this module.
from controlnet_data import (  # pylint: disable=import-error
    ControlNetData, ControlType, Model, Names, Preprocessor, all_errors, load_all)

SECTION = 'controlnet'
TYPES_DIR = f'{SECTION}/types'
PREPROCESSORS_DIR = f'{SECTION}/preprocessors'
MODELS_DIR = f'{SECTION}/models'
NAMES_PAGE = f'{SECTION}/names.md'
SOURCES_PAGE = f'{SECTION}/sources.md'

# Example images from the example-images issue (#184) go at `<EXAMPLES_DIR>/<kind dir>/<id>.<ext>` inside the guide's
# docs_dir. A record page shows its image when one exists and renders without one otherwise.
EXAMPLES_DIR = f'{SECTION}/examples'
EXAMPLE_EXTENSIONS = ('webp', 'png', 'jpg')

OPEN_QUESTIONS_ISSUE = 'https://github.com/centuryglass/sd-backend-client/issues/176'
REPO_BLOB = 'https://github.com/centuryglass/sd-backend-client/blob/main'

BASE_MODEL_LABELS = {'sd15': 'SD 1.5', 'sdxl': 'SDXL'}
ADAPTER_LABELS = {
    'controlnet': 'ControlNet',
    't2i_adapter': 'T2I-Adapter',
    'control_lora': 'Control-LoRA',
    'controlnet_lllite': 'ControlNet-LLLite',
    'ip_adapter': 'IP-Adapter',
    'union': 'Union ControlNet',
    'other': 'Other',
}
BACKEND_LABELS = {'webui': 'A1111 WebUI (sd-webui-controlnet)', 'forge': 'Forge, reForge, Forge Neo',
                  'comfyui': 'ComfyUI'}
DEVICE_LABELS = {
    'cpu': 'CPU',
    'gpu': 'GPU',
    'gpu_preferred': 'GPU preferred, runs on CPU more slowly',
    'none': 'No neural network: runs on CPU',
}
SOURCE_KIND_LABELS = {
    'official_docs': 'Official documentation',
    'model_card': 'Model card',
    'paper': 'Paper',
    'repo_code': 'Repository code',
    'tutorial': 'Tutorial',
    'local_code': 'Backend source code',
}

# The record page directory for each `[[kind:id]]` kind.
KIND_DIRS = {'type': TYPES_DIR, 'pre': PREPROCESSORS_DIR, 'model': MODELS_DIR}
LINK_PATTERN = re.compile(r'\[\[(type|pre|model|source):([a-z0-9_]+)(?:\|([^\]]+))?\]\]')


class RecordLinkError(ValueError):
    """A `[[kind:id]]` token names a record or source that doesn't exist."""


def type_path(record_id: str) -> str:
    """Page path of a control type."""
    return f'{TYPES_DIR}/{record_id}.md'


def preprocessor_path(record_id: str) -> str:
    """Page path of a preprocessor."""
    return f'{PREPROCESSORS_DIR}/{record_id}.md'


def model_path(record_id: str) -> str:
    """Page path of a model."""
    return f'{MODELS_DIR}/{record_id}.md'


def expand_record_links(markdown: str, page_path: str, data: ControlNetData) -> str:
    """Replace each `[[kind:id]]` token with a markdown link relative to `page_path`, labelled with the record's label.

    Raises `RecordLinkError` naming every token whose id is unknown.
    """
    unknown: list[str] = []

    def replace(match: re.Match) -> str:
        kind, record_id, text = match.groups()
        if kind == 'source':
            source = data.sources.get(record_id)
            if source is None:
                unknown.append(match.group(0))
                return match.group(0)
            target = f'{SOURCES_PAGE}#{record_id}'
            label = text or source.title
        else:
            records = {'type': data.control_types, 'pre': data.preprocessors, 'model': data.models}[kind]
            record = records.get(record_id)
            if record is None:
                unknown.append(match.group(0))
                return match.group(0)
            target = f'{KIND_DIRS[kind]}/{record_id}.md'
            label = text or record.label
        return f'[{label}]({_relative(target, page_path)})'

    result = LINK_PATTERN.sub(replace, markdown)
    if unknown:
        raise RecordLinkError(f'{page_path}: unknown record link(s) {", ".join(unknown)}')
    return result


def _relative(target: str, page_path: str) -> str:
    """`target` as a link relative to the page at `page_path`, keeping any `#anchor`."""
    path, _, anchor = target.partition('#')
    relative = posixpath.relpath(path, posixpath.dirname(page_path) or '.')
    return f'{relative}#{anchor}' if anchor else relative


def _cell(text: str) -> str:
    """Text made safe for one markdown table cell."""
    return ' '.join(text.split()).replace('|', '\\|')


def _code(name: str) -> str:
    """A name as inline code, safe in a table cell."""
    return f'`{name}`'.replace('|', '\\|')


def _size(size_bytes: int | None) -> str:
    """A file size in MB or GB, or empty when unknown."""
    if size_bytes is None:
        return ''
    if size_bytes >= 1024 ** 3:
        return f'{size_bytes / 1024 ** 3:.2f} GB'
    return f'{size_bytes / 1024 ** 2:.1f} MB'


def _base_models(ids: list[str]) -> str:
    return ', '.join(BASE_MODEL_LABELS[base] for base in ids)


def _paragraphs(text: str) -> str:
    """Record prose, whose paragraphs are separated by blank lines, as markdown."""
    return '\n\n'.join(' '.join(paragraph.split()) for paragraph in text.split('\n\n') if paragraph.strip())


def _status_note(record, kind_name: str) -> list[str]:
    """An admonition for a record that isn't verified, listing its open questions."""
    if record.research_status == 'verified':
        return []
    lines = ['!!! warning "Drafted record"',
             f'    Parts of this {kind_name} are not yet confirmed by a source. The open questions are tracked in '
             f'[issue #176]({OPEN_QUESTIONS_ISSUE}):', '']
    lines.extend(f'    - {" ".join(question.split())}' for question in record.open_questions)
    return lines + ['']


def _example(kind_dir: str, record_id: str, label: str, docs_dir: Path | None) -> list[str]:
    """The record's example image, when `docs_dir` holds one, as a section."""
    if docs_dir is None:
        return []
    for extension in EXAMPLE_EXTENSIONS:
        image = f'{EXAMPLES_DIR}/{posixpath.basename(kind_dir)}/{record_id}.{extension}'
        if (docs_dir / image).is_file():
            page = f'{kind_dir}/{record_id}.md'
            return ['## Example', '', f'![{label} example]({_relative(image, page)})', '']
    return []


def _sources(source_ids: list[str]) -> list[str]:
    """A record's sources section."""
    if not source_ids:
        return []
    return ['## Sources', ''] + [f'- [[source:{source_id}]]' for source_id in source_ids] + ['']


def _names_table(names: Names, name_parameters: dict | None = None) -> list[str]:
    """Each backend's names for a control type or preprocessor, as a table."""
    rows = []
    for backend, label in BACKEND_LABELS.items():
        backend_names = getattr(names, backend)
        if not backend_names:
            continue
        cell = ', '.join(_code(name) for name in backend_names)
        parameters = (name_parameters or {}).get(backend)
        if parameters:
            cell += ' with ' + ', '.join(_code(f'{key} = {value}') for key, value in parameters.items())
        rows.append(f'| {label} | {cell} |')
    if not rows:
        return []
    return ['## Names in your backend', '', '| Backend | Name |', '|---|---|'] + rows + ['']


def _record_rows(ids: list[str], records: dict, columns: list[tuple[str, Callable]]) -> list[str]:
    """A table with one row per record id, its first column linking to the record."""
    header = '| ' + ' | '.join(title for title, _ in columns) + ' |'
    rule = '|' + '---|' * len(columns)
    lines = [header, rule]
    for record_id in ids:
        record = records[record_id]
        lines.append('| ' + ' | '.join(_cell(render(record)) for _, render in columns) + ' |')
    return lines + ['']


def _type_link(record: ControlType) -> str:
    return f'[[type:{record.id}]]'


def _pre_link(record: Preprocessor) -> str:
    return f'[[pre:{record.id}]]'


def _model_link(record: Model) -> str:
    return f'[[model:{record.id}]]'


PREPROCESSOR_COLUMNS: list[tuple[str, Callable]] = [
    ('Preprocessor', _pre_link),
    ('What it does', lambda record: record.summary),
]
MODEL_COLUMNS: list[tuple[str, Callable]] = [
    ('Model', _model_link),
    ('Base', lambda record: _base_models(record.base_models)),
    ('Kind', lambda record: ADAPTER_LABELS[record.adapter]),
    ('What it does', lambda record: record.summary),
]


def control_type_page(record: ControlType, data: ControlNetData, docs_dir: Path | None = None) -> str:
    """The page for one control type."""
    lines = [f'# {record.label}', '', f'**{record.summary}**', '']
    lines += _status_note(record, 'control type')
    if record.base_models:
        lines += [f'**Base models:** {_base_models(record.base_models)}', '']
    lines += [_paragraphs(record.description), '']
    if record.control_image:
        lines += ['## Control image', '', ' '.join(record.control_image.split()), '']
    lines += _example(TYPES_DIR, record.id, record.label, docs_dir)
    if record.uses:
        lines += ['## What to use it for', '']
        for kind, heading in (('standard', 'Everyday uses'), ('creative', 'Creative uses')):
            uses = [use for use in record.uses if use.kind == kind]
            if uses:
                lines += [f'### {heading}', '']
                for use in uses:
                    lines += [f'**{use.title}.** {" ".join(use.description.split())}', '']
    if record.how_it_works:
        lines += ['## How it works', '', _paragraphs(record.how_it_works), '']
    choosing = record.choosing
    if choosing.guide or choosing.preprocessors or choosing.models:
        lines += ['## Choosing a preprocessor and model', '']
        if choosing.guide:
            lines += [_paragraphs(choosing.guide), '']
        if choosing.preprocessors:
            lines += ['### Preprocessors, in recommended order', '']
            lines += _record_rows(choosing.preprocessors, data.preprocessors, PREPROCESSOR_COLUMNS)
        if choosing.models:
            lines += ['### Models, in recommended order', '']
            lines += _record_rows(choosing.models, data.models, MODEL_COLUMNS)
    lines += _names_table(record.names)
    lines += _sources(record.sources)
    return '\n'.join(lines)


def _parameter_value(value) -> str:
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parameters(record: Preprocessor) -> list[str]:
    """A preprocessor's settings: a summary table, then each setting's effect and backend names."""
    if not record.parameters:
        return []
    lines = ['## Settings', '', '| Setting | Default | Range | When to adjust |', '|---|---|---|---|']
    for parameter in record.parameters:
        if parameter.range:
            allowed = ' to '.join(_parameter_value(bound) for bound in parameter.range)
        elif parameter.choices:
            allowed = ', '.join(_code(_parameter_value(choice)) for choice in parameter.choices)
        elif parameter.type == 'bool':
            allowed = 'on, off'
        else:
            allowed = ''
        default = '' if parameter.default is None else _code(_parameter_value(parameter.default))
        lines.append(f'| {_cell(parameter.label)} | {default} | {_cell(allowed)} | '
                     f'{_cell(parameter.when_to_adjust)} |')
    lines.append('')
    for parameter in record.parameters:
        lines += [f'### {parameter.label}', '']
        text = ' '.join(part for part in (parameter.summary, parameter.effect) if part)
        if text:
            lines += [' '.join(text.split()), '']
        if parameter.names:
            names = '; '.join(f'{BACKEND_LABELS[backend]}: {_code(name)}'
                              for backend, name in parameter.names.items())
            lines += [f'**Names:** {names}', '']
    return lines


def _hardware(record: Preprocessor) -> list[str]:
    """What a preprocessor needs to run, and the files it downloads."""
    hardware = record.hardware
    facts = []
    if hardware.device:
        facts.append(f'- **Runs on:** {DEVICE_LABELS[hardware.device]}')
    if hardware.vram:
        facts.append(f'- **Memory:** {" ".join(hardware.vram.split())}')
    if hardware.speed:
        facts.append(f'- **Speed:** {" ".join(hardware.speed.split())}')
    if not facts and not hardware.files:
        return []
    lines = ['## Requirements', ''] + facts + ['']
    if hardware.files:
        lines += ['Downloaded on first use:', '', '| File | Variant | Size | Download |', '|---|---|---|---|']
        for file in hardware.files:
            link = f'[link]({file.source_url})' if file.source_url else ''
            lines.append(f'| {_code(file.filename)} | {_cell(file.variant)} | {_size(file.size_bytes)} | {link} |')
        lines.append('')
    return lines


def preprocessor_page(record: Preprocessor, data: ControlNetData, docs_dir: Path | None = None) -> str:
    """The page for one preprocessor."""
    lines = [f'# {record.label}', '', f'**{record.summary}**', '']
    lines += _status_note(record, 'preprocessor')
    if record.control_types:
        types = ', '.join(f'[[type:{type_id}]]' for type_id in record.control_types)
        lines += [f'**Control type:** {types}', '']
    lines += [_paragraphs(record.description), '']
    facts = [('Effect', record.effect), ('Output', record.output), ('Works best on', record.best_input),
             ('Struggles with', record.weak_input)]
    facts = [(label, text) for label, text in facts if text]
    if facts:
        lines += [f'- **{label}:** {" ".join(text.split())}' for label, text in facts] + ['']
    lines += _example(PREPROCESSORS_DIR, record.id, record.label, docs_dir)
    if record.how_it_works:
        lines += ['## How it works', '', _paragraphs(record.how_it_works), '']
    lines += _parameters(record)
    lines += _hardware(record)
    if record.models:
        lines += ['## Models that read its maps', '']
        lines += _record_rows(_ordered_models(record.models, data), data.models, MODEL_COLUMNS)
    lines += _names_table(record.names, record.name_parameters)
    lines += _sources(record.sources)
    return '\n'.join(lines)


def _ordered_models(model_ids: list[str], data: ControlNetData) -> list[str]:
    """Model ids with SD 1.5 models first, keeping the record's order within each base model."""
    return sorted(model_ids, key=lambda model_id: 'sd15' not in data.models[model_id].base_models)


def _webui_name(filename: str, webui_hash: str) -> str:
    """The name the WebUI ControlNet extensions show for a model file: its stem and short hash."""
    stem = filename.rsplit('.', 1)[0]
    return f'{stem} [{webui_hash}]' if webui_hash else stem


def model_page(record: Model, data: ControlNetData, docs_dir: Path | None = None) -> str:
    """The page for one model."""
    lines = [f'# {record.label}', '', f'**{record.summary}**', '']
    lines += _status_note(record, 'model')
    facts = [('Base model', _base_models(record.base_models)), ('Kind', ADAPTER_LABELS[record.adapter])]
    if record.control_types:
        facts.append(('Control type', ', '.join(f'[[type:{type_id}]]' for type_id in record.control_types)))
    if record.trainer:
        facts.append(('Trained by', record.trainer))
    if record.license:
        facts.append(('License', _code(record.license)))
    if record.homepage:
        facts.append(('Homepage', f'<{record.homepage}>'))
    lines += [f'- **{label}:** {value}' for label, value in facts] + ['']
    lines += [_paragraphs(record.description), '']
    lines += _example(MODELS_DIR, record.id, record.label, docs_dir)
    for heading, items in (('Good for', record.good_for), ('Strengths', record.strengths),
                           ('Weaknesses', record.weaknesses)):
        if items:
            lines += [f'## {heading}', ''] + [f'- {" ".join(item.split())}' for item in items] + ['']
    if record.how_it_works:
        lines += ['## How it works', '', _paragraphs(record.how_it_works), '']
    if record.preprocessors:
        lines += ['## Preprocessors', '']
        lines += _record_rows(record.preprocessors, data.preprocessors, PREPROCESSOR_COLUMNS)
    if record.files:
        lines += ['## Files', '', '| File | Variant | Precision | Size | WebUI hash | Download |',
                  '|---|---|---|---|---|---|']
        for file in record.files:
            link = f'[link]({file.source_url})' if file.source_url else ''
            webui_hash = _code(file.webui_hash) if file.webui_hash else ''
            lines.append(f'| {_code(file.filename)} | {_cell(file.variant)} | {file.precision or ""} | '
                         f'{_size(file.size_bytes)} | {webui_hash} | {link} |')
        lines.append('')
        hashes = [file for file in record.files if file.sha256]
        if hashes:
            lines += ['??? note "SHA-256 checksums"', '']
            lines += [f'    - {_code(file.filename)}: {_code(file.sha256)}' for file in hashes] + ['']
        lines += ['## Names in your backend', '',
                  'A1111 WebUI, Forge and reForge show the file name without its extension, followed by the '
                  'WebUI hash. Forge Neo and ComfyUI show the file name.', '',
                  '| File | A1111 WebUI, Forge, reForge |', '|---|---|']
        for file in record.files:
            for filename in [file.filename, *file.also_seen_as]:
                lines.append(f'| {_code(filename)} | {_code(_webui_name(filename, file.webui_hash))} |')
        lines.append('')
    lines += _sources(record.sources)
    return '\n'.join(lines)


def _by_label(records: dict) -> list[str]:
    return sorted(records, key=lambda record_id: records[record_id].label.lower())


def control_types_index(data: ControlNetData) -> str:
    """The list of every control type."""
    lines = ['# Control types', '',
             'A control type is one kind of guidance: what a ControlNet keeps from your control image. Each page '
             'explains what the type does, what to use it for, and which preprocessors and models belong to it.',
             '', '| Control type | What it keeps | Base models |', '|---|---|---|']
    for record_id in _by_label(data.control_types):
        record = data.control_types[record_id]
        lines.append(f'| [[type:{record_id}]] | {_cell(record.summary)} | {_base_models(record.base_models)} |')
    return '\n'.join(lines + [''])


def preprocessors_index(data: ControlNetData) -> str:
    """Every preprocessor, grouped by control type."""
    lines = ['# Preprocessors', '',
             'A preprocessor turns your image into the control image a model reads, such as an edge map or a '
             'pose skeleton. They are grouped here by control type; a preprocessor that serves several types '
             'appears under each.', '']
    for type_id in _by_label(data.control_types):
        control_type = data.control_types[type_id]
        if not control_type.choosing.preprocessors:
            continue
        lines += [f'## {control_type.label}', '']
        lines += _record_rows(control_type.choosing.preprocessors, data.preprocessors, PREPROCESSOR_COLUMNS)
    return '\n'.join(lines)


def models_index(data: ControlNetData) -> str:
    """Every model, grouped by control type."""
    lines = ['# Models', '',
             'A model is the trained network that reads a control image during generation. Each one is built '
             'for one base model, SD 1.5 or SDXL, and works only with checkpoints of that base. They are grouped '
             'here by control type, in recommended order; a model that covers several types appears under each.',
             '']
    for type_id in _by_label(data.control_types):
        control_type = data.control_types[type_id]
        if not control_type.choosing.models:
            continue
        lines += [f'## {control_type.label}', '']
        lines += _record_rows(control_type.choosing.models, data.models, MODEL_COLUMNS)
    return '\n'.join(lines)


def name_lookup_rows(data: ControlNetData) -> list[tuple[str, str, str]]:
    """(name, where it appears, record token) for every backend name, file name and WebUI model name."""
    rows: list[tuple[str, str, str]] = []
    for kind, token, records in (('control type', 'type', data.control_types),
                                 ('preprocessor', 'pre', data.preprocessors)):
        for record_id in _by_label(records):
            record = records[record_id]
            for backend, label in BACKEND_LABELS.items():
                for name in getattr(record.names, backend):
                    rows.append((name, f'{label} {kind}', f'[[{token}:{record_id}]]'))
    for record_id in _by_label(data.models):
        for file in data.models[record_id].files:
            for filename in [file.filename, *file.also_seen_as]:
                where = 'model file' if filename == file.filename else 'model file, other name'
                rows.append((filename, where, f'[[model:{record_id}]]'))
                if file.webui_hash:
                    rows.append((_webui_name(filename, file.webui_hash), 'A1111 WebUI, Forge, reForge model',
                                 f'[[model:{record_id}]]'))
    return rows


def name_lookup_page(data: ControlNetData) -> str:
    """One filterable table of every name a backend shows, mapping it to its record."""
    lines = ['# Name lookup', '',
             'Every name a backend shows for a control type, preprocessor or model, and every model file name, '
             'with the record it belongs to. The WebUI model names end in the short hash the WebUI ControlNet '
             'extensions compute from the file, as in `control_v11p_sd15_canny [d14c016b]`. Type part of a name, '
             'a file name or a hash to filter the table.', '',
             '<input type="search" class="guide-table-filter" placeholder="Filter names" '
             'aria-label="Filter names">', '',
             '| Name | Where it appears | Record |', '|---|---|---|']
    for name, where, token in sorted(name_lookup_rows(data), key=lambda row: row[0].lower()):
        lines.append(f'| {_code(name)} | {where} | {token} |')
    return '\n'.join(lines + [''])


def sources_page(data: ControlNetData) -> str:
    """Every cited source, each with an anchor named by its id."""
    lines = ['# Sources', '',
             'Every source the ControlNet records cite. Each record page lists its own sources, which link here. '
             'Backend source code is pinned to the commit that was read.', '']
    for source_id in sorted(data.sources, key=lambda source_id: data.sources[source_id].title.lower()):
        source = data.sources[source_id]
        title = f'[{source.title}]({source.url})' if source.url else source.title
        details = [SOURCE_KIND_LABELS[source.kind]]
        if source.kind == 'local_code':
            details.append(f'{source.project} at `{source.commit[:12]}`, `{source.path}`')
        details.append(f'accessed {source.accessed}')
        lines += [f'<span id="{source_id}"></span>', f'- {title}  ', f'  {"; ".join(details)}', '']
    return '\n'.join(lines)


def generate_pages(data: ControlNetData, docs_dir: Path | None = None) -> dict[str, str]:
    """Every generated page's markdown by path, with `[[kind:id]]` tokens still unexpanded.

    `docs_dir` is the guide's docs directory, searched for example images; without it no page shows one.
    """
    pages = {
        f'{TYPES_DIR}/index.md': control_types_index(data),
        f'{PREPROCESSORS_DIR}/index.md': preprocessors_index(data),
        f'{MODELS_DIR}/index.md': models_index(data),
        NAMES_PAGE: name_lookup_page(data),
        SOURCES_PAGE: sources_page(data),
    }
    for record_id, control_type in data.control_types.items():
        pages[type_path(record_id)] = control_type_page(control_type, data, docs_dir)
    for record_id, preprocessor in data.preprocessors.items():
        pages[preprocessor_path(record_id)] = preprocessor_page(preprocessor, data, docs_dir)
    for record_id, model in data.models.items():
        pages[model_path(record_id)] = model_page(model, data, docs_dir)
    return pages


def nav_sections(data: ControlNetData) -> dict[str, list]:
    """MkDocs nav entries for each generated directory, keyed by the directory with a trailing slash."""
    sections = {}
    for directory, records, path_of in ((TYPES_DIR, data.control_types, type_path),
                                        (PREPROCESSORS_DIR, data.preprocessors, preprocessor_path),
                                        (MODELS_DIR, data.models, model_path)):
        entries: list = [f'{directory}/index.md']
        entries += [{records[record_id].label: path_of(record_id)} for record_id in _by_label(records)]
        sections[f'{directory}/'] = entries
    return sections


def load_checked() -> ControlNetData:
    """Load every record and run the cross-record checks, raising `ValueError` listing any errors."""
    data = load_all()
    errors = all_errors(data)
    if errors:
        raise ValueError('data/controlnet/ has errors:\n' + '\n'.join(errors))
    return data


def main() -> int:
    """Write every generated page, links expanded, under the directory given as the only argument."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1])
    data = load_checked()
    for path, markdown in generate_pages(data).items():
        target = out_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(expand_record_links(markdown, path, data), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
