"""Schema and loader for the ControlNet reference data in `data/controlnet/`.

The data is TOML, one file per record, outside the package (`data/controlnet/README.md` describes the layout and the
writing rules). These pydantic models are its schema: every model forbids unknown keys, so a misspelled field fails
to load instead of vanishing. `load_all` parses and validates every file; the `*_errors` functions run the checks
that span records, which `tests/unit/test_controlnet_data.py` asserts are empty.

Usage: python scripts/controlnet_data.py    # load everything and print a status count, or the errors
"""
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, \
    model_validator

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'controlnet'

LABEL_MAX_CHARS = 32
SUMMARY_MAX_CHARS = 160
# The README promises "~120 words" for the prose fields; the check allows some slack over that target.
PROSE_MAX_WORDS = 150

# Characters the writing rules exclude from every text field: em and en dashes.
FORBIDDEN_CHARACTERS = ('—', '–')

RecordId: TypeAlias = Annotated[str, StringConstraints(pattern=r'^[a-z0-9]+(_[a-z0-9]+)*$')]
BaseModelId: TypeAlias = Literal['sd15', 'sdxl']
ResearchStatus: TypeAlias = Literal['stub', 'drafted', 'verified']
Scalar: TypeAlias = str | int | float | bool
Sha256: TypeAlias = Annotated[str, StringConstraints(pattern=r'^([0-9a-f]{64})?$')]
WebuiHash: TypeAlias = Annotated[str, StringConstraints(pattern=r'^([0-9a-f]{8})?$')]

# The `names` keys. `forge` covers Forge, reForge and Forge Neo, which share one ControlNet extension.
BACKENDS = ('webui', 'forge', 'comfyui')


class _Strict(BaseModel):
    """Base for every record model: unknown keys fail validation."""
    model_config = ConfigDict(extra='forbid')


class Names(_Strict):
    """Every name each backend uses for a control type or preprocessor.

    Lists, since one backend can expose several aliases. For a preprocessor, `webui` holds sd-webui-controlnet's
    module name and then its UI label where they differ, and `comfyui` holds node class names. A control type's
    `comfyui` list is empty, since ComfyUI has no control types.
    """
    webui: list[str] = []
    forge: list[str] = []
    comfyui: list[str] = []


class _Record(_Strict):
    """Fields every record kind shares: id, the four text tiers, sources and research status."""
    id: RecordId
    label: str = Field(min_length=1, max_length=LABEL_MAX_CHARS)
    summary: str = Field('', max_length=SUMMARY_MAX_CHARS)
    description: str = ''
    how_it_works: str = ''
    sources: list[str] = []
    research_status: ResearchStatus = 'stub'
    open_questions: list[str] = []

    @field_validator('label', 'summary')
    @classmethod
    def _single_line(cls, value: str) -> str:
        if '\n' in value:
            raise ValueError('must be a single line')
        if value != value.strip():
            raise ValueError('must not start or end with whitespace')
        return value

    @field_validator('label')
    @classmethod
    def _no_trailing_period(cls, value: str) -> str:
        if value.endswith('.'):
            raise ValueError('a label has no trailing period')
        return value

    @field_validator('description', 'how_it_works')
    @classmethod
    def _word_limit(cls, value: str) -> str:
        if len(value.split()) > PROSE_MAX_WORDS:
            raise ValueError(f'over {PROSE_MAX_WORDS} words')
        return value

    @model_validator(mode='after')
    def _prose_required_past_stub(self) -> '_Record':
        if self.research_status != 'stub':
            missing = [name for name in ('summary', 'description', 'how_it_works') if not getattr(self, name)]
            if missing:
                raise ValueError(f'a {self.research_status} record needs {", ".join(missing)}')
        return self


class Use(_Strict):
    """One thing an artist does with a control type."""
    kind: Literal['standard', 'creative']
    title: str
    description: str


class Choosing(_Strict):
    """How to pick a preprocessor and model within a control type, with each list in recommended order."""
    guide: str = ''
    preprocessors: list[RecordId] = []
    models: list[RecordId] = []


class ControlType(_Record):
    """One kind of control, such as Depth or OpenPose: what it keeps from the control image and how to use it."""
    base_models: list[BaseModelId] = []
    control_image: str = ''
    names: Names = Names()
    uses: list[Use] = []
    choosing: Choosing = Choosing()


class AnnotatorFile(_Strict):
    """A weights file a preprocessor downloads on first use."""
    filename: str
    size_bytes: Optional[int] = None
    sha256: Sha256 = ''
    source_url: str = ''
    variant: str = ''


class Hardware(_Strict):
    """What a preprocessor needs to run. `vram` and `speed` are qualitative notes, sourced or left empty."""
    device: Optional[Literal['cpu', 'gpu', 'gpu_preferred', 'none']] = None
    vram: str = ''
    speed: str = ''
    files: list[AnnotatorFile] = []


class Parameter(_Strict):
    """One preprocessor setting, with its name on each backend."""
    id: RecordId
    label: str = Field(min_length=1, max_length=LABEL_MAX_CHARS)
    summary: str = Field('', max_length=SUMMARY_MAX_CHARS)
    names: dict[Literal['webui', 'forge', 'comfyui'], str] = {}
    type: Literal['int', 'float', 'bool', 'enum']
    default: Optional[Scalar] = None
    range: Optional[Annotated[list[float], Field(min_length=2, max_length=2)]] = None
    choices: Optional[list[Scalar]] = None
    effect: str = ''
    when_to_adjust: str = ''


class Preprocessor(_Record):
    """One preprocessor: what it extracts from an image, and its name and settings on each backend."""
    control_types: list[RecordId] = []
    effect: str = ''
    output: str = ''
    best_input: str = ''
    weak_input: str = ''
    models: list[RecordId] = []
    names: Names = Names()
    name_parameters: dict[Literal['webui', 'forge', 'comfyui'], dict[str, Scalar]] = {}
    """Parameter values that select this record on a backend where its name is shared with another record.

    ComfyUI folds several WebUI preprocessors into one node with a switch: `LineArtPreprocessor` is both
    `lineart_realistic` and, with `coarse = "enable"`, `lineart_coarse`. The record with the switch set carries it
    here. `name_collision_errors` requires it on every record but one that shares a name."""
    hardware: Hardware = Hardware()
    parameters: list[Parameter] = []


class ModelFile(_Strict):
    """One downloadable copy of a model's weights.

    `webui_hash` is the short hash the WebUI ControlNet extensions append to model names (`name [d14c016b]`): the
    first 8 hex digits of the sha256 of the 64 KiB starting 1 MiB into the file (`modules.sd_models.model_hash`).
    It never matches a prefix of `sha256`, the hash of the whole file.
    """
    filename: str
    also_seen_as: list[str] = []
    format: Literal['pth', 'safetensors', 'ckpt', 'bin']
    precision: Optional[Literal['fp32', 'fp16', 'bf16']] = None
    size_bytes: Optional[int] = None
    sha256: Sha256 = ''
    webui_hash: WebuiHash = ''
    source_url: str = ''
    variant: str = ''


class Model(_Record):
    """One trained network. Its pruned, fp16, renamed or LoRA-extracted copies are `files` entries."""
    base_models: list[BaseModelId] = Field(min_length=1)
    adapter: Literal['controlnet', 't2i_adapter', 'control_lora', 'controlnet_lllite', 'ip_adapter', 'union',
                     'other']
    control_types: list[RecordId] = []
    preprocessors: list[RecordId] = []
    trainer: str = ''
    license: str = ''
    homepage: str = ''
    strengths: list[str] = []
    weaknesses: list[str] = []
    good_for: list[str] = []
    files: list[ModelFile] = []


class Source(_Strict):
    """One cited source. A `local_code` source names a project, its upstream commit and a path inside it."""
    title: str
    url: str = ''
    kind: Literal['official_docs', 'model_card', 'paper', 'repo_code', 'tutorial', 'local_code']
    accessed: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    project: str = ''
    commit: str = ''
    path: str = ''

    @model_validator(mode='after')
    def _local_code_fields(self) -> 'Source':
        if self.kind == 'local_code':
            if not (self.project and self.commit and self.path):
                raise ValueError('a local_code source needs project, commit and path')
            if self.path.startswith(('/', '~')) or re.match(r'^[A-Za-z]:', self.path):
                raise ValueError('path is inside the project, never a local filesystem path')
        return self


@dataclass
class ControlNetData:
    """Every record in `data/controlnet/`, by id."""
    control_types: dict[str, ControlType] = field(default_factory=dict)
    preprocessors: dict[str, Preprocessor] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)


class DataLoadError(Exception):
    """One or more data files failed to parse or validate. The message lists every failure."""


_RECORD_DIRS: dict[str, type[_Record]] = {
    'control_types': ControlType,
    'preprocessors': Preprocessor,
    'models': Model,
}


def load_all(data_dir: Path = DATA_DIR) -> ControlNetData:
    """Parse and validate every data file, raising `DataLoadError` listing every file that fails."""
    data = ControlNetData()
    errors: list[str] = []
    try:
        with open(data_dir / 'sources.toml', 'rb') as file:
            for source_id, source_dict in tomllib.load(file).items():
                try:
                    data.sources[source_id] = Source.model_validate(source_dict)
                except ValidationError as err:
                    errors.append(f'sources.toml [{source_id}]: {err}')
    except (OSError, tomllib.TOMLDecodeError) as err:
        errors.append(f'sources.toml: {err}')
    for dir_name, record_class in _RECORD_DIRS.items():
        records: dict[str, _Record] = getattr(data, dir_name)
        for path in sorted((data_dir / dir_name).glob('*.toml')):
            relative = f'{dir_name}/{path.name}'
            try:
                with open(path, 'rb') as file:
                    record = record_class.model_validate(tomllib.load(file))
            except (OSError, tomllib.TOMLDecodeError, ValidationError) as err:
                errors.append(f'{relative}: {err}')
                continue
            if record.id != path.stem:
                errors.append(f'{relative}: id "{record.id}" does not match the file name')
                continue
            records[record.id] = record
    if errors:
        raise DataLoadError('\n'.join(errors))
    return data


def _all_records(data: ControlNetData) -> list[tuple[str, _Record]]:
    """(kind/id, record) for every record, for error messages."""
    return [(f'{dir_name}/{record_id}', record)
            for dir_name in _RECORD_DIRS
            for record_id, record in getattr(data, dir_name).items()]


def reference_errors(data: ControlNetData) -> list[str]:
    """Every id reference that doesn't resolve to a record of the right kind."""
    errors: list[str] = []

    def check(owner: str, field_name: str, ids: list[str], known: dict) -> None:
        errors.extend(f'{owner}: {field_name} names unknown id "{ref}"' for ref in ids if ref not in known)

    for owner, record in _all_records(data):
        check(owner, 'sources', record.sources, data.sources)
        if isinstance(record, ControlType):
            check(owner, 'choosing.preprocessors', record.choosing.preprocessors, data.preprocessors)
            check(owner, 'choosing.models', record.choosing.models, data.models)
        elif isinstance(record, Preprocessor):
            check(owner, 'control_types', record.control_types, data.control_types)
            check(owner, 'models', record.models, data.models)
        elif isinstance(record, Model):
            check(owner, 'control_types', record.control_types, data.control_types)
            check(owner, 'preprocessors', record.preprocessors, data.preprocessors)
    return errors


def status_errors(data: ControlNetData) -> list[str]:
    """Every `verified` record without a non-tutorial source, or with open questions left."""
    errors: list[str] = []
    for owner, record in _all_records(data):
        if record.research_status != 'verified':
            continue
        if record.open_questions:
            errors.append(f'{owner}: verified, but has open_questions')
        if not any(data.sources[ref].kind != 'tutorial' for ref in record.sources if ref in data.sources):
            errors.append(f'{owner}: verified, but has no non-tutorial source')
    return errors


def text_errors(data: ControlNetData) -> list[str]:
    """Every record whose text contains a character the writing rules exclude."""
    errors: list[str] = []
    for owner, record in _all_records(data):
        text = record.model_dump_json()
        for character in FORBIDDEN_CHARACTERS:
            if character in text:
                errors.append(f'{owner}: contains {character!r}; use ASCII hyphens')
    return errors


def name_collision_errors(data: ControlNetData) -> list[str]:
    """Every backend name claimed by several records of one kind without `name_parameters` telling them apart.

    Records sharing a name must all set `name_parameters` for that backend, except at most one (the node's default
    behavior).
    """
    errors: list[str] = []
    for kind in ('control_types', 'preprocessors'):
        for backend in BACKENDS:
            claims: dict[str, list[_Record]] = {}
            for record in getattr(data, kind).values():
                for name in getattr(record.names, backend):
                    claims.setdefault(name, []).append(record)
            for name, records in claims.items():
                if len(records) < 2:
                    continue
                unparameterized = [record.id for record in records
                                   if not getattr(record, 'name_parameters', {}).get(backend)]
                if len(unparameterized) > 1:
                    errors.append(f'{kind}: {backend} name "{name}" is claimed by {", ".join(unparameterized)}, '
                                  'and only one of them may omit name_parameters')
    return errors


def all_errors(data: ControlNetData) -> list[str]:
    """Every cross-record check's errors."""
    return reference_errors(data) + status_errors(data) + text_errors(data) + name_collision_errors(data)


def main() -> int:
    """Load everything and print a status count, or every error."""
    try:
        data = load_all()
    except DataLoadError as err:
        print(err)
        return 1
    errors = all_errors(data)
    for error in errors:
        print(error)
    for owner_kind in _RECORD_DIRS:
        records = getattr(data, owner_kind).values()
        counts = {status: sum(1 for record in records if record.research_status == status)
                  for status in ('stub', 'drafted', 'verified')}
        print(f'{owner_kind}: {len(records)} records, '
              + ', '.join(f'{count} {status}' for status, count in counts.items()))
    print(f'sources: {len(data.sources)}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
