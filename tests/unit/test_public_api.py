"""The package root's `__all__` is the supported API: every name imports, and the types it uses are exported too."""
import dataclasses
import importlib.metadata
import inspect
import typing

import pytest
from PIL import Image
from pydantic import BaseModel

import sd_backend_client
from sd_backend_client import Backend, GenerationHandle, connect_to_backend

_PACKAGE = 'sd_backend_client'


def _package_classes(hint: object) -> set[type]:
    """Return the package's own classes that appear anywhere in type hint `hint`."""
    found: set[type] = set()
    if isinstance(hint, type) and hint.__module__.startswith(_PACKAGE):
        found.add(hint)
    for arg in typing.get_args(hint):
        found |= _package_classes(arg)
    return found


def _signature_classes(func: object) -> set[type]:
    found: set[type] = set()
    for hint in typing.get_type_hints(func).values():
        found |= _package_classes(hint)
    return found


def _interface_classes() -> set[type]:
    """Package classes used by the backend-agnostic interface: its methods, and the fields of exported types."""
    found = _signature_classes(connect_to_backend)
    for interface in (Backend, GenerationHandle):
        for name, member in inspect.getmembers(interface, inspect.isfunction):
            if not name.startswith('_'):
                found |= _signature_classes(member)
    for name in sd_backend_client.__all__:
        exported = getattr(sd_backend_client, name)
        if isinstance(exported, type) and issubclass(exported, BaseModel):
            for field in exported.model_fields.values():
                found |= _package_classes(field.annotation)
        elif dataclasses.is_dataclass(exported):
            found |= _signature_classes(exported)
    return found


@pytest.mark.parametrize('name', sd_backend_client.__all__)
def test_every_exported_name_imports(name: str) -> None:
    """`from sd_backend_client import <name>` works for each name in `__all__`."""
    namespace: dict[str, object] = {}
    exec(f'from sd_backend_client import {name}', namespace)  # pylint: disable=exec-used
    assert namespace[name] is getattr(sd_backend_client, name)


def test_all_has_no_duplicates() -> None:
    """Each name appears in `__all__` once."""
    assert len(sd_backend_client.__all__) == len(set(sd_backend_client.__all__))


def test_star_import_matches_all() -> None:
    """`from sd_backend_client import *` binds the names in `__all__` and nothing else."""
    namespace: dict[str, object] = {}
    exec('from sd_backend_client import *', namespace)  # pylint: disable=exec-used
    assert set(namespace) - {'__builtins__'} == set(sd_backend_client.__all__)


def test_interface_types_are_exported() -> None:
    """A package class the public interface takes, returns or holds can be imported from the package root."""
    exported = {getattr(sd_backend_client, name) for name in sd_backend_client.__all__}
    missing = {f'{cls.__module__}.{cls.__qualname__}' for cls in _interface_classes() - exported}
    assert not missing, f'add these to sd_backend_client.__all__: {sorted(missing)}'


def test_version_matches_installed_distribution() -> None:
    """`__version__` is exported and reports the installed distribution's version."""
    assert '__version__' in sd_backend_client.__all__
    assert sd_backend_client.__version__ == importlib.metadata.version('sd-backend-client')


_EXPORTED_MODELS = [
    candidate for name in sd_backend_client.__all__
    if isinstance(candidate := getattr(sd_backend_client, name), type) and issubclass(candidate, BaseModel)
]


@pytest.mark.parametrize('model', _EXPORTED_MODELS)
def test_exported_model_fields_have_descriptions(model: type[BaseModel]) -> None:
    """Every field of an exported model carries its attribute docstring as its runtime `description`.

    This needs `use_attribute_docstrings` in the model's config, and pydantic 2.7 or later, the pyproject.toml floor.
    """
    missing = [name for name, field in model.model_fields.items() if not (field.description or '').strip()]
    assert not missing, f'{model.__name__} fields without a docstring: {missing}'


@pytest.mark.parametrize('model', _EXPORTED_MODELS)
def test_exported_models_generate_json_schema(model: type[BaseModel]) -> None:
    """Every exported Pydantic model exposes a JSON schema."""
    schema = model.model_json_schema()
    assert schema['title'] == model.__name__
    assert schema['type'] == 'object'


def test_diffusion_json_schema_omits_python_only_images() -> None:
    """Schema consumers see scalar options, not runtime-only Pillow fields."""
    properties = sd_backend_client.DiffusionParams.model_json_schema()['properties']
    assert {'init_images', 'mask'}.isdisjoint(properties)
    assert properties['steps']['type'] == 'integer'

    properties = sd_backend_client.ControlNetUnit.model_json_schema()['properties']
    assert 'image' not in properties
    assert properties['control_strength']['type'] == 'number'


def test_python_only_images_remain_in_runtime_model_dumps() -> None:
    """Omitting a Pillow field from the JSON schema does not remove it from Python data."""
    image = Image.new('RGB', (2, 2))
    params = sd_backend_client.DiffusionParams(init_images=[image], mask=image)
    dumped = params.model_dump()
    assert dumped['init_images'] == [image]
    assert dumped['mask'] is image
    assert sd_backend_client.ControlNetUnit(image=image).model_dump()['image'] is image
