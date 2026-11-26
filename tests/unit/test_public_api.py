"""The package root's `__all__` is the supported API: every name imports, and the types it uses are exported too."""
import dataclasses
import importlib.metadata
import inspect
import typing

import pytest
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
