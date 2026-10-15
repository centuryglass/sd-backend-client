"""Describes a setting a backend defines at runtime, so callers can build controls for it and validate its values.

ControlNet preprocessors and soft inpainting publish their settings as `ParameterDef` lists. Callers send the chosen
values back as a dict keyed by `ParameterDef.key`, which `validate_parameter_values` checks against the definitions.
"""
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel


def _value_type_matches(value: Any, default: Any) -> bool:
    """Whether a provided parameter value is type-compatible with its default.

    `bool` is a subclass of `int`, so it's checked exactly (a bool is never a valid int value and
    vice versa); other ints and floats are treated as interchangeable numbers.
    """
    if isinstance(value, bool) or isinstance(default, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float))
    return isinstance(value, type(default))


class ParameterDef(BaseModel):
    """Defines one setting a backend accepts: a ControlNet preprocessor parameter or a soft inpainting setting."""
    key: str
    default_value: str|int|float|bool
    """Default for the parameter, sent whenever the caller leaves it unset (see `required`). Every parameter carries
       one so there's always a type to validate provided values against, too."""
    display_name: str = ''
    """Short label for the parameter in a UI. Empty when the backend provides none (ComfyUI); fall back to `key`."""
    description: str = ''
    """Tooltip-style explanation of the parameter. Empty when the backend provides none (WebUI)."""
    required: bool = False
    """Whether the backend's schema marks this parameter as required rather than optional.

       `PreprocessorParams.build_params_from_defaults` fills every parameter's default regardless of this flag: some
       backend nodes (e.g. ComfyUI's `comfyui_controlnet_aux` `TilePreprocessor`) don't apply their own declared
       default when an optional input is left out of the request entirely, so leaving it unset there isn't safe."""

    option_list: Optional[list[str|int|float|bool]] = None
    """List of valid options. If None, all options are valid."""

    min_val: Optional[int|float] = None
    """Minimum value, ignored if non-numeric. If unset, value has no minimum bounds."""

    max_val: Optional[int|float] = None
    """Maximum value, ignored if non-numeric. If unset, value has no maximum bounds."""

    step_val: Optional[int|float] = None
    """Smallest value change, ignored if non-numeric."""


def validate_parameter_values(parameters: Sequence[ParameterDef], values: Mapping[str, Any], owner: str) -> None:
    """Raise `ValueError` unless every entry in `values` matches a definition in `parameters`.

    Each value must have its definition's default type, be one of its `option_list` if it has one, and lie within its
    `min_val` and `max_val`. Keys absent from `values` are not checked. `owner` names the parameters' owner in the
    error message.
    """
    expected_keys = set()
    for param_def in parameters:
        expected_keys.add(param_def.key)
        if param_def.key not in values:
            continue
        value = values[param_def.key]
        if not _value_type_matches(value, param_def.default_value):
            raise ValueError(f"{owner}.{param_def.key}: expected {type(param_def.default_value).__name__}, found "
                             f"{type(value).__name__}")
        if param_def.option_list is not None and value not in param_def.option_list:
            raise ValueError(f"{owner}.{param_def.key}: invalid option {value}, expected selection from "
                             f"{param_def.option_list}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if param_def.max_val is not None and value > param_def.max_val:
                raise ValueError(f"{owner}.{param_def.key}: invalid value {value}, expected value <= "
                                 f"{param_def.max_val}")
            if param_def.min_val is not None and value < param_def.min_val:
                raise ValueError(f"{owner}.{param_def.key}: invalid value {value}, expected value >= "
                                 f"{param_def.min_val}")
    for key in values:
        if key not in expected_keys:
            raise ValueError(f"{owner}: unexpected {key} param, expected one of {sorted(expected_keys)}")
