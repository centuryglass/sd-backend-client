"""Defines a ControlNet preprocessor's name and parameters, for use with a Stable Diffusion API"""

from typing import Optional, Any, Self

from pydantic import BaseModel, ConfigDict, model_validator


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
    """Defines a ControlNet preprocessor parameter."""
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


class ControlNetPreprocessor(BaseModel):
    """Defines a ControlNet preprocessor's name and parameters for use with a Stable Diffusion API."""
    model_config = ConfigDict(protected_namespaces=('model_validate', 'model_dump'))

    name: str

    category_name: str = ""
    """Category label for grouping with similar preprocessors."""

    description: str = ""
    """Description string provided by a server API."""

    has_image_input: bool = True
    """Whether this preprocessor accepts an image input when converted into a ComfyUI node."""

    has_mask_input: bool = True
    """Whether this preprocessor accepts a mask input when converted into a ComfyUI node."""

    model_free: bool = False
    """Whether this preprocessor can be used without a ControlNet model."""

    parameters: list[ParameterDef] = []
    """Definitions for all additional parameter types the preprocessor accepts."""

class PreprocessorParams(BaseModel):
    """Selected ControlNet Preprocessor and parameter values."""
    typedef: ControlNetPreprocessor
    parameter_values: dict[str, Any]

    @model_validator(mode='before')
    @classmethod
    def build_params_from_defaults(cls, data: Any) -> Any:
        """Fill in every parameter's default value left unset by the caller, required or optional.

        See `ParameterDef.required` for why optional parameters are defaulted too, rather than left absent.
        """
        if isinstance(data, dict) and "typedef" in data:
            typedef = data['typedef']
            if not isinstance(typedef, ControlNetPreprocessor):
                typedef = ControlNetPreprocessor.model_validate(typedef)
            param_values = data.setdefault('parameter_values', {})
            for param_def in typedef.parameters:
                param_values.setdefault(param_def.key, param_def.default_value)
        return data

    @model_validator(mode='after')
    def validate_parameters(self) -> Self:
        expected_keys = set()
        for param_def in self.typedef.parameters:
            expected_keys.add(param_def.key)
            if param_def.key not in self.parameter_values:
                continue  # no value provided for this parameter
            value = self.parameter_values[param_def.key]
            if not _value_type_matches(value, param_def.default_value):
                raise ValueError(f"Preprocessor {self.typedef.name}.{param_def.key}: expected "
                                 f"{type(param_def.default_value).__name__}, found {type(value).__name__}")
            if param_def.option_list is not None and value not in param_def.option_list:
                raise ValueError(f"Preprocessor {self.typedef.name}.{param_def.key}: invalid option {value}, expected "
                                 f"selection from {param_def.option_list}")
            if (param_def.min_val is not None or param_def.max_val is not None) and isinstance(value, (int, float)):
                if param_def.max_val is not None and value > param_def.max_val:
                    raise ValueError( f"Preprocessor {self.typedef.name}.{param_def.key}: invalid value {value}, "
                                      f"expected value <= {param_def.max_val}")
                if param_def.min_val is not None and value < param_def.min_val:
                    raise ValueError( f"Preprocessor {self.typedef.name}.{param_def.key}: invalid value {value}, "
                                      f"expected value >= {param_def.min_val}")
        for key in self.parameter_values:
            if key not in expected_keys:
                raise ValueError(f"Preprocessor {self.typedef.name}: unexpected {key} param")
        return self



