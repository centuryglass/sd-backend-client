"""Defines a ControlNet preprocessor's name and parameters, for use with a Stable Diffusion API"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

# ParameterDef is re-exported here, its home before it was shared with soft inpainting.
from sd_backend_client.api.shared_data.parameter_def import ParameterDef, validate_parameter_values

__all__ = ['ControlNetPreprocessor', 'ParameterDef', 'PreprocessorParams']


class ControlNetPreprocessor(BaseModel):
    """A ControlNet preprocessor a server offers, with the parameters it accepts.

    `Backend.get_controlnet_preprocessors` lists them for either backend. To use one, wrap it in a
    `PreprocessorParams` with the chosen parameter values and attach that to a `ControlNetUnit`.
    """
    # use_attribute_docstrings copies each field's docstring into its FieldInfo.description.
    model_config = ConfigDict(protected_namespaces=('model_validate', 'model_dump'), use_attribute_docstrings=True)

    name: str
    """The server's identifier for the preprocessor: a WebUI module name, or a ComfyUI node class type."""

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
    """A chosen ControlNet preprocessor and the parameter values to run it with, for `ControlNetUnit.preprocessor`.

    Construction fills every parameter the caller leaves out with its default, then raises `ValueError` for an
    unknown key or a value that doesn't fit its `ParameterDef`.
    """
    # use_attribute_docstrings copies each field's docstring into its FieldInfo.description.
    model_config = ConfigDict(use_attribute_docstrings=True)

    typedef: ControlNetPreprocessor
    """The preprocessor to run, as listed by `Backend.get_controlnet_preprocessors`."""

    parameter_values: dict[str, Any]
    """Values for the preprocessor's settings, keyed by `ParameterDef.key` from `typedef.parameters`.

    May be omitted or partial: each missing key gets its `ParameterDef.default_value`.
    """

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
        validate_parameter_values(self.typedef.parameters, self.parameter_values, f'Preprocessor {self.typedef.name}')
        return self
