"""Sorts ControlNet models and preprocessors into categories, automatically filtering out unavailable options."""
import re
from copy import deepcopy
from typing import Optional

from sd_backend_client.api.shared_data.controlnet.controlnet_constants import (
    STATIC_CONTROL_TYPE_DEFS, ControlTypeDef,
)
from sd_backend_client.api.webui.controlnet_webui_constants import ControlTypeResponse, PREPROCESSOR_MODEL_FREE

# TODO: temp constants, remove or relocate after figuring out how to properly reduce the use of 'None' strings to minimum
PREPROCESSOR_NONE = 'None'
"""Placeholder preprocessor name for "no preprocessing". It appears in `ControlTypeDef.module_list`, and as
`ControlTypeDef.default_option` when a control type has no preferred preprocessor."""

CONTROLNET_MODEL_NONE = 'None'
"""Placeholder model name for "no model". It appears in `ControlTypeDef.model_list`, and as
`ControlTypeDef.default_model` when a control type has no preferred model."""

class ControlNetCategoryBuilder:
    """Sorts ControlNet models and preprocessors into categories, automatically filtering out unavailable options."""

    def __init__(self, preprocessor_names: list[str], model_names: list[str],
                 preprocessor_comfyui_categories: Optional[dict[str, str]] = None,
                 control_type_defs: Optional[ControlTypeResponse] = None):
        self._preprocessors: list[str] = preprocessor_names
        self._models: list[str] = model_names
        self._control_types: dict[str, ControlTypeDef] = {}
        if PREPROCESSOR_NONE not in self._preprocessors and PREPROCESSOR_NONE.lower() not in self._preprocessors:
            self._preprocessors.append(PREPROCESSOR_NONE)
        if CONTROLNET_MODEL_NONE not in self._models and CONTROLNET_MODEL_NONE.lower() not in self._models:
            self._models.append(CONTROLNET_MODEL_NONE)

        # Server categories and defaults take precedence over the built-in table.
        if control_type_defs is not None:
            for type_name, definition in control_type_defs['control_types'].items():
                self._add_type_options(type_name, ControlTypeDef(
                    module_list=definition['module_list'], model_list=definition['model_list'],
                    default_option=definition['default_option'], default_model=definition['default_model'],
                ))
        for type_name, definition in STATIC_CONTROL_TYPE_DEFS.items():
            self._add_type_options(type_name, ControlTypeDef(
                module_list=definition['module_list'], model_list=definition['model_list'],
                default_option=definition['default_option'], default_model=definition['default_model'],
            ))
            self._add_pattern_matches(type_name, definition.get('preprocessor_pattern'),
                                      definition.get('model_pattern'), preprocessor_comfyui_categories)

    def _add_type_options(self, name: str, definition: ControlTypeDef) -> None:
        saved = self._control_types.setdefault(name, {
            'module_list': [PREPROCESSOR_NONE],
            'model_list': [CONTROLNET_MODEL_NONE],
            'default_option': PREPROCESSOR_NONE,
            'default_model': CONTROLNET_MODEL_NONE,
        })
        for preprocessor in definition['module_list']:
            if preprocessor in self._preprocessors and preprocessor not in saved['module_list']:
                saved['module_list'].append(preprocessor)
        for model in definition['model_list']:
            if model in self._models and model not in saved['model_list']:
                saved['model_list'].append(model)
        if saved['default_option'] == PREPROCESSOR_NONE and definition['default_option'] in self._preprocessors:
            saved['default_option'] = definition['default_option']
        if saved['default_model'] == CONTROLNET_MODEL_NONE and definition['default_model'] in self._models:
            saved['default_model'] = definition['default_model']

    def _add_pattern_matches(self, name: str, preprocessor_regex: Optional[str], model_regex: Optional[str],
                             categories: Optional[dict[str, str]]) -> None:
        saved = self._control_types[name]
        if preprocessor_regex is not None:
            pattern = re.compile(preprocessor_regex)
            category_by_name = categories or {}
            for preprocessor in self._preprocessors:
                if preprocessor not in saved['module_list'] and \
                        (pattern.search(preprocessor) or pattern.search(category_by_name.get(preprocessor, ''))):
                    saved['module_list'].append(preprocessor)
        if model_regex is not None:
            pattern = re.compile(model_regex)
            for model in self._models:
                if model not in saved['model_list'] and pattern.search(model):
                    saved['model_list'].append(model)

    def get_control_types(self) -> dict[str, ControlTypeDef]:
        """Returns all control types that have at least one preprocessor that's not 'none', and either at least one
           model that's not 'None' or are model-free ControlNet types."""
        non_empty_types: dict[str, ControlTypeDef] = {}
        for name, type_def in self._control_types.items():
            if len(type_def['module_list']) > 1 and (len(type_def['model_list']) > 1
                                                     or name in PREPROCESSOR_MODEL_FREE):
                non_empty_types[name] = deepcopy(type_def)
        return non_empty_types
