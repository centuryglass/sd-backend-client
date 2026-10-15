# ControlNet

ControlNet units are backend-agnostic. Add `ControlNetUnit`s to `DiffusionParams.controlnet_units`, and each client
serializes them into its own form. `Backend.list_controlnet_models` and `Backend.get_controlnet_preprocessors` list
the models and preprocessors a server offers. Each preprocessor lists its settings as `ParameterDef`s (see
Discovery).

::: sd_backend_client.ControlNetUnit

::: sd_backend_client.ControlNetModel

::: sd_backend_client.ControlNetPreprocessor

::: sd_backend_client.PreprocessorParams

::: sd_backend_client.ControlTypeDef

## Placeholder values

::: sd_backend_client.PREPROCESSOR_NONE

::: sd_backend_client.CONTROLNET_MODEL_NONE

## WebUI unit-level setting keys

::: sd_backend_client.CONTROL_MODE_PARAM_KEY

::: sd_backend_client.RESIZE_MODE_PARAM_KEY

::: sd_backend_client.PREPROCESSOR_RES_PARAM_KEY
