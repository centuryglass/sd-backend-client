# API reference

The supported public API is the set of names in the package root's `__all__`. Import them from `sd_backend_client`
itself:

```python
from sd_backend_client import connect_to_backend, DiffusionParams, GenerationResult
```

Every other module, including the ComfyUI node graph and workflow builders, the HTTP layer and the wire formats under
`sd_backend_client.api`, stays importable but is internal and may change or move in any release. These pages cover
only the public names.

| Page | Names |
| --- | --- |
| [Clients](clients.md) | `__version__`, `Backend`, `connect_to_backend`, `A1111Webservice`, `ComfyUiWebservice` |
| [Generation parameters](parameters.md) | `DiffusionParams`, `DiffusionRequestBody`, `ResizeMode`, `InpaintFillOption`, `ScriptRequestData`, `ComfyUIDiffusionParams`, `DiffusionUpscalingParams` |
| [ControlNet](controlnet.md) | `ControlNetUnit`, `ControlNetModel`, `ControlNetPreprocessor`, `PreprocessorParams`, `ParameterDef`, `ControlTypeDef` |
| [Discovery](discovery.md) | `BackendOption`, `BackendCapabilities` |
| [Local detection](detection.md) | `detect`, `SearchLocations`, `default_search_locations`, `DetectionResult`, `DetectedInstall`, `DetectedServer`, `Finding`, `FindingCode`, `FindingSeverity`, `BackendKind`, `InstallOwner` |
| [Generation handles](generation.md) | `GenerationHandle`, `GenerationStatus`, `GenerationProgress`, `GenerationResult`, `ProgressCallback` |
| [Errors](errors.md) | `SDBackendError` and its subclasses |
