# Local detection

`detect` finds the ComfyUI and WebUI installs on this machine and the servers answering on it, before the caller
knows a URL to pass to `connect_to_backend`. It reports evidence only: each `Finding` carries a `FindingCode` and a
details dict, and the caller decides what to show and what to launch.

::: sd_backend_client.detect

::: sd_backend_client.SearchLocations

::: sd_backend_client.default_search_locations

::: sd_backend_client.DetectionResult

::: sd_backend_client.DetectedInstall

::: sd_backend_client.DetectedServer

::: sd_backend_client.Finding

::: sd_backend_client.FindingCode

::: sd_backend_client.FindingSeverity

::: sd_backend_client.BackendKind

::: sd_backend_client.InstallOwner
