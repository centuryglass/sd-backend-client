"""The result model of local backend detection: installs found on disk, running servers, and findings about each.

Findings are evidence, not verdicts. A broken or stale install is still reported, with findings saying why, and
detection never ranks results or words them for users: the caller maps each `FindingCode` to its own text.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

DEFAULT_COMFYUI_URL = 'http://localhost:8188'
DEFAULT_WEBUI_URL = 'http://localhost:7860'
DEFAULT_WEBUI_PORT = 7860


class BackendKind(Enum):
    """The API a backend serves: ComfyUI's, or the WebUI API that A1111 and Forge share."""
    COMFYUI = 'comfyui'
    WEBUI = 'webui'


class InstallOwner(Enum):
    """The app that installed and manages a backend, or `NONE` for one the user set up themselves.

    `CALLER` marks the install passed to `detect` as `own_install`: the one the calling app manages. Only `NONE`
    installs get a `DetectedInstall.launch_command`.
    """
    NONE = 'none'
    STABILITY_MATRIX = 'stability_matrix'
    KRITA_AI_DIFFUSION = 'krita_ai_diffusion'
    COMFY_DESKTOP = 'comfy_desktop'
    CALLER = 'caller'


class FindingSeverity(Enum):
    """How much a finding affects whether an install or server can be used."""
    BLOCKER = 'blocker'
    """It can't work as it is."""
    WARNING = 'warning'
    """It may not work."""
    INFO = 'info'
    """Something detection couldn't tell, or that the caller may need to act on."""


class FindingCode(Enum):
    """What a finding reports. `Finding.severity` gives each code's severity."""
    ENTRY_POINT_MISSING = 'entry_point_missing'
    INTERPRETER_MISSING = 'interpreter_missing'
    BASE_PYTHON_MISSING = 'base_python_missing'
    TORCH_MISSING = 'torch_missing'
    INSTALL_INCOMPLETE = 'install_incomplete'
    API_DISABLED = 'api_disabled'
    NO_MODELS = 'no_models'
    PYTHON_VERSION_MISMATCH = 'python_version_mismatch'
    CPU_ONLY_TORCH = 'cpu_only_torch'
    INTERPRETER_UNKNOWN = 'interpreter_unknown'
    API_FLAG_UNKNOWN = 'api_flag_unknown'
    AUTH_REQUIRED = 'auth_required'


FINDING_SEVERITY: dict[FindingCode, FindingSeverity] = {
    FindingCode.ENTRY_POINT_MISSING: FindingSeverity.BLOCKER,
    FindingCode.INTERPRETER_MISSING: FindingSeverity.BLOCKER,
    FindingCode.BASE_PYTHON_MISSING: FindingSeverity.BLOCKER,
    FindingCode.TORCH_MISSING: FindingSeverity.BLOCKER,
    FindingCode.INSTALL_INCOMPLETE: FindingSeverity.BLOCKER,
    FindingCode.API_DISABLED: FindingSeverity.BLOCKER,
    FindingCode.NO_MODELS: FindingSeverity.WARNING,
    FindingCode.PYTHON_VERSION_MISMATCH: FindingSeverity.WARNING,
    FindingCode.CPU_ONLY_TORCH: FindingSeverity.WARNING,
    FindingCode.INTERPRETER_UNKNOWN: FindingSeverity.INFO,
    FindingCode.API_FLAG_UNKNOWN: FindingSeverity.INFO,
    FindingCode.AUTH_REQUIRED: FindingSeverity.INFO,
}


@dataclass
class Finding:
    """One piece of evidence about an install or server.

    Attributes
    ----------
    code: FindingCode
        What the finding reports.
    details: dict[str, str]
        Data for the caller to format, such as paths and versions. Never user-facing text.
    """
    code: FindingCode
    details: dict[str, str] = field(default_factory=dict)

    @property
    def severity(self) -> FindingSeverity:
        """The severity of the finding's code."""
        return FINDING_SEVERITY[self.code]


@dataclass
class DetectedServer:
    """A server that answered at a probed URL.

    Attributes
    ----------
    url: str
        The URL that was probed.
    kind: BackendKind, optional
        The API the server answered as, or None when it answered as neither backend.
    version: str, optional
        A ComfyUI server's version, from `/system_stats`.
    argv: list[str]
        A ComfyUI server's command line, from `/system_stats`.
    install_path: Path, optional
        The folder holding the `main.py` in `argv`, when that path is absolute.
    findings: list[Finding]
        Evidence about the server, such as `AUTH_REQUIRED` or `API_DISABLED`.
    """
    url: str
    kind: Optional[BackendKind]
    version: Optional[str] = None
    argv: list[str] = field(default_factory=list)
    install_path: Optional[Path] = None
    findings: list[Finding] = field(default_factory=list)


@dataclass
class DetectedInstall:
    """A backend install found on disk, and what its files say about whether it works.

    Attributes
    ----------
    kind: BackendKind
        The API the install serves.
    owner: InstallOwner
        The app that manages the install.
    path: Path
        The folder the user knows as the install. For ComfyUI portable this is the portable build's root, which holds
        `entry_point` in its `ComfyUI/` folder.
    entry_point: Path
        ComfyUI's `main.py` or WebUI's `launch.py`. It may not exist; `ENTRY_POINT_MISSING` then says so.
    webui_variant: str, optional
        'forge' or 'a1111' for WebUI installs, None for ComfyUI.
    interpreter: Path, optional
        The Python interpreter in the install's environment or its owner's. None when none was found.
    launch_command: list[str], optional
        A command that starts the install. Set only for ComfyUI installs owned by `NONE` whose interpreter was
        found.
    launch_args: list[str], optional
        A WebUI install's recorded arguments, from `webui-user` or its owner's settings, with credential values
        redacted. None if none were found.
    version: str, optional
        A ComfyUI install's version, from `comfyui_version.py`.
    git_commit: str, optional
        The commit the install's checkout is on, read from its `.git` files.
    last_used: datetime, optional
        The newest modification time among the files the backend writes when it runs, in UTC.
    findings: list[Finding]
        Evidence about whether the install works.
    server: DetectedServer, optional
        The running server linked to this install, when `detect` could tell which install it runs from.
    """
    kind: BackendKind
    owner: InstallOwner
    path: Path
    entry_point: Path
    webui_variant: Optional[str] = None
    interpreter: Optional[Path] = None
    launch_command: Optional[list[str]] = None
    launch_args: Optional[list[str]] = None
    version: Optional[str] = None
    git_commit: Optional[str] = None
    last_used: Optional[datetime] = None
    findings: list[Finding] = field(default_factory=list)
    server: Optional[DetectedServer] = None

    def has_finding(self, code: FindingCode) -> bool:
        """Whether any of the install's findings has `code`."""
        return any(finding.code == code for finding in self.findings)


@dataclass
class DetectionResult:
    """What `detect` found.

    Attributes
    ----------
    installs: list[DetectedInstall]
        Every install found, with the servers linked to them.
    servers: list[DetectedServer]
        The running servers that weren't linked to an install.
    """
    installs: list[DetectedInstall] = field(default_factory=list)
    servers: list[DetectedServer] = field(default_factory=list)
