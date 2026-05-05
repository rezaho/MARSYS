"""
Configuration for code execution toolkit.

This module defines the configuration class for controlling security,
permissions, resource limits, and execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from marsys.environment.filesystem import RunFileSystem


DEFAULT_ALLOWED_ENV_VARS = [
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "PYTHONPATH",
]

DEFAULT_BLOCKED_SHELL_PATTERNS = [
    "rm -rf /",
    "sudo",
    "dd if=",
    "mkfs",
    "format",
    "> /dev/",
    "chmod -R 777",
    ":(){ :|:& };:",
]

DEFAULT_BLOCKED_PYTHON_MODULES = [
    "subprocess",
    "shutil",
    "ctypes",
    "multiprocessing",
]


@dataclass
class CodeExecutionConfig:
    """
    Configuration for code execution toolkit.

    This class controls security settings, resource limits,
    and execution behavior for Python and shell code execution.
    """

    # === Directory Settings ===

    base_directory: Path = field(default_factory=lambda: Path.cwd() / "code_exec")
    """Base directory for code execution. Working directories are relative to this."""

    output_subdir: str = "outputs"
    """Subdirectory under base_directory where images and artifacts are saved."""

    # === Resource Limits ===

    timeout_default: int = 30
    """Default timeout in seconds for code execution."""

    max_output_bytes: int = 1_000_000
    """Maximum output size in bytes (stdout + stderr). Larger outputs are truncated."""

    max_images: int = 4
    """Maximum number of images to return per execution."""

    max_memory_mb: int = 1024
    """Maximum memory usage in MB (Linux only, via resource.setrlimit)."""

    max_cpu_seconds: int = 30
    """Maximum CPU time in seconds (Linux only, via resource.setrlimit)."""

    # === Network Settings ===

    allow_network: bool = False
    """
    If False (default), block network-related commands and Python modules.
    Network commands (curl, wget, nc, ssh) and modules (socket, urllib, requests) are blocked.
    """

    # === Environment Settings ===

    allowed_env_vars: List[str] = field(
        default_factory=lambda: DEFAULT_ALLOWED_ENV_VARS.copy()
    )
    """
    List of environment variables that can be inherited or set.
    Only these variables will be passed to subprocess environments.
    """

    # === Python Settings ===

    allowed_python_modules: List[str] = field(default_factory=list)
    """
    Whitelist of allowed Python modules (empty = all allowed except blocked).
    When non-empty, only listed modules can be imported.
    """

    blocked_python_modules: List[str] = field(
        default_factory=lambda: DEFAULT_BLOCKED_PYTHON_MODULES.copy()
    )
    """
    Blacklist of blocked Python modules. These modules cannot be imported.
    Network-related modules are automatically added when allow_network=False.
    """

    python_executable: Optional[str] = None
    """
    Path to Python executable. If None, uses venv_path or system Python.
    """

    venv_path: Optional[Path] = None
    """
    Path to virtual environment. If None, auto-detects .venv in current directory.
    """

    session_persistent_python: bool = False
    """
    If True, use a persistent Python session that maintains state between calls.
    Useful for iterative data analysis. Session is terminated on cleanup.
    """

    # === Shell Settings ===

    shell_path: str = "/bin/sh"
    """Path to shell executable. Default is POSIX /bin/sh, can be /bin/bash if needed."""

    allowed_shell_commands: List[str] = field(default_factory=list)
    """
    Whitelist of allowed shell commands (empty = all allowed except blocked patterns).
    When non-empty, only commands starting with these are allowed.
    """

    blocked_shell_patterns: List[str] = field(
        default_factory=lambda: DEFAULT_BLOCKED_SHELL_PATTERNS.copy()
    )
    """Patterns that are blocked in shell commands for safety."""

    # === Virtual Filesystem Settings ===

    run_filesystem: Optional["RunFileSystem"] = None
    """
    Shared run filesystem for unified path resolution across tools.
    If provided, working directory resolution uses this object.
    If None, one is auto-created from base_directory.
    """

    output_virtual_dir: str = "./outputs"
    """
    Virtual path for output directory where images and artifacts are saved.
    Maps to base_directory/output_subdir on the host.
    """

    def __post_init__(self) -> None:
        """Validate and initialize configuration after dataclass init."""
        # Auto-detect venv if not specified
        if self.venv_path is None:
            candidate = Path.cwd() / ".venv"
            if candidate.exists():
                self.venv_path = candidate

        # Convert base_directory to Path if string
        if isinstance(self.base_directory, str):
            self.base_directory = Path(self.base_directory)

        # Convert venv_path to Path if string
        if self.venv_path is not None and isinstance(self.venv_path, str):
            self.venv_path = Path(self.venv_path)

        # Create base directory if it doesn't exist
        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Create output directory
        (self.base_directory / self.output_subdir).mkdir(parents=True, exist_ok=True)

        # Auto-create RunFileSystem if not provided
        if self.run_filesystem is None:
            from marsys.environment.filesystem import RunFileSystem

            self.run_filesystem = RunFileSystem.local(
                run_root=self.base_directory,
            )

    def resolve_output_dir(self) -> Path:
        """
        Resolve the output directory for images and artifacts (host path).

        Returns:
            Path to output directory (created if doesn't exist)
        """
        out_dir = self.base_directory / self.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def resolve_virtual_output_dir(self) -> str:
        """
        Resolve the virtual output directory path.

        Returns:
            Virtual path to output directory (e.g., '/outputs')
        """
        return self.output_virtual_dir

    def resolve_python_executable(self) -> str:
        """
        Resolve the Python executable path.

        Priority:
        1. Explicit python_executable if set
        2. Python from venv_path if available
        3. System 'python' command

        Returns:
            Path to Python executable
        """
        if self.python_executable:
            return self.python_executable

        if self.venv_path:
            venv_python = self.venv_path / "bin" / "python"
            if venv_python.exists():
                return str(venv_python)

        return "python"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CodeExecutionConfig("
            f"base_dir={self.base_directory}, "
            f"timeout={self.timeout_default}s, "
            f"network={'enabled' if self.allow_network else 'disabled'}, "
            f"persistent_python={self.session_persistent_python})"
        )
