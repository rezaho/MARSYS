"""Virtual filesystem abstraction for MARSYS tools and agents.

This module provides a unified virtual filesystem that allows tools and agents
to work with paths consistently, regardless of the underlying host filesystem.

Example:
    >>> from marsys.environment.filesystem import RunFileSystem
    >>> fs = RunFileSystem.local(run_root=Path.cwd())
    >>> resolved = fs.resolve("./downloads/file.txt")
    >>> print(resolved.virtual_path)
    /downloads/file.txt
"""

from .core import LocalBackend, ResolvedPath, RunFileSystem

__all__ = ["RunFileSystem", "ResolvedPath", "LocalBackend"]
