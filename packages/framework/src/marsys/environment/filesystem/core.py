"""Run-scoped virtual filesystem with mount support.

Agent-facing paths are always virtual POSIX paths rooted at '/'.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class ResolvedPath:
    """Result of resolving a virtual path to a host path.

    Attributes:
        virtual_path: The canonical virtual path (e.g., '/downloads/file.txt')
        host_path: The actual filesystem path on the host
        mount_prefix: The mount prefix this path belongs to (e.g., '/' or '/memory')
    """

    virtual_path: str
    host_path: Path
    mount_prefix: str


class LocalBackend:
    """Local filesystem backend rooted at a host directory.

    This backend provides path resolution within a specified root directory,
    with traversal protection to prevent escaping the root.
    """

    def __init__(self, root: Path, allow_symlink_escape: bool = False) -> None:
        """Initialize the local backend.

        Args:
            root: The host directory that serves as the root for this backend
            allow_symlink_escape: If True, allow symlinks that point outside the root
        """
        self.root = root.resolve()
        self.allow_symlink_escape = allow_symlink_escape
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel_path: PurePosixPath) -> Path:
        """Resolve a relative path within this backend's root.

        Args:
            rel_path: A relative POSIX path to resolve

        Returns:
            The resolved host Path

        Raises:
            ValueError: If the resolved path escapes the backend root
        """
        rel_str = "" if str(rel_path) in (".", "") else rel_path.as_posix()
        candidate = (self.root / rel_str).resolve(strict=False)

        if not self.allow_symlink_escape:
            try:
                candidate.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(
                    f"Resolved path escapes backend root: {candidate}"
                ) from exc

        return candidate

    def to_relative(self, host_path: Path) -> PurePosixPath:
        """Convert a host path to a relative path within this backend.

        Args:
            host_path: An absolute host path

        Returns:
            A relative POSIX path from the root

        Raises:
            ValueError: If the host path is not within this backend's root
        """
        resolved = host_path.resolve(strict=False)
        rel = resolved.relative_to(self.root)
        return PurePosixPath(rel.as_posix()) if rel.as_posix() else PurePosixPath(".")


class RunFileSystem:
    """Virtual filesystem abstraction used by tools and agents.

    Provides a unified virtual path space where:
    - '/' is always the run root directory
    - Optional mounts can map virtual prefixes to different host directories
    - All path operations prevent escaping the virtual filesystem boundaries

    Example:
        >>> fs = RunFileSystem.local(run_root=Path("/tmp/run-123"))
        >>> resolved = fs.resolve("./downloads/file.txt")
        >>> resolved.virtual_path
        '/downloads/file.txt'
        >>> resolved.host_path
        PosixPath('/tmp/run-123/downloads/file.txt')
    """

    def __init__(
        self,
        mounts: Dict[str, LocalBackend],
        cwd: str = "/",
    ) -> None:
        """Initialize the virtual filesystem.

        Args:
            mounts: Dictionary mapping virtual prefixes to LocalBackend instances.
                    Must include a '/' mount.
            cwd: Initial virtual working directory (default: '/')

        Raises:
            ValueError: If no '/' mount is provided
        """
        normalized_mounts: Dict[str, LocalBackend] = {}
        for prefix, backend in mounts.items():
            p = self._normalize_absolute_prefix(prefix)
            normalized_mounts[p.as_posix()] = backend

        if "/" not in normalized_mounts:
            raise ValueError("RunFileSystem requires a '/' mount")

        self._mounts = normalized_mounts
        self.cwd = self._normalize_virtual(cwd, PurePosixPath("/")).as_posix()

    @classmethod
    def local(
        cls,
        run_root: Path,
        cwd: str = "/",
        memory_root: Optional[Path] = None,
        extra_mounts: Optional[Dict[str, Path]] = None,
        allow_symlink_escape: bool = False,
    ) -> "RunFileSystem":
        """Create a local filesystem rooted at the given directory.

        Args:
            run_root: The host directory that will be mounted at '/'
            cwd: Initial virtual working directory (default: '/')
            memory_root: Optional separate directory for '/memory' mount
                         (shorthand for extra_mounts={"/memory": memory_root})
            extra_mounts: Optional dictionary mapping virtual prefixes to host
                          directories. Each entry creates a mount point that
                          agents can access via the virtual prefix.
                          Example: {"/datasets": Path("/shared/data")}
            allow_symlink_escape: If True, allow symlinks pointing outside roots

        Returns:
            A configured RunFileSystem instance
        """
        mounts: Dict[str, LocalBackend] = {
            "/": LocalBackend(run_root, allow_symlink_escape=allow_symlink_escape),
        }
        if memory_root is not None:
            mounts["/memory"] = LocalBackend(
                memory_root,
                allow_symlink_escape=allow_symlink_escape,
            )
        if extra_mounts:
            for virtual_prefix, host_dir in extra_mounts.items():
                mounts[virtual_prefix] = LocalBackend(
                    host_dir,
                    allow_symlink_escape=allow_symlink_escape,
                )
        return cls(mounts=mounts, cwd=cwd)

    def resolve(
        self,
        path: Optional[str],
        cwd: Optional[str] = None,
    ) -> ResolvedPath:
        """Resolve a virtual path to a host path.

        Args:
            path: The virtual path to resolve (relative or absolute).
                  If None or empty, defaults to '.'
            cwd: Optional working directory override. If None, uses self.cwd

        Returns:
            ResolvedPath with virtual path, host path, and mount prefix

        Raises:
            ValueError: If the path would escape the virtual root
        """
        cwd_pp = self._normalize_virtual(cwd or self.cwd, PurePosixPath("/"))
        virtual = self._normalize_virtual(path or ".", cwd_pp)

        mount_prefix, backend = self._select_mount(virtual)
        rel = self._relative_to_mount(virtual, mount_prefix)
        host = backend.resolve(rel)

        return ResolvedPath(
            virtual_path=virtual.as_posix(),
            host_path=host,
            mount_prefix=mount_prefix.as_posix(),
        )

    def to_virtual(self, host_path: Path) -> str:
        """Convert a host path to a relative virtual path.

        Returns paths relative to the virtual root so they work both in
        file-operation tools (which resolve through RunFileSystem) and in
        code-execution tools (where CWD is the host root directory).

        Args:
            host_path: A host filesystem path

        Returns:
            A relative virtual path (e.g. ``./downloads/file.txt``, ``.``)

        Raises:
            ValueError: If the host path doesn't belong to any mount
        """
        resolved = host_path.resolve(strict=False)

        for prefix in self._sorted_mount_prefixes():
            backend = self._mounts[prefix]
            try:
                rel = backend.to_relative(resolved)
            except ValueError:
                continue

            prefix_pp = PurePosixPath(prefix)
            if prefix_pp == PurePosixPath("/"):
                rel_str = rel.as_posix()
                return "." if rel_str in (".", "") else f"./{rel_str}"
            mount_name = prefix_pp.as_posix().lstrip("/")
            if str(rel) in (".", ""):
                return f"./{mount_name}"
            return f"./{mount_name}/{rel.as_posix()}"

        raise ValueError(f"Host path does not belong to any mount: {resolved}")

    def to_agent_path(
        self, host_or_virtual_path: Union[str, Path], cwd: Optional[str] = None
    ) -> str:
        """Convert a path to an agent-friendly relative format.

        Args:
            host_or_virtual_path: Either a host Path or a virtual path string
            cwd: Optional working directory override

        Returns:
            A relative path string (e.g. ``./downloads/file.txt``, ``.``)
        """
        if isinstance(host_or_virtual_path, Path):
            return self.to_virtual(host_or_virtual_path)

        text = str(host_or_virtual_path)
        if text.startswith("/"):
            text = text.lstrip("/")
        return f"./{text}" if text else "."

    def _select_mount(
        self, virtual: PurePosixPath
    ) -> Tuple[PurePosixPath, LocalBackend]:
        """Select the appropriate mount for a virtual path.

        Uses longest-prefix matching to find the best mount.

        Args:
            virtual: The normalized virtual path

        Returns:
            Tuple of (mount_prefix, backend)

        Raises:
            ValueError: If no mount matches (should never happen with '/' mount)
        """
        for prefix in self._sorted_mount_prefixes():
            prefix_pp = PurePosixPath(prefix)
            if virtual == prefix_pp or str(virtual).startswith(
                prefix.rstrip("/") + "/"
            ):
                return prefix_pp, self._mounts[prefix]
        raise ValueError(f"No mount found for virtual path: {virtual}")

    def _sorted_mount_prefixes(self) -> List[str]:
        """Get mount prefixes sorted by length (longest first).

        This ensures longest-prefix matching works correctly.
        """
        return sorted(self._mounts.keys(), key=len, reverse=True)

    @staticmethod
    def _relative_to_mount(
        virtual: PurePosixPath, mount_prefix: PurePosixPath
    ) -> PurePosixPath:
        """Get the relative path from a mount prefix.

        Args:
            virtual: The full virtual path
            mount_prefix: The mount prefix to strip

        Returns:
            The relative path within the mount
        """
        if mount_prefix == PurePosixPath("/"):
            parts = virtual.parts[1:]  # Skip the leading '/'
            return PurePosixPath(*parts) if parts else PurePosixPath(".")

        if virtual == mount_prefix:
            return PurePosixPath(".")

        rel_parts = virtual.relative_to(mount_prefix).parts
        return PurePosixPath(*rel_parts) if rel_parts else PurePosixPath(".")

    @staticmethod
    def _normalize_absolute_prefix(prefix: str) -> PurePosixPath:
        """Normalize a mount prefix to a canonical form.

        Args:
            prefix: The mount prefix string

        Returns:
            Normalized PurePosixPath

        Raises:
            ValueError: If the prefix is not absolute
        """
        p = PurePosixPath(prefix)
        if not p.is_absolute():
            raise ValueError(f"Mount prefix must be absolute virtual path: {prefix}")
        parts = [part for part in p.parts if part not in ("", "/", ".")]
        return (
            PurePosixPath("/") / PurePosixPath(*parts)
            if parts
            else PurePosixPath("/")
        )

    @staticmethod
    def _normalize_virtual(path: str, cwd: PurePosixPath) -> PurePosixPath:
        """Normalize a virtual path to canonical form.

        Resolves relative paths against cwd and handles '..' components.

        Args:
            path: The path to normalize (relative or absolute)
            cwd: The current working directory for relative paths

        Returns:
            Canonical absolute virtual path

        Raises:
            ValueError: If the path escapes the virtual root via '..'
        """
        candidate = PurePosixPath(path)
        combined = candidate if candidate.is_absolute() else (cwd / candidate)

        normalized: List[str] = []
        for part in combined.parts:
            if part in ("", "/", "."):
                continue
            if part == "..":
                if not normalized:
                    raise ValueError(f"Path escapes virtual root: {path}")
                normalized.pop()
                continue
            normalized.append(part)

        return (
            PurePosixPath("/") / PurePosixPath(*normalized)
            if normalized
            else PurePosixPath("/")
        )
