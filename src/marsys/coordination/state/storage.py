"""Snapshot storage abstraction.

The framework ships ``FileStorageBackend`` rooted at a caller-configured
path; Hosted S3/GCS backends and CI integrations' artifact-store
backends live in their own repos and satisfy ``StorageBackend`` purely
structurally (no inheritance from a framework class).

Atomic write contract: ``StorageBackend.write`` must guarantee that a
crash mid-write leaves either the previous file or no file at all — never
a torn or partial file. ``FileStorageBackend`` achieves this with
write-temp + ``fsync(fd)`` + ``os.replace`` + ``fsync(parent_dir_fd)``
(POSIX; the parent-dir fsync is a no-op on Windows but ``os.replace``
itself is atomic via ``MoveFileEx`` semantics).

Reference: ``man rename(2)`` (POSIX), Python ``os.replace`` docs.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class StorageEntry(BaseModel):
    """One entry's metadata for ``StorageBackend.list_with_metadata()``."""

    model_config = ConfigDict(extra="forbid")

    key: str
    size_bytes: int
    modified_at: datetime


@runtime_checkable
class StorageBackend(Protocol):
    """Generic storage abstraction for snapshots.

    Implementations MUST guarantee atomic writes — a crash mid-write
    leaves either the previous value or no value at the key, never a
    partial value.
    """

    async def read(self, key: str) -> bytes:
        """Read the raw bytes stored at ``key``. Raises ``FileNotFoundError``
        (or backend-specific equivalent) if the key does not exist.
        """
        ...

    async def write(self, key: str, data: bytes) -> None:
        """Atomically write ``data`` to ``key``. Overwrites any existing
        value at ``key``."""
        ...

    async def delete(self, key: str) -> None:
        """Delete the entry at ``key``. Idempotent — a missing key is not
        an error."""
        ...

    async def list_with_metadata(self) -> list[StorageEntry]:
        """Enumerate all entries with their metadata, without loading
        bodies. Used by ``Orchestra.list_paused_sessions()``."""
        ...

    async def expire_older_than(self, age: timedelta) -> int:
        """Delete entries whose ``modified_at`` is older than ``age``
        before now. Returns the count deleted."""
        ...


class FileStorageBackend:
    """File-backed snapshot storage.

    Layout: ``<root>/<key>`` where ``key`` is typically
    ``"<session_id>/snapshot.json"``. The backend creates parent
    directories on write.

    Atomic writes:
        1. open(target.tmp, 'wb', os.O_CLOEXEC)
        2. write data
        3. fsync(file_fd)
        4. close
        5. os.replace(target.tmp, target)        # atomic on POSIX + Windows
        6. fsync(parent_dir_fd)                  # POSIX-only; no-op on Windows

    Source: ``man rename(2)``, ``man fsync(2)``, Python ``os.replace`` docs.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Treat keys with separators as relative paths under root.
        # Reject absolute paths and parent-traversal to prevent escaping root.
        key_path = Path(key)
        if key_path.is_absolute() or ".." in key_path.parts:
            raise ValueError(f"invalid key: {key!r}")
        return self.root / key_path

    async def read(self, key: str) -> bytes:
        path = self._path(key)
        return await asyncio.to_thread(path.read_bytes)

    async def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        await asyncio.to_thread(self._atomic_write_sync, path, data)

    @staticmethod
    def _atomic_write_sync(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        # Open with O_CLOEXEC so a forked subprocess doesn't inherit the fd.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(str(tmp_path), flags, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp_path), str(path))
        # fsync the parent dir so the rename is durable across crash.
        # Windows lacks O_DIRECTORY; skip silently there.
        if os.name == "posix":
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                return
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(self._delete_sync, path)

    @staticmethod
    def _delete_sync(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        # Best-effort: remove the empty parent directory if this was the
        # only file there (a session directory). Don't fail if the directory
        # is not empty.
        parent = path.parent
        try:
            parent.rmdir()
        except OSError:
            pass

    async def list_with_metadata(self) -> list[StorageEntry]:
        return await asyncio.to_thread(self._list_with_metadata_sync)

    def _list_with_metadata_sync(self) -> list[StorageEntry]:
        entries: list[StorageEntry] = []
        for child in self.root.rglob("*"):
            if not child.is_file():
                continue
            stat = child.stat()
            rel = child.relative_to(self.root)
            entries.append(
                StorageEntry(
                    key=str(rel).replace(os.sep, "/"),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
        return entries

    async def expire_older_than(self, age: timedelta) -> int:
        return await asyncio.to_thread(self._expire_older_than_sync, age)

    def _expire_older_than_sync(self, age: timedelta) -> int:
        cutoff = time.time() - age.total_seconds()
        deleted = 0
        for child in list(self.root.rglob("*")):
            if not child.is_file():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    child.unlink()
                    deleted += 1
                    # Same parent-cleanup as delete(): remove empty parent.
                    parent = child.parent
                    if parent != self.root:
                        try:
                            parent.rmdir()
                        except OSError:
                            pass
            except OSError:
                continue
        return deleted


__all__ = ["StorageBackend", "FileStorageBackend", "StorageEntry"]
