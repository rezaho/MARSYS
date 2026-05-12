"""Tests for StorageBackend Protocol + FileStorageBackend.

Covers:
- AC-42: StorageBackend Protocol contract
- AC-43: FileStorageBackend(root) constructor
- AC-44: atomic write semantics (write-temp + os.replace + fsync(parent))
- AC-45: simulated mid-write crash leaves prior snapshot intact
- AC-46: expire_older_than deletes stale entries, returns count
- AC-47: snapshot path is <root>/<session_id>/snapshot.json
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from marsys.coordination.state import (
    FileStorageBackend,
    StorageBackend,
    StorageEntry,
)


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    return tmp_path / "snapshots"


@pytest.fixture
def backend(storage_root: Path) -> FileStorageBackend:
    return FileStorageBackend(storage_root)


def test_file_storage_backend_satisfies_protocol(backend: FileStorageBackend):
    """AC-42 + AC-43: FileStorageBackend satisfies StorageBackend (Protocol)."""
    assert isinstance(backend, StorageBackend)


@pytest.mark.asyncio
async def test_basic_write_then_read(backend: FileStorageBackend):
    await backend.write("sess-1/snapshot.json", b"hello world")
    data = await backend.read("sess-1/snapshot.json")
    assert data == b"hello world"


@pytest.mark.asyncio
async def test_write_uses_documented_path_layout(
    backend: FileStorageBackend, storage_root: Path,
):
    """AC-47: default snapshot path is <root>/<session_id>/snapshot.json."""
    await backend.write("sess-99/snapshot.json", b"payload")
    expected = storage_root / "sess-99" / "snapshot.json"
    assert expected.exists()
    assert expected.read_bytes() == b"payload"


@pytest.mark.asyncio
async def test_write_is_atomic_via_os_replace(
    backend: FileStorageBackend, storage_root: Path,
):
    """AC-44: writes go through a .tmp file then os.replace.

    We can't easily intercept fsync at the test level without OS-level
    tooling, but we can verify the .tmp path exists during the write
    by overriding the synchronous helper to inspect mid-flight state.
    """
    seen_tmp_existence = []
    real_replace = os.replace

    def spy_replace(src, dst):
        # The .tmp file must exist when os.replace is called.
        seen_tmp_existence.append(Path(src).exists())
        real_replace(src, dst)

    target = "sess-atomic/snapshot.json"
    import unittest.mock as mock
    with mock.patch.object(os, "replace", side_effect=spy_replace):
        await backend.write(target, b"first")
        await backend.write(target, b"second")
    assert seen_tmp_existence == [True, True]
    data = await backend.read(target)
    assert data == b"second"
    # The .tmp sibling must NOT linger after the write completes.
    final = storage_root / "sess-atomic" / "snapshot.json"
    sibling = final.with_suffix(final.suffix + ".tmp")
    assert not sibling.exists()


@pytest.mark.asyncio
async def test_simulated_crash_mid_write_preserves_prior_snapshot(
    backend: FileStorageBackend, storage_root: Path,
):
    """AC-45: a crash between fsync(file) and os.replace leaves the prior
    snapshot intact.

    We simulate the crash by raising inside `os.replace` after the .tmp
    file has been written. The target file (the prior snapshot) must
    remain unchanged; the .tmp file may linger (we accept either outcome
    on the .tmp file — it's a leaked-temp file, not a torn snapshot).
    """
    target = "sess-crash/snapshot.json"
    await backend.write(target, b"prior")

    real_replace = os.replace
    crash_count = {"n": 0}

    def crashing_replace(src, dst):
        crash_count["n"] += 1
        raise OSError("simulated crash mid-write")

    import unittest.mock as mock
    with mock.patch.object(os, "replace", side_effect=crashing_replace):
        with pytest.raises(OSError, match="simulated crash"):
            await backend.write(target, b"new value that never landed")

    # The prior snapshot is intact.
    assert (await backend.read(target)) == b"prior"
    assert crash_count["n"] == 1


@pytest.mark.asyncio
async def test_delete_removes_entry_and_is_idempotent(backend: FileStorageBackend):
    await backend.write("sess-x/snapshot.json", b"payload")
    await backend.delete("sess-x/snapshot.json")
    with pytest.raises(FileNotFoundError):
        await backend.read("sess-x/snapshot.json")
    # Idempotent: deleting again is not an error.
    await backend.delete("sess-x/snapshot.json")


@pytest.mark.asyncio
async def test_list_with_metadata_returns_storage_entries(
    backend: FileStorageBackend,
):
    await backend.write("a/snapshot.json", b"aaa")
    await backend.write("b/snapshot.json", b"bb")
    entries = await backend.list_with_metadata()
    keys = sorted(e.key for e in entries)
    assert keys == ["a/snapshot.json", "b/snapshot.json"]
    sizes = {e.key: e.size_bytes for e in entries}
    assert sizes["a/snapshot.json"] == 3
    assert sizes["b/snapshot.json"] == 2
    for e in entries:
        assert isinstance(e, StorageEntry)
        assert e.modified_at.tzinfo is not None


@pytest.mark.asyncio
async def test_expire_older_than_deletes_stale_entries(
    backend: FileStorageBackend, storage_root: Path,
):
    """AC-46: expire_older_than deletes entries older than the threshold,
    leaves newer entries, and returns the count deleted.
    """
    await backend.write("old/snapshot.json", b"old")
    await backend.write("new/snapshot.json", b"new")
    # Backdate the "old" file to 100 days ago.
    old_path = storage_root / "old" / "snapshot.json"
    backdated = time.time() - timedelta(days=100).total_seconds()
    os.utime(old_path, (backdated, backdated))

    deleted = await backend.expire_older_than(timedelta(days=30))
    assert deleted == 1
    # Newer entry survives.
    assert (await backend.read("new/snapshot.json")) == b"new"
    # Older entry is gone.
    with pytest.raises(FileNotFoundError):
        await backend.read("old/snapshot.json")


@pytest.mark.asyncio
async def test_invalid_keys_rejected(backend: FileStorageBackend):
    """Defense: keys with absolute paths or .. components are rejected."""
    with pytest.raises(ValueError):
        await backend.write("/etc/passwd", b"x")
    with pytest.raises(ValueError):
        await backend.write("../escape/snapshot.json", b"x")
