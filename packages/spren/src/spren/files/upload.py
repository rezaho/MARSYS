"""Multipart upload handler.

Streams the body to ``<data-dir>/data/files/{file_id}/<sanitized_name>``
via write-temp + atomic rename. Computes sha256 + content-type detection
during the stream. Enforces per-file size cap and aggregate-storage cap;
rejects with 413 on either, cleaning up partial bytes.

Cap config is hardcoded for v0.3 (settings.files.* table is a v0.4
concern); defaults are 100MB per file, 5GB aggregate.

Mime detection uses ``mimetypes.guess_type()`` (stdlib, extension-based).
For v0.3 personal use this is sufficient; ``python-magic`` would add a
``libmagic`` system dep that v0.3 doesn't justify. The frontend's icon
map keys off whatever the server returns, so either approach works.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from ulid import ULID

from spren.models import FileMetadata
from spren.storage.files import insert_file, total_bytes_used


# Defaults — v0.3 hardcodes these. v0.4 will surface them via the
# settings table per docs/architecture/spren/02-data-model.md.
DEFAULT_MAX_PER_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_TOTAL_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

# Filename safety: allow alphanumeric + a small set; everything else → '_'.
# Decision §8.8: preserve the user's filename in the path for Finder/Explorer
# inspection; sanitize for filesystem safety.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
_DEFAULT_FILENAME = "upload.bin"

# 64 KB read chunk — large enough to amortize syscalls, small enough to be
# responsive to mid-stream cap cancellation.
_CHUNK_BYTES = 64 * 1024


class FileTooLargeError(Exception):
    """Per-file size cap exceeded."""

    def __init__(self, *, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"file size {size_bytes} exceeds per-file cap of {max_bytes} bytes"
        )


class StorageCapExceededError(Exception):
    """Aggregate-storage cap would be exceeded by this upload."""

    def __init__(self, *, current_total: int, attempted_size: int, max_bytes: int) -> None:
        self.current_total = current_total
        self.attempted_size = attempted_size
        self.max_bytes = max_bytes
        super().__init__(
            f"adding {attempted_size} bytes to current total {current_total} would exceed "
            f"storage cap of {max_bytes} bytes"
        )


class InvalidFilenameError(Exception):
    """Uploaded file has an empty or unusable filename."""


@dataclass(frozen=True)
class UploadResult:
    """The on-disk + DB-row outcome of one successful upload."""

    metadata: FileMetadata
    on_disk_path: Path


def sanitize_filename(name: str) -> str:
    """Strip path separators + replace unsafe chars; preserve user-visible name in DB.

    The DB stores ``original_name`` verbatim; this helper produces the
    sanitized form used for the on-disk file. Empty / all-unsafe input
    falls back to ``upload.bin``.
    """
    base = os.path.basename(name).strip()
    if not base:
        return _DEFAULT_FILENAME
    sanitized = _SAFE_NAME_RE.sub("_", base)
    # Avoid leading dot (hidden file on macOS/Linux) for the disk name only.
    sanitized = sanitized.lstrip(".") or _DEFAULT_FILENAME
    return sanitized


def detect_mime_type(filename: str, declared: str | None) -> str:
    """Resolve the file's mime type.

    Trust client-declared content-type for non-octet-stream values
    (e.g., browsers send ``application/pdf`` for actual PDFs); for
    ``application/octet-stream`` (the multipart default) or absent,
    infer from extension via ``mimetypes.guess_type()``; final fallback
    is ``application/octet-stream``.
    """
    if declared and declared != "application/octet-stream":
        return declared
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


async def stream_upload(
    *,
    conn: sqlite3.Connection,
    upload_file: UploadFile,
    data_dir: Path,
    max_per_file_bytes: int = DEFAULT_MAX_PER_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> UploadResult:
    """Stream a multipart upload to disk; insert row; return the result.

    On any cap rejection or I/O error, partial bytes are cleaned up.
    The DB row is inserted ONLY after the file is durably on disk +
    final size is known.

    Caller (the route handler) wraps this in try/except on
    ``FileTooLargeError`` / ``StorageCapExceededError`` /
    ``InvalidFilenameError`` and renders the appropriate 4xx envelope.
    """
    original_name = upload_file.filename or ""
    if not original_name.strip():
        raise InvalidFilenameError("upload missing filename")

    # Pre-check aggregate storage cap. We can't know the final size until
    # the stream completes, but we can reject early when the current total
    # is already at the cap; the mid-stream check below catches the rest.
    current_total = total_bytes_used(conn)
    if current_total >= max_total_bytes:
        raise StorageCapExceededError(
            current_total=current_total,
            attempted_size=0,
            max_bytes=max_total_bytes,
        )

    file_id = str(ULID())
    safe_name = sanitize_filename(original_name)

    files_root = data_dir / "data" / "files" / file_id
    files_root.mkdir(parents=True, exist_ok=True)
    final_path = files_root / safe_name
    temp_path = files_root / f".{safe_name}.tmp"

    sha = hashlib.sha256()
    size = 0
    try:
        with open(temp_path, "wb") as fh:
            while True:
                chunk = await upload_file.read(_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_per_file_bytes:
                    raise FileTooLargeError(
                        size_bytes=size, max_bytes=max_per_file_bytes
                    )
                # Mid-stream aggregate cap: the contributing total grows.
                if current_total + size > max_total_bytes:
                    raise StorageCapExceededError(
                        current_total=current_total,
                        attempted_size=size,
                        max_bytes=max_total_bytes,
                    )
                fh.write(chunk)
                sha.update(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, final_path)
    except Exception:
        # Clean up partials so a misbehaving / mid-cap-exceeded upload
        # doesn't leave bytes on disk.
        for p in (temp_path, final_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            files_root.rmdir()
        except OSError:
            pass
        raise

    mime_type = detect_mime_type(safe_name, upload_file.content_type)
    metadata = insert_file(
        conn,
        file_id=file_id,
        original_name=original_name,
        mime_type=mime_type,
        size_bytes=size,
        path=str(final_path),
        sha256=sha.hexdigest(),
    )
    return UploadResult(metadata=metadata, on_disk_path=final_path)
