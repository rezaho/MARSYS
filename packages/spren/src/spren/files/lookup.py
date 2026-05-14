"""Resolve ``file_id`` → ``(metadata, on_disk_path)``.

Used by the ``GET /v1/files/{id}/download`` endpoint, by the run
lifecycle coordinator's attachments-resolution step (re-resolves the
file paths to build the system-context block), and by the ``DELETE``
reference-check (consults ``runs_referencing_file``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, NamedTuple

from spren.models import FileMetadata
from spren.storage.files import fetch_file_path


class ResolvedAttachment(NamedTuple):
    """One ``file_id`` resolved to its on-disk path + metadata."""

    file_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    path: Path


def resolve_attachment(
    conn: sqlite3.Connection, file_id: str
) -> ResolvedAttachment | None:
    """Return resolved attachment or ``None`` if the ``file_id`` is unknown."""
    result = fetch_file_path(conn, file_id)
    if result is None:
        return None
    metadata, path_str = result
    return ResolvedAttachment(
        file_id=metadata.id,
        original_name=metadata.original_name,
        mime_type=metadata.mime_type,
        size_bytes=metadata.size_bytes,
        path=Path(path_str),
    )


def resolve_attachments(
    conn: sqlite3.Connection, file_ids: Iterable[str]
) -> tuple[list[ResolvedAttachment], list[str]]:
    """Bulk-resolve ``file_ids``.

    Returns ``(resolved, missing)`` so the caller can choose between
    "fail fast on first miss" (POST handler) and "resolve survivors only"
    (re-run with stale attachment confirmation).
    """
    resolved: list[ResolvedAttachment] = []
    missing: list[str] = []
    for fid in file_ids:
        att = resolve_attachment(conn, fid)
        if att is None:
            missing.append(fid)
        else:
            resolved.append(att)
    return resolved, missing


def fetch_file_metadata(conn: sqlite3.Connection, file_id: str) -> FileMetadata | None:
    """Wire-shape metadata (no on-disk path)."""
    result = fetch_file_path(conn, file_id)
    return result[0] if result is not None else None


def format_attachments_block(resolved: list[ResolvedAttachment]) -> str:
    """Build the system-context block appended to ``task_input.text``.

    Format matches plan §3:

      Files attached to this run:
      - report.pdf (/.../report.pdf, application/pdf, 1.2 MB)
      - data.csv (/.../data.csv, text/csv, 87 KB)

      Use the `read_file` tool to access them.
    """
    if not resolved:
        return ""
    lines = ["", "Files attached to this run:"]
    for att in resolved:
        size_human = _human_size(att.size_bytes)
        lines.append(
            f"- {att.original_name} ({att.path}, {att.mime_type}, {size_human})"
        )
    lines.append("")
    lines.append("Use the `read_file` tool to access them.")
    return "\n".join(lines)


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"
