"""Artifact listing + per-artifact download for ``GET /v1/runs/{id}/artifacts*``.

Artifacts live at ``<data-dir>/data/runs/{run_id}/artifacts/`` (created
by future framework tools that produce structured outputs). v0.3 ships
no in-tree tool that writes artifacts, so the common case is an empty
directory.

Path traversal is hardened via ``pathlib.Path.resolve()`` + parent-dir
containment check (per plan §10.13). The filename is also pre-validated
against ``_FILENAME_RE`` to reject ``..``, slashes, and backslashes
before any path resolution happens.
"""
from __future__ import annotations

import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from spren.models import ArtifactInfo


# Plan §10.13: filename allow-list. Reject any char that could form a
# path-traversal — confines ``name`` to a single filename component.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ArtifactNotFoundError(Exception):
    """The named artifact does not exist (or is outside the artifacts dir)."""


class InvalidArtifactNameError(Exception):
    """The provided artifact name failed the safety check."""


def list_artifacts(*, data_dir: Path, run_id: str) -> List[ArtifactInfo]:
    """List artifact files for the run; empty list if the dir doesn't exist."""
    artifacts_dir = data_dir / "data" / "runs" / run_id / "artifacts"
    if not artifacts_dir.is_dir():
        return []
    items: List[ArtifactInfo] = []
    for entry in sorted(artifacts_dir.iterdir()):
        if not entry.is_file():
            continue
        stat = entry.stat()
        mime, _ = mimetypes.guess_type(entry.name)
        items.append(
            ArtifactInfo(
                name=entry.name,
                size_bytes=stat.st_size,
                mime_type=mime or "application/octet-stream",
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            )
        )
    return items


def resolve_artifact_path(*, data_dir: Path, run_id: str, name: str) -> Path:
    """Resolve a per-run artifact filename to its on-disk path.

    Path-confined: rejects URL-encoded path traversal + filenames
    outside the artifacts dir. Raises ``InvalidArtifactNameError`` for
    rejected input, ``ArtifactNotFoundError`` for not-on-disk.
    """
    if not _FILENAME_RE.match(name):
        raise InvalidArtifactNameError(f"invalid artifact name: {name!r}")

    artifacts_dir = (data_dir / "data" / "runs" / run_id / "artifacts").resolve()
    candidate = (artifacts_dir / name).resolve()

    # Belt-and-braces: even though the regex blocks `..`, verify the
    # resolved path is under the artifacts dir.
    try:
        candidate.relative_to(artifacts_dir)
    except ValueError:
        raise InvalidArtifactNameError(
            f"resolved path escapes artifacts directory: {name!r}"
        )

    if not candidate.is_file():
        raise ArtifactNotFoundError(f"artifact {name!r} not found in run {run_id}")

    return candidate
