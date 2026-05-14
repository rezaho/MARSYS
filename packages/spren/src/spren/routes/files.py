"""File upload + metadata + download + delete routes.

REST under ``/v1/files``:

- ``POST   /v1/files``               — multipart upload; streams to disk; returns metadata + sha256
- ``GET    /v1/files/{id}``          — metadata only (no on-disk path)
- ``GET    /v1/files/{id}/download`` — raw bytes; ``Content-Disposition: attachment``
- ``DELETE /v1/files/{id}``          — soft-rejects if referenced by any run (409); else hard-deletes

All routes require auth (router-level dependency); CORS regex is set
app-wide in ``server.create_app``.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response

from spren.files import upload as upload_module
from spren.files.lookup import fetch_file_metadata, resolve_attachment
from spren.files.upload import (
    FileTooLargeError,
    InvalidFilenameError,
    StorageCapExceededError,
    stream_upload,
)
from spren.models import (
    ErrorCode,
    ErrorEnvelope,
    ErrorPayload,
    FileMetadata,
    FileUploadResponse,
)
from spren.storage.files import (
    delete_file_row,
    fetch_file_path,
    runs_referencing_file,
)

logger = logging.getLogger(__name__)


def _error(
    code: ErrorCode,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload = ErrorEnvelope(
        error=ErrorPayload(code=code, message=message, details=details or {})
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def make_files_router(
    require_auth: Callable[..., Any],
    db_factory: Callable[[], sqlite3.Connection],
    data_dir: Path,
) -> APIRouter:
    """Build the /v1/files router."""

    router = APIRouter(prefix="/v1/files", dependencies=[Depends(require_auth)])

    def get_conn() -> sqlite3.Connection:
        return db_factory()

    # ---- POST /v1/files ----

    @router.post(
        "",
        responses={
            201: {"model": FileUploadResponse},
            400: {"model": ErrorEnvelope},
            413: {"model": ErrorEnvelope},
        },
        status_code=status.HTTP_201_CREATED,
    )
    async def upload(
        file: UploadFile = File(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        try:
            # Read cap values fresh on each call so monkeypatching them
            # in tests works as expected (function-default capture would
            # freeze the cap at module-load time).
            result = await stream_upload(
                conn=conn,
                upload_file=file,
                data_dir=data_dir,
                max_per_file_bytes=upload_module.DEFAULT_MAX_PER_FILE_BYTES,
                max_total_bytes=upload_module.DEFAULT_MAX_TOTAL_BYTES,
            )
        except InvalidFilenameError as exc:
            return _error(
                "INVALID_FILENAME",
                str(exc),
                status.HTTP_400_BAD_REQUEST,
            )
        except FileTooLargeError as exc:
            return _error(
                "FILE_TOO_LARGE",
                str(exc),
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={"size_bytes": exc.size_bytes, "max_bytes": exc.max_bytes},
            )
        except StorageCapExceededError as exc:
            return _error(
                "STORAGE_CAP_EXCEEDED",
                str(exc),
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={
                    "current_total": exc.current_total,
                    "attempted_size": exc.attempted_size,
                    "max_bytes": exc.max_bytes,
                },
            )
        conn.commit()
        body = FileUploadResponse(
            file_id=result.metadata.id,
            original_name=result.metadata.original_name,
            mime_type=result.metadata.mime_type,
            size_bytes=result.metadata.size_bytes,
            sha256=result.metadata.sha256,
        ).model_dump(mode="json")
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=body)

    # ---- GET /v1/files/{id} ----

    @router.get(
        "/{file_id}",
        response_model=FileMetadata,
        responses={404: {"model": ErrorEnvelope}},
    )
    def get_metadata(
        file_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        meta = fetch_file_metadata(conn, file_id)
        if meta is None:
            return _error(
                "FILE_NOT_FOUND",
                f"file {file_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(content=meta.model_dump(mode="json"))

    # ---- GET /v1/files/{id}/download ----

    @router.get(
        "/{file_id}/download",
        response_class=FileResponse,
        responses={404: {"model": ErrorEnvelope}},
    )
    def download(
        file_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        att = resolve_attachment(conn, file_id)
        if att is None:
            return _error(
                "FILE_NOT_FOUND",
                f"file {file_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        # Path-confine: the resolved file MUST live under <data-dir>/data/files/.
        # Defends against a tampered-DB row pointing outside the files root.
        files_root = (data_dir / "data" / "files").resolve()
        try:
            resolved = att.path.resolve()
        except (OSError, RuntimeError):
            return _error(
                "FILE_NOT_FOUND",
                f"file {file_id} on-disk path is not accessible",
                status.HTTP_404_NOT_FOUND,
            )
        if not _is_subpath(resolved, files_root) or not resolved.exists():
            return _error(
                "FILE_NOT_FOUND",
                f"file {file_id} on-disk content is missing",
                status.HTTP_404_NOT_FOUND,
            )
        return FileResponse(
            path=resolved,
            media_type=att.mime_type,
            filename=att.original_name,
        )

    # ---- DELETE /v1/files/{id} ----

    @router.delete(
        "/{file_id}",
        responses={
            204: {"description": "Deleted"},
            404: {"model": ErrorEnvelope},
            409: {"model": ErrorEnvelope},
        },
    )
    def delete_handler(
        file_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        result = fetch_file_path(conn, file_id)
        if result is None:
            return _error(
                "FILE_NOT_FOUND",
                f"file {file_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        _, on_disk_path = result
        # Reference check first — element-equality via json_each.
        run_ids = runs_referencing_file(conn, file_id)
        if run_ids:
            return _error(
                "FILE_REFERENCED_BY_RUNS",
                f"file {file_id} is referenced by {len(run_ids)} run(s)",
                status.HTTP_409_CONFLICT,
                details={"run_ids": run_ids},
            )
        delete_file_row(conn, file_id)
        conn.commit()
        # Best-effort on-disk cleanup. The row deletion is the source of truth;
        # an orphaned file is benign (sweepers can pick it up later).
        try:
            disk_path = Path(on_disk_path)
            disk_path.unlink(missing_ok=True)
            try:
                disk_path.parent.rmdir()
            except OSError:
                pass
        except OSError:
            logger.warning(
                "delete_file: row removed but on-disk cleanup failed for %s",
                file_id,
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


def _is_subpath(child: Path, parent: Path) -> bool:
    """Path-confinement check: ``child`` must be inside ``parent``.

    Both paths are assumed pre-resolved.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
