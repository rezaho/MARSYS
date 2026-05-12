"""Workflow CRUD + Python-file import endpoints.

All endpoints under ``/v1/workflows`` require the per-launch auth token (route
level via ``APIRouter(dependencies=[Depends(require_auth)])`` — no
``Annotated[None, Depends(...)]`` because that surface returns 422 not 401).

Pagination uses the ULID of the last returned row as an opaque cursor; the
server queries ``WHERE id > :cursor ORDER BY id`` and trims the result. The
``Idempotency-Key`` header is honored for POST/PUT/PATCH/DELETE: replays
within 24 hours return the cached body + status; cache key is
``(method, path, key)`` so cross-method/cross-path collisions are treated as
fresh.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Callable

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from ulid import ULID

from spren.importers.python_workflow import (
    MAX_IMPORT_BYTES,
    PythonImportError,
    parse_python_workflow,
)
from spren.models import (
    ErrorCode,
    ErrorEnvelope,
    ErrorPayload,
    ImportWarningPayload,
    Workflow,
    WorkflowCreateRequest,
    WorkflowImportResponse,
    WorkflowListResponse,
    WorkflowProvenance,
    WorkflowUpdateRequest,
)
from spren.storage import idempotency, workflows

PROVENANCE_VALUES: tuple[WorkflowProvenance, ...] = (
    "visual_builder",
    "meta_agent",
    "code_import",
    "template",
    "api",
)


def _error(code: ErrorCode, message: str, status_code: int, details: dict[str, Any] | None = None) -> JSONResponse:
    payload = ErrorEnvelope(error=ErrorPayload(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def make_workflows_router(
    require_auth: Callable[..., Any],
    db_factory: Callable[[], sqlite3.Connection],
) -> APIRouter:
    """Build the /v1/workflows router with explicit dep injection."""

    router = APIRouter(prefix="/v1/workflows", dependencies=[Depends(require_auth)])

    def get_conn() -> sqlite3.Connection:
        return db_factory()

    # ---- Cursor-pagination list ----

    @router.get(
        "",
        response_model=WorkflowListResponse,
        responses={400: {"model": ErrorEnvelope}},
    )
    def list_workflows(
        conn: sqlite3.Connection = Depends(get_conn),
        cursor: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
        provenance: WorkflowProvenance | None = Query(default=None),
        archived: bool | None = Query(default=None),
        include_drafts: bool = Query(default=False),
    ) -> Response:
        items, has_more = workflows.list_workflows(
            conn,
            cursor=cursor,
            limit=limit,
            provenance=provenance,
            archived=archived,
            include_drafts=include_drafts,
        )
        next_cursor = items[-1].id if has_more and items else None
        body = WorkflowListResponse(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
        )
        return JSONResponse(content=body.model_dump(mode="json"))

    # ---- Create ----

    @router.post(
        "",
        responses={
            201: {"model": Workflow},
            422: {"model": ErrorEnvelope},
        },
        status_code=status.HTTP_201_CREATED,
    )
    async def create_workflow(
        payload: WorkflowCreateRequest,
        request: Request,
        conn: sqlite3.Connection = Depends(get_conn),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        cached = _maybe_replay(conn, request, idempotency_key)
        if cached is not None:
            return cached

        workflow = workflows.insert_workflow(
            conn,
            workflow_id=str(ULID()),
            name=payload.name,
            description=payload.description,
            definition=payload.definition,
            provenance=payload.provenance,
            provenance_metadata=payload.provenance_metadata,
        )
        body = workflow.model_dump(mode="json")
        response = JSONResponse(status_code=status.HTTP_201_CREATED, content=body)
        _maybe_store(conn, request, idempotency_key, response, body)
        return response

    # ---- Read ----

    @router.get(
        "/{workflow_id}",
        response_model=Workflow,
        responses={404: {"model": ErrorEnvelope}},
    )
    def read_workflow(workflow_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> Response:
        wf = workflows.fetch_workflow(conn, workflow_id)
        if wf is None:
            return _error(
                "WORKFLOW_NOT_FOUND",
                f"workflow {workflow_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(content=wf.model_dump(mode="json"))

    # ---- Replace (PUT) ----

    @router.put(
        "/{workflow_id}",
        responses={
            200: {"model": Workflow},
            404: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
        },
    )
    async def replace_workflow(
        workflow_id: str,
        payload: WorkflowCreateRequest,
        request: Request,
        conn: sqlite3.Connection = Depends(get_conn),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        cached = _maybe_replay(conn, request, idempotency_key)
        if cached is not None:
            return cached

        existing = workflows.fetch_workflow(conn, workflow_id)
        if existing is None:
            return _error(
                "WORKFLOW_NOT_FOUND",
                f"workflow {workflow_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )

        updated = workflows.replace_workflow(
            conn,
            workflow_id,
            name=payload.name,
            description=payload.description,
            definition=payload.definition,
            provenance=payload.provenance,
            provenance_metadata=payload.provenance_metadata,
        )
        assert updated is not None
        body = updated.model_dump(mode="json")
        response = JSONResponse(content=body)
        _maybe_store(conn, request, idempotency_key, response, body)
        return response

    # ---- Patch ----

    @router.patch(
        "/{workflow_id}",
        responses={
            200: {"model": Workflow},
            404: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
        },
    )
    async def patch_workflow(
        workflow_id: str,
        payload: WorkflowUpdateRequest,
        request: Request,
        conn: sqlite3.Connection = Depends(get_conn),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        cached = _maybe_replay(conn, request, idempotency_key)
        if cached is not None:
            return cached

        existing = workflows.fetch_workflow(conn, workflow_id)
        if existing is None:
            return _error(
                "WORKFLOW_NOT_FOUND",
                f"workflow {workflow_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )

        fields: dict[str, Any] = {}
        # Use only fields the client explicitly set so PATCH semantics hold.
        provided = payload.model_dump(exclude_unset=True)
        if "name" in provided:
            fields["name"] = payload.name
        if "description" in provided:
            fields["description"] = payload.description
        if "definition" in provided:
            fields["definition"] = payload.definition
        if "is_archived" in provided:
            fields["is_archived"] = payload.is_archived
        if "provenance_metadata" in provided:
            fields["provenance_metadata"] = payload.provenance_metadata

        updated = workflows.patch_workflow(conn, workflow_id, fields=fields)
        assert updated is not None
        body = updated.model_dump(mode="json")
        response = JSONResponse(content=body)
        _maybe_store(conn, request, idempotency_key, response, body)
        return response

    # ---- Delete ----

    @router.delete(
        "/{workflow_id}",
        responses={
            204: {"description": "deleted"},
            404: {"model": ErrorEnvelope},
            409: {"model": ErrorEnvelope},
        },
    )
    def delete_workflow(
        workflow_id: str,
        request: Request,
        conn: sqlite3.Connection = Depends(get_conn),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        cached = _maybe_replay(conn, request, idempotency_key)
        if cached is not None:
            return cached

        existing = workflows.fetch_workflow(conn, workflow_id)
        if existing is None:
            return _error(
                "WORKFLOW_NOT_FOUND",
                f"workflow {workflow_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )

        if workflows.count_runs_referencing(conn, workflow_id) > 0:
            return _error(
                "WORKFLOW_HAS_RUNS",
                f"workflow {workflow_id} has runs referencing it; archive instead",
                status.HTTP_409_CONFLICT,
            )

        workflows.delete_workflow(conn, workflow_id)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        _maybe_store(conn, request, idempotency_key, response, b"")
        return response

    # ---- Python-file importer ----

    @router.post(
        "/import-python",
        responses={
            201: {"model": WorkflowImportResponse},
            413: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
        },
        status_code=status.HTTP_201_CREATED,
    )
    async def import_python_workflow(
        request: Request,
        file: UploadFile = File(...),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        cached = _maybe_replay(conn, request, idempotency_key)
        if cached is not None:
            return cached

        # Stream-bounded read. Fail loud if the upload exceeds the budget.
        first_chunk = await file.read(MAX_IMPORT_BYTES + 1)
        if len(first_chunk) > MAX_IMPORT_BYTES:
            return _error(
                "PYTHON_IMPORT_REJECTED",
                f"upload exceeds {MAX_IMPORT_BYTES} bytes",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                {"reason": "too_large", "max_bytes": MAX_IMPORT_BYTES},
            )

        try:
            source = first_chunk.decode("utf-8")
        except UnicodeDecodeError:
            return _error(
                "PYTHON_IMPORT_REJECTED",
                "file must be UTF-8 encoded",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                {"reason": "non_utf8"},
            )

        try:
            result = parse_python_workflow(source)
        except PythonImportError as exc:
            return _error(
                "PYTHON_IMPORT_REJECTED",
                exc.message,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                exc.details,
            )

        sha256 = hashlib.sha256(first_chunk).hexdigest()
        provenance_metadata: dict[str, Any] = {
            "source_filename": file.filename or "uploaded.py",
            "sha256": sha256,
        }
        if result.description:
            provenance_metadata["source_description"] = result.description

        workflow = workflows.insert_workflow(
            conn,
            workflow_id=str(ULID()),
            name=result.name,
            description=result.description,
            definition=result.definition,
            provenance="code_import",
            provenance_metadata=provenance_metadata,
        )
        warning_payloads = [
            ImportWarningPayload(
                code=w.code,  # type: ignore[arg-type] -- literal-set narrowing
                source=w.source,
                target=w.target,
                original_pattern=w.original_pattern,
                message=w.message,
            )
            for w in result.warnings
        ]
        envelope = WorkflowImportResponse(workflow=workflow, warnings=warning_payloads)
        body = envelope.model_dump(mode="json")
        response = JSONResponse(status_code=status.HTTP_201_CREATED, content=body)
        _maybe_store(conn, request, idempotency_key, response, body)
        return response

    return router


# ---- Idempotency helpers ----

def _maybe_replay(
    conn: sqlite3.Connection,
    request: Request,
    key: str | None,
) -> Response | None:
    if not key:
        return None
    cached = idempotency.fetch(
        conn,
        method=request.method,
        path=request.url.path,
        key=key,
    )
    if cached is None:
        return None
    return Response(
        content=cached.body,
        status_code=cached.status,
        headers=cached.headers,
    )


def _maybe_store(
    conn: sqlite3.Connection,
    request: Request,
    key: str | None,
    response: Response,
    body: Any,
) -> None:
    if not key:
        return
    body_bytes: bytes
    if isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    elif isinstance(body, str):
        body_bytes = body.encode("utf-8")
    elif isinstance(response, JSONResponse):
        body_bytes = response.body if isinstance(response.body, bytes) else json.dumps(body).encode("utf-8")
    elif isinstance(body, dict):
        body_bytes = json.dumps(body).encode("utf-8")
    else:
        body_bytes = b""
    idempotency.store(
        conn,
        method=request.method,
        path=request.url.path,
        key=key,
        status=response.status_code,
        body=body_bytes,
        headers={"content-type": response.headers.get("content-type", "application/json")},
    )
