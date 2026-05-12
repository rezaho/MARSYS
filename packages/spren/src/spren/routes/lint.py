"""POST /v1/workflows/{id}/lint — non-blocking lint surface for the canvas.

The endpoint loads a stored workflow, runs the Spren-side linter (which
produces structured findings the UI can pin to nodes and edges), and returns
``{"findings": [...]}``. Lint never raises 5xx for a known workflow id; only
401 (auth) and 404 (unknown id) are non-200.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Callable

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from spren.lint import lint_workflow
from spren.models import (
    ErrorCode,
    ErrorEnvelope,
    ErrorPayload,
    LintResponse,
)
from spren.storage import workflows


def _error(code: ErrorCode, message: str, status_code: int) -> JSONResponse:
    payload = ErrorEnvelope(error=ErrorPayload(code=code, message=message, details={}))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def make_lint_router(
    require_auth: Callable[..., Any],
    db_factory: Callable[[], sqlite3.Connection],
    *,
    known_tools_provider: Callable[[], list[str]],
) -> APIRouter:
    """Build the workflow-lint router.

    ``known_tools_provider`` is injected so the linter sees the same registry
    the ``GET /v1/tools`` endpoint exposes — the route can't pull the list
    itself without re-importing the framework, which we've already done once
    at app construction.
    """
    router = APIRouter(prefix="/v1/workflows", dependencies=[Depends(require_auth)])

    def get_conn() -> sqlite3.Connection:
        return db_factory()

    @router.post(
        "/{workflow_id}/lint",
        response_model=LintResponse,
        responses={
            200: {"model": LintResponse},
            404: {"model": ErrorEnvelope},
        },
    )
    def lint_endpoint(
        workflow_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Any:
        workflow = workflows.fetch_workflow(conn, workflow_id)
        if workflow is None:
            return _error(
                "WORKFLOW_NOT_FOUND",
                f"workflow {workflow_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        findings = lint_workflow(workflow.definition, known_tools=known_tools_provider())
        return LintResponse(findings=findings)

    return router
