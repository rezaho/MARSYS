"""POST /v1/workflows/{id}/lint — non-blocking lint surface for the canvas.

The endpoint lints the **workflow definition supplied in the request
body** (the live canvas), runs the Spren-side linter (which produces
structured findings the UI can pin to nodes and edges), and returns
``{"findings": [...]}``.

WF-BUG-LINT-REACTIVITY: this previously linted the *stored* definition
(``fetch_workflow`` by id, no body), so a fix made on the canvas was
never reflected until it was saved AND the lint effect happened to
re-fire — the chip reported errors that no longer existed until a page
reload. A linter must lint the artifact you are editing, not the last
save; the definition is now the request body. The ``{workflow_id}`` path
segment is kept for route identity/observability but is not read — lint
is a pure function of the submitted definition (it also now works for a
brand-new, never-saved canvas, which previously 404'd). Invalid bodies
surface as the app-wide ``VALIDATION_FAILED`` envelope (422).
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends

from spren.lint import lint_workflow
from spren.models import LintResponse, WorkflowDefinition


def make_lint_router(
    require_auth: Callable[..., Any],
    *,
    known_tools_provider: Callable[[], list[str]],
) -> APIRouter:
    """Build the workflow-lint router.

    ``known_tools_provider`` is injected so the linter sees the same
    registry the ``GET /v1/tools`` endpoint exposes — the route can't
    pull the list itself without re-importing the framework, which we've
    already done once at app construction.
    """
    router = APIRouter(prefix="/v1/workflows", dependencies=[Depends(require_auth)])

    @router.post(
        "/{workflow_id}/lint",
        response_model=LintResponse,
        responses={200: {"model": LintResponse}},
    )
    def lint_endpoint(
        workflow_id: str,
        definition: WorkflowDefinition,
    ) -> Any:
        del workflow_id  # route identity only; lint is body-driven
        findings = lint_workflow(definition, known_tools=known_tools_provider())
        return LintResponse(findings=findings)

    return router
