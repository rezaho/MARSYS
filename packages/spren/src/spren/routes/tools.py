"""GET /v1/tools — tool registry surface for the canvas agent-config form.

v0.3 returns framework tools only (``marsys.environment.tools.AVAILABLE_TOOLS``).
Spren-side tools and user-authored tools arrive in v0.4 (see SP-019 +
``docs/implementation/spren/sessions/v0.3.0/03-visual-builder.md`` §8 Q9).

The registry is import-time-static on the framework side, so the response is
computed once per process and cached. The cache lives in the router closure;
each FastAPI worker has its own copy, which is fine because the framework
module itself is loaded once per worker.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends

from spren.models import ToolInfo, ToolListResponse


def _first_doc_line(callable_: Any) -> str | None:
    """First non-empty stripped line of the callable's ``__doc__``.

    Returns ``None`` when the callable has no docstring or the docstring is
    blank. We deliberately drop multi-paragraph framework docstrings — v0.3's
    tool picker shows a single line; full descriptions arrive with the v0.4
    parameter-schema work.
    """
    doc = getattr(callable_, "__doc__", None)
    if not doc:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def make_tools_router(require_auth: Callable[..., Any]) -> APIRouter:
    """Build the /v1/tools router with explicit auth-dep injection."""

    router = APIRouter(prefix="/v1/tools", dependencies=[Depends(require_auth)])

    # Import lazily so test fixtures that monkey-patch the framework registry
    # see their patched view, AND so the router factory remains importable in
    # environments without marsys installed (the FastAPI app fails to start
    # earlier in those cases — this is the natural ordering).
    from marsys.environment.tools import AVAILABLE_TOOLS  # type: ignore[import-not-found]

    _cached_items: list[ToolInfo] = sorted(
        (
            ToolInfo(name=name, source="framework", description=_first_doc_line(fn))
            for name, fn in AVAILABLE_TOOLS.items()
        ),
        key=lambda info: info.name,
    )

    @router.get("", response_model=ToolListResponse)
    def list_tools() -> ToolListResponse:
        return ToolListResponse(items=_cached_items)

    return router
