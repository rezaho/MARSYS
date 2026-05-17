"""Run execution + inspection endpoints.

REST + SSE under /v1/runs:

- ``POST   /v1/runs``                  — create run; race-free wiring
- ``GET    /v1/runs``                  — cursor-paginated list with filters
- ``GET    /v1/runs/{id}``             — read run row
- ``POST   /v1/runs/{id}/cancel``      — cancel (queued → cancelled, running → cancelling → cancelled)
- ``GET    /v1/runs/{id}/events``      — per-run SSE (AG-UI events; gated on Framework 06)
- ``GET    /v1/runs/events``           — aggregate SSE for the list page (RunsBroker)

All routes require auth (router-level dependency); CORS regex is set
app-wide in ``server.create_app``.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from ulid import ULID

from spren.models import (
    ArtifactInfo,
    ArtifactListResponse,
    ErrorCode,
    ErrorEnvelope,
    ErrorPayload,
    RunCancelledEvent,
    RunCreate,
    RunCreateResponse,
    RunCreatedEvent,
    RunFinishedEvent,
    RunListResponse,
    RunRead,
    RunStatus,
    RunTrace,
    RunUpdatedEvent,
    TERMINAL_STATUSES,
    WorkflowDefinition,
)
from spren.files.lookup import fetch_file_metadata
from spren.runs import lifecycle as lifecycle_module
from spren.runs.broker import RunsBroker, StreamLaggedMarker
from spren.runs.lifecycle import (
    ActiveRun,
    cancel_run,
    register_run,
    schedule_run,
    _freeze_workflow_snapshot,
    _to_list_item,
)
from spren.runs.artifacts import (
    ArtifactNotFoundError,
    InvalidArtifactNameError,
    list_artifacts,
    resolve_artifact_path,
)
from spren.runs.materialize import MaterializationError, materialize_run
from spren.runs.sse import stream_run_events
from spren.runs.trace import TraceNotAvailableError, build_run_trace
from spren.storage import workflows as workflows_store
from spren.storage.runs import fetch_run, insert_run, list_runs

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


def make_runs_router(
    require_auth: Callable[..., Any],
    db_factory: Callable[[], sqlite3.Connection],
    broker: RunsBroker,
    data_dir: Path,
) -> APIRouter:
    """Build the /v1/runs router."""

    router = APIRouter(prefix="/v1/runs", dependencies=[Depends(require_auth)])

    def get_conn() -> sqlite3.Connection:
        return db_factory()

    # ---- Create run ----

    @router.post(
        "",
        responses={
            201: {"model": RunCreateResponse},
            400: {"model": ErrorEnvelope},
            404: {"model": ErrorEnvelope},
        },
        status_code=status.HTTP_201_CREATED,
    )
    async def create_run(
        payload: RunCreate,
        request: Request,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        # Validate trigger
        if payload.trigger != "manual":
            return _error(
                "TRIGGER_NOT_YET_SUPPORTED",
                f"trigger={payload.trigger!r} not supported in v0.3 (manual only)",
                status.HTTP_400_BAD_REQUEST,
            )
        # Synchronously validate task_input.attachments — Session 05 enables
        # non-empty arrays; each file_id must exist before the 201 returns
        # so an unknown id surfaces as 400 here, not as RUN_FAILED 100ms
        # later. Plan §3 + decision §19.
        if payload.task_input.attachments:
            for fid in payload.task_input.attachments:
                if fetch_file_metadata(conn, fid) is None:
                    return _error(
                        "ATTACHMENT_NOT_FOUND",
                        f"attached file {fid!r} does not exist",
                        status.HTTP_400_BAD_REQUEST,
                        details={"file_id": fid},
                    )

        workflow = workflows_store.fetch_workflow(conn, payload.workflow_id)
        if workflow is None:
            return _error(
                "WORKFLOW_NOT_FOUND",
                f"workflow {payload.workflow_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        if workflow.is_archived:
            return _error(
                "WORKFLOW_ARCHIVED",
                f"workflow {payload.workflow_id} is archived; archive must be reverted before running",
                status.HTTP_400_BAD_REQUEST,
            )

        # Generate run_id first so materialize_run can wire per-run tracing
        # (TracingConfig.output_dir = <data-dir>/data/runs/{run_id}).
        run_id = str(ULID())

        # Materialize before inserting so a materialization error doesn't leave
        # an orphan queued row.
        try:
            bundle = materialize_run(
                definition=workflow.definition,
                data_dir=data_dir,
                run_id=run_id,
            )
        except MaterializationError as exc:
            return _error(
                "VALIDATION_FAILED",
                str(exc),
                status.HTTP_400_BAD_REQUEST,
            )

        # Freeze workflow snapshot to disk (SP-009)
        _freeze_workflow_snapshot(
            run_id=run_id,
            definition_json=workflow.definition.model_dump_json(),
            data_dir=data_dir,
        )
        # Insert runs row
        run_row = insert_run(
            conn,
            run_id=run_id,
            workflow_id=payload.workflow_id,
            task_input=payload.task_input,
            trigger=payload.trigger,
        )
        conn.commit()

        # Construct + wire Orchestra synchronously so SSE subscribers find a
        # wired translator before the 201 returns.
        active = await register_run(
            run_id=run_id,
            workflow_id=payload.workflow_id,
            task_input=payload.task_input,
            bundle=bundle,
            data_dir=data_dir,
            db_factory=db_factory,
            trigger=payload.trigger,
        )
        # Schedule the lifecycle task (sets status → running, awaits Orchestra,
        # writes terminal state).
        schedule_run(
            active=active,
            task_input=payload.task_input,
            db_factory=db_factory,
            broker=broker,
        )

        # Publish RunCreated to the aggregate stream.
        broker.publish(RunCreatedEvent(run=_to_list_item(run_row)))

        body = RunCreateResponse(run_id=run_id, status=RunStatus.QUEUED).model_dump(mode="json")
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=body)

    # ---- List runs ----

    @router.get(
        "",
        response_model=RunListResponse,
        responses={400: {"model": ErrorEnvelope}},
    )
    def list_runs_handler(
        conn: sqlite3.Connection = Depends(get_conn),
        cursor: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
        workflow_id: str | None = Query(default=None),
        run_status: str | None = Query(default=None, alias="status"),
        since: datetime | None = Query(default=None),
        until: datetime | None = Query(default=None),
    ) -> Response:
        # Session 05 multi-value semantics: ``workflow_id`` and ``status``
        # accept comma-separated lists. Backward-compat with Session 04 —
        # a value without a comma is treated as a single-element list and
        # routed through the original single-value path.
        wf_ids: list[str] | None = None
        wf_id_single: str | None = None
        if workflow_id is not None:
            parts = [s.strip() for s in workflow_id.split(",") if s.strip()]
            if not parts:
                pass
            elif len(parts) == 1:
                wf_id_single = parts[0]
            else:
                wf_ids = parts

        statuses_multi: list[RunStatus] | None = None
        status_single: RunStatus | None = None
        if run_status is not None:
            try:
                parts = [s.strip() for s in run_status.split(",") if s.strip()]
                if not parts:
                    pass
                elif len(parts) == 1:
                    status_single = RunStatus(parts[0])
                else:
                    statuses_multi = [RunStatus(p) for p in parts]
            except ValueError as exc:
                return _error(
                    "VALIDATION_FAILED",
                    f"invalid status value: {exc}",
                    status.HTTP_400_BAD_REQUEST,
                )

        items, has_more = list_runs(
            conn,
            cursor=cursor,
            limit=limit,
            workflow_id=wf_id_single,
            workflow_ids=wf_ids,
            status=status_single,
            statuses=statuses_multi,
            since=since,
            until=until,
        )
        next_cursor = items[-1].id if has_more and items else None
        body = RunListResponse(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
        )
        return JSONResponse(content=body.model_dump(mode="json"))

    # ---- Schema export ----
    # The aggregate SSE endpoint emits JSON payloads of the discriminated
    # union; SSE responses don't auto-surface their payload types in the
    # OpenAPI schema, so we expose them via a tiny meta-endpoint that
    # returns a sample of each type. Clients use the generated types to
    # parse the SSE stream client-side.

    class _RunsEventSchemas(BaseModel):
        created: RunCreatedEvent | None = None
        updated: RunUpdatedEvent | None = None
        finished: RunFinishedEvent | None = None
        cancelled: RunCancelledEvent | None = None

    @router.get("/_event_schemas", response_model=_RunsEventSchemas)
    def event_schemas() -> _RunsEventSchemas:
        """Returns null-shaped schemas for the aggregate SSE event union.

        Forces the OpenAPI schema to include `RunCreatedEvent` /
        `RunUpdatedEvent` / `RunFinishedEvent` / `RunCancelledEvent` so
        the generated TypeScript client can discriminate them.
        """
        return _RunsEventSchemas()

    # ---- Aggregate SSE ----
    # Defined BEFORE /{run_id} so /events isn't shadowed.

    @router.get("/events", response_model=None)
    async def aggregate_events(request: Request) -> Response:
        async def event_gen() -> AsyncIterator[dict[str, Any]]:
            async with broker.subscription() as sub:
                while not await request.is_disconnected():
                    item = await sub.get()
                    if isinstance(item, StreamLaggedMarker):
                        yield {
                            "event": "marsys.stream.lagged",
                            "data": '{"dropped_count": %d}' % item.dropped_count,
                        }
                        continue
                    yield {
                        "event": item.type,
                        "data": item.model_dump_json(),
                    }

        return EventSourceResponse(event_gen())

    # ---- Read run ----

    @router.get(
        "/{run_id}",
        response_model=RunRead,
        responses={404: {"model": ErrorEnvelope}},
    )
    def read_run(run_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> Response:
        run = fetch_run(conn, run_id)
        if run is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(content=run.model_dump(mode="json"))

    # ---- Run trace ----

    @router.get(
        "/{run_id}/trace",
        response_model=RunTrace,
        responses={404: {"model": ErrorEnvelope}},
    )
    def read_trace(run_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> Response:
        run = fetch_run(conn, run_id)
        if run is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        try:
            trace = build_run_trace(data_dir=data_dir, run_id=run_id)
        except TraceNotAvailableError:
            return _error(
                "TRACE_NOT_AVAILABLE",
                f"no trace available for run {run_id} yet",
                status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(content=trace.model_dump(mode="json"))

    # ---- Run workflow snapshot (frozen at run start, SP-009) ----

    @router.get(
        "/{run_id}/workflow",
        response_model=WorkflowDefinition,
        responses={404: {"model": ErrorEnvelope}},
    )
    def read_workflow_snapshot(
        run_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> Response:
        run = fetch_run(conn, run_id)
        if run is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        snapshot_path = data_dir / "data" / "runs" / run_id / "workflow.json"
        if not snapshot_path.is_file():
            return _error(
                "RUN_NOT_FOUND",
                f"workflow snapshot for run {run_id} not on disk",
                status.HTTP_404_NOT_FOUND,
            )
        try:
            definition = WorkflowDefinition.model_validate_json(
                snapshot_path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            return _error(
                "INTERNAL_ERROR",
                f"failed to parse workflow snapshot: {exc}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return JSONResponse(content=definition.model_dump(mode="json"))

    # ---- Run artifacts ----

    @router.get(
        "/{run_id}/artifacts",
        response_model=ArtifactListResponse,
        responses={404: {"model": ErrorEnvelope}},
    )
    def read_artifacts(
        run_id: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> Response:
        run = fetch_run(conn, run_id)
        if run is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        items = list_artifacts(data_dir=data_dir, run_id=run_id)
        body = ArtifactListResponse(items=items).model_dump(mode="json")
        return JSONResponse(content=body)

    @router.get(
        "/{run_id}/artifacts/{name}",
        response_class=FileResponse,
        responses={
            400: {"model": ErrorEnvelope},
            404: {"model": ErrorEnvelope},
        },
    )
    def download_artifact(
        run_id: str, name: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> Response:
        run = fetch_run(conn, run_id)
        if run is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        try:
            path = resolve_artifact_path(data_dir=data_dir, run_id=run_id, name=name)
        except InvalidArtifactNameError as exc:
            return _error(
                "INVALID_FILENAME",
                str(exc),
                status.HTTP_400_BAD_REQUEST,
            )
        except ArtifactNotFoundError as exc:
            return _error(
                "ARTIFACT_NOT_FOUND",
                str(exc),
                status.HTTP_404_NOT_FOUND,
            )
        import mimetypes

        mime, _ = mimetypes.guess_type(name)
        return FileResponse(
            path=path,
            media_type=mime or "application/octet-stream",
            filename=name,
        )

    # ---- Cancel run ----

    @router.post(
        "/{run_id}/cancel",
        responses={
            200: {"model": RunRead},
            404: {"model": ErrorEnvelope},
            409: {"model": ErrorEnvelope},
        },
    )
    async def cancel_run_handler(
        run_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        # Read the raw status first so a paused row (v0.4 forward-compat)
        # doesn't blow up RunStatus's enum validator. AC-47.
        row = conn.execute(
            "SELECT status FROM runs WHERE id = ?", (run_id,),
        ).fetchone()
        if row is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        raw_status = row["status"]
        if raw_status == "paused":
            return _error(
                "RUN_NOT_CANCELLABLE",
                f"run {run_id} is paused; pause/resume ships in v0.4",
                status.HTTP_409_CONFLICT,
            )

        run = fetch_run(conn, run_id)
        if run is None:
            return _error(
                "RUN_NOT_FOUND",
                f"run {run_id} does not exist",
                status.HTTP_404_NOT_FOUND,
            )
        if run.status in TERMINAL_STATUSES:
            return _error(
                "RUN_NOT_CANCELLABLE",
                f"run {run_id} is in terminal state {run.status.value}",
                status.HTTP_409_CONFLICT,
            )
        if run.status == RunStatus.CANCELLING:
            return _error(
                "RUN_NOT_CANCELLABLE",
                f"run {run_id} is already cancelling",
                status.HTTP_409_CONFLICT,
            )

        await cancel_run(
            run_id=run_id,
            db_factory=db_factory,
            broker=broker,
        )

        # Re-read to return the latest row state.
        with db_factory() as fresh_conn:
            updated = fetch_run(fresh_conn, run_id)
        return JSONResponse(content=updated.model_dump(mode="json") if updated else {})

    # ---- Per-run SSE ----

    @router.get("/{run_id}/events", response_model=None)
    async def per_run_events(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        active = lifecycle_module.get(run_id)
        if active is None:
            # Either unknown run_id or already terminal. Distinguish via DB.
            with db_factory() as conn:
                run = fetch_run(conn, run_id)
            if run is None:
                return _error(
                    "RUN_NOT_FOUND",
                    f"run {run_id} does not exist",
                    status.HTTP_404_NOT_FOUND,
                )
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # Wait for startup latch (race guard for SSE arriving before lifecycle flips status)
        try:
            await asyncio.wait_for(active.started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            err_resp = _error(
                "INTERNAL_ERROR",
                f"run {run_id} did not start within 2s; retry",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
            err_resp.headers["Retry-After"] = "1"
            return err_resp

        async def event_gen() -> AsyncIterator[str]:
            async for sse_payload in stream_run_events(
                active=active, last_event_id=last_event_id
            ):
                yield sse_payload

        return EventSourceResponse(event_gen())

    return router
