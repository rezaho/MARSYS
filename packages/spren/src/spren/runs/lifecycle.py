"""Run lifecycle coordinator.

Owns the in-process registry of active runs and the
``queued → running → terminal`` state machine.

``_active_runs`` is module-level (mirrors ``workers/draft_sweeper.py``'s
module-pure idiom). Per-run records hold the ``Orchestra`` instance, the
asyncio Task running ``Orchestra.execute()``, an ``asyncio.Event``
``started`` set after status flips to ``running``, and a
``collections.deque(maxlen=1024)`` AG-UI events replay buffer for
reconnect-with-Last-Event-ID.

The materialized ``Orchestra`` is constructed inside ``register_run``
synchronously so ``orchestra.aggui_translator`` is guaranteed to be set
when the SSE endpoint asks for it.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from marsys.agents.registry import AgentRegistry
from marsys.coordination.orchestra import Orchestra
from marsys.coordination.state.storage import FileStorageBackend

from spren.cost import GenerationCost, apply_to_run as apply_cost_to_run, calculate_cost
from spren.files.lookup import format_attachments_block, resolve_attachments
from spren.models import (
    RunCancelledEvent,
    RunCreatedEvent,
    RunFinishedEvent,
    RunListItem,
    RunStatus,
    RunUpdatedEvent,
    RunsListEvent,
    TaskInput,
)
from spren.runs.broker import RunsBroker
from spren.runs.materialize import RuntimeBundle
from spren.storage.runs import (
    fetch_run,
    update_run_status,
)

logger = logging.getLogger(__name__)


@dataclass
class ActiveRun:
    """One active run's runtime state."""

    run_id: str
    workflow_id: str
    orchestra: Orchestra
    bundle: RuntimeBundle
    task: asyncio.Task | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    replay: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=1024)
    )


_active_runs: dict[str, ActiveRun] = {}
_lock = asyncio.Lock()


def get(run_id: str) -> ActiveRun | None:
    return _active_runs.get(run_id)


def is_active(run_id: str) -> bool:
    return run_id in _active_runs


def active_run_ids() -> list[str]:
    return list(_active_runs.keys())


async def deregister(run_id: str) -> None:
    async with _lock:
        _active_runs.pop(run_id, None)


async def shutdown_all_active(*, timeout: float = 5.0) -> None:
    """Cancel every in-flight lifecycle task and await drain.

    Called from the FastAPI lifespan handler on daemon shutdown so
    in-flight runs land in a terminal state before the process exits.
    """
    active_ids = list(_active_runs.keys())
    for run_id in active_ids:
        active = _active_runs.get(run_id)
        if active and active.task and not active.task.done():
            active.task.cancel()
    # Drain
    for run_id in active_ids:
        active = _active_runs.get(run_id)
        if active and active.task and not active.task.done():
            try:
                await asyncio.wait_for(active.task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:  # noqa: BLE001
                logger.exception("shutdown_all_active: task for %s raised", run_id)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_list_item(run: Any) -> RunListItem:
    return RunListItem(
        id=run.id,
        workflow_id=run.workflow_id,
        status=run.status,
        created_at=run.created_at,
        finished_at=run.finished_at,
        total_duration_ms=run.total_duration_ms,
        total_cost_usd=run.total_cost_usd,
    )


def _publish_event(broker: RunsBroker, run: Any, event_type: str = "updated") -> None:
    """Dispatch the right RunsListEvent for the run's transition.

    ``event_type``: 'created' (initial insert), 'finished' (succeeded/failed),
    'cancelled', or 'updated' (everything else).
    """
    item = _to_list_item(run)
    event: RunsListEvent
    if event_type == "created":
        event = RunCreatedEvent(run=item)
    elif event_type == "finished":
        event = RunFinishedEvent(run=item)
    elif event_type == "cancelled":
        event = RunCancelledEvent(run=item)
    else:
        event = RunUpdatedEvent(run=item)
    broker.publish(event)


# Kept as a thin alias for backwards-compat with internal callers; callers
# that know the terminal type should use `_publish_event` with the right
# discriminator.
def _publish_update(broker: RunsBroker, run: Any) -> None:
    _publish_event(broker, run, "updated")


def _freeze_workflow_snapshot(*, run_id: str, definition_json: str, data_dir: Path) -> Path:
    """Write the frozen workflow.json under <data-dir>/data/runs/{run_id}/."""
    run_dir = data_dir / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "workflow.json"
    path.write_text(definition_json, encoding="utf-8")
    return path


async def register_run(
    *,
    run_id: str,
    workflow_id: str,
    task_input: TaskInput,
    bundle: RuntimeBundle,
    data_dir: Path,
    db_factory: Callable[[], sqlite3.Connection] | None = None,
    trigger: str = "manual",
) -> ActiveRun:
    """Construct + wire ``Orchestra`` synchronously; insert ActiveRun record.

    Caller (the POST /v1/runs handler) calls this BEFORE returning 201
    so the SSE endpoint can find ``orchestra.aggui_translator`` set on
    first subscribe. ``schedule_run`` schedules the actual asyncio task;
    they're separated so the handler can return without awaiting.

    ``db_factory`` is wired into the cost-aggregation EventBus listener so
    each ``GenerationEvent`` updates the runs row's per-run aggregates.
    Tests that stub ``Orchestra`` may pass ``None`` and skip cost wiring.
    """
    storage_backend = FileStorageBackend(data_dir / "data" / "runs")
    orchestra = Orchestra(
        agent_registry=AgentRegistry,
        execution_config=bundle.execution_config,
        storage_backend=storage_backend,
    )

    active = ActiveRun(
        run_id=run_id,
        workflow_id=workflow_id,
        orchestra=orchestra,
        bundle=bundle,
    )
    if db_factory is not None:
        _wire_cost_aggregation(orchestra, run_id, db_factory)
    async with _lock:
        _active_runs[run_id] = active
    return active


def _wire_cost_aggregation(
    orchestra: Orchestra,
    run_id: str,
    db_factory: Callable[[], sqlite3.Connection],
) -> None:
    """Subscribe to ``GenerationEvent`` on the Orchestra's EventBus to
    aggregate cost per run. Aggregation writes happen on the same DB
    connection the lifecycle uses; we use a single short transaction
    per event so aggregates land even if the run later fails.
    """
    from marsys.coordination.tracing.events import GenerationEvent

    async def on_generation(event: GenerationEvent) -> None:
        # Filter by session_id (== run_id) so a shared EventBus across
        # parallel runs aggregates each run independently.
        if getattr(event, "session_id", None) != run_id:
            return
        cost = calculate_cost(
            provider=event.provider or "",
            model=event.model_name or "",
            prompt_tokens=event.prompt_tokens or 0,
            completion_tokens=event.completion_tokens or 0,
            reasoning_tokens=event.reasoning_tokens or 0,
        )
        try:
            with db_factory() as conn:
                apply_cost_to_run(conn, run_id=run_id, cost=cost)
                conn.commit()
        except Exception:  # noqa: BLE001
            logger.exception("cost aggregation failed for run_id=%s", run_id)

    bus = getattr(orchestra, "event_bus", None)
    if bus is None:
        logger.debug("register_run: orchestra has no event_bus; cost wiring skipped")
        return
    bus.subscribe(GenerationEvent.__name__, on_generation)


def schedule_run(
    *,
    active: ActiveRun,
    task_input: TaskInput,
    db_factory: Callable[[], sqlite3.Connection],
    broker: RunsBroker,
) -> asyncio.Task:
    """Kick off ``run_lifecycle`` as a background task. Returns the task."""
    task = asyncio.create_task(
        _run_lifecycle(
            active=active,
            task_input=task_input,
            db_factory=db_factory,
            broker=broker,
        ),
        name=f"spren-run-{active.run_id}",
    )
    active.task = task
    return task


async def _run_lifecycle(
    *,
    active: ActiveRun,
    task_input: TaskInput,
    db_factory: Callable[[], sqlite3.Connection],
    broker: RunsBroker,
) -> None:
    """Drive the run from queued → terminal."""
    started_at = _utc_now()
    try:
        # Transition queued → running. Re-resolve attachments to disk
        # paths in the same DB context so the system-context block lands
        # before Orchestra.execute() runs.
        task_text = task_input.text
        with db_factory() as conn:
            row = update_run_status(
                conn,
                run_id=active.run_id,
                status=RunStatus.RUNNING,
                started_at=started_at,
            )
            if task_input.attachments:
                resolved, missing = resolve_attachments(conn, task_input.attachments)
                if missing:
                    # The POST handler validated these; if they vanished
                    # between then and now (e.g., concurrent DELETE), log
                    # but proceed with the survivors. The agents see only
                    # the resolved files.
                    logger.warning(
                        "lifecycle: attachments missing for run %s: %s",
                        active.run_id,
                        missing,
                    )
                task_text = task_text + format_attachments_block(resolved)
            conn.commit()
            _publish_event(broker, row, "updated")
        active.started.set()

        # Run the orchestration
        result = await active.orchestra.execute(
            task=task_text,
            topology=active.bundle.topology,
            context={"session_id": active.run_id},
        )

        finished_at = _utc_now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        success = bool(getattr(result, "success", True))
        final_response: Any | None = getattr(result, "final_response", None)
        error_msg: str | None = None if success else (
            str(getattr(result, "error", None) or "execution failed")
        )

        terminal_status = RunStatus.SUCCEEDED if success else RunStatus.FAILED
        event_kind = "finished"

        # If we were transitioning through CANCELLING, honor that as terminal
        with db_factory() as conn:
            current = fetch_run(conn, active.run_id)
            if current is not None and current.status == RunStatus.CANCELLING:
                terminal_status = RunStatus.CANCELLED
                error_msg = error_msg or "cancelled"
                event_kind = "cancelled"

            row = update_run_status(
                conn,
                run_id=active.run_id,
                status=terminal_status,
                finished_at=finished_at,
                total_duration_ms=duration_ms,
                total_steps=getattr(result, "total_steps", None),
                final_response=final_response,
                error=error_msg,
            )
            conn.commit()
            _publish_event(broker, row, event_kind)

    except asyncio.CancelledError:
        # Cooperative cancel via task.cancel(); persist as cancelled.
        finished_at = _utc_now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)
        with db_factory() as conn:
            row = update_run_status(
                conn,
                run_id=active.run_id,
                status=RunStatus.CANCELLED,
                finished_at=finished_at,
                total_duration_ms=duration_ms,
                error="cancelled via task.cancel()",
            )
            conn.commit()
            _publish_event(broker, row, "cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        finished_at = _utc_now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)
        logger.exception("run lifecycle failed for run_id=%s", active.run_id)
        with db_factory() as conn:
            current = fetch_run(conn, active.run_id)
            # Honor an in-flight CANCELLING transition: if the user requested
            # cancel and the framework's exception handler raised, we still
            # want to record cancelled (not failed).
            if current is not None and current.status == RunStatus.CANCELLING:
                terminal_status = RunStatus.CANCELLED
                event_kind = "cancelled"
            else:
                terminal_status = RunStatus.FAILED
                event_kind = "finished"
            row = update_run_status(
                conn,
                run_id=active.run_id,
                status=terminal_status,
                finished_at=finished_at,
                total_duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
            conn.commit()
            _publish_event(broker, row, event_kind)
    finally:
        await deregister(active.run_id)


async def cancel_run(
    *,
    run_id: str,
    db_factory: Callable[[], sqlite3.Connection],
    broker: RunsBroker,
    force_after: float = 5.0,
    watchdog_total: float = 10.0,
) -> None:
    """Transition a running run to cancelled.

    - If queued: flip directly to cancelled (no Orchestra to signal).
    - If running: flip to ``cancelling``; call
      ``Orchestra.cancel_session(force_after)`` if present (Framework 07);
      else fallback to ``task.cancel()``. Watchdog total budget bounds
      the wait; on timeout, force-mark cancelled and log WARN.

    Caller (POST /v1/runs/{id}/cancel) checks current status / 409 first.
    """
    active = get(run_id)

    # Transition status first so the broker emits an update.
    now = _utc_now()
    with db_factory() as conn:
        current = fetch_run(conn, run_id)
        if current is None:
            return
        if current.status == RunStatus.QUEUED:
            row = update_run_status(
                conn,
                run_id=run_id,
                status=RunStatus.CANCELLED,
                finished_at=now,
                error="cancelled while queued",
            )
            conn.commit()
            _publish_event(broker, row, "cancelled")
            await deregister(run_id)
            return
        # status == running → enter cancelling
        row = update_run_status(conn, run_id=run_id, status=RunStatus.CANCELLING)
        conn.commit()
        _publish_event(broker, row, "updated")

    if active is None:
        # No active runtime record but DB says running. Treat as already terminal.
        with db_factory() as conn:
            row = update_run_status(
                conn,
                run_id=run_id,
                status=RunStatus.CANCELLED,
                finished_at=_utc_now(),
                error="cancelled (no active runtime)",
            )
            conn.commit()
            _publish_event(broker, row, "cancelled")
        return

    # Try Framework 07's Orchestra.cancel_session first.
    if hasattr(active.orchestra, "cancel_session"):
        try:
            await asyncio.wait_for(
                active.orchestra.cancel_session(run_id, force_after=force_after),
                timeout=watchdog_total,
            )
            return  # _run_lifecycle will write the terminal row
        except asyncio.TimeoutError:
            logger.warning(
                "cancel_run: Orchestra.cancel_session(%s) exceeded watchdog "
                "(%.1fs); marking cancelled and proceeding",
                run_id,
                watchdog_total,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cancel_run: Orchestra.cancel_session(%s) raised %s; falling back",
                run_id,
                exc,
            )

    # Fallback: cancel the asyncio task directly.
    if active.task is not None and not active.task.done():
        active.task.cancel()
        try:
            await asyncio.wait_for(active.task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            logger.exception("cancel_run: lifecycle task raised on cancel")
