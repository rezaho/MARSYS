"""Spec → runtime materializer.

SP-005: the framework owns the canonical wire shape and its hydration.
``materialize_run`` therefore does exactly three things:

1. ``pydantic_to_topology(definition, AVAILABLE_TOOLS, handler_registry={})``
   — the framework constructs every ``Agent`` once (registered in
   ``AgentRegistry``), binds them onto ``Node.agent_ref``, and returns a
   runnable ``Topology``. No Spren-side node/edge/agent re-implementation.
2. ``pydantic_to_execution_config(definition.execution_config)`` — the
   framework rebuilds the typed runtime ``ExecutionConfig``.
3. A Spren-only per-run override: tracing → the per-run NDJSON dir, AG-UI
   → enabled (SP-004).

Credentials: Spren imposes ZERO assumption. ``ModelConfigSpec`` carries no
``api_key``; the framework's ``ModelConfig`` validator resolves the
*per-provider* env var itself (``ANTHROPIC_API_KEY`` for anthropic,
``OPENROUTER_API_KEY`` for openrouter, …). Spren never reads a
``SPREN_<PROVIDER>_API_KEY`` variable and never injects ``api_key``. A
genuinely missing key surfaces the framework's own ``ValidationError``,
wrapped here as ``MaterializationError`` (→ HTTP 400).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from marsys.coordination.aggui.config import AGGUIConfig
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.serialize import pydantic_to_execution_config
from marsys.coordination.topology.core import Topology
from marsys.coordination.topology.exceptions import (
    UnknownHandlerError,
    UnknownToolError,
)
from marsys.coordination.topology.serialize import (
    WorkflowDefinition,
    pydantic_to_topology,
)
from marsys.coordination.tracing.config import TracingConfig
from marsys.environment.tools import AVAILABLE_TOOLS
from pydantic import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class RuntimeBundle:
    """Materialized runtime types for one run.

    Agents are constructed + registered by ``pydantic_to_topology`` and are
    reachable via ``topology.nodes[*].agent_ref``; there is no separate agent
    list (nothing deregisters one).
    """

    topology: Topology
    execution_config: ExecutionConfig


class MaterializationError(Exception):
    """Raised when a workflow definition cannot be converted to runtime types."""


def materialize_run(
    *,
    definition: WorkflowDefinition,
    enable_aggui: bool = True,
    data_dir: Path | None = None,
    run_id: str | None = None,
) -> RuntimeBundle:
    """Convert one ``WorkflowDefinition`` to a runnable bundle.

    ``data_dir`` + ``run_id`` together set the per-run tracing ``output_dir``
    to ``<data_dir>/data/runs/{run_id}``; if either is ``None`` tracing stays
    at the framework default (useful for unit tests not exercising traces).
    ``enable_aggui=True`` opts into AG-UI translator construction at Orchestra
    wiring time (SP-004 — never optional in product).
    """
    try:
        topology = pydantic_to_topology(
            definition, AVAILABLE_TOOLS, handler_registry={}
        )
        execution_config = pydantic_to_execution_config(
            definition.execution_config
        )
    except (UnknownToolError, UnknownHandlerError, ValidationError) as exc:
        raise MaterializationError(str(exc)) from exc

    if data_dir is not None and run_id is not None:
        tracing_output_dir = data_dir / "data" / "runs" / run_id
        tracing_output_dir.mkdir(parents=True, exist_ok=True)
        execution_config.tracing = TracingConfig(
            enabled=True,
            output_dir=str(tracing_output_dir),
            include_message_content=True,
        )

    if enable_aggui:
        execution_config.aggui = AGGUIConfig(enabled=True)

    return RuntimeBundle(
        topology=topology,
        execution_config=execution_config,
    )
