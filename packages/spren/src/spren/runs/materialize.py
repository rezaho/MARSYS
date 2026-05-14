"""Spec → runtime materializer.

Converts a Spren-side ``WorkflowDefinition`` (Pydantic mirror of the
framework's runtime types) into the runnable types the framework's
``Orchestra`` consumes:

- ``TopologySpec`` → ``marsys.coordination.topology.core.Topology``
- ``dict[str, AgentSpec]`` → list of ``marsys.agents.Agent`` instances
  (registered via ``AgentRegistry`` on construction)
- tool name strings → callables via ``marsys.environment.tools.AVAILABLE_TOOLS``
- assembled ``ExecutionConfig`` (with ``aggui=AGGUIConfig(enabled=True)``
  when Framework 06 is available; otherwise no-op)

Returns a ``RuntimeBundle`` consumed by ``runs/lifecycle.py``.

Secret resolution: ``ModelConfigSpec`` carries no ``api_key``. The
materializer's ``secrets_lookup`` callable resolves the api_key from a
secrets store keyed by ``provider``. v0.3 uses an environment-variable
lookup (`SPREN_<PROVIDER>_API_KEY`); v0.4 swaps in the OS-keychain
+ encrypted-SQLite path.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from marsys.agents.agents import Agent
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.topology.core import (
    Edge,
    EdgePattern,
    EdgeType,
    Node,
    NodeType,
    Topology,
)
from marsys.coordination.tracing.config import TracingConfig
from marsys.environment.tools import AVAILABLE_TOOLS
from marsys.models.models import ModelConfig

from spren.models import (
    AgentSpec,
    ModelConfigSpec,
    NodeType as SprenNodeType,
    EdgeType as SprenEdgeType,
    EdgePattern as SprenEdgePattern,
    WorkflowDefinition,
)


logger = logging.getLogger(__name__)


SecretsLookup = Callable[[str], Optional[str]]


@dataclass
class RuntimeBundle:
    """Materialized runtime types for one run.

    The framework's ``AgentRegistry`` holds strong refs to the registered
    agents while a run is in flight; ``agents`` here is the explicit list
    so the lifecycle coordinator can deregister them after terminal.
    """

    topology: Topology
    agents: list[Agent]
    execution_config: ExecutionConfig


class MaterializationError(Exception):
    """Raised when a workflow definition cannot be converted to runtime types."""


def _env_secrets_lookup(provider: str) -> str | None:
    """Default secrets lookup: env var ``SPREN_<PROVIDER>_API_KEY``."""
    canonical = provider.replace("-", "_").upper()
    return os.environ.get(f"SPREN_{canonical}_API_KEY")


def _materialize_node(node_spec: Any) -> Node:
    """Convert one ``NodeSpec`` to a framework ``Node``.

    The framework's ``Node.agent_ref`` is ``Optional[Any]`` and can carry
    a string name or an Agent instance. We keep the spec's string name —
    the registry-based lookup is how agents wire up at execution time.
    """
    return Node(
        name=node_spec.name,
        node_type=NodeType(node_spec.node_type.value),
        agent_ref=node_spec.agent_ref,
        is_convergence_point=node_spec.is_convergence_point,
        metadata=dict(node_spec.metadata),
    )


def _materialize_edge(edge_spec: Any) -> Edge:
    """Convert one ``EdgeSpec`` to a framework ``Edge``."""
    return Edge(
        source=edge_spec.source,
        target=edge_spec.target,
        edge_type=EdgeType(edge_spec.edge_type.value),
        bidirectional=edge_spec.bidirectional,
        pattern=(EdgePattern(edge_spec.pattern.value) if edge_spec.pattern is not None else None),
        metadata=dict(edge_spec.metadata),
    )


def _materialize_topology(topology_spec: Any) -> Topology:
    nodes = [_materialize_node(n) for n in topology_spec.nodes]
    edges = [_materialize_edge(e) for e in topology_spec.edges]
    return Topology(nodes=nodes, edges=edges)


def _materialize_model_config(
    spec: ModelConfigSpec,
    *,
    secrets_lookup: SecretsLookup,
) -> ModelConfig:
    """Convert a ``ModelConfigSpec`` (no api_key) to a runnable ``ModelConfig``.

    Resolves ``api_key`` for API-typed models via ``secrets_lookup`` keyed
    by provider. Local-typed models pass through unchanged.
    """
    payload = spec.model_dump()
    if payload.get("type") == "api":
        provider = payload.get("provider")
        if provider is None:
            raise MaterializationError(
                "API-typed ModelConfigSpec is missing 'provider'; cannot resolve api_key"
            )
        api_key = secrets_lookup(provider)
        if api_key is None:
            raise MaterializationError(
                f"No api_key in secrets store for provider '{provider}' "
                f"(checked SPREN_{provider.replace('-', '_').upper()}_API_KEY env var)"
            )
        payload["api_key"] = api_key
    # ModelConfig validator handles base_url inference + thinking-budget guards
    return ModelConfig(**payload)


def _materialize_agent(
    name: str,
    agent_spec: AgentSpec,
    *,
    secrets_lookup: SecretsLookup,
) -> Agent:
    """Construct one framework ``Agent`` from an ``AgentSpec``.

    Agent.__init__ registers the instance with ``AgentRegistry`` under
    ``name``. Tools are resolved by name from
    ``marsys.environment.tools.AVAILABLE_TOOLS``; unknown names raise.
    """
    tool_callables: dict[str, Any] = {}
    for tool_name in agent_spec.tools:
        if tool_name not in AVAILABLE_TOOLS:
            raise MaterializationError(
                f"Agent '{name}' references unknown tool '{tool_name}'. "
                f"Available tools: {sorted(AVAILABLE_TOOLS.keys())}"
            )
        tool_callables[tool_name] = AVAILABLE_TOOLS[tool_name]

    runtime_model = _materialize_model_config(agent_spec.agent_model, secrets_lookup=secrets_lookup)

    return Agent(
        model_config=runtime_model,
        goal=agent_spec.goal,
        instruction=agent_spec.instruction,
        tools=tool_callables or None,
        name=name,
        allowed_peers=list(agent_spec.allowed_peers) or None,
        memory_retention=agent_spec.memory_retention,
    )


def _build_execution_config(
    *,
    spec_execution_config: Any,
    enable_aggui: bool,
    tracing_output_dir: Path | None = None,
) -> ExecutionConfig:
    """Assemble the framework ``ExecutionConfig`` for a run.

    - mirrors the spec's execution-config fields where they're stable
    - **Always enables tracing** with ``output_dir = tracing_output_dir``
      (when provided) so the NDJSON writer drops the per-run trace file.
      Without this, ``trace.ndjson`` is never written and Session 05's
      trace endpoint reads an empty path. Plan §8.16.
    - opts into AG-UI translator construction when Framework 06 is
      available (gated by a runtime presence check; falls back to a
      plain ``ExecutionConfig`` otherwise so Spren can run without
      Framework 06 merged)
    """
    spec_dump = spec_execution_config.model_dump() if spec_execution_config is not None else {}

    # Filter to fields the framework's ExecutionConfig actually accepts.
    # The spec-side mirror may carry additional Spren-only fields.
    framework_fields = set(ExecutionConfig.__dataclass_fields__.keys()) if hasattr(
        ExecutionConfig, "__dataclass_fields__"
    ) else set()
    filtered = {k: v for k, v in spec_dump.items() if k in framework_fields}
    # ``tracing`` is constructed below with per-run output_dir; drop any
    # spec-side tracing payload (the spec-mirror's tracing shape isn't
    # the framework's TracingConfig, and we want the per-run output_dir
    # regardless).
    filtered.pop("tracing", None)

    config = ExecutionConfig(**filtered) if filtered else ExecutionConfig()

    # Per-run tracing wiring. Plan §8.16 — without this, no NDJSON file
    # exists for the trace endpoint to read.
    if tracing_output_dir is not None:
        tracing_output_dir.mkdir(parents=True, exist_ok=True)
        config.tracing = TracingConfig(
            enabled=True,
            output_dir=str(tracing_output_dir),
            include_message_content=True,  # framework default; preserved per plan §8.17
        )

    if enable_aggui:
        try:
            # Framework 06 ships ExecutionConfig.aggui = AGGUIConfig
            from marsys.coordination.aggui.config import AGGUIConfig  # type: ignore[import-not-found]

            if hasattr(config, "aggui"):
                config.aggui = AGGUIConfig(enabled=True)
            else:
                logger.debug(
                    "materialize: ExecutionConfig has no 'aggui' field; "
                    "Framework 06 not yet present in this build"
                )
        except ImportError:
            logger.debug(
                "materialize: marsys.coordination.aggui not importable; "
                "Framework 06 not yet present"
            )

    return config


def materialize_run(
    *,
    definition: WorkflowDefinition,
    secrets_lookup: SecretsLookup | None = None,
    enable_aggui: bool = True,
    data_dir: Path | None = None,
    run_id: str | None = None,
) -> RuntimeBundle:
    """Convert one ``WorkflowDefinition`` to a runnable bundle.

    ``secrets_lookup`` defaults to env-var lookup. Pass a callable to
    plug in the OS-keychain / encrypted-SQLite path when available.
    ``enable_aggui=True`` opts into AG-UI translator construction at
    Orchestra wiring time when Framework 06 is present (no-op when
    absent).
    ``data_dir`` + ``run_id`` together set the per-run tracing
    ``output_dir`` to ``<data_dir>/data/runs/{run_id}``. Both are
    required to enable tracing; if either is ``None``, tracing is left
    at the framework default (``enabled=False``) — useful in unit tests
    that don't exercise the trace path.
    """
    secrets_lookup = secrets_lookup or _env_secrets_lookup

    topology = _materialize_topology(definition.topology)

    agents: list[Agent] = []
    for name, agent_spec in definition.agents.items():
        agents.append(_materialize_agent(name, agent_spec, secrets_lookup=secrets_lookup))

    tracing_output_dir: Path | None = None
    if data_dir is not None and run_id is not None:
        tracing_output_dir = data_dir / "data" / "runs" / run_id

    execution_config = _build_execution_config(
        spec_execution_config=definition.execution_config,
        enable_aggui=enable_aggui,
        tracing_output_dir=tracing_output_dir,
    )

    return RuntimeBundle(
        topology=topology,
        agents=agents,
        execution_config=execution_config,
    )
