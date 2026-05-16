"""
Pydantic wire-shape mirror of the marsys :class:`Agent` constructor surface.

``AgentSpec`` is the JSON-safe form of an agent's identity + behavior knobs.
Tools are referenced by name (``list[str]``) — the runtime callables are
supplied at hydration time via the ``tool_registry: Dict[str, Callable]``
parameter on :func:`pydantic_to_agents`.

The model-config field is named ``agent_model`` (NOT ``model``) because
Pydantic v2 reserves the ``model_*`` namespace for class-level configuration
(e.g., ``model_config = ConfigDict(...)``). The nested type is
:class:`marsys.models.serialize.ModelConfigSpec`, the storage-boundary mirror
of :class:`marsys.models.ModelConfig` — see ``models/serialize.py`` for why a
mirror is necessary instead of a direct re-export.

``agent_to_pydantic`` is typed against the concrete ``Agent`` subclass
(``packages/framework/src/marsys/agents/agents.py:2727``) rather than
``BaseAgent`` because only ``Agent._model_config`` retains the originating
``ModelConfig`` instance — ``BaseAPIModel`` / ``BaseLocalModel`` do not.

``is_convergence_point`` round-trips through a post-construction attribute set
in :func:`pydantic_to_agents` because ``Agent.__init__`` does not accept it
as a constructor kwarg (only ``BaseAgent.__init__`` does, but ``Agent`` does
not forward it). The asymmetry is documented as a known gap; fixing it
properly requires extending ``Agent.__init__``, which is out of scope for
this PR.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..coordination.topology.exceptions import UnknownToolError
from ..coordination.topology.serialize import (
    WorkflowDefinition,
    _rebuild_workflow_definition,
)
from ..models.serialize import (
    ModelConfigSpec,
    model_config_spec_from_runtime,
    runtime_model_config_from_spec,
)


MemoryRetention = Literal["single_run", "session", "persistent"]


class AgentSpec(BaseModel):
    """Wire mirror of :class:`marsys.agents.Agent`'s constructor surface."""

    model_config = ConfigDict(extra="forbid")

    name: str
    goal: str
    instruction: str
    agent_model: ModelConfigSpec
    tools: List[str] = Field(default_factory=list)
    max_tokens: Optional[int] = 10000
    allowed_peers: List[str] = Field(default_factory=list)
    bidirectional_peers: bool = False
    is_convergence_point: Optional[bool] = None
    memory_retention: MemoryRetention = "session"
    memory_storage_path: Optional[str] = None
    plan_config: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


# Rebuild ``WorkflowDefinition`` now that ``AgentSpec`` is defined; resolves the
# forward reference on the ``agents: Dict[str, "AgentSpec"]`` field.
_rebuild_workflow_definition()


def agent_to_pydantic(agent: Any) -> AgentSpec:
    """Build an :class:`AgentSpec` from a live :class:`Agent` (concrete subclass).

    Reads ``agent._model_config`` (only the ``Agent`` subclass retains it; see
    module docstring). Raises ``AttributeError`` when called on a ``BaseAgent``
    subclass that does not retain the originating ``ModelConfig``.
    """
    from .agents import Agent

    if not isinstance(agent, Agent):
        raise TypeError(
            f"agent_to_pydantic expects an Agent (subclass that retains its "
            f"originating ModelConfig as _model_config); got {type(agent).__name__}. "
            f"If you need to serialize a custom BaseAgent subclass, make it "
            f"retain its source ModelConfig and update agent_to_pydantic."
        )

    plan_config_dict: Optional[Dict[str, Any]] = None
    planning_config = getattr(agent, "_planning_config", None)
    if planning_config is not None:
        to_dict = getattr(planning_config, "to_dict", None)
        if callable(to_dict):
            plan_config_dict = to_dict()
        else:
            plan_config_dict = None

    # Framework-injected planning tools (`plan_*`) are NOT user-supplied; the
    # rehydration path re-injects them automatically based on `plan_config`.
    # Excluding them from the wire shape avoids `UnknownToolError` on the
    # round-trip when the consumer's tool registry only covers user tools.
    user_tool_names = [
        name for name in agent.tools.keys() if not name.startswith("plan_")
    ]

    return AgentSpec(
        name=agent.name,
        goal=agent.goal,
        instruction=agent.instruction,
        agent_model=model_config_spec_from_runtime(agent._model_config),
        tools=user_tool_names,
        max_tokens=agent.max_tokens,
        allowed_peers=sorted(agent._allowed_peers_init),
        bidirectional_peers=agent._bidirectional_peers,
        is_convergence_point=agent._is_convergence_point,
        memory_retention=agent._memory_retention,
        memory_storage_path=agent._memory_storage_path,
        plan_config=plan_config_dict,
        input_schema=agent.input_schema,
        output_schema=agent.output_schema,
    )


def pydantic_to_agents(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> List[Any]:
    """Materialize the agent set declared in a :class:`WorkflowDefinition`.

    Resolves each tool name in ``AgentSpec.tools`` against ``tool_registry``;
    a missing name raises :class:`UnknownToolError` with the offending tool +
    agent name.

    ``is_convergence_point`` is restored by setting the private attribute on
    the constructed Agent because ``Agent.__init__`` does not accept the
    kwarg. Documented in module docstring as a known asymmetry.
    """
    from .agents import Agent

    agents: List[Any] = []
    for agent_name, agent_spec in spec.agents.items():
        tools_dict: Dict[str, Callable] = {}
        for tool_name in agent_spec.tools:
            if tool_name not in tool_registry:
                raise UnknownToolError(
                    f"Tool '{tool_name}' (used by agent '{agent_name}') is "
                    f"not registered. Pass a tool_registry mapping name → "
                    f"callable to pydantic_to_agents / pydantic_to_topology."
                )
            tools_dict[tool_name] = tool_registry[tool_name]

        runtime_model_config = runtime_model_config_from_spec(agent_spec.agent_model)

        agent = Agent(
            name=agent_spec.name,
            goal=agent_spec.goal,
            instruction=agent_spec.instruction,
            model_config=runtime_model_config,
            tools=tools_dict,
            max_tokens=agent_spec.max_tokens,
            allowed_peers=list(agent_spec.allowed_peers),
            bidirectional_peers=agent_spec.bidirectional_peers,
            input_schema=agent_spec.input_schema,
            output_schema=agent_spec.output_schema,
            memory_retention=agent_spec.memory_retention,
            memory_storage_path=agent_spec.memory_storage_path,
            plan_config=agent_spec.plan_config,
        )
        if agent_spec.is_convergence_point is not None:
            agent._is_convergence_point = agent_spec.is_convergence_point
        agents.append(agent)
    return agents
