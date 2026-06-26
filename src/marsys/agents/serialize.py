"""
Pydantic wire-shape mirror of the marsys :class:`Agent` constructor surface,
plus the specialized-agent ``kind``/``params`` contract (Session 09 / ADR-009).

``AgentSpec`` is the JSON-safe form of an agent's identity + behavior knobs.
Tools are referenced by name (``list[str]``) — the runtime callables are
supplied at hydration time via the ``tool_registry: Dict[str, Callable]``
parameter on :func:`pydantic_to_agents`.

Specialized ``Agent`` subclasses (``WebSearchAgent``, ``BrowserAgent``,
``CodeExecutionAgent``, ``DataAnalysisAgent``, ``FileOperationAgent``) round-trip
**as that subclass** via:

- ``AgentSpec.kind`` — a closed string discriminator (default ``"agent"`` →
  base ``Agent``, byte-identical to pre-S09, no migration). The 1:1 analog of
  ``NodeSpec.kind`` / ``NODE_KIND_BEHAVIOUR`` (ADR-008 Decision 2/9).
- ``AGENT_KIND_REGISTRY`` — the single authoritative ``kind → Agent subclass``
  map (NOT ``marsys.agents.__all__``, NOT reflection). Each subclass declares
  its own stable wire key as ``WIRE_KIND``; the reverse ``class → kind`` map is
  *derived* by inverting the registry, never hand-maintained twice.
- ``AgentSpec.params`` — a typed, per-subclass ``*ParamsSpec`` (Pydantic
  discriminated union keyed on ``spec_kind``) carrying ONLY the subclass's
  JSON-safe declarative config. Secrets (e.g. ``WebSearchAgent`` API keys) and
  runtime objects (``RunFileSystem``, ``CodeExecutionConfig``) never travel —
  the subclass ``__init__`` re-resolves them on hydrate exactly as it does from
  a Python file (the ``ModelConfigSpec``-drops-``api_key`` precedent).

``kind``/``params`` are **additive** on the existing flat base surface — every
base field still round-trips for a subclass.

The model-config field is named ``model`` (renamed from ``agent_model`` in
S09 / ADR-009 Decision 6: a field named exactly ``model`` is Pydantic-v2-clean;
only the ``model_`` *prefix* namespace and the ``model_config`` attribute name
are reserved — verified). The nested type is
:class:`marsys.models.serialize.ModelConfigSpec`.

Hydration is **async** (single entrypoint, ADR-009 Decision 4 / option B′):
``BrowserAgent`` is only fully constructible via its async ``create_safe``;
a sync hydrate physically cannot produce a browser-ready instance, and a
sync/async dual API forces the caller to choose by workflow contents it has
not parsed yet. ``pydantic_to_agents`` / ``pydantic_to_topology`` are
coroutines; there is no sync variant.

``is_convergence_point`` round-trips through a post-construction attribute set
because ``Agent.__init__`` does not accept it as a constructor kwarg. The
asymmetry is documented as a known gap; fixing it properly requires extending
``Agent.__init__``, which is out of scope for this session.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
)

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
from .browser_agent import BrowserAgent
from .code_execution_agent import CodeExecutionAgent
from .data_analysis_agent import DataAnalysisAgent
from .file_operation_agent import FileOperationAgent
from .web_search_agent import WebSearchAgent


MemoryRetention = Literal["single_run", "session", "persistent"]


# ---------------------------------------------------------------------------
# Per-subclass declarative params specs (typed; no secrets; no runtime objects)
#
# Each spec mirrors ONLY the subclass's JSON-safe declarative constructor
# surface — the inputs that determine the rebuilt tools/instructions/behaviour.
# ``goal``/``instruction`` are captured *as the user gave them* (the delta),
# NOT the resolved form: subclasses like BrowserAgent prepend mode defaults in
# __init__, so re-feeding the resolved goal would double-prepend on round-trip.
# ``spec_kind`` is the Pydantic discriminator (mirrors ``AgentSpec.kind``).
# ---------------------------------------------------------------------------


class _BaseParamsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # As-given user goal/instruction delta (None → the subclass auto-builds).
    goal: Optional[str] = None
    instruction: Optional[str] = None


class WebSearchAgentParamsSpec(_BaseParamsSpec):
    spec_kind: Literal["web_search"] = "web_search"
    search_mode: str = "all"
    search_types: Optional[List[str]] = None
    enabled_tools: Optional[List[str]] = None
    include_google: bool = True


class BrowserAgentParamsSpec(_BaseParamsSpec):
    spec_kind: Literal["browser"] = "browser"
    mode: str = "advanced"
    headless: bool = True
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    tmp_dir: Optional[str] = None
    browser_channel: Optional[str] = None
    vision_model_config: Optional[ModelConfigSpec] = None
    auto_screenshot: bool = False
    element_detection_mode: str = "auto"
    timeout: int = 5000
    memory_type: str = "conversation_history"
    session_path: Optional[str] = None
    show_mouse_helper: bool = True
    downloads_subdir: str = "downloads"
    downloads_virtual_dir: str = "./downloads"
    fetch_file_tool_name: str = "download_file"


class CodeExecutionAgentParamsSpec(_BaseParamsSpec):
    spec_kind: Literal["code_execution"] = "code_execution"
    # As-given declarative directory inputs (NOT the resolved cwd — a
    # cwd-derived absolute path must never reach the wire; None → the
    # subclass re-defaults to Path.cwd() on hydrate, exactly as today).
    base_directory: Optional[str] = None
    working_directory: Optional[str] = None


class DataAnalysisAgentParamsSpec(_BaseParamsSpec):
    spec_kind: Literal["data_analysis"] = "data_analysis"
    base_directory: Optional[str] = None
    working_directory: Optional[str] = None


class FileOperationAgentParamsSpec(_BaseParamsSpec):
    spec_kind: Literal["file_operation"] = "file_operation"
    enable_shell: bool = False
    working_directory: Optional[str] = None
    base_directory: Optional[str] = None
    allowed_shell_commands: Optional[List[str]] = None
    blocked_shell_patterns: Optional[List[str]] = None
    shell_timeout_default: int = 30


AgentParamsSpec = Annotated[
    Union[
        WebSearchAgentParamsSpec,
        BrowserAgentParamsSpec,
        CodeExecutionAgentParamsSpec,
        DataAnalysisAgentParamsSpec,
        FileOperationAgentParamsSpec,
    ],
    Field(discriminator="spec_kind"),
]


# ---------------------------------------------------------------------------
# The single authoritative kind → Agent-subclass registry.
#
# THIS dict is the source of truth (the ``NODE_KIND_BEHAVIOUR`` analog,
# det_nodes.py:188). NOT ``marsys.agents.__all__`` (that is the package public
# API — includes BaseAgent/AgentPool/MemoryManager/...). NOT reflection /
# ``Agent.__subclasses__()`` (instantiating an arbitrary class named in stored
# JSON is an RCE-class risk, and is import-order fragile). Each subclass
# declares its own ``WIRE_KIND``; the reverse + the params-spec map are DERIVED
# from this one dict. Adding a specialized agent = one ``WIRE_KIND`` attr + one
# entry here, no dispatch-site edits (extension-open; AC-13/14/16/17).
# ---------------------------------------------------------------------------

AGENT_KIND_REGISTRY: Dict[str, Type[Any]] = {
    WebSearchAgent.WIRE_KIND: WebSearchAgent,
    BrowserAgent.WIRE_KIND: BrowserAgent,
    CodeExecutionAgent.WIRE_KIND: CodeExecutionAgent,
    DataAnalysisAgent.WIRE_KIND: DataAnalysisAgent,
    FileOperationAgent.WIRE_KIND: FileOperationAgent,
}

# Derived (NOT hand-maintained): class → kind, by inverting the registry.
_CLASS_TO_KIND: Dict[Type[Any], str] = {
    cls: kind for kind, cls in AGENT_KIND_REGISTRY.items()
}

# Derived (NOT a hand-maintained third copy): kind → typed params spec class,
# read off each discriminated-union member's ``spec_kind`` default. The
# ``AgentParamsSpec`` union is the single source; the CI test asserts this
# key-set equals ``AGENT_KIND_REGISTRY``'s, so a registry/spec mismatch fails
# the build, not a user's load.
_PARAMS_SPEC_CLASSES = get_args(get_args(AgentParamsSpec)[0])
_KIND_TO_PARAMS_SPEC: Dict[str, Type[_BaseParamsSpec]] = {
    cls.model_fields["spec_kind"].default: cls
    for cls in _PARAMS_SPEC_CLASSES
}

BASE_AGENT_KIND = "agent"


class AgentSpec(BaseModel):
    """Wire mirror of :class:`marsys.agents.Agent`'s constructor surface."""

    model_config = ConfigDict(extra="forbid")

    name: str
    goal: str
    instruction: str
    model: ModelConfigSpec
    tools: List[str] = Field(default_factory=list)
    max_tokens: Optional[int] = 10000
    allowed_peers: List[str] = Field(default_factory=list)
    bidirectional_peers: bool = False
    # ADR-013: the escalate_to_user grant, mirrored from Agent.can_escalate so it
    # round-trips through a serialized workflow (the live Agent stays canonical).
    can_escalate: bool = False
    is_convergence_point: Optional[bool] = None
    memory_retention: MemoryRetention = "session"
    memory_storage_path: Optional[str] = None
    plan_config: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Specialized-agent contract (S09 / ADR-009). ``kind`` defaults to
    # ``"agent"`` → base ``Agent``; existing specs are byte-identical and need
    # no migration. ``params`` is non-None iff ``kind`` is a specialized kind.
    kind: str = BASE_AGENT_KIND
    params: Optional[AgentParamsSpec] = None


# Rebuild ``WorkflowDefinition`` now that ``AgentSpec`` is defined; resolves the
# forward reference on the ``agents: Dict[str, "AgentSpec"]`` field.
_rebuild_workflow_definition()


def _params_spec_from_agent(agent: Any, kind: str) -> _BaseParamsSpec:
    """Build the typed ``*ParamsSpec`` for a specialized agent instance.

    Reads ``agent._wire_params`` — the as-given declarative kwargs each
    specialized ``__init__`` retains for serialization (the base ``Agent``
    ``_allowed_peers_init`` precedent). Nested ``ModelConfig`` values are
    converted to the storage-safe :class:`ModelConfigSpec` (drops api_key).
    """
    spec_cls = _KIND_TO_PARAMS_SPEC[kind]
    raw = dict(getattr(agent, "_wire_params", {}))
    vision = raw.get("vision_model_config")
    if vision is not None and not isinstance(vision, ModelConfigSpec):
        raw["vision_model_config"] = model_config_spec_from_runtime(vision)
    return spec_cls(**raw)


def agent_to_pydantic(agent: Any) -> AgentSpec:
    """Build an :class:`AgentSpec` from a live :class:`Agent` (concrete subclass).

    Reads ``agent._model_config`` (only the ``Agent`` subclass retains it; see
    module docstring). For a specialized subclass, additionally emits ``kind``
    + the typed ``params`` — *additive* on the base surface (every base field
    still round-trips). A base ``Agent`` yields ``kind="agent"``, ``params=None``
    — byte-identical to pre-S09.

    Raises ``TypeError`` when called on a ``BaseAgent`` subclass that is not an
    ``Agent`` (e.g. ``LearnableAgent``) — behaviour unchanged from pre-S09
    (explicitly out of scope; not a silent flatten).
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

    kind = _CLASS_TO_KIND.get(type(agent), BASE_AGENT_KIND)
    params: Optional[_BaseParamsSpec] = None
    if kind != BASE_AGENT_KIND:
        params = _params_spec_from_agent(agent, kind)
        # A specialized subclass rebuilds its OWN tools from `params` on
        # hydrate (ADR-009 Decision 1; AC-6) — those tools are a function of
        # the declarative config, not an independent user-supplied set, so
        # they are NOT carried on the wire (carrying them would serialize
        # derived state and force a tool_registry entry for tools the class
        # rebuilds itself). Base `Agent` keeps its by-name user tools.
        user_tool_names = []

    return AgentSpec(
        name=agent.name,
        goal=agent.goal,
        instruction=agent.instruction,
        model=model_config_spec_from_runtime(agent._model_config),
        tools=user_tool_names,
        max_tokens=agent.max_tokens,
        allowed_peers=sorted(agent._allowed_peers_init),
        bidirectional_peers=agent._bidirectional_peers,
        can_escalate=agent.can_escalate,
        is_convergence_point=agent._is_convergence_point,
        memory_retention=agent._memory_retention,
        memory_storage_path=agent._memory_storage_path,
        plan_config=plan_config_dict,
        input_schema=agent.input_schema,
        output_schema=agent.output_schema,
        kind=kind,
        params=params,
    )


def _base_surface_kwargs(
    agent_spec: AgentSpec,
    runtime_model_config: Any,
    tools_dict: Dict[str, Callable],
) -> Dict[str, Any]:
    """The base ``Agent`` constructor kwargs shared by every kind.

    ``is_convergence_point`` is intentionally excluded — ``Agent.__init__``
    does not accept it; it is restored post-construction by the caller.
    """
    return dict(
        name=agent_spec.name,
        model_config=runtime_model_config,
        max_tokens=agent_spec.max_tokens,
        allowed_peers=list(agent_spec.allowed_peers),
        bidirectional_peers=agent_spec.bidirectional_peers,
        input_schema=agent_spec.input_schema,
        output_schema=agent_spec.output_schema,
        memory_retention=agent_spec.memory_retention,
        memory_storage_path=agent_spec.memory_storage_path,
        plan_config=agent_spec.plan_config,
        tools=tools_dict,
    )


def _accepted_kwargs(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the target callable cannot accept.

    Specialized constructors are heterogeneous: the sync subclasses take
    ``**kwargs`` and forward the base surface to ``super().__init__``, but
    ``BrowserAgent.create_safe`` is a fixed-signature factory with no
    ``**kwargs`` — a base-surface kwarg it does not declare (e.g.
    ``bidirectional_peers``) is genuinely not part of its contract. We
    synthesize the call, so we match the callee's signature: pass everything
    when it has ``**kwargs``; otherwise only the parameters it declares.

    Recorded consequence (ADR-009 Consequences): for a fixed-signature
    factory, a base ``AgentSpec`` field the factory does not expose is not
    round-tripped for that subclass — concretely, ``BrowserAgent`` cannot
    carry ``bidirectional_peers`` because neither ``BrowserAgent.__init__``
    nor ``create_safe`` accepts it, so the ``Agent`` default always applies.
    This is a property of that class's own constructor contract, not a
    serializer drop, and it is not a silent feature loss: the value is
    structurally always the class default for such a class (it cannot be set
    non-default through the class in the first place).
    """
    sig = inspect.signature(target)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


async def pydantic_to_agents(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> List[Any]:
    """Materialize the agent set declared in a :class:`WorkflowDefinition`.

    **Async, single entrypoint** (ADR-009 Decision 4 / option B′). Hydration is
    intrinsically async (a workflow may contain a ``BrowserAgent``, only fully
    constructible via its async ``create_safe``); there is no sync variant.

    Resolves each tool name in ``AgentSpec.tools`` against ``tool_registry``;
    a missing name raises :class:`UnknownToolError`. Specialized kinds dispatch
    via :data:`AGENT_KIND_REGISTRY` and the framework's own async-construction
    idiom (``pool_factory.py:44`): a class exposing an async ``create_safe`` is
    built via ``await create_safe(...)``; otherwise via the plain constructor.
    A specialized subclass rebuilds its own tools/instructions from ``params``
    — its declarative config — exactly as it would from a Python file.

    ``is_convergence_point`` is restored by setting the private attribute on
    the constructed agent because ``Agent.__init__`` does not accept the kwarg.
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

        runtime_model_config = runtime_model_config_from_spec(agent_spec.model)

        if agent_spec.kind == BASE_AGENT_KIND:
            agent = Agent(
                name=agent_spec.name,
                goal=agent_spec.goal,
                instruction=agent_spec.instruction,
                model_config=runtime_model_config,
                tools=tools_dict,
                max_tokens=agent_spec.max_tokens,
                allowed_peers=list(agent_spec.allowed_peers),
                bidirectional_peers=agent_spec.bidirectional_peers,
                can_escalate=agent_spec.can_escalate,
                input_schema=agent_spec.input_schema,
                output_schema=agent_spec.output_schema,
                memory_retention=agent_spec.memory_retention,
                memory_storage_path=agent_spec.memory_storage_path,
                plan_config=agent_spec.plan_config,
            )
        else:
            cls = AGENT_KIND_REGISTRY.get(agent_spec.kind)
            if cls is None:
                raise UnknownToolError(
                    f"Agent '{agent_name}' declares kind "
                    f"'{agent_spec.kind}' which is not in AGENT_KIND_REGISTRY "
                    f"(have: {sorted(AGENT_KIND_REGISTRY)}). A specialized "
                    f"agent kind must be a registered Agent subclass."
                )
            params = agent_spec.params
            params_kwargs: Dict[str, Any] = (
                params.model_dump(exclude={"spec_kind"})
                if params is not None
                else {}
            )
            # Nested ModelConfigSpec → runnable ModelConfig (e.g. BrowserAgent
            # vision_model_config). Re-resolved here, never carried as an
            # object on the wire.
            vision = params_kwargs.get("vision_model_config")
            if vision is not None:
                params_kwargs["vision_model_config"] = (
                    runtime_model_config_from_spec(
                        ModelConfigSpec(**vision)
                        if isinstance(vision, dict)
                        else vision
                    )
                )
            # Reconstruct runtime-typed kwargs from their JSON form (the
            # plan's "non-JSON / runtime-derived kwargs reconstructed at
            # hydrate"): the code/data/file agents annotate ``base_directory``
            # as ``Optional[Path]`` and call ``.resolve()`` on it, so the
            # declarative wire string must become a ``Path`` before the
            # constructor runs.
            bd = params_kwargs.get("base_directory")
            if bd is not None:
                params_kwargs["base_directory"] = Path(bd)
            # The subclass rebuilds its own tools from params; base `tools`
            # are user-supplied extras resolved via the registry and passed
            # through (the subclass merges them in **kwargs → super().__init__).
            kwargs = dict(
                name=agent_spec.name,
                model_config=runtime_model_config,
                max_tokens=agent_spec.max_tokens,
                allowed_peers=list(agent_spec.allowed_peers),
                bidirectional_peers=agent_spec.bidirectional_peers,
                can_escalate=agent_spec.can_escalate,
                input_schema=agent_spec.input_schema,
                output_schema=agent_spec.output_schema,
                memory_retention=agent_spec.memory_retention,
                memory_storage_path=agent_spec.memory_storage_path,
                plan_config=agent_spec.plan_config,
                **params_kwargs,
            )
            if hasattr(cls, "create_safe"):
                agent = await cls.create_safe(
                    **_accepted_kwargs(cls.create_safe, kwargs)
                )
            else:
                agent = cls(**_accepted_kwargs(cls, kwargs))

        if agent_spec.is_convergence_point is not None:
            agent._is_convergence_point = agent_spec.is_convergence_point
        agents.append(agent)
    return agents
