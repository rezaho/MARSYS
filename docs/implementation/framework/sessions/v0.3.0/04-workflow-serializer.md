# Framework Session 04: Workflow Definition Serializer

Required by Spren v0.4 (Python-file workflow round-trip, workflow CRUD persistence, frozen run snapshots) and by anyone wanting to persist a runtime topology to disk.

---

## Working rules

Same peer-collaboration norms as [`_template.md`](./_template.md). Read first.

### Foundational project rules

- The framework worktree's `CLAUDE.md`
- Framework architecture docs (topology module especially)
- [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md)

---

## The big picture

The framework's runtime topology is built from `@dataclass` types (`Node`, `Edge`, `Topology`, `PatternConfig`, `ExecutionConfig`, `ConvergencePolicyConfig`, `TracingConfig`, `StatusConfig`) and from `BaseAgent`, an ABC carrying live runtime state plus `tools: Dict[str, Callable]`. Today, anyone wanting a JSON shape for a workflow has to either roll their own conversion (Spren currently does this in `packages/spren/src/spren/models/`) or parse Python source via AST. There is no canonical, framework-owned, Pydantic-shaped representation.

This PR adds one. A new module `coordination/topology/serialize.py` (with thin sibling helpers in `agents/serialize.py` and `coordination/serialize.py`) defines a Pydantic `WorkflowDefinition` that mirrors the runtime shape, and a pair of pure conversion functions that round-trip between the runtime objects and the Pydantic spec. The Pydantic model is the canonical wire shape; its JSON Schema export is what every non-Python consumer reads.

### Multi-consumer (mandatory)

- **Spren** — Python-file import, workflow CRUD persistence (`workflows.definition` SQLite column), frozen run snapshots (SP-009)
- **CI integrations** — build pipelines that load a topology JSON from a config file and call `Orchestra.run()`
- **MARSYS Cloud** — pre-deploy validation of submitted topologies against the canonical schema
- **Community workflow templates / starter library** — share topologies as JSON; the JSON Schema export means non-Python consumers can validate locally
- **MARSYS Studio** — hosted authoring UI consuming the same shape
- **Workflow versioning UIs** — diff topologies across versions

If only one consumer can use this shape, the design is wrong — escalate.

---

## What came before this session

**Previous framework PRs from this dir:** Session 02 (`TelemetrySink` protocol) and Session 03 (pause/resume completion) are independent of this PR. Session 01 (NDJSON streaming tracing writer) is independent.

**State at start of this session:**

- Topology runtime types live in `packages/framework/src/marsys/coordination/topology/`. `Node`, `Edge`, and `Topology` are dataclasses (`core.py:46`, `core.py:79`, `core.py:147`). `Topology` carries a `metadata: Dict[str, Any]` field (`core.py:159`) that is currently used by the legacy-shim path but not by `pattern_converter`.
- Pattern presets live in `coordination/topology/patterns.py`. `PatternConfig` (`patterns.py:25`) is a dataclass with seven `PatternType` variants (`patterns.py:13`). The runtime-side conversion is in `coordination/topology/converters/pattern_converter.py`; today it produces raw `Node` / `Edge` lists and does not record the source pattern.
- `Agent` / `BaseAgent` constructor lives at `agents/agents.py:131`. The constructor accepts `tools: Optional[Dict[str, Callable[..., Any]]]` (`agents.py:137`), `goal`, `instruction`, `allowed_peers`, `bidirectional_peers`, `is_convergence_point`, `memory_retention`, `memory_storage_path`, `plan_config`, `input_schema`, `output_schema`, and `model: Union[BaseLocalModel, BaseAPIModel]`.
- `ModelConfig` is already a Pydantic `BaseModel` at `models/models.py:85` with `ConfigDict(extra="allow")` and a `_set_base_url_from_provider` validator. Its `model_dump()` / `model_validate()` are usable directly as the wire shape.
- `ExecutionConfig` is a dataclass at `coordination/config.py:232`. Its `convergence_policy` field is `Union[float, str, ConvergencePolicyConfig]` (`config.py:250`); `ConvergencePolicyConfig.from_value` (`config.py:156`) collapses the three forms to the canonical config. `TracingConfig` is a dataclass at `coordination/tracing/config.py:9`. `StatusConfig` is a dataclass at `coordination/config.py:20`.
- `Orchestra` is the public entry point (`coordination/orchestra.py:124`). Its constructor takes an `agent_registry`, optional `state_manager`, optional `communication_manager`, and optional `execution_config`. Topology is supplied per `Orchestra.run(task, topology, …)` call (`orchestra.py:544`); there is no unified `orchestra_state` bundle held inside `Orchestra`. Reading the source for these symbols is the first step of this session.
- `coordination/topology/graph.py` (`TopologyGraph`) is TRUNK-CRITICAL per `CLAUDE.md`. This PR does not touch it.
- No `WorkflowDefinition` Pydantic types exist in the framework today. Spren v0.3 Session 02 ships its own Pydantic mirrors at `packages/spren/src/spren/models/topology.py`, `packages/spren/src/spren/models/agent.py`, and `packages/spren/src/spren/models/execution_config.py` because the canonical serializer does not yet exist.

**Verify state with:**
```bash
cd /home/rezaho/research_projects/marsys-spren-work/packages/framework/
source ../../.venv/bin/activate
pytest tests/ -x --tb=short                                          # baseline test counts
git log --oneline -20 src/marsys/coordination/topology/
grep -rn 'class Node\b\|class Edge\b\|class Topology\b' src/marsys/coordination/topology/core.py
grep -rn 'class PatternConfig\b\|class PatternType\b' src/marsys/coordination/topology/patterns.py
grep -rn 'class ExecutionConfig\b\|class ConvergencePolicyConfig\b' src/marsys/coordination/config.py
grep -rn 'class ModelConfig\b' src/marsys/models/models.py
grep -rn 'class BaseAgent\b\|class Agent\b' src/marsys/agents/agents.py
grep -rn 'metadata' src/marsys/coordination/topology/converters/pattern_converter.py    # confirm pattern not yet recorded
```

---

## What this session ships

After merge, the framework owns a canonical Pydantic wire shape for an entire runnable workflow (topology + agent specs + execution config). Two pure conversion functions round-trip between this shape and runtime objects. A JSON Schema export pinned to JSON Schema 2020-12 is reachable as a standalone Python API. Every standard pattern preset, every node type, and every edge type round-trips losslessly under property-based testing. Framework regression suite stays green.

The merged PR exposes:

- `coordination/topology/serialize.py`:
  - `class WorkflowDefinition(BaseModel)` — top-level wire shape: `topology: TopologySpec`, `agents: dict[str, AgentSpec]` (key = agent name; same key referenced by `NodeSpec.agent_ref`), `execution_config: ExecutionConfigSpec`
  - `class TopologySpec(BaseModel)` — `nodes: list[NodeSpec]`, `edges: list[EdgeSpec]`, `metadata: dict[str, Any]` (carries `original_pattern` if set), `rules: list[str]` (rule references; full rule serialization is out of scope for this PR)
  - `class NodeSpec`, `class EdgeSpec`, `class PatternConfigSpec`
  - `def workflow_to_pydantic(orchestra: Orchestra, topology: Topology) -> WorkflowDefinition`
  - `def pydantic_to_topology(spec: WorkflowDefinition, tool_registry: Dict[str, Callable]) -> Topology`
- `agents/serialize.py`:
  - `class AgentSpec(BaseModel)` — Pydantic mirror of the framework's agent constructor surface; `tools` is `list[str]` of names, not callables
  - `def agent_to_pydantic(agent: BaseAgent) -> AgentSpec`
  - `def pydantic_to_agents(spec: WorkflowDefinition, tool_registry: Dict[str, Callable]) -> list[BaseAgent]`
- `coordination/serialize.py`:
  - `class ExecutionConfigSpec(BaseModel)`, `class ConvergencePolicyConfigSpec(BaseModel)`, `class TracingConfigSpec(BaseModel)`, `class StatusConfigSpec(BaseModel)`
  - `def execution_config_to_pydantic(config: ExecutionConfig) -> ExecutionConfigSpec`
  - `def pydantic_to_execution_config(spec: ExecutionConfigSpec) -> ExecutionConfig`
- `coordination/topology/exceptions.py`:
  - `class UnknownToolError(ValueError)` — raised when a `tool_registry` lookup misses
  - `class NonSerializableTopologyError(ValueError)` — raised on serialization paths that cannot be represented (e.g., a custom subclass with extra runtime state the spec cannot capture)
- `ModelConfig` is exported from `coordination.topology.serialize` as a re-export of `marsys.models.models.ModelConfig` (it is already Pydantic; no mirror)
- A 1-line addition to `coordination/topology/converters/pattern_converter.py` (the `PatternConfigConverter.convert` method) writes the source `PatternConfig` into `topology.metadata["original_pattern"]` as a `PatternConfigSpec.model_dump()` payload before returning the constructed `Topology`. This is a non-TRUNK-CRITICAL change.
- A standalone Python API for the JSON Schema: `from marsys.coordination.topology.serialize import WorkflowDefinition; WorkflowDefinition.model_json_schema()`. The schema declares JSON Schema 2020-12 (Pydantic v2's default) explicitly via `$schema`.
- Framework `CHANGELOG.md` entry under `## [Unreleased]`.

### Acceptance criteria

- [ ] `workflow_to_pydantic` and `pydantic_to_topology` (+ `pydantic_to_agents` + `pydantic_to_execution_config`) round-trip every standard `NodeType`, every `EdgeType`, every `EdgePattern`, and every `PatternType` preset under semantic equality
- [ ] `topology.metadata["original_pattern"]` is populated by `pattern_converter.convert` for every `PatternType` (HUB_AND_SPOKE, HIERARCHICAL, PIPELINE, MESH, STAR, RING, BROADCAST). Round-tripping a pattern-built topology through `workflow_to_pydantic` → `pydantic_to_topology` produces a re-applicable `PatternConfig` recoverable from `Topology.metadata`
- [ ] Polymorphic `convergence_policy` discriminator: bare-`float`, named-string (`"strict"`, `"majority"`, `"fail"`, `"user"`, `"any"`), and full `ConvergencePolicyConfigSpec` all round-trip through `ExecutionConfigSpec` and reduce to the same `ConvergencePolicyConfig` post-`pydantic_to_execution_config`. All three discriminant branches tested
- [ ] Bidirectional edge handling: `Topology.add_edge` auto-inserts a reverse edge for `bidirectional=True`; `workflow_to_pydantic` consolidates the user-declared edge + its synthesized reverse into a single `EdgeSpec` with `bidirectional=True`. Round-tripping does not double the edge count
- [ ] `AgentSpec.tools: list[str]` carries tool names; `pydantic_to_agents` resolves each name against the supplied `tool_registry`. Unknown name raises `UnknownToolError` with the missing name and a hint pointing at the `tool_registry` parameter. The error is a hard failure, not silent
- [ ] `ModelConfig` flows through `AgentSpec.agent_model: ModelConfig` directly (re-exported, not mirrored)
- [ ] Enums (`NodeType`, `EdgeType`, `EdgePattern`, `PatternType`) serialize via `Literal[...]` declarations or `model_config = ConfigDict(use_enum_values=True)`; the wire values match the framework's StrEnum values exactly (`"agent"`, `"user"`, `"system"`, `"tool"`, `"invoke"`, `"notify"`, `"query"`, `"stream"`, `"alternating"`, `"symmetric"`, `"hub_and_spoke"`, `"hierarchical"`, `"pipeline"`, `"mesh"`, `"star"`, `"ring"`, `"broadcast"`)
- [ ] `WorkflowDefinition.model_json_schema()` returns a non-empty schema declaring JSON Schema 2020-12 in the `$schema` field. The schema is documented in module docstring + framework docs as the source of truth for non-Python consumers
- [ ] Property-based round-trip tests via `hypothesis`: a constrained strategy generates valid topologies (concrete strategies for node count, valid edge connectivity over node-pairs, RESERVED_NODE_NAMES exclusion, valid pattern-arg shapes); each generated topology round-trips losslessly. Strategies are reviewed code, not hand-waved comments
- [ ] Test matrix covers (a) every `NodeType` × `EdgeType` × `EdgePattern` cell as an example-based test; (b) every `PatternType` preset round-trip end-to-end; (c) Hypothesis property tests for randomized topologies (≥100 examples per shape)
- [ ] Constructor path on rehydration: `pydantic_to_topology` constructs the `Topology` via `Topology(nodes=[...], edges=[...])`, never `model_construct` shortcuts, so `__post_init__` builds indices and runs validation
- [ ] `pydantic_to_topology(spec, tool_registry={})` raises `UnknownToolError` (not silently lossy) when any agent in `spec.agents` references a tool name with no callable in the registry. Empty `tools: []` is valid and does not require a registry entry
- [ ] **PR-coordination acceptance: open the Spren-side cleanup PR within 24 hours of this PR merging**; the cleanup PR deletes `packages/spren/src/spren/models/topology.py`, `packages/spren/src/spren/models/agent.py`, `packages/spren/src/spren/models/execution_config.py` and replaces their imports with `from marsys.coordination.topology.serialize import WorkflowDefinition, TopologySpec, NodeSpec, EdgeSpec, ExecutionConfigSpec` (and friends). The Spren-side cleanup is small (~50 lines deleted, ~10 lines of imports added). This PR's description tags the cleanup PR by branch name
- [ ] **Multi-consumer justification documented in PR description**: Spren / CI integrations / MARSYS Cloud / community templates / MARSYS Studio / workflow versioning UIs
- [ ] Framework regression suite green (zero new failures vs. baseline)
- [ ] No TRUNK-CRITICAL changes. Files NOT touched: `coordination/topology/graph.py`, `coordination/orchestra.py`, `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`. The `pattern_converter.py` edit is non-TRUNK-CRITICAL
- [ ] No Spren type imported (Spren is in `packages/spren/`, not `packages/framework/`)
- [ ] Framework `CHANGELOG.md` entry added under `## [Unreleased]`
- [ ] PR description references this session brief

---

## Background reading

1. The framework's `CLAUDE.md` — TRUNK-CRITICAL components and design principles DP-001..DP-007
2. Framework architecture docs in the framework worktree (especially the topology module)
3. [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md)
4. [`../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-005 (Pydantic source of truth), SP-018 (framework knows nothing of Spren)
5. [`../../../spren/sessions/02-workflow-crud-types.md`](../../../spren/sessions/02-workflow-crud-types.md) — the Spren-side mirrors that get deleted in the coordinated cleanup PR
6. `packages/framework/src/marsys/coordination/topology/core.py` — `Node`, `Edge`, `Topology` (read end-to-end; line numbers cited in this brief)
7. `packages/framework/src/marsys/coordination/topology/patterns.py` — `PatternConfig`, `PatternType`
8. `packages/framework/src/marsys/coordination/topology/converters/pattern_converter.py` — the file the 1-line `metadata["original_pattern"]` write lands in
9. `packages/framework/src/marsys/coordination/config.py` — `ExecutionConfig`, `ConvergencePolicyConfig`, `StatusConfig`
10. `packages/framework/src/marsys/coordination/tracing/config.py` — `TracingConfig`
11. `packages/framework/src/marsys/agents/agents.py` — `BaseAgent.__init__` constructor surface
12. `packages/framework/src/marsys/models/models.py` — `ModelConfig` (already Pydantic at line 85)
13. Pydantic v2 docs: discriminator unions, `Literal` enum handling, `model_validator(mode="before")` for normalizing union variants, `model_json_schema(schema_dialect=...)` defaults
14. Hypothesis docs: composite strategies, `assume()`, shrinking; for the constrained-topology generator

**Verify before proceeding:**
- Capture baseline test counts BEFORE any change
- `git log --oneline -20 packages/framework/src/marsys/coordination/topology/` and `…/coordination/config.py` and `…/agents/agents.py` to see recent activity
- Read every line-cited file end-to-end; if a citation has drifted (a refactor renamed a symbol or moved a line), update the brief or escalate before writing code

---

## Detailed plan

### Step 0 — Baseline + audit

Capture baseline regression counts. Read every file the brief cites. Confirm `ModelConfig` is the only Pydantic type in scope; everything else needs mirroring. Confirm `Topology.metadata` exists and is currently unused by `pattern_converter`. Confirm `BaseAgent.tools` is `Dict[str, Callable]` and that the supported value space for tool names is plain identifiers.

### Step 1 — Define the Pydantic wire shape

Create `coordination/topology/serialize.py`. Define the model hierarchy in dependency order:

1. `class PatternConfigSpec(BaseModel)`: `pattern: Literal["hub_and_spoke", "hierarchical", "pipeline", "mesh", "star", "ring", "broadcast"]`, `params: dict[str, Any]`, `metadata: dict[str, Any] = {}`
2. `class NodeSpec(BaseModel)`: `name: str`, `node_type: Literal["user", "agent", "system", "tool"] = "agent"`, `agent_ref: str | None = None`, `is_convergence_point: bool = False`, `metadata: dict[str, Any] = {}`. The `agent_ref` field carries the agent name; the runtime `Node.agent_ref` Python reference is reconstructed at `pydantic_to_topology` time from the agent registry built by `pydantic_to_agents`
3. `class EdgeSpec(BaseModel)`: `source: str`, `target: str`, `edge_type: Literal["invoke", "notify", "query", "stream"] = "invoke"`, `bidirectional: bool = False`, `pattern: Literal["alternating", "symmetric"] | None = None`, `metadata: dict[str, Any] = {}`
4. `class TopologySpec(BaseModel)`: `nodes: list[NodeSpec]`, `edges: list[EdgeSpec]`, `metadata: dict[str, Any] = {}`, `rules: list[str] = []`
5. `class WorkflowDefinition(BaseModel)`: `topology: TopologySpec`, `agents: dict[str, AgentSpec]`, `execution_config: ExecutionConfigSpec`

Do the same for execution-side specs in `coordination/serialize.py`:

6. `class ConvergencePolicyConfigSpec(BaseModel)`: `min_ratio: float`, `on_insufficient: Literal["proceed", "fail", "user"]`, `terminate_orphans: bool`, `log_level: Literal["info", "warning", "error"]`
7. `class StatusConfigSpec(BaseModel)`, `class TracingConfigSpec(BaseModel)` mirroring the dataclass fields
8. `class ExecutionConfigSpec(BaseModel)` with the polymorphic `convergence_policy` field: declare it `Union[float, str, ConvergencePolicyConfigSpec]` and add a `model_validator(mode="before")` that:
   - leaves a bare-float as-is (no normalization at the spec level — the runtime conversion via `pydantic_to_execution_config` lets `ConvergencePolicyConfig.from_value` resolve it)
   - leaves a bare-str as-is
   - passes a dict / nested-spec through unchanged
   The validator preserves the discriminant branch chosen by the producer; round-trip emits the same branch the input used. Bidirectional consistency between the three discriminants is documented and tested.

`AgentSpec` lives in `agents/serialize.py`:

9. `class AgentSpec(BaseModel)`: `name: str`, `goal: str`, `instruction: str`, `agent_model: ModelConfig` (re-exported from `marsys.models.models`), `tools: list[str] = []`, `max_tokens: int | None = 10000`, `allowed_peers: list[str] = []`, `bidirectional_peers: bool = False`, `is_convergence_point: bool | None = None`, `memory_retention: Literal["single_run", "session", "persistent"] = "session"`, `memory_storage_path: str | None = None`, `plan_config: dict[str, Any] | None = None`, `input_schema: dict[str, Any] | None = None`, `output_schema: dict[str, Any] | None = None`. The `agent_model` field name is deliberately not `model` (Pydantic v2 reserves the `model_*` namespace).

All Pydantic models declare `model_config = ConfigDict(extra="forbid")` to fail loudly when a producer sends an unknown field, except where existing framework dataclasses use `extra="allow"`-like flexibility (`Topology.metadata`, `Node.metadata`, `Edge.metadata` are unconstrained dicts already; `ConfigDict(extra="forbid")` applies to the spec models, not their nested `metadata` dicts).

### Step 2 — Conversion functions

`agents/serialize.py`:

```python
def agent_to_pydantic(agent: BaseAgent) -> AgentSpec:
    return AgentSpec(
        name=agent.name,
        goal=agent.goal,
        instruction=agent.instruction,
        agent_model=agent.model.config,        # ModelConfig is on the model adapter
        tools=list(agent.tools.keys()),
        max_tokens=agent.max_tokens,
        allowed_peers=sorted(agent._allowed_peers_init),
        bidirectional_peers=agent._bidirectional_peers,
        is_convergence_point=agent._is_convergence_point,
        memory_retention=agent._memory_retention,
        plan_config=agent._planning_config.to_dict() if agent._planning_config else None,
        input_schema=agent.input_schema,
        output_schema=agent.output_schema,
    )

def pydantic_to_agents(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> list[BaseAgent]:
    agents: list[BaseAgent] = []
    for agent_name, agent_spec in spec.agents.items():
        tools_dict: Dict[str, Callable] = {}
        for tool_name in agent_spec.tools:
            if tool_name not in tool_registry:
                raise UnknownToolError(
                    f"Tool '{tool_name}' (used by agent '{agent_name}') "
                    f"is not registered. Pass a tool_registry mapping name → callable "
                    f"to pydantic_to_agents."
                )
            tools_dict[tool_name] = tool_registry[tool_name]
        agents.append(Agent(
            name=agent_spec.name,
            goal=agent_spec.goal,
            instruction=agent_spec.instruction,
            model=ModelConfigToModel(agent_spec.agent_model),    # whichever adapter the framework already exposes
            tools=tools_dict,
            ...
        ))
    return agents
```

`coordination/topology/serialize.py`:

```python
def workflow_to_pydantic(orchestra: Orchestra, topology: Topology) -> WorkflowDefinition:
    """
    Capture a runnable workflow as a Pydantic spec.

    The Orchestra is the source of execution config + agent registry; the
    topology is the structural source. Bidirectional edges are consolidated
    (the auto-inserted reverse is folded back into the single EdgeSpec with
    bidirectional=True). Pattern provenance from Topology.metadata['original_pattern']
    is preserved into TopologySpec.metadata.
    """
    ...

def pydantic_to_topology(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> Topology:
    """
    Hydrate a runnable Topology from a spec. Constructs Topology via the
    canonical Topology(nodes=..., edges=...) path so __post_init__ runs and
    indices build. Resolves agent_ref Names against the agents materialized
    by pydantic_to_agents.
    """
    ...
```

The bidirectional consolidation in `workflow_to_pydantic` walks `topology.edges`, identifies pairs `(a → b, b → a)` where both have `bidirectional=True` and `edge_type` matches, and emits one `EdgeSpec(source=a, target=b, bidirectional=True)` for the canonical direction (lex-smaller endpoint first). Single-direction edges (`bidirectional=False`) emit one `EdgeSpec` each.

### Step 3 — Pattern preservation

Edit `coordination/topology/converters/pattern_converter.py` so `PatternConfigConverter.convert` writes the source `PatternConfig` into the constructed topology before returning. The change:

```python
@staticmethod
def convert(config: PatternConfig) -> Topology:
    topology = ...        # existing branch dispatch
    topology.metadata["original_pattern"] = PatternConfigSpec(
        pattern=config.pattern.value,
        params=config.params,
        metadata=config.metadata,
    ).model_dump()
    return topology
```

The forward dispatch (`_hub_and_spoke`, `_hierarchical`, …) is untouched. The 1-line edit lives at the top of `convert`. Round-trip semantics: `pydantic_to_topology` reads `spec.topology.metadata["original_pattern"]` (if present), reconstructs a `PatternConfig`, and the caller can re-apply it via `PatternConfigConverter.convert` if they want a freshly-rebuilt topology. The on-disk topology's nodes/edges are still the source of truth; `original_pattern` is the round-trippable provenance.

### Step 4 — JSON Schema export

Confirm Pydantic v2's default schema dialect is JSON Schema 2020-12. Document this in the module docstring of `coordination/topology/serialize.py` and in the framework docs entry that this PR adds. Provide a small helper:

```python
def workflow_definition_schema() -> dict:
    """Return the JSON Schema for WorkflowDefinition. Dialect: JSON Schema 2020-12."""
    schema = WorkflowDefinition.model_json_schema()
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    return schema
```

This is the reachable Python API path the `openapi-typescript` + `datamodel-code-generator` Spren-side pipeline reads from.

### Step 5 — Tests

Tests live in `packages/framework/tests/coordination/topology/test_serialize.py` and adjacent files mirroring the module layout.

Example-based tests:
- One test per `NodeType` value verifying `NodeSpec` round-trip
- One test per `EdgeType` × `EdgePattern` combination
- One test per `PatternType` preset: build via `PatternConfigConverter.convert(PatternConfig.<factory>(...))`, serialize, deserialize, compare nodes/edges by semantic equality, confirm `metadata["original_pattern"]` round-trips
- `convergence_policy` discriminator: one test for bare-float (`1.0`, `0.5`), one for each named string (`"strict"`, `"majority"`, `"fail"`, `"user"`, `"any"`), one for full `ConvergencePolicyConfigSpec`. Each round-trips through `ExecutionConfigSpec` and reduces to the same canonical `ConvergencePolicyConfig` after `pydantic_to_execution_config` (compared via `ConvergencePolicyConfig.from_value` semantic-equality)
- Bidirectional consolidation: build a topology with `Edge(a, b, bidirectional=True)` (which auto-inserts reverse). Serialize → exactly one `EdgeSpec`. Deserialize → topology with two edges (forward + reverse) again. Confirm round-trip is fixed-point (no edge-doubling on repeat)
- Tool-registry resolution: `pydantic_to_agents(spec, tool_registry={"web_search": stub})` builds agent with `tools={"web_search": stub}`. Missing-key raises `UnknownToolError` with the missing name and the agent name in the error message

Property-based tests (Hypothesis, `≥100` examples):
- Composite strategy `topology_strategy()` builds a valid `Topology` by:
  1. Sampling N ∈ [2, 8] node names from a `text(min_size=1, max_size=12, alphabet=ascii_letters)` strategy filtered to exclude `RESERVED_NODE_NAMES`
  2. Sampling node types per node from `sampled_from(NodeType)`
  3. Sampling K ∈ [N-1, N*(N-1)] directed edges over distinct node-pairs (`assume` no self-loops)
  4. Sampling `bidirectional` and `pattern` independently per edge
  Each generated topology is assembled, serialized, deserialized, and asserted equal under semantic equality (node sets, edge sets, edge_type / bidirectional flag / pattern preservation).
- Composite strategy `pattern_strategy()` builds a valid `PatternConfig` per `PatternType`, with constrained `params` (e.g., HUB_AND_SPOKE: hub + spokes lists; PIPELINE: stages list-of-dicts with `agents` keys). Each pattern is converted via `PatternConfigConverter.convert`, serialized, deserialized, and the recovered `original_pattern` metadata re-builds an equivalent `PatternConfig`.

The constrained strategies are concrete code in `tests/coordination/topology/strategies.py`, not implementer-discretion comments in the brief.

Multi-consumer test:
- A test fixture acts as a "non-Spren consumer" by exercising the JSON Schema + a sample payload via the `jsonschema` library. The schema validates the sample; an intentionally-broken sample fails validation. This proves the schema is usable by non-Python tooling, not just the Pydantic models themselves.

### Step 6 — Framework docs + CHANGELOG

- Module docstring in `coordination/topology/serialize.py` documents the JSON Schema dialect, the round-trip contract, and the `tool_registry` parameter
- `packages/framework/CHANGELOG.md` adds an entry under `## [Unreleased]` describing the new `WorkflowDefinition` shape, the conversion functions, the `original_pattern` metadata convention, and the JSON Schema export
- A short framework docs page (location TBD by the framework team during review) explains the canonical wire shape and how non-Python consumers should validate against the JSON Schema

### Files to create

- `packages/framework/src/marsys/coordination/topology/serialize.py`
- `packages/framework/src/marsys/coordination/topology/exceptions.py`
- `packages/framework/src/marsys/agents/serialize.py`
- `packages/framework/src/marsys/coordination/serialize.py`
- `packages/framework/tests/coordination/topology/test_serialize.py`
- `packages/framework/tests/coordination/topology/strategies.py`
- `packages/framework/tests/agents/test_serialize.py`
- `packages/framework/tests/coordination/test_serialize.py`

### Files to modify

- `packages/framework/src/marsys/coordination/topology/converters/pattern_converter.py` — add the 1-line `topology.metadata["original_pattern"] = PatternConfigSpec(...).model_dump()` write at the top of `PatternConfigConverter.convert`
- `packages/framework/CHANGELOG.md` — `## [Unreleased]` entry

### Files NOT to touch

- TRUNK-CRITICAL: `coordination/topology/graph.py`, `coordination/orchestra.py`, `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`
- `coordination/topology/core.py` — no edits to `Node` / `Edge` / `Topology` themselves; the metadata field already exists
- `agents/agents.py` — no edits to `BaseAgent`; the constructor surface is read-only for this PR
- `models/models.py` — `ModelConfig` is re-exported, not modified
- Spren-side code (`packages/spren/`) — that's the coordinated cleanup PR, not this one

### Pre-flight escalation gate

If, during implementation, any work proposed in this brief turns out to require modifying a TRUNK-CRITICAL file or a non-additive change to `coordination/topology/core.py`, **escalate via `AskUserQuestion` for ADR review before writing code**. The brief's expectation is that all work fits in the new modules + the 1-line `pattern_converter.py` edit.

### Load-bearing shapes

```python
# coordination/topology/serialize.py — reference signatures
from marsys.coordination.topology.core import Topology
from marsys.coordination.orchestra import Orchestra
from marsys.coordination.config import ExecutionConfig

def workflow_to_pydantic(
    orchestra: Orchestra,
    topology: Topology,
) -> WorkflowDefinition: ...

def pydantic_to_topology(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> Topology: ...

def pydantic_to_agents(
    spec: WorkflowDefinition,
    tool_registry: Dict[str, Callable],
) -> list[BaseAgent]: ...

def pydantic_to_execution_config(
    spec: ExecutionConfigSpec,
) -> ExecutionConfig: ...
```

```python
# WorkflowDefinition top-level shape
class WorkflowDefinition(BaseModel):
    topology: TopologySpec
    agents: dict[str, AgentSpec]                  # key = agent name
    execution_config: ExecutionConfigSpec

# ExecutionConfigSpec convergence_policy is polymorphic
class ExecutionConfigSpec(BaseModel):
    convergence_policy: Union[float, str, ConvergencePolicyConfigSpec] = 1.0
    convergence_timeout: float = 300.0
    branch_timeout: float = 600.0
    # ... mirror of ExecutionConfig dataclass fields
    status: StatusConfigSpec = Field(default_factory=StatusConfigSpec)
    tracing: TracingConfigSpec = Field(default_factory=TracingConfigSpec)

    @model_validator(mode="before")
    @classmethod
    def normalize_convergence_policy(cls, data: Any) -> Any:
        # bare-float and bare-str pass through; dict is parsed as ConvergencePolicyConfigSpec
        ...
```

---

## Hard rules

### Multi-consumer justification (mandatory)

- [ ] At least one consumer beyond Spren named in the PR description (Spren / CI integrations / MARSYS Cloud / community templates / MARSYS Studio / workflow versioning UIs)
- [ ] No Spren type imported (Spren is in `packages/spren/`, not `packages/framework/`)
- [ ] No "if running under Spren" code paths

### Framework design principles

Per the framework's `CLAUDE.md` § design principles:
- DP-001: pure agent logic — n/a (this PR doesn't touch agent execution)
- DP-002: centralized validation — n/a
- DP-003: unified-barrier orchestration — n/a
- DP-004: branch isolation — n/a
- DP-005: topology-driven routing — preserved (round-trip preserves nodes / edges / `bidirectional` / `edge_type`, which is what `TopologyGraph` reads)
- DP-006: adapter pattern — `ModelConfig` is the adapter-config shape; this PR re-exports it, doesn't introduce a parallel
- DP-007: format pluggability — n/a

If this feature would force a violation of any of these, **escalate** before writing code.

### No TRUNK-CRITICAL changes

This PR adds one new sibling module per affected package and edits one non-TRUNK-CRITICAL file (`pattern_converter.py`). If implementation requires anything else, escalate.

### Clean code rules

- Smallest implementation that passes acceptance criteria
- Pure functions; no side effects in the conversion path (the `pattern_converter.py` write is a deterministic record, not a side effect on global state)
- One concern per file: `serialize.py` modules export specs + conversions; `exceptions.py` exports the two error types; tests mirror the source layout
- No descriptive comments for self-naming code — only WHY when not obvious

---

## Tests (required for "done")

### Unit tests

- `tests/coordination/topology/test_serialize.py` — node / edge / topology / pattern round-trips per the example-based matrix
- `tests/coordination/test_serialize.py` — execution-config round-trips, all three `convergence_policy` discriminant branches, `TracingConfig` + `StatusConfig` round-trips
- `tests/agents/test_serialize.py` — agent round-trips, tool-registry resolution, `UnknownToolError` raising, `ModelConfig` direct flow

### Integration tests

- `tests/integration/test_workflow_definition_round_trip.py` — full `WorkflowDefinition` round-trip on a non-trivial pattern-built topology (e.g., `PatternConfig.hub_and_spoke(...)` with 5 spokes + parallel rule + tracing enabled). Build → serialize → deserialize → re-run via `Orchestra.run()` against the rehydrated topology. The rehydrated topology must produce a structurally identical `OrchestraResult` envelope.

### Property-based tests

- `tests/coordination/topology/test_serialize_hypothesis.py` — Hypothesis tests over `topology_strategy()` and `pattern_strategy()`, ≥100 examples per shape

### Framework regression test

- Entire framework test suite passes with the same counts as the pre-change baseline. Document baseline + post-change counts in "What was actually built"

### Multi-consumer test

- `tests/coordination/topology/test_schema_consumer.py` — exercises `WorkflowDefinition.model_json_schema()` via the `jsonschema` library against a sample payload. Confirms a non-Python consumer can validate a wire payload against the canonical schema

---

## Open questions for the framework team

1. **Module placement of execution-side specs.** Brief proposes `coordination/serialize.py` for `ExecutionConfigSpec`, `ConvergencePolicyConfigSpec`, `TracingConfigSpec`, `StatusConfigSpec`. Alternative is colocating each spec next to its dataclass (e.g., a `tracing/serialize.py`). Confirm placement.
2. **`AgentSpec.agent_model` access path on the runtime side.** The brief reads `agent.model.config` on `agent_to_pydantic`; verify which attribute on the live model adapter exposes the originating `ModelConfig`. If the adapter does not retain the `ModelConfig` instance, escalate — the model rehydration path needs an additional design step.
3. **Plan-config and schema-field round-trips.** `plan_config: dict[str, Any] | None` and `input_schema` / `output_schema: dict[str, Any] | None` are passed through opaquely. Confirm this is sufficient for v0.4 consumers; if the framework wants typed schemas for these, they belong in a follow-up PR.

---

## Sign-off

On completion:

1. Update **What was actually built** below with the delta from the plan, if any
2. Update [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md) — check Session 04's row
3. Note the framework release version that ships this feature (e.g., "shipped in framework v0.4.0")
4. Open the coordinated Spren-side cleanup PR within 24 hours and tag it in this PR's description
5. Add a **Lessons / Surprises** entry below

### What was actually built (filled by implementer)

> _Implementer fills this in._
>
> Include: baseline test counts (before change), post-change test counts (must match for regression suite + new tests added), framework PR number + URL, framework release version that includes this feature, the coordinated Spren-side cleanup PR URL + merge timing, anything done differently from the plan with reasons.

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._
