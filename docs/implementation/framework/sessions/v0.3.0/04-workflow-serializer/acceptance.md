# Acceptance criteria — Framework Session 04: Workflow Definition Serializer

Frozen at 2026-05-12T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

## Public API surface

- AC-1: A module `marsys.coordination.topology.serialize` exposes class `WorkflowDefinition` as a Pydantic `BaseModel` with fields `topology: TopologySpec`, `agents: dict[str, AgentSpec]`, and `execution_config: ExecutionConfigSpec`.
- AC-2: The same module exposes class `TopologySpec` as a Pydantic `BaseModel` with fields `nodes: list[NodeSpec]`, `edges: list[EdgeSpec]`, `metadata: dict[str, Any]`, and `rules: list[str]`.
- AC-3: The same module exposes class `NodeSpec` as a Pydantic `BaseModel` with fields `name: str`, `node_type: Literal["user", "agent", "system", "tool"]` (default `"agent"`), `agent_ref: str | None` (default `None`), `is_convergence_point: bool` (default `False`), and `metadata: dict[str, Any]`.
- AC-4: The same module exposes class `EdgeSpec` as a Pydantic `BaseModel` with fields `source: str`, `target: str`, `edge_type: Literal["invoke", "notify", "query", "stream"]` (default `"invoke"`), `bidirectional: bool` (default `False`), `pattern: Literal["alternating", "symmetric"] | None` (default `None`), and `metadata: dict[str, Any]`.
- AC-5: The same module exposes class `PatternConfigSpec` as a Pydantic `BaseModel` with fields `pattern: Literal["hub_and_spoke", "hierarchical", "pipeline", "mesh", "star", "ring", "broadcast"]`, `params: dict[str, Any]`, and `metadata: dict[str, Any]`.
- AC-6: The same module exposes top-level functions `workflow_to_pydantic(orchestra, topology) -> WorkflowDefinition` and `pydantic_to_topology(spec, tool_registry) -> Topology`.
- AC-7: A module `marsys.agents.serialize` exposes class `AgentSpec` as a Pydantic `BaseModel` and top-level functions `agent_to_pydantic(agent) -> AgentSpec` and `pydantic_to_agents(spec, tool_registry) -> list[Agent]`.
- AC-8: A module `marsys.coordination.serialize` exposes classes `ExecutionConfigSpec`, `ConvergencePolicyConfigSpec`, `TracingConfigSpec`, `StatusConfigSpec` (all Pydantic `BaseModel`) and top-level functions `execution_config_to_pydantic(config) -> ExecutionConfigSpec` and `pydantic_to_execution_config(spec) -> ExecutionConfig`.
- AC-9: A module `marsys.models.serialize` exposes class `ModelConfigSpec` and top-level functions `model_config_spec_from_runtime(config) -> ModelConfigSpec` and `runtime_model_config_from_spec(spec, api_key=None) -> ModelConfig`.
  - _[amended 2026-05-16 — Session 07] signature is now `runtime_model_config_from_spec(spec, api_key=None, *, runnable=True) -> ModelConfig` (additive keyword-only param; default flips to runnable). Rationale: `../07-canonical-workflow-repro.md` §7._
- AC-10: A module `marsys.coordination.topology.exceptions` exposes class `UnknownToolError` (subclass of `ValueError`) and class `NonSerializableTopologyError` (subclass of `ValueError`).
- AC-11: A public helper `topology_equals(a: Topology, b: Topology) -> bool` is exposed from `marsys.coordination.topology.serialize`.

## Functional — round-trip semantic equality

- AC-12: For every value of `NodeType` (i.e., `"user"`, `"agent"`, `"system"`, `"tool"`), constructing a `Topology` containing a node of that type, then calling `workflow_to_pydantic` followed by `pydantic_to_topology`, yields a `Topology` that satisfies `topology_equals(original, round_tripped) is True`.
- AC-13: For every value of `EdgeType` (i.e., `"invoke"`, `"notify"`, `"query"`, `"stream"`) crossed with every value of `EdgePattern` (i.e., `"alternating"`, `"symmetric"`), constructing a `Topology` containing such an edge round-trips losslessly under `topology_equals`.
- AC-14: For every `PatternType` preset (`HUB_AND_SPOKE`, `HIERARCHICAL`, `PIPELINE`, `MESH`, `STAR`, `RING`, `BROADCAST`), building a topology via `PatternConfigConverter.convert(PatternConfig.<factory>(...))`, then round-tripping through `workflow_to_pydantic` and `pydantic_to_topology`, yields a `Topology` satisfying `topology_equals(original, round_tripped) is True`.
- AC-15: `topology_equals` compares nodes by name + type + metadata, compares edges as a multiset over `(source, target, edge_type, bidirectional, pattern, metadata)`, and compares `metadata["original_pattern"]`. It returns `False` when any of these differ and `True` when all are equal.

## Functional — pattern preservation

- AC-16: After invoking `PatternConfigConverter.convert(config)` for any `PatternType` preset, the returned `Topology` carries `metadata["original_pattern"]` equal to the dict produced by `PatternConfigSpec(pattern=config.pattern.value, params=config.params, metadata=config.metadata).model_dump()`.
- AC-17: Round-tripping a pattern-built topology through `workflow_to_pydantic` → `pydantic_to_topology` preserves `metadata["original_pattern"]` such that a caller can reconstruct an equivalent `PatternConfig` from the recovered metadata and rebuild an equivalent topology.

## Functional — convergence policy discriminator

- AC-18: An `ExecutionConfigSpec` populated with a bare-float `convergence_policy` (e.g., `1.0`, `0.5`) round-trips through JSON serialization and emits the bare-float branch on output (no normalization at the spec level).
- AC-19: An `ExecutionConfigSpec` populated with a named-string `convergence_policy` from the set `{"strict", "majority", "fail", "user", "any"}` round-trips through JSON serialization and emits the bare-string branch on output.
- AC-20: An `ExecutionConfigSpec` populated with a full `ConvergencePolicyConfigSpec` object round-trips through JSON serialization and emits the full-spec branch on output.
- AC-21: All three discriminant branches (bare-float, named-string, full spec) when passed to `pydantic_to_execution_config` produce an `ExecutionConfig` whose `convergence_policy` is the same canonical `ConvergencePolicyConfig` as produced by `ConvergencePolicyConfig.from_value` applied to the input.

## Functional — bidirectional edge handling

- AC-22: When the input `Topology` has a user-declared bidirectional edge (which `Topology.add_edge` auto-inserts as a reverse edge), `workflow_to_pydantic` emits exactly one `EdgeSpec` with `bidirectional=True` for the canonical direction (lex-smaller endpoint as `source`), not two `EdgeSpec` entries.
- AC-23: Calling `pydantic_to_topology` on a `WorkflowDefinition` containing one bidirectional `EdgeSpec` produces a `Topology` with both the forward and reverse edges materialized.
- AC-24: The round-trip is a fixed point: round-tripping a bidirectional-edge topology N times (N ≥ 2) produces the same edge count as round-tripping once; edges do not double on repeat.

## Functional — tool registry resolution

- AC-25: `pydantic_to_agents(spec, tool_registry={"web_search": stub})` for an `AgentSpec` with `tools=["web_search"]` produces an `Agent` whose `tools` mapping contains `{"web_search": stub}`.
- AC-26: `pydantic_to_agents(spec, tool_registry={})` (or any registry missing a name referenced by an `AgentSpec.tools` entry) raises `UnknownToolError`.
- AC-27: The raised `UnknownToolError` message includes both the missing tool name and the agent name that referenced it, and points at the `tool_registry` parameter.
- AC-28: An `AgentSpec` with `tools=[]` is valid and `pydantic_to_agents` succeeds with no entries in the registry needed.
- AC-29: `pydantic_to_topology(spec, tool_registry={})` raises `UnknownToolError` (not a silent loss) when any agent in `spec.agents` references a tool name with no callable in the registry.

## Functional — ModelConfigSpec

- AC-30: `ModelConfigSpec` has no `api_key` field; instantiating it does not run an `api_key`-resolving validator.
- AC-31: `AgentSpec.agent_model` is typed as `ModelConfigSpec` (not `marsys.models.ModelConfig`).
- AC-32: `agent_to_pydantic(agent)` populates `AgentSpec.agent_model` by calling `model_config_spec_from_runtime(agent._model_config)`.
- AC-33: `runtime_model_config_from_spec(spec)` produces a `ModelConfig` whose non-secret fields equal those of `spec` (round-trip preservation of every non-`api_key` field, including `oauth_profile`).
  - _[amended 2026-05-16 — Session 07] this preservation/no-raise guarantee now holds for `runtime_model_config_from_spec(spec, runnable=False)` (the explicit inspection opt-in). The bare default is now `runnable=True` and resolves the credential / raises if unreachable, exactly like a directly-constructed `ModelConfig`. Reason: the silent-non-runnable default was a footgun that made canonical workflows fail on standard-API-key providers. See `../07-canonical-workflow-repro.md`._
- AC-34: `runtime_model_config_from_spec(spec, api_key="sk-...")` produces a valid `ModelConfig` with the supplied key.
  - _[unaffected 2026-05-16 — Session 07] explicit-key path still validates; behavior unchanged under the new default._
- AC-35: A `ModelConfigSpec` (e.g., `type="api"`, `name="gpt-4o"`, `provider="openai"`) round-trips through JSON with no `api_key` field present in the JSON payload.

## Functional — WorkflowDefinition cross-reference validator

- AC-36: Constructing a `WorkflowDefinition` whose `topology.nodes` contains a `NodeSpec` with `agent_ref="X"` where `"X"` is not a key in `agents` raises a Pydantic `ValidationError`.
- AC-37: Constructing a `WorkflowDefinition` whose `topology.edges` contains an `EdgeSpec` with `source` not equal to any `NodeSpec.name` raises a Pydantic `ValidationError`.
- AC-38: Constructing a `WorkflowDefinition` whose `topology.edges` contains an `EdgeSpec` with `target` not equal to any `NodeSpec.name` raises a Pydantic `ValidationError`.
- AC-39: A `WorkflowDefinition` with all cross-references valid (all `agent_ref` values present in `agents`; all edge endpoints present in `nodes`) constructs successfully.

## Functional — enum / Literal wire values

- AC-40: When serialized to JSON, enum-valued fields emit exactly these string values: `NodeType` → `"agent"`, `"user"`, `"system"`, `"tool"`; `EdgeType` → `"invoke"`, `"notify"`, `"query"`, `"stream"`; `EdgePattern` → `"alternating"`, `"symmetric"`; `PatternType` → `"hub_and_spoke"`, `"hierarchical"`, `"pipeline"`, `"mesh"`, `"star"`, `"ring"`, `"broadcast"`.

## Functional — JSON Schema export

- AC-41: `WorkflowDefinition.model_json_schema()` returns a non-empty dict.
- AC-42: The returned schema has `schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"`.
- AC-43: A standalone Python API path `from marsys.coordination.topology.serialize import WorkflowDefinition; WorkflowDefinition.model_json_schema()` works without additional setup.

## Functional — constructor path on rehydration

- AC-44: `pydantic_to_topology` constructs the `Topology` via `Topology(nodes=..., edges=...)` such that `Topology.__post_init__` runs and indices are built. (Verifiable behaviorally: rehydrated topology exposes the same indices and validation behavior as a freshly-constructed topology, and any `__post_init__`-driven invariant is observable.)

## Functional — is_convergence_point post-construction

- AC-45: Given an `AgentSpec` with `is_convergence_point=True`, `pydantic_to_agents` produces an `Agent` whose `_is_convergence_point` attribute is `True` even though `Agent.__init__` does not accept `is_convergence_point` as a constructor kwarg.
- AC-46: Given an `AgentSpec` with `is_convergence_point=False`, the resulting `Agent._is_convergence_point` is `False`.
- AC-47: Given an `AgentSpec` with `is_convergence_point=None`, `pydantic_to_agents` does not override the default `_is_convergence_point` set by `Agent.__init__`.

## Functional — workflow_definition_schema helper

- AC-48: `marsys.coordination.topology.serialize.workflow_definition_schema()` returns the JSON Schema for `WorkflowDefinition`, identical to `WorkflowDefinition.model_json_schema()`.
- AC-49: `workflow_definition_schema()` asserts (or guarantees) the returned schema declares `"$schema": "https://json-schema.org/draft/2020-12/schema"`.

## Functional — property-based coverage

- AC-50: Hypothesis-based property tests for randomized topology round-trips execute at least 100 examples per shape (per `topology_strategy` and per `pattern_strategy`).
- AC-51: The `topology_strategy()` generator is concrete code in `tests/coordination/topology/strategies.py` (not implementer-discretion text), and produces valid topologies obeying: node count N ∈ [2, 8] from an ASCII-letters identifier strategy excluding `RESERVED_NODE_NAMES`; node types sampled from `NodeType`; K ∈ [N-1, N*(N-1)] directed edges over distinct node-pairs (no self-loops); `bidirectional` and `pattern` sampled independently per edge.
- AC-52: The `pattern_strategy()` generator is concrete code in `tests/coordination/topology/strategies.py` and produces a valid `PatternConfig` per `PatternType` with constrained `params`.
- AC-53: Every Hypothesis-generated topology round-trips losslessly under semantic equality (node sets, edge sets, edge_type, bidirectional flag, pattern preservation).
- AC-54: Every Hypothesis-generated `PatternConfig` converted via `PatternConfigConverter.convert`, serialized, and deserialized produces a recovered `original_pattern` metadata that re-builds an equivalent `PatternConfig`.

## Functional — multi-consumer JSON Schema test

- AC-55: A test (located at `tests/coordination/topology/test_schema_consumer.py`) loads `WorkflowDefinition.model_json_schema()` and validates a sample payload against it via the `jsonschema` library (i.e., not via Pydantic's own validators).
- AC-56: The same test demonstrates that an intentionally-broken sample fails `jsonschema` validation against the same schema.

## Functional — integration round-trip

- AC-57: A non-trivial pattern-built topology (e.g., `PatternConfig.hub_and_spoke(...)` with 5 spokes + parallel rule + tracing enabled) round-trips through `workflow_to_pydantic` → `pydantic_to_topology`. **Amended 2026-05-12 [structural-only substitution]**: the integration test asserts that the rehydrated topology is structurally identical (every node, every edge, every rule, every metadata key preserved under `topology_equals`) rather than re-running via `Orchestra.run()`. Justification: re-running requires live LLM calls (cost + flakiness); the structural equality is the load-bearing contract because the orchestrator (`TopologyGraph`, `RealRuntime`, `Orchestrator`) reads only from the same node/edge surface the test asserts equal. A separate live-tests harness exercising `Orchestra.run()` against rehydrated topologies is a follow-up.

## Non-functional — error handling

- AC-58: `UnknownToolError` is raised (never silently swallowed; never replaced by a generic exception) on any missing tool name during `pydantic_to_agents` / `pydantic_to_topology`.
- AC-59: `NonSerializableTopologyError` is raised by `workflow_to_pydantic` when a `topology.nodes` entry is a `DeterministicNode` (instance of `StartNode`, `EndNode`, or `UserNode` from `coordination/execution/det_nodes.py`). Det-nodes carry execution-runtime state beyond what `NodeSpec` captures, so the wire shape refuses them explicitly rather than producing a partial spec. Tests demonstrate this by constructing a topology with at least one `DeterministicNode` and asserting the exception is raised with a message that names the offending node and points the caller at the workaround (drop the det-node before serializing, or open a separate PR to extend the spec).

## Non-functional — strictness

- AC-60: Pydantic spec models declare `model_config = ConfigDict(extra="forbid")` such that passing an unknown top-level field to any spec raises `ValidationError`. (Nested `metadata` dicts on `Topology`, `Node`, `Edge` remain unconstrained.)

## Non-functional — regression and scope guards

- AC-61: The framework regression test suite passes with zero new failures vs. the pre-change baseline.
- AC-62: No changes are made to TRUNK-CRITICAL files: `coordination/topology/graph.py`, `coordination/orchestra.py`, `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`.
- AC-63: No Spren types are imported by any file under `packages/framework/`.
- AC-64: No "if running under Spren" conditional code paths are introduced under `packages/framework/`.
- AC-65: `coordination/topology/core.py` is not edited; the existing `metadata` field is reused.
- AC-66: `agents/agents.py` is not edited; the `BaseAgent` / `Agent` constructor surface is read-only for this PR.
- AC-67: `models/models.py` is not edited; `ModelConfig` is re-exported (via `models/serialize.py`), not modified.
- AC-68: The only modified non-new file in the framework source tree (besides `CHANGELOG.md`) is `coordination/topology/converters/pattern_converter.py`, which receives a localized addition writing `topology.metadata["original_pattern"]`.

## Non-functional — CHANGELOG and PR metadata

- AC-69: `packages/framework/CHANGELOG.md` contains a new entry under the `## [Unreleased]` heading describing the `WorkflowDefinition` shape, the conversion functions, the `original_pattern` metadata convention, and the JSON Schema export.

## Out of scope

- Full rule serialization. `TopologySpec.rules` is a `list[str]` of rule references only; the rule object structure itself is intentionally not serialized in this PR.
- Editing the `BaseAgent` / `Agent` constructor to accept `is_convergence_point` as a kwarg. The current PR sets it post-construction.
- Editing `ModelConfig` to remove or alter the `_validate_api_key` validator. The mirror (`ModelConfigSpec`) is introduced instead.
- Editing TRUNK-CRITICAL files or `coordination/topology/core.py`.
- The Spren-side cleanup that deletes mirror files under `packages/spren/`. That is a coordinated follow-up PR, not part of this session's deliverable for the framework.
- Typed `plan_config`, `input_schema`, `output_schema`. These are passed through as opaque `dict[str, Any] | None` per the open question (3) deferred to a follow-up PR.

## Open / needs clarification

- ~~AC-59 trigger conditions~~ **RESOLVED 2026-05-12**: `NonSerializableTopologyError` is raised when `workflow_to_pydantic` encounters a `DeterministicNode` instance. See AC-59 above for the test contract.
- PR-coordination acceptance items from the plan (open Spren cleanup PR within 24 hours of merge; multi-consumer justification in PR description; PR description references session brief) are process/PR-metadata requirements not testable from the test suite alone; they are intentionally omitted from the test-auditable acceptance list above. Tracked in the PR checklist of the session plan itself, not in this acceptance file.
