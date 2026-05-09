# ADR-006: Deprecation Timeline (v0.2.x → v0.4)

**Status**: Accepted
**Date**: 2026-05-02

## Context

Phase 3 cutover (commit `bc19b98`) replaced `BranchSpawner` and `BranchExecutor` with `Orchestrator` + `RealRuntime`. Phase 3.5 (commits `2689a39` … `402b223`) then added topology-gated coordination tools, det-node singletons (`StartNode`/`EndNode`/`UserNode`), the `AgentInput` abstraction at the orchestrator-agent boundary, retry-tiered steering, a content-only hard limit, and a legacy migration shim.

These changes renamed several v0.2.x APIs and removed others. Some legacy symbols are kept as deprecated aliases for one release so existing user code continues to work; the rest emit `DeprecationWarning` via runtime auto-translation. This ADR codifies the deprecation timeline and provides the canonical migration table that documentation banners across the project link to.

## Decision

Maintain a single migration table (below) as the source of truth for v0.2.x → v0.3.0 renames. Every deprecated API emits `DeprecationWarning` at runtime; the auto-translation shim in `Orchestra._apply_legacy_topology_shim` (`src/marsys/coordination/orchestra.py:296`) handles topology-level legacy forms idempotently. **Removal target: MARSYS v0.4.**

Documentation banners on every page that mentions a legacy term link back to the migration table fragment in this ADR. When the parallel code track removes an alias, the corresponding row's status updates to "removed in v0.3.0" and banner text adjusts.

## Migration table {#migration-table}

| Legacy term (≤ v0.2) | Canonical term (v0.3.0+) | Status | Removal target |
|---|---|---|---|
| `BranchSpawner` / `DynamicBranchSpawner` | `Orchestrator` (event loop) + `RealRuntime` (per-branch driver) | Deleted in commit `bc19b98` | already gone |
| `BranchExecutor` (large class) | `RealRuntime.step()` driving `StepExecutor` | Deleted in commit `bc19b98` | already gone |
| `return_final_response` (tool name) | `terminate_workflow` | Alias kept for one release (still in `COORDINATION_TOOL_NAMES` line 30) | v0.4 |
| `can_return_final_response` (`CoordinationContext` field) | `can_terminate_workflow` | **Removed in v0.3.0** (commit `82ff393`, step-7 cleanup); shim was @property aliasing `can_terminate_workflow` | already gone |
| `entry_point` / `exit_points` (topology metadata) | explicit `Start → A` / `X → End` det-node edges | Auto-shim emits `DeprecationWarning` | v0.4 |
| `User(Node)` (regular user node) | `UserNode` (det-node singleton) | Auto-translated by shim | v0.4 |
| `has_user_access(agent)` | `has_edge_to_endnode(agent)` / `has_edge_to_usernode(agent)` | Still exists at `graph.py:804`; kept while validators transition | v0.4 |
| JSON `{"next_action": "...", "action_input": "..."}` response format | native tool calls (`invoke_agent` / `terminate_workflow` / `ask_user` / `end_conversation`) | Removed (no shim) | already gone |
| `final_response` action (string) | `terminate_workflow` tool | `ActionType.FINAL_RESPONSE` retained as enum alias (`response_validator.py:32`) | v0.4 |
| `parallel_invoke` action (single string) | repeated `invoke_agent` tool calls in same model turn (concurrent dispatch by Orchestrator) | Removed | already gone |
| `wait_and_aggregate` action | implicit (Orchestrator barrier) | Removed | already gone |
| `coordination_action` / `coordination_data` (dict-key dispatch on string action) | `ActionType` enum + `parse_coordination_call` for tool calls | Internal; doc-only relevance | n/a |

## Per-deprecation entries

### `entry_point` / `exit_points` topology metadata

- **Trigger**: any `Topology.metadata['entry_point']` or `['exit_points']` non-empty at `Orchestra.run`.
- **Shim**: `src/marsys/coordination/orchestra.py:296` (`_apply_legacy_topology_shim`). Synthesizes `StartNode` + edge `Start → A` for `entry_point=A`; synthesizes `EndNode` + edges `X → End`, `Y → End` for `exit_points=[X, Y]`. Idempotent.
- **Migration**: replace metadata with explicit det-node edges:

```python
# Before (v0.2.x — emits DeprecationWarning, removed in v0.4):
topology = {
    "agents": ["Coordinator", "Researcher"],
    "flows": ["Coordinator -> Researcher", "Researcher -> Coordinator"],
    "entry_point": "Coordinator",
    "exit_points": ["Coordinator"],
}

# After (canonical):
from marsys.coordination.execution.det_nodes import StartNode, EndNode
from marsys.coordination.topology import Topology, Node, Edge

topology = Topology(
    nodes=[StartNode(), Node("Coordinator"), Node("Researcher"), EndNode()],
    edges=[
        Edge("Start", "Coordinator"),
        Edge("Coordinator", "Researcher"),
        Edge("Researcher", "Coordinator"),
        Edge("Coordinator", "End"),
    ],
)
```

### `return_final_response` tool name

- **Trigger**: an agent emits a tool call named `return_final_response`.
- **Alias**: `src/marsys/coordination/formats/coordination_tools.py:COORDINATION_TOOL_NAMES` (line 25, alias entry at line 30) keeps the name; `_validate_return_final_response` routes to `_validate_terminate_workflow` and emits `ActionType.FINAL_RESPONSE` for back-compat.
- **Migration**: rename the agent's instruction wording from `final_response` / `return_final_response` to `terminate_workflow`. No topology change required.

### `User(Node)` regular user node

- **Trigger**: topology contains a node with `node_type == NodeType.USER` and no `UserNode` det-node.
- **Shim**: same `_apply_legacy_topology_shim` registers a `UserNode` det-node alongside.
- **Migration**: use `UserNode()` directly in topology nodes:

```python
from marsys.coordination.execution.det_nodes import UserNode

topology = Topology(
    nodes=[StartNode(), Node("Assistant"), UserNode(), EndNode()],
    edges=[
        Edge("Start", "Assistant"),
        Edge("Assistant", "User"),
        Edge("User", "Assistant"),
        Edge("Assistant", "End"),
    ],
)
```

### `has_user_access(agent)`

- **Trigger**: internal — called by validators and tool-gating.
- **Replacement**: `has_edge_to_endnode(agent)` (`graph.py:846`) checks the `terminate_workflow` gate; `has_edge_to_usernode(agent)` (`graph.py:862`) checks the `ask_user` gate.
- **Current state**: `has_user_access` still exists at `src/marsys/coordination/topology/graph.py:804` alongside the new methods. It will be removed once internal validators are fully migrated.
- **Migration**: internal — most user code does not call this method directly.

### `can_return_final_response` field on `CoordinationContext` (REMOVED in v0.3.0)

- **Trigger**: previously, anyone constructing or reading the field.
- **Status**: the `@property` shim was removed in commit `82ff393` (step-7 cleanup) along with the constructor kwarg shim. There is no longer a fallback. Code that constructs `CoordinationContext` with `can_return_final_response=...` or reads `.can_return_final_response` will raise `TypeError` / `AttributeError`.
- **Migration**: rename to `can_terminate_workflow`. `BaseAgent.can_return_final_response` (the agent-level property) was migrated in the same commit to read `has_edge_to_endnode` internally.

### Removed (no shim) — JSON `next_action` response format

- **Removed in commit `bc19b98`**.
- The legacy format expected agents to emit JSON like `{"next_action": "invoke_agent", "action_input": "Researcher"}`. MARSYS now uses native tool calls; coordination is driven by tool names that match `COORDINATION_TOOL_NAMES`.
- **Migration**: agents emit native tool calls produced by the model's tool-use API. No user code change for agents that already use the default response format. If any custom code constructed `next_action` dicts, replace with `parse_coordination_call(tool_call)` flow.

### Retained as enum alias — `ActionType.FINAL_RESPONSE`

- The enum member `ActionType.FINAL_RESPONSE` (`response_validator.py:32`) is retained as an alias for `ActionType.TERMINATE_WORKFLOW` (line 33) so legacy `return_final_response` tool calls validate cleanly. Removal aligned with v0.4.

## Forward-compatibility rules

- **v0.3.0** (this release): warnings emitted, all aliases functional. New code should use canonical names exclusively.
- **v0.4**: warnings escalate to errors; aliases removed.
- **v0.5+**: cleanup (no longer in changelog).

## Consequences

- Positive:
  - Single source of truth for migration mapping; doc banners point to one anchor.
  - One-release alias window gives downstream users time to migrate without breaking existing scripts.
  - Auto-shim makes topology migration risk-free for examples and notebooks.

- Negative:
  - Two parallel code paths during the v0.3.0 → v0.4 window (canonical + legacy alias).
  - `DeprecationWarning` noise for users on legacy code; mitigated by clear migration text in this ADR.

- Risk:
  - If the parallel code track removes an alias before this ADR is updated, banner links may go stale. Mitigation: keep the migration table the authoritative source and update it with each removal.

## References

- Commit `bc19b98` — phase 3 cutover (deleted `BranchSpawner` / `BranchExecutor`).
- Commits `2689a39`, `e83a6ca`, `662ae5c`, `3a17667`, `402b223` — phase 3.5.
- ADR-001 — original branch-based parallel execution (decision unchanged; implementation superseded).
- ADR-005 — unified-barrier algorithm (companion to this ADR).
