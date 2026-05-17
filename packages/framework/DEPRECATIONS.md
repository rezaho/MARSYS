# Deprecations

Canonical tracker for everything deprecated in MARSYS. Sectioned by the version
where the deprecation will be **removed**. Inside each section, list each
deprecated thing with its replacement, the support code locations (marked in
source with `# REMOVE-IN-V0.4:` etc.), and a migration recipe.

To find every support site for a target version:

```bash
grep -rn "REMOVE-IN-V0.4" src/
```

The marker count in source should track the entries in the corresponding
section here. When the version ships, removing every marker should leave the
section here removable too.

---

## v0.4 — slated for removal in next major release

### Legacy topology metadata: `entry_point` / `exit_points`

**Replacement**: explicit `Start` and `End` det-node instances in topology
nodes/edges.

**Why**: the v0.3 redesign introduced unique reserved det-nodes (`Start`,
`End`, `User`) as the canonical way to express workflow entry, termination,
and user-channel I/O. Legacy `entry_point=A` and `exit_points=[X, Y]`
metadata are translated into `Start → A` and `X → End`, `Y → End` edges
by a backward-compat shim, which becomes unnecessary once users opt in.

**Support code (delete at v0.4)**:
- `src/marsys/coordination/orchestra.py:_apply_legacy_topology_shim` —
  the entire method is the translator. Delete its call site too
  (orchestra.py wherever it's invoked).
- `src/marsys/coordination/topology/graph.py:find_exit_points_with_manual` —
  legacy auto-detection of exit points used by the analyzer.
- `src/marsys/coordination/orchestra.py:_find_entry_agents` — fallback
  when no `StartNode` is registered. Plus its call site.

**Migration recipe**:

```python
# v0.3 (deprecated)
topology = {
    "agents": ["Coordinator", "Worker1", "Worker2"],
    "flows": [
        "Coordinator -> Worker1",
        "Coordinator -> Worker2",
    ],
    "entry_point": "Coordinator",
    "exit_points": ["Coordinator"],
}

# v0.4
topology = {
    "agents": ["Start", "Coordinator", "Worker1", "Worker2", "End"],
    "flows": [
        "Start -> Coordinator",
        "Coordinator -> Worker1",
        "Coordinator -> Worker2",
        "Coordinator -> End",
    ],
}
```

---

### Legacy User pattern: `agent → User` edge as implicit terminal

**Replacement**: explicit `End` det-node + `agent → End` edge for terminal
delivery; `User` det-node remains for `ask_user` (question-with-reply).

**Why**: the v0.3 model separates two semantics that legacy User-edges
conflated: (a) *terminate workflow with final answer* — now
`terminate_workflow`, gated on `agent → End`; (b) *ask user a question
expecting a reply* — now `ask_user`, gated on `agent → User`. The shim
infers terminal intent from `agent → User` legacy edges and synthesizes
`agent → End`.

**Support code (delete at v0.4)**:
- `src/marsys/coordination/orchestra.py:_apply_legacy_topology_shim` —
  the User-edge translation block at the end of the method.
- `src/marsys/coordination/topology/graph.py:has_user_access` —
  superseded by `has_edge_to_endnode` and `has_edge_to_usernode`.
- `src/marsys/coordination/topology/graph.py:auto_inject_user_node` —
  legacy auto-injection driven by `auto_inject_user=True` metadata.
- `src/marsys/coordination/topology/analyzer.py:_add_nodes` — the
  `node.kind is not NodeKind.USER` carve-out (Session 08, ADR-008): USER
  stays a regular `NodeInfo(kind=USER)` so the shim above performs the
  legacy-User translation + `UserNode` registration. At v0.4, delete the
  carve-out so USER joins the generic kind→behaviour materialization branch
  (no dispatch restructure — the comment at that site is the exact recipe).

**Migration recipe**:

```python
# v0.3 (deprecated): "Planner → User" implicitly meant Planner can deliver final
topology = {
    "agents": ["User", "Planner", "Worker"],
    "flows": [
        "User -> Planner",
        "Planner -> Worker",
        "Worker -> Planner",
        "Planner -> User",   # implicit "Planner terminates"
    ],
}

# v0.4: explicit Start and End, plus User if interactive Q&A is needed
topology = {
    "agents": ["Start", "Planner", "Worker", "End"],
    "flows": [
        "Start -> Planner",
        "Planner -> Worker",
        "Worker -> Planner",
        "Planner -> End",
    ],
    # Add "User" det-node and `Planner -> User` if the planner needs to ask
    # questions mid-flow.
}
```

---

### Coordination tool name: `return_final_response`

**Replacement**: `terminate_workflow` (same semantics, clearer name).

**Why**: the new name reflects what the tool actually does — emit the
workflow's final answer and terminate the branch, delivering to the
workflow output channel. The legacy name was ambiguous between
"return-to-caller" and "terminate-workflow".

**Support code (delete at v0.4)**:
- `src/marsys/coordination/formats/coordination_tools.py:COORDINATION_TOOL_NAMES` —
  drop the `"return_final_response"` entry.
- `src/marsys/coordination/validation/response_validator.py:_validate_return_final_response` —
  alias dispatcher.
- `src/marsys/coordination/validation/response_validator.py:ActionType.FINAL_RESPONSE` —
  enum member kept for legacy-action emission. After removal, only
  `ActionType.TERMINATE_WORKFLOW` exists.
- 3 test cases in `tests/coordination/test_response_validator.py`
  (`test_return_final_response_with_access`,
  `test_return_final_response_without_access`,
  `test_return_final_response_structured_output`) that assert on the
  legacy name. Either delete or rewrite to use `terminate_workflow`.

**Migration recipe**:

```python
# v0.3 (deprecated): emit tool call with the legacy name
tool_calls=[ToolCall(name="return_final_response", arguments={"response": "..."})]

# v0.4
tool_calls=[ToolCall(name="terminate_workflow", arguments={"answer": "..."})]
```

---

### `auto_detect_convergence` config flag

**Replacement**: explicit convergence-point declaration in topology.

**Why**: auto-detecting convergence based on multi-incoming-edge nodes is
heuristic; the new model expects users to mark convergence explicitly via
det-nodes or explicit metadata.

**Support code (delete at v0.4)**:
- `src/marsys/coordination/config.py:ExecutionConfig.auto_detect_convergence` —
  the field itself.
- `src/marsys/coordination/topology/graph.py:mark_dynamic_convergence_points` —
  the auto-detection branch (the explicit-marking branch stays).

**Migration recipe**: no user-facing API change; auto-detection just stops
firing. If a workflow relied on it, declare convergence explicitly.

---

## How to add a new deprecation

1. Decide the removal target version (e.g., v0.5).
2. Add a `# REMOVE-IN-V0.5: <reason>` marker on every code site that
   supports the deprecated thing.
3. Add a section here under the matching version header, listing the
   thing, replacement, support sites, and migration recipe.
4. Emit a `DeprecationWarning` at runtime when the user invokes the
   deprecated path, naming the version where it'll be removed.

When the version ships:
1. `grep -rn "REMOVE-IN-V<x>" src/` to find every site.
2. Delete each one in a single PR.
3. Remove the corresponding section from this file.
