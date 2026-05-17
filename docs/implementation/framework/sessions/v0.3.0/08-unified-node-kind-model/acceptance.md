# Acceptance criteria — Framework Session 08: Unified Node-Kind Model for the Canonical Topology & Wire Format

Frozen at 2026-05-17T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Source contract: `docs/implementation/framework/sessions/v0.3.0/08-unified-node-kind-model.md` §4 (AC-1…AC-18, characterization cases (a)-(e)). Design authority: ADR-008 (context only).

**Legend.** Each criterion is tagged:
- `[new-behaviour]` — asserts behaviour that did not exist or is intentionally changed in S08.
- `[characterization]` — a baseline of *current* executed behaviour captured BEFORE any production change (no prior baseline exists).
- `[regression-parity]` — post-change executed outcome must equal the captured pre-change baseline.
- `[regression-pin]` — pre-existing behaviour that must remain unchanged; pinned so a future change cannot silently alter it.
- `[structural / diff-verifiable]` — verified by inspecting the resulting code structure or a diff against the pre-session baseline (the test asserts the structural fact; the auditor checks the test exists and asserts it).

A criterion may carry more than one tag where it has distinct halves.

---

## Functional

### Core taxonomy

- **AC-1** `topology/core.py` exposes a `NodeKind` enum whose members are *exactly* `{AGENT, START, END, USER}` — no `SYSTEM`, no `TOOL`, no other members. `[new-behaviour]`
- **AC-2** No in-repo framework module references the old `NodeType` symbol; every prior `NodeType` reference has been migrated to `NodeKind` in place. An alias is permitted ONLY if a primary-source caller forces it, and if present it is explicitly surfaced (not assumed); absent that, no alias exists. `[structural / diff-verifiable]` `[new-behaviour]`
- **AC-3** `core.User` (the `core.User(Node)` class) no longer exists. `[new-behaviour]`
- **AC-4** No framework module imports `core.User`. `[structural / diff-verifiable]`
- **AC-5** `topology/__init__` no longer exports `User`; the package export surface is updated accordingly. `[new-behaviour]`
- **AC-6** A user node is expressible solely as a `Node` with `kind=USER` (there is no separate user node class). `[new-behaviour]`
- **AC-7** `Topology.__post_init__` accepts a homogeneous list of `Node` instances; constructing a `Topology` whose nodes include `kind ∈ {START, END, USER}` succeeds without raising. `[new-behaviour]`
- **AC-8** `Topology.__post_init__` accepts `Node` only — the prior dual-accept of `(Node, DeterministicNode)` is removed; passing a `DeterministicNode` instance into `Topology` nodes is no longer the supported construction path. `[new-behaviour]`

### Single-sourced behaviour registry (extension-openness)

- **AC-9** There is exactly one authoritative `NodeKind → behaviour-class` map serving as the single source of truth for deterministic-node behaviour. `[structural / diff-verifiable]`
- **AC-10** The reserved-name string lookup (e.g. `"Start" → StartNode`) is *derived* from the behaviour classes' `RESERVED_NAME` attributes — there is no second hand-maintained `RESERVED_DETNODE_NAMES` dict, and the previously separate spellings (`RESERVED_NODE_NAMES`, `RESERVED_DETNODE_NAMES`, the `parse_node` carve-out) are collapsed to one source. `[structural / diff-verifiable]` `[new-behaviour]`
- **AC-11** Adding a new (throwaway) node kind end-to-end requires only three additions — an enum value, a behaviour class, and one registry entry — with **no** edits to any dispatch site; a test exercises a throwaway kind through this path and demonstrates it works without dispatch-site edits. `[new-behaviour]`
- **AC-12** The deterministic-node behaviour classes (`DeterministicNode`, `StartNode`, `EndNode`, `UserNode`) continue to exist as behaviour. No speculative `conditional`/`loop` (or other unused) behaviour classes are introduced. `[structural / diff-verifiable]`

### Materialization seam (load-bearing — Option A, true uniform)

- **AC-13** `parse_node` returns a uniform `core.Node(kind=...)` for *every* input. In particular a `"Start"` / `"End"` / `"User"` string input yields `core.Node(kind=START)` / `core.Node(kind=END)` / `core.Node(kind=USER)` respectively. `[new-behaviour]`
- **AC-14** `parse_node` never returns a `DeterministicNode` instance for any input (reversing the prior behaviour where strings produced det-node instances). `[new-behaviour]`
- **AC-15** `Topology.nodes` is homogeneous — every element is a `Node`; no element is a `DeterministicNode` instance, including after parsing string-notation or explicit-det-node topologies. `[new-behaviour]`
- **AC-16** For a `core.Topology` produced by `pydantic_to_topology`, after analysis the resulting `TopologyGraph` answers `get_start_node()`, `is_det_node()`, and `get_det_node()` with the correct deterministic-node instances for `START`/`END`/`USER` nodes (i.e. a deserialized `core.Node(kind=START)` becomes a registered det-node — closes the prior gap where it did not). `[new-behaviour]`
- **AC-17** The `DeterministicNode` instance for a `START`/`END`/`USER` node is materialized and `register_det_node`'d during analysis (analyzer `_add_nodes`), not at `Topology` construction or at parse time — observable via: a parsed/constructed `Topology` carries no det-node instances, but the post-analysis `TopologyGraph` det-node registry does. `[new-behaviour]`

### Wire format (NodeSpec / serialize)

- **AC-18** `NodeSpec` has a `kind` field typed as the closed `NodeKind`; `NodeSpec` has no `node_type` field. `[new-behaviour]`
- **AC-19** `NodeSpec` rejects unknown/extra fields (`extra="forbid"` behaviour holds): a `NodeSpec` payload with an unexpected field fails validation. `[regression-pin]`
- **AC-20** `workflow_to_pydantic` serializes a topology containing `START`, `END`, and `USER` nodes without raising (the serialization is total over all `NodeKind`). `[new-behaviour]`
- **AC-21** `NonSerializableTopologyError` no longer exists as a symbol and is unreferenced repo-wide. `[structural / diff-verifiable]` `[new-behaviour]`
- **AC-22** `pydantic_to_topology` accepts the signature `pydantic_to_topology(spec, tool_registry, handler_registry)`; round-tripping a workflow through `workflow_to_pydantic` then `pydantic_to_topology` succeeds for every `NodeKind`. `[new-behaviour]`
- **AC-23** A `USER` node's handler is resolved from the injected `handler_registry` (the same dependency-injection pattern tools use via `tool_registry`). `[new-behaviour]`
- **AC-24** When the handler required by a `USER` node is missing from `handler_registry`, `pydantic_to_topology` fails with a clear, specifically-named error mirroring `UnknownToolError` — it does NOT silently bind `None`. `[new-behaviour]`

### Equality oracle (`topology_equals`)

- **AC-25** `topology_equals`'s `node_key` keys on `kind` (e.g. `kind.value`), not on `node_type.value`. `[new-behaviour]`
- **AC-26** `topology_equals`'s `node_key` includes `is_convergence_point`; two topologies differing only in a node's `is_convergence_point` value compare as NOT equal under `topology_equals`. `[new-behaviour]`

### Round-trip matrix & property

- **AC-27** An exhaustive round-trip matrix holds: for every `NodeKind` × every `EdgeType` × every link mode in `{none, bidirectional, alternating, symmetric}`, `pydantic_to_topology(workflow_to_pydantic(t))` is equal to `t` under the updated `topology_equals`. `[new-behaviour]`
- **AC-28** The round-trip matrix includes pattern-provenance: a topology carrying `metadata["original_pattern"]` round-trips with that provenance preserved (equal under `topology_equals`). `[new-behaviour]`
- **AC-29** The round-trip matrix sets `is_convergence_point=True` on at least one node and full round-trip equality (under the `is_convergence_point`-aware oracle) still holds for that case — i.e. `is_convergence_point` round-trips correctly across the matrix. `[new-behaviour]`
- **AC-30** Any convergence round-trip discrepancy that the `is_convergence_point` oracle change exposes — whether in S08's own changed path OR pre-existing/unrelated — is fixed within S08. (User-directed scope inclusion, 2026-05-17; recorded as deliberate, not creep — anti-pattern #8 exception is explicit user instruction; tradeoff: coupled rollback + larger diff.) **Auditable proxy:** AC-29 (full `is_convergence_point` round-trip correctness across the exhaustive matrix) is the enforceable observable; additionally, each *specific* pre-existing discrepancy found is pinned by its own dedicated, named regression test enumerated in the plan §12 sign-off. The auditor verifies AC-29 + each named sign-off regression test — NOT an unbounded set. `[new-behaviour]`
- **AC-31** A Hypothesis property test asserts `pydantic_to_topology(workflow_to_pydantic(t)) ≈ t` (equal under `topology_equals`) for generated topologies that include deterministic-node kinds. `[new-behaviour]`
- **AC-32** The Hypothesis node-name generation strategy is updated for the new reserved-name set (so generated names do not collide with the new reserved names and do collide-avoid the changed membership). `[new-behaviour]`

### Start invariant (complete)

- **AC-33** `[amended 2026-05-17 — Session 08; clean option (3rd characterization refutation; user-approved). Supersedes the Option-1 wording.]` "Exactly one Start" is **already fully enforced** for the explicit/canonical path and Session 08 adds **no new enforcement code** (a post-shim guard would be defensive code for an impossible state — anti-pattern #5). Regression-pin: a no-entry/cycle topology raises `TopologyError` at analyze (`graph.py:516`); a no-terminal topology raises `TopologyError` at analyze under strict/non-auto_run (`graph.py:644`). These existing errors must remain after the node-kind migration. `[regression-pin]`
- **AC-34** A topology with two or more Start nodes raises `TopologyError`. `[regression-pin]`
- **AC-35** `[amended 2026-05-17 — Session 08]` A topology with exactly one Start node runs successfully. Single-Start is already auto-synthesized by the analyzer→shim chain for *every* topology with ≥1 terminal node (not "the now-total shim" — the shim was already total; corrected P6). `[regression-pin]` (parity; see AC-38–AC-42)

### Characterization-first parity (AC-12 cases (a)-(f))

- **AC-36** `[amended 2026-05-17 — Session 08; added case (f), user-approved Option 1]` Characterization tests capturing the *current executed behaviour* are written BEFORE any production change (no pre-existing baseline; the captured outcomes are the parity reference). Six cases: (a) pure-agent topology with no Start; (b) `entry_point` metadata; (c) `exit_points` metadata; (d) `User(Node)`-terminal; (e) string-notation `"Start -> … -> End"` explicit-det-node; **(f) degenerate no-terminal topology** (pure cycle / isolated node — the genuine zero-Start case). `[characterization]`
- **AC-37** Case (e) is characterized specifically as: a string-notation `"Start -> … -> End"` topology where `"Start"`/`"End"` strings become `core.Node(kind=START/END)` materialized at the analyzer (rather than `StartNode`/`EndNode` instances placed into `Topology.nodes`) — the case `parse_node`'s Option-A change most directly affects. `[characterization]`
- **AC-38** Case (b) `entry_point`-metadata topology: post-change executed outcome equals the captured pre-change baseline. `[regression-parity]`
- **AC-39** Case (c) `exit_points`-metadata topology: post-change executed outcome equals the captured pre-change baseline. `[regression-parity]`
- **AC-40** Case (d) `User(Node)`-terminal topology: post-change executed outcome equals the captured pre-change baseline. `[regression-parity]`
- **AC-41** Case (e) string-notation explicit-det-node topology: post-change executed outcome equals the captured pre-change baseline. `[regression-parity]`
- **AC-42** `[amended 2026-05-17 — Session 08; clean option]` Case (a) pure-agent topology: **parity** — already gets a synthesized Start+End today; post-change executed outcome equals the captured pre-change baseline, exactly like (b)-(e). `[regression-parity]`
- **AC-42b** `[amended 2026-05-17 — Session 08; clean option supersedes Option 1]` Cases (f1)/(f2) degenerate topologies (no-entry cycle; has-entry-no-terminal): **regression-pin, NOT new behaviour.** Both already raise an explicit `TopologyError` at analyze today (`graph.py:516` / `graph.py:644`); the f1/f2 characterization tests pin that this existing enforcement is not regressed by the node-kind migration. No Start-area behaviour change in Session 08. `[regression-pin]`

### pattern_converter parity

- **AC-43** Every topology built by `pattern_converter` (all 7 construction sites) produces user nodes as `kind=USER` (no `NodeType.USER` constructor remains in `pattern_converter`). `[new-behaviour]` `[structural / diff-verifiable]`
- **AC-44** `pattern_converter`-built `kind=USER` nodes still traverse the retained legacy-User shim with an executed outcome unchanged from the captured pre-change baseline; pattern semantics and edges are unchanged. `[regression-parity]`

### Backward-compat load policy

- **AC-45** A legacy stored `WorkflowDefinition` JSON with no explicit Start node still deserializes successfully and emits a `DeprecationWarning`. `[new-behaviour]` (permissive v0.3 wire validator; parity of user-visible run outcome covered by AC-42/AC-36)
- **AC-46** A stored document whose node `kind`/value is `system` or `tool` is rejected at load with an error containing a migration message (it is NOT silently coerced to `agent`). `[new-behaviour]`

### Docs / schema / contract artifacts

- **AC-47** `workflow_definition_schema()` returns a JSON Schema using dialect 2020-12 that reflects the `kind` field (not `node_type`). `[new-behaviour]`
- **AC-48** A CHANGELOG entry for this change is present. `[structural / diff-verifiable]`
- **AC-49** The wire schema version is bumped. `[structural / diff-verifiable]` `[new-behaviour]`
- **AC-50** `DEPRECATIONS.md` is updated (heuristic-entry superseded by the total shim; destination model live). `[structural / diff-verifiable]`
- **AC-51** The Session-04 `acceptance.md` carries an amendment recording that AC-59 is reversed by ADR-008/Session-08, dated and Session-08-attributed. `[structural / diff-verifiable]`

### Orchestrator-untouched guarantee (P4 falsifier)

- **AC-52** Diff-verified against the pre-session baseline: `execution/orchestrator.py`'s dispatch body is unmodified. `[structural / diff-verifiable]` `[regression-pin]`
- **AC-53** Diff-verified against the pre-session baseline: the `TopologyLike` Protocol signature (`orchestrator_types.py`) is unmodified. `[structural / diff-verifiable]` `[regression-pin]`

### Surfaced cross-package / registration behaviour change

- **AC-54** The agent-registration legality change is pinned by an explicit test: an agent (or pool) named `system` or `tool` is now *allowed* to register; an agent named `start` or `end` is now *forbidden* — making the `RESERVED_NODE_NAMES` membership change intentional and visible, not incidental. `[new-behaviour]`
- **AC-55** `RESERVED_NODE_NAMES` remains a module-level frozenset with the same name and import path as before; its *value* is derived from `NodeKind` (the non-AGENT kinds, lowercased). Importing `RESERVED_NODE_NAMES` does not raise (the symbol/path/type is preserved; only membership changes). `[structural / diff-verifiable]` `[new-behaviour]`

### AC-59 inversion (explicit spec reversal — anti-pattern #1 surfaced)

- **AC-56** The Session-04 tests that previously asserted `pytest.raises(NonSerializableTopologyError)` for det-node serialization are inverted to assert a *successful* round-trip. The reversal is explicit and surfaced (not a silent test deletion); no skip/xfail is used to neutralize them. `[new-behaviour]` `[structural / diff-verifiable]`

### Regression suite

- **AC-57** The framework regression suite is green at session tip: no new test failures and no new skips relative to the pre-session baseline. `[regression-pin]`
- **AC-58** The Spren import of `RESERVED_NODE_NAMES` does not break (it remains a same-named, same-path frozenset). Any divergence in Spren's *own* suite is confined to assertions of the old reserved-value semantics or reliance on Spren's stale local `NodeType`/mirror — this divergence is expected and contracted (P9), and is NOT to be "fixed" by reverting the framework change. `[structural / diff-verifiable]` (Spren-suite divergence itself is out of scope to fix here — see Out of scope.)

---

## Out of scope

Tests asserting behaviour for any of the following would be wrong for this session:

- `agent_type` round-trip / specialized-`Agent`-subclass serialization (Session 09).
- v0.4 shim / heuristic-entry-path deletion (the heuristic branch and `_find_entry_agents` remain in place, unreached, `REMOVE-IN-V0.4`).
- Spren-side mirror-drop and stored-data migration (Spren-side follow-up PR, P9). Spren-suite divergence caused by the contracted `RESERVED_NODE_NAMES` value change is expected, not a defect to fix here.
- Edge / pattern *semantics* changes (only the node constructor in `pattern_converter` migrates; edges and pattern behaviour are unchanged).
- New deterministic-node behaviour classes (e.g. `conditional`/`loop`) — none are introduced.
- Modifying `execution/orchestrator.py` dispatch body, the `TopologyLike` Protocol signature, the `entry_agent` run signature, or `coordination/serialize.py` (ExecutionConfig).
- Any Spren file modification.
- A dual-field wire backward-compat shim (e.g. accepting both `node_type` and `kind`) — explicitly rejected; the version boundary is the contract.

---

## Open / needs clarification

- **AC-30 / AC-58 verifiability note (no weakening intended):** AC-30 ("any convergence round-trip discrepancy … pre-existing or unrelated … is fixed in S08") is observable as written via AC-29's positive assertion (full `is_convergence_point` round-trip correctness across the matrix), so it is testable. But its scope ("*any* discrepancy the oracle exposes, including pre-existing/unrelated") is open-ended — the test auditor cannot enumerate, from this file plus tests alone, every discrepancy that *should* have been caught. If the implementer's sign-off (§12 of the plan) records a specific pre-existing convergence discrepancy that was found and fixed, that specific case should also have a dedicated regression test; absent such a sign-off entry, AC-29's matrix coverage is the contracted observable. Flagged so the auditor treats AC-29 as the enforceable proxy for AC-30 rather than searching for an unbounded set. `(RESOLVED 2026-05-17 — AC-29 is the enforceable proxy; each specific pre-existing discrepancy found is pinned by a dedicated, named regression test enumerated in plan §12 sign-off; the auditor checks AC-29 + those named tests, not an unbounded set. User-directed fix-in-S08 scope unchanged.)`
