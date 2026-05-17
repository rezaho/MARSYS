# Framework Session 08: Unified Node-Kind Model for the Canonical Topology & Wire Format

**Status**: PLAN — awaiting ADR-008 approval (escalation gate) + Phase-A acceptance freeze
**Role**: implementer (executes after the gate; this doc is the contract)
**Branch**: feature/tracing-streaming
**Created**: 2026-05-16 · **Revised**: 2026-05-16 (Phase-A validation refuted v1's mechanism — see ADR-008 §Revision note)
**Decided by**: [ADR-008](../../../../architecture/framework/decisions/ADR-008-unified-node-kind-model.md) — read it whole first; it is the authoritative design, premise ledger, alternatives. This plan executes ADR-008 Decisions 1-8.

---

## 1. The big picture

The Session-04 serializer mirrored the *legacy enum-only* node model and hard-rejects every `DeterministicNode`, while the v0.3 redesign had already made explicit `Start`/`End`/`User` det-nodes **the** canonical model (`DEPRECATIONS.md:24-31`). This session makes a single closed `NodeKind` discriminator the canonical taxonomy — owned in `topology/core.py`, mirrored 1:1 on the wire, deterministic behaviour materialized from `kind` at the **analyzer/parsing seam** via one extension-open registry. Outcome: lossless round-trip of every runnable workflow (incl. explicit entry/exit/human nodes), survives the v0.4 shim removal, Start invariant complete now.

**Mechanism (corrected — `kind` threads three layers):** wire `NodeSpec.kind` ↔ canonical `core.Node.kind` ↔ execution `TopologyGraph.det_nodes` (instance registry). The real work is the **materialization seam**: a `core.Node(kind∈{START,END,USER})` or `"Start"/"End"/"User"` string → `DeterministicNode` instance via the registry → `register_det_node`. The `Orchestrator` body + `TopologyLike` signature are unmodified (they keep reading `det_nodes`); we change *what feeds* it. This is not a one-line branch swap — the analyzer/parsing/graph materialization is the substance.

### Multi-consumer (mandatory)

Only live in-repo model consumer: Spren (hand-mirror) + the framework's own `agents/serialize.py`. Non-Python consumers aspirational (`v0.4-spren-support.md:43-48`). Spren mirror-drop + stored-data migration = Spren-side follow-up (`v0.4-spren-support.md` S04 row) — **out of scope here**. `RESERVED_NODE_NAMES` is imported by Spren (`spren/models/topology.py:13`); the import does **not** break (derived → same name/path/type) — only its *value* changes, so Spren's name-validator semantics diverge (and Spren's docstring @3-6 says it imports the set to *track the framework*, so this aligns with its stated intent). Surfaced (ADR-008 P10), not silent; contracted P9.

## 2. What came before

- **Session 04** built the serializer; AC-59 froze the det-node rejection. ADR-008 §Context shows from primary source why that was backwards.
- **Session 07** exercised the canonical path (ModelConfig/adapter chain); never touched the node model.
- **ADR-006** scheduled `entry_point`/`exit_points`/`User(Node)`/`return_final_response` for v0.4 removal. This session builds the *destination* model; it does **not** perform the removals.

## 3. What this session ships (ADR-008 Decisions 1-8)

1. `NodeKind = {AGENT, START, END, USER}` in `core.py`; drop `SYSTEM`/`TOOL`; remove `User(Node)`; `Topology` holds uniform `Node`s; `RESERVED_NODE_NAMES` **derived from `NodeKind`** (same name/import path — Spren imports the value), behaviour + cross-package change surfaced.
2. One authoritative extension-open `NodeKind → behaviour-class` registry; det-node classes stay; no speculative `conditional`/`loop` classes.
3. **Materialization seam (Option A — true uniform)**: `parse_node` returns a uniform `core.Node(kind=...)` for *every* input (no longer returns `DeterministicNode` instances); `Topology.__post_init__` accepts `Node` only (drop the `core.py:171` dual-accept). The `DeterministicNode` *instance* is materialized **only** at `analyzer._add_nodes` (branch on `node.kind` → registry → instance → `register_det_node`, retained — `analyzer.py:185` `isinstance` becomes a `kind` check at the *same site*, not deleted). `isinstance(DeterministicNode)` readers of `Topology.nodes` migrate to `.kind` (bounded: `parsing.py:56`, `serialize.py:279`-being-deleted, `core.py:171`, `analyzer.py:185`). Single chokepoint @`orchestra.py:936-955`. Closes the deserialized-`core.Node(kind=START)`-doesn't-register gap (P4b). (Option B — keep instances in `Topology.nodes` — is the rejected bounded-depth; would make "uniform" false.)
4. `NodeSpec.node_type`→`kind`; total `workflow_to_pydantic`; delete `NonSerializableTopologyError`; `pydantic_to_topology` `handler_registry` param; `topology_equals` oracle `node_type.value`→`kind.value` **AND add `is_convergence_point` to `node_key`** (wire round-trips it; oracle blind to it today).
5. `pattern_converter` 7 `Node(NodeType.USER)` sites → `kind=USER` (pattern semantics/edges unchanged).
6. **(final 2026-05-17 — clean option)** NO production code. "Exactly one Start" is already fully enforced for the explicit path (analyze hard-errors @graph.py:516/644; multi-Start @graph.py:1069). A guard would be anti-pattern #5. Step 6 = regression-pins only (f1/f2 in `test_s08_characterization.py`). `auto_run` soft-exit path out of scope.
7. Wire validator permissive in v0.3 (legacy-no-Start deserialises + `DeprecationWarning`); stored `system`/`tool` value → reject-at-load with migration message; CHANGELOG/schema-version/DEPRECATIONS.md.
8. AC-59 tests inverted to assert successful round-trip (explicit intentional spec reversal, surfaced; anti-pattern #1).

**Out of scope** (ADR-008 §Scope): `agent_type` round-trip (Session 09); v0.4 shim/heuristic deletion; Spren-side migration; edge/pattern semantics; new det-node classes.

## 4. Acceptance criteria (extracted + frozen in Phase A6 before code)

- **AC-1** `core.py` exposes `NodeKind` = exactly `{AGENT,START,END,USER}`; no `SYSTEM`/`TOOL`; every in-repo framework `NodeType` reference migrated in place (no alias unless a primary-source caller forces it — surfaced, not assumed).
- **AC-2** `core.User(Node)` removed; no framework module imports it; `topology/__init__` export surface updated; a user node is solely `kind=USER`.
- **AC-3** `Topology.__post_init__` accepts a homogeneous `Node` list; a topology with `START`/`END`/`USER` nodes constructs successfully.
- **AC-4** One authoritative `NodeKind→behaviour-class` map is the single source; the reserved-name string lookup is *derived* from the classes' `RESERVED_NAME` attrs (no second hand-maintained dict; `RESERVED_NODE_NAMES`/`RESERVED_DETNODE_NAMES`/`parse_node` carve-out collapsed to one). A test registers a throwaway kind end-to-end (enum value + class + one registry entry) with **no** edits to dispatch sites — proving single-source extension-openness.
- **AC-5 (materialization, load-bearing)** `parse_node` returns a uniform `core.Node(kind=...)` (a `"Start"/"End"/"User"` string → `core.Node(kind=START/END/USER)`; never a `DeterministicNode`); `Topology.nodes` is homogeneous `Node`; `Topology.__post_init__` accepts `Node` only. The `DeterministicNode` instance is built **only** at `analyzer._add_nodes` (kind→registry→instance→`register_det_node`). A `core.Topology` as produced by `pydantic_to_topology` analyzes such that `TopologyGraph.get_start_node()`/`is_det_node()`/`get_det_node()` return the correct instances. (Closes P4b.)
- **AC-6** `NodeSpec` has `kind` (closed `NodeKind`); `node_type` gone; `extra="forbid"` holds.
- **AC-7** `workflow_to_pydantic` serializes a topology with `START`/`END`/`USER` without raising; `NonSerializableTopologyError` no longer exists and is unreferenced repo-wide.
- **AC-8** `pydantic_to_topology(spec, tool_registry, handler_registry)` round-trips every `NodeKind`; a `USER` node's handler resolves from the injected registry exactly as tools do; a missing handler fails with a clear named error mirroring `UnknownToolError` (no silent `None`).
- **AC-9** `topology_equals` `node_key` keys on `kind` (not `node_type.value`) **and includes `is_convergence_point`**; the exhaustive matrix sets `is_convergence_point=True` on at least one node and full round-trip equality holds: every `NodeKind` × every `EdgeType` × `{none,bidirectional,alternating,symmetric}` + pattern-provenance (`metadata["original_pattern"]`). Any convergence round-trip discrepancy the oracle exposes (in-path OR pre-existing/unrelated) = **fix in S08** (user-directed 2026-05-17; deliberate scope inclusion recorded as intentional — anti-pattern #8 exception is explicit user instruction; tradeoff: coupled rollback + larger diff). Decided at acceptance-freeze.
- **AC-10** Hypothesis property: `pydantic_to_topology(workflow_to_pydantic(t)) ≈ t` for generated topologies including det-node kinds; the Hypothesis node-name strategy (`strategies.py:22`) updated for the new reserved set.
- **AC-11 (final 2026-05-17 — clean option, regression-pin only)** "Exactly one Start" is already fully enforced for the explicit path; Session 08 adds NO new enforcement code. Regression-pin (f1/f2) that the existing analyze-layer `TopologyError`s (no-entry @graph.py:516, no-terminal @graph.py:644) and the multi-Start error (@graph.py:1069) remain — so the node-kind migration cannot silently regress them.
- **AC-12 (characterization-first; reframed 2026-05-17)** Before any production change, characterization tests capture current executed behaviour for: (a) pure-agent no-Start, (b) `entry_point` metadata, (c) `exit_points` metadata, (d) `User(Node)`-terminal, (e) string-notation `"Start -> … -> End"` explicit-det-node topology, **(f) degenerate no-terminal topology** (pure cycle / isolated node — the genuine zero-Start case). After the change, **(a)(b)(c)(d)(e) executed-outcome parity holds** (the corrected P6: (a) already gets a synthesised Start today, so it is parity like the rest — NOT a behaviour change); **(f)** changes from the current silent heuristic/vague-error to the explicit post-shim `TopologyError` (documented, surfaced — the only Start-area behaviour change; the input is already malformed/no-exit).
- **AC-13** `pattern_converter`-built topologies (all 7 sites) produce `kind=USER` nodes that still traverse the retained legacy-User shim with unchanged executed outcome.
- **AC-14** Legacy stored `WorkflowDefinition` JSON without explicit Start deserialises + `DeprecationWarning`; stored doc with `kind`/value `system`/`tool` rejected-at-load with a migration message.
- **AC-15** `workflow_definition_schema()` returns dialect-2020-12 reflecting `kind`; CHANGELOG + schema-version + DEPRECATIONS.md + S04 acceptance amendment present.
- **AC-16** Diff-verified: `execution/orchestrator.py` dispatch body and the `TopologyLike` Protocol signature are unmodified (P4 falsifier).
- **AC-17** `RESERVED_NODE_NAMES` change surfaced: a test pins the new agent-registration legality (`system`/`tool` now allowed; `start`/`end` now forbidden) so the behaviour change is intentional and visible, not incidental.
- **AC-18** Framework regression suite green at session tip (no new failures; no new skips). Spren import does NOT break (derived frozenset, same name/path); Spren's *suite* diverges only where it asserts old reserved-value semantics or relies on its stale local `NodeType`/mirror — expected, contracted P9, surfaced, not "fixed" by reverting. Green-bar target scoped in Step 0.

## 5. Premise Ledger & frame check

Authoritative ledger in ADR-008 §Premise Ledger (P1-P13, all CONFIRMED@cite) + §Frame check. Phase-A validator MUST re-confirm before code:

- **P4** (revised): `execution/orchestrator.py` dispatch + `TopologyLike` signature unmodified. Falsifier: any required edit there. Refute → STOP/escalate (different risk tier).
- **P4b** (new, load-bearing): the analyzer/parsing materialization path is the real work; a deserialized `core.Node(kind=START)` does not become a det-node today (AC-5 closes this). Falsifier: materialization is already kind-driven (it is not — `analyzer.py:185`, `parsing.py:62`).

RISK-LOG: the node-kind migration must not change executed outcomes for (a)(b)(c)(d)(e) (AC-12 parity guards — (a) is parity, corrected P6); only (f) degenerate-no-terminal changes (silent heuristic → explicit `TopologyError`), documented + characterized pre/post, not papered over.

## 6. Background reading (before Step 0)

ADR-008 (whole); `core.py`; `det_nodes.py`; `serialize.py`; `exceptions.py`; `converters/parsing.py`; `converters/pattern_converter.py`; `analyzer.py:160-200`; `graph.py:38-48,1042-1075`; `orchestra.py:452-616,1005-1055,1189-1212`; `orchestrator.py:140-235`; `orchestrator_types.py:185-205`; `agents/registry.py:55-70,250-265`; `agents/agents.py:238-248`; `tests/coordination/topology/test_serialize.py:600-650`; `tests/coordination/topology/strategies.py`; `DEPRECATIONS.md`; S04 acceptance (AC-59).

## 7. Detailed plan

- **Step 0 — Baseline + P4/P4b re-confirm + characterization tests.** [DONE 2026-05-17] Diff baseline `e9d6003`. Characterization tests for (a)-(e) written + green on baseline; **(a) refuted P6** (pure-agent already gets synthesised Start/End — see corrected P6); add (f) degenerate-no-terminal characterization. Validator P4/P4b re-confirm + the `NodeType`/`DeterministicNode`/`node_type` site enumeration carry into Step 1.
- **Step 1 — Core taxonomy.** `NodeKind`; `Node.kind`; remove `User(Node)`; reconcile `RESERVED_NODE_NAMES` + pin the registration behaviour change (AC-17); `Topology` homogeneous. Migrate every framework `NodeType` ref in place.
- **Step 2 — Behaviour registry.** Generalise `RESERVED_DETNODE_NAMES` → authoritative `NodeKind→class` map; det-node classes unchanged.
- **Step 3 — Materialization seam (Option A).** `parse_node` → uniform `core.Node(kind=...)` only; `Topology.__post_init__` Node-only. `analyzer._add_nodes` branches on `node.kind` → registry → instance → `register_det_node` (the `analyzer.py:185` isinstance becomes a kind check at the same site, not removed). Migrate the bounded `isinstance(DeterministicNode)`-on-`Topology.nodes` readers to `.kind`. AC-5 the load-bearing test.
- **Step 4 — Wire mirror.** `NodeSpec.kind`; total `workflow_to_pydantic`; delete `NonSerializableTopologyError`; `pydantic_to_topology` handler-registry; `topology_equals` oracle → `kind`; permissive legacy deserialize + warning; reject vestigial values.
- **Step 5 — pattern_converter.** 7 sites → `kind=USER`; AC-13 parity.
- **Step 6 — Start invariant (final: regression-pin only, no code).** Already enforced (analyze @graph.py:516/644 + multi-Start @graph.py:1069). f1/f2 characterization tests pin it; Step 6 adds no production code.
- **Step 7 — Tests.** Matrix (AC-9), Hypothesis + strategy update (AC-10), Start invariant (AC-11), characterization parity (AC-12), pattern parity (AC-13), load policy (AC-14), schema (AC-15), orchestrator-untouched diff (AC-16), reserved-name behaviour (AC-17), full regression (AC-18). **AC-59 tests inverted, surfaced.** Failures → root-cause protocol, never YOLO.
- **Step 8 — Docs.** CHANGELOG, schema-version, DEPRECATIONS.md (heuristic-entry superseded by total shim; destination model live), S04 acceptance amendment (AC-59 reversed by ADR-008, dated/Session-08-attributed).

## 8. Files

- **Modify**: `topology/core.py`, `topology/serialize.py`, `topology/exceptions.py`, `topology/converters/parsing.py`, `topology/converters/pattern_converter.py`, `topology/analyzer.py`, `topology/graph.py` (det-feed/`register_det_node` path only — NOT the Protocol-facing query signatures), `topology/__init__.py` (drop `User` export), `execution/det_nodes.py` (registry), `coordination/orchestra.py` (shim final block + invariant), `agents/serialize.py`/`agents/registry.py`/`agents/agents.py` (only where a `NodeType` symbol import or reserved-name set breaks — minimal, surfaced), `tests/coordination/topology/test_serialize.py` (AC-59 inversion), `tests/coordination/topology/strategies.py` (reserved set), other tests, `packages/framework/CHANGELOG.md`, `packages/framework/DEPRECATIONS.md`, S04 `acceptance.md` (amendment).
- **NOT touch**: `execution/orchestrator.py` dispatch body; `orchestrator_types.TopologyLike` signature; the `entry_agent` run signature; edge/pattern *semantics*; `coordination/serialize.py` (ExecutionConfig); any Spren file; v0.4 removal code sites (DEPRECATIONS.md prose only).
- **Create**: characterization + new tests in place (no variant filenames; edit existing modules where they exist).

## 9. Pre-flight escalation gate

Edits `topology/core.py` (canonical model) + `orchestra.py` (TRUNK-CRITICAL) + a wire contract + a cross-package symbol. Per CLAUDE.md: **ADR-008 written approval before any code**. Phase A: validator (P4/P4b re-confirm + ledger) → improver → frozen `acceptance.md`. No code until the gate clears.

## 10. Risks

| Risk | Mitigation |
|---|---|
| P4/P4b wrong → Orchestrator change / mis-scoped materialization | Step 0 re-confirms @cite first; refute → STOP/escalate |
| Round-trip regressions, exotic combos | Exhaustive matrix + Hypothesis (AC-9/10), mandated `v0.4-spren-support.md:56` |
| Node-kind migration changes legacy outcomes | Characterization-first (a-e + f1/f2, AC-12) then parity — all 7 are parity (Start enforcement already exists; no behaviour change in the Start sub-area at all) |
| `RESERVED_NODE_NAMES` cross-package + agent-registration behaviour change | Surfaced (ADR-008 P10); AC-17 pins it; Step 0 scopes green bar; Spren breakage = contracted P9 |
| `topology_equals` oracle change hides a round-trip bug | Oracle now includes `kind`+`is_convergence_point` (decided at freeze); any convergence round-trip discrepancy exposed (in-path OR pre-existing) = fix in S08 (user-directed 2026-05-17; deliberate, recorded intentional; tradeoff: coupled rollback) |
| `NodeType`→`NodeKind` rename churn / smuggled refactors | Single in-place pass, grep-verified; reviewer scope-checks (anti-pattern #8/#15) |
| AC-59 test inversion read as silent removal | Explicit intentional reversal, surfaced in sign-off + S04 amendment (anti-pattern #1) |

## 11. Open questions for the team

None blocking — ADR-008 resolved field-name, reserved-names, load-policy, specialized-agent scope, oracle `is_convergence_point`, split-vs-single. Any new fork in Phase A is surfaced before code.

## 12. Sign-off (filled by implementer)

**Built (Steps 1–8).** `NodeType`→`NodeKind` {AGENT,START,END,USER} (SYSTEM/TOOL dropped); `Node.kind`; `core.User(Node)` removed; `Topology` homogeneous; `RESERVED_NODE_NAMES` derived from `NodeKind` (same name/path). Single-sourced `NODE_KIND_BEHAVIOUR` registry; `parse_node` returns uniform `Node(kind=...)` (never a det-node); materialization at `analyzer._add_nodes` (USER carve-out → shim, the one documented REMOVE-IN-V0.4 legacy bridge). `NodeSpec.kind`; `workflow_to_pydantic` total; `NonSerializableTopologyError` deleted; `pydantic_to_topology(spec, tool_registry, handler_registry)`; `topology_equals` keys on `kind`+`is_convergence_point`; permissive legacy-no-Start deserialize; stored `system`/`tool` rejected-at-load; `WIRE_SCHEMA_VERSION=2` embedded in `workflow_definition_schema()`. `pattern_converter` 7 sites → `kind=USER`. AC-59 tests inverted (surfaced, no skip/xfail). Docs: CHANGELOG (+ 2 stale lines corrected), DEPRECATIONS.md (analyzer carve-out support site), S04 acceptance AC-59-reversal amendment.

**AC-30 resolution (the proxy enumeration).** The independent reviewer derived from primary source that `is_convergence_point` was **never** carried for START/END pre-S08 either (the old `isinstance(DeterministicNode)` analyzer branch also `continue`d before the convergence transfer) — so the oracle change exposed **no pre-existing convergence round-trip discrepancy**. There is therefore no specific discrepancy to pin; AC-29 (full `is_convergence_point` round-trip correctness across the exhaustive matrix, `test_serialize.py` matrix + explicit-case) is the sole and sufficient AC-30 observable, per the acceptance's stated proxy contract.

**Important #2 — handler seam wired now (user-directed 2026-05-17, overriding the defer recommendation).** `Orchestra._bind_user_node_handlers` (extracted, static, unit-tested) binds each `UserNode` post-shim, preferring an explicitly-injected per-node handler (`handler_registry`→`Node.agent_ref`→`NodeInfo.agent`) over the process-wide one, and binds even with no process-wide handler. Verified end-to-end by `test_s08_coverage_gaps.py` (per-node wins / binds-without-process-wide / process-wide fallback / no-handler-stays-None). The previously-dead DI seam is now live for the canonical path.

**Notable (no apology — root causes).** Characterization-first refuted **three** Start-area premises before production code (P6 "pure-agent has no Start" → false: analyzer's unconditional `exit_points` write already drives the shim to synthesize Start+End for any terminal-bearing topology; both degenerate forms already hard-error at analyze, `graph.py:516/644`). Net: "exactly one Start" was already fully enforced for the explicit path → Step 6 became regression-pins only, no code (anti-pattern #5 avoided). Recurring root cause: reasoning at the shim/orchestra layer without first characterizing what the analyze layer admits — fixed by characterizing at the true entry point. Also fixed an internal inconsistency I introduced: `Topology.to_dict` emits `"kind"` while `parse_node`'s dict branch read `"type"` → `parse_node` now reads `"kind"` (legacy `"type"` alias retained, no regression).
