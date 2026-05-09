# Framework Session 05: Advanced Semantic Linter

Required by Spren v0.4 (visual builder + meta-agent's `lint_workflow` tool surface).

> **Status: brief-pending.** This session is held until Spren v0.3 Session 03 (the visual builder, primary consumer of `LintIssue`) is drafted. The lint rule list, `LintIssue` shape, and `severity` semantics are consumer-driven — Session 03's Researcher stage surfaces what the `@xyflow/react` canvas needs to render inline, what the meta-agent needs to act on, and what `MARSYS Cloud`'s pre-deploy validation needs to gate on. This brief is rewritten from scratch after those needs are concrete.
>
> The rest of this file below the next `---` is preserved as a placeholder reference only and is NOT the design contract. Re-read after Spren Session 03 ships and rewrite this brief from the consumer's actual requirements.

---

## Working rules

Same peer-collaboration norms as [`_template.md`](./_template.md). Read first.

### Foundational project rules

- The framework worktree's `CLAUDE.md`
- Framework architecture docs (analyzer module especially)
- [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md)

---

## The big picture

The framework already has compile-time topology validation (post-unified-barrier merge: reachability + cycle-without-escape). This PR extends `coordination/topology/analyzer.py` with a semantic-level lint pass that surfaces issues with `severity=warning` (never `error` — that stays compile-time):

- **Cost projection** — multiply per-node historical-mean token cost × topology shape; warn if estimated cost exceeds a threshold
- **Infinite-loop heuristic** — detect topology shapes that historical data shows tend to loop
- **Rate-limit warnings** — per-provider concurrency hints
- **Dependency-graph metadata** — emit edge / node metadata useful for visualization

### Multi-consumer (mandatory)

- **Spren's visual builder** consumes lint warnings inline on the canvas
- **Spren's meta-agent** uses `lint_workflow` as a read-tool to assess workflows before running
- **MARSYS Studio** consumes the same lint output in its hosted authoring UI
- **MARSYS Cloud** uses lint as a pre-deploy validation gate
- **Framework users iterating locally** see warnings in CLI output

---

## What came before this session

**Previous framework PRs from this dir:** Sessions 01–04 (independent of this PR).

**State at start of this session:**

- `coordination/topology/analyzer.py` exists with reachability + cycle-without-escape (post-phase 3.5 merge)
- Cost rate table proposed in Spren's v0.3 plan (Session 04) — by the time this PR is implemented, the framework or Spren ships a default rate table util

**Verify state with:**
```bash
pytest tests/ -x --tb=short                      # baseline
grep -rn 'def.*lint\|def.*analyze\|def.*validate' src/marsys/coordination/topology/
git log --oneline -20 src/marsys/coordination/topology/analyzer.py
```

---

## What this session ships

After merge:

- `coordination/topology/lint.py` (or extends `analyzer.py`) exposes `lint_topology(topology, config) -> list[LintIssue]`
- `LintIssue` Pydantic shape: `code, severity, message, node_or_edge_ref, suggestion`
- Implementations of:
  - `CostProjectionRule` — projects expected $ cost per run
  - `InfiniteLoopHeuristicRule` — detects shapes prone to looping
  - `RateLimitRule` — per-provider concurrency warnings
  - `DependencyGraphMetadataRule` — emits structured metadata
- All rules are `severity=warning`; existing compile-time checks remain at `severity=error`
- Configurable: caller can opt out of specific rules, set thresholds, choose strict mode
- Framework regression suite green
- New tests cover each rule's positive + negative cases

### Acceptance criteria

- [ ] `lint_topology(topology, config)` returns `list[LintIssue]`
- [ ] All four new rules implemented
- [ ] All four rules emit `severity=warning`; never `error`
- [ ] Each rule has positive + negative test cases (warning fires when it should; doesn't fire when it shouldn't)
- [ ] Configuration: `LintConfig` Pydantic with per-rule on/off + thresholds; default sensible
- [ ] CLI surface: `python -m marsys.coordination.topology.lint <workflow.json>` prints lint output
- [ ] **Multi-consumer justification documented**: at minimum Spren visual builder / Spren meta-agent / Studio / Cloud / framework local users
- [ ] Framework regression suite green
- [ ] New tests:
  - [ ] Each rule: at least one positive test (warning fires)
  - [ ] Each rule: at least one negative test (warning doesn't fire)
  - [ ] Threshold tunability: change config, observe different warnings
  - [ ] Strict mode: any warning becomes an error
  - [ ] CLI smoke test
- [ ] CHANGELOG entry
- [ ] No TRUNK-CRITICAL changes (lint pass is additive; doesn't change Orchestra behavior)
- [ ] No Spren type imported

---

## Background reading

1. The framework's `CLAUDE.md`
2. Framework architecture docs (analyzer module)
3. [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md)
4. Existing analyzer code (read end-to-end)
5. Cost rate table source (framework default util OR Spren's at the time)
6. Any historical-cost data model (token-mean per agent if it exists; otherwise the rule projects from `model.max_tokens` × `len(agents)` × empirical-multiplier)

---

## Detailed plan

### Step 0 — Capture baseline + design

Read existing analyzer; identify lint rule plug-in pattern. If none exists, design one.

### Step 1 — `LintIssue` + `LintConfig`

```python
class LintIssue(BaseModel):
    code: str                          # e.g., "COST_PROJECTION_HIGH"
    severity: Literal["info", "warning", "error"]
    message: str
    node_ref: str | None = None
    edge_ref: tuple[str, str] | None = None  # (source, target)
    suggestion: str | None = None
    metadata: dict = {}

class LintConfig(BaseModel):
    enabled_rules: list[str] | None = None       # None = all
    cost_projection_threshold_usd: float = 5.0
    infinite_loop_max_depth: int = 10
    strict: bool = False                          # warnings become errors
```

### Step 2 — Rule plug-in pattern

```python
class LintRule(Protocol):
    code: str
    def check(self, topology: Topology, config: LintConfig) -> list[LintIssue]: ...
```

### Step 3 — Implement each rule

- `CostProjectionRule` — sum `agents[i].model.estimated_cost_per_call × estimated_calls(node)`. `estimated_calls` is a topology-shape heuristic.
- `InfiniteLoopHeuristicRule` — pattern-match topologies with cycles and minimal escape paths; warn if escape weight is low.
- `RateLimitRule` — for each provider, count agents using it; warn if concurrent calls would exceed published per-provider rate limits.
- `DependencyGraphMetadataRule` — emit `LintIssue(severity="info")` per node with metadata fields that visualizers can render.

### Step 4 — CLI surface

`python -m marsys.coordination.topology.lint <workflow.json>` reads a JSON workflow, calls `lint_topology`, prints output (text / JSON via flag).

### Step 5 — Tests + CHANGELOG + framework docs

Per acceptance criteria.

### Files NOT to touch

- TRUNK-CRITICAL files
- Spren-side code

---

## Open questions for the framework team

1. **Where lint lives.** Extend `analyzer.py` vs new `lint.py`. Brief proposes new file (analyzer is about correctness; lint is about quality).
2. **Cost projection inputs.** Where do per-agent historical means live? Framework or Spren?
3. **CLI invocation.** `python -m` vs a Click subcommand on the framework's CLI. Confirm framework convention.
4. **Strict mode semantics.** `strict=True` makes warnings into errors. Useful for CI gates. Confirm.

---

## Sign-off

(Standard.)
