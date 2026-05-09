# Session 02: Workflow CRUD + Pydantic + TypeScript Type Generation

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of Spren. We work together — not above-and-below.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if a structural choice will cause us pain later, say so. Defend your position with evidence. Don't change your mind just because someone asked a question — only when you're genuinely convinced.
- **Stay engaged.** This is collaborative work, not request-response. Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Tests passing isn't enough — manually verify acceptance criteria.
- **You own follow-through.** Update this file's "What was actually built". Update [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) checkboxes. Add "Lessons / Surprises" if anything surprised you.

### Double-check before any decision

- **Read the code before changing it.** Don't assume; verify.
- **Verify file paths and symbols** still exist before referencing them.
- **Re-confirm assumptions** in tool-call comments.
- **Run tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints** after each criterion + green tests.

### Critically assess the plan itself

This brief was written before you started. It might be wrong. **Don't follow it blindly.**

- **Read the actual code** for any module the brief references.
- **Spawn an independent sub-agent for verification** when material doubt exists. Use Agent / general-purpose: "I'm about to do X based on the assumption that Y. Read [files] and tell me whether Y holds."
- **Run online research** for any "best practice" or library-version claim. If 2026 docs disagree, defer to current docs and push back.
- **Cite sources when you challenge the plan.**

### Ask the user when blocked on intent

Strategic, opinionated, or subjective decisions belong to the user. Use `AskUserQuestion` for:
- Strategic choices
- Product-vision interpretation
- Opinionated/subjective preferences
- "Something doesn't feel right"
- Scope-expansion questions

Do NOT ask for purely technical implementation choices you can decide.

### Build the smallest thing that works first; don't expand scope silently.

### Foundational project rules

- [`/CLAUDE.md`](../../../../CLAUDE.md)
- [`/BRAINDUMP.md`](../../../../BRAINDUMP.md)
- [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md)

---

## The big picture — what we're building and why

### What Spren is

Spren is a **continuously-active personal AI assistant** that lives as a local daemon on the user's machine. It has its own event loop (responds to messages, scheduled triggers, workflow lifecycle events, and heartbeat ticks), persistent memory across sessions and channels, the ability to spawn sub-agents and team managers, and the authority to act on the user's behalf with appropriate confirmation flows.

It is **the agent that runs your other agents.** It sits on top of the open-source [marsys Python framework](../../../../packages/framework/src/marsys/) and uses marsys as a tool to execute user-defined multi-agent workflows. The seam between framework and Spren is exactly three doors (SP-018): `Orchestra.run()`, `EventBus.subscribe()`, `TelemetrySink`.

It is **NOT** a chat surface. The home page is a four-surface command center: "Now" (what the agent is currently doing), "Since you were away" (items needing decision), "Activity" (chronological log), and a chat input. The chat is one of four surfaces, not the page.

Spren ships three product surfaces that all consume the same FastAPI backend (SP-019): the desktop GUI (Tauri webview), the browser GUI (system browser), and the TUI (Textual; v0.4). Distribution is via native installer per platform with secondary channels (brew / winget / apt / npm / pipx / Docker).

It is **distinct from** two proprietary products in sibling repos: MARSYS Cloud and MARSYS Studio. Spren stays focused on the **single-user-local** case so it doesn't cannibalize the proprietary product line.

### Why this work matters

Today the only way to use the marsys framework is to write Python. Spren lowers the barrier dramatically with a visual builder + always-on personal assistant. Beyond that, Spren's pitch is **observability + agency** — the user can see why their agents made the decisions they made, AND can delegate the operational work of running and tuning them.

### Your role as an implementer

1. Honor the architecture (`docs/architecture/spren/`)
2. Honor the design principles (SP-001..SP-019)
3. Ship a working artifact at session end
4. Write all required tests
5. Push back when something is wrong

### Where to read deeper if you need it

- High-level: [`docs/architecture/spren/00-overview.md`](../../../architecture/spren/00-overview.md)
- System context: [`docs/architecture/spren/01-system-context.md`](../../../architecture/spren/01-system-context.md)
- **Data model (load-bearing for this session):** [`docs/architecture/spren/02-data-model.md`](../../../architecture/spren/02-data-model.md)
- **API design (load-bearing for this session):** [`docs/architecture/spren/03-api-design.md`](../../../architecture/spren/03-api-design.md)
- v0.3 plan: [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md)

---

## What came before this session

**Previous sessions:**
- **Session 01 — Foundation** ([`01-foundation.md`](./01-foundation.md)): Restructured the umbrella monorepo into `packages/framework/` (marsys, relocated), `packages/spren/` (FastAPI sidecar), `apps/web/` (Vite + React + TanStack Router), `apps/desktop/` (Tauri 2 shell), `apps/tui/` (placeholder); set up uv workspace + pnpm workspace + Cargo workspace + Justfile; minimal FastAPI server with `/healthz`, `/v1/bootstrap`, per-launch auth token, CORS locked to localhost + `tauri://localhost`; Vite skeleton with placeholder home route; Tauri shell that spawns sidecar and opens webview; framework regression tests pass after relocation.

**State at start of this session (verified post-Session-01-merge):**
- `just dev` starts FastAPI sidecar (port 8765) + Vite dev server (port 5173)
- `just dev-desktop` starts Vite dev server + Tauri shell (which internally spawns the sidecar via `std::process::Command`)
- `GET /healthz` returns 200 without auth; `GET /v1/bootstrap` returns 200 with `Authorization: Bearer <token>`, rejects 401 without
- The FastAPI app is at `packages/spren/src/spren/server.py` and uses an `app.add_middleware(CORSMiddleware, allow_origin_regex=r"^(http://(127\.0\.0\.1|localhost)(:\d+)?|tauri://localhost)$", ...)` pattern — already permissive enough for any future endpoint
- Auth is wired via `make_auth_dependency(token)` factory from `packages/spren/src/spren/auth.py` and applied **at route level**: `@app.get("/v1/bootstrap", dependencies=[Depends(require_auth)])`. **Use this pattern for all new endpoints in this session.** Do NOT use `Annotated[None, Depends(require_auth)]` — FastAPI returns 422 not 401 in that shape (Session 01 hit + fixed this).
- Vite placeholder route at `apps/web/src/routes/index.tsx` reads token from `window.__SPREN_AUTH__` (Tauri injection) or `#token=...` URL fragment (browser fallback) and renders bootstrap. Uses inline styles only (NO Tailwind / shadcn / radix yet — Session 03 adds those).
- API client in `apps/web/src/lib/api.ts` resolves base URL via `resolveBaseUrl()`: prefers `window.__SPREN_PORT__` (Tauri injection), falls back to `import.meta.env.VITE_SPREN_API_URL` (browser dev), then same-origin (production). **Use this helper for all new endpoints; do NOT hardcode `http://127.0.0.1:8765`.**
- The framework's existing test suite passes at its new location: **841 collected, 764 passed, 20 failed, 43 skipped, 14 errors** (the 20 fail + 14 errors are pre-existing fixture issues needing `OPENROUTER_API_KEY` and are not this session's responsibility — flag if any new test surfaces them differently).
- Web package name is `@marsys/spren-web`. Use `pnpm --filter @marsys/spren-web ...` (NOT bare `web`).
- `uv run --package <pkg> ...` is the workspace-aware Python invocation (e.g., `uv run --package spren pytest packages/spren/tests`).
- No `workflows` table exists yet; no Pydantic schemas for workflows; no TS type generation pipeline yet.

**Verify state with:**
```bash
cd /home/rezaho/research_projects/marsys-spren-work/
git log --oneline -10                      # see Session 01 commits (e2e6a17, 650749b, e44e285, e77d497)
just install                                # uv sync + pnpm install + cargo fetch + tauri-cli install
just test                                   # all green at the baseline counts above
just dev &                                  # start sidecar + Vite
sleep 3
curl http://127.0.0.1:8765/healthz          # 200 ok (no auth)
jobs -p | xargs kill 2>/dev/null            # stop servers
ls packages/spren/src/spren/                # __init__.py, __main__.py, auth.py, server.py, _webui/
ls apps/web/src/                            # lib/, providers/, routes/, styles/, types/, main.tsx, routeTree.gen.ts
```

**Pinned versions** (the Session 02 type-generation tooling must compose with these):

| Tool | Version |
|---|---|
| Python | 3.12.9 |
| uv | 0.9.21 |
| pnpm | 10.33.0 |
| Vite | ^8.0.0 |
| React / React DOM | ^19.2.0 |
| TypeScript | ^6.0.0 |
| TanStack Router | ^1.169.0 |
| TanStack Router Plugin | ^1.167.0 |
| TanStack Query | ^5.100.0 |
| Vitest | ^4.1.0 |
| Playwright | ^1.59.0 |
| Tauri | 2.x |

Verify the chosen `openapi-typescript` and `json-schema-to-typescript` versions emit code that compiles against TypeScript ^6.0.0 + JSON Schema 2020-12 (NPM registry checks + the smoke procedure in Step 4).

**Relevant prior session briefs to consult if needed:**
- [`01-foundation.md`](./01-foundation.md) — for the FastAPI server setup details, the `make_auth_dependency` pattern, CORS regex, the Tauri `__SPREN_AUTH__` / `__SPREN_PORT__` injection, the `resolveBaseUrl()` helper, the `_webui/` build trick

---

## Bundle position + tier

- **Bundle**: A — Visual workflow builder ([test scenarios](../bundles/A-visual-builder/test-scenarios.md))
- **Position in bundle**: 2 of 3 (after foundation, before visual builder)
- **Tier**: HIGH — Designer + Validator/Critic (no Researcher; CRUD over SQLite + FastAPI + Pydantic is well-understood territory; only `json-schema-to-typescript` and `openapi-typescript` invocation syntax warrants a quick verification at session start)
- **Bundle outcome this session contributes to**: workflows persist in SQLite + REST CRUD works + Python file imports round-trip + TS types are generated from Pydantic. (Session 03 turns this into a real visual UI.)

## Files to DELETE from prior sessions

n/a — Session 01 produced foundation scaffolding, not workflow CRUD code. Nothing to remove.

**Note for Session 03**: this session creates a placeholder workflow CRUD UI in `apps/web/src/routes/index.tsx` (or wherever appropriate post-TanStack-Router-restructure). Session 03's brief MUST list this file for deletion when the real visual builder ships.

## What this session ships

After this session, the user can fully CRUD workflows via REST and via the Vite frontend (the latter with hand-written placeholder UI for now — Session 03 builds the proper UI). The user can also import a workflow from a Python file using the marsys framework. Workflows persist in SQLite at `<data-dir>/data/spren.db` (where `<data-dir>` is the platform-resolved per-user data directory via `platformdirs`). Pydantic schemas mirror marsys topology types and execution config; JSON Schema is exported automatically; TypeScript types are generated from JSON Schema at build time and consumed by `apps/web`.

The bootstrap response from Session 01 is extended to include the workflow list endpoint URL pattern, so the frontend knows where to fetch workflows from. The visual builder (Session 03) sits on top of this CRUD foundation.

### Acceptance criteria

- [ ] SQLite database initialized at `<data-dir>/data/spren.db` on first server start (creates parent directory if needed)
- [ ] Migrations runner: schema versioned in a `_migrations` table; idempotent; migration scripts under `packages/spren/src/spren/storage/migrations/`
- [ ] First migration creates the `workflows` table per the data model in `02-data-model.md` (including `provenance` and `provenance_metadata` columns)
- [ ] Pydantic schemas for `Workflow`, `WorkflowDefinition`, `TopologySpec`, `NodeSpec`, `EdgeSpec`, `AgentSpec`, `ExecutionConfigSpec` mirror marsys's dataclass types; `ModelConfig` is imported directly from `marsys.models` (already Pydantic)
- [ ] REST CRUD endpoints under `/v1/workflows`: GET (list, paginated, filterable by `provenance` and `archived`), POST (create), GET (read), PUT (replace), PATCH (partial update / archive), DELETE (hard delete only when no runs reference; if runs reference, return HTTP 409 with `error.code = "WORKFLOW_HAS_RUNS"` and leave the row intact)
- [ ] `POST /v1/workflows/import-python` accepts a multipart upload of a `.py` file, parses agent / topology / execution_config constructs in the constructor-only style (Step 3), materializes a workflow record with `provenance="code_import"` and `provenance_metadata={source_filename, sha256}`. Parsing uses two-pass Python AST inspection (no `exec`/`eval`); rejects dict-DSL, comprehensions, conditionals, function/class defs, f-strings, and files >1 MB
- [ ] All endpoints require auth (per session 01's auth pattern)
- [ ] Cursor-based pagination on the list endpoint (`?cursor=...&limit=N`, max 100)
- [ ] OpenAPI schema available at `/openapi.json` (FastAPI auto-generates)
- [ ] Build script in `apps/web/` fetches `/openapi.json` from a running dev server (or static snapshot) and runs `openapi-typescript` to generate `apps/web/src/lib/api-types.generated.ts`
- [ ] Pydantic-only models that aren't represented in OpenAPI request/response schemas are exported to JSON Schema (via `model.model_json_schema()`) and then to TypeScript (via `json-schema-to-typescript`) into `apps/web/src/lib/types.generated.ts`
- [ ] `apps/web` placeholder route is updated to: list workflows from `/v1/workflows` (with provenance shown), allow creating a workflow with a hand-typed name + JSON definition (placeholder textarea — Session 03 replaces with canvas), allow deleting one, allow uploading a `.py` file to test the Python import endpoint
- [ ] All required tests written and passing (see Tests section)
- [ ] No mocks of in-codebase features (SP-007)
- [ ] No backward-compat code (SP-006) — first migration is the only schema
- [ ] No TRUNK-CRITICAL framework changes (SP-001) — all new code in `packages/spren/`
- [ ] No Spren type imported into `packages/framework/` (SP-018)
- [ ] No hand-written TS types mirroring Pydantic (SP-005)

---

## Background reading (do this before writing code)

1. [`/CLAUDE.md`](../../../../CLAUDE.md) — project rules, TRUNK criticality
2. [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md) — especially SP-005 (Pydantic source of truth), SP-006 (no backward-compat), SP-009 (run snapshots immutable), SP-018 (framework knows nothing of Spren), SP-019 (API is single source of truth)
3. [`docs/architecture/spren/02-data-model.md`](../../../architecture/spren/02-data-model.md) — the SQLite schema for `workflows` (including `provenance` columns) and the JSON shape of `definition`
4. [`docs/architecture/spren/03-api-design.md`](../../../architecture/spren/03-api-design.md) — the workflow REST endpoints (including `import-python`), error format, idempotency, pagination
5. [`packages/framework/src/marsys/coordination/topology/core.py`](../../../../packages/framework/src/marsys/coordination/topology/core.py) — marsys's Node, Edge, NodeType, EdgeType
6. [`packages/framework/src/marsys/coordination/topology/patterns.py`](../../../../packages/framework/src/marsys/coordination/topology/patterns.py) — PatternConfig presets
7. [`packages/framework/src/marsys/agents/agents.py`](../../../../packages/framework/src/marsys/agents/agents.py) — Agent constructor params
8. [`packages/framework/src/marsys/coordination/config.py`](../../../../packages/framework/src/marsys/coordination/config.py) — ExecutionConfig fields
9. [`packages/framework/src/marsys/models/models.py`](../../../../packages/framework/src/marsys/models/models.py) — ModelConfig fields
10. [json-schema-to-typescript docs](https://www.npmjs.com/package/json-schema-to-typescript) — verify current version + JSON Schema 2020-12 compatibility
11. [openapi-typescript docs](https://openapi-ts.dev/) — verify current version + CLI invocation
12. Python `ast` module documentation — used by the Python-file import endpoint to parse marsys constructs without executing the file

**Verify before proceeding:**
- `git log --oneline -20 packages/framework/src/marsys/coordination/topology/core.py` — see if Node/Edge/NodeType/EdgeType have moved or changed signatures
- `grep -rn 'class Node\b\|class Edge\b\|class NodeType\b\|class EdgeType\b' packages/framework/src/marsys/coordination/topology/` — confirm these classes are at the paths the brief assumes
- Check current pinned versions of `pydantic`, `fastapi`, `openapi-typescript`, `json-schema-to-typescript`. Pin within the same major; bump patches as needed.

---

## Detailed plan

### Step 1 — Add storage module skeleton

Create:
- `packages/spren/src/spren/storage/__init__.py` — empty
- `packages/spren/src/spren/storage/db.py` — thin `SQLiteConnection` wrapper that opens `<data-dir>/data/spren.db` via `platformdirs.user_data_dir("spren")` and ensures the parent directory exists
- `packages/spren/src/spren/storage/migrations/__init__.py` — `MigrationsRunner` class: reads `*.py` migration files in order, runs unapplied ones, records applied IDs in a `_migrations` table

Migration file convention: `<NN>__<slug>.py` — each exports `def upgrade(conn: sqlite3.Connection) -> None`. Forward-only (SP-006); no downgrade. The runner is roll-our-own (~50 lines); Alembic is heavier than v0.3 needs.

The first migration `01__create_workflows.py` creates the `workflows` table per `docs/architecture/spren/02-data-model.md`: `id` (TEXT PRIMARY KEY, ULID), `name`, `description`, `definition` (TEXT JSON), `definition_version` (INTEGER), `provenance` (TEXT NOT NULL), `provenance_metadata` (TEXT), `is_archived` (INTEGER), `created_at` (TEXT), `updated_at` (TEXT).

IDs are ULIDs generated via `python-ulid ^3.0` (added to `packages/spren/pyproject.toml` deps): `from ulid import ULID; str(ULID())`. Verify the pin against current PyPI before installing.

Timestamps in `created_at` / `updated_at` are ISO 8601 UTC with microsecond precision, generated in Pydantic via `datetime.now(timezone.utc).isoformat()`. Do not use SQLite's `CURRENT_TIMESTAMP` — it is second-precision only and breaks the architecture doc's microsecond contract.

Migrations run on FastAPI app startup (lifespan event).

### Step 2 — Pydantic models

`marsys.models.ModelConfig` is the only marsys type Spren needs that is already a Pydantic `BaseModel` (`packages/framework/src/marsys/models/models.py`). The rest — `Node`, `Edge`, `Topology`, `PatternConfig`, `ExecutionConfig`, `ConvergencePolicyConfig`, `TracingConfig`, `StatusConfig`, `BaseAgent` — are dataclasses or runtime objects.

Pattern: import `ModelConfig` directly from `marsys.models`; mirror everything else as Pydantic. The mirror is the API-boundary contract; the conversion to runnable marsys topology happens at execution time in Session 04, not at storage time.

**Future cleanup commitment (post-Framework Session 04)**: when the framework's canonical Pydantic ↔ runtime-topology serializer (Framework Session 04) merges and is released, Spren opens a follow-up cleanup PR within 24 hours that deletes `packages/spren/src/spren/models/topology.py`, `packages/spren/src/spren/models/agent.py`, `packages/spren/src/spren/models/execution_config.py` and replaces their imports with `from marsys.coordination.topology.serialize import WorkflowDefinition, TopologySpec, NodeSpec, EdgeSpec, AgentSpec, ExecutionConfigSpec`. Framework Session 04's brief carries the symmetric commitment.

Create:
- `packages/spren/src/spren/models/__init__.py` — re-exports including `from marsys.models import ModelConfig`
- `packages/spren/src/spren/models/workflow.py`:
  - `class WorkflowDefinition(BaseModel)` — topology + agents map + execution_config
  - `class Workflow(BaseModel)` — SQLite row shape: id, name, description, definition, definition_version, provenance: Literal["visual_builder","meta_agent","code_import","template","api"], provenance_metadata: dict | None, is_archived, created_at, updated_at
  - `class WorkflowCreateRequest`, `class WorkflowUpdateRequest`, `class WorkflowListResponse`
- `packages/spren/src/spren/models/topology.py`:
  - `class TopologySpec(BaseModel)` — nodes: list[NodeSpec], edges: list[EdgeSpec], rules: list[str]
  - `class NodeSpec`, `class EdgeSpec` — mirror marsys's Node/Edge dataclasses
  - `NodeType` and `EdgeType` enums — values match marsys's enum values exactly. Read `marsys.coordination.topology.core` to confirm: they're plain `Enum` (NOT `StrEnum`) with lowercase string values. Mirror that shape exactly.
- `packages/spren/src/spren/models/agent.py`:
  - `class AgentSpec(BaseModel)` — `agent_model: ModelConfig` (NOT `model_config` — that name is reserved by Pydantic v2's `BaseModel`), name, goal, instruction, `tools: list[str]` (string names; runtime tool registry resolves them at execution time), memory_retention, allowed_peers, plan_config (optional)
- `packages/spren/src/spren/models/execution_config.py`:
  - `class ExecutionConfigSpec(BaseModel)` — mirror marsys's ExecutionConfig + ConvergencePolicyConfig (the convergence policy is a polymorphic union: `Union[float, str, ConvergencePolicyConfigSpec]` with a discriminator)
- `packages/spren/src/spren/models/errors.py`:
  - `ErrorCode = Literal["WORKFLOW_NOT_FOUND", "PYTHON_IMPORT_REJECTED", "MIGRATION_FAILED", "VALIDATION_FAILED", "WORKFLOW_HAS_RUNS", ...]` — exhaustive string union for the `error.code` field across endpoints

`docs/architecture/spren/02-data-model.md` already uses `agent_model` and lowercase enum values for the `definition` JSON shape — no doc edit needed. If you spot any remaining mismatch (anywhere in the architecture or implementation docs) between the doc's stated shape and the Pydantic mirror you create, that's a doc bug to fix as part of this PR.

### Step 3 — REST endpoints + Python importer

Create:
- `packages/spren/src/spren/routes/__init__.py`
- `packages/spren/src/spren/routes/workflows.py` — APIRouter with the CRUD endpoints + the `import-python` endpoint
- `packages/spren/src/spren/importers/__init__.py`
- `packages/spren/src/spren/importers/python_workflow.py` — AST-based parser. Reads a marsys-framework `.py` file, walks the AST to extract `Agent(...)` constructors, `Topology(...)` constructor calls, ModelConfig literals, tool name references; returns a `WorkflowDefinition` Pydantic model. Refuses any file that contains constructs outside the expected pattern.

The Python-file importer reads source written against the **marsys framework's own constructor surface** (NOT the Spren API-boundary shape) and translates to a Spren `WorkflowDefinition`. The parser must understand the real marsys signatures:

- `Agent(name=..., model_config=<ModelConfig>, goal=..., instruction=..., tools=<Dict[str, Callable]>, ...)` — note the kwarg is `model_config` (matching `agents.py:131`), NOT `agent_model`. The `tools` value is a dict whose keys are tool names; values are callables. The parser extracts the dict's keys (string names) and discards the callables — Spren's API mirror uses `tools: list[str]` of names; the runtime tool registry resolves names to callables at execution time per Framework Session 04's pattern.
- `Topology(nodes=[Node(...), ...], edges=[Edge(...), ...])` — explicit constructor calls.
- `ModelConfig(...)`, `ExecutionConfig(...)` — literal kwargs only.
- Module-level constants (`AGENT_NAME = "researcher"`) resolved via a two-pass AST walk: pass 1 builds a `Name → Constant` table; pass 2 resolves constructor-arg `Name` references against it.

The parser-output translation:
- marsys `Agent(model_config=ModelConfig(...))` → Spren `AgentSpec(agent_model=ModelConfig(...))` (Spren renames the field at the API boundary; marsys's own kwarg stays `model_config`).
- marsys `Agent(tools={"search_web": <callable>, "browse_url": <callable>})` → Spren `AgentSpec(tools=["search_web", "browse_url"])`.
- marsys `Topology(nodes=[...], edges=[...])` → Spren `TopologySpec(nodes=[...], edges=[...])`, with `node_type` / `edge_type` enum values mirrored from `marsys.coordination.topology.core` (read the file at brief start to confirm casing — they're plain `Enum`, NOT `StrEnum`, with lowercase values).

The parser rejects any file containing:
- `exec(...)`, `eval(...)`, `compile(...)`, `__import__(...)`
- Class definitions, function definitions, decorators, comprehensions, conditionals
- The dict-of-string-arrows topology DSL (e.g., `{"Start -> Researcher": ...}`) — many of the framework's own example files at `packages/framework/examples/` use this style; clean AST parsing of it is out of scope for v0.3. Users with dict-DSL workflows rewrite to `Topology(...)` constructor style or use the visual builder.
- f-strings in user-facing fields
- `Subscript` expressions inside `tools=` dict values (e.g., `tools={"x": something[0]}`) — the parser only resolves dict keys, but a malformed value at parse time signals a non-trivial Python file outside the supported subset
- Multi-file imports of workflow constructs (the parser reads one file)
- Files larger than 1 MB
- Non-UTF-8 encoding

Rejections return `422 Unprocessable Entity` with `{"error": {"code": "PYTHON_IMPORT_REJECTED", "message": "<human-readable>", "details": {"reason": "<short-tag>", "line": <N>}}}`. The human-readable message names the unsupported construct and points the user at the visual builder.

Test fixtures live at `packages/spren/tests/fixtures/python_workflows/` and use synthetic constructor-style files written against marsys's real signatures:
- `valid_minimal.py` — 2-agent linear pipeline using `Agent(model_config=ModelConfig(...), tools={...})` + explicit `Topology(nodes=[Node(...)], edges=[Edge(...)])`
- `valid_with_constants.py` — same shape but uses module-level `AGENT_NAME = "researcher"` Name refs
- `invalid_dict_dsl.py` — uses `{"Start -> Researcher": ...}`
- `invalid_exec.py` — contains `exec(...)`
- `invalid_dynamic_topology.py` — builds edges via comprehension
- `invalid_too_large.py` — 1.5 MB

The framework's own example files at `packages/framework/examples/` (e.g., `example_01_Deep_Research.py`, `example_02_local_models.py`) predominantly use the dict-DSL style and are NOT used as fixtures — they're the reason the rejected-DSL test fixture exists. Step 7's manual integration loop should cite a synthetic constructor-style fixture as the import target, not a real example file (real examples 422 by design until they're rewritten to constructor style).

**Auth pattern (mandatory)**: each endpoint takes auth at the **route level**, not the function signature, matching Session 01's pattern. Build the router with the auth dep from `spren.server` (or pass via `APIRouter(dependencies=[Depends(require_auth)])`). Do NOT use `Annotated[None, Depends(require_auth)]` in handler signatures — that surface returns 422 not 401.

```python
# packages/spren/src/spren/routes/workflows.py — pattern
from fastapi import APIRouter, Depends
def make_workflows_router(require_auth) -> APIRouter:
    router = APIRouter(prefix="/v1/workflows", dependencies=[Depends(require_auth)])
    @router.get("")
    def list_workflows(...) -> WorkflowListResponse: ...
    return router
```

Mount the router in `spren.server.create_app()` after the CORS middleware: `app.include_router(make_workflows_router(require_auth))`. Endpoints per `docs/architecture/spren/03-api-design.md` § Workflows.

Error response format: `{"error": {"code": ErrorCode, "message": str, "details": dict}}` where `ErrorCode` is the `Literal[...]` union from `spren.models.errors`. A global exception handler maps Pydantic `ValidationError` → `VALIDATION_FAILED`, marsys exceptions → their respective codes, and unexpected errors → `INTERNAL_ERROR` (with the original message in `details` only in dev mode).

**Cursor pagination on the list endpoint**: `?cursor=<opaque>&limit=<N>` (max 100, default 20). The cursor is a base64-encoded JSON `{"id": "<ulid>", "ts": "<iso>"}` HMAC-signed with a per-launch secret (regenerated on each server start, alongside the auth token, in `auth.py`). The server validates the HMAC before consuming the cursor; tampered cursors return 400. The `next_cursor` field on `WorkflowListResponse` is `null` when no more pages exist.

**Idempotency-Key header** (optional on `POST` / `PUT` / `PATCH` / `DELETE` under `/v1/workflows*`; ignored on `GET`): when present, the server caches the response for 24 hours and replays the cached response on a key-match within that window. Cache lives in a `_idempotency` table in `<data-dir>/data/spren.db` so it survives daemon restart inside the 24-hour window; expired rows are swept on startup. Cache key is `(method, path, idempotency_key)`; mismatched method/path with the same key is treated as a fresh request.

### Step 4 — Type generation pipeline

The TS type-gen pipeline produces two files: `apps/web/src/lib/api-types.generated.ts` (request/response shapes from `/openapi.json`) and `apps/web/src/lib/types.generated.ts` (standalone Pydantic types not in OpenAPI request/response schemas).

`openapi-typescript@7.x` declares `peerDependencies: { typescript: "^5.x" }`; the workspace pins `typescript: ^6.0.0`. Configure a pnpm peer-dep override (`apps/web/package.json` → `pnpm.overrides`) to allow TS 6 and verify empirically that the emitted `.ts` compiles under TS 6:

```bash
cd apps/web/
pnpm add -D openapi-typescript@^7
echo '{"openapi":"3.1.0","info":{"title":"t","version":"1"},"paths":{},"components":{"schemas":{"Foo":{"type":"object","properties":{"x":{"type":"integer"}}}}}}' > /tmp/openapi-smoke.json
pnpm exec openapi-typescript /tmp/openapi-smoke.json -o /tmp/types-smoke.ts
pnpm exec tsc --noEmit /tmp/types-smoke.ts
```

If the smoke check fails (emitted `.ts` has TS 6 syntactic incompatibilities) or produces deprecation warnings on common shapes, switch to `@hey-api/openapi-ts` (verify its TS 6 peer-dep first); ask the user via `AskUserQuestion` if the alternative meaningfully changes the API surface.

For Pydantic → TypeScript on standalone models that aren't in OpenAPI request/response shapes, use `json-schema-to-typescript` (npm package, actively maintained, accepts JSON Schema 2020-12). **Do not use `datamodel-code-generator`** — despite its name, that package only emits Python (Pydantic / dataclasses / msgspec / TypedDict); it has no TypeScript output mode. Verify current `json-schema-to-typescript` version + JSON Schema 2020-12 compatibility on npm before pinning.

Then create:
- `apps/web/scripts/generate-types.ts` — fetches `/openapi.json` from a running dev server (resolved via `__SPREN_PORT__` or `VITE_SPREN_API_URL`), runs the chosen OpenAPI generator to produce `api-types.generated.ts`. Falls back to a static snapshot at `apps/web/openapi-snapshot.json` for CI-style builds without a live server.
- A second script (`apps/web/scripts/generate-pydantic-types.ts`) that:
  1. Calls a small Python helper (`uv run --package spren python -m spren.scripts.export_json_schema`) to export JSON Schema for each Pydantic model in `spren.models` via `model.model_json_schema()` (using `mode='serialization'` for response shapes and `mode='validation'` for request shapes per Pydantic v2). Writes consolidated schema to `apps/web/types-source.json`.
  2. Runs `json-schema-to-typescript` against `types-source.json` into `apps/web/src/lib/types.generated.ts`.

Wire both into `package.json`'s `predev` and `prebuild` scripts so types are always fresh before the dev server starts and before each build. `generate-types.ts` and `generate-pydantic-types.ts` run under `tsx` (declared as a devDep).

### Step 5 — Frontend placeholder UI for workflows

Update `apps/web/src/routes/index.tsx` to add minimal workflow CRUD controls. Use the existing patterns from Session 01:
- Reuse `useCapabilities()` from `apps/web/src/providers/capabilities.tsx` to read the auth token from context
- Build URLs via `resolveBaseUrl()` from `apps/web/src/lib/api.ts` — do NOT hardcode `http://127.0.0.1:8765`
- Use `fetch(...)` directly (Session 02 doesn't add TanStack Query data fetching wiring; Session 03 will)
- **Inline styles only** (`style={{ ... }}`) matching `routes/index.tsx`'s current pattern. **Do NOT add Tailwind / shadcn / radix / cmdk / Geist as CSS** — those are deferred to Session 03 (Session 01's "What was actually built" explicitly defers them; adding them here is scope creep)

Add the new TS types (`Workflow`, `WorkflowDefinition`, etc.) to `apps/web/src/lib/api.ts` as **placeholder interfaces** (matching the existing pattern of `BootstrapResponse`). Do NOT replace `api.ts`'s placeholder types with the generated ones in this same edit — that's Step 4's job (the generated `api-types.generated.ts` and `types.generated.ts` files). Once those generate cleanly, switch the inline interfaces to imports from the generated files in a final cleanup commit at the end of the session.

**End-state requirement**: when the session is declared done, no TypeScript file in `apps/web/src/lib/` (other than the two `*.generated.ts` files) declares interfaces or types named after Pydantic models (`Workflow`, `WorkflowDefinition`, `TopologySpec`, `NodeSpec`, `EdgeSpec`, `AgentSpec`, `ExecutionConfigSpec`). The acceptance auditor verifies this end-state.

The placeholder route renders:
- A list of workflow names + a provenance badge per row (small inline `<span>` with provenance-specific background color)
- A "Create" button → simple form with `name` text input + `definition` textarea (JSON) → POST → refresh list
- A "Delete" button per workflow → DELETE → refresh list
- An "Import Python file" button → file input → multipart POST to `/v1/workflows/import-python` → refresh list

This UI is INTENTIONALLY ugly. Session 03 replaces it with the real visual builder + design system. Don't invest in styling.

**Justfile note**: any new pnpm invocation in `Justfile` uses the scoped name: `pnpm --filter @marsys/spren-web ...` (NOT bare `web`). Any new Python invocation uses `uv run --package spren ...`.

### Step 6 — Update bootstrap response

Extend `BootstrapResponse` in Session 01's contract to add:
- `endpoints.workflows: str` — the URL prefix for workflow CRUD (`/v1/workflows`)
- (Other endpoint hints can come in later sessions)

### Step 7 — Run the integration loop

```bash
just install
just dev &                                  # start sidecar + Vite (sidecar prints `spren-ready: port=N token=T data-dir=P` on stdout)
TOKEN=<from sidecar output>
sleep 3
curl localhost:8765/v1/bootstrap -H "Authorization: Bearer $TOKEN"          # 200 with new shape (endpoints.workflows present)
curl -X POST localhost:8765/v1/workflows -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "definition": {...}, "provenance": "api"}'           # 201 with new workflow
curl localhost:8765/v1/workflows -H "Authorization: Bearer $TOKEN"          # 200 list with the workflow
curl -X POST localhost:8765/v1/workflows/import-python -H "Authorization: Bearer $TOKEN" \
  -F "file=@packages/spren/tests/fixtures/python_workflows/valid_minimal.py"  # 201 with the imported workflow
# (Real example files at packages/framework/examples/ predominantly use the dict-DSL style and 422 by design.)
DATA_DIR=$(uv run --package spren python -c "import platformdirs; print(platformdirs.user_data_dir('spren'))")
sqlite3 "$DATA_DIR/data/spren.db" "SELECT id, name, provenance FROM workflows;"
```

Visit the placeholder UI at `http://localhost:5173/#token=$TOKEN` and verify CRUD + Python import work. Also visit via `just dev-desktop` and confirm the same flows work in the Tauri webview (the auth token comes from `__SPREN_AUTH__` injection rather than the URL fragment).

### Files NOT to touch

- Anything inside `packages/framework/src/marsys/coordination/`, `agents/`, `models/`, `environment/` — TRUNK-CRITICAL or TRUNK-STABLE; we mirror their types but don't modify them
- `docs/` — only update if links break
- Existing marsys framework tests under `packages/framework/tests/` — those belong to the marsys repo; don't modify from the Spren side

### Load-bearing shapes

```python
# packages/spren/src/spren/models/workflow.py
class Workflow(BaseModel):
    id: str               # ULID
    name: str
    description: str | None = None
    definition: WorkflowDefinition
    definition_version: int = 1
    provenance: Literal["visual_builder", "meta_agent", "code_import", "template", "api"]
    provenance_metadata: dict | None = None
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime

class WorkflowDefinition(BaseModel):
    topology: TopologySpec
    agents: dict[str, AgentSpec]    # key = agent_id (also used as Node.agent_ref)
    execution_config: ExecutionConfigSpec

# Pagination response
class WorkflowListResponse(BaseModel):
    items: list[Workflow]
    next_cursor: str | None
    has_more: bool
```

```python
# packages/spren/src/spren/models/topology.py
# NodeType / EdgeType / EdgePattern enum values match marsys's plain Enum values exactly —
# read marsys.coordination.topology.core to confirm; values are lowercase strings.

class TopologySpec(BaseModel):
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    rules: list[str] = []           # marsys rules engine references

class NodeSpec(BaseModel):
    name: str                                # cannot collide with reserved system identifiers
    node_type: NodeType                      # mirror of marsys.coordination.topology.core.NodeType
    agent_ref: str | None = None             # references an entry in WorkflowDefinition.agents
    is_convergence_point: bool = False
    metadata: dict = {}

class EdgeSpec(BaseModel):
    source: str
    target: str
    edge_type: EdgeType                      # mirror of marsys.coordination.topology.core.EdgeType
    bidirectional: bool = False
    pattern: EdgePattern | None = None       # mirror of marsys.coordination.topology.core.EdgePattern
    metadata: dict = {}
```

```python
# packages/spren/src/spren/models/agent.py
from marsys.models import ModelConfig         # imported directly; already Pydantic

class AgentSpec(BaseModel):
    agent_model: ModelConfig                  # NOT named `model_config` (Pydantic v2 reserves that)
    name: str
    goal: str
    instruction: str
    tools: list[str] = []                     # tool names; runtime registry resolves at execution time
    memory_retention: Literal["session", "ephemeral", "persistent"] = "session"
    allowed_peers: list[str] = []
    plan_config: PlanConfigSpec | None = None
```

---

## Hard rules (per session)

### No mocks of in-codebase features (SP-007)
- The placeholder UI is real (it really does CRUD workflows). It's just ugly. Session 03 replaces it wholesale.
- The Pydantic mirrors of marsys types are NOT mocks — they're the API-boundary representation. The conversion to/from marsys types happens at execution time and is real code.

### No backward compatibility (SP-006)
- The first migration is the schema. No `if old_format then ... else` code paths.
- If you change a Pydantic field name later, write a migration; don't keep both names alive.

### No TRUNK-CRITICAL framework changes (SP-001)
- Mirror types in `spren.models`; do not import + re-export from `marsys.coordination.topology` if doing so requires modifying those files.

### Other Spren design principles
- **SP-005**: Pydantic is the source of truth for types. Hand-writing TS types is FORBIDDEN. Generate them.
- **SP-009**: Run snapshots immutable — n/a yet (no runs in v0.3 till Session 04), but: when you eventually link runs to workflows, the run will snapshot the workflow's `definition` at run start. Keep this in mind for the schema design.

### Clean code rules
- One concern per module: storage, models, routes, services. Don't smear them.

---

## Tests (required for "done")

### Unit tests

- `packages/spren/tests/test_models_workflow.py`:
  - Pydantic models accept valid examples and reject invalid (missing required fields, wrong types, reserved node names, invalid provenance values)
  - JSON Schema export produces a non-empty schema for each model
- `packages/spren/tests/test_storage_migrations.py`:
  - Migrations runner is idempotent
  - First migration creates the `workflows` table with all expected columns + indexes (including `provenance` and `provenance_metadata`)
- `packages/spren/tests/test_routes_workflows.py`:
  - With FastAPI test client and a temp-file SQLite, exercise each endpoint:
    - POST creates and returns the new workflow with generated `id`, `created_at`, `updated_at`, `provenance` defaulted to `api` if not specified
    - GET list returns it; filterable by `provenance` and `archived`
    - GET by id returns it
    - PUT replaces; updated_at changes; created_at preserved
    - PATCH `is_archived=true` works
    - DELETE removes
    - Pagination: create 25 workflows; GET list with `limit=10` returns 10 items + non-null `next_cursor`; following the cursor returns the next batch
- `packages/spren/tests/test_importers_python.py`:
  - Provide a fixture file with a known marsys-framework workflow → AST parse extracts the right agents, topology, configs
  - Provide a malformed file (missing topology) → returns a 422 with a clear error
  - Provide a file containing `exec(...)` or other forbidden constructs → rejects without execution

### Integration tests

- `packages/spren/tests/integration/test_workflow_crud_e2e.py`:
  - Spin up the FastAPI app with a real SQLite file at a temp path
  - Issue real HTTP requests via TestClient or httpx with auth
  - Round-trip a complex workflow (with multiple agents + edges + execution_config) and verify shape preservation
  - Round-trip a Python file import → resulting workflow record matches the file's structure
  - Verify OpenAPI schema is reachable at `/openapi.json` and contains expected paths

### End-to-end tests

- `apps/web/tests/e2e/workflow-crud.spec.ts` (Playwright):
  - Start dev server fixtures (FastAPI + Vite)
  - Visit page with auth token in fragment
  - Create a workflow via the placeholder UI
  - Import a workflow from a Python file
  - Verify both appear in the list with correct provenance badges
  - Delete each
  - Verify they're gone

### Type-generation tests

- `apps/web/scripts/generate-types.test.ts` (Vitest):
  - Generated `api-types.generated.ts` is non-empty and parses as TS
  - Generated `types.generated.ts` is non-empty and parses as TS
  - The generated `Workflow` type round-trips with a JSON sample

### Test framework conventions

- Python: pytest; fixtures in `conftest.py`
- TypeScript: Vitest for unit/integration; Playwright for E2E

---

## Manual-verify checklist (implementer fills before declaring done)

Tests passing isn't enough. Before marking this session done, run through the list below in a real browser pointed at a real running stack (`just dev`). Capture screenshots/observations into `./tmp/spren/sessions/02-workflow-crud-types/manual-verify.md`.

- [ ] `just install && just dev` boots cleanly; no Python or pnpm errors in either prefixed log stream
- [ ] First-launch creates `<data-dir>/data/spren.db` and runs the migration; subsequent launches do not re-run it (verify via `_migrations` table contents)
- [ ] Open `http://localhost:5173/#token=<token-from-stdout>`; placeholder workflow page loads without console errors
- [ ] Create a workflow via the placeholder UI with a name + minimal definition JSON; row appears in list with `provenance=api`
- [ ] Refresh the page (preserving the token in fragment); workflow still appears (DB persistence works)
- [ ] Upload a known-good marsys-framework `.py` file (use `examples/` as fixture source); workflow appears with `provenance=code_import` and `provenance_metadata` showing source filename + sha256
- [ ] Upload a `.py` file containing `exec(...)` or any forbidden construct → 422 with clear error; nothing inserted
- [ ] Delete a workflow via the placeholder UI; row disappears; SQLite confirms the row is gone
- [ ] Patch a workflow with `is_archived=true`; archived workflow disappears from default list, reappears with `?archived=true` filter
- [ ] Hit `/openapi.json` directly; spec is non-empty and includes `/v1/workflows` paths
- [ ] In Tauri shell (`just dev-desktop`): same flows work; auth token is injected via `window.__SPREN_AUTH__` not URL fragment
- [ ] Generated `apps/web/src/lib/api-types.generated.ts` and `types.generated.ts` are present + non-empty after build; `pnpm --filter web build` succeeds
- [ ] `grep -rn 'Mock\|mock.patch\|vi.mock\|MagicMock' packages/spren/src/ apps/web/src/` returns ZERO hits in product code (test files only)
- [ ] No `if version` / `# legacy` / `# TODO: remove` patterns in product code
- [ ] All tests green: `just test` exits 0

If any item fails: investigate and fix before declaring done. Do NOT close out the session with a known regression.

---

## Open questions for the user

The implementer must surface these via `AskUserQuestion` BEFORE writing code:

1. **If the openapi-typescript × TypeScript 6 smoke check (Step 4) fails or produces meaningful incompatibilities**: surface the alternative-tool choice (`@hey-api/openapi-ts` is the leading candidate; `swagger-typescript-api` is the fallback). Particularly important if the alternative changes the emitted API surface for handlers (different naming conventions, different response-typing helpers).

---

## Sign-off

On completion:

1. Update **What was actually built** below with the delta from the plan, if any
2. Update [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) — check Session 02's row
3. Add a **Lessons / Surprises** entry below if anything surprising came up

### What was actually built (filled by implementer)

> _Implementer fills this in._
>
> Include: which migrations runner approach taken (custom vs Alembic), whether marsys types were mirrored or imported directly (with reasons), TS type generation tooling versions pinned, database path resolution approach, anything done differently from the plan with reasons.

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._
