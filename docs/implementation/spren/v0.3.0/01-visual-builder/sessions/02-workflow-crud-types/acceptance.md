# Acceptance criteria — Spren v0.3 Session 02: Workflow CRUD + Pydantic + TypeScript Type Generation

Frozen at 2026-05-10T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Manual-only criteria are tagged `(manual-only)` — they're part of the contract but cannot be automated; the implementer runs them and the auditor checks whether tests + manual-verify together cover the AC.

## Functional — Storage / migrations

- AC-1: On first server startup, a SQLite database file exists at `<data-dir>/data/spren.db` (where `<data-dir>` is the platform-resolved per-user data directory via `platformdirs.user_data_dir("spren")`); the parent directory is created if missing.
- AC-2: The SQLite connection enables `PRAGMA foreign_keys = ON` and `PRAGMA journal_mode = WAL` (observable via `PRAGMA` queries against the connection).
- AC-3: A `_migrations` table exists after first startup and records the ID of every successfully applied migration along with an `applied_at` timestamp.
- AC-4: Running the migrations runner twice in a row applies each migration only once (idempotent — second run is a no-op).
- AC-5: A migration that fails partway through leaves no row in `_migrations` for that migration (no half-written state); the schema is in the pre-migration state.
- AC-6: After the first migration, a `workflows` table exists with columns: `id` (TEXT PRIMARY KEY), `name` (TEXT NOT NULL), `description` (TEXT), `definition` (TEXT NOT NULL), `definition_version` (INTEGER NOT NULL), `provenance` (TEXT NOT NULL), `provenance_metadata` (TEXT), `is_archived` (INTEGER NOT NULL DEFAULT 0), `created_at` (TEXT NOT NULL), `updated_at` (TEXT NOT NULL).
- AC-7: The `workflows` table has indexes on `provenance`, `is_archived`, and `created_at`.
- AC-8: After the first migration, an `_idempotency` table exists with a compound primary key over `(method, path, key)` (or equivalent) and columns `response_status` (INTEGER), `response_body` (BLOB), `response_headers` (TEXT JSON), `created_at` (TEXT), `expires_at` (TEXT). An index on `expires_at` exists.
- AC-9: On every FastAPI app startup, expired `_idempotency` rows (`expires_at` < current UTC time) are deleted.
- AC-10: Workflow `id` values are 26-character ULIDs (string form). (resolved-13: `python-ulid >= 3.0,<4.0`; observable shape only.)
- AC-11: `created_at` and `updated_at` are ISO 8601 UTC timestamps with microsecond precision (NOT second-precision SQLite `CURRENT_TIMESTAMP`).
- AC-12: The migrations runner class is at `packages/spren/src/spren/storage/migrations/runner.py`; the migrations package's `__init__.py` re-exports `MigrationsRunner` but does NOT contain the runner logic itself. (resolved-8)

## Functional — Pydantic models + validation

- AC-13: A `Workflow` Pydantic model exists with fields: `id: str`, `name: str`, `description: str | None`, `definition: WorkflowDefinition`, `definition_version: int = 1`, `provenance: Literal["visual_builder","meta_agent","code_import","template","api"]`, `provenance_metadata: dict | None`, `is_archived: bool = False`, `created_at: datetime`, `updated_at: datetime`.
- AC-14: `Workflow.provenance` rejects any string outside the literal set above with a Pydantic validation error.
- AC-15: A `WorkflowDefinition` Pydantic model exists with fields: `topology: TopologySpec`, `agents: dict[str, AgentSpec]`, `execution_config: ExecutionConfigSpec`.
- AC-16: A `TopologySpec` Pydantic model exists with fields: `nodes: list[NodeSpec]`, `edges: list[EdgeSpec]`, `rules: list[str]`.
- AC-17: A `NodeSpec` Pydantic model exists with fields: `name: str`, `node_type: NodeType`, `agent_ref: str | None`, `is_convergence_point: bool`, `metadata: dict`.
- AC-18: An `EdgeSpec` Pydantic model exists with fields: `source: str`, `target: str`, `edge_type: EdgeType`, `bidirectional: bool`, `pattern: EdgePattern | None`, `metadata: dict`.
- AC-19: An `AgentSpec` Pydantic model exists with fields: `agent_model: ModelConfigSpec`, `name: str`, `goal: str`, `instruction: str`, `tools: list[str]`, `memory_retention: Literal["single_run","session","persistent"]`, `allowed_peers: list[str]`. The literal mirrors `marsys.agents.Agent`'s memory-retention contract exactly (NOT a Spren-side rename — this was caught during implementation review when the original AC's `"ephemeral"` value would have rejected real marsys files using `memory_retention="single_run"`). The model-config field is named `agent_model` (NOT `model_config` — that name is reserved by Pydantic v2's `BaseModel`). (resolved-12: `plan_config` is NOT a field on `AgentSpec`.)
- AC-20: A `ModelConfigSpec` Pydantic model exists mirroring every field of `marsys.models.ModelConfig` EXCEPT `api_key` (which is a credential value and lives only in the per-user secrets store, never in workflow definitions). `oauth_profile` (a profile *name*, not a credential) IS preserved on the spec. The spec validates without requiring any environment variable to be set, on a machine with no provider keys configured. Different agents in the same workflow can have different `provider` values, with credential resolution happening only at execution time (Session 04+) by looking up `<PROVIDER>_API_KEY` in the secrets store.
- AC-21: An `ExecutionConfigSpec` Pydantic model exists mirroring marsys's `ExecutionConfig` + `ConvergencePolicyConfig` shape; the convergence policy field accepts `Union[float, str, ConvergencePolicyConfigSpec]` with a discriminator.
- AC-22: `NodeType`, `EdgeType`, and `EdgePattern` enum values are lowercase strings exactly matching marsys's plain `Enum` values: `NodeType ∈ {"user","agent","system","tool"}`, `EdgeType ∈ {"invoke","notify","query","stream"}`, `EdgePattern ∈ {"alternating","symmetric"}`.
- AC-23: `NodeSpec.name` validation rejects any name whose lowercase form is in `RESERVED_NODE_NAMES` (i.e., `"user"`, `"system"`, `"tool"`); rejection is case-insensitive (`"User"`, `"USER"`, `"SYSTEM"`, etc. are all rejected). (resolved-10)
- AC-24: `RESERVED_NODE_NAMES` is sourced (imported) from `marsys.coordination.topology.core`, not redefined locally. (resolved-10; observable: shape of the validator follows marsys's constant.)
- AC-25: `WorkflowDefinition` enforces a `model_validator(mode='after')` cross-validation that rejects any model where a node with `node_type == "agent"` has an `agent_ref` not present as a key of `agents`. (resolved-11)
- AC-26: `WorkflowDefinition`'s `model_validator(mode='after')` rejects any model where any `edge.source` or `edge.target` does not match the `name` of some node in `topology.nodes`. (resolved-11)
- AC-27: Pydantic models reject missing required fields and wrong types with structured Pydantic validation errors.
- AC-28: `Workflow`, `WorkflowDefinition`, `TopologySpec`, `NodeSpec`, `EdgeSpec`, `AgentSpec`, `ModelConfigSpec`, and `ExecutionConfigSpec` each return a non-empty JSON Schema from `model_json_schema()`.
- AC-29: An `ErrorCode = Literal[...]` union is defined and includes at least `"WORKFLOW_NOT_FOUND"`, `"PYTHON_IMPORT_REJECTED"`, `"MIGRATION_FAILED"`, `"VALIDATION_FAILED"`, `"WORKFLOW_HAS_RUNS"`.

## Functional — Workflow CRUD endpoints

- AC-30: `POST /v1/workflows` with a valid body returns HTTP 201 and a `Workflow` JSON object with server-generated `id`, `created_at`, `updated_at` (these are NOT supplied by the client).
- AC-31: `POST /v1/workflows` defaults `provenance` to `"api"` when omitted from the request body.
- AC-32: `POST /v1/workflows` accepts an explicit `provenance` value from the allowed literal set and persists it as supplied.
- AC-33: `GET /v1/workflows` returns HTTP 200 with body shaped `{"items": [...], "next_cursor": <string|null>, "has_more": <bool>}`.
- AC-34: `GET /v1/workflows?provenance=<value>` filters items to only those matching the given provenance.
- AC-35: `GET /v1/workflows?archived=true` returns archived rows; the default list (no flag, or `archived=false`) excludes archived rows.
- AC-36: `GET /v1/workflows/{id}` returns HTTP 200 with the matching workflow when the id exists.
- AC-37: `GET /v1/workflows/{id}` returns HTTP 404 with body `{"error": {"code": "WORKFLOW_NOT_FOUND", "message": ..., "details": ...}}` when the id does not exist.
- AC-38: `PUT /v1/workflows/{id}` replaces the workflow; on success the response's `created_at` is preserved (unchanged from prior state) and `updated_at` is later than the prior `updated_at`.
- AC-39: `PATCH /v1/workflows/{id}` with `{"is_archived": true}` archives the workflow; it disappears from the default list and reappears with `?archived=true`. Fields not in the patch are unchanged.
- AC-40: `DELETE /v1/workflows/{id}` hard-deletes the row when no runs reference it; subsequent `GET /v1/workflows/{id}` returns 404.
- AC-41: `DELETE /v1/workflows/{id}` returns HTTP 409 with body `{"error": {"code": "WORKFLOW_HAS_RUNS", "message": ..., "details": ...}}` when at least one row in a `runs` table references the workflow id; the workflow row remains in the database.
- AC-42: All non-2xx responses from `/v1/workflows*` endpoints have body shape `{"error": {"code": <string>, "message": <string>, "details": <object>}}`.
- AC-43: Validation errors on request bodies return an error response with `code: "VALIDATION_FAILED"`.
- AC-44: `GET /openapi.json` returns HTTP 200 with a non-empty OpenAPI 3.x document.
- AC-45: The OpenAPI document includes paths for `/v1/workflows`, `/v1/workflows/{id}`, and `/v1/workflows/import-python`.

## Functional — Pagination

- AC-46: `GET /v1/workflows?limit=N` returns at most `N` items in `items`. The default `limit` is 20; the maximum is 100 (a request with `limit > 100` is either clamped to 100 or rejected — the response is observable either way).
- AC-47: When more workflows exist beyond the page, `next_cursor` is the string-form ULID of the last returned row and `has_more` is `true`.
- AC-48: When no further results exist, `next_cursor` is `null` and `has_more` is `false`.
- AC-49: `GET /v1/workflows?cursor=<ulid>&limit=N` returns the next page (rows where `id > cursor`, ordered by `id`, up to `limit` items); none of the returned rows appeared in the prior page.
- AC-50: With 25 workflows persisted and `limit=10`, three sequential page fetches return 10 + 10 + 5 items; the third page has `next_cursor=null` and `has_more=false`.
- AC-51: The cursor is the BARE string-form ULID — no HMAC signature is required for the cursor to be accepted. (resolved-2)

## Functional — Idempotency

- AC-52: `POST` / `PUT` / `PATCH` / `DELETE` requests under `/v1/workflows*` with an `Idempotency-Key` header cache the response (status, body, headers) for 24 hours. (resolved-3)
- AC-53: A second request with the same `(method, path, Idempotency-Key)` triple within 24 hours returns the cached body and status (no second side effect).
- AC-54: A request with the same `Idempotency-Key` value but a DIFFERENT method or path is treated as a fresh request (cache key includes method + path).
- AC-55: The idempotency cache survives daemon restart inside the 24-hour window; the cache is stored inside `<data-dir>/data/spren.db` in the `_idempotency` table — NOT a separate `*.db._idempotency` file. (resolved-3, resolved-4)
- AC-56: `Idempotency-Key` on `GET` requests is ignored.
- AC-57: Expired `_idempotency` rows are swept on FastAPI app startup. (Mirrors AC-9 from the idempotency-feature angle.)

## Functional — Python AST importer + rejection cases

- AC-58: `POST /v1/workflows/import-python` accepts a `multipart/form-data` upload with a `.py` file part and, on success, returns HTTP 201 with a `Workflow` JSON shape whose `provenance == "code_import"` and `provenance_metadata` contains `source_filename` and `sha256` (hex digest of the uploaded file content).
- AC-59: The importer parses input via Python AST inspection only — it does NOT call `exec`, `eval`, `compile`, `runpy`, or `importlib.import_module` against the uploaded source. (Observable via a fixture whose execution would set a global; the global remains unset after rejection or success.)
- AC-60: A valid Python workflow file in constructor-only style (`Agent(...)`, `Topology(...)`, `ExecutionConfig(...)`, `ModelConfig(...)` with literal kwargs) is materialized into a workflow row.
- AC-61: `valid_minimal.py` (constructor-style 2-agent linear pipeline) round-trips: every `Agent(...)` field (`name`, `goal`, `instruction`, `model_config=ModelConfig(...)`, `tools={...}`, `memory_retention`, `allowed_peers`) is preserved in the resulting `WorkflowDefinition.agents` entry; every `Node(...)` field (`name`, `node_type`, `agent_ref`) is preserved in `topology.nodes`; every `Edge(...)` field (`source`, `target`, `edge_type`, `bidirectional`, `pattern`) is preserved in `topology.edges`; the `ExecutionConfig(...)` kwargs are preserved in `execution_config`.
- AC-62: The importer translates marsys `Agent(model_config=ModelConfig(...))` into Spren `AgentSpec(agent_model=ModelConfig(...))` (kwarg renamed at the Spren API boundary).
- AC-63: The importer translates marsys `Agent(tools={"name1": <callable>, "name2": <callable>})` into Spren `AgentSpec(tools=["name1", "name2"])` — keys preserved as opaque strings; callables discarded.
- AC-64: A two-pass AST walker resolves module-level `Name = Constant` assignments and substitutes them when used as constructor arguments (`valid_with_constants.py` round-trips identically to `valid_minimal.py` field-by-field).
- AC-65: A file containing `exec(...)` is rejected with HTTP 422 and `error.code = "PYTHON_IMPORT_REJECTED"`. The human-readable message names the unsupported construct.
- AC-66: A file containing `eval(...)`, `compile(...)`, or `__import__(...)` is rejected with HTTP 422 + `PYTHON_IMPORT_REJECTED`.
- AC-67: A file containing class definitions, decorators on module-level definitions, comprehensions, or conditionals is rejected with HTTP 422 + `PYTHON_IMPORT_REJECTED`. **Bare `def`/`async def` at module scope is ALLOWED** — marsys's `tools=Dict[str, Callable]` contract requires users to define tool callables somewhere; the importer extracts only the dict KEYS from `tools={...}` and ignores function bodies, so the function body content is never inspected. Decorators on those functions, however, are rejected (they normally execute at import time and signal logic outside the supported subset).
- AC-68: A file using the dict-of-string-arrows topology DSL (e.g., `{"Start -> Researcher": ...}`) is rejected with HTTP 422 + `PYTHON_IMPORT_REJECTED`; the message names the dict-DSL construct.
- AC-69: A file containing f-strings in user-facing fields is rejected with HTTP 422 + `PYTHON_IMPORT_REJECTED`.
- AC-70: A file containing `Subscript` expressions inside `tools=` dict values is rejected with HTTP 422 + `PYTHON_IMPORT_REJECTED`.
- AC-71: A file with non-UTF-8 encoding is rejected with HTTP 422 + `PYTHON_IMPORT_REJECTED`.
- AC-72: A file larger than 1 MB is rejected at the FastAPI handler boundary BEFORE AST parse (fast-fail) with HTTP 422 + `PYTHON_IMPORT_REJECTED`. (resolved-14)
- AC-73: Rejection responses for the import endpoint include `details = {"reason": <short-tag>, "line": <N>}` (line number provided when applicable to the construct).
- AC-74: A malformed file missing a `Topology(...)` construct returns HTTP 422 with a clear error.
- AC-75: When a file is rejected, no row is inserted into `workflows` (database state is unchanged across the rejection).

## Functional — Type generation pipeline + smoke

- AC-76: A Node script exists at `apps/web/scripts/generate-types.mjs` (Node native ESM) that, when run, produces `apps/web/src/lib/api-types.generated.ts`. The extension is `.mjs`, NOT `.ts` / `.tsx`. (resolved-5)
- AC-77: The type-gen script spawns the sidecar in a transient subprocess, parses the `spren-ready: port=N token=T data-dir=P` line from stdout, fetches `/openapi.json` with the auth token, writes a snapshot to `apps/web/openapi-snapshot.json`, runs `openapi-typescript` to emit the generated `.ts`, then sends `shutdown\n` over the sidecar's stdin and waits for clean exit.
- AC-78: `apps/web/openapi-snapshot.json` is gitignored (entry present in `.gitignore` and the file is NOT committed). (resolved-9)
- AC-79: `apps/web/src/lib/api-types.generated.ts` IS committed and tracked in git.
- AC-80: There is exactly ONE TypeScript-type emitter in the build pipeline: `openapi-typescript`. `json-schema-to-typescript` is NOT installed and NOT invoked anywhere. (resolved-1)
- AC-81: The type-gen script is wired into `apps/web/package.json`'s `prebuild` hook ONLY; it is NOT in `predev`. (resolved-9)
- AC-82: The generated `api-types.generated.ts` is non-empty and parses as valid TypeScript (verifiable via `typescript`'s `createSourceFile`).
- AC-83: The generated `api-types.generated.ts` exposes a path entry for `/v1/workflows` with a reachable response schema for the GET 200 response (`paths['/v1/workflows'].get.responses[200].content['application/json'].schema`).
- AC-84: A small JSON sample matching the `Workflow` Pydantic shape typechecks against the generated TypeScript type when assigned to a typed variable (no TS errors).
- AC-85: `pnpm-workspace.yaml` declares `peerDependencyRules.allowedVersions["openapi-typescript>typescript"] = "^6"`. The rule lives at the workspace root, NOT in `apps/web/package.json` — pnpm 10 ignores `pnpm.peerDependencyRules` declared in package-level `package.json` for workspace packages and emits a warning to that effect. The `pnpm.overrides` field is NOT used to relax this peer. (resolved-7; corrected during implementation when pnpm warned that the package-level location is ignored.)
- AC-86: `pnpm --filter @marsys/spren-web build` succeeds with the type-gen pipeline wired in.
- AC-87: The Vitest type-gen smoke test lives at `apps/web/tests/generate-types.test.ts` (NOT under `apps/web/scripts/`). (resolved-6)
- AC-88: No Python helper script for emitting JSON Schema (e.g., `spren.scripts.export_json_schema`) is created in this session — the single-emitter approach makes it unnecessary.

## Functional — Frontend placeholder UI

- AC-89: The placeholder workflow page in `apps/web` lists workflows fetched from `GET /v1/workflows`, showing each workflow's name and a provenance badge.
- AC-90: The placeholder UI exposes a "Create" affordance that submits a `name` + `definition` JSON to `POST /v1/workflows`, then refreshes the list.
- AC-91: The placeholder UI exposes a "Delete" affordance per row that calls `DELETE /v1/workflows/{id}`, then refreshes the list.
- AC-92: The placeholder UI exposes an "Import Python file" affordance that performs a multipart `POST /v1/workflows/import-python` with the chosen file, then refreshes the list.
- AC-93: The placeholder UI uses inline `style={{ ... }}` only — no Tailwind / shadcn / radix / cmdk / Geist CSS / `@xyflow/react` is added. (Observable: no Tailwind classes present, no `tailwind.config.*` added.)
- AC-94: The placeholder UI resolves the API base URL via the existing `resolveBaseUrl()` helper from `apps/web/src/lib/api.ts` (no hardcoded `http://127.0.0.1:8765`).
- AC-95: The placeholder UI reads the auth token via the existing `useCapabilities()` provider from `apps/web/src/providers/capabilities.tsx` (does not read directly from the URL fragment in route components).
- AC-96: A workflow created via the UI persists across a page reload (preserving the auth token in the URL fragment) — the row reappears in the list.
- AC-97 (manual-only): Visiting `http://localhost:5173/#token=<token>` shows the placeholder workflow page with no console errors.
- AC-98 (manual-only): The same flows (create, delete, import) work in the Tauri webview via `just dev-desktop`, with the auth token sourced from `window.__SPREN_AUTH__` rather than the URL fragment.

## Functional — Bootstrap response extension

- AC-99: `GET /v1/bootstrap` response includes `endpoints.workflows` with the value `"/v1/workflows"`.

## Functional — Auth gating

- AC-100: Every endpoint under `/v1/workflows*` (list, create, read, replace, patch, delete, import-python) returns HTTP 401 when called without an `Authorization: Bearer <token>` header. (NOT 422 — the response status is observable.)
- AC-101: An invalid bearer token on `/v1/workflows*` returns HTTP 401.
- AC-102: `GET /healthz` returns HTTP 200 without any authentication header (regression — must remain unauth'd).
- AC-103: The workflows router is constructed via a factory `make_workflows_router(require_auth, db)` (or equivalent) taking the auth dep as a constructor parameter; auth is applied at the router level (no module-level globals dependent on `server.create_app()`'s call order).

## Functional — End-to-end flows

- AC-104: A Playwright E2E test at `apps/web/tests/e2e/workflow-crud.spec.ts` starts the FastAPI app + Vite dev server, navigates to the placeholder page with a token in the URL fragment, creates a workflow via the UI, imports a workflow from a Python file, verifies both appear with correct provenance badges, deletes each, and verifies the list is empty.

## Functional — Architecture-doc fixes (this PR)

- AC-105: `docs/architecture/spren/03-api-design.md` (idempotency cache section, around line 164) describes the cache as living inside `<data-dir>/data/spren.db` in the `_idempotency` table — NOT a separate `*.db._idempotency` file. (resolved-4)
- AC-106: `docs/architecture/spren/03-api-design.md` (type-generation pipeline section, around lines 119-128) describes the pipeline as single-emitter (`openapi-typescript` only); `json-schema-to-typescript` is removed from the description. (resolved-1, resolved-4)
- AC-107: `docs/architecture/spren/08-design-principles.md` SP-005 reference to `datamodel-code-generator` is fixed (it's Python-only, doesn't emit TS).
- AC-108: Any link broken by file moves in this session is updated.

## Non-functional — Code quality + SP-rule audits

- AC-109: No mocks of in-codebase features in product code. `grep -rn 'Mock\|mock.patch\|vi.mock\|MagicMock' packages/spren/src/ apps/web/src/` returns zero hits in non-test files. (SP-007)
- AC-110: No backward-compatibility branches or shims (`if version`, `# legacy`, `# TODO: remove` markers as conditional logic) in product code; the first migration is the only schema. (SP-006)
- AC-111: No file under `packages/framework/src/marsys/coordination/`, `agents/`, `models/`, or `environment/` is modified by this session. (SP-001)
- AC-112: No file in `packages/framework/` imports anything from `spren.*`. (SP-018)
- AC-113: At session end-state, no TypeScript file under `apps/web/src/lib/` (other than `api-types.generated.ts`) declares interfaces or types named after the Pydantic models `Workflow`, `WorkflowDefinition`, `TopologySpec`, `NodeSpec`, `EdgeSpec`, `AgentSpec`, or `ExecutionConfigSpec`. (SP-005; intermediate states during Step 5 may still hold inline placeholder interfaces — the auditor evaluates the end-state.)
- AC-114: `json-schema-to-typescript` is NOT in `apps/web/package.json` `dependencies` or `devDependencies`. (resolved-1, SP-005)
- AC-115: Stale comments referencing `datamodel-code-generator` in `apps/web/src/lib/api.ts:1` and `apps/web/src/types/api.ts:1` are removed or updated to reflect the actual generator (`openapi-typescript`).
- AC-116: `apps/web/src/types/api.ts` either re-exports from `api-types.generated.ts` or is deleted (no other purpose post-Session-02).

## Non-functional — Workspace + tooling

- AC-117: `packages/spren/pyproject.toml` pins `python-ulid >= 3.0,<4.0`; the unrelated `ulid-py` package is NOT a dependency. (resolved-13)
- AC-118: ULID values are produced via the `ulid` import name (distribution name `python-ulid`); no use of `ulid_py` is present. (resolved-13; observable indirectly via AC-10 and via dependency manifest.)
- AC-119: Any new `Justfile` recipe uses scoped `pnpm --filter @marsys/spren-web ...` for pnpm invocations (NOT bare `web`).
- AC-120: Any new `Justfile` recipe uses `uv run --package spren ...` for Python invocations.

## Non-functional — Tests required

- AC-121: Pytest unit tests at `packages/spren/tests/test_models_workflow.py` cover Pydantic acceptance / rejection, `RESERVED_NODE_NAMES` rejection (each member, case-insensitive), `WorkflowDefinition` cross-validation (agent_ref ↔ agents, edge endpoints ↔ nodes), JSON Schema export non-empty for each model.
- AC-122: Pytest unit tests at `packages/spren/tests/test_storage_migrations.py` cover migrations runner idempotency, `workflows` + `_idempotency` table creation, partial-failure leaves no row in `_migrations`.
- AC-123: Pytest unit tests at `packages/spren/tests/test_routes_workflows.py` cover each CRUD endpoint, archived filter, DELETE 409 path, pagination 25-row scenario, auth 401 path, `Idempotency-Key` cache + same-key-different-method-or-path treated as fresh.
- AC-124: Pytest unit tests at `packages/spren/tests/test_importers_python.py` cover both valid fixtures (round-trip preservation), all rejection fixtures (dict-DSL, exec, dynamic topology, too-large), the no-exec/eval/compile property (via side-effect-free fixture), and a missing-topology malformed case.
- AC-125: Pytest integration tests at `packages/spren/tests/integration/test_workflow_crud_e2e.py` exercise the full FastAPI app with real SQLite at a temp path, round-trip a complex workflow (multiple agents + edges + execution_config), round-trip a Python file import, and verify `/openapi.json` is reachable + contains `/v1/workflows*` paths.
- AC-126: Playwright E2E test at `apps/web/tests/e2e/workflow-crud.spec.ts` exists per AC-104.
- AC-127: Vitest test at `apps/web/tests/generate-types.test.ts` covers existence + non-emptiness + valid-TS-parse + path-reachability + Workflow-shape typecheck.
- AC-128: `just test` exits 0 with all suites passing.
- AC-129: Framework regression baseline preserved: `pytest packages/framework/tests` collects 841 tests; pass / fail / skip / error counts do not regress from the 764 / 20 / 43 / 14 baseline. The 20 failures + 14 errors are pre-existing fixture issues (require `OPENROUTER_API_KEY`) and are out of scope.

## Non-functional — Manual-verify items

- AC-130 (manual-only): `just install && just dev` boots cleanly; no Python or pnpm errors in either prefixed log stream.
- AC-131 (manual-only): First-launch creates `<data-dir>/data/spren.db` and runs the migration; subsequent launches do NOT re-run it (verified by inspecting the `_migrations` table).
- AC-132 (manual-only): `pnpm --filter @marsys/spren-web build` succeeds and produces a populated `apps/web/src/lib/api-types.generated.ts`. Running `git status` after the build shows no untracked snapshot file (i.e., `apps/web/openapi-snapshot.json` is gitignored).

## Out of scope (tests for these in this session would be wrong)

- Run execution (no runs exist in v0.3 until Session 04). Tests requiring a live runtime tool registry, agent invocation, or a populated `runs` table are out of scope. The DELETE-409 path may be exercised by stub-inserting a row into `runs` if that table exists, OR by a small mock at the count-runs query — surfaced if needed.
- Real visual builder UI (canvas, node editing, drag-and-drop) — Session 03.
- Tailwind / shadcn / radix-ui / cmdk / Geist / `@xyflow/react` design system — Session 03.
- TanStack Query data-fetching wiring on the workflow page — Session 03.
- `plan_config` field on `AgentSpec` — dropped from v0.3, revisit in Session 04. (resolved-12)
- Tool-name → callable runtime registry — Session 04. v0.3 stores tool names as opaque strings.
- Snapshot-on-run-start linkage from runs to workflows — Session 04+.
- The `runs`, `files`, `schedules`, `channels`, `secrets`, `settings` SQLite tables — later sessions.
- Pause/resume endpoints (v0.4 framework feature) — out of v0.3 entirely.
- `POST /v1/workflows/{id}/lint` and `POST /v1/workflows/{id}/duplicate` endpoints — not in this session's scope.
- TUI workflow CRUD — v0.4.
- The dict-of-string-arrows topology DSL parser — explicitly rejected; users rewrite to constructor style or use the visual builder.
- HMAC-signed pagination cursors — explicitly rejected for v0.3 (auth-token-gated localhost-only scope makes signing ceremonial; regenerate-secret-on-restart failure mode is unnecessary cost). (resolved-2)
- Future cleanup PR that replaces `spren.models.topology` / `agent` / `execution_config` with imports from `marsys.coordination.topology.serialize` — that comes after Framework Session 04 ships, NOT this session.
- Modifying any TRUNK-CRITICAL framework file (per SP-001).
- Spren type imports inside `packages/framework/` (per SP-018).
