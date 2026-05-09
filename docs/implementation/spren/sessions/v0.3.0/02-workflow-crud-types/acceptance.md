# Acceptance criteria — Session 02: Workflow CRUD + Pydantic + TypeScript Type Generation

Frozen at 2026-05-09T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

## Functional

### Database initialization & migrations

- AC-1: On first server start, a SQLite database file is created at `<data-dir>/data/spren.db`, where `<data-dir>` is the platform-resolved per-user data directory (`platformdirs.user_data_dir("spren")`).
- AC-2: The parent directory of `spren.db` is created automatically if it does not exist.
- AC-3: A `_migrations` table exists in `spren.db` after first start, recording applied migration IDs, names, and `applied_at` timestamps.
- AC-4: Migrations are forward-only and idempotent: running the migrations runner twice in a row leaves the schema unchanged and does not re-apply already-recorded migrations.
- AC-5: After the first migration runs, a `workflows` table exists in `spren.db`.
- AC-6: The `workflows` table has all of these columns: `id` (TEXT PRIMARY KEY), `name` (TEXT NOT NULL), `description` (TEXT), `definition` (TEXT NOT NULL), `definition_version` (INTEGER NOT NULL), `provenance` (TEXT NOT NULL), `provenance_metadata` (TEXT), `is_archived` (INTEGER NOT NULL DEFAULT 0), `created_at` (TEXT NOT NULL), `updated_at` (TEXT NOT NULL).
- AC-7: `created_at` and `updated_at` values stored in `workflows` are ISO 8601 UTC with microsecond precision (not second-precision).
- AC-8: `id` values stored in `workflows` are ULIDs.

### Pydantic model contracts

- AC-9: A `Workflow` Pydantic model is defined whose JSON shape contains exactly: `id`, `name`, `description`, `definition`, `definition_version`, `provenance`, `provenance_metadata`, `is_archived`, `created_at`, `updated_at`.
- AC-10: The `provenance` field on `Workflow` only accepts the literal values `visual_builder`, `meta_agent`, `code_import`, `template`, `api`. Any other string value is rejected with a validation error.
- AC-11: A `WorkflowDefinition` Pydantic model contains a `topology`, an `agents` map (string keys), and an `execution_config`.
- AC-12: A `TopologySpec` Pydantic model contains `nodes: list[NodeSpec]`, `edges: list[EdgeSpec]`, and `rules: list[str]`.
- AC-13: A `NodeSpec` Pydantic model contains at minimum: `name`, `node_type`, `agent_ref`, `is_convergence_point`, `metadata`.
- AC-14: An `EdgeSpec` Pydantic model contains at minimum: `source`, `target`, `edge_type`, `bidirectional`, `pattern`, `metadata`.
- AC-15: An `AgentSpec` Pydantic model exposes the model-config field as `agent_model` (NOT `model_config`, since that name is reserved by Pydantic v2's `BaseModel`).
- AC-16: An `AgentSpec` Pydantic model contains at minimum: `agent_model`, `name`, `goal`, `instruction`, `tools`, `memory_retention`, `allowed_peers`, `plan_config`.
- AC-17: An `ExecutionConfigSpec` Pydantic model exists and mirrors the marsys `ExecutionConfig` fields.
- AC-18: Each Spren Pydantic model (`Workflow`, `WorkflowDefinition`, `TopologySpec`, `NodeSpec`, `EdgeSpec`, `AgentSpec`, `ExecutionConfigSpec`) produces a non-empty JSON Schema when `model_json_schema()` is called.
- AC-19: `NodeType` and `EdgeType` enum values used by `NodeSpec` / `EdgeSpec` exactly match the string values of `marsys.coordination.topology.core.NodeType` and `EdgeType` (same casing).

### REST endpoints — auth

- AC-20: Every endpoint under `/v1/workflows` (list, create, read, replace, patch, delete, import-python) returns HTTP 401 when called without an `Authorization: Bearer <token>` header.
- AC-21: `GET /healthz` returns 200 without any authentication header (regression — must remain unauth'd).

### REST endpoints — workflow CRUD

- AC-22: `POST /v1/workflows` with a valid body creates a new workflow row, returns HTTP 201, and the response body is a `Workflow` JSON object.
- AC-23: `POST /v1/workflows` populates `id`, `created_at`, and `updated_at` server-side; the client does not supply them.
- AC-24: `POST /v1/workflows` defaults `provenance` to `"api"` when the request body does not specify a `provenance`.
- AC-25: `POST /v1/workflows` accepts an explicit `provenance` value from the allowed set and persists it as supplied.
- AC-26: `GET /v1/workflows` returns HTTP 200 with a JSON body shaped `{"items": [...], "next_cursor": <string|null>, "has_more": <bool>}`.
- AC-27: `GET /v1/workflows?provenance=<value>` filters returned items to only those matching the given provenance.
- AC-28: `GET /v1/workflows?archived=true` returns archived workflows; the default list (no archived filter, or `archived=false`) excludes archived workflows.
- AC-29: `GET /v1/workflows/{id}` returns HTTP 200 and the matching workflow when the id exists.
- AC-30: `GET /v1/workflows/{id}` returns HTTP 404 with body `{"error": {"code": "WORKFLOW_NOT_FOUND", "message": ..., "details": {...}}}` when the id does not exist.
- AC-31: `PUT /v1/workflows/{id}` replaces a workflow; on success the returned `updated_at` is later than the prior `updated_at`, and `created_at` is preserved unchanged.
- AC-32: `PATCH /v1/workflows/{id}` with `{"is_archived": true}` archives the workflow; the row's `is_archived` becomes `1` (true) without otherwise mutating fields not in the patch.
- AC-33: `DELETE /v1/workflows/{id}` hard-deletes the workflow row when no runs reference it; subsequent `GET /v1/workflows/{id}` returns 404.
- AC-34: `DELETE /v1/workflows/{id}` returns HTTP 409 with body `{"error": {"code": "WORKFLOW_HAS_RUNS", "message": ..., "details": {...}}}` when one or more rows in the `runs` table reference the workflow id; the workflow row remains in the database (delete is a no-op).

### REST endpoints — pagination

- AC-35: `GET /v1/workflows?limit=N` returns at most N items in `items` (default 20, max 100).
- AC-36: When more workflows exist than fit in one page, `next_cursor` is a non-null string and `has_more` is `true`.
- AC-37: Following the returned `next_cursor` via `GET /v1/workflows?cursor=<next_cursor>&limit=N` returns the next batch of items, none of which appeared in the prior page.
- AC-38: When the final page is reached, `next_cursor` is `null` and `has_more` is `false`.
- AC-39: A tampered cursor (any modification to a previously-issued cursor string) returns HTTP 400.

### REST endpoints — error format

- AC-40: All non-2xx responses from `/v1/workflows*` endpoints have a JSON body shaped `{"error": {"code": <string>, "message": <string>, "details": <object>}}`.
- AC-41: Validation errors on request bodies return an error response with `code: "VALIDATION_FAILED"`.

### REST endpoints — Python file import

- AC-42: `POST /v1/workflows/import-python` accepts a `multipart/form-data` upload with a file part containing a `.py` file.
- AC-43: A valid Python workflow file (constructor-only style: `Agent(...)`, `Topology(...)`, `ExecutionConfig(...)`, `ModelConfig(...)` with literal kwargs) is materialized into a workflow row, returned with HTTP 201.
- AC-44: The materialized workflow has `provenance == "code_import"`.
- AC-45: The materialized workflow's `provenance_metadata` JSON contains `source_filename` and `sha256` (where `sha256` is the hex digest of the uploaded file content).
- AC-46: A Python file containing module-level constants (e.g., `AGENT_NAME = "researcher"`) referenced by `Name` from constructor kwargs is accepted; the parser resolves the `Name` references against the constant table.
- AC-47: A Python file containing the dict-of-string-arrows topology DSL (e.g., `{"Start -> Researcher": ...}`) is rejected with HTTP 422 and `code: "PYTHON_IMPORT_REJECTED"`.
- AC-48: A Python file containing `exec(...)`, `eval(...)`, `compile(...)`, or `__import__(...)` is rejected with HTTP 422 and `code: "PYTHON_IMPORT_REJECTED"` without executing the file.
- AC-49: A Python file containing class definitions, function definitions, decorators, comprehensions, or conditionals is rejected with HTTP 422 and `code: "PYTHON_IMPORT_REJECTED"`.
- AC-50: A Python file containing f-strings in user-facing fields is rejected with HTTP 422 and `code: "PYTHON_IMPORT_REJECTED"`.
- AC-51: A Python file larger than 1 MB is rejected with HTTP 422 and `code: "PYTHON_IMPORT_REJECTED"`.
- AC-52: A Python file with non-UTF-8 encoding is rejected with HTTP 422 and `code: "PYTHON_IMPORT_REJECTED"`.
- AC-53: Rejection responses for Python import include `details.reason` (a short tag) and `details.line` (the line number where the offending construct appears, when applicable).
- AC-54: Rejection messages name the unsupported construct in human-readable text.
- AC-55: When a file is rejected, no row is inserted into `workflows` (database state is unchanged).

### Idempotency

- AC-56: `Idempotency-Key` is honored on `POST`, `PUT`, `PATCH`, and `DELETE` requests under `/v1/workflows*` (and ignored on `GET`). When the same `(method, path, Idempotency-Key)` triple has been seen within the last 24 hours, the server returns the cached prior response and does NOT re-perform the mutation.
- AC-57: The idempotency cache persists in `<data-dir>/data/spren.db` in an `_idempotency` table; cache hits work across server restart as long as the original entry is within its 24-hour window. Expired rows are swept on server startup.

### OpenAPI

- AC-58: `GET /openapi.json` returns HTTP 200 with a non-empty OpenAPI 3.x document.
- AC-59: The OpenAPI document includes paths for `/v1/workflows`, `/v1/workflows/{id}`, and `/v1/workflows/import-python`.

### Bootstrap response

- AC-60: `GET /v1/bootstrap` response includes `endpoints.workflows` with the value `"/v1/workflows"`.

### TypeScript type generation

- AC-61: After running the type generation pipeline, the file `apps/web/src/lib/api-types.generated.ts` exists and is non-empty.
- AC-62: After running the type generation pipeline, the file `apps/web/src/lib/types.generated.ts` exists and is non-empty.
- AC-63: Both generated files parse as valid TypeScript (compile cleanly under the workspace's TypeScript compiler with `--noEmit`).
- AC-64: The generated `Workflow` TypeScript type round-trips successfully against a JSON sample of a valid `Workflow` (the sample parses as the generated type without TypeScript errors).
- AC-65: At session end-state, no TypeScript file in `apps/web/src/lib/` (other than the two `*.generated.ts` files) declares interfaces or types named after the Pydantic models `Workflow`, `WorkflowDefinition`, `TopologySpec`, `NodeSpec`, `EdgeSpec`, `AgentSpec`, or `ExecutionConfigSpec`. Intermediate states during Step 5 may still hold inline placeholder interfaces; the auditor evaluates the end-state of the session, not commits in between.

### Frontend placeholder UI

- AC-66: The placeholder workflow page in `apps/web` lists workflows fetched from `GET /v1/workflows`, showing each workflow's name and a provenance badge.
- AC-67: The placeholder UI exposes a "Create" affordance that submits a `name` + `definition` JSON to `POST /v1/workflows`, then refreshes the list.
- AC-68: The placeholder UI exposes a "Delete" affordance per row that calls `DELETE /v1/workflows/{id}`, then refreshes the list.
- AC-69: The placeholder UI exposes an "Import Python file" affordance that performs a multipart `POST /v1/workflows/import-python` with the chosen file, then refreshes the list.
- AC-70: The placeholder UI resolves the API base URL via the existing `resolveBaseUrl()` helper (does not hardcode `http://127.0.0.1:8765`).
- AC-71: The placeholder UI reads the auth token via the existing capabilities provider (does not read directly from the URL fragment in route components).
- AC-72: A workflow created via the UI persists across a page reload (preserving the auth token in the URL fragment).

### End-to-end flows

- AC-73: An end-to-end test starts the FastAPI app + Vite dev server, navigates to the placeholder page with a token in the URL fragment, creates a workflow via the UI, imports a workflow from a Python file, verifies both appear with correct provenance badges, deletes each, and verifies the list is empty.

## Non-functional

### Security / auth

- AC-74: The Python importer never executes the uploaded file (no `exec`, `eval`, `compile`, `runpy`, `importlib.import_module` against the upload path); parsing is AST-only.
- AC-75: Cursor strings are HMAC-signed with a per-launch secret; the secret is regenerated on each server restart (alongside the auth token).

### Persistence integrity

- AC-76: Migrations run inside a transaction — a migration that raises mid-execution leaves the schema in the pre-migration state.

### Test baseline

- AC-77: The framework regression baseline holds: `pytest packages/framework/tests` collects 841 tests with the same pass/fail/skip/error counts as Session 01's recorded baseline (764 passed / 20 failed / 43 skipped / 14 errors). New work in this session does not regress the baseline.

## Out of scope

- The visual workflow builder UI (canvas, node editor, design system) — Session 03.
- Tailwind, shadcn, radix, cmdk, Geist CSS, MSW — deferred to Session 03.
- TanStack Query data-fetching wiring on the workflow page — Session 03.
- Run execution endpoints (`/v1/runs*`) and the SSE stream — Session 04.
- The `runs`, `files`, `schedules`, `channels`, `secrets`, `settings` SQLite tables — later sessions.
- Pause/resume endpoints (v0.4 framework feature) — out of v0.3 entirely.
- The `POST /v1/workflows/{id}/lint` and `POST /v1/workflows/{id}/duplicate` endpoints — not listed in this session's acceptance scope.
- The dict-of-string-arrows topology DSL parser — explicitly rejected; users rewrite to constructor style or use the visual builder.
- Modifying any TRUNK-CRITICAL framework file (per SP-001).
- Spren type imports inside `packages/framework/` (per SP-018).

## Notes

This acceptance file is frozen at the timestamp at the top of the document. Subsequent changes to the brief that affect observable behavior should be appended here as `AC-NN-amend` entries with a date, NOT applied by editing existing criteria. The auditor reads this file as the contract for the session.
