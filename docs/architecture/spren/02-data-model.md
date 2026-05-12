# 02 — Data Model

All persistent data lives under the per-user data directory resolved via `platformdirs`:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/spren/` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/spren/` |
| Windows | `%LOCALAPPDATA%/spren/` |

Inside that directory:

```
spren/
├── data/
│   ├── spren.db                  # SQLite — workflows, runs, schedules, channels, settings, secrets
│   ├── files/
│   │   └── {file_id}/...             # uploaded user files (raw bytes)
│   └── runs/                         # framework's FileStorageBackend root for pause/resume snapshots
│       └── {run_id}/                 # Spren's run_id maps 1:1 to the framework's session_id
│           ├── trace.ndjson          # append-only trace events (one JSON per line)
│           ├── workflow.json         # frozen snapshot of workflow at run start
│           ├── snapshot.json         # (v0.4) pause/resume StateSnapshot; only present for paused runs
│           └── artifacts/            # files produced by the run (screenshots, agent outputs)
├── sandbox/
│   ├── shared/memory/                # markdown KB
│   ├── shared/skills/                # skill catalog
│   └── teams/<slug>/                 # team-scoped sandboxes (later releases)
├── logs/spren.log
└── runtime/auth-token                # current session's per-launch token (0600)
```

Subsequent paths in this document use `<data-dir>` as shorthand for the platform-resolved data directory.

## SQLite schema

All tables include `created_at` and `updated_at` (UTC, microsecond precision). All IDs are ULIDs unless stated.

### `workflows`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PRIMARY KEY | ULID |
| `name` | TEXT NOT NULL | User-visible label |
| `description` | TEXT | Optional |
| `definition` | TEXT NOT NULL | JSON: full workflow spec (topology + agents + execution_config) |
| `definition_version` | INTEGER NOT NULL | Schema version of the JSON in `definition` (bumps on migration) |
| `provenance` | TEXT NOT NULL | One of `visual_builder` / `meta_agent` / `code_import` / `template` / `api`. Annotates how the workflow was authored. UI filters and badges read this. |
| `provenance_metadata` | TEXT | JSON. For `code_import`: source filename + sha256. For `meta_agent`: which conversation produced it. Optional for other provenances. |
| `is_archived` | INTEGER NOT NULL DEFAULT 0 | Soft-delete; kept for run-history references |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC |
| `updated_at` | TEXT NOT NULL | ISO 8601 UTC |

`definition` JSON shape (mirrors marsys topology + execution config):

```json
{
  "topology": {
    "nodes": [{"name": "Researcher", "node_type": "agent", "agent_ref": "agent_id"}, ...],
    "edges": [{"source": "Researcher", "target": "Writer", "edge_type": "invoke"}, ...],
    "rules": [...]
  },
  "agents": {
    "agent_id_1": {
      "name": "Researcher",
      "agent_model": {...ModelConfig fields...},
      "goal": "...",
      "instruction": "...",
      "tools": ["search_web", "browse_url"],
      "memory_retention": "session",
      "allowed_peers": ["Writer"],
      "plan_config": {...}
    }
  },
  "execution_config": {
    "convergence_timeout": 300.0,
    "user_interaction": "web",
    "response_format": "json",
    ...
  }
}
```

`node_type`, `edge_type`, and `edge.pattern` values are lowercase strings matching `marsys.coordination.topology.core.NodeType` / `EdgeType` / `EdgePattern`: `node_type ∈ {user, agent, system, tool}`, `edge_type ∈ {invoke, notify, query, stream}`, `pattern ∈ {alternating, symmetric}` (or null). The agent's model-config field is named `agent_model` rather than `model` because Pydantic v2 reserves the attribute name `model_config`; storage JSON shape mirrors the Pydantic mirror.

### `runs`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PRIMARY KEY | ULID; doubles as the marsys `session_id` |
| `workflow_id` | TEXT NOT NULL | FK → `workflows.id` (no cascade — keep history if workflow deleted) |
| `status` | TEXT NOT NULL | `queued` / `running` / `paused` / `succeeded` / `failed` / `cancelled` (`paused` ships in v0.4) |
| `task_input` | TEXT NOT NULL | JSON: `{text: "...", attachments: ["file_id_1"]}` |
| `started_at` | TEXT | NULL until `running` |
| `finished_at` | TEXT | NULL until terminal state |
| `total_steps` | INTEGER | Filled on completion |
| `total_duration_ms` | INTEGER | Filled on completion |
| `total_tokens_input` | INTEGER NOT NULL DEFAULT 0 | Sum across all GenerationEvents |
| `total_tokens_output` | INTEGER NOT NULL DEFAULT 0 | |
| `total_cost_usd` | REAL NOT NULL DEFAULT 0.0 | Sum from rate table |
| `final_response` | TEXT | JSON-encoded final response if any |
| `error` | TEXT | NULL on success |
| `trigger` | TEXT NOT NULL | `manual` / `scheduled` / `webhook` / `messenger:telegram` / etc. |
| `created_at` | TEXT NOT NULL | |
| `updated_at` | TEXT NOT NULL | |

Trace events for a run live at `<data-dir>/data/runs/{run_id}/trace.ndjson` — NOT in SQLite. SQLite holds aggregates only.

The frozen workflow snapshot lives at `<data-dir>/data/runs/{run_id}/workflow.json` per SP-009.

### `files`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PRIMARY KEY | ULID |
| `original_name` | TEXT NOT NULL | User-uploaded filename |
| `mime_type` | TEXT NOT NULL | Detected on upload |
| `size_bytes` | INTEGER NOT NULL | |
| `path` | TEXT NOT NULL | Absolute path under `<data-dir>/data/files/{id}/` |
| `sha256` | TEXT NOT NULL | Content hash; for deduping if needed later |
| `created_at` | TEXT NOT NULL | |

Files are referenced by `id` from `runs.task_input.attachments` and from agent tool inputs.

### `schedules` (later release)

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PRIMARY KEY | ULID |
| `workflow_id` | TEXT NOT NULL | FK → `workflows.id` |
| `cron_expr` | TEXT NOT NULL | Standard cron with seconds (APScheduler format) |
| `timezone` | TEXT NOT NULL | IANA tz name |
| `task_template` | TEXT NOT NULL | JSON: `{text: "...", attachments: []}` |
| `is_enabled` | INTEGER NOT NULL DEFAULT 1 | |
| `last_fired_at` | TEXT | |
| `created_at` | TEXT NOT NULL | |

### `channels` (later release)

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PRIMARY KEY | ULID |
| `kind` | TEXT NOT NULL | `telegram` / `discord` / `slack` / `whatsapp` |
| `display_name` | TEXT NOT NULL | User label |
| `config` | TEXT NOT NULL | JSON, kind-specific (bot tokens, channel IDs — encrypted at rest) |
| `meta_agent_enabled` | INTEGER NOT NULL DEFAULT 0 | If set, meta-agent is reachable from this channel |
| `allowlist` | TEXT NOT NULL | JSON array of platform user IDs allowed to interact (per SP-008 / security) |
| `is_enabled` | INTEGER NOT NULL DEFAULT 1 | |
| `created_at` | TEXT NOT NULL | |

### `secrets`

| Column | Type | Notes |
|--------|------|-------|
| `key_name` | TEXT PRIMARY KEY | e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN` |
| `keychain_ref` | TEXT | If OS keychain available, opaque ref; the value lives in the keychain |
| `ciphertext` | BLOB | Fallback: AES-GCM encrypted with a master key derived from a user-set passphrase |
| `kdf_salt` | BLOB | If using ciphertext path |
| `created_at` | TEXT NOT NULL | |
| `updated_at` | TEXT NOT NULL | |

See [`07-security.md`](./07-security.md) for the storage policy.

### `settings`

| Column | Type | Notes |
|--------|------|-------|
| `key` | TEXT PRIMARY KEY | e.g., `meta_agent.model`, `cost.daily_budget_usd` |
| `value` | TEXT NOT NULL | JSON-encoded |
| `updated_at` | TEXT NOT NULL | |

## Trace event format (NDJSON)

The framework's NDJSON streaming writer (Framework v0.3 Session 01) writes one JSON object per line, append-only, flushed per closed span. The wire format is the framework's contract — Spren reads it as-is. Schema-versioned, lowercase `kind`, float epoch-second timestamps, ULIDs for span/trace IDs.

Example shape (one closed `generation` span):

```json
{"schema_version":1,"ts":1714220645.123,"kind":"generation","span_id":"01J9X4ABCDEFGHJKMP","parent_span_id":"01J9X4ABCDEFGHJKMR","trace_id":"01J9X4ABCDEFGHJKMN","name":"Generation: claude-opus-4.7","start_time":1714220641.000,"end_time":1714220645.123,"duration_ms":4123.0,"status":"ok","attributes":{"model_name":"claude-opus-4.7","provider":"anthropic","prompt_tokens":1234,"completion_tokens":567,"reasoning_tokens":120,"response_time_ms":4321,"finish_reason":"stop","has_thinking":true,"has_tool_calls":false}}
```

`kind` ∈ `{execution, branch, step, generation, tool}` (the five `Span.kind` values; validation, user-interaction, and final-response are recorded as events on the surrounding step / execution span, not as separate kinds). `ts` is the writer's emission timestamp as a float epoch-second (not the span's `start_time`). `start_time` / `end_time` mirror `Span.start_time` / `Span.end_time` (`float`, epoch seconds via `time.time()`); clients format for display. `span_id` / `parent_span_id` / `trace_id` are ULIDs (uppercase Crockford-base32, monotonic-within-process).

Two non-span line types are emitted by the writer and must be filtered by readers consuming individual spans:
- `kind == "stream_event"` — diagnostic (e.g. `{"event": "dropped_span", "dropped_span_count": N}`).
- `kind == "stream_completed"` — terminal marker, exactly once on close. Missing marker on EOF means the writer crashed.

The framework-side AG-UI translator (`marsys.transport.aggui`, Framework v0.3 Session 06) is a separate `EventBus` consumer that translates framework lifecycle events into AG-UI events. Spren v0.3 Session 04 wraps the framework's `AGUIEventStream(orchestra, run_id) -> AsyncIterator[AGUIEvent]` adapter in an SSE HTTP endpoint at `GET /v1/runs/{id}/events`; it does not consume `trace.ndjson` for live streaming. The trace file is for the run inspector + cold reads (reconnect replay).

## Migrations

Located at `packages/spren/src/spren/storage/migrations/<version>__<name>.py`. Forward-only (no down-migration). Each migration runs in a transaction. Schema version tracked in a `_migrations` SQLite table (id, name, applied_at).

Per SP-006: the post-migration code must only know the new shape. Forward-only migrations are not "backward compatibility" — they are one-shot transformations.

## Storage volume planning

- Workflows: ~5KB each in JSON. Even 1000 workflows = 5MB. Negligible.
- Runs: trace.ndjson can be 50KB to 10MB depending on verbosity. Default retention: keep last 200 runs OR 7 days of runs, whichever is larger. Older trace files moved to `<data-dir>/data/archived/runs/` and gzipped; the SQLite row stays.
- Files: bound by user upload behavior. Soft cap proposed at 5GB; user-configurable.
- Settings + schedules + channels: bytes.
