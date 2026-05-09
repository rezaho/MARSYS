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
│   └── runs/
│       └── {run_id}/
│           ├── trace.ndjson          # append-only trace events (one JSON per line)
│           ├── workflow.json         # frozen snapshot of workflow at run start
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
    "nodes": [{"name": "Researcher", "node_type": "AGENT", "agent_ref": "agent_id"}, ...],
    "edges": [{"source": "Researcher", "target": "Writer", "edge_type": "INVOKE"}, ...],
    "rules": [...]
  },
  "agents": {
    "agent_id_1": {
      "name": "Researcher",
      "model": {...ModelConfig fields...},
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

Each line is a single JSON object conforming to a Spren-internal schema that mirrors AG-UI events. Example shape:

```json
{"ts":"2026-04-27T12:30:45.123Z","trace_id":"01J...","span_id":"01J...","parent_span_id":"01J...","kind":"GENERATION","name":"opus-4.7 call","attrs":{"model":"claude-opus-4.7","provider":"anthropic","prompt_tokens":1234,"completion_tokens":567,"reasoning_tokens":120,"response_time_ms":4321,"finish_reason":"stop","has_thinking":true,"has_tool_calls":false}}
```

Append-only writer flushes per event. Reader streams via SSE for live UI; can also tail-read for late subscribers.

## Migrations

Located at `packages/spren/src/spren/storage/migrations/<version>__<name>.py`. Forward-only (no down-migration). Each migration runs in a transaction. Schema version tracked in a `_migrations` SQLite table (id, name, applied_at).

Per SP-006: the post-migration code must only know the new shape. Forward-only migrations are not "backward compatibility" — they are one-shot transformations.

## Storage volume planning

- Workflows: ~5KB each in JSON. Even 1000 workflows = 5MB. Negligible.
- Runs: trace.ndjson can be 50KB to 10MB depending on verbosity. Default retention: keep last 200 runs OR 7 days of runs, whichever is larger. Older trace files moved to `<data-dir>/data/archived/runs/` and gzipped; the SQLite row stays.
- Files: bound by user upload behavior. Soft cap proposed at 5GB; user-configurable.
- Settings + schedules + channels: bytes.
