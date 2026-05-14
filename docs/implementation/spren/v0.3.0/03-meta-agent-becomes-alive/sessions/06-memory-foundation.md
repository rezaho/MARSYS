# Spren Session 06 — Memory Foundation

> Session plan. The implementer reads this as the primary source of truth for what Session 06 ships, how the sandbox + session log + markdown KB + indexer + memory CLI + pending facts queue fit together, and what's in vs out of scope. Captures bundle position, scope boundaries, dependency check, files-to-CREATE / MODIFY / DELETE, the user journeys that anchor Bundle 03's first demo gate, wireframes for the small UI lift (no UI — Session 06 is all backend + CLI), data-model considerations, the locked decisions, polish items the implementer addresses in-session, success criteria, and open research items the implementer resolves in-flight.
>
> Status: **draft — subject to user redirect**. Acceptance criteria are frozen separately at [`./06-memory-foundation/acceptance.md`](./06-memory-foundation/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Architectural anchors (read before coding):
- [`../../../../architecture/spren/10-memory-architecture.md`](../../../../architecture/spren/10-memory-architecture.md) — the locked memory architecture. §1 Tier layout, §2 write paths (incl. §2.D escape valve), §3 read tools, §4 predicate metadata, §5 supersession algorithm, §11 indexer, §12 procedural memory.
- [`../../../../architecture/spren/09-meta-agent.md`](../../../../architecture/spren/09-meta-agent.md) §"Sandbox and filesystem permissions" — the two-layer sandbox model (OS-level outer envelope + application-layer inner permission tiers).
- [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — especially SP-014 (atomic scratchpad), SP-015 (untrusted-channel writes never live-touch memory), SP-016 (markdown is source of truth), SP-017 (forget tombstones never deletes), SP-023 (generic-runtime vs Spren-specific boundary).
- [`../../../../../tmp/spren/research/06-memory-foundations/00-synthesis.md`](../../../../../tmp/spren/research/06-memory-foundations/00-synthesis.md) — the four-research synthesis that produced the locked deltas (M1-M18 + S1-S5).
- [`../../../../../tmp/spren/research/06-memory-foundations/01-memory-architecture-deltas.md`](../../../../../tmp/spren/research/06-memory-foundations/01-memory-architecture-deltas.md) — May-2026 literature deltas.

---

## 1. Bundle position + tier

- **Bundle**: 03 — Meta-agent becomes alive (Sessions 06 + 07 + 08). Demo-able outcome: user has a working meta-agent that reads workflows + traces + memory, suggests-with-confirm on writes, dispatches workflows, respects hard rails, ticks heartbeat, tracks budget. The bond mechanism produces persona-evolution proposals over time. Memory plumbing on its own (Session 06) is too thin to demo; bundling all three so the meta-agent breathes is the demo gate.
- **Session 06 scope**: the memory foundation. Sandbox (both layers) + Tier 2 session log + Tier 3 markdown KB scaffold + deterministic markdown→SQL indexer + `pending_facts` queue + `spren memory` CLI. Sessions 07 + 08 build on top.
- **Tier**: CRITICAL. Sandbox is the security boundary; the markdown→SQL indexer is forward-mode (a bad supersession decision can't be unwritten retroactively); SP-014 / SP-015 / SP-016 / SP-017 are load-bearing for Sessions 07-08 and everything downstream. Full meta-process pipeline.
- **Approval gate**: peer Stage 0 conversation already complete (this brief). Researcher + Designer + Validator/Critic + Fact-checker + Synthesis run before implementation. Multi-checkpoint user review per CRITICAL tier.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 01 (foundation) | shipped | FastAPI sidecar, auth, capabilities, `make_auth_dependency` factory, route-level auth pattern, CORS regex. Session 06 reuses all of this. |
| Spren Session 02 (workflow CRUD + types + Python import) | shipped | Pydantic types + SQLite + migrations runner pattern. Session 06 adds new tables (`events`, `facts`, `entities`, `pending_facts`) using the same migration pattern. |
| Spren Session 03 (visual builder) | shipped | Provides the design system. Session 06 has no UI surface beyond the CLI; the design system isn't directly consumed but the cmdk navigation pattern is the model for the future memory-browse surface (v0.4 lands the UI). |
| Spren Session 04 (run execution + AG-UI streaming + cost) | shipping in parallel | NDJSON trace.ndjson per run. Session 06's session log (Tier 2) coexists with the per-run trace (different scopes — Tier 2 is daemon-wide; trace.ndjson is per-run). |
| Spren Session 05 (run inspection) | shipping in parallel | `GET /v1/runs/{id}/trace` reads trace.ndjson. Same independence as Session 04. |
| Framework Session 01 (NDJSON streaming tracing writer) | shipped | Not directly consumed by Session 06; consumed by Session 04. |
| Framework Session 02 (`TelemetrySink` + `SecretRedactor`) | shipped on `feature/tracing-streaming` | Not consumed in v0.3 by Spren; v0.4 `SprenTelemetrySink` consumer. The `SecretRedactor` from this work is the reference for our Spren-side redaction in the session log. |
| Framework Session 03 (pause/resume) | shipped on `feature/tracing-streaming` | Not consumed in v0.3 by Spren; v0.4 `v0.4-29` consumer. |
| `platformdirs` | already a dep | Resolves the per-user data directory (`<data-dir>`); the sandbox tree lives under it. |
| `watchdog` | new dep — must be added | Cross-platform file-watcher (`inotify` / `FSEvents` / `ReadDirectoryChangesW`). The indexer's primary mechanism. Latest stable as of session start. |
| `python-frontmatter` | new dep — must be added | YAML frontmatter parser. Battle-tested; used by Hugo, Jekyll-equivalents, lots of static-site tooling. |
| `bubblewrap` (Linux) / `sandbox-exec` profile (macOS) / Windows AppContainer + Job Object | system tooling | Per-platform; the OS-level outer envelope. Not Python deps. The sandbox launcher invokes them via subprocess. macOS: `sandbox-exec -f profile.sb`. Linux: `bwrap --bind ... --tmpfs /tmp -- ...`. Windows: requires a Rust shim that wraps the subprocess in `CreateProcess` with a restricted token + Job Object. |

Session 06 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). All work lands inside `packages/spren/src/spren/` and adds two new top-level subpackages (`spren/runtime/` per SP-023 generic side; `spren/memory/` per SP-023 Spren-specific side).

## 3. SP-023 boundary — what goes in `spren/runtime/` vs `spren/memory/`

This section is load-bearing. The boundary lives in code from this session onward.

**Generic always-on runtime — `packages/spren/src/spren/runtime/`** (no Spren-specific imports beyond approved deps + `packages/framework/`):
- `runtime/sandbox/__init__.py` — the `Sandbox` class (application-layer wrapper) and the `SandboxLauncher` interface (OS-level outer envelope).
- `runtime/sandbox/wrapper.py` — application-layer permission tiers: `Sandbox(root: Path, agent_role: AgentRole)` with `read_file`, `write_file`, `list_dir`, `exists` methods that resolve paths and enforce per-tier rules. **Note: SandboxFilesystem methods are exposed as the agent's typed tools later in Session 07/08.** Session 06 builds the wrapper itself; tool-level invocation is downstream.
- `runtime/sandbox/launcher.py` — `SandboxLauncher` ABC + per-platform implementations (`BubblewrapLauncher` for Linux, `SandboxExecLauncher` for macOS, `AppContainerLauncher` for Windows). Each takes `cmd, cwd, timeout_s, allow_network, sandbox_root` and returns `SubprocessResult`. Generic API; per-platform body. Used by Session 07's `execute_shell` tool, by workflow agents (CodeAgent), by the indexer's full-scan path if it ever invokes a subprocess.
- `runtime/storage/sessionlog.py` — `SessionLog` writer + reader. Append-only SQLite events table + JSONL daily files under `<sandbox-root>/logs/events/YYYY/MM/DD.jsonl`. Generic event-shaped log; no Spren-specific event types. The reader supports BM25 over event text via SQLite FTS5.
- `runtime/storage/atomic.py` — atomic-write primitives (`write-temp + fsync(fd) + os.replace + fsync(parent_dir_fd)`). Used by the session log writer, the indexer's SQL writes, the future scratchpad writes (SP-014).
- `runtime/indexer/__init__.py` — `MarkdownIndexer` ABC. Generic markdown→SQL indexer that takes a directory + a parser strategy + a SQL backend.
- `runtime/indexer/watcher.py` — wraps `watchdog`. Generic file-watcher with debounce + idempotent reindex callback. Recovers from `inotify`-class delivery loss via 5-min full-scan fallback.
- `runtime/indexer/parser.py` — markdown parser primitives: YAML frontmatter, fenced-div (`::: name ... :::`) blocks, prose body extraction. No knowledge of fact-blocks or activity-logs specifically — those are *uses* of the parser.

**Spren-specific layer — `packages/spren/src/spren/memory/`**:
- `memory/__init__.py` — module init. Re-exports the Spren-specific facts API.
- `memory/predicate_metadata.py` — the predicate → `{volatility, cardinality, default_trust_threshold}` mapping (renamed from architecture's "volatility.py"). Starter table covers ~30 common predicates.
- `memory/sandbox_layout.py` — the Spren-specific sandbox tree: `~/.spren/sandbox/shared/memory/{people,projects,procedures,decisions,profile,personas,journal,vault,security}/`, plus the per-archetype path resolution. Consumes generic `runtime/sandbox/Sandbox` to enforce permissions.
- `memory/indexer.py` — Spren-specific application of the generic `MarkdownIndexer`: knows about the `::: facts` block syntax + per-entity-type frontmatter (`type: project | person | ...`), produces `entities` + `facts` SQL rows. Composes with `runtime/indexer/parser.py` for the syntactic primitives.
- `memory/storage.py` — SQL schema for `entities` + `facts` + `pending_facts` + `_migrations` tables. Forward-only migrations.
- `memory/cli.py` — the `spren memory` command catalog: `show`, `edit`, `remember`, `forget`, `why`, `journal`, `verify-index`, `rebuild-index`, `rebuild-embeddings`. Click-based; reuses the auth token from the running daemon (or refuses with a clear message if the daemon isn't running).
- `memory/bootstrap.py` — first-run scaffold that creates the directory tree, seeds `_self.md` from the user's settings (name, pronouns), creates `personas/main.yaml` with the chosen archetype defaults (Session 07 wires the picker; Session 06 ships the seeding mechanism + a default archetype at "Vesper" if nothing's chosen yet — this default goes away when Session 07's picker lands).

**Cross-checks for SP-023 enforcement:**
- A linter rule (or PR-review check; we ship the linter rule for v0.3) at the package level: `runtime/` files cannot `from spren.memory ...`. They CAN import from `spren.runtime.*`, `marsys.*`, and approved third-party deps.
- Each new module's session-level acceptance criteria include "no Spren-specific imports in runtime/ files" verified by `grep -rE "from spren\.memory|import spren\.memory" packages/spren/src/spren/runtime/` returning empty.
- Documentation: every file in `runtime/` has a docstring noting it's generic; every file in `memory/` consumes `runtime/` via the documented interface.

## 4. What ships in Session 06

### 4.1 Sandbox foundations

**Application-layer wrapper** (`spren/runtime/sandbox/wrapper.py`):

```python
class AgentRole(StrEnum):
    main_agent = "main"
    team_manager = "team"
    working_instance = "instance"
    workflow_agent = "workflow"
    user = "user"  # CLI / vim direct edits

class Sandbox:
    def __init__(self, root: Path, agent_role: AgentRole, scope: str | None = None):
        """
        root: the resolved <sandbox-root> (e.g., <data-dir>/sandbox/)
        agent_role: the calling agent's role (drives permission tier)
        scope: optional sub-scope (team slug, instance id) for narrower roles
        """
    def read_file(self, path: str | Path) -> bytes: ...
    def write_file(self, path: str | Path, data: bytes) -> None: ...  # atomic
    def list_dir(self, path: str | Path) -> list[Path]: ...
    def exists(self, path: str | Path) -> bool: ...
    def resolve(self, path: str | Path) -> Path: ...  # abs path; raises PathTraversalError on escape
```

Permission tiers per `09-meta-agent.md` §"Sandbox and filesystem permissions":
- `main_agent`: full RW under `shared/`, RW under any team or instance scope it explicitly addresses, R on the audit log.
- `team_manager` (with `scope=<team-slug>`): RW under `teams/<team-slug>/`, R on `shared/`.
- `working_instance` (with `scope=<instance-id>`): RW under `teams/<team>/instances/<id>/scratchpad/`, R on shared and team memory.
- `workflow_agent` (with `scope=<run-id>`): RW under `runs/<run-id>/artifacts/`, R on shared.
- `user`: full RW (CLI + vim are the user's authority surface).

Path-resolution algorithm: every `read_file`/`write_file` resolves the path via `Path(root, path).resolve()`, then verifies the resolved path is contained inside `root` (no `..` escape, no absolute-path injection, no symlink escape). On escape: `raise PathTraversalError`. Verified via `pathlib.Path.resolve(strict=False)` + parent-dir check.

**OS-level outer envelope** (`spren/runtime/sandbox/launcher.py`):

```python
@dataclass
class SubprocessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_ms: int

class SandboxLauncher(ABC):
    @abstractmethod
    def run(
        self,
        cmd: list[str],
        cwd: Path,
        sandbox_root: Path,
        timeout_s: int = 30,
        allow_network: bool = False,
        env: dict[str, str] | None = None,
    ) -> SubprocessResult: ...

class BubblewrapLauncher(SandboxLauncher): ...    # Linux
class SandboxExecLauncher(SandboxLauncher): ...   # macOS
class AppContainerLauncher(SandboxLauncher): ...  # Windows
class NoopLauncher(SandboxLauncher): ...          # Docker mode (container is the envelope) + dev override
```

Each per-platform implementation:
- Bind-mounts only `<sandbox-root>` (rw) + a tmpdir (rw) + system libraries (ro) + `cwd` if outside `<sandbox-root>` and explicitly granted by the caller.
- Drops network unless `allow_network=True`. Linux: `bwrap --unshare-net`. macOS: deny `network*` operations in the profile. Windows: AppContainer's network capability omitted from the token.
- Time-bounds via `timeout_s`. CPU + memory bounds via cgroups (Linux), Job Object limits (Windows), and `ulimit`-equivalent on macOS.
- Returns `SubprocessResult` synchronously. Async variant (`run_async`) is v0.4 if needed; v0.3 is sync only since the only callers are CLI and (in Session 07) `execute_shell`.

`launcher_for_platform()` factory returns the right subclass based on `sys.platform` + container detection. Docker mode (detected via `/proc/1/cgroup` reading or env var `SPREN_RUNNING_IN_CONTAINER=1`) returns `NoopLauncher`. Dev override `SPREN_SANDBOX_NOOP=1` also returns `NoopLauncher` for tests where setting up bubblewrap is impractical.

### 4.2 Tier 2 session log

**SQLite events table** (in `<data-dir>/data/spren.db`, alongside the existing `workflows` + `runs` + `_idempotency` tables):

```sql
CREATE TABLE events (
    id              TEXT PRIMARY KEY,         -- ULID
    ts              REAL NOT NULL,            -- epoch seconds, float
    kind            TEXT NOT NULL,            -- e.g., "user_message", "agent_response", "tool_call", "fact_proposed", "live_commit", "consolidation_completed", "discord_event", "voice_drift"
    actor           TEXT,                     -- agent_id or "user" or channel id
    channel         TEXT,                     -- "cli" / "web" / "telegram" / "tauri" / "internal"
    payload         TEXT NOT NULL,            -- JSON; kind-specific shape
    schema_version  INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_events_ts ON events(ts);
CREATE INDEX idx_events_kind_ts ON events(kind, ts);
CREATE INDEX idx_events_actor_ts ON events(actor, ts);

CREATE VIRTUAL TABLE events_fts USING fts5(payload, content='events', content_rowid='rowid');
```

**JSONL daily files** at `<data-dir>/sandbox/logs/events/YYYY/MM/DD.jsonl`. One JSON object per line, append-only. Mirror of the SQLite events table; serves cold storage + offline analysis. The session-log writer writes both atomically (SQLite first; JSONL append; on JSONL failure, the SQLite row stays — JSONL is recoverable from SQLite via `SessionLog.export_jsonl(date_range)`).

The session log writer is generic (no Spren-specific event-type knowledge). Spren-specific code records events by calling `session_log.write(SessionEvent(kind="fact_proposed", ...))`. Event kinds are documented in `spren/memory/events.py` (Spren-specific) but the writer doesn't validate kind values — that's the caller's responsibility.

### 4.3 Tier 3 markdown KB scaffold

The directory tree from `10-memory-architecture.md` §1 Tier 3, materialized on first daemon launch via `memory/bootstrap.py`:

```
<data-dir>/sandbox/shared/memory/
├── people/
│   └── _self.md                      # seeded from user settings (name, pronouns)
├── projects/                         # empty
├── procedures/                       # empty (skills come in v0.4)
├── decisions/                        # empty
├── profile/
│   ├── identity.md                   # seeded with frontmatter + empty facts block
│   ├── constraints.md                # seeded with sensitive: true frontmatter + empty
│   ├── values.md                     # empty
│   ├── preferences.md                # empty
│   ├── capabilities.md               # empty
│   ├── rhythm.md                     # empty
│   └── goals.md                      # empty
├── personas/
│   ├── main.yaml                     # written by Session 07's archetype picker
│   ├── pending_persona_changes/      # empty dir
│   └── journal/                      # empty dir
├── journal/                          # empty (date-partitioned files created on first consolidation)
├── active_context.md                 # empty
├── active_todos.md                   # empty
├── vault/                            # empty; reserved for v0.4 (encrypted; not parsed by indexer)
└── security/
    └── poisoning_patterns.yaml       # empty in v0.3, seeded v0.4
```

**Bootstrap mechanism:** on daemon launch, `memory/bootstrap.py:ensure_kb_exists(sandbox: Sandbox)` checks the tree's existence and creates missing subdirs + seed files. Idempotent. Run from `server.py`'s `lifespan` startup hook before any agent code touches memory.

**`_self.md` seed:** if `<data-dir>/data/spren.db` settings table has `user.name` and `user.pronouns`, the bootstrap fills them in. If not, `_self.md` is written with empty frontmatter and a comment line: `# This is your identity record. Tell Spren your name and it'll be filled in.` First user-message that contains a `name` extraction will populate this via the live-commit path.

**`personas/main.yaml` seed:** Session 06 ships a placeholder persona (Vesper-archetype defaults from the archetype research) so the daemon has *some* persona to read on day one before Session 07's picker ships. When Session 07 lands, the picker overwrites this on first run.

### 4.4 Markdown→SQL indexer

**Generic primitives** (`spren/runtime/indexer/`):

```python
# parser.py
@dataclass
class Frontmatter:
    data: dict[str, Any]
    raw: str

@dataclass
class FencedDiv:
    name: str           # the name after ::: (e.g., "facts", "todo", "activity-log")
    body: str           # raw body text between ::: name and :::
    line_start: int
    line_end: int

def parse_frontmatter(content: str) -> Frontmatter | None: ...
def parse_fenced_divs(content: str) -> list[FencedDiv]: ...
def extract_prose_body(content: str) -> str: ...  # everything outside frontmatter + fenced divs

# watcher.py
class FileWatcher:
    def __init__(self, root: Path, on_change: Callable[[Path], None], debounce_ms: int = 100): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

# __init__.py — the generic indexer ABC
class MarkdownIndexer(ABC):
    @abstractmethod
    def reindex_file(self, path: Path) -> ReindexResult: ...
    @abstractmethod
    def full_scan(self) -> FullScanResult: ...
```

**Spren-specific application** (`spren/memory/indexer.py`):

```python
class SprenMarkdownIndexer(MarkdownIndexer):
    """Watches <sandbox>/shared/memory/, parses ::: facts blocks +
    YAML frontmatter, produces entities + facts SQL rows."""

    def __init__(self, sandbox: Sandbox, db: SQLiteConnection): ...

    def reindex_file(self, path: Path) -> ReindexResult:
        content = self.sandbox.read_file(path).decode()
        frontmatter = parse_frontmatter(content)
        if frontmatter is None:
            return ReindexResult.skipped("no frontmatter")
        if frontmatter.data.get("type") not in ENTITY_TYPES:
            return ReindexResult.skipped("not an entity file")

        fenced = parse_fenced_divs(content)
        facts_block = next((f for f in fenced if f.name == "facts"), None)
        prose = extract_prose_body(content)

        entity = self._entity_from_frontmatter(frontmatter, source_path=path)
        facts = self._facts_from_block(facts_block, entity_id=entity.id) if facts_block else []

        # Format validation — reject malformed
        for f in facts:
            if not self._validate_fact(f):
                self._log_format_error(path, f)
                return ReindexResult.format_error(f)

        # Compute content_hash for each fact
        for f in facts:
            f.content_hash = sha256(f"{f.asserted_at}|{f.source}|{f.value}".encode()).hexdigest()

        # UPSERT
        with self.db.transaction():
            self.db.upsert("entities", entity, key="id")
            current_fact_ids = {f.id for f in facts}
            for f in facts:
                self.db.upsert("facts", f, key="id")
            # Mark facts that USED to be in this file but are no longer
            stale = self.db.query(
                "SELECT id FROM facts WHERE entity_id=? AND id NOT IN (?)",
                (entity.id, current_fact_ids))
            for s in stale:
                self.db.update("facts", s.id, {"status": "orphaned"})

        # Re-index prose for recall (Tier 4 stub in v0.3 — see §4.5)
        self.discovery_index.upsert_prose(path, prose, entity_ref=entity.id)

        return ReindexResult.indexed(entity.id, len(facts))
```

**Properties:**
- Deterministic. Same markdown input → same SQL state.
- Cheap. Pure parsing + SQL UPSERTs.
- Fast. File-system events trigger reindex within milliseconds.
- Resilient. 5-min periodic full-scan catches anything missed.
- Format-validated. Malformed fact-blocks log to `<data-dir>/sandbox/logs/consolidation_errors.md` and don't commit.

### 4.5 Tier 4 discovery index — stub in v0.3

The full hybrid (BM25 + vector + reranker) discovery index is a v0.4 deliverable. Session 06 ships:
- A `DiscoveryIndex` interface (`spren/runtime/indexer/discovery.py`) with `upsert_prose(path, prose, entity_ref)`, `search(query, k=5) -> list[Snippet]`.
- A no-op default implementation (`NoopDiscoveryIndex`) that records prose into a SQLite `prose_chunks` FTS5 table (BM25 only, no vectors). This is enough for Session 08's `recall` tool to function at v0.3 scale (small KB).
- `embedding_model_id` field on every chunk row, defaulted to `null` in v0.3 (no embeddings yet); v0.4's vector path populates it.

`recall(query, k=5)` against the no-op index returns BM25-ranked prose snippets. v0.4 swaps the implementation for real hybrid retrieval; the caller-side API doesn't change.

### 4.6 `pending_facts` queue

```sql
CREATE TABLE pending_facts (
    id              TEXT PRIMARY KEY,         -- ULID
    entity_id       TEXT,                     -- nullable; consolidation pass resolves to a real entity
    predicate       TEXT NOT NULL,
    value           TEXT NOT NULL,            -- JSON for non-string values
    source          TEXT NOT NULL,            -- "user_direct" | "meta_agent_observed" | "sub_agent_branch_summary" | "channel_*" | "webhook_inbound"
    source_event_id TEXT NOT NULL,            -- FK to events.id
    confidence      REAL DEFAULT 0.5,
    queued_at       REAL NOT NULL,            -- epoch seconds
    schema_version  INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_pending_facts_queued_at ON pending_facts(queued_at);
CREATE INDEX idx_pending_facts_source ON pending_facts(source);
```

Session 06 ships:
- The schema + migration.
- `pending_facts.write(fact)` writer — used by Session 07's LLM-judged extraction step (Path B in `10-memory-architecture.md` §2.B).
- `pending_facts.read_since(timestamp)` reader — used by Session 08's consolidation pass.
- A 4-hour-interval watcher (`PendingFactsWatcher` from `09-meta-agent.md` §3) that surfaces a `ConsolidationDue` event to the meta-agent inbox when the queue exceeds a threshold (default: 50 facts) — Session 06 ships the watcher + threshold check; Session 07 wires it into the actual inbox.

Note: the consolidation pass itself ships in Session 08, not Session 06. Session 06 ships the queue + the writer + the reader; the consumer is Session 08.

### 4.7 `spren memory` CLI

Click-based CLI surface, lives in `spren/memory/cli.py`. Commands per `10-memory-architecture.md` §9 + the additions from research synthesis:

```bash
spren memory show <entity-id-or-path>      # cat the entity / category file
spren memory edit <entity-id-or-path>      # opens in $EDITOR
spren memory remember "<fact>"             # appends to the right file with provenance line
spren memory forget <fact-id>              # tombstones the fact (SP-017)
spren memory restore <fact-id>             # reverses a tombstone
spren memory why <fact-id-or-claim>        # walks the supersession chain back to source events
spren memory journal [<date>]              # opens today's (or specified date's) journal entry
spren memory verify-index                  # walks every fact, confirms source file + content_hash
spren memory rebuild-index                 # rebuild SQL projection from markdown (one-shot full-scan)
spren memory rebuild-embeddings            # rebuild Tier 4 from markdown (no-op until v0.4 vectors)
```

Each command:
- Reads the auth token from the running daemon (default token path `<data-dir>/runtime/auth-token`; chmod 0600 per Session 01's pattern). Refuses with a clear error if the daemon isn't running.
- Calls into the daemon's CLI bridge (a small UNIX socket at `<data-dir>/runtime/spren.sock` OR direct DB+filesystem access if the daemon isn't running for read-only commands like `show` / `why`).
- Writes operations always require the daemon (so the write goes through the indexer + SQL projection cleanly).

`spren memory remember "<fact>"` parses the fact via a small heuristic: if the fact is `"<predicate>: <value>"`, it routes to the right entity / category file via the bubble-up routing rules table (§8.2 of memory architecture). If the fact is freeform prose, it lands in today's `journal/YYYY-MM-DD.md` with a `> source: user, <ts>` provenance line. The user can edit the file directly to clean it up.

`spren memory forget <fact-id>` requires confirmation (`Are you sure? (y/N)`) because tombstones are auditable but reversible only via `restore`.

### 4.8 New deps

`pyproject.toml` for `packages/spren/` adds:

```toml
[project]
dependencies = [
    # ... existing ...
    "watchdog ~= 6.0",        # file watcher (latest stable; verify version at implementation time)
    "python-frontmatter ~= 1.1",
    "click ~= 8.2",           # CLI framework
]
```

Implementer verifies versions at session start via `pip index versions <pkg>`. The `~=` operator pins major+minor; patch updates allowed.

### 4.9 Tests

- **Vitest unit:** N/A — Session 06 has no UI surface. CLI tests use `click.testing.CliRunner`.
- **Pytest unit:** `Sandbox` permission tier matrix (every role × every path → expected allow/deny); `SandboxLauncher` per-platform smoke (file-system isolation; network drop verification; timeout enforcement); markdown frontmatter + fenced-div parser exhaustive (well-formed, malformed, edge cases like nested fenced divs, BOM, CRLF); `SprenMarkdownIndexer.reindex_file` against fixture files (well-formed entity, malformed `::: facts`, file with no frontmatter — skipped, file with stale facts — orphaned); content_hash computation; `pending_facts` queue read-write; predicate_metadata lookups.
- **Pytest integration:** end-to-end memory CLI flow (`spren memory remember`, then `show`, then `why`); `verify-index` against a deliberately corrupted fact-block; bootstrap creates the full directory tree on first launch; format validator rejects bad YAML and logs to `consolidation_errors.md`; full-scan catches a file that watchdog missed (simulated by writing while watchdog is off, then triggering scan).
- **Per-platform sandbox tests:** Linux+macOS+Windows CI matrix. The OS-level outer envelope tests run only on the matching platform; tests are marked `@pytest.mark.platform_linux` etc.; the no-op launcher path tests run everywhere.
- **Manual-verify checklist** (implementer self-verification before claiming done): vim a fact-block, see SQL update within 1s; `spren memory why` walks back to source event; `forget` + `restore` roundtrip preserves history; deliberately tampered fact (edit SQL behind the indexer's back) surfaces as `IndexDriftDetected` on next `verify-index`.

## 5. What is OUT of scope

| Out of scope in Session 06 | Lands in |
|---|---|
| Consolidation pass (extract → adjudicate → TMS gate → routing → conflict resolution → apply) | Session 08 |
| Supersession algorithm (cardinality dispatch, confidence floor, trust check) | Session 08 |
| TMS gate at consolidation | Session 08 |
| Read tools surfaced to the agent (`read_file`, `grep`, `lookup_facts`, `recall`, `verify_fact`, `confirm_with_user`) | Session 07 (basic shape) + Session 08 (full set) |
| Escape-valve write tool (`commit_fact_now`) | Session 08 |
| Persona file write at first run via archetype picker | Session 07 |
| Persona-evolution mechanism | Session 08 |
| Vector embeddings (sqlite-vec) | v0.4 |
| Hybrid retrieval (BM25 + vector + reranker) | v0.4 |
| Procedural memory consolidation from BranchSummary | v0.4 |
| Sub-instance scratchpad (per-instance dir under `teams/<slug>/instances/<id>/`) | v0.4 (sub-instances ship in v0.4) |
| Channel-sourced fact extraction | v0.4 (channels ship in v0.4) |
| `vault/` encryption | v0.4 |
| `poisoning_patterns.yaml` seeding | v0.4 (sub-instances drive the threat) |
| UI surface for memory browse / edit | v0.4 |

Anything labeled out-of-scope renders as "not available" or empty placeholder where the route exists (per SP-019).

## 6. Files to CREATE / MODIFY / DELETE in Session 06

### To CREATE — `spren/runtime/` (generic side, SP-023)

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/runtime/__init__.py` | Module init. Re-exports the public runtime API. |
| `packages/spren/src/spren/runtime/sandbox/__init__.py` | Re-exports `Sandbox`, `AgentRole`, `PathTraversalError`, `SandboxLauncher`, `launcher_for_platform`. |
| `packages/spren/src/spren/runtime/sandbox/wrapper.py` | `Sandbox` application-layer wrapper + permission tiers. |
| `packages/spren/src/spren/runtime/sandbox/launcher.py` | `SandboxLauncher` ABC + `BubblewrapLauncher`, `SandboxExecLauncher`, `AppContainerLauncher`, `NoopLauncher`. |
| `packages/spren/src/spren/runtime/sandbox/profiles/sandbox.sb` | macOS sandbox-exec profile. |
| `packages/spren/src/spren/runtime/sandbox/profiles/bwrap_args.py` | Linux bubblewrap arg builder. |
| `packages/spren/src/spren/runtime/storage/__init__.py` | Re-exports `SessionLog`, `SessionEvent`, atomic write primitives. |
| `packages/spren/src/spren/runtime/storage/sessionlog.py` | `SessionLog` SQLite + JSONL writer + reader. |
| `packages/spren/src/spren/runtime/storage/atomic.py` | atomic-write primitives (write-temp + fsync + replace + parent-dir-fsync). |
| `packages/spren/src/spren/runtime/indexer/__init__.py` | `MarkdownIndexer` ABC + `ReindexResult`, `FullScanResult`. |
| `packages/spren/src/spren/runtime/indexer/parser.py` | Frontmatter + fenced-div + prose-body parser primitives. |
| `packages/spren/src/spren/runtime/indexer/watcher.py` | `FileWatcher` (watchdog wrapper with debounce + 5-min full-scan fallback). |
| `packages/spren/src/spren/runtime/indexer/discovery.py` | `DiscoveryIndex` interface + `NoopDiscoveryIndex` (FTS5 BM25-only). |

### To CREATE — `spren/memory/` (Spren-specific side, SP-023)

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/memory/__init__.py` | Module init. Re-exports SprenMarkdownIndexer, predicate_metadata, bootstrap, cli. |
| `packages/spren/src/spren/memory/predicate_metadata.py` | predicate → `{volatility, cardinality, default_trust_threshold}` mapping. ~30 starter predicates. |
| `packages/spren/src/spren/memory/sandbox_layout.py` | Spren KB tree paths + per-archetype path resolution. |
| `packages/spren/src/spren/memory/indexer.py` | `SprenMarkdownIndexer` (subclass of generic indexer; knows `::: facts` syntax + entity-type frontmatter). |
| `packages/spren/src/spren/memory/storage.py` | SQL schema for `entities`, `facts`, `pending_facts`. |
| `packages/spren/src/spren/memory/events.py` | Spren-specific event-kind constants + payload Pydantic types. |
| `packages/spren/src/spren/memory/bootstrap.py` | First-run scaffolding (`ensure_kb_exists`); seeds `_self.md` and a placeholder persona. |
| `packages/spren/src/spren/memory/cli.py` | `spren memory` Click commands. |
| `packages/spren/src/spren/memory/cli_bridge.py` | UNIX socket bridge (CLI ↔ daemon). |

### To CREATE — Migrations + tests + fixtures

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/storage/migrations/03__create_events_table.py` | Forward-only. `events` + `events_fts` + indexes. |
| `packages/spren/src/spren/storage/migrations/04__create_memory_tables.py` | `entities`, `facts`, `pending_facts`, `prose_chunks` (FTS5). |
| `packages/spren/tests/runtime/test_sandbox_wrapper.py` | Sandbox permission tier matrix. |
| `packages/spren/tests/runtime/test_sandbox_launcher.py` | Per-platform launcher smoke (gated on `@pytest.mark.platform_*`). |
| `packages/spren/tests/runtime/test_parser.py` | Frontmatter + fenced-div + prose parser unit tests. |
| `packages/spren/tests/runtime/test_watcher.py` | FileWatcher debounce + 5-min fallback scan. |
| `packages/spren/tests/runtime/test_sessionlog.py` | SessionLog write+read roundtrip; FTS5 BM25 search. |
| `packages/spren/tests/runtime/test_atomic.py` | Atomic write primitives (kill-during-write recovery). |
| `packages/spren/tests/memory/test_predicate_metadata.py` | Predicate lookup table. |
| `packages/spren/tests/memory/test_indexer.py` | SprenMarkdownIndexer reindex_file + full_scan. |
| `packages/spren/tests/memory/test_bootstrap.py` | First-run tree creation. |
| `packages/spren/tests/memory/test_cli.py` | Click CLI commands via CliRunner. |
| `packages/spren/tests/memory/test_pending_facts.py` | Queue read+write. |
| `packages/spren/tests/memory/test_sp023_boundary.py` | Verifies no `from spren.memory` imports in `spren/runtime/` files (regex grep + AST walk). |
| `packages/spren/tests/integration/test_memory_e2e.py` | End-to-end CLI roundtrip. |
| `packages/spren/tests/fixtures/memory/well_formed_person.md` | Fixture: valid `people/p-mike.md`. |
| `packages/spren/tests/fixtures/memory/malformed_facts_block.md` | Fixture: bad `::: facts` syntax. |
| `packages/spren/tests/fixtures/memory/no_frontmatter.md` | Fixture: file the indexer skips. |
| `packages/spren/tests/fixtures/memory/orphaned_facts.md` | Fixture: file that previously had facts now removed. |

### To MODIFY

| Path | Edit |
|---|---|
| `packages/spren/pyproject.toml` | Add `watchdog`, `python-frontmatter`, `click` deps. |
| `packages/spren/src/spren/server.py` | Add `lifespan` hook that calls `memory.bootstrap.ensure_kb_exists` + starts the SprenMarkdownIndexer + starts the PendingFactsWatcher. |
| `packages/spren/src/spren/__init__.py` | Re-export `runtime` and `memory` subpackages. |
| `Justfile` | Add `just memory-cli` recipe (alias for `uv run --package spren spren memory`). |
| `docs/architecture/spren/10-memory-architecture.md` | Mark §1 / §4 / §11 as "shipped in v0.3 Session 06" with the implementation paths. (One-paragraph addition; not a rewrite.) |

### To DELETE

None. Session 06 is purely additive.

## 7. User journeys (anchor for Bundle 03 demo gate)

Bundle 03's demo gate is "the meta-agent breathes" — that demo is shipped in Session 07 + 08. Session 06 contributes the foundation; the journeys for Session 06 are the user's *direct interactions with memory*, not the agent's interactions.

### J-1 — Fresh install + first memory edit via CLI

State: fresh data dir, daemon launched for first time.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User runs `just dev` (or installs + launches Spren). | Terminal | Daemon starts; sidecar prints `spren-ready: port=<N> token=<T> data-dir=<P>`. Bootstrap creates the full `<data-dir>/sandbox/shared/memory/` tree. |
| 2 | User runs `spren memory show _self`. | CLI | Prints contents of `<data-dir>/sandbox/shared/memory/people/_self.md` — empty frontmatter with the comment line "Tell Spren your name". |
| 3 | User runs `spren memory remember "my name is Reza"`. | CLI | The CLI bridge writes the fact to the daemon. Indexer parses; `_self.md` updates with `name: Reza` frontmatter; SQL `entities` row updated. |
| 4 | User runs `spren memory show _self` again. | CLI | The updated record. |
| 5 | User opens `_self.md` in vim, adds `pronouns: he/him` to the frontmatter, saves. | $EDITOR | watchdog catches the change within ~50ms; indexer reparses; SQL updates. |
| 6 | User runs `spren memory why "name is Reza"`. | CLI | Prints the supersession chain: the fact's `id`, `source: user_direct`, `source_event_id: <ULID>` (linked to the events table row from step 3), the timestamp. |

### J-2 — Forget + restore

State: from J-1, `_self.md` has `name: Reza` and `pronouns: he/him`.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User runs `spren memory forget <fact-id-of-pronouns>`. | CLI | Confirms `Are you sure? (y/N)`. User confirms. The fact's frontmatter gets `tombstoned: true, tombstoned_at: <ts>`; the entry moves to `archive/forgotten/YYYY-MM-DD.md` with a backlink. |
| 2 | User runs `spren memory show _self`. | CLI | `_self.md` no longer shows `pronouns`. The tombstone is hidden from the show by default. |
| 3 | User runs `spren memory restore <fact-id>`. | CLI | The fact returns to `_self.md` with `tombstoned: false`. |

### J-3 — Verify-index detects manual SQL tampering

State: daemon running.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User opens `<data-dir>/data/spren.db` directly with `sqlite3` and updates a fact's `value` column to a different value (simulating tampering). | sqlite3 CLI | The SQL is changed. The markdown still has the old value. |
| 2 | User runs `spren memory verify-index`. | CLI | Walks every fact: each fact's `content_hash` is recomputed and compared against the stored hash; the tampered fact surfaces as `IndexDriftDetected: fact <id> hash mismatch`. The report lists the discrepancy and recommends `spren memory rebuild-index`. |
| 3 | User runs `spren memory rebuild-index`. | CLI | Full-scan repopulates SQL from markdown (the source of truth per SP-016); the tampered value is gone. |

## 8. Decisions locked

These are the calls I'm making in this brief; the implementer doesn't re-litigate them.

1. **Generic-runtime vs Spren-specific boundary in code from this session forward.** SP-023 enforced via test (`test_sp023_boundary.py`) that grep+AST-walks `spren/runtime/` for `from spren.memory` or `import spren.memory` imports. The boundary is mechanical and verifiable.
2. **Sandbox is two layers, both shipped in v0.3.** Application-layer `Sandbox` wrapper (typed FS access) + OS-level `SandboxLauncher` (subprocess sandboxing). Per-platform launchers ship in v0.3 because shell tools ship in v0.3 (Session 07's `execute_shell` + workflow agents already running shell). Reference: `09-meta-agent.md` §"Sandbox and filesystem permissions" + `07-security.md` §"Shell execution and the OS-level sandbox envelope".
3. **`<data-dir>/sandbox/` is the sandbox root.** The whole Spren execution environment runs inside this envelope. NoopLauncher mode for Docker (the container is the envelope) and for tests. CI runs the per-platform launchers natively on macOS / Linux / Windows; WSL2 runs Linux launcher (note: bubblewrap on WSL2 has quirks — implementer benchmarks; if blocked, NoopLauncher is the fallback for the affected test).
4. **Tier 2 dual-write: SQLite + JSONL.** Both written; SQLite is primary; JSONL is recoverable from SQLite. JSONL is for cold storage + offline analysis + future grep-friendly access (SQL FTS5 covers the in-process search need).
5. **No vector embeddings in v0.3.** `prose_chunks` table uses FTS5 BM25 only; `embedding_model_id` column nullable. v0.4 adds sqlite-vec + the hybrid retriever; the `recall` tool API doesn't change.
6. **`spren memory remember` parses fact-shaped input via heuristic.** `"<predicate>: <value>"` → routed by predicate-class to entity / category file. Freeform prose → today's journal. The user can edit further via `spren memory edit`. We deliberately do NOT call an LLM in the CLI write path — the CLI is for explicit user authority, not LLM judgment.
7. **`forget` is a tombstone, not a delete (SP-017).** `restore` is supported. The audit trail is preserved.
8. **`verify-index` checks `content_hash` against recomputation, not just markdown ↔ SQL agreement.** Hash mismatch surfaces as `IndexDriftDetected` and recommends `rebuild-index`. The hash is `sha256(asserted_at | source | value)`.
9. **Format errors at write boundary log to `consolidation_errors.md` and don't commit.** This is M4 from the synthesis; it ships in v0.3 because format errors at 1.2-30.4% rates would silently corrupt the KB if uncaught.
10. **Per-platform tests gated by marker.** `@pytest.mark.platform_linux`, `_macos`, `_windows`. CI matrix runs all three; local dev runs whichever matches.

11. **CLI bridge: TCP-localhost-with-token across all platforms.** The `spren memory` CLI talks to the running daemon over `http://127.0.0.1:<port>` using the per-launch auth token from Session 01 — same pattern as the FastAPI clients (Tauri webview, browser tab, future TUI). Rejected: per-platform IPC (Unix sockets on macOS/Linux + named pipes on Windows). Reasoning: the CLI runs a few commands per minute, not thousands per second; localhost-TCP overhead is microseconds (irrelevant); IPC's marginal correctness isn't worth maintaining two platform-specific code paths. The CLI just becomes another HTTP client. Auth flow: CLI reads the token from `<data-dir>/runtime/auth-token` (mode 0600 — same file Session 01 writes) and passes it as `Authorization: Bearer <token>`. Refuses with a clear error if the daemon isn't running OR if the token file is missing/unreadable.

## 9. Polish items to address inside Session 06

Gaps the architect-stage draft surfaced that the implementer addresses in-session, not as nice-to-haves.

1. **Watchdog reliability across platforms.** Implementer benchmarks watchdog on macOS / Linux / Windows; verifies that file edits via vim (which uses temp-file + rename rather than direct write) trigger the right events. If a platform's events are unreliable, the 5-min full-scan fallback catches drift; document the per-platform behavior in the indexer's docstring.
2. **JSONL daily file rotation at midnight.** When the day rolls over, SessionLog writer must close the previous day's file and open a new one cleanly. Edge case: if the daemon is mid-write at 23:59:59.999, the event might land in either day's file. The rule: bucket by the event's `ts`, not by wall-clock-at-write — this means writes can be slightly out-of-order in the JSONL for ~1ms around midnight. Acceptable; document.
3. **CLI bridge socket path on Windows.** UNIX sockets aren't native on Windows. Use a named pipe or fall back to TCP-on-localhost-with-token. Implementer picks; document.
4. **`spren memory edit` editor selection.** Defaults to `$EDITOR` env var, falls back to `nano` (Linux), `pico` (macOS), `notepad` (Windows). Implementer wires; tests cover the env-var path with a fake editor that just prints args.
5. **Migrations idempotency.** Both new migrations (03 + 04) run cleanly twice without error. Standard pattern from Session 02; verify by running the test suite twice in a row against a persisted DB.
6. **Format validator error message UX.** When `consolidation_errors.md` gets an entry, the entry includes the file path, line range, raw input, and a parser-state snapshot showing where parsing failed. The user reads this when debugging; useful diagnostic information matters.
7. **Sandbox launcher: time-bound enforcement.** Verify that `timeout_s` actually kills the subprocess after the limit. Edge case: a child process that ignores `SIGTERM`. The launcher follows up with `SIGKILL` after a 1-second grace.
8. **CLI errors surface root cause.** When the daemon isn't running and a CLI command needs it, the error is `"Spren daemon is not running. Start it with `just dev` and try again."` — not a Python traceback.
9. **The placeholder Vesper persona seeded at first launch must be cleanly overwriteable** by Session 07's archetype picker. Implementer verifies that `personas/main.yaml` is not write-protected; the picker can replace it without permission errors.
10. **Empty-state UX in the CLI.** `spren memory show` against an empty `values.md` doesn't print "(empty file)" — it prints the actual file content (frontmatter + a comment hint about what goes in this file). The seed files include 1-2 lines of comment-prose in each empty category file: `# Values, ethics, stances. What you care about.` etc. The user can delete the comments when they fill in real content.

## 10. Success criteria

Bundle 03 demo gate ("the meta-agent breathes") ships across Sessions 06 + 07 + 08. Session 06's contribution to that gate:

- **G-17** (memory bootstrap): on first `just dev` against a fresh data dir, the full `<data-dir>/sandbox/shared/memory/` tree exists with all expected files; SQL has empty `entities` + `facts` + `pending_facts` tables.
- **G-18** (memory CLI roundtrip): user can `spren memory remember "name is Reza"`, then `spren memory show _self`, then `spren memory why <fact-id>` and see the chain.
- **G-19** (vim → indexer → SQL): user edits `_self.md` in vim, adds a fact; within 1 second, SQL reflects the change; `lookup_facts(_self, name)` returns the new value.
- **G-20** (forget + restore): user forgets a fact, sees it gone from `show`; restores, sees it back.
- **G-21** (verify-index detects tampering): SQL tampered behind the indexer's back; verify-index surfaces the drift.
- **G-22** (sandbox permission tier): a working_instance role tries to write to `shared/memory/` directly; the wrapper raises `PermissionDeniedError` with the role + path.
- **G-23** (OS-level launcher denies network when off): subprocess invoked with `allow_network=False` cannot reach `8.8.8.8` (verified per-platform).
- **C-06** (no SP-023 violation): `grep -rE "from spren\.memory|import spren\.memory" packages/spren/src/spren/runtime/` returns empty.
- **C-07** (format validator): a deliberately malformed fact-block in a fixture file does not commit to SQL and does log to `consolidation_errors.md`.
- **C-08** (atomic writes survive a kill): test that simulates a kill mid-write produces either the old content fully or the new content fully — never a partial write.
- **U-14** (manual smoke): from a fresh install, run through J-1 + J-2 + J-3.

## 11. Open research items the implementer resolves in-flight

Empirical version + behavior verification done during implementation, not via a separate pre-implementation research stage.

- Watchdog version pin and platform behavior — implementer runs the benchmark across macOS / Linux / Windows.
- python-frontmatter version pin + edge cases (BOM, CRLF, malformed YAML).
- Bubblewrap argument list for the Linux outer envelope — exact flags vary slightly across distros and bubblewrap versions; verify against `bwrap --version` ≥ 0.4.
- macOS sandbox-exec profile shape — Apple deprecated the public sandbox-exec interface; it still works but the profile syntax should be tested against the current macOS version (15.x as of session start). If sandbox-exec is broken, the fallback is to use `posix_spawn` with `chroot`-equivalent and document that the macOS envelope is weaker than the Linux/Windows ones (acceptable for v0.3 single-user-local).
- Windows AppContainer + Job Object integration — complex; the implementer benchmarks. Fallback: a minimal restricted-token implementation if the full AppContainer setup is too heavy.
- SQLite FTS5 with JSON payload search performance at 100k events — benchmark; if slow, switch to indexing extracted text columns rather than the JSON payload directly.
- The 4-hour interval for `PendingFactsWatcher` — verify against a synthetic queue-grow scenario; tune if 50 facts is too aggressive or too lax.

If any of these surface a conflict with the locked decisions in §8, the implementer flags it and asks before deviating.

## 12. Status

- [ ] Tier confirmed (CRITICAL).
- [ ] Scope boundaries confirmed (sections 4 + 5).
- [ ] SP-023 boundary confirmed (section 3).
- [ ] Files-to-CREATE list approved (section 6).
- [ ] Three user journeys approved (section 7).
- [ ] Decisions locked (section 8).
- [ ] Polish items captured for in-session work (section 9).
- [ ] Success criteria affirmed (section 10).
- [ ] Acceptance criteria frozen at [`./06-memory-foundation/acceptance.md`](./06-memory-foundation/acceptance.md) — extracted by the `acceptance-criteria-extractor` agent on the first implementation turn, before any code is written.
- [ ] Architect peer review at Stages 1, 3, 4 (CRITICAL multi-checkpoint).
- [ ] Session implementation complete (all acceptance criteria pass; polish items addressed; tests green; manual verify done).

**Next step:** acceptance-criteria-extractor freezes `./06-memory-foundation/acceptance.md` on the first implementation turn. Implementer begins. Sessions 07 + 08 build on top of this foundation; their architect drafts begin once Session 06 is in implementation (parallel architect work + sequential implementation).

## 13. Open questions for user input

One remaining; the prior question on the CLI bridge transport (Q1) was resolved in §8 decision 11 (TCP-localhost-with-token across all platforms).

1. **Format validator strictness — silent rejection vs surfaced error?** When a fact-block fails parsing, it's logged to `consolidation_errors.md`. But should the *user* see this immediately (e.g., a notification, a banner in the daemon log), or is the log file enough? **My pick: log file is enough for v0.3.** A user editing markdown via vim doesn't want a popup; a malformed save shows up the next time they `spren memory verify-index`. v0.4 might add a watcher that surfaces the error. Want your call.
