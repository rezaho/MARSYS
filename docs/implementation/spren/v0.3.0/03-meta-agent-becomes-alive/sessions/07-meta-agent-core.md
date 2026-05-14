# Spren Session 07 — Meta-Agent Core + Persona

> Session plan. The implementer reads this as the primary source of truth for what Session 07 ships, how the daemon + EventIngress + per-agent inbox + scheduler + watcher catalog + heartbeat + main agent loop + 8-axis system prompt assembly + 5-archetype onboarding + execute_shell tool + voice-drift warning layer fit together. Captures bundle position, scope boundaries, dependency check, files-to-CREATE / MODIFY, the user journeys that anchor Bundle 03's middle demo gate, wireframes for the small UI lift (the archetype picker is the only Session 07 UI surface), data-model considerations, the locked decisions, polish items, success criteria, and open research items the implementer resolves in-flight.
>
> Status: **draft — subject to user redirect**. Acceptance criteria are frozen separately at [`./07-meta-agent-core/acceptance.md`](./07-meta-agent-core/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Architectural anchors (read before coding):
- [`../../../../architecture/spren/09-meta-agent.md`](../../../../architecture/spren/09-meta-agent.md) — the full meta-agent architecture. §"Why this is its own execution model", §"The agent hierarchy", §"Event ingress and the inbox model", §"Decision flow on each event", §"Authority tiers", §"Scope of the meta-agent", §"System prompt structure — six axes of context", §"Persona — the 8 sub-axes", §"Onboarding — five archetypes", §"Persona-evolution mechanism", §"Bond-violation handling", §"The supervision surface", §"Heartbeat", §"Lifecycle and crash recovery", §"Sandbox and filesystem permissions", §"Tool surface and tiered creation".
- [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — especially SP-011 (bounded triggers), SP-012 (hard rails always confirm), SP-013 (cost ceiling load-bearing), SP-014 (working instances persist scratchpad atomically), SP-019 (API single source of truth), SP-023 (generic-runtime vs Spren-specific boundary).
- [`../../../../../tmp/spren/research/06-memory-foundations/00-synthesis.md`](../../../../../tmp/spren/research/06-memory-foundations/00-synthesis.md) — M13' (archetype picker), M14 (persona-evolution mechanism), M18 (async-default + sync-escape-valve write paths).
- [`../../../../../tmp/spren/research/06-memory-foundations/02-spren-soul-and-bond.md`](../../../../../tmp/spren/research/06-memory-foundations/02-spren-soul-and-bond.md) — Stormlight bond mechanic translated to product rules; voice spec.
- [`../../../../../tmp/spren/research/06-memory-foundations/05-five-archetypes.md`](../../../../../tmp/spren/research/06-memory-foundations/05-five-archetypes.md) — full 5-archetype designs (Ember / Flint / Quill / Vesper / Kindle); 8-axis YAMLs; voice samples; visual orb variants.
- [`./06-memory-foundation.md`](./06-memory-foundation.md) — Session 06 (the foundation Session 07 builds on).

---

## 1. Bundle position + tier

- **Bundle**: 03 — Meta-agent becomes alive (Sessions 06 + 07 + 08). Session 07 ships the agent's *body*: it boots, loops, listens, ticks, breathes. The orb is alive. But it doesn't yet *do anything productive* — that's Session 08 (read tools + write tools + supersession + persona-evolution + suggest-with-confirm).
- **Session 07 scope**: meta-agent runtime (daemon, inbox, scheduler, watchers, heartbeat, main agent loop), six-axis system prompt assembly, 5-archetype onboarding picker (CLI + web), `execute_shell` tool, voice-drift warning layer, hard-rail confirmation flow infrastructure (the actual hard-rail tool list is Session 08).
- **Tier**: CRITICAL. The agent runtime is load-bearing for everything downstream; SP-011 (bounded triggers), SP-013 (cost ceiling), SP-019 (API source of truth) all funnel through the loop in this session. Full meta-process pipeline.
- **Approval gate**: peer Stage 0 conversation already complete (this brief). Researcher + Designer + Validator/Critic + Fact-checker + Synthesis. Multi-checkpoint user review per CRITICAL tier.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 06 (memory foundation) | ships first | Sandbox + session log + KB scaffold + indexer + pending_facts queue + memory CLI. Session 07 hard-depends on all of these. |
| Spren Session 01 (foundation) | shipped | FastAPI sidecar, auth, capabilities, bootstrap endpoint. Session 07 extends server.py with new routes. |
| Spren Session 03 (visual builder) | shipped | Design system + Spren orb component + reactive states (idle/typing/thinking/speaking). Session 07's orb wires to live agent state via SSE. |
| Spren Session 04 (run execution) | shipped | `Orchestra.run()` invocation pattern. Session 07's main agent uses this when dispatching workflows (the actual `run_workflow` tool ships in Session 08). |
| `marsys.coordination.orchestra.Orchestra` | live | The framework's request-response entry. The meta-agent calls `Orchestra.run()` when dispatching. (Read-only consumption; SP-001 holds.) |
| `marsys.coordination.event_bus.EventBus` | live | The framework's pub/sub. The meta-agent subscribes to lifecycle events for in-flight observability of dispatched workflows. |
| `apscheduler` | new dep — must be added | Time scheduler. APScheduler 4.x. Used for one-off + recurring scheduled events. |
| `litellm` (or equivalent) | already present via marsys | LLM provider abstraction. The meta-agent uses it for: extraction-classifier prompts (Session 07), main agent turns (Session 07), consolidation-pass extraction + adjudication (Session 08). |
| Existing settings store (`<data-dir>/data/spren.db` settings table) | exists from Session 01 | New keys: `meta_agent.model.cheap`, `meta_agent.model.expensive`, `meta_agent.heartbeat_interval_minutes`, `cost.daily_budget_usd`, `cost.per_think_token_cap`, `meta_agent.archetype` (set on first-run picker), `meta_agent.persona_path`. |

Session 07 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). All work lands inside `packages/spren/src/spren/` and adds two new top-level subpackages: `spren/runtime/agent/` (generic agent-loop, inbox, scheduler, watchers, heartbeat) and `spren/agent/` (Spren-specific persona, six-axis assembly, archetype picker, execute_shell tool wiring, voice-drift layer).

## 3. SP-023 boundary — what goes in `spren/runtime/agent/` vs `spren/agent/`

**Generic always-on runtime — `spren/runtime/agent/`** (no Spren-specific imports):
- `runtime/agent/inbox.py` — `Inbox` class with priority queue (P0-P3), bounded async-queue per subscriber (default 256, drop-oldest with `STREAM_LAGGED` notice), durable persistence to `<sandbox-root>/inbox/pending/<priority>/<event-id>.json` for crash recovery. Generic event-shape; no Spren-specific event types.
- `runtime/agent/event_ingress.py` — `EventIngress` central router. Takes events from all sources, routes by `event.target_agent` (default: main); emits to that agent's inbox. Generic; producer registration pattern for sources to attach.
- `runtime/agent/scheduler.py` — APScheduler wrapper. `schedule_one_off(at: datetime, payload: dict)`, `schedule_recurring(cron_expr: str, payload: dict)`, `schedule_recurring_at_interval(interval: timedelta, payload: dict)`. Generic dispatch; payloads are opaque dicts.
- `runtime/agent/watcher.py` — `Watcher` ABC. Subclasses run as Python coroutines; `tick()` returns `WatcherResult` (`{fired: bool, event: dict | None}`). The catalog of concrete watchers is per-product (some generic — `BudgetWatcher` is generic-shape; some Spren-specific — `PendingFactsWatcher` knows about Spren's `pending_facts` table).
- `runtime/agent/heartbeat.py` — `Heartbeat` primitive. Configurable cadence; emits a `HeartbeatTick` event into the agent's inbox at P3.
- `runtime/agent/loop.py` — `AgentLoop` ABC. Takes an `Inbox`, a `ContextLoader`, a `Reasoner`. Loops: `dequeue → load_context → reason → process_outputs → repeat`. Process-outputs is generic (handle inline / delegate / spawn / ask user / defer — but the *shape*, not the Spren-specific delegation rules).
- `runtime/agent/scratchpad.py` — atomic-write scratchpad pattern (SP-014). `Scratchpad(path)` with `write_state(dict)` (atomic) and `read_state() -> dict` (load on agent boot). Generic.
- `runtime/agent/cost_ceiling.py` — `CostMeter` class. `check_budget()`, `record_call(tokens_in, tokens_out, model)`, `daily_total_usd()`, `per_think_total_tokens()`. Generic.
- `runtime/agent/sandbox_exec.py` — generic `execute_shell` primitive. `ExecuteShell(launcher: SandboxLauncher, audit_log: SessionLog)` exposes the tool. Logs every call to the audit log + the session log. The hard-rail confirmation is implemented at the *tool-binding* layer (Spren-specific) — the primitive itself just runs the subprocess sandboxed.

**Spren-specific layer — `spren/agent/`**:
- `agent/__init__.py` — module init.
- `agent/main_agent.py` — Spren's main agent definition. Orchestrates the six-axis system prompt assembly + the Spren-specific decision flow (`handle inline / delegate to team / spawn working instance / ask user / defer`). Consumes generic `runtime/agent/AgentLoop`.
- `agent/persona.py` — persona file reader + writer; the 8 sub-axes; voice-drift warning layer (regex post-pass).
- `agent/archetypes.py` — the 5 archetypes' YAML defaults loaded from `data/archetypes/<slug>.yaml` files. Each archetype is one YAML file; the bootstrap copies it to `personas/main.yaml` on pick.
- `agent/onboarding.py` — first-run archetype picker logic. Detects "no archetype chosen" state (settings key `meta_agent.archetype` is null); routes the user to the picker (CLI + web).
- `agent/system_prompt.py` — six-axis assembly. Reads persona + doctrine + active_context + capabilities + active_todos; calls memory tools for axis #6 lazy retrieval (Session 06 + Session 08 ship the tools); concatenates into the system prompt.
- `agent/doctrine.py` — `rules.md` reader. Defaults shipped at `data/rules_default.md`; user-editable.
- `agent/tools/__init__.py` — tool registry for the meta-agent's own tools (distinct from the workflow tool catalog from Session 02). Session 07 ships: `execute_shell` (with hard-rail wrapper), `schedule_event`, `schedule_reminder`. Session 08 ships the rest.
- `agent/tools/execute_shell.py` — Spren-specific tool binding for the generic `runtime/agent/sandbox_exec.ExecuteShell`. Adds the hard-rail confirmation flow + standing approvals lookup.
- `agent/watchers/__init__.py` — Spren's watcher catalog. Session 07 ships: `BudgetWatcher`, `DiskWatcher`, `ProviderHealthWatcher`, `PendingFactsWatcher`, `IndexDriftWatcher`, `WorkflowRunWatcher`. Each is a `runtime/agent/Watcher` subclass with Spren-specific `tick()` logic.
- `agent/orb_state.py` — derives the orb's reactive state (`idle | thinking | speaking | typing`) from the agent's loop phase; emits via the existing SSE stream from Session 04 (run-events) extended with a new `meta_agent_orb` channel.
- `agent/api.py` — new FastAPI routes: `POST /v1/meta/messages`, `GET /v1/meta/conversations/{id}/events` (SSE), `GET /v1/meta/conversations`, `GET /v1/meta/conversations/{id}`, `GET /v1/onboarding/state`, `POST /v1/onboarding/archetype` (the picker), `POST /v1/agent/confirm/<confirmation-id>` (hard-rail confirmation flow).
- `agent/data/archetypes/{ember,flint,quill,vesper,kindle}.yaml` — the 5 archetype YAMLs.
- `agent/data/rules_default.md` — default doctrine.
- `agent/data/active_context_template.md` — seed for `active_context.md` on first launch.

**SP-023 enforcement test extends:** `tests/sp023_boundary.py` from Session 06 grows to include the new files. Same rule: `runtime/` files cannot import from `spren.agent` or `spren.memory`.

## 4. What ships in Session 07

### 4.1 The daemon shape

The Spren daemon is the FastAPI process from Session 01, extended. On startup (`server.py:lifespan`):

1. Memory bootstrap (Session 06's `memory.bootstrap.ensure_kb_exists`).
2. Onboarding state check — if no archetype chosen (settings `meta_agent.archetype is null`), the daemon enters "onboarding mode" — agent loop doesn't start; only the bootstrap + onboarding API routes are active. Once the picker submits, the persona is written and the daemon transitions into "running mode".
3. Agent runtime startup (in running mode):
   - Load `personas/main.yaml`, `rules.md`, `active_context.md`, `active_todos.md`.
   - Initialize `Inbox` (durable; replay any persisted events from the previous session per crash recovery).
   - Initialize `EventIngress` with producers: user_direct (web/CLI), channel adapters (none in v0.3 except web/CLI), schedule, watchers, sub-instance-completions (none in v0.3, deferred to v0.4).
   - Start `Heartbeat` (default 30 min, configurable via `meta_agent.heartbeat_interval_minutes`).
   - Start the watcher catalog (each watcher is a coroutine).
   - Start the main agent loop coroutine.

The agent loop is async; the FastAPI process has a single asyncio loop running both the agent + the HTTP routes. Sub-agents (when v0.4 ships sub-instances) get their own loops on separate asyncio tasks.

On graceful shutdown:
- Stop accepting new events into the EventIngress.
- Wait for the current agent turn to complete (with a timeout).
- Persist any unprocessed inbox events to `<sandbox-root>/inbox/pending/`.
- Shutdown the watcher coroutines.
- Stop the FastAPI server.

On crash:
- Next startup, the durable inbox replays any `<sandbox-root>/inbox/pending/` events into the Inbox before the loop starts.
- Sub-agents persist state in scratchpads (SP-014); v0.4 brings them back. v0.3 has no sub-agents to recover.

### 4.2 EventIngress + Inbox

```python
# runtime/agent/event_ingress.py
class EventIngress:
    def __init__(self):
        self._producers: list[EventProducer] = []
        self._inboxes: dict[AgentId, Inbox] = {}

    def register_producer(self, producer: EventProducer) -> None: ...
    def register_inbox(self, agent_id: AgentId, inbox: Inbox) -> None: ...
    async def route(self, event: Event) -> None:
        target = event.target_agent or AgentId.MAIN
        await self._inboxes[target].put(event)

# runtime/agent/inbox.py
class Inbox:
    def __init__(self, agent_id: AgentId, sandbox: Sandbox, max_size: int = 256):
        self.agent_id = agent_id
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._sandbox = sandbox  # for durable persistence

    async def put(self, event: Event) -> None: ...
    async def get(self) -> Event: ...  # blocks if empty; pops highest priority
    async def peek_priority(self) -> Priority | None: ...  # for chunked thinking preemption check
    def persist_pending(self) -> None: ...  # graceful-shutdown drain
    def replay_pending(self) -> None: ...  # boot recovery
```

`Event` is a Pydantic model:

```python
class Priority(IntEnum):
    P0 = 0  # user-direct, critical workflow failures, InterruptNow
    P1 = 1  # channel messages (allowlisted), instance completions
    P2 = 2  # workflow lifecycle, scheduled triggers
    P3 = 3  # heartbeat, webhook fires (later release)

class Event(BaseModel):
    id: str  # ULID
    ts: float
    kind: str  # "user_message" | "scheduled_run" | "budget_threshold_crossed" | "heartbeat_tick" | "instance_completed" | "needs_decision" | etc.
    priority: Priority
    target_agent: AgentId | None  # default = MAIN
    actor: str | None  # producer-side identifier
    channel: str | None  # source channel for routing/trust
    payload: dict
    schema_version: int = 1
```

Inbox event-kinds list lives in `agent/events.py` as enums; the runtime code doesn't validate kinds.

### 4.3 Scheduler

```python
# runtime/agent/scheduler.py — APScheduler 4.x wrapper
class TimeScheduler:
    def __init__(self, ingress: EventIngress, sandbox: Sandbox):
        self._scheduler = AsyncIOScheduler(jobstores={"default": SQLAlchemyJobStore(...)})

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def schedule_one_off(self, at: datetime, payload: dict) -> str: ...  # returns job_id
    def schedule_recurring_cron(self, cron_expr: str, payload: dict) -> str: ...
    def schedule_recurring_interval(self, interval: timedelta, payload: dict) -> str: ...
    def cancel(self, job_id: str) -> bool: ...
```

The job store is SQLAlchemy-backed against the existing `<data-dir>/data/spren.db`. Jobs survive daemon restart.

When a scheduled job fires, the scheduler emits a `ScheduledRun` event (or `ScheduledReminder` for agent-issued schedules) into the EventIngress at P2 (default; customizable via the payload).

### 4.4 Watcher catalog

`runtime/agent/watcher.py` defines the ABC. `agent/watchers/` ships the v0.3 catalog:

```python
# Each is a Watcher subclass.
class BudgetWatcher(Watcher):  # cadence: 1 min
    """Spend reaches 75% / 90% / 100% of daily cap → BudgetThresholdCrossed event."""

class DiskWatcher(Watcher):  # cadence: 5 min
    """Free space < 5GB on the data-dir partition → DiskLow event."""

class ProviderHealthWatcher(Watcher):  # rolling window over LLM API calls
    """N consecutive 429s OR sustained 5xx → ProviderRateLimited / ProviderUnavailable event."""

class PendingFactsWatcher(Watcher):  # cadence: 5 min
    """Queue grows past threshold (default 50) → ConsolidationDue event (early trigger,
    independent of the 24h+5-session time trigger from Session 08's consolidation pass)."""

class IndexDriftWatcher(Watcher):  # cadence: 15 min
    """Full-scan finds SQL ↔ markdown disagreement → IndexDriftDetected event."""

class WorkflowRunWatcher(Watcher):  # per active run, cadence: 30 sec
    """Run idle > timeout, cost > per-run cap, wall-clock > limit →
    WorkflowRunStalled / WorkflowRunOverBudget / WorkflowRunOvertime event."""
```

Each watcher:
- Runs as a Python asyncio coroutine in the daemon's event loop.
- Has a fixed cadence per `09-meta-agent.md` §"System watchers"; configurable via settings.
- Emits events at appropriate priority (e.g., `BudgetThresholdCrossed(100%)` is P0; `IndexDriftDetected` is P3).
- Has zero LLM calls — purely mechanical.
- Is registered with the EventIngress at daemon startup.

### 4.5 Heartbeat

```python
# runtime/agent/heartbeat.py
class Heartbeat:
    def __init__(self, ingress: EventIngress, interval: timedelta = timedelta(minutes=30)):
        self._ingress = ingress
        self._interval = interval

    async def start(self) -> None:
        while True:
            await asyncio.sleep(self._interval.total_seconds())
            await self._ingress.route(Event(
                kind="heartbeat_tick",
                priority=Priority.P3,
                target_agent=AgentId.MAIN,
                payload={},
            ))
```

When the main agent dequeues a `HeartbeatTick`, it runs a "review turn" — see `09-meta-agent.md` §"Heartbeat — the proactive thinking cycle". The actual review-turn behavior is the main agent's job (`agent/main_agent.py`); the heartbeat primitive just emits the event.

### 4.6 Main agent loop + decision flow

```python
# agent/main_agent.py
class MainAgent:
    """The Spren main agent. Composes the generic AgentLoop with Spren-specific
    context loader, reasoner, and process-outputs handler."""

    def __init__(
        self,
        inbox: Inbox,
        sandbox: Sandbox,
        session_log: SessionLog,
        memory: SprenMemoryAccess,        # from Session 06
        persona: PersonaReader,
        doctrine: DoctrineReader,
        active_context: ActiveContextStore,
        cost_meter: CostMeter,
        llm_client: LLMClient,             # litellm or equivalent
        tool_registry: ToolRegistry,        # Session 07 partial; Session 08 full
    ): ...

    async def run(self) -> None:
        while not self._shutdown:
            event = await self.inbox.get()
            try:
                await self._process_event(event)
            except Exception as e:
                # Log, do not crash the loop.
                self._session_log.write(SessionEvent(kind="agent_error", payload={"event_id": event.id, "error": str(e)}))

    async def _process_event(self, event: Event) -> None:
        # 1. Cost-budget guard
        if not self._cost_meter.has_budget():
            if event.priority > Priority.P0:
                # Defer; schedule re-attempt at next budget cycle.
                self._scheduler.schedule_one_off(at=next_budget_window(), payload={"deferred_event": event.dict()})
                return
            # P0 = critical; process anyway, surface budget exhaustion.

        # 2. Assemble system prompt from six axes.
        system_prompt = self._system_prompt.assemble()

        # 3. Reason — call the LLM with system prompt + event as user message.
        response = await self._llm_client.chat(
            model=self._select_model(event),  # cheap for routine, expensive for hard reasoning
            system=system_prompt,
            messages=[{"role": "user", "content": format_event_for_llm(event)}],
            tools=self._tool_registry.schemas_for(self.agent_id),
        )

        # 4. Process outputs.
        await self._process_response(event, response)

    async def _process_response(self, event: Event, response: LLMResponse) -> None:
        # Branch on decision per 09-meta-agent.md §"Decision flow on each event":
        # (a) Handle inline — emit response to user, update state.
        # (b) Delegate to team manager — enqueue on their inbox. (v0.3: no teams; this branch is unreachable.)
        # (c) Spawn working instance. (v0.3: no sub-instances; this branch is unreachable.)
        # (d) Ask user — enqueue a "needs decision" inbox item.
        # (e) Defer — schedule re-attempt.

        # In v0.3, only (a) and (d) are live paths.
        # The response's tool_calls drive everything: text-only response = (a); tool call = either inline tool execution or (d).

        for tool_call in response.tool_calls or []:
            tool = self._tool_registry.get(tool_call.name)
            if tool.requires_confirmation and not self._has_standing_approval(tool, tool_call.arguments):
                # (d) Ask user — surface a confirmation request.
                await self._enqueue_confirmation_request(tool_call)
                continue
            result = await tool.execute(tool_call.arguments, agent=self)
            self._session_log.write(SessionEvent(kind="tool_call", payload={"tool": tool.name, "args": tool_call.arguments, "result": result.summary}))
```

### 4.7 Six-axis system prompt assembly

```python
# agent/system_prompt.py
class SystemPromptAssembler:
    """Assembles axis #1-5 eagerly into the system prompt; axis #6 (Memory)
    is left to the agent's tool calls."""

    def __init__(
        self,
        persona: PersonaReader,
        doctrine: DoctrineReader,
        active_context: ActiveContextStore,
        active_todos: ActiveTodosStore,
        capabilities: CapabilitiesProvider,  # tools available, budget remaining, etc.
        recent_activity: RecentActivityProvider,
    ): ...

    def assemble(self) -> str:
        sections = [
            self._render_persona(),                # axis 1
            self._render_doctrine(),               # axis 2
            self._render_situational_awareness(),  # axis 3 — time, timezone, DND, "weather"
            self._render_capabilities(),           # axis 4 — available tools, budget remaining
            self._render_active_context(),         # axis 5 — TODO list, recent N actions
        ]
        return "\n\n---\n\n".join(sections)
```

The output is one long system-prompt string. Axis #6 (Memory — `read_file`, `grep`, `lookup_facts`, `recall`) is mentioned in the persona + doctrine sections as tools the agent can call; the *content* is fetched on demand.

The persona section is rendered from `personas/main.yaml`'s 8 sub-axes. Each axis becomes a paragraph in the system prompt. The voice-drift warning layer's blacklisted phrases are surfaced explicitly: "Avoid: \[list of forbidden phrases from `voice.style_tells.avoid`\]."

### 4.8 Onboarding flow — three steps

First-run onboarding has three required steps in this order. The daemon stays in onboarding mode (agent loop NOT running) until all three complete; only `/v1/onboarding/*` routes are active. Each step writes settings + transitions to the next; the user cannot skip steps.

**Step 1 — Provider authentication** (REQUIRED first). Without a configured LLM provider, no LLM call is possible; the agent loop cannot start. The picker shows the supported providers as cards:

| Provider card | Auth method | Default selection |
|---|---|---|
| **Anthropic** | API key | (selected by default — Claude-friendly is our default lane) |
| **Anthropic Claude.ai (OAuth)** | OAuth flow with the user's Claude.ai account | |
| **AWS Bedrock** | AWS credentials (key + secret + region) | |
| **OpenRouter** | API key | |
| **OpenAI** | API key | |
| **Azure OpenAI** | API key + endpoint + deployment | |
| **Google AI** | API key | |
| **DeepSeek** | API key (uses OpenAI-compatible adapter) | |

The Anthropic API card is highlighted as default — most users we expect to onboard with v0.3 are Claude-friendly. The user can pick any other; the highlight is a starting point, not a forced choice.

After provider pick, the user enters credentials. Step 1 completes when credentials are validated by an actual test call against the provider's lightest endpoint (e.g., `GET /v1/models` for OpenAI; `GET /messages` smoke for Anthropic). On failure: clear error + back to credential entry. On success: settings write `meta_agent.provider = <name>` + `meta_agent.credentials_path` (or keychain reference per Session 01's secrets pattern) + advance to Step 2.

**Step 2 — Model selection (cheap + expensive)**. Once a provider is configured, the model picker shows the per-provider defaults pre-selected. User can change either; can't skip.

Per-provider defaults (verify model identifiers at session start — implementer task — but these match the latest available as of session draft):

| Provider | Cheap default | Expensive default |
|---|---|---|
| Anthropic API | `claude-haiku-4-5` | `claude-sonnet-4-6` |
| Anthropic OAuth (Claude.ai) | `claude-haiku-4-5` | `claude-opus-4-7` |
| AWS Bedrock | `claude-haiku-4-5` (Bedrock-routed) | `claude-sonnet-4-6` (Bedrock-routed) |
| OpenRouter | `anthropic/claude-haiku-4-5` | `anthropic/claude-sonnet-4-6` |
| OpenAI | `gpt-5.4-mini` | `gpt-5.4` (or `gpt-5.5` if available) |
| Azure OpenAI | `gpt-5.4-mini` (Azure deployment) | `gpt-5.4` (Azure deployment) |
| Google AI | `gemini-3.1-flash` | `gemini-3.1-pro` |
| DeepSeek | `deepseek-v4-mini` | `deepseek-v4-pro` |

The defaults are stored in `agent/data/provider_defaults.yaml`; the implementer verifies against current provider catalogs at implementation time and updates the file. The settings write `meta_agent.model.cheap = <id>` + `meta_agent.model.expensive = <id>`. User can change later via the settings UI.

**Step 3 — Archetype picker** (the existing 5-card picker from Research 2). Same flow as before — Ember / Flint / Quill / Vesper / Kindle cards with their orb visual variants + 3-line voice descriptions. User clicks one; the chosen orb morphs into the home page's main orb.

After Step 3 submission, daemon transitions to running mode (agent loop, watchers, heartbeat all start).

**Backend routes (`agent/api.py`):**

```python
# GET /v1/onboarding/state
# Returns the onboarding state. Default: {"completed": false, "step": "provider_auth", "completed_steps": []}.
# After step 1: {"completed": false, "step": "model_selection", "completed_steps": ["provider_auth"], "provider": "<name>"}.
# After step 2: {"completed": false, "step": "archetype_picker", "completed_steps": ["provider_auth", "model_selection"], "provider": ..., "models": {...}}.
# After step 3: {"completed": true, "archetype": "<name>", "completed_at": "<ts>"}.

# POST /v1/onboarding/provider
# Body: {"provider": "anthropic" | "anthropic_oauth" | "bedrock" | "openrouter" | "openai" | "azure_openai" | "google" | "deepseek", "credentials": <provider-specific>}
# Validates credentials via a smoke call to the provider. On success: writes settings.meta_agent.provider + credentials.
# Returns: {"success": true, "provider": "<name>"} or {"success": false, "error": "..."}.

# GET /v1/onboarding/models
# Returns the per-provider default model ids for the configured provider, plus a list of selectable alternatives.

# POST /v1/onboarding/models
# Body: {"cheap": "<model-id>", "expensive": "<model-id>"}
# Validates that both models are available for the configured provider (uses framework's adapter SUPPORTED_MODELS list where present, OR a smoke call).
# Writes settings.meta_agent.model.cheap + settings.meta_agent.model.expensive.

# POST /v1/onboarding/archetype
# Body: {"archetype": "ember" | "flint" | "quill" | "vesper" | "kindle"}
# Validates the archetype name. Loads the archetype YAML from data/archetypes/<name>.yaml.
# Writes personas/main.yaml with the archetype's 8-axis defaults + identity block + bonded_at = now() + archetype = name.
# Sets settings.meta_agent.archetype = name.
# Transitions the daemon from onboarding mode to running mode (starts the agent loop, watchers, heartbeat).
# Returns: {"success": true, "archetype": "<name>"}.
```

**Frontend (the picker UI):**

The onboarding flow lives at `apps/web/src/routes/onboarding.tsx` with sub-routes `/onboarding/provider`, `/onboarding/models`, `/onboarding/archetype` for each step. A progress indicator at the top shows step 1 of 3 / 2 of 3 / 3 of 3.

Each step's UI reuses Session 03's design system. Step 1 (provider): a 2-row grid of provider cards; click → expand to credential entry inline (no separate screen). Step 2 (models): two dropdowns side-by-side (Cheap / Expensive) pre-populated with the provider's defaults; "Use defaults" button skips manual selection. Step 3 (archetype): the 5-card picker per Research 2 (5 cards in a row at desktop, with their orb variants animating in incommensurate cadences).

After the archetype is picked → the chosen orb scales up + transitions to the home page; the daemon agent loop has started.

Routing logic:
- On daemon-running, the web client calls `GET /v1/onboarding/state`. If `completed: false`, route to `/onboarding/<current_step>`. Else route to home (`/`).
- The picker is non-skippable in v0.3 for the GUI flow. CLI override: `spren onboarding pick --provider <name> --cheap <model> --expensive <model> --archetype <name>` for headless / scripted setups (e.g., demo videos, test fixtures).

### 4.9 `execute_shell` tool

Tool schema:

```python
class ExecuteShellArgs(BaseModel):
    cmd: list[str]                    # argv-style; not a shell string
    cwd: str | None = None             # path inside sandbox (or absolute path the user explicitly granted)
    timeout_s: int = 30
    allow_network: bool = False

class ExecuteShellResult(BaseModel):
    returncode: int
    stdout: str                        # tail-truncated to 4K chars
    stderr: str                        # tail-truncated to 4K chars
    duration_ms: int
    truncated: bool                    # true if stdout/stderr were tail-truncated
```

Tool implementation:

```python
# agent/tools/execute_shell.py
class ExecuteShellTool:
    """Hard-rail-confirmed shell execution.
    
    Wraps the generic runtime/agent/sandbox_exec.ExecuteShell primitive
    with the Spren-specific confirmation flow + standing approvals."""

    def __init__(self, sandbox_exec: ExecuteShell, persona_policy: PolicyReader):
        self._sandbox_exec = sandbox_exec
        self._policy = persona_policy

    @property
    def requires_confirmation(self) -> bool:
        return True  # Hard rail — see SP-012. Standing approvals checked at confirmation time.

    async def execute(self, args: ExecuteShellArgs, agent: Agent) -> ExecuteShellResult:
        # Confirmation flow handled by the calling agent's _process_response.
        # By the time this runs, confirmation has already been granted.
        result = await self._sandbox_exec.run(
            cmd=args.cmd,
            cwd=Path(args.cwd) if args.cwd else agent.sandbox.root,
            sandbox_root=agent.sandbox.root,
            timeout_s=args.timeout_s,
            allow_network=args.allow_network,
        )
        return ExecuteShellResult(
            returncode=result.returncode,
            stdout=truncate_tail(result.stdout.decode(errors="replace"), 4096),
            stderr=truncate_tail(result.stderr.decode(errors="replace"), 4096),
            duration_ms=result.duration_ms,
            truncated=len(result.stdout) > 4096 or len(result.stderr) > 4096,
        )
```

The hard-rail confirmation flow:

1. Agent emits a tool_call for `execute_shell` with the cmd.
2. Main loop's `_process_response` checks `tool.requires_confirmation` → True.
3. Loop checks standing approvals: `policy.is_approved(tool="execute_shell", scope=cwd, cmd=cmd)`. Standing approvals are scoped — e.g., "git in `<repo>`" allows `git status`, `git diff`, but not `rm -rf`.
4. If approved → execute directly.
5. If not approved → emit a `NeedsDecision` event into the inbox (P1) with the tool_call payload + a draft message to surface to the user. The agent's response to the user includes "I want to run \[cmd\]; confirm to proceed."
6. User confirms via `POST /v1/agent/confirm/<confirmation-id>` (web/CLI) → the deferred tool call executes, result feeds back into the agent's next turn.

Audit log: every `execute_shell` invocation logs to `<data-dir>/logs/shell-audit.log` with `{ts, agent_id, workflow_id?, cmd, cwd, exit_code, duration_ms}`. No content of stdout/stderr is logged (could contain secrets); just the command and result code.

### 4.10 Voice-drift warning layer

Every agent-emitted message passes through a regex post-pass that scans for blacklisted phrases declared in `personas/main.yaml` `voice.style_tells.avoid`. Hits log `voice_drift` events into the session log (kind = `voice_drift`, payload = `{phrase, message_id, agent_id, archetype}`). The warning is **logging-only** — the message is not modified. Consolidation pass (Session 08) reads voice_drift events; if drift accumulates, the persona-evolution mechanism proposes a re-emphasis of the voice axes.

```python
# agent/persona.py
class VoiceDriftDetector:
    def __init__(self, persona: PersonaReader, session_log: SessionLog):
        self._persona = persona
        self._session_log = session_log

    def scan(self, message: str, agent_id: AgentId) -> list[VoiceDriftHit]:
        avoid = self._persona.get("voice.style_tells.avoid", [])
        hits = []
        for phrase in avoid:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", message, re.IGNORECASE):
                hits.append(VoiceDriftHit(phrase=phrase, span=(match.start(), match.end())))
        for hit in hits:
            self._session_log.write(SessionEvent(
                kind="voice_drift",
                payload={"phrase": hit.phrase, "agent_id": agent_id, "archetype": self._persona.get("archetype")},
            ))
        return hits
```

The blacklist comes from the archetype's YAML (each archetype has its own `voice.style_tells.avoid` list per `05-five-archetypes.md`). Plus a global floor of forbidden phrases (`"Great question"`, `"Happy to help"`, `"In conclusion"`, etc.) loaded from `agent/data/global_voice_floor.yaml`.

### 4.11 New deps

`pyproject.toml` for `packages/spren/` adds:

```toml
[project]
dependencies = [
    # ... existing ...
    "apscheduler ~= 4.0",
    "litellm ~= 1.x",   # already present via marsys, but verify export
]
```

### 4.12 Tests

- **Pytest unit:** Inbox priority queue ordering; durable persistence + replay; EventIngress routing; Scheduler one-off + recurring + cancel; each watcher's tick logic against a fixture state; Heartbeat cadence (with mocked clock); Cost-meter accounting; Voice-drift detector against fixture personas; Six-axis system prompt assembly produces stable output for stable inputs; ExecuteShellTool hard-rail-confirmation gating.
- **Pytest integration:** Daemon-startup + onboarding-mode + picker-submission + running-mode transition; Heartbeat fires within a minute of expected interval (slow test, run nightly); Watcher emits an event when its condition trips (synthetic state); Crash recovery — kill the daemon mid-event-processing, restart, verify the durable inbox replays; Hard-rail confirmation flow end-to-end (agent emits tool_call → confirmation event in inbox → user confirms via API → tool runs).
- **Per-platform sandbox tests:** Linux + macOS + Windows CI matrix for `ExecuteShellTool` against the OS-level outer envelope (Session 06's launcher).
- **Manual-verify checklist:** Pick each archetype on first run; verify the persona YAML matches; verify the orb visual variant matches; trigger a heartbeat; check the inbox replay after a kill.

## 5. What is OUT of scope

| Out of scope in Session 07 | Lands in |
|---|---|
| Read tools beyond `read_file` (lookup_facts, recall, verify_fact, confirm_with_user) | Session 08 |
| Write tools (`commit_fact_now`, `set_active_context`, `add_run_note`, etc.) | Session 08 |
| Workflow-dispatch tool (`run_workflow`) | Session 08 |
| Persona-evolution mechanism (PersonaReflection consolidation stage, proposals, CLI approve/reject) | Session 08 |
| Consolidation pass (extract → adjudicate → TMS gate → routing → conflict resolution → apply) | Session 08 |
| Supersession algorithm | Session 08 |
| Suggest-with-confirm flow for write tools beyond `execute_shell` | Session 08 |
| Hard-rail tool catalog (delete_workflow, modify_settings, etc.) | Session 08 |
| Standing approvals UI | Session 09 (the home page surfaces them) |
| Sub-instance spawning + skills | v0.4 |
| Team managers + team-scoped sandboxes | v0.4 |
| Channel adapters (Telegram, Discord, Slack) | v0.4 |
| Webhook receiver + cloudflared | v0.4 |
| Cron / schedule UI | v0.4 |
| The home page's full four-surface command center (Now / Since-you-were-away / Activity / Chat) | Session 09 (it expands the orb-only home shipped in Session 03) |
| Bond reset CLI command (`spren bond reset`) | Session 08 (alongside the persona-evolution CLI) |

## 6. Files to CREATE / MODIFY in Session 07

### To CREATE — `spren/runtime/agent/` (generic, SP-023)

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/runtime/agent/__init__.py` | Re-exports public runtime/agent API. |
| `packages/spren/src/spren/runtime/agent/inbox.py` | `Inbox` priority queue + durable persistence. |
| `packages/spren/src/spren/runtime/agent/event_ingress.py` | `EventIngress` central router. |
| `packages/spren/src/spren/runtime/agent/scheduler.py` | APScheduler wrapper. |
| `packages/spren/src/spren/runtime/agent/watcher.py` | `Watcher` ABC + `WatcherResult`. |
| `packages/spren/src/spren/runtime/agent/heartbeat.py` | `Heartbeat` primitive. |
| `packages/spren/src/spren/runtime/agent/loop.py` | `AgentLoop` ABC. |
| `packages/spren/src/spren/runtime/agent/scratchpad.py` | Atomic-write scratchpad pattern. |
| `packages/spren/src/spren/runtime/agent/cost_ceiling.py` | `CostMeter`. |
| `packages/spren/src/spren/runtime/agent/sandbox_exec.py` | Generic `ExecuteShell` primitive. |
| `packages/spren/src/spren/runtime/agent/events.py` | Generic `Event`, `Priority`, `AgentId` Pydantic types. |

### To CREATE — `spren/agent/` (Spren-specific)

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/agent/__init__.py` | Module init. |
| `packages/spren/src/spren/agent/main_agent.py` | `MainAgent` Spren-specific composition. |
| `packages/spren/src/spren/agent/persona.py` | Persona reader/writer; voice-drift detector. |
| `packages/spren/src/spren/agent/archetypes.py` | 5-archetype loader. |
| `packages/spren/src/spren/agent/onboarding.py` | First-run state machine. |
| `packages/spren/src/spren/agent/system_prompt.py` | Six-axis assembly. |
| `packages/spren/src/spren/agent/doctrine.py` | `rules.md` reader. |
| `packages/spren/src/spren/agent/active_context.py` | `active_context.md` reader/writer (plus the active-session-buffer pattern from M18). |
| `packages/spren/src/spren/agent/active_todos.py` | `active_todos.md` reader/writer. |
| `packages/spren/src/spren/agent/orb_state.py` | Orb reactive-state derivation. |
| `packages/spren/src/spren/agent/api.py` | New FastAPI routes. |
| `packages/spren/src/spren/agent/events.py` | Spren-specific event-kind constants + payload types. |
| `packages/spren/src/spren/agent/policy.py` | Standing-approvals policy reader (`main.policy.yaml`). |
| `packages/spren/src/spren/agent/tools/__init__.py` | Tool registry init. |
| `packages/spren/src/spren/agent/tools/execute_shell.py` | `ExecuteShellTool` Spren binding. |
| `packages/spren/src/spren/agent/tools/schedule_event.py` | `schedule_event` tool — agent can schedule its own deferred wake-ups. |
| `packages/spren/src/spren/agent/tools/schedule_reminder.py` | `schedule_reminder` tool. |
| `packages/spren/src/spren/agent/watchers/__init__.py` | Watcher registry init. |
| `packages/spren/src/spren/agent/watchers/budget_watcher.py` | `BudgetWatcher`. |
| `packages/spren/src/spren/agent/watchers/disk_watcher.py` | `DiskWatcher`. |
| `packages/spren/src/spren/agent/watchers/provider_health_watcher.py` | `ProviderHealthWatcher`. |
| `packages/spren/src/spren/agent/watchers/pending_facts_watcher.py` | `PendingFactsWatcher` (uses Session 06's queue). |
| `packages/spren/src/spren/agent/watchers/index_drift_watcher.py` | `IndexDriftWatcher`. |
| `packages/spren/src/spren/agent/watchers/workflow_run_watcher.py` | `WorkflowRunWatcher` (uses Session 04's runs table). |
| `packages/spren/src/spren/agent/data/archetypes/ember.yaml` | Ember 8-axis YAML (from `05-five-archetypes.md`). |
| `packages/spren/src/spren/agent/data/archetypes/flint.yaml` | Flint 8-axis YAML. |
| `packages/spren/src/spren/agent/data/archetypes/quill.yaml` | Quill 8-axis YAML. |
| `packages/spren/src/spren/agent/data/archetypes/vesper.yaml` | Vesper 8-axis YAML. |
| `packages/spren/src/spren/agent/data/archetypes/kindle.yaml` | Kindle 8-axis YAML. |
| `packages/spren/src/spren/agent/data/global_voice_floor.yaml` | Global voice-tells blacklist (applies regardless of archetype). |
| `packages/spren/src/spren/agent/data/rules_default.md` | Default doctrine. |
| `packages/spren/src/spren/agent/data/active_context_template.md` | Seed for first-launch active_context.md. |
| `packages/spren/src/spren/agent/data/provider_defaults.yaml` | Per-provider cheap/expensive default model identifiers (verified at session start). |
| `packages/spren/src/spren/agent/onboarding_provider.py` | Provider auth step (validates credentials via smoke call to the provider's API). |
| `packages/spren/src/spren/agent/onboarding_models.py` | Model selection step (loads provider_defaults.yaml; validates against framework's SUPPORTED_MODELS where applicable). |
| `packages/spren/src/spren/agent/credentials_store.py` | Wraps Session 01's secrets storage; provider-credentials read/write/validate API. |

### To CREATE — frontend

| Path | Purpose |
|---|---|
| `apps/web/src/routes/onboarding.tsx` | Onboarding parent route + progress indicator. |
| `apps/web/src/routes/onboarding/provider.tsx` | Step 1 — provider auth picker. |
| `apps/web/src/routes/onboarding/models.tsx` | Step 2 — cheap + expensive model selection. |
| `apps/web/src/routes/onboarding/archetype.tsx` | Step 3 — 5-archetype picker. |
| `apps/web/src/components/ProviderCard/ProviderCard.tsx` | One provider card with credential entry inline-expand. |
| `apps/web/src/components/ProviderCard/ProviderCard.css` | Provider card styling. |
| `apps/web/src/components/ArchetypeCard/ArchetypeCard.tsx` | One archetype card with embedded orb variant. |
| `apps/web/src/components/ArchetypeCard/ArchetypeCard.css` | Card styling. |
| `apps/web/src/components/Spren/Variants.ts` | Per-archetype orb visual variants (gradient stops, breath period, grain density). |
| `apps/web/src/lib/onboarding.ts` | Onboarding API client (provider + models + archetype). |
| `apps/web/src/main.tsx` | (modify) onboarding-state check on mount; route to `/onboarding/<current_step>` if not completed. |
| `apps/web/src/routes/__root.tsx` | (modify) load onboarding state in the root loader. |

### To CREATE — tests

| Path | Purpose |
|---|---|
| `packages/spren/tests/runtime/agent/test_inbox.py` | Priority queue + durable persistence + replay. |
| `packages/spren/tests/runtime/agent/test_event_ingress.py` | Routing logic. |
| `packages/spren/tests/runtime/agent/test_scheduler.py` | One-off + recurring + cancel. |
| `packages/spren/tests/runtime/agent/test_heartbeat.py` | Cadence with mocked clock. |
| `packages/spren/tests/runtime/agent/test_cost_meter.py` | Daily total + per-think cap + budget guard. |
| `packages/spren/tests/runtime/agent/test_sandbox_exec.py` | (extends Session 06's runtime/sandbox tests) ExecuteShell primitive. |
| `packages/spren/tests/agent/test_persona.py` | PersonaReader + voice-drift detector. |
| `packages/spren/tests/agent/test_archetypes.py` | Archetype YAML load + write to personas/main.yaml. |
| `packages/spren/tests/agent/test_onboarding.py` | First-run state machine across 3 steps. |
| `packages/spren/tests/agent/test_onboarding_provider.py` | Per-provider auth smoke (mocked provider responses). |
| `packages/spren/tests/agent/test_onboarding_models.py` | Per-provider default model selection + custom override validation. |
| `packages/spren/tests/agent/test_credentials_store.py` | Read/write/validate flow against the secrets storage. |
| `packages/spren/tests/agent/test_system_prompt.py` | Six-axis assembly stability. |
| `packages/spren/tests/agent/test_main_agent.py` | Main agent loop end-to-end (with mocked LLM). |
| `packages/spren/tests/agent/test_orb_state.py` | Reactive-state derivation. |
| `packages/spren/tests/agent/test_execute_shell_tool.py` | Hard-rail confirmation gating. |
| `packages/spren/tests/agent/watchers/test_budget_watcher.py` | Each watcher gets its own test file. |
| `packages/spren/tests/agent/watchers/test_*.py` | (One per watcher.) |
| `packages/spren/tests/integration/test_daemon_lifecycle.py` | Boot → onboarding → picker → running → shutdown → restart. |
| `packages/spren/tests/integration/test_archetype_voice_samples.py` | (Voice-tests-as-CI per S4 from synthesis.) For each archetype, given a canonical scenario (e.g., failed-run report), verify Spren's output matches the voice spec — no banned phrases, contains expected vocabulary, length within bounds. Uses cassette-replayed LLM responses. |
| `apps/web/tests/onboarding-picker.test.tsx` | Archetype picker UI. |
| `apps/web/tests/e2e/onboarding.spec.ts` | Playwright: fresh data dir → daemon launches → picker visible → click archetype → home loads with correct orb. |

### To MODIFY

| Path | Edit |
|---|---|
| `packages/spren/pyproject.toml` | Add `apscheduler` dep. |
| `packages/spren/src/spren/server.py` | Lifespan startup hook orchestrates: memory bootstrap (S06), onboarding-state check, agent runtime start (in running mode). New routes registered (onboarding, meta, agent). |
| `packages/spren/src/spren/storage/migrations/05__create_apscheduler_tables.py` | APScheduler 4.x SQLAlchemy job-store schema. |
| `packages/spren/src/spren/__init__.py` | Re-export `runtime.agent` and `agent` subpackages. |
| `apps/web/src/components/Spren/Spren.tsx` | (Session 03 already shipped the orb component.) Modify to support archetype variants via a new `archetype` prop that swaps gradient stops, breath period, grain density. |
| `Justfile` | Add `just onboarding-skip ARCH=<name>` recipe (CLI override). |
| `docs/architecture/spren/09-meta-agent.md` | Mark §"Onboarding — five archetypes" + §"Persona-evolution mechanism" + §"Bond-violation handling" sections as "shipping in Session 07 (onboarding) + Session 08 (evolution)". |

### To DELETE

The Vesper placeholder persona seeded in Session 06 (`memory/bootstrap.py:write_placeholder_persona`) is no longer needed once Session 07's archetype picker lands. The Session 07 implementer removes that placeholder write and the associated test. The bootstrap path becomes: KB tree + empty `personas/main.yaml` directory; the picker writes the actual persona on first user click.

## 7. User journeys (anchor for Bundle 03 demo gate)

### J-4 — First-run onboarding flow (3 steps)

State: fresh data dir; daemon launched.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User runs `just dev-desktop`. | Tauri shell + sidecar | Daemon starts in onboarding mode (memory bootstrap from Session 06 runs; agent loop NOT started). Tauri webview loads. |
| 2 | App detects `onboarding.completed: false`; routes to `/onboarding/provider`. | Onboarding step 1 | Provider cards visible: Anthropic, Anthropic OAuth, AWS Bedrock, OpenRouter, OpenAI, Azure OpenAI, Google AI, DeepSeek. Anthropic API is highlighted as default. |
| 3 | User clicks Anthropic API. | Card expands inline | API-key entry field appears in the card; "Save" button. |
| 4 | User pastes API key, clicks Save. | `POST /v1/onboarding/provider` | Backend smoke-tests the key with a lightweight Anthropic API call (e.g., `GET /v1/models`). On success: settings write `meta_agent.provider = "anthropic"` + credentials stored via Session 01's secrets store. Advance to step 2. |
| 5 | App routes to `/onboarding/models`. | Onboarding step 2 | Two dropdowns visible: Cheap (default `claude-haiku-4-5`) + Expensive (default `claude-sonnet-4-6`). User can change either. "Use defaults" button accepts both. |
| 6 | User clicks "Use defaults". | `POST /v1/onboarding/models` | Settings write `meta_agent.model.cheap = "claude-haiku-4-5"` + `meta_agent.model.expensive = "claude-sonnet-4-6"`. Advance to step 3. |
| 7 | App routes to `/onboarding/archetype`. | Onboarding step 3 | 5 archetype cards visible: Ember, Flint, Quill, Vesper, Kindle. Each shows its orb variant + 3-line description. Orbs animate in incommensurate cadences (9s/10s/11.5s/12s/15s). |
| 8 | User hovers each card. | Card hover state | Each card lifts slightly; the orb's breath subtly accelerates on hover. |
| 9 | User clicks "Quill". | Card click | The chosen orb scales up; the others fade out. `POST /v1/onboarding/archetype` fires with `{"archetype": "quill"}`. |
| 10 | Backend writes `personas/main.yaml` with Quill's 8-axis defaults + `archetype: quill, archetype_chosen_at: <ts>, evolved_axes: []`. Sets `settings.meta_agent.archetype = "quill"`. Transitions daemon from onboarding mode to running mode (starts agent loop, watchers, heartbeat). | (server-side) | — |
| 11 | Frontend receives the success response. | Redirect to `/` | Home page loads with the chosen Quill orb at center. The orb breathes in Quill's cadence. |
| 12 | Spren's first message (the welcome from Quill's voice samples in `05-five-archetypes.md` §Quill §Voice samples §"First-run welcome"). | Home page | Greeting in Quill's voice. |

### J-5 — Heartbeat tick

State: daemon running, user picked Vesper, hour 0 since last activity.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | 30 minutes pass. | (no user action) | `Heartbeat.start()` coroutine sleeps 30 min, emits `HeartbeatTick` event into main agent's inbox at P3. |
| 2 | Main agent loop dequeues the event. | (server-side) | System prompt assembled; LLM call with the heartbeat-prompt extension ("review since last tick — what changed, what needs attention, what should I surface?"). |
| 3 | LLM responds — for v0.3 with no real workflows running, the response is typically "nothing to surface". | (server-side) | Activity log gets an entry (`heartbeat_completed`); cost meter records the call (~$0.0001 with the cheap model). No user-visible action. |
| 4 | Hour 1: user has run a workflow that failed twice in a row. | Background event | `WorkflowRunWatcher` emits `WorkflowRunStalled` event. Main agent receives at P2. |
| 5 | Main agent surfaces the issue: "I noticed `research-pipeline` failed twice today — same step both times. Want me to investigate?" | Home page chat surface (Session 09 wires this fully; Session 07 ships the underlying SSE event) | The orb's reactive state shifted to `thinking` while the agent reasoned, then `idle`. |

### J-6 — Hard-rail shell-execute confirmation

State: Spren running (any archetype); user is in a chat conversation; user asks "can you grep for 'TODO' in my project"?

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User sends message. | Home page chat | `POST /v1/meta/messages`. |
| 2 | Main agent receives event; reasons; emits a tool_call for `execute_shell` with `cmd=["grep", "-rn", "TODO", "."]`, `cwd=<project-path>`. | (server-side) | — |
| 3 | Main loop checks `tool.requires_confirmation` → True. Checks standing approvals → none granted yet. Enqueues a `NeedsDecision` event with the tool_call payload. | (server-side) | — |
| 4 | Spren's chat response: "I want to run `grep -rn TODO .` in `<project-path>`. Confirm to proceed." With buttons or a CLI prompt to confirm. | Home page chat | Confirmation UI surfaces. |
| 5 | User clicks "Confirm" (or replies "yes"). | Frontend → `POST /v1/agent/confirm/<id>` | The deferred tool call executes; subprocess runs sandboxed; returns stdout. |
| 6 | The result feeds back into the agent's next turn; agent's response includes the grep output. | Home page chat | "Found 7 TODOs..." |
| 7 | Audit log records the call: `<data-dir>/logs/shell-audit.log` gets `{ts, agent_id: main, cmd: "grep -rn TODO .", cwd: <project-path>, exit_code: 0, duration_ms: 45}`. | (server-side) | — |

## 8. Decisions locked

1. **Daemon startup state machine.** Onboarding mode (memory bootstrap; only `/v1/onboarding/*` routes active; agent loop NOT started) → archetype picker submission → running mode (agent loop, watchers, heartbeat all started). The transition is irreversible from running mode without `spren bond reset` (Session 08).

2. **Inbox uses Python `asyncio.PriorityQueue` for in-memory state + atomic-write JSON files in `<sandbox-root>/inbox/pending/<priority>/<event-id>.json` for durability.** Persistence happens on `put()` for P0+P1 events (so they survive crash); P2+P3 are best-effort. On graceful shutdown, all queued events flush to disk. On crash, restart replays.

3. **Watcher catalog is fixed in code per version.** Agents do NOT spin up new watchers at runtime in v0.3/v0.4. The catalog ships with: `BudgetWatcher`, `DiskWatcher`, `ProviderHealthWatcher`, `PendingFactsWatcher`, `IndexDriftWatcher`, `WorkflowRunWatcher`. v0.4 adds `SubInstanceLivenessWatcher`, `ChannelHealthWatcher`, `TunnelWatcher`, `VolatilityReverifyWatcher`.

4. **Heartbeat default 30 min, configurable per-user in settings.** Common patterns: 5 min for active monitoring; 6 hours for low-cost mode.

5. **Six-axis system prompt assembly is one large string per turn.** Axes 1-5 eagerly loaded; axis #6 (Memory) is via tool calls. The prompt is rebuilt every turn (not cached) so changes to persona / doctrine / active_context / capabilities reflect immediately. Rebuild cost is ~ms per turn — negligible.

6. **`execute_shell` is a hard rail by default.** Standing approvals scoped per-tool + per-cwd-prefix + per-cmd-prefix. Examples of approvable scopes: `git in <repo>` (allows `git status`, `git diff`, etc., but not `git push`), `pnpm in <project>`. The user can edit `<sandbox-root>/shared/personas/main.policy.yaml` directly to grant standing approvals.

7. **Voice-drift warning is logging-only in v0.3.** Hits go into the session log as `voice_drift` events; the message itself is not modified. Consolidation pass (Session 08) reads the events; persona-evolution may propose recalibration based on accumulated drift.

8. **Cost meter check happens before each agent turn, not per LLM call within a turn.** A turn might make multiple LLM calls (extraction classifier + main reasoner + tool result analyzer); the cost meter accumulates. The check is "do I have budget for at least the cheapest reasonable turn?" before dequeueing the event.

9. **Provider-aware model defaults.** Settings `meta_agent.model.cheap` and `meta_agent.model.expensive` defaults are per-provider, picked at onboarding Step 2 from `agent/data/provider_defaults.yaml`. Implementer verifies model identifiers against the framework's adapter `SUPPORTED_MODELS` lists at session start; updates the defaults file. The starter table (Anthropic API: haiku-4-5 / sonnet-4-6; Anthropic OAuth: haiku-4-5 / opus-4-7; Bedrock: haiku-4-5 / sonnet-4-6 routed; OpenRouter: anthropic/haiku-4-5 / anthropic/sonnet-4-6; OpenAI: gpt-5.4-mini / gpt-5.4; Azure: same as OpenAI Azure-routed; Google: gemini-3.1-flash / gemini-3.1-pro; DeepSeek: deepseek-v4-mini / deepseek-v4-pro). Selection per turn: routine events (workflow status, heartbeat, simple confirmations) use cheap; complex reasoning (consolidation, multi-step planning, hard-rail decisions) use expensive. The selection logic (`agent/model_selection.py`) is a simple mapping of event kind → model tier; refined organically.

10. **Onboarding is 3 steps in v0.3, all required.** Step 1 provider auth → Step 2 model selection (with provider defaults) → Step 3 archetype picker. The user picks an LLM provider, validates credentials via a real smoke call to the provider, picks cheap + expensive models (or accepts the per-provider defaults), then picks an archetype before the daemon enters running mode. The GUI flow doesn't have skip buttons. CLI override (`spren onboarding pick --provider <name> --cheap <model> --expensive <model> --archetype <name>`) exists for headless / scripted setups (demo videos, test fixtures, scripted installs).

11. **Default provider for first-time users: Anthropic API (Claude-friendly).** The provider picker highlights Anthropic API as the default selection. Reasoning: most v0.3 early users are Claude-friendly (the product team's dogfood + early-adopter audience). The user can pick any other provider; the default is just a starting point. v0.4 may revisit the default once we have onboarding telemetry.

12. **Provider-credentials live in the existing secrets store from Session 01.** OS keychain primary; encrypted SQLite fallback. The credentials_store module wraps Session 01's pattern + adds per-provider validation. No new credential storage mechanism.

11. **`active_context.md` does double duty per M18 from synthesis.** Current focus state + active-session adjustments to behavior. Re-read every turn. Cleared / condensed by the consolidation pass (Session 08). Soft cap on size (default 2K tokens); on overflow, agent compresses older entries into a summary line.

## 9. Polish items to address inside Session 07

1. **Onboarding picker layout responsiveness.** Step 1 provider cards: 4 cards per row at desktop (1280+); 2 per row at tablet; vertical at mobile. Step 3 archetype cards: 5 in a row at desktop, 2x2 + 1 at tablet, vertical at mobile. The orbs maintain their incommensurate-cadence animation regardless of layout.

1a. **Provider credential validation UX.** When the credential smoke call fails (bad API key, network issue, model access not granted on Bedrock), the error needs to be specific. Implementer wires per-provider error mapping from the framework adapter's exceptions into actionable user-facing messages. Generic "auth failed" is not acceptable.

1b. **Anthropic OAuth flow.** The OAuth callback needs to land on a localhost URL the daemon listens on temporarily during onboarding. Implementer uses a one-shot HTTP listener at a free port; verifies the OAuth token; closes the listener. The token (or refresh token) goes into the secrets store. v0.3 implementer: confirm whether the framework's existing `anthropic_oauth.py` adapter handles the OAuth flow internally OR expects Spren to drive the OAuth interaction.

2. **Inbox replay on crash recovery.** Verify that if the daemon crashes mid-event-processing, the replay doesn't double-process the in-flight event. Use a lockfile (`<sandbox-root>/inbox/processing/<event-id>.lock`) that's created when the event is dequeued and removed when processing completes. On restart, any event in the `processing/` dir is moved back to `pending/<priority>/` for replay.

3. **APScheduler with the SQLAlchemy job store: schema migrations.** APScheduler 4.x's job store schema may differ from 3.x; the implementer pins the version + the schema, runs migrations as Spren's own migration (number 05), and verifies idempotent restart.

4. **Watcher coroutine lifetime.** Each watcher is `asyncio.create_task(...)`-spawned. On graceful shutdown, all are cancelled with timeout; their state (last-checked, last-fired) is saved to settings table for restart continuity.

5. **`execute_shell` truncation at 4K chars per stream.** Tail-truncation (keep the end) — failures usually surface near the bottom. The agent sees the truncated content + a `truncated: true` flag and can call again with a more specific cmd if it needs more.

6. **Voice-drift regex performance.** At 5K-token agent responses, the post-pass scan should complete in <10ms. If a per-archetype `avoid` list grows past ~30 phrases, build a single compiled regex (alternation) instead of looping per phrase.

7. **Onboarding picker accessibility.** Each card is keyboard-navigable (Tab to focus, Enter to select); orbs respect `prefers-reduced-motion` (drop the breath animation, replace with static gradient). Screen-reader labels match the visible card text.

8. **Cost meter persistence.** Daily totals reset at midnight UTC (configurable per-user). The meter persists state in `<data-dir>/data/spren.db` settings table (`cost.daily_total_usd`, `cost.daily_window_start`); on restart, it loads.

9. **Cheap-vs-expensive model fallback.** If the configured cheap model is unavailable (provider down), the cost-meter check + a fallback table determines whether to upgrade to expensive (allowed for P0; refused for P3 — heartbeat just skips).

10. **The orb's reactive state during a hard-rail confirmation pause.** While waiting for user confirmation, the orb's state is `idle` (not `thinking` — the agent has paused, not is reasoning). The Run button on the canvas (Session 04) and the home-page orb both reflect this consistently.

## 10. Success criteria

- **G-24** (onboarding flow): fresh data dir → daemon launches → step 1 (provider Anthropic API + key) → step 2 (default models accepted) → step 3 (user picks Ember) → home loads with the Ember orb breathing at 12s cadence; `personas/main.yaml` matches Ember's defaults; settings has `meta_agent.provider = anthropic`, `meta_agent.model.cheap`, `meta_agent.model.expensive`, `meta_agent.archetype = ember`.
- **G-24a** (per-provider onboarding): fresh data dir → user picks each provider in turn (across separate test runs) → each completes its auth flow → defaults pre-populate per the provider table → user accepts defaults → onboarding completes.
- **G-25** (each archetype loads correctly): repeat G-24 for each of the 5 archetypes; each produces a distinct orb visual + a distinct persona YAML.
- **G-26** (heartbeat tick): with daemon running and 30-min cadence configured, a heartbeat fires within the expected window; main agent processes it; activity log records `heartbeat_completed`.
- **G-27** (watcher fires): synthetic state simulating "spend reaches 100% of daily cap" → `BudgetWatcher` emits `BudgetThresholdCrossed` event at P0; main agent dequeues it; surfaces a notification to the user.
- **G-28** (`execute_shell` hard rail): user asks Spren to run a shell command; confirmation flow surfaces; user confirms; subprocess runs sandboxed; result feeds back to the agent.
- **G-29** (voice-drift logged): a fixture agent response containing "Great question!" (banned phrase) produces a `voice_drift` event in the session log.
- **G-30** (crash recovery): kill the daemon mid-event-processing; restart; the in-flight event is replayed without double-processing.
- **G-31** (cost-budget guard): daily budget exhausted → P3 heartbeat events are deferred; P0 user-direct events still process; user sees a "budget exhausted" notification.
- **C-09** (no SP-023 violation): SP-023 boundary test extended for new files; `runtime/agent/` does not import from `spren.agent` or `spren.memory`.
- **C-10** (per-archetype voice tests): each archetype's voice CI test passes — no banned phrases, expected vocabulary, length within bounds, against canonical scenarios from `05-five-archetypes.md`.
- **U-15** (manual smoke): from a fresh data dir, complete onboarding with each archetype in turn (clear data dir between); verify the felt-experience matches `05-five-archetypes.md`'s voice samples.

## 11. Open research items the implementer resolves in-flight

- APScheduler 4.x version + the SQLAlchemy job-store schema. Verify against current major version at session start.
- **LLM model identifiers per provider — VERIFY AT IMPLEMENTATION TIME.** The defaults in `provider_defaults.yaml` (haiku-4-5, sonnet-4-6, opus-4-6, gpt-5.4-mini, gemini-3.1-pro, etc.) are draft-time best-guesses. Implementer cross-checks each against:
  1. The framework's per-adapter `SUPPORTED_MODELS` lists where present (`packages/framework/src/marsys/models/adapters/anthropic_oauth.py` lists Claude variants explicitly; `openai_oauth.py` lists GPT-5+ variants; verify the Anthropic API + OpenAI + Google + Bedrock + OpenRouter + DeepSeek adapters' supported lineup).
  2. Each provider's current public model catalog at the time of implementation. Update `provider_defaults.yaml` accordingly. Pricing at the time of implementation determines what counts as "cheap" vs "expensive" for that provider — the cheap default should be ≤ $0.50/M tokens input; the expensive default should be the strongest reasoning model in the provider's lineup.
  3. The framework adapter's model-name normalization (some adapters accept aliases like `claude-sonnet-4.6` AND `claude-sonnet-4-6`; pick the canonical form for settings storage).
- Voice-tests-as-CI: how to record + replay LLM responses for deterministic test assertions. VCR-style cassettes via `vcrpy` or `litellm` cassette mode; the implementer picks. The fixture flow: real LLM call once during cassette generation; recorded response replays in CI.
- Inbox lockfile semantics on Windows. UNIX `flock` doesn't have a direct Windows equivalent; pick a portable mechanism (e.g., `portalocker` library or atomic-rename-pattern lockfiles).
- The `voice.style_tells.avoid` regex compilation strategy at 30+ phrases — benchmark loop-per-phrase vs compiled-alternation; pick the faster.
- Interaction between agent loop and FastAPI request handlers. Both run on the same asyncio loop; verify that long agent reasoning doesn't block HTTP responses (the FastAPI endpoints should always return quickly; agent work happens in background tasks).

## 12. Status

- [ ] Tier confirmed (CRITICAL).
- [ ] Scope boundaries confirmed.
- [ ] SP-023 boundary confirmed.
- [ ] Files-to-CREATE list approved.
- [ ] Three user journeys approved.
- [ ] Decisions locked.
- [ ] Polish items captured.
- [ ] Success criteria affirmed.
- [ ] Acceptance criteria frozen at [`./07-meta-agent-core/acceptance.md`](./07-meta-agent-core/acceptance.md).
- [ ] Architect peer review at Stages 1, 3, 4 (CRITICAL multi-checkpoint).
- [ ] Session 06 implementation complete (Session 07 hard-depends on it).
- [ ] Session implementation complete.

## 13. Open questions — resolved

All Session 07 questions are resolved.

1. **Anthropic OAuth in v0.3.** **Locked: ships in v0.3.** The `anthropic_oauth.py` adapter is already present; the OAuth flow follows the `gh auth login` pattern (browser handoff to Claude.ai, callback to a one-shot localhost listener). Implementer wires the credential exchange + storage via the existing secrets store. OpenAI OAuth + Google OAuth defer to v0.4.

2. **AWS Bedrock and other-provider error handling — single error class, not per-error-class.** **Locked: Bedrock ships in v0.3 with one unified provider-credential error class.** Provider-credential failures (bad key, model access denied, region wrong, token expired) all surface through the same `ProviderAuthError` path with structured remediation hints in the message. We do NOT add a separate "model access denied" error class — that would mean the architecture is wrong: every credential or access problem at onboarding-validate time funnels through one auth-error path with informative remediation strings. Implementer maps the framework adapter's exception types into structured `{kind: "auth", remediation: <action>, message: <user-facing>}` payloads; the UI renders the remediation as a clickable next step (e.g., "Request `anthropic.claude-haiku-4-5` access in your AWS console: <link>"). If we ever find ourselves wanting a separate error class for a specific failure mode, that's a signal the auth-error path itself needs to be broader, not that we need more classes.

3. **Default models — implementer picks the latest in the tier; `claude-opus-4-7` is now in the catalog.** **Locked.** The brief specifies the *tier* (cheap = haiku-class; expensive = sonnet-or-opus-class per provider). At implementation time, the implementer cross-checks the current public model catalog for each provider; updates `provider_defaults.yaml` to whichever specific identifiers are the latest in each tier. The Anthropic OAuth row's expensive-default lifts to `claude-opus-4-7` (which exists at session-implementation time). Settings tracks the chosen identifier so the user knows what's running and can switch via the settings UI later.
