# Framework Session 02: `TelemetrySink` ABC + `TraceCollector` Hook

Required by Spren v0.4 (`SprenTelemetrySink` adapter at `packages/spren/src/spren/telemetry/`, the `python my_workflow.py` → Spren UI flow, and the `v0.4-27` plan entry in [`../../../spren/v0.4-extensions.md`](../../../spren/v0.4-extensions.md)).

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of the marsys framework.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if a structural choice will cause us pain later, say so. Defend with evidence.
- **Stay engaged.** Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it. If a Spren-side assumption embedded in this brief doesn't hold from the framework's perspective, push back loudly.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Manually verify acceptance criteria, not just unit tests.
- **You own follow-through.** Update this file's "What was actually built". Update [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md) checkbox. Add "Lessons / Surprises" if anything surprised you.

### Double-check before any decision

- **Read the framework code before changing it.** Don't assume; verify.
- **Verify file paths and symbols still exist** before referencing them.
- **Run framework tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints.**

### Critically assess the plan itself

This brief was rewritten after Session 01 shipped. Verify the framework-internal facts (current `TraceCollector` API, the `Span` shape, the `Orchestra._initialize_components` wiring) before writing code. If anything has drifted since this brief was written, **stop and update the brief or escalate**.

### Multi-consumer justification (mandatory)

Every framework user who needs to forward execution traces to a hosted observability backend benefits from a generic sink seam. Spren is one consumer; the abstraction is intentionally shaped to fit LangSmith, Phoenix (OTel), and Langfuse adapters one-to-one. The vendors' SDKs share a common shape (span-based, hierarchical, async-batched) and the protocol matches it so a third party can ship a `LangSmithTelemetrySink` / `PhoenixTelemetrySink` / `LangfuseTelemetrySink` package without touching the framework.

**Forbidden:** any code path special-cased for Spren, or any protocol concept that only Spren can satisfy. The framework knows about the protocol; it knows nothing about specific backends.

### Foundational project rules

- The framework worktree's `CLAUDE.md` — TRUNK-CRITICAL list, framework design principles DP-001..DP-007
- Framework architecture docs in the framework worktree
- [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md)
- [`../../../architecture/spren/01-system-context.md`](../../../../architecture/spren/01-system-context.md) § "Framework contributions Spren motivates"
- [`../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-018 (framework purity), SP-019 (API as truth)

---

## The big picture — what we're building and why

### What this PR ships

Session 01 already shipped streaming NDJSON tracing: `TraceCollector._stream_span` (`coordination/tracing/collector.py:403-417`) fans every closed span to a list of writers via the `TraceWriter` ABC's `write_span` hook. That gets traces to disk, line by line, as the run unfolds.

What's missing: a way for external observability backends — Spren's daemon, a hosted LangSmith project, a Phoenix server, a Langfuse instance, MARSYS Cloud, custom HTTP collectors — to receive the same per-span stream over the network.

This PR replaces `TraceWriter` with `TelemetrySink`, an abstraction at the right level: a sink consumes closed spans and forwards them somewhere. NDJSON-to-disk is one kind of sink. Spren-over-HTTP is another. LangSmith via `create_run` is another. The framework knows about the abstraction; adapters live in their own packages and translate to whatever vendor API they target.

### Writer is a special case of sink

The current `TraceWriter` ABC was named when on-disk persistence was the only output path. After Session 01 introduced streaming, the abstract `write(trace: TraceTree)` method became vestigial — `NDJSONTraceWriter.write` is a no-op kept only for ABC compat (`writers/ndjson_writer.py:148-155`). The real work happens in `write_span(span)`, which any sink needs.

This PR:

- Introduces `TelemetrySink` at `coordination/tracing/sink.py` with two abstract methods: `publish_span(span)` and `close()`.
- Deletes `TraceWriter` entirely (no deprecated alias). The vestigial `write(trace)` method goes with it.
- Reclassifies `NDJSONTraceWriter` to inherit from `TelemetrySink` directly. Same class name, same module path, same behavior. The `write_span` method renames to `publish_span`.
- Renames `TraceCollector.writers` (kwarg + attribute) to `sinks`.
- Adds `SecretRedactor` at `coordination/tracing/redactor.py`. It runs once at the fan-out boundary inside `TraceCollector._stream_span`, scrubbing API keys, tokens, passwords, and similar from `span.attributes`, every `event['attributes']` dict in `span.events`, and every `link['attributes']` dict in `span.links`. All consumers (NDJSON-to-disk, vendor sinks, in-memory `TraceTree`) see the same redacted view.
- Plumbs sinks through `TracingConfig.sinks` and the redactor through `TracingConfig.redactor`. Zero new kwargs on `Orchestra`.

### Vendor mapping (current 2026 state)

All three reference vendors converge on OpenTelemetry by mid-2026, but the framework keeps a custom abstraction to avoid pulling in `opentelemetry-sdk` as a runtime dependency. Adapters wrap their preferred OTel exporter or vendor SDK; that coupling lives in the adapter package, not the framework.

| Vendor | Native API | Adapter shape |
|---|---|---|
| **LangSmith** | `Client.create_run(name, run_type, inputs, outputs, parent_run_id, start_time, end_time)`. OTel-native (`LANGSMITH_OTEL_ENABLED=true`) is now also recommended. | 5-line `publish_span` body translating `span.kind` → `run_type`, `span.parent_span_id` → `parent_run_id`. |
| **Phoenix** | Pure OTLP through `opentelemetry-sdk`'s `BatchSpanProcessor` pointed at `PHOENIX_COLLECTOR_ENDPOINT`. | Adapter holds an OTel `Tracer`, calls `start_span(name, attributes=...)` then `.end(end_time)` per `publish_span`. |
| **Langfuse** v4 | `langfuse.start_observation(as_type="span"|"generation", ...)` and the context-manager `start_as_current_observation`. Explicit `flush()` / `shutdown()`. | `publish_span` opens and immediately closes an observation per closed span; `close()` calls `langfuse.shutdown()`. |
| **MARSYS Spren** | HTTP POST per span to the daemon. | `publish_span` enqueues `span.to_dict()` onto an `asyncio.Queue`; a drain task POSTs in batches. Mirrors the NDJSON queue-and-drain pattern. |
| **Generic HTTP** | User-defined endpoint. | `publish_span` POSTs `span.to_dict()` directly. |

### Why hooked into `TraceCollector`, not `EventBus` directly

`TraceCollector` is the framework's existing "subscribe to lifecycle events → produce spans" component, and it is load-bearing. Hooking sinks into it reuses the span-tree-building logic at `tracing/collector.py:82-397`, the per-listener exception handling at `_stream_span` (L403-417), and the timeout-bounded shutdown wired through `Orchestra.execute`'s finalize block (`orchestra.py:966-995`). Sinks register through `TracingConfig.sinks`; they do not subscribe to `EventBus` independently.

### Multi-consumer (mandatory)

- **Spren** ships `SprenTelemetrySink` in `packages/spren/src/spren/telemetry/` (added in Spren v0.4-27).
- **LangSmith / Phoenix / Langfuse adapters** ship as third-party packages — `marsys-langsmith`, `marsys-phoenix`, `marsys-langfuse`. Their SDK shapes match this protocol one-to-one.
- **MARSYS Cloud** uses sinks for hosted observability. Its API ingests the same per-span shape.
- **OpenInference / OpenTelemetry users** wrap a `TelemetrySink` around their pre-configured OTel `TracerProvider`.
- **Framework users running raw `python my_workflow.py`** can write a 30-line custom HTTP sink and POST spans to their own backend.

If any of these consumers cannot be served by the protocol as written, the protocol is wrong — escalate before coding.

### Your role as a framework implementer

1. Honor the framework's architecture (tracing module + `TraceCollector` as the single span-tree builder).
2. Honor the framework's design principles (DP-001..DP-007).
3. Honor multi-consumer justification (no Spren-specific paths).
4. Ship a single coherent PR with green tests.
5. Push back when something is wrong.

---

## What came before this session

**Previous framework PRs from this dir:**

- Session 01 — NDJSON streaming tracing writer ([`./01-ndjson-streaming-tracing-writer.md`](./01-ndjson-streaming-tracing-writer.md)). Shipped: `TraceCollector._stream_span` (collector.py:403-417), `TraceWriter.write_span` virtual hook (writers/base.py:34-41), `NDJSONTraceWriter` (writers/ndjson_writer.py), `NDJSONTraceReader` (readers/ndjson_reader.py), ULID migration for `event_id` / `span_id` / `trace_id`, `Orchestra` finalize block with timeout-bounded close (orchestra.py:966-995), `_collect_tracing_metadata` introspection (orchestra.py:999-1017), `result.metadata["tracing"]` surface.

**State at start of this session** (verified against current code):

- `TraceCollector.__init__` at `coordination/tracing/collector.py:39-44` accepts `event_bus`, `config`, `writers`. Subscribes to 11 event types via `_subscribe_to_events` at L59-80. Span-tree building L82-397.
- `_stream_span` helper at `collector.py:403-417` — calls `await writer.write_span(span)` on every writer, swallows + logs exceptions.
- `finalize` at `collector.py:421-471` — closes orphan spans, then loops `for writer in self.writers: await writer.write(trace)` at L463-468 (the loop this PR deletes; it only invokes the no-op `write(trace)`).
- `close` at `collector.py:473-479` — iterates writers, calls `await writer.close()`.
- `Span` and `TraceTree` at `coordination/tracing/types.py:14-89, 92-139`. `Span.attributes` is a mutable dict (line 41); `Span.events` is a list of dicts each carrying their own `attributes` sub-dict (line 42, 53-59); `Span.links` is the same shape (line 44, 61-67).
- `TraceWriter` ABC at `coordination/tracing/writers/base.py:12-41` with `write` and `close` as abstractmethod, `write_span` as a virtual default-no-op.
- `NDJSONTraceWriter` at `coordination/tracing/writers/ndjson_writer.py`. Class-level constants `SCHEMA_VERSION=1`, `DISK_ERROR_DISABLE_THRESHOLD=100`, `DISK_ERROR_LOG_PERIOD=100`, `DEFAULT_QUEUE_MAXSIZE=10000`, `CLOSE_TIMEOUT_SECONDS=5.0` at L77-81. The `write_span` method is at L131-146. The vestigial `write(trace)` no-op is at L148-155.
- `TracingConfig` at `coordination/tracing/config.py:8-22`. Five fields after `fe61a9b` removed `detail_level` and `max_content_length`.
- `Orchestra._initialize_components` constructs `TraceCollector` at `orchestra.py:207-216`.
- `Orchestra.execute` finalizes at `orchestra.py:966-995`: calls `await trace_collector.finalize(session_id)` then `await asyncio.wait_for(trace_collector.close(), timeout=NDJSONTraceWriter.CLOSE_TIMEOUT_SECONDS)` then `_collect_tracing_metadata`.
- `Orchestra._collect_tracing_metadata` at `orchestra.py:999-1017` iterates `self.trace_collector.writers` and reads `total_spans`, `disk_error_count`, `dropped_span_count`, `disabled_dropped_count`, `disabled` from the first writer that exposes them.
- `EventBus._max_listener_errors = 5` at `coordination/event_bus.py:29` (the framework's existing 5-strike pattern; not used in this session — sinks have their own 100-failure precedent from NDJSON's `DISK_ERROR_DISABLE_THRESHOLD`).
- `StatusEvent.event_id` is ULID (`coordination/status/events.py:14-23`). `Span.span_id` is ULID (factory in `tracing/types.py:142-159`).
- `python-ulid>=3.1.0,<4` is in `packages/framework/pyproject.toml` runtime deps (added by Session 01).

**Verify state with:**

```bash
cd /home/rezaho/research_projects/marsys-tracing-work/packages/framework/
source ../../.venv/bin/activate

pytest tests/ -x --tb=short                                # capture baseline test counts

grep -n 'class TraceCollector\|def _stream_span\|def finalize\|def close' src/marsys/coordination/tracing/collector.py
grep -n 'class TraceWriter\|abstractmethod\|write_span' src/marsys/coordination/tracing/writers/base.py
grep -n 'class NDJSONTraceWriter\|write_span\|async def write\b' src/marsys/coordination/tracing/writers/ndjson_writer.py
grep -n 'tracing.enabled\|writers=' src/marsys/coordination/orchestra.py
grep -n 'class TracingConfig' src/marsys/coordination/tracing/config.py
```

If any cited line number has drifted, **stop and update this brief or escalate** before coding.

---

## What this session ships

After merge:

- `coordination/tracing/sink.py` — `TelemetrySink` ABC with two abstract async methods: `publish_span(span)` and `close()`.
- `coordination/tracing/redactor.py` — `SecretRedactor` with default deny-list (`api_key, apikey, token, authorization, auth, secret, password, bearer, cookie, session, credential`, case-insensitive substring match). `redact_span(span)` walks `span.attributes`, every `event['attributes']` dict in `span.events`, and every `link['attributes']` dict in `span.links`. Mutates in place. Optional `NoRedactionRedactor` no-op for explicit opt-out.
- `coordination/tracing/writers/base.py` — DELETED. `TraceWriter` ABC removed entirely (no deprecated alias).
- `coordination/tracing/writers/ndjson_writer.py` — inheritance change `TraceWriter` → `TelemetrySink`. `write_span` renamed to `publish_span`. The vestigial no-op `write(trace)` method dropped. All other behavior identical.
- `coordination/tracing/collector.py` — `writers` kwarg + attribute renamed to `sinks`. `_stream_span` calls `redactor.redact_span(span)` before fan-out. `finalize` no longer loops over writers calling `write(trace)`. Imports updated.
- `coordination/tracing/config.py` — `TracingConfig` gains `sinks: list = []` and `redactor: SecretRedactor | None = None`.
- `coordination/orchestra.py:207-216` — reads sinks from `execution_config.tracing.sinks`. Combines with default `NDJSONTraceWriter` (NDJSON first, user sinks after). Default `SecretRedactor` instantiated lazily inside `TraceCollector` if config field is None.
- `coordination/orchestra.py:999-1017` — `_collect_tracing_metadata` iterates `trace_collector.sinks`.
- All test files importing `TraceWriter` or calling `.write_span()` updated.
- New tests covering: ABC contract, NDJSON inheritance regression, redactor (attributes + events + links), redaction at chokepoint, full `Orchestra.run()` integration with `RecordingTelemetrySink`, multi-consumer test with three fake vendor adapters.
- `CHANGELOG.md` entry under next release version.

### Acceptance criteria

- [ ] `coordination/tracing/sink.py` exists. Defines `TelemetrySink` ABC with `publish_span(span)` and `close()` as abstract async methods.
- [ ] `coordination/tracing/redactor.py` exists. Defines `SecretRedactor` with default deny-list and `redact_span(span)` that walks attributes + events + links recursively. Defines `NoRedactionRedactor` for opt-out.
- [ ] `coordination/tracing/writers/base.py` deleted. `TraceWriter` ABC and the vestigial `write(trace)` method are gone.
- [ ] `NDJSONTraceWriter` inherits from `TelemetrySink`. `write_span` method renamed to `publish_span`. The no-op `write` method is gone. All Session 01 behavior unchanged.
- [ ] `TraceCollector.__init__` takes `sinks: Optional[List[TelemetrySink]]` (renamed from `writers`). Reads redactor from `config.redactor`; default-instantiates `SecretRedactor()` if None.
- [ ] `TraceCollector._stream_span` calls `self._redactor.redact_span(span)` before the fan-out loop. Sinks receive the redacted view.
- [ ] `TraceCollector.finalize` no longer iterates writers calling `write(trace)`. The deleted loop's behavior is preserved (it was a no-op in practice).
- [ ] `TracingConfig.sinks: list = []` and `TracingConfig.redactor: Optional[SecretRedactor] = None` fields added.
- [ ] `Orchestra._initialize_components` constructs `sinks = [NDJSONTraceWriter(cfg)] + list(cfg.sinks)` and passes via `TraceCollector(sinks=sinks)`. Zero new kwargs on `Orchestra`.
- [ ] `Orchestra._collect_tracing_metadata` iterates `trace_collector.sinks`.
- [ ] All imports updated: `from .sink import TelemetrySink` replaces `from .writers.base import TraceWriter` in collector, ndjson_writer, tests.
- [ ] All `.write_span()` call sites in tests renamed to `.publish_span()` (approximately 24 in `test_ndjson_writer.py` + 8 in `test_ndjson_reader.py`, plus the fake-writer subclass method definition at `test_ndjson_writer.py:355`).
- [ ] `coordination/tracing/__init__.py` drops `TraceWriter` export, adds `TelemetrySink` and `SecretRedactor` exports.
- [ ] `coordination/tracing/writers/__init__.py` drops `TraceWriter` export.
- [ ] **Multi-consumer justification documented in PR description**: at minimum LangSmith, Phoenix, Langfuse, Spren, generic HTTP — each shown as a one-paragraph adapter sketch with `publish_span` body.
- [ ] Framework regression suite green (zero new failures vs. baseline).
- [ ] New tests:
  - [ ] `test_telemetry_sink.py` — ABC contract; `NDJSONTraceWriter` `isinstance(sink, TelemetrySink)` regression.
  - [ ] `test_redactor.py` — default deny-list; case-insensitive matching; nested dicts; walks `span.events[*].attributes` and `span.links[*].attributes`; `extra_deny`; `replacement` override; `NoRedactionRedactor` no-op.
  - [ ] `test_collector_redaction.py` — `_stream_span` redacts before fan-out; sinks see redacted view; `redactor=None` opts out.
  - [ ] `test_sink_integration.py` — full `Orchestra.run()` with `RecordingTelemetrySink` fixture; recorded spans match the `TraceTree`; `close()` called per sink at run end.
  - [ ] `test_multi_consumer.py` — `FakeLangSmithSink` + `FakePhoenixSink` + `FakeSprenSink` all attached; each translates the protocol's spans into its target shape; `tool_arguments` carrying secrets are redacted in all three.
- [ ] Refreshed tests:
  - [ ] `test_ndjson_writer.py` — `write_span` → `publish_span` calls; `isinstance(writer, TelemetrySink)` assertion; same behavior as Session 01 baseline.
  - [ ] `test_ndjson_reader.py` — green (reader unaffected; imports may need updating).
- [ ] `CHANGELOG.md` entry under next version.
- [ ] No TRUNK-CRITICAL behavior changes. `Orchestra` constructor and `Orchestra.run` signatures unchanged.
- [ ] No Spren type imported in this PR.
- [ ] PR description references this brief.

---

## No TRUNK-CRITICAL touches

`Orchestra.__init__` signature and `Orchestra.run` signature are unchanged. Sinks plumb through `TracingConfig.sinks`; the redactor through `TracingConfig.redactor`. `Orchestra._initialize_components` reads from config — no new kwargs. The pre-flight TRUNK-CRITICAL escalation gate that the previous draft of this brief required no longer applies.

`TraceCollector` is not in the TRUNK-CRITICAL table, but it is load-bearing for every existing tracing consumer. Adding redaction to `_stream_span` and renaming `writers` → `sinks` is additive in semantic terms — the data flow is unchanged, only the names and the in-flight redaction step are new.

If implementation requires a non-additive change to `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, or `TopologyGraph`, **stop and escalate via `AskUserQuestion`** before any edit.

---

## Background reading (do this before writing code)

1. The framework worktree's `CLAUDE.md` — TRUNK-CRITICAL list; design principles
2. Framework architecture docs in the framework worktree
3. [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md) — Spren's expectation of this PR
4. [`../../../spren/v0.4-extensions.md`](../../../spren/v0.4-extensions.md) — `v0.4-27` consumer (`SprenTelemetrySink`)
5. [`../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-018 (the seam Spren expects)
6. The full `coordination/tracing/` module — read every `.py`:
   - `collector.py` — esp. `_stream_span` (L403-417), `finalize` (L421-471) for the deletion site at L463-468
   - `types.py` — `Span` shape and the `events` / `links` lists with their nested `attributes` dicts
   - `writers/base.py` — the ABC about to be deleted
   - `writers/ndjson_writer.py` — the class about to be reclassified; note `SCHEMA_VERSION`, `DISK_ERROR_DISABLE_THRESHOLD`, `CLOSE_TIMEOUT_SECONDS` constants; `write_span` at L131-146; vestigial `write` at L148-155
   - `config.py` — `TracingConfig` to extend
7. `coordination/orchestra.py` § `_initialize_components` (L163-247) and § the finalize block (L966-995) and `_collect_tracing_metadata` (L999-1017)

**Reference vendor docs (current 2026):**

- LangSmith Python SDK now recommends OTel-native tracing (`LANGSMITH_OTEL_ENABLED=true`); `Client.create_run` still works.
- Phoenix tracing is a thin wrapper over OpenTelemetry; adapters use a `TracerProvider` pointed at `PHOENIX_COLLECTOR_ENDPOINT`.
- Langfuse v4 (March 2026): unified `start_observation(as_type="span"|"generation", ...)` plus context-manager `start_as_current_observation`; explicit `flush()` / `shutdown()`.

**Verify before proceeding:**

- Capture baseline test counts before any change.
- `git log --oneline -20 src/marsys/coordination/tracing/ src/marsys/coordination/orchestra.py`.
- Confirm `TraceCollector` constructor still matches L39-44.
- Confirm `Orchestra._initialize_components` still constructs the collector at L207-216.
- If anything has moved or renamed since this brief was written, **update the brief or escalate** before writing code.

---

## Detailed plan

### Step 1 — Confirm vendor mappings still hold

Before writing code, confirm in the PR description that the `TelemetrySink` shape proposed in "Load-bearing shapes" maps cleanly to:

- LangSmith's `Client.create_run` (or its OTel-native equivalent)
- Phoenix's OTel `Tracer.start_span`
- Langfuse v4's `start_observation`

One paragraph per vendor; show the call-site translation. If any one of the three doesn't fit, **stop and escalate** before coding.

### Step 2 — Add the `TelemetrySink` ABC

Create `coordination/tracing/sink.py`:

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Span


class TelemetrySink(ABC):
    """
    Async streaming consumer for closed spans.

    TraceCollector calls publish_span once per span close. Adapters may do
    network I/O, queue/batch internally, and may take time to shut down.
    Exceptions raised from publish_span are caught + logged at the framework
    boundary; a misbehaving sink does not stop the run.

    Adapters live outside the framework (spren.telemetry, marsys-langsmith,
    marsys-phoenix, marsys-langfuse). The framework knows about TelemetrySink;
    it knows nothing about specific vendor backends.
    """

    @abstractmethod
    async def publish_span(self, span: 'Span') -> None:
        """Forward a closed span to the backend. May do network I/O."""

    @abstractmethod
    async def close(self) -> None:
        """Flush pending data and release resources. Idempotent."""
```

### Step 3 — Add `SecretRedactor`

Create `coordination/tracing/redactor.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Span


class SecretRedactor:
    """Redacts known-secret keys from span attribute payloads.

    Walks span.attributes recursively. Also walks the attribute dicts inside
    span.events (each event has its own 'attributes' sub-dict) and span.links.
    Matched keys at any nesting depth have their values replaced with the
    configured replacement string. Mutates in place.
    """

    DEFAULT_DENY = (
        "api_key", "apikey", "token", "authorization", "auth",
        "secret", "password", "bearer", "cookie", "session", "credential",
    )

    def __init__(
        self,
        *,
        extra_deny: tuple[str, ...] = (),
        replacement: str = "[REDACTED]",
    ):
        self._deny = tuple(s.lower() for s in (*self.DEFAULT_DENY, *extra_deny))
        self._replacement = replacement

    def redact(self, attributes: dict) -> None:
        """Mutate the attributes dict in place. Recurses into nested dicts."""
        for key in list(attributes.keys()):
            if isinstance(key, str) and any(d in key.lower() for d in self._deny):
                attributes[key] = self._replacement
                continue
            value = attributes[key]
            if isinstance(value, dict):
                self.redact(value)

    def redact_span(self, span: 'Span') -> None:
        """Apply redact() to attributes, events, and links sub-attributes."""
        if span.attributes:
            self.redact(span.attributes)
        for event in span.events:
            attrs = event.get('attributes')
            if isinstance(attrs, dict):
                self.redact(attrs)
        for link in span.links:
            attrs = link.get('attributes')
            if isinstance(attrs, dict):
                self.redact(attrs)


class NoRedactionRedactor(SecretRedactor):
    """Opt-in no-op redactor. Caller accepts the leak risk."""

    def redact_span(self, span):
        return None
```

### Step 4 — Wire sinks + redactor into `TraceCollector`

Update `coordination/tracing/collector.py`:

1. Update import at line 20: `from .sink import TelemetrySink` (replacing `from .writers.base import TraceWriter`). Add `from .redactor import SecretRedactor`.
2. Update constructor (currently L39-47):

   ```python
   def __init__(
       self,
       event_bus: 'EventBus',
       config: TracingConfig,
       sinks: Optional[List['TelemetrySink']] = None,
   ):
       self.event_bus = event_bus
       self.config = config
       self.sinks = sinks or []
       self._redactor = config.redactor if config.redactor is not None else SecretRedactor()
       # ... rest unchanged
   ```

3. Update `_stream_span` (currently L403-417):

   ```python
   async def _stream_span(self, span: Span) -> None:
       self._redactor.redact_span(span)
       for sink in self.sinks:
           try:
               await sink.publish_span(span)
           except Exception as e:
               logger.error(
                   "Telemetry sink %s.publish_span failed: %s",
                   type(sink).__name__, e, exc_info=True,
               )
   ```

4. Delete the entire `for writer in self.writers: try: await writer.write(trace) ...` loop at L463-468 inside `finalize`. The remaining `finalize` body stays unchanged.

5. Update `close` (L473-479):

   ```python
   async def close(self) -> None:
       for sink in self.sinks:
           try:
               await sink.close()
           except Exception as e:
               logger.error(f"Error closing telemetry sink: {e}")
   ```

### Step 5 — Extend `TracingConfig`

Update `coordination/tracing/config.py`:

```python
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .sink import TelemetrySink
    from .redactor import SecretRedactor


@dataclass
class TracingConfig:
    enabled: bool = False
    output_dir: str = "./traces"
    include_generation_details: bool = True
    include_message_content: bool = True
    include_tool_results: bool = True
    sinks: List['TelemetrySink'] = field(default_factory=list)
    redactor: Optional['SecretRedactor'] = None
```

### Step 6 — Migrate `NDJSONTraceWriter`

Update `coordination/tracing/writers/ndjson_writer.py`:

1. Update import at line 40: `from ..sink import TelemetrySink` (replacing `from .base import TraceWriter`).
2. Change class declaration: `class NDJSONTraceWriter(TelemetrySink):`.
3. Rename method `write_span(self, span)` to `publish_span(self, span)` (currently at L131-146). Update its docstring.
4. Delete the no-op `async def write(self, trace: 'TraceTree') -> None: return None` method (currently at L148-155). Drop the `write(trace)` line from the class lifecycle docstring (currently at L71-73).
5. Refresh tests at `tests/coordination/tracing/test_ndjson_writer.py` and `tests/coordination/tracing/test_ndjson_reader.py`: rename all `await writer.write_span(...)` calls to `await writer.publish_span(...)` (approximately 24 sites in `test_ndjson_writer.py` plus 8 in `test_ndjson_reader.py`). Also rename the fake-writer subclass method definition at `test_ndjson_writer.py:355` (`async def write_span(self, span):` → `async def publish_span(self, span):`) and any in-line comments that reference the old name. Update any `from ...writers.base import TraceWriter` imports to `from ...sink import TelemetrySink`. Add `isinstance(writer, TelemetrySink)` regression assertion in `test_ndjson_writer.py`.

### Step 7 — Delete `TraceWriter`

1. Delete `coordination/tracing/writers/base.py` entirely.
2. Update `coordination/tracing/writers/__init__.py` — drop `TraceWriter` export. Keep `NDJSONTraceWriter` export.
3. Update `coordination/tracing/__init__.py` — drop `TraceWriter` export. Add `TelemetrySink` and `SecretRedactor` exports.
4. `grep -rn 'TraceWriter\|write_span' packages/framework/` and update every remaining reference. None should remain after this step.

### Step 8 — Wire sinks through `Orchestra`

Update `coordination/orchestra.py:207-216`:

```python
if execution_config.tracing.enabled:
    from .tracing.collector import TraceCollector
    from .tracing.writers.ndjson_writer import NDJSONTraceWriter

    sinks: list = [NDJSONTraceWriter(execution_config.tracing)]
    sinks.extend(execution_config.tracing.sinks)
    self.trace_collector = TraceCollector(
        event_bus=self.event_bus,
        config=execution_config.tracing,
        sinks=sinks,
    )
    logger.info("Tracing enabled, output dir: %s", execution_config.tracing.output_dir)
```

Update `_collect_tracing_metadata` (L999-1017): change `for writer in self.trace_collector.writers:` to `for sink in self.trace_collector.sinks:` and rename loop variable `writer` → `sink` inside the loop body.

### Step 9 — Tests, framework docs, CHANGELOG

Per acceptance criteria. Test layout:

```
packages/framework/tests/coordination/tracing/
  test_telemetry_sink.py        # ABC contract; NDJSON isinstance regression
  test_redactor.py              # deny-list, attributes/events/links walking
  test_collector_redaction.py   # _stream_span applies redactor; opt-out
  test_sink_integration.py      # RecordingTelemetrySink + Orchestra.run
  test_multi_consumer.py        # 3 fake vendor adapters, all see same tree
  test_ndjson_writer.py         # REFRESHED: write_span → publish_span; isinstance
  test_ndjson_reader.py         # REFRESHED: imports + assertions
```

Framework architecture doc: short page covering protocol shape, redaction policy, three-vendor adapter sketch.

CHANGELOG entry under the next release version: `TelemetrySink` ABC introduced; `TraceWriter` removed; `SecretRedactor` added; `NDJSONTraceWriter` reclassified to `TelemetrySink`; vestigial `write(trace)` removed; `TracingConfig.sinks` and `TracingConfig.redactor` added.

### Files to create

- `packages/framework/src/marsys/coordination/tracing/sink.py` — `TelemetrySink` ABC
- `packages/framework/src/marsys/coordination/tracing/redactor.py` — `SecretRedactor` and `NoRedactionRedactor`
- `packages/framework/tests/coordination/tracing/test_telemetry_sink.py`
- `packages/framework/tests/coordination/tracing/test_redactor.py`
- `packages/framework/tests/coordination/tracing/test_collector_redaction.py`
- `packages/framework/tests/coordination/tracing/test_sink_integration.py`
- `packages/framework/tests/coordination/tracing/test_multi_consumer.py`

### Files to modify

- `packages/framework/src/marsys/coordination/tracing/__init__.py` — drop `TraceWriter` export, add `TelemetrySink` + `SecretRedactor`
- `packages/framework/src/marsys/coordination/tracing/writers/__init__.py` — drop `TraceWriter` export
- `packages/framework/src/marsys/coordination/tracing/collector.py` — rename `writers` → `sinks`; integrate redactor; drop `write(trace)` loop in `finalize`
- `packages/framework/src/marsys/coordination/tracing/config.py` — add `sinks` + `redactor` fields
- `packages/framework/src/marsys/coordination/tracing/writers/ndjson_writer.py` — inheritance to `TelemetrySink`; `write_span` → `publish_span`; drop `write(trace)`
- `packages/framework/src/marsys/coordination/orchestra.py:207-216` — read sinks from `TracingConfig`
- `packages/framework/src/marsys/coordination/orchestra.py:999-1017` — iterate `sinks` instead of `writers`
- `packages/framework/CHANGELOG.md` — release entry
- `packages/framework/tests/coordination/tracing/test_ndjson_writer.py` — refresh `write_span` calls; `isinstance` regression
- `packages/framework/tests/coordination/tracing/test_ndjson_reader.py` — refresh imports if any

### Files to delete

- `packages/framework/src/marsys/coordination/tracing/writers/base.py` — `TraceWriter` ABC

### Files NOT to touch

- TRUNK-CRITICAL: `coordination/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`, `coordination/topology/graph.py`
- `coordination/event_bus.py` — sinks ride on `TraceCollector`, not directly on `EventBus`
- `coordination/tracing/types.py` — `Span` shape unchanged
- `coordination/tracing/readers/` — readers untouched; reader is independent of writer/sink ABC

### Load-bearing shapes

The ABC IS the contract. After this PR ships, third-party adapters depend on its stability. Treat as semver — additive changes only after v1.0; pre-1.0 the framework docs mark it as `pre-1.0`.

```python
class TelemetrySink(ABC):
    @abstractmethod
    async def publish_span(self, span: Span) -> None: ...
    @abstractmethod
    async def close(self) -> None: ...
```

`Span` is the existing `coordination.tracing.types.Span` after `SecretRedactor` has run. Adapters see `attributes`, `events[*].attributes`, and `links[*].attributes` redacted by default.

---

## Hard rules

### Multi-consumer justification (mandatory)

- [ ] PR description includes a one-paragraph adapter sketch for each of: Spren, LangSmith, Phoenix, Langfuse, generic HTTP. Each sketch shows the translation from this Protocol's calls to the vendor's native API.
- [ ] No Spren type imported in this PR.
- [ ] No "if running under Spren" code paths.

### Framework design principles

- DP-001 (pure agent logic): n/a — telemetry observes, doesn't run agent logic.
- DP-002 (centralized validation): n/a.
- DP-003 (unified-barrier orchestration): n/a — sinks are downstream of orchestration.
- DP-004 (branch isolation): preserved — `TraceCollector` is the single subscriber; sinks receive closed spans whose branch identity is already isolated by collector logic.
- DP-005 (topology-driven routing): n/a.
- DP-006 (adapter pattern): the `TelemetrySink` ABC IS the adapter pattern for observability backends. `TraceWriter` was a misnamed precursor; this session corrects the abstraction.
- DP-007 (format pluggability): on-disk NDJSON and on-the-wire vendor formats are now homogeneous from the framework's perspective — both are sinks.

If this feature would force a violation of any DP, escalate before coding.

### No TRUNK-CRITICAL behavioral changes

`Orchestra.__init__` and `Orchestra.run` signatures unchanged. The brief's previous draft proposed two new kwargs; that approach is replaced by `TracingConfig.sinks` plumbing. Pre-flight gate not required.

If implementation requires a non-additive change to `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, or `TopologyGraph`, **stop and escalate via `AskUserQuestion`** before any edit.

### Clean code rules

- Smallest implementation that passes acceptance criteria.
- No backward-compat shims (SP-006-equivalent for the framework). `TraceWriter` is deleted, not deprecated.
- One concern per file: ABC, redactor, config, collector hook, tests.
- No descriptive comments for self-naming code — only WHY when not obvious.

---

## Tests (required for "done")

### Unit tests

`tests/coordination/tracing/test_telemetry_sink.py`:

- `TelemetrySink` is an ABC; instantiating without overriding raises `TypeError`.
- `NDJSONTraceWriter` is `isinstance(TelemetrySink)` (regression — locks the new inheritance).
- A minimal `RecordingTelemetrySink(TelemetrySink)` subclass that overrides both abstract methods works.

`tests/coordination/tracing/test_redactor.py`:

- Default deny-list redacts each key (case-insensitive) at top-level.
- Default deny-list redacts at arbitrary nesting depth in dict values.
- `redact_span` walks `span.attributes`, every `span.events[*].attributes`, every `span.links[*].attributes`.
- Custom `extra_deny` extends the deny-list.
- `replacement` override produces the configured string.
- `NoRedactionRedactor` returns input unchanged.
- Redaction mutates in place (assert `id` of attribute dict before and after — same object).

`tests/coordination/tracing/test_collector_redaction.py`:

- `TraceCollector(config=TracingConfig(redactor=SecretRedactor()), sinks=[fake])` — span emitted via `_stream_span` reaches `fake.publish_span` with redacted attributes.
- `redactor=None` in config → default `SecretRedactor` instantiated lazily.
- `redactor=NoRedactionRedactor()` → sinks see raw attributes.
- Sink raising in `publish_span` does not propagate; other sinks still receive.
- Three sinks all receive each span in registration order.

### Integration tests

`tests/coordination/tracing/test_sink_integration.py`:

- A `RecordingTelemetrySink` (in-memory list of all `publish_span` calls) attached to a real `Orchestra.run()` records:
  - one `publish_span` per closed span; the recorded spans match `TraceCollector.finalize`'s `TraceTree`
  - exactly one `await sink.close()` at run end
- Two `RecordingTelemetrySink` instances both receive identical streams in same order.

`tests/coordination/tracing/test_multi_consumer.py`:

- Three sinks attached to a single `Orchestra.run()`:
  - `FakeLangSmithSink` records `(name, run_type, parent_run_id, start_time, end_time, inputs, outputs)` tuples translated from the Span.
  - `FakePhoenixSink` records OTel-shaped span dicts (`(parent_id, start_time, end_time, attributes, kind)`).
  - `FakeSprenSink` records the protocol's calls verbatim.
- All three see the same span tree; each translates correctly into its target shape.
- A run with `ToolCallEvent.arguments={"api_key": "sk-..."}` produces redacted values in all three.

### Refreshed tests (regression coverage)

`tests/coordination/tracing/test_ndjson_writer.py`:

- All `await writer.write_span(...)` calls renamed to `publish_span`.
- All imports of `TraceWriter` updated to `TelemetrySink`.
- New assertion: `assert isinstance(writer, TelemetrySink)`.
- All Session 01 behaviors still verified: queue overflow drop-oldest, fsync_per_span, schema_version, stream_completed marker, close idempotency, disk-write error self-disable.

`tests/coordination/tracing/test_ndjson_reader.py`:

- Confirm green; reader is unaffected by the writer's ABC change.
- Update imports if any reference `TraceWriter`.

### Framework regression test

Entire `packages/framework/tests/` passes with the SAME counts as baseline. Document baseline + post-change counts in "What was actually built."

### Edge-case coverage matrix

| Scenario | Test |
|---|---|
| Default redactor catches `api_key` at top level of attributes | `test_redactor.py` |
| Default redactor catches `password` nested two levels deep | `test_redactor.py` |
| Redactor walks `span.events[*].attributes` | `test_redactor.py` |
| Redactor walks `span.links[*].attributes` | `test_redactor.py` |
| `redactor=None` defaults to `SecretRedactor` instance | `test_collector_redaction.py` |
| `NoRedactionRedactor` opt-out passes raw values through | `test_collector_redaction.py` |
| Sink failure does not propagate to other sinks | `test_collector_redaction.py` |
| `NDJSONTraceWriter isinstance TelemetrySink` | `test_ndjson_writer.py` (regression) |
| `Orchestra.run` end-to-end with custom sinks via `TracingConfig.sinks` | `test_sink_integration.py` |
| Three vendor adapters in one run, all see same redacted tree | `test_multi_consumer.py` |

---

## Open questions for the framework team

The implementer surfaces these via `AskUserQuestion` BEFORE writing code if any are still unanswered:

1. **Default replacement string.** Brief proposes `"[REDACTED]"`. Alternatives: `"***"`, `""`, configurable-required (no default). Confirm.
2. **`extra_deny` matching mode.** Brief proposes substring match (case-insensitive). Stricter alternative: full key-name match. Substring is more ergonomic but may over-match — e.g., `"auth"` deny would match `"authority"` or `"authorial"` field names. Confirm acceptable.
3. **Opt-out from redaction.** Brief ships both `SecretRedactor()` (default-on) and `NoRedactionRedactor` (no-op subclass). Confirm both are wanted, or whether reconfiguring `SecretRedactor` with an empty `DEFAULT_DENY` override is sufficient.

---

## Sign-off

On completion:

1. Update **What was actually built** below with delta from plan
2. Update [`../../v0.4-spren-support.md`](../../v0.4-spren-support.md) — check Session 02's row
3. Note the framework release version that ships this feature
4. Add **Lessons / Surprises** below

### What was actually built (filled by implementer)

> _Implementer fills this in._
>
> Include: baseline test counts (before change), post-change test counts (regression suite must match; include new tests separately), framework PR number + URL, framework release version, anything done differently from the plan with reasons.

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._
