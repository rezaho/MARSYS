# Framework Session 01: NDJSON Streaming Tracing Writer

Required by Spren v0.3 (Session 04: run execution + tracing).

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of the marsys framework.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if the proposed touch points are wrong for the framework's current shape, say so. Defend with evidence.
- **Stay engaged.** Comment in this session file as you go.
- **Be proactive.** If you see something this session is missing, raise it.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Manually verify acceptance criteria.
- **You own follow-through.** Update this file's "What was actually built". Update [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) checkbox. Add "Lessons / Surprises" if anything surprised you.

### Double-check before any decision

- **Read the framework code before changing it.**
- **Verify file paths and symbols still exist** before referencing them.
- **Run framework tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints.**

### Critically assess the plan itself

This brief was written from the Spren consumer perspective with a careful read of the current `coordination/tracing/` module. Verify the symbols and line numbers cited are still accurate before writing code. If anything has drifted (a symbol moved, a method changed signature), update the brief or escalate.

### Multi-consumer justification (mandatory)

Any framework user who runs `Orchestra.run()` and wants on-disk durability for the resulting trace benefits from streaming-on-emit: mid-run crashes preserve everything emitted up to the crash, and tail-followers can read live state. Spren is one such consumer. Other consumers documented below in "Multi-consumer scope (honest)".

**Forbidden:** any code path special-cased for Spren. The writer behaves identically regardless of who's consuming the traces.

### Foundational project rules

- The framework worktree's `CLAUDE.md` — TRUNK-CRITICAL component map, framework design principles DP-001..DP-007
- Framework architecture docs in the framework worktree (especially the tracing module's docs if present)
- Spren's view of why this is needed: [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md), [`../../../architecture/spren/06-observability.md`](../../../../architecture/spren/06-observability.md)

---

## The big picture — what we're building and why

### What this PR ships

The framework's tracing layer collects events from `EventBus` into a hierarchical `TraceTree` via `TraceCollector`, then writes the completed tree once at finalization through registered `TraceWriter` implementations (see `tracing/collector.py:39-48`, `tracing/collector.py:401-448`, `tracing/writers/base.py:19-25`). The shipped writer is `JSONFileTraceWriter`, which serializes the whole tree to a single `{output_dir}/{session_id}.json` file inside `TraceCollector.finalize()`.

Two practical problems with that:

1. **Mid-run process crashes lose the entire trace.** All in-memory state evaporates before `finalize()` runs.
2. **Late subscribers cannot tail-follow.** The file does not exist until completion; SSE-style streaming consumers (Spren, any future hosted observability frontend) cannot read partial state.

This PR adds a streaming NDJSON writer that hooks into `TraceCollector` at the per-span level — every time a span closes, the writer emits one JSON object on a new line and flushes. The legacy `JSONFileTraceWriter` is removed in the same PR (forward-only, per SP-006). Crash mid-run = the file is exactly the spans that closed before the crash. Late subscribers tail-read the file from the start to current EOF, then switch to live `EventBus` subscription.

### Architectural placement: hook the existing `TraceCollector`

The current pipeline is single-subscriber:

```
EventBus → TraceCollector (one subscriber; builds TraceTree)
                          → registered TraceWriter.write(tree) (called once at finalize)
```

`TraceCollector` is the canonical builder for the hierarchical tree. It owns non-trivial state — clamped generation start times (`collector.py:248`), pending convergence inheritance (`collector.py:55`, `collector.py:344-376`), step span persistence beyond closure for late `ValidationDecisionEvent` attachment (`collector.py:340-342`). Re-implementing any of that in a parallel EventBus subscriber would diverge from the canonical view.

The streaming NDJSON writer is therefore a **per-span hook on `TraceCollector`**, not a parallel EventBus subscriber. The collector remains the sole subscriber on the bus and the sole tree builder. The writer receives `Span` objects as they close and serializes one NDJSON line per closed span. At finalize, the writer emits a `stream_completed` marker and closes the file. Reconstruction of the hierarchical tree from streamed lines is straightforward (parent_span_id is on every span) and does not require duplicating the collector's logic.

### Why the framework needs this (multi-consumer scope, honest)

This PR ships **on-disk durability** for run traces. It does not claim wire-format export to third-party observability vendors — those backends have their own protocols.

- **Spren** subscribes to the file for SSE event streaming AND uses it for trace inspection on completed runs.
- **Framework users running long workflows locally** stop losing traces on Ctrl-C / kernel-panic / power-loss / OOM-kill.
- **MARSYS Cloud's hosted runs** can stream completed-span events to subscribers without keeping the entire tree in memory until finalize. Cloud may also re-emit the same NDJSON over its own wire protocol; that's a separate Cloud-side concern.
- **Custom in-process consumers** (CLI inspectors, internal dashboards, batch trace post-processors) read the file with the shipped reader.

What this PR does **not** ship: protocol-level export to LangSmith / Phoenix / Langfuse. All three speak OTLP over HTTP, not NDJSON. An OTel exporter is a separate framework concern that lands via the `TelemetrySink` protocol in Framework Session 02. NDJSON-on-disk and OTel-on-the-wire are complementary concerns; this PR is the durability half. State this in the PR description verbatim so reviewers don't expect functionality this PR doesn't include.

### Your role as a framework implementer

1. Honor the framework's architecture (single-subscriber `TraceCollector`)
2. Honor the framework's design principles (DP-001..DP-007)
3. Honor multi-consumer justification — no Spren-specific paths
4. Ship a single coherent PR with green tests
5. Push back when something is wrong

---

## What came before this session

**Previous framework PRs from this dir:** None — first framework session driven by the Spren-side backlog.

**State at start of this session:**

- Framework on main, post-unified-barrier merge: `Orchestra` + `Orchestrator` + `RealRuntime` are the live execution path; `BranchExecutor` / `BranchSpawner` are gone; `EventBus` lives at `coordination/event_bus.py`; lifecycle events `BranchCreatedEvent` / `BranchCompletedEvent` are in `coordination/events.py`.
- Tracing module at `coordination/tracing/`:
  - `tracing/collector.py` — `TraceCollector` (single EventBus subscriber, builds `TraceTree`)
  - `tracing/types.py` — `Span` / `TraceTree` / `create_span()`
  - `tracing/config.py` — `TracingConfig`
  - `tracing/events.py` — tracing-specific `StatusEvent` subclasses
  - `tracing/writers/base.py` — `TraceWriter` ABC (`async write(trace) / async close()`)
  - `tracing/writers/json_writer.py` — `JSONFileTraceWriter` (write-at-end JSON)
- `Orchestra.__init__()` instantiates `JSONFileTraceWriter` at `coordination/orchestra.py:209-215` when tracing is enabled, and passes it to `TraceCollector(writers=[json_writer])`.
- `Span.span_id` is `uuid4` per `tracing/types.py:123`. `Span.kind` is lowercase per `tracing/types.py:34`. `Span.start_time` is a float per `tracing/types.py:35`.
- `StatusEvent.event_id` is `uuid.uuid4()` per `coordination/status/events.py:15`.
- `EventBus.emit()` removes a listener after 5 consecutive errors via the `_listener_errors` counter in `event_bus.py:50-60`.
- Existing tracing tests live at `packages/framework/tests/coordination/test_trace_collector.py` and `tests/coordination/orchestrator/test_convergence_trace_links.py`.

**Verify state with:**

```bash
cd /home/rezaho/research_projects/marsys-spren-work/packages/framework/
source ../../.venv/bin/activate

# Capture baseline test counts BEFORE any change
pytest tests/ -x --tb=short
# → record total, passed, failed, skipped

# Confirm the symbols this brief cites
grep -n 'class TraceCollector\|class TraceWriter\|class Span\|class TraceTree\|JSONFileTraceWriter\|event_id\|_max_listener_errors' \
  src/marsys/coordination/tracing/collector.py \
  src/marsys/coordination/tracing/writers/base.py \
  src/marsys/coordination/tracing/writers/json_writer.py \
  src/marsys/coordination/tracing/types.py \
  src/marsys/coordination/status/events.py \
  src/marsys/coordination/event_bus.py

# See recent activity
git log --oneline -20 src/marsys/coordination/tracing/
git log --oneline -20 src/marsys/coordination/event_bus.py
git log --oneline -20 src/marsys/coordination/status/events.py
```

If anything has drifted: stop and update the brief or escalate.

### Pre-flight escalation gate (TRUNK-CRITICAL)

`Orchestra` is TRUNK-CRITICAL. The expected change in `coordination/orchestra.py` is replacing two lines: the `JSONFileTraceWriter` import and its construction at lines 209-215 (instantiating the new `NDJSONTraceWriter` and passing it to `TraceCollector(writers=[ndjson_writer])`). This is a wiring substitution, not a change to `Orchestra.run()` semantics — it does not alter the public surface of `Orchestra`, the constructor signature, or the run lifecycle. SP-001 forbids non-additive TRUNK-CRITICAL changes; a like-for-like writer substitution at the construction site is additive in semantic terms (the surrounding code is unchanged).

**Implementer responsibility before editing `orchestra.py`:**

1. Re-read `Orchestra.__init__()` (`coordination/orchestra.py:1-260`).
2. Confirm the change is genuinely confined to the writer-construction lines (no spread into `run()` or other lifecycle methods).
3. If the change grows beyond a like-for-like writer substitution at the existing call site, **stop and escalate via `AskUserQuestion`** for an ADR review. Do not silently expand scope into `Orchestra` internals.

Edits to `Orchestrator`, `RealRuntime`, `ValidationProcessor`, or `TopologyGraph` are out of scope for this session — escalate before any of them.

### `StatusEvent.event_id`: UUID4 → ULID

`StatusEvent.event_id` is currently `uuid.uuid4()` per `coordination/status/events.py:15`. This is random, not monotonic. Spren's SSE-resume contract relies on monotonic-orderable event IDs (`Last-Event-ID` header, see `docs/architecture/spren/06-observability.md:46`).

This session changes the default factory for `StatusEvent.event_id` from `uuid.uuid4()` to a ULID. ULIDs are 26-character lexicographic-monotonic identifiers (see [github.com/ulid/spec](https://github.com/ulid/spec)) that sort correctly by emission time within a process and are JSON-safe strings, so changing the factory does not change `event_id`'s declared type or external shape (still a `str`). All call sites that read `event_id` continue working unchanged; tests that compare `event_id` values to specific UUID4 strings (if any exist) get migrated.

This change is the only TRUNK-adjacent change in this session: `StatusEvent` is the base class for events that flow through `EventBus`, so the change is observable to all subscribers. Verify per SP-001:

- The change is **additive** in shape: `event_id: str` remains a string.
- The change is **substitutive** in value: random UUID → monotonic ULID. No subscriber that treats `event_id` as an opaque identifier breaks.
- The change is **single-version** (SP-006-equivalent): no flag selecting old vs new ID format.

Apply the same pre-flight check before editing `status/events.py`: re-read the file, confirm the change is contained to the default factory and import, and run the framework regression suite immediately after the edit.

`Span.span_id` (set in `tracing/types.py:123` as `uuid.uuid4()`) is also migrated to ULID in this session for consistency: span IDs and event IDs share lexicographic-monotonic ordering, which simplifies cross-referencing in the NDJSON stream and the SSE resume flow.

The chosen ULID library: `python-ulid` (PyPI: `python-ulid`, MIT-licensed, single dependency, `ULID()` returns a `str`-castable monotonic ID). Add it to `packages/framework/pyproject.toml` runtime deps.

---

## What this session ships

After merge:

- A streaming NDJSON writer at `packages/framework/src/marsys/coordination/tracing/writers/ndjson_writer.py` that subclasses `TraceWriter` and adds a `write_span(span)` method called by `TraceCollector` whenever a span closes.
- A small extension to `TraceCollector` that calls `await writer.write_span(span)` on every span close (added to a single helper invoked from each `_handle_*` method that closes a span). The collector remains the single EventBus subscriber.
- A reader at `coordination/tracing/readers/ndjson_reader.py` that streams events one span at a time, supports tail-follow, and reconstructs a `TraceTree` from streamed lines.
- The legacy `JSONFileTraceWriter` is **deleted** along with its test references; the writers package no longer exports it.
- A one-shot conversion utility (`marsys.coordination.tracing.compat.json_to_ndjson`) for users with archived legacy `.json` traces.
- `Orchestra.__init__()` instantiates `NDJSONTraceWriter` instead of `JSONFileTraceWriter` when tracing is enabled.
- `StatusEvent.event_id` and `Span.span_id` migrate from UUID4 to ULID.
- Framework regression suite green (zero new failures vs. baseline); existing tests that asserted on the JSON-at-end format migrate to assert against NDJSON.
- New tests cover the wire format, span-close streaming, tail-follow, reconstruction parity, and every edge case in the "Tests" section below.

### Acceptance criteria

- [ ] `coordination/tracing/writers/ndjson_writer.py` exists; subclasses `TraceWriter`; implements `write_span(span)` per closed span and `write(trace)` as a final marker (delegates to internal close + writes `stream_completed` line).
- [ ] `TraceCollector` calls `await writer.write_span(span)` on every span close (root close in `_handle_final_response`, branch close in `_handle_branch_completed`, step close in `_handle_agent_complete`, tool close in `_handle_tool_call`, generation close in `_handle_generation`, and orphan close in `finalize()`).
- [ ] Writer flushes per span close (`f.flush()`); `os.fsync()` is opt-in via constructor flag (default off).
- [ ] Writer opens file with `mode='a', encoding='utf-8', newline='\n'`. No BOM. Line terminator is `\n` on all platforms.
- [ ] Writer uses a bounded `asyncio.Queue` (default `maxsize=10000`) and a single dedicated drain task — `write_span` is non-blocking on the disk write path, so a slow disk does not back-pressure `EventBus.emit`. Drop-oldest policy on overflow with a `dropped_span_count` metric exposed via the writer's introspection surface.
- [ ] On disk-write error, the writer logs and increments an internal error counter but does **not** raise out of `write_span`. `EventBus`'s 5-strike auto-unsubscribe rule (`event_bus.py:50-60`) does not get triggered because `write_span` is called from `TraceCollector`, not directly from a subscriber callback. The disk error counter is observable; if it exceeds a threshold, the writer logs at warning level periodically.
- [ ] Writer emits a `stream_completed` final marker line on `close()`; closes the file cleanly.
- [ ] Legacy `JSONFileTraceWriter` DELETED — no dual-write, no compatibility shim, no opt-in flag. The writers package's `__init__.py` no longer exports it. SP-006-equivalent.
- [ ] One-shot conversion utility `coordination/tracing/compat/json_to_ndjson.py` reads a legacy `JSONFileTraceWriter`-format file and writes the equivalent NDJSON. Provides a `python -m` CLI entry.
- [ ] Reader at `coordination/tracing/readers/ndjson_reader.py`:
  - [ ] Streaming iteration (generator yielding span dicts)
  - [ ] `follow=True` mode (tail-follow with EOF retry; opens read-only on Windows so no exclusive write lock)
  - [ ] Tolerates a truncated trailing line: yields N-1 spans on partial line, exposes `truncated_line_count` on the reader instance, does not raise.
  - [ ] Detects missing `stream_completed` marker on EOF and surfaces it via a clear API (`reader.completion_status` returns `complete | truncated | crashed`).
  - [ ] `to_tree()` method returns hierarchical `TraceTree` from streamed spans (uses `parent_span_id` for parent-child reconstruction).
- [ ] `Orchestra.__init__()` updated at `coordination/orchestra.py:209-215`: instantiate `NDJSONTraceWriter` instead of `JSONFileTraceWriter`. No other change to `Orchestra`.
- [ ] `StatusEvent.event_id` default factory migrated to ULID (`coordination/status/events.py:15`).
- [ ] `Span.span_id` factory migrated to ULID in `create_span()` (`coordination/tracing/types.py:113-130`).
- [ ] `python-ulid` added to `packages/framework/pyproject.toml` runtime dependencies.
- [ ] **Multi-consumer justification documented in PR description**: at minimum the four consumers listed in "Multi-consumer scope (honest)" above. PR description explicitly says "OTel-protocol export is a separate concern handled by Framework Session 02."
- [ ] Framework regression suite green (zero new failures vs. baseline).
- [ ] New tests covering every scenario in "Tests" below.
- [ ] Framework's `CHANGELOG.md` entry under the next release version, calling out: NDJSON streaming writer, `JSONFileTraceWriter` removal, ULID migration for `event_id` and `span_id`, conversion utility for legacy traces.
- [ ] Spren architecture doc `docs/architecture/spren/06-observability.md` updated to use lowercase `kind` values matching the framework's `Span.kind` (per `tracing/types.py:34`) and to reflect the framework-side (not Spren-side) location of the writer.
- [ ] No TRUNK-CRITICAL changes beyond the like-for-like writer substitution in `Orchestra.__init__()` and the ULID migration in `StatusEvent`.
- [ ] No Spren type imported in this PR.
- [ ] PR description references this brief.

---

## Background reading (do this before writing code)

1. The framework worktree's `CLAUDE.md` — TRUNK-CRITICAL list; design principles
2. Framework architecture docs in the framework worktree
3. [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) — Spren's expectation of this PR
4. [`../../../architecture/spren/06-observability.md`](../../../../architecture/spren/06-observability.md) — Spren's downstream consumption pattern (informs reader's tail-follow API)
5. The full `coordination/tracing/` module — read every `.py` end-to-end:
   - `collector.py` (esp. `_subscribe_to_events`, every `_handle_*`, and `finalize()`)
   - `types.py` (`Span` / `TraceTree` / `create_span`)
   - `writers/base.py` (the ABC the new writer extends)
   - `writers/json_writer.py` (the writer being removed)
   - `events.py` (tracing-specific events)
   - `config.py`
6. `coordination/event_bus.py` — subscription mechanics, 5-strike rule
7. `coordination/status/events.py` — `StatusEvent.event_id` definition
8. `coordination/orchestra.py:195-220` — current writer wiring
9. NDJSON spec — one JSON object per line, no inter-line whitespace, `\n`-terminated. See [ndjson.org](https://ndjson.org/).
10. ULID spec — [github.com/ulid/spec](https://github.com/ulid/spec). Confirm `python-ulid`'s monotonic factory matches the spec.

**Verify before proceeding:**

- Capture baseline test counts (already noted in "State at start").
- Re-read every line of `tracing/collector.py` (cited in this brief).
- Verify the `_handle_*` method names and the span-close points haven't drifted from this brief's expectations.
- Identify all consumers of the legacy `JSONFileTraceWriter` (search results from `grep -rn 'JSONFileTraceWriter' packages/framework/`):
  - `coordination/orchestra.py:209-215` (construction site)
  - `coordination/tracing/writers/__init__.py:6,10` (exports)
  - Test files that assert on the legacy `.json` shape
- Note where the framework's NDJSON line-format choice intersects with `Span.to_dict()` (the writer reuses the existing serialization path).

---

## Detailed plan

### Step 0 — Capture baseline + map the current state

```bash
pytest tests/ -x --tb=short                          # record baseline counts
find src/marsys/coordination/tracing -name '*.py'
grep -rn 'JSONFileTraceWriter' src/marsys/ tests/
grep -rn 'event_id' src/marsys/coordination/status/events.py
git log --oneline -20 src/marsys/coordination/tracing/
```

Document baseline test counts and the JSONFileTraceWriter touch sites in PR description.

### Step 1 — Add ULID factory + migrate `event_id` and `span_id`

- Add `python-ulid` to `packages/framework/pyproject.toml` runtime dependencies.
- Pick the canonical helper: `from ulid import ULID; lambda: str(ULID())` (the package's monotonic factory). Wrap in a small helper if reused: `coordination/tracing/_ids.py` exposing `new_id() -> str`.
- Update `coordination/status/events.py:15`: replace `uuid.uuid4()` with the ULID helper.
- Update `coordination/tracing/types.py:123` (`create_span()`): replace `uuid.uuid4()` with the ULID helper.
- Run the framework regression suite. If any test asserts on UUID4-shaped IDs, migrate it to assert on ULID-shaped IDs (26-character base32 string). Do not skip; fix.

### Step 2 — Define `NDJSONTraceWriter`

Create `coordination/tracing/writers/ndjson_writer.py`. Subclass `TraceWriter`. The writer's surface adds one method beyond the ABC — `write_span(span)` — and provides a non-blocking implementation:

```python
class NDJSONTraceWriter(TraceWriter):
    """
    Streaming NDJSON writer.

    One JSON object per closed span on its own line, plus a final
    'stream_completed' marker on close. The writer does not subscribe
    to EventBus directly — it receives spans from TraceCollector as
    they close, preserving TraceCollector as the single source of
    truth for hierarchical tree construction.

    NOT Spren-specific. Used by any consumer that wants line-by-line
    persistence of a marsys execution trace.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        config: TracingConfig,
        *,
        fsync_per_span: bool = False,
        queue_maxsize: int = 10000,
    ): ...

    async def write_span(self, span: Span) -> None:
        """Enqueue a closed span for serialization. Non-blocking on disk."""
        ...

    async def write(self, trace: TraceTree) -> None:
        """Called by TraceCollector at finalize. Drains the queue, emits the
        stream_completed marker, and triggers close()."""
        ...

    async def close(self) -> None:
        """Stop the drain task, flush, close the file. Idempotent."""
        ...
```

Drain task: a single `asyncio.Task` reads from the bounded queue and writes lines. On overflow, the writer drops the oldest queued span and increments `dropped_span_count` (also written as a `stream_event` warning line into the file so consumers see it).

Disk error handling: the drain task catches `OSError` (disk full, permission denied, file-system unmount), increments `disk_error_count`, logs at warning level, and continues. The drain task does not raise back into the event-loop scheduler; the bus's 5-strike rule is not in play because `write_span` is called from `TraceCollector`, not directly from a bus listener.

File open: `open(path, mode='a', encoding='utf-8', newline='\n')`. Path: `{config.output_dir}/{session_id}.ndjson`.

### Step 3 — Hook `TraceCollector` into the writer

Add a small helper to `TraceCollector`:

```python
async def _stream_span(self, span: Span) -> None:
    for writer in self.writers:
        if hasattr(writer, "write_span"):
            try:
                await writer.write_span(span)
            except Exception as e:
                logger.error(f"Trace writer {type(writer).__name__} write_span failed: {e}")
```

Call `await self._stream_span(span)` at the end of every span-close site:

- `_handle_final_response` after `trace.root_span.close(...)` (`collector.py:394-397`)
- `_handle_branch_completed` after `span.close(...)` (`collector.py:166-171`)
- `_handle_agent_complete` after `span.close(...)` (`collector.py:333-342`)
- `_handle_tool_call` (the `completed | failed` branch) after `tool_span.close(...)` (`collector.py:291-300`)
- `_handle_generation` after `gen_span.close(...)` (`collector.py:256-259`)
- `finalize()` for each orphaned span being force-closed (`collector.py:415-423`)

The `TraceWriter` ABC stays unchanged. Writers that don't implement `write_span` (none after this PR — `JSONFileTraceWriter` is removed) gracefully skip.

### Step 4 — Define `NDJSONTraceReader`

Create `coordination/tracing/readers/ndjson_reader.py`:

```python
class NDJSONTraceReader:
    """
    Streaming NDJSON reader with tail-follow support.

    NOT Spren-specific. Any consumer (UI, CLI inspector, post-processor) reads here.
    """

    def __init__(self, path: pathlib.Path): ...

    @property
    def completion_status(self) -> Literal["complete", "truncated", "crashed"]:
        """After streaming ends, indicates whether the file ended with a
        stream_completed marker (complete), a truncated last line (truncated),
        or neither (crashed)."""
        ...

    @property
    def truncated_line_count(self) -> int: ...

    def stream(self, follow: bool = False) -> Iterator[dict]:
        """Yield span dicts line-by-line. If follow=True, block on EOF and
        retry until stream_completed is read or consumer breaks out."""
        ...

    def to_tree(self) -> TraceTree:
        """Materialize hierarchical TraceTree from the file's spans.

        Reconstruction: read all spans into a flat list keyed by span_id,
        then attach each span as a child of its parent_span_id. The root
        span (parent_span_id is None and kind == 'execution') becomes
        TraceTree.root_span.
        """
        ...
```

Open mode for the reader: `open(path, mode='r', encoding='utf-8')` (read-only — Windows write locks will not block the reader).

Tail-follow strategy: read until current EOF, sleep briefly (configurable, default 100ms), re-`stat()` the file, read new bytes from the last byte offset. Stop when a `stream_completed` line is parsed or the consumer breaks out.

Truncation tolerance: a final line that does not end with `\n` (or fails `json.loads`) is recorded as truncated and not yielded. `truncated_line_count` exposes the count; `completion_status` becomes `truncated`. The reader does not raise.

Crash detection: if the stream ends at EOF without ever seeing `stream_completed`, `completion_status` is `crashed`. Tail-followers that are watching a file from a known-dead process can use this to surface a "run terminated unexpectedly" notice in the consumer UI rather than waiting forever.

### Step 5 — Wire the new writer into `Orchestra`

In `coordination/orchestra.py:209-215`:

```python
# Before
from .tracing.writers.json_writer import JSONFileTraceWriter
json_writer = JSONFileTraceWriter(execution_config.tracing)
self.trace_collector = TraceCollector(
    event_bus=self.event_bus,
    config=execution_config.tracing,
    writers=[json_writer],
)

# After
from .tracing.writers.ndjson_writer import NDJSONTraceWriter
ndjson_writer = NDJSONTraceWriter(execution_config.tracing)
self.trace_collector = TraceCollector(
    event_bus=self.event_bus,
    config=execution_config.tracing,
    writers=[ndjson_writer],
)
```

Re-confirm before saving: only the import line and the writer construction are touched. No other change in `Orchestra`.

### Step 6 — Delete the legacy writer

- Delete `coordination/tracing/writers/json_writer.py`.
- Update `coordination/tracing/writers/__init__.py` to drop `JSONFileTraceWriter`.
- Search the test suite (`grep -rn 'JSONFileTraceWriter' tests/`) and either migrate each test to assert against NDJSON output (for tests that exercise actual writer behavior) or delete (for tests that test the legacy writer's specifics, e.g. JSON file shape).

### Step 7 — Conversion utility

Create `coordination/tracing/compat/__init__.py` and `coordination/tracing/compat/json_to_ndjson.py`:

```python
def convert_legacy_json_to_ndjson(input_path: pathlib.Path, output_path: pathlib.Path) -> int:
    """Read a legacy JSONFileTraceWriter trace; emit NDJSON. Returns span count."""
    ...

def main() -> None:
    """CLI entry: python -m marsys.coordination.tracing.compat.json_to_ndjson <in> <out>"""
    ...
```

Tree-walk the legacy file's `root_span` recursively, emit one NDJSON line per span in pre-order traversal, append a synthetic `stream_completed` marker.

### Step 8 — Update the Spren observability doc

`docs/architecture/spren/06-observability.md` currently shows uppercase `kind` values (`EXECUTION|BRANCH|STEP|GENERATION|TOOL|...`) at line 61, which contradicts the framework's lowercase convention in `tracing/types.py:34`. The framework's wire format is the source of truth. Update the doc:

- Change the `kind` enumeration to lowercase: `execution|branch|step|generation|tool` (the framework's actual five kinds; `validation`, `user_interaction`, `final_response` are not separate `Span.kind` values — validation decisions are events on step spans per `tracing/types.py:26`).
- Change the writer location reference at line 40 from `src/spren/tracing/ndjson_writer.py` (Spren-side) to the framework-side path `packages/framework/src/marsys/coordination/tracing/writers/ndjson_writer.py`.
- Update the timestamp note at line 57 to clarify: framework emits `start_time` / `end_time` as floats internally; the NDJSON writer translates to ISO 8601 UTC strings on serialization.
- Reference the schema version field (`schema_version: 1`) on every NDJSON line so future consumers can pin against a contract.

### Step 9 — Benchmark and document

Synthetic 1000-span run against `NDJSONTraceWriter`:

- Default settings (queue + flush, no `fsync`)
- `fsync_per_span=True`
- (For comparison) the legacy `JSONFileTraceWriter` — re-implement the bench against the deleted writer in a one-off branch if needed; report the comparison in PR description.

Target: > 5000 spans/second on a typical Linux laptop SSD with default settings. Document the result in PR description; if Windows / network mounts are reachable, run the bench there too and note the throughput delta.

### Step 10 — Update CHANGELOG + framework docs

- `CHANGELOG.md` entry under the next release version: NDJSON writer added, `JSONFileTraceWriter` removed, `event_id` and `span_id` migrated to ULID, conversion utility for legacy traces, schema version 1.
- Framework architecture docs for the tracing module (if applicable in this worktree).
- PR description: feature summary, multi-consumer list (with explicit "OTel export is Session 02 territory"), link to this brief, baseline / post-change test counts, benchmark numbers, ULID migration note.

### Files to create

- `packages/framework/src/marsys/coordination/tracing/writers/ndjson_writer.py`
- `packages/framework/src/marsys/coordination/tracing/readers/__init__.py`
- `packages/framework/src/marsys/coordination/tracing/readers/ndjson_reader.py`
- `packages/framework/src/marsys/coordination/tracing/compat/__init__.py`
- `packages/framework/src/marsys/coordination/tracing/compat/json_to_ndjson.py`
- `packages/framework/src/marsys/coordination/tracing/_ids.py` (small ULID helper, optional)
- `packages/framework/tests/coordination/tracing/test_ndjson_writer.py`
- `packages/framework/tests/coordination/tracing/test_ndjson_reader.py`
- `packages/framework/tests/coordination/tracing/test_streaming_integration.py`
- `packages/framework/tests/coordination/tracing/test_json_to_ndjson_compat.py`
- `packages/framework/tests/coordination/tracing/test_multi_consumer.py`

### Files to modify

- `packages/framework/src/marsys/coordination/tracing/collector.py` — add `_stream_span` helper; call it at every span-close point.
- `packages/framework/src/marsys/coordination/tracing/types.py` — `create_span()` uses ULID for `span_id`.
- `packages/framework/src/marsys/coordination/tracing/writers/__init__.py` — drop `JSONFileTraceWriter` export, add `NDJSONTraceWriter`.
- `packages/framework/src/marsys/coordination/tracing/__init__.py` — re-export the new reader if convenient.
- `packages/framework/src/marsys/coordination/status/events.py:15` — `event_id` default factory uses ULID.
- `packages/framework/src/marsys/coordination/orchestra.py:209-215` — instantiate `NDJSONTraceWriter` instead of `JSONFileTraceWriter`.
- `packages/framework/pyproject.toml` — add `python-ulid` runtime dep.
- `packages/framework/CHANGELOG.md` — release entry.
- `docs/architecture/spren/06-observability.md` — kind casing, writer location, timestamp note, schema version.
- Any test file that asserts on the legacy JSON shape — migrate or delete.

### Files to delete

- `packages/framework/src/marsys/coordination/tracing/writers/json_writer.py`

### Files NOT to touch

- TRUNK-CRITICAL: `packages/framework/src/marsys/coordination/orchestra.py` (beyond the like-for-like writer substitution in lines 209-215 documented above), `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`, `coordination/topology/graph.py`.
- `TraceCollector`'s event-handler logic — only the helper-call additions described above. The `_handle_*` method internals (clamped `gen_start`, pending convergence, branch_spans/step_spans persistence) remain untouched.
- `TraceWriter` ABC at `tracing/writers/base.py` — unchanged. New `write_span` is added on the concrete `NDJSONTraceWriter` only.
- Spren-side code (`packages/spren/`) — Spren consumes this in its own follow-up session.

### Load-bearing shapes

The NDJSON wire format is the contract. One JSON object per line, schema-versioned, lowercase `kind`, ISO 8601 UTC timestamps:

```
{"schema_version":1,"ts":"2026-04-27T12:30:45.123Z","kind":"generation","span_id":"01J9X4...","parent_span_id":"01J9X3...","trace_id":"01J9X0...","name":"Generation: claude-opus-4.7","start_time":"2026-04-27T12:30:41.000Z","end_time":"2026-04-27T12:30:45.123Z","duration_ms":4123,"status":"ok","attributes":{"model_name":"claude-opus-4.7","provider":"anthropic","prompt_tokens":1234,"completion_tokens":567,"reasoning_tokens":120,"response_time_ms":4321,"finish_reason":"stop","has_thinking":true,"has_tool_calls":false}}
```

Per-line field summary:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `int` | Pin against changes; bumped only on breaking format change. |
| `ts` | `string (ISO 8601 UTC)` | Wall-clock at the moment this line was written by the writer. Emission time, not the span's `start_time`. |
| `kind` | `string` | Lowercase, one of `execution|branch|step|generation|tool` per `tracing/types.py:34`. |
| `span_id` | `string (ULID)` | 26-char base32, monotonic-orderable per process. |
| `parent_span_id` | `string (ULID) | null` | Null only on the root execution span. |
| `trace_id` | `string (ULID)` | Same trace_id for every span in one `Orchestra.run()`. |
| `name` | `string` | Human-readable label as set in `create_span`. |
| `start_time` | `string (ISO 8601 UTC)` | Translated from internal `Span.start_time: float` (epoch seconds). |
| `end_time` | `string (ISO 8601 UTC)` | Translated from internal `Span.end_time: float`. |
| `duration_ms` | `float` | From `Span.duration_ms`. |
| `status` | `string` | `ok | error`, per `Span.status`. |
| `attributes` | `object` | Kind-specific. Mirrors `Span.attributes` after `TracingConfig` filtering. |
| `events` | `array` | Optional; present only when the span has `Span.events`. |
| `links` | `array` | Optional; present only when the span has `Span.links`. |

The `stream_completed` final marker:

```
{"schema_version":1,"ts":"...","kind":"stream_completed","attributes":{"total_spans":N,"final_status":"success|failure|cancelled","disk_error_count":0,"dropped_span_count":0}}
```

Lines emitted for diagnostic events the writer wants to surface (for example, queue overflow):

```
{"schema_version":1,"ts":"...","kind":"stream_event","attributes":{"event":"dropped_span","dropped_span_count":42}}
```

The Spren observability doc's NDJSON example gets reconciled against this contract in Step 8.

---

## Hard rules

### Multi-consumer justification (mandatory)

- [ ] PR description lists at least the four consumers in "Multi-consumer scope (honest)" above, with the explicit "OTel-protocol export is Session 02 territory" disclaimer.
- [ ] No Spren type imported in this PR.
- [ ] No "if running under Spren" code paths.
- [ ] The reader's tail-follow API is generic (file path + boolean) — no Spren-specific knobs.

### Framework design principles

Per `CLAUDE.md` § design principles:

- DP-001 (pure agent logic): n/a — tracing observes, does not run agent logic.
- DP-002 (centralized validation): n/a — no validation surface touched.
- DP-003 (unified-barrier orchestration): n/a — no orchestration surface touched.
- DP-004 (branch isolation): preserved — `TraceCollector` is the single subscriber; writer receives closed spans whose branch identity is already isolated by collector logic.
- DP-005 (topology-driven routing): n/a.
- DP-006 (adapter pattern): the writer extends `TraceWriter` ABC. The new `write_span` method is an additive hook on the concrete subclass; the ABC's `write` and `close` contracts remain valid for any future writer that doesn't want streaming.
- DP-007 (format pluggability): the on-disk format is one of potentially many — adding an OTel exporter later (Session 02) does not require touching the NDJSON writer.

### No TRUNK-CRITICAL behavior changes

Two TRUNK-adjacent edits in this session, each scoped tight:

1. `coordination/orchestra.py:209-215` — like-for-like writer substitution in the existing tracing-init block. No change to `Orchestra.run()`, constructor signature, or run lifecycle.
2. `coordination/status/events.py:15` — default factory for `event_id` migrates UUID4 → ULID. The declared type (`str`) is unchanged; subscribers that treat `event_id` as opaque continue working.

If implementation requires a non-additive change to `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, or `TopologyGraph`, **stop and escalate via `AskUserQuestion`** before any edit.

### Clean code rules

- Smallest implementation that passes acceptance criteria.
- Delete the legacy writer in this PR; no commented-out blocks, no dual-write paths.
- One concern per file: writer, reader, compat utility, types.
- No descriptive comments for self-naming code — comment WHY when not obvious.

---

## Tests (required for "done")

### Unit tests

`tests/coordination/tracing/test_ndjson_writer.py`:

- Constructor wires up; opens file in append mode with UTF-8 + `\n` newline.
- `write_span` enqueues; the drain task writes one line per span; `f.flush()` is called per span.
- `fsync_per_span=True` calls `os.fsync` (mocked).
- `stream_completed` marker emitted on `close()`; closes the file.
- Closing twice is idempotent.
- Disk-write error: write to a path on a tmpfs that's been remounted read-only (or simulate via mock `OSError` on `f.write`). Writer logs, increments `disk_error_count`, does not raise. Subsequent `write_span` calls keep being attempted (do not silently disable).
- Queue overflow: with `queue_maxsize=10`, enqueue 1000 spans rapidly. Drop-oldest behavior verified; `dropped_span_count > 0`; a `stream_event` warning line is written into the file.
- Schema version present on every line.
- `Span.kind` lowercase round-trip: write a span with each of `execution | branch | step | generation | tool`; reader yields lowercase.

`tests/coordination/tracing/test_ndjson_reader.py`:

- Streaming iteration yields spans in file order.
- `follow=True` blocks on EOF and resumes on append (write some, read, write more, verify reader yields the new spans).
- Truncated trailing line: write 10 lines, manually truncate the last 30%; reader yields 9 spans, `truncated_line_count == 1`, `completion_status == "truncated"`, no exception raised.
- Missing `stream_completed` marker (full lines but no marker — simulating a crash): reader's `completion_status == "crashed"` after streaming completes.
- `to_tree()` reproduces hierarchical `TraceTree` with correct parent-child links.
- Reader opens read-only — verified by attempting to open under a write-locked file on Windows (skip on non-Windows).

`tests/coordination/tracing/test_json_to_ndjson_compat.py`:

- Convert a legacy `JSONFileTraceWriter`-shaped fixture (one is committed under `tests/fixtures/`); reader reconstructs the same `TraceTree` shape.

### Integration tests

`tests/coordination/tracing/test_streaming_integration.py`:

- Full `Orchestra.run()` with the new writer; assert spans arrive in close order; reader reconstructs the tree; matches what the legacy writer would have produced (semantic parity, modulo ID format).
- Two parallel runs writing to two files; spans stay separated by `trace_id`.
- Synthetic crash: kill the orchestration mid-run via `asyncio.CancelledError`; verify the file is valid NDJSON up to the cancellation point; reader recovers what was written and reports `completion_status == "crashed"`.
- Tail-follow during a live run: a second coroutine calls `reader.stream(follow=True)` while the orchestration is still running; spans are delivered to the reader as they're written.
- Slow disk: wrap the writer's drain task with an artificial 100ms delay per write; verify `EventBus.emit` latency does not increase (the bus does not block on the writer).
- Concurrent emit from parallel branches: spawn 5 parallel branches each emitting at high frequency; the file has one valid JSON object per line with no interleaved corruption (each `f.write(line)` is one syscall on the drain task; the drain task is single-threaded so writes are serialized).

### Multi-consumer test

`tests/coordination/tracing/test_multi_consumer.py`:

- A "fake observability sink" that subscribes via `NDJSONTraceReader.stream(follow=True)` and forwards each span to a stub destination — proves the surface is genuinely consumable by non-Spren backends.
- A second test imports the writer, instantiates it directly (bypassing `Orchestra`), and writes spans manually — proves the writer is usable outside the orchestration entry point.

### Framework regression test

Entire `packages/framework/tests/` passes with the SAME counts as baseline. Document baseline + post-change counts in "What was actually built." If a test was migrated (NDJSON-asserting instead of JSON-asserting), document it in the same section.

### Edge-case coverage matrix (required for "done")

| Scenario | Test |
|---|---|
| Partial-line tolerance on read | `test_ndjson_reader.py` — truncated trailing line |
| Disk full during write | `test_ndjson_writer.py` — disk-write error, no raise |
| Crash without `stream_completed` marker | `test_streaming_integration.py` — synthetic cancellation; reader's `completion_status` |
| `EventBus` 5-strike rule does not silently kill the writer | `test_ndjson_writer.py` — error path verifies writer keeps being called |
| Windows file-locking + UTF-8 encoding | `test_ndjson_writer.py` — file opened with explicit encoding + newline; reader opened read-only (Windows-skipped tests where needed) |
| Slow-disk backpressure | `test_streaming_integration.py` — bus latency under slow drain |

---

## Open questions for the framework team

The implementer surfaces these via `AskUserQuestion` BEFORE writing code if any are still unanswered after reading the brief and the cited code:

1. **Default trace file location.** Spren passes `output_dir` per run via `TracingConfig`. Is `{output_dir}/{session_id}.ndjson` the right path scheme, or does the framework prefer `{output_dir}/{trace_id}.ndjson` (since `trace_id` is the run-stable ULID and `session_id` is the orchestrator-internal identifier)?
2. **Schema version field placement.** Brief proposes `schema_version` on every line. Alternative: `schema_version` on a single header line at file start, omitted from subsequent lines. Either is defensible; per-line is simpler for tail-follow consumers that may join mid-file.
3. **Queue overflow policy.** Drop-oldest is the default in this brief; alternative is drop-newest or a hard back-pressure on `_stream_span`. Which fits the framework's stance on "trace loss vs. orchestration delay"?
4. **`fsync_per_span` default.** Off by default for performance. If the framework has a stronger durability stance (Cloud worker scenario), push back.
5. **Conversion utility scope.** Provided as a module function + `python -m` CLI. If the framework prefers a `marsys` CLI subcommand, align.
6. **ULID library choice.** Brief picks `python-ulid` (PyPI). Framework may prefer a vendored implementation to avoid a runtime dep — confirm before pinning.

---

## Sign-off

On completion:

1. Update **What was actually built** below with delta from plan.
2. Update [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) — check Session 01's row.
3. Note the framework release version that ships this feature.
4. Add **Lessons / Surprises** below.

### What was actually built (filled by implementer)

> _Implementer fills this in._
>
> Include: baseline test counts (before change), post-change test counts (regression suite must match; new tests counted separately), framework PR number + URL, framework release version, benchmark numbers (default + `fsync_per_span=True` + comparison vs deleted legacy writer if measured), anything done differently from the plan with reasons.

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._
