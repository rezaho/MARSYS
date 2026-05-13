# Bundle 03 — Meta-Agent Becomes Alive — Test Scenarios

> Status: **stub**. Fleshed out fully when Session 08 plan is drafted (the bundle-end scenarios depend on the full feature set 06+07+08 ships).
>
> **Bundle**: 03 — Meta-agent becomes alive
> **Sessions**: 06 (Memory foundation), 07 (Meta-agent core + persona), 08 (Meta-agent capabilities + memory write paths)
> **Demo-able outcome**: from a fresh data dir, user installs Spren, completes the 5-archetype onboarding picker, lands on the Spren orb home with the chosen archetype's voice + visual variant. The meta-agent reads workflows + traces + memory; suggests-with-confirm on writes; dispatches workflows with hard-rail confirmation; respects the per-day budget cap; ticks heartbeat every 30 minutes; runs the consolidation pass on its own cadence (24h+5sessions); evolves its persona through the bond mechanism over time. The agent's voice is recognisably the chosen archetype; the bond grows through use; user can `spren persona log/why/diff/approve/reject/revert` and `spren bond reset` from the CLI.

## Cross-bundle dependencies

- Bundle 01 (visual-builder) ships the workflows the meta-agent dispatches via `run_workflow`.
- Bundle 02 (run execution + inspection) provides the run lifecycle + trace endpoints the agent's `read_run` and `read_trace` tools wrap.
- Framework Session 06 (AG-UI translator) — already required by Bundle 02; same dependency for Bundle 03's `run_workflow` tool.
- No new framework dependencies for Bundle 03 (v0.4 introduces `TelemetrySink` + pause/resume Spren-side surfaces).

## Setup steps

(Filled in when this stub is promoted. Will mirror Bundle 01 + Bundle 02's preamble; adds: `OPENAI_API_KEY` or equivalent for the meta-agent's cheap + expensive models; allowance for the consolidation pass to run during testing — typically tests will trigger consolidation via a CLI override (`just consolidate-now`) rather than wait 24 hours.)

## Golden-path scenarios

(Filled in when Session 08 is drafted. Will cover G-17 through G-44 from the per-session success criteria, organized by user journey.)

Sketch:

- **G-17..G-23** (memory foundation, Session 06): bootstrap, CLI roundtrip, vim → indexer → SQL, forget + restore, verify-index detects tampering, sandbox permission tier, OS-level launcher denies network.
- **G-24..G-31** (meta-agent core, Session 07): onboarding flow per archetype, heartbeat tick, watcher fires, execute_shell hard rail, voice-drift logged, crash recovery, cost-budget guard.
- **G-32..G-44** (meta-agent capabilities, Session 08): each tool roundtrip, suggest-with-confirm with standing approval, consolidation pass end-to-end, TMS gate triggers, CUPMem 4-state output, supersession decision matrix, PersonaReflection generates proposal, persona approve, bond reset, budget exhaustion behavior, stated-vs-observed surfacing.

## Edge-case scenarios

(Filled in when Session 08 is drafted. Will cover):

- Indexer drift after manual SQL tampering (the user runs `verify-index`).
- Crash mid-consolidation-pass (next start: pass replays from the same delta; no double-commit).
- A user-direct fact contradicts a stable core fact via the escape valve → TMS gate triggers → user sees a dispute prompt before the fact lands.
- A fact's content_hash mismatch on read → IndexDriftDetected event.
- LLM provider returns malformed JSON during consolidation Stage 2 → format validator at indexer rejects → consolidation_errors.md captures the raw output.
- Standing approval revoked mid-conversation: agent should re-prompt on the next invocation of the formerly-approved tool.
- The agent's reasoning loops over a hard-rail-blocked tool — it should NOT keep trying; the inbox flow naturally handles this (NeedsDecision event → conversation continues only after user response).
- `spren persona approve` for a proposal that became stale (axis was already evolved by another approved proposal in between) → CLI surfaces conflict, asks for re-review.
- Two tools that overlap on functionality (e.g., `read_file` for a fact file vs `lookup_facts` for the same fact) — the agent's prompt should prefer `lookup_facts` for typed retrieval; verify in voice tests.
- A fact file with malformed YAML frontmatter is hand-edited via vim; on save, the indexer's format validator rejects; user sees the error in `consolidation_errors.md`; markdown stays editable (the indexer doesn't refuse the file; it skips it until the YAML is repaired).

## Exploratory scenarios

(For the testing agent to vary against the golden + edge scenarios — different archetypes, different conversation lengths, different memory-store sizes, different tool-call sequences, different user-pushback patterns.)

Sketch:

- Conversation in each of the 5 archetypes: the agent's voice should be recognisably distinct.
- Persona evolution after 30 sessions: does the bond shape feel earned? Does the user feel surprised by which axes drifted?
- Memory volume stress: load 10K facts; verify lookup_facts performance, recall performance, indexer responsiveness.
- Cost stress: many small low-budget LLM calls vs few expensive ones — verify the budget meter accounts correctly.
- Bond reset edge: reset mid-conversation; verify the daemon transitions cleanly without corrupting state.
- Concurrent CLI + GUI: user makes a memory edit via vim while the agent is reasoning over the same fact in the chat — verify no race conditions.

## Visual regression baselines

(Filled in when Session 08 is drafted. Will extend Bundle 01 + Bundle 02's visual baseline set with):

- Onboarding picker (5 archetype cards with their orb variants).
- Each archetype's home page (5 baselines).
- Suggest-with-confirm UI for each tool kind.
- Persona-evolution proposal surface (CLI output + future v0.4 UI).
- Bond status UI.
- Voice-drift surfacing (when consolidation surfaces drift in active_context).

## Manual smoke checklist

(Filled in when Session 08 is drafted. Will be ≤8 items, ≤8 minutes.)

Sketch:

- [ ] **U-14** — Session 06 manual smoke (memory CLI roundtrip).
- [ ] **U-15** — Session 07 manual smoke (each archetype onboarding).
- [ ] **U-16** — Session 08 manual smoke (J-7 + J-8 + J-9 — workflow run + bond reset + budget exhaustion).
- [ ] U-Bundle03-1 — daemon boots clean from a fresh data dir; bootstrap creates the full KB tree; user picks archetype; orb appears.
- [ ] U-Bundle03-2 — heartbeat fires within 30 min of cadence; no LLM call cost when nothing's pending.
- [ ] U-Bundle03-3 — agent dispatches a workflow via the home-page chat; trace visible in /runs/{id}; cost tracked correctly.
- [ ] U-Bundle03-4 — consolidation pass runs (manually triggered); produces a git commit on `personas/` directory; CLI shows the new history.
- [ ] U-Bundle03-5 — bond reset preserves memory + workflows; new archetype pick produces visibly different voice in the next session.

If any item fails: bundle does not ship.
