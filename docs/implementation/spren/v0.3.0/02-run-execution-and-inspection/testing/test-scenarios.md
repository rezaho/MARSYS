# Bundle 02 — Run Execution and Inspection — Test Scenarios

> Status: **stub**. Fleshed out when Session 05 plan is drafted (the bundle-end scenarios depend on the full feature set 04+05 ships).
>
> **Bundle**: 02 — Run execution and inspection
> **Sessions**: 04 (Run execution + AG-UI streaming + cost), 05 (Run inspection: trace viewer + file upload + run history UI)
> **Demo-able outcome**: from Bundle 01's saved workflow, user clicks Run, watches the workflow execute live with token-by-token streaming visible in the Spren orb's "speaking" state, sees cost computed and displayed on completion, navigates to `/runs/{id}` to see the full nested-span trace viewer with per-span cost chips. Cancel mid-run cleanly. Reconnect after a network blip without losing event continuity.

## Cross-bundle dependencies

- Bundle 01 (visual-builder) ships the workflow the user will run + the design system + the canvas Run button.
- Framework Session 06 (AG-UI translator) lands before Bundle 02 implementation begins.
- Framework Session 01 (NDJSON streaming tracing) is already shipped — the trace.ndjson file backs the SSE reconnect-replay path.

## Golden-path scenarios

(Filled in when Session 05 is drafted. Will cover G-08 through G-12: run a workflow, cancel mid-run, reconnect after network blip, browse run list, click into run inspector with full trace.)

## Edge-case scenarios

(Filled in when Session 05 is drafted. Will cover: failed runs, runs that exceed budget, runs with mid-run user-interaction prompts, runs with file attachments, multiple simultaneous runs, draft cleanup correctness, stale rate-table fallback.)

## Exploratory variations

(For the testing agent to vary against the golden + edge scenarios — different workflow shapes, different network conditions, different time-of-day, etc.)

## Visual regression baselines

(Filled in when Session 05 is drafted. Will extend Bundle 01's visual baseline set with: Run button states, completion toast, `/runs` list page, `/runs/{id}` trace viewer, agent config rail during a running workflow.)

## Manual smoke checklist

(Filled in when Session 05 is drafted.)
