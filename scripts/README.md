# Spren assessment harness

Tooling for autonomous product assessment — not pass/fail tests, but
journey-shaped probes that exercise the real product and produce output
Claude Code can read and triage.

Two complementary surfaces:

- **`scenarios/`** — Python scripts that boot a real sidecar and walk
  through realistic user journeys against the live FastAPI surface
  (create workflow → lint → save → run → inspect → re-run, etc.).
  Output is structured findings printed to stdout + JSON in
  `scenarios/output/<journey>-<timestamp>.json`. The scenarios assert
  outcomes a real user would see, not endpoint-level contracts (those
  are the pytest suite's job).

- **`visual_audit/`** — Playwright spec that boots the dev stack,
  navigates every implemented route in realistic states (empty,
  populated, focused, error), captures full-page screenshots at
  `deviceScaleFactor: 2` for retina detail, plus element-scoped clips
  for fine inspection (the orb, the lint chip, agent config form,
  trace tree, provenance badges). PNGs land in
  `visual_audit/output/<route>__<state>.png`. Claude Code reads them
  via the `Read` tool, which handles images natively.

## Running

```powershell
# Backend journey
uv run --package spren python scripts/scenarios/workflow_crud_journey.py

# All scenarios
uv run --package spren python scripts/scenarios/run_all.py

# Visual audit
pnpm --filter @marsys/spren-web exec playwright test ../../scripts/visual_audit/capture.spec.ts
```

## Output directories

Both `scenarios/output/` and `visual_audit/output/` are gitignored.
The harness code itself is checked in; runs are local artifacts.
