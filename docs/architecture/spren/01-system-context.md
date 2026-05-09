# 01 — System Context

## The MARSYS product family

There are four distinct systems in the MARSYS product family. Spren sits among them at a specific position.

```
┌────────────────────────────────────────────────────────────────────────┐
│                       MARSYS PRODUCT FAMILY                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────┐         ┌──────────────────────────────────┐    │
│  │  marsys (this    │ ◄──────┤ Spren (this app)                 │    │
│  │  repo, Python    │ uses    │ Python + Rust + Vite/React + TUI │    │
│  │  framework, OSS) │ as tool │ Local-first, OSS                  │    │
│  └────────▲─────────┘         └──────────────────────────────────┘    │
│           │ depends on                                                 │
│           │                                                            │
│  ┌────────┴──────────┐         ┌──────────────────────────────────┐   │
│  │ MARSYS Cloud      │ ◄──────┤ MARSYS Studio (proprietary)      │   │
│  │ (proprietary,     │         │ React 19 + TS                     │   │
│  │  Go, separate     │         │ Hosted UI; client of Cloud        │   │
│  │  repo)            │         └──────────────────────────────────┘   │
│  └───────────────────┘                                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

| System | Repo | Tech | License | Distribution | Audience |
|--------|------|------|---------|--------------|----------|
| **marsys framework** | this umbrella (`packages/framework/`) | Python | Open-source | `pip install marsys` (PyPI) | Developers building agents in code |
| **MARSYS Spren** | this umbrella (`packages/spren/` + `apps/web/` + `apps/desktop/` + `apps/tui/`) | Python + Rust shell + Vite/React + Textual | Open-source | Native installer (front door); secondary: brew / winget / apt / npm / pipx / Docker | AI builders who want a visual / always-on surface; terminal-resident builders via the TUI |
| **MARSYS Cloud** | `/home/rezaho/research_projects/MARSYS_Cloud` | Go | Proprietary | SaaS (commercial) | Enterprises and teams paying for hosted execution |
| **MARSYS Studio** | `/home/rezaho/research_projects/MARSYS_Studio` | React 19 + TS | Proprietary | SaaS (client of Cloud) | Same audience as Cloud |

## Spren's relationship to each

### Spren → marsys framework (CONSUMES, with a separate execution model)

Spren has its **own continuous-execution runtime** ([`09-meta-agent.md`](./09-meta-agent.md)) that does NOT go through `Orchestra.run()`. The meta-agent's needs (long-running daemon, persistent memory across sessions, event-driven ingress, dynamic sub-agent spawning, multi-channel I/O) don't fit Marsys's bounded request/response orchestration. Reshaping Marsys to support both modes would touch every TRUNK-CRITICAL component (Orchestra, Orchestrator, RealRuntime, ValidationProcessor, TopologyGraph) — too invasive, and out of scope for the framework's contract with its other consumers (Cloud, Studio, third-party Python users).

The seam is exactly three doors (SP-018):

1. **`Orchestra.run(topology, task) → OrchestraResult`** — the meta-agent calls this as a tool when it decides "run workflow X." Finite workflow runs go through Marsys; the meta-agent itself does not.
2. **`EventBus.subscribe(event_type, listener)`** — Spren subscribes to in-process workflow lifecycle events (`BranchCreatedEvent`, `BranchCompletedEvent`, tracing/status events) and translates them to AG-UI events for the SSE stream.
3. **`TelemetrySink` Protocol** — a span-shaped, async, hierarchical observability hook. Sinks register with `TraceCollector` (NOT `EventBus` directly); the collector fans closed spans out at every `span.close(...)` site, after running them through the configured `SecretRedactor`. Spren ships one implementation (`SprenTelemetrySink` at `packages/spren/src/spren/telemetry/`) so a Python developer can run a workflow from raw code with `Orchestra.run(..., telemetry=[SprenTelemetrySink(...)])` and have it appear in the local Spren UI. Vendor adapters (LangSmith, Phoenix, Langfuse) fit the same protocol shape — contributor-welcome, none shipped here. The framework knows about the sink protocol; it knows nothing about Spren.

Spren also uses MARSYS topology types (Node, Edge, PatternConfig) as the data model for the visual builder, and uses MARSYS agent + tool primitives inside the meta-agent's sub-agents (each sub-instance and team manager is an instance of `marsys.agents.Agent` with scoped tools and persona configuration). These are library-level imports — no special framework hooks.

Spren MUST NOT modify TRUNK-CRITICAL framework components (SP-001). The two systems are intentionally orthogonal: Marsys stays great at finite multi-agent orchestration; Spren stays great at being a long-running personal AI ops engineer that uses Marsys when it needs to run a workflow.

### Spren ⟂ MARSYS Cloud (INDEPENDENT)

Spren does NOT use MARSYS Cloud and is not aware of it. It runs the framework in-process. Users who want hosted execution use MARSYS Cloud + Studio instead.

### Spren ⟂ MARSYS Studio (NOT competing on feature parity)

Spren overlaps in concept (visual builder + run inspection) but not in audience or distribution model:

| Dimension | Spren | MARSYS Studio |
|-----------|-----------|---------------|
| Hosting | Local (laptop or self-hosted) | Hosted SaaS (client of Cloud) |
| Audience | OSS framework users | Enterprise/team customers |
| Source | Open-source | Proprietary |
| Multi-user | No (single-user local) | Yes (orgs, teams, RBAC) |
| Persistence | Local SQLite + filesystem | Cloud DB |
| Accounts | None — BYO API keys | Org/user accounts |

**Positioning rule:** when a feature could go either way (e.g., team workflows, role-based access, sharing), it goes to Studio. Spren stays focused on the single-user-local case so we don't accidentally cannibalize the proprietary product.

This rule is owned by the user — see open question in [`docs/implementation/00-overview.md`](../../implementation/spren/00-overview.md). Defer ambiguous calls.

## External integrations (planned)

| Integration | Purpose | Version | Mechanism |
|-------------|---------|---------|-----------|
| LLM providers (OpenAI / Anthropic / Google / OpenRouter / xAI / local HF / vLLM) | Execute agent calls | v0.3 | marsys framework's existing model adapters; Spren just routes BYO keys |
| OS keychain (macOS Keychain / Windows Credential Manager / Linux Secret Service) | API key storage | v0.3 | `keyring` Python library |
| Telegram (Bot API long-polling) | Inbound messenger trigger | v0.4 | Polling — no inbound tunnel needed |
| Discord (Gateway WebSocket) | Inbound messenger trigger | v0.4 | Gateway connection — no inbound tunnel needed |
| Cloudflare Tunnel (`cloudflared` binary) | Expose local server for webhook-only integrations | v0.4 | Bundled `spren expose` subcommand |
| Slack | Webhook + Slash commands | v0.5 | Requires tunnel |
| WhatsApp Business API | Webhook | v0.5 | Requires tunnel |

## Framework contributions Spren motivates

Spren consumes the framework cleanly via the three seams (SP-018), but a few framework-side improvements are useful broadly and Spren happens to be the proximate motivator:

1. **NDJSON streaming trace writer** to replace the existing write-at-end JSON writer. Fixes a latent bug (crash mid-run loses the trace). Useful for any framework user. Lives in `packages/framework/src/marsys/coordination/tracing/`.
2. **`TelemetrySink` Protocol via `TraceCollector`** — a span-shaped, async, hierarchical interface so any external observability backend (Spren, LangSmith, Phoenix, Langfuse, custom) receives closed spans without hooking into framework internals. The framework defines the protocol and routes through `TraceCollector` (with a configurable `SecretRedactor` at the boundary); consumers register sinks via `Orchestra(telemetry=[...])`. Spren provides one implementation; vendor adapters fit the same shape.
3. **Workflow definition serializer** — round-trip Pydantic ↔ runtime topology conversion in `packages/framework/src/marsys/coordination/topology/serialize.py`, used by Spren's Python-file import path and by anyone who wants to persist a topology.
4. **Semantic linter pass** in `coordination/topology/analyzer.py` (validate tool/agent refs beyond structural checks). Useful framework-wide.

These are NOT TRUNK-CRITICAL — they extend rather than replace existing framework code. They land via the framework's own PR flow.
