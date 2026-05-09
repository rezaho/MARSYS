# 00 — Overview

## What Spren is

A local-first, open-source product on top of the [marsys Python framework](../framework/overview.md). It is one product with two faces:

1. **A continuously-active personal AI assistant** — the "meta-agent." A daemon with persistent memory, a multi-channel inbox (web / Slack / Telegram / Discord / etc.), the ability to spawn sub-agents and team managers, and the authority to act on the user's behalf with appropriate confirmation flows. It is the agent that runs your other agents. See [09-meta-agent.md](./09-meta-agent.md).
2. **A visual builder + run inspector** — a Vite + React + TanStack Router single-page app for designing marsys workflows and inspecting traces. Workflows authored visually, by the meta-agent, by Python code (imported), or from a template all land in the same workflow store with a `provenance` annotation.

The meta-agent is the headline capability. The visual builder is the canonical authoring surface. The two share one workflow store, one run store, one memory KB. Whether the meta-agent is "armed" (handling events autonomously) is a runtime setting; manual visual building works alongside.

Compared to Langflow / Open WebUI, the difference is the always-on assistant — Spren is not a request/response chat surface, and the home page is not a chat box.

## Surfaces

Spren ships three product surfaces, all consuming one FastAPI API (SP-019):

- **Desktop GUI** (Tauri webview) — the default experience. Native installer per platform; tray, autostart, OS integration grow over releases.
- **Browser GUI** — same Vite bundle, served by the Python sidecar at `http://127.0.0.1:<port>/`. For users who prefer their existing browser. Available in every channel.
- **TUI** (Textual, Python) — `spren tui` opens a four-pane terminal interface (Now / Inbox / Activity / Chat). Same domain, different rendering. For users who live in the terminal.

## Who it's for

- Developers and enthusiasts who already use the marsys framework programmatically and want a visual surface
- AI builders who pay for ChatGPT Plus, use Cursor or Claude Code, may have built a custom GPT — comfortable with technical concepts but not necessarily writing Python every day
- People who want to run agents on their own machine, with their own keys, without subscribing to a hosted service

## Primary user surfaces

1. **Meta-agent home** — the main page. NOT a chat box. A four-surface command center:
   - **Now** — what the agent is currently working on; active sub-instances and their focus
   - **Since you were away** — items awaiting your decision (suggestions, alerts, completed actions you should know about)
   - **Activity** — chronological log of agent actions; drill into any for the full thought trail
   - **Chat input** — talk to the agent directly; one of four surfaces, not the page
   See [09-meta-agent.md](./09-meta-agent.md) for the full design.
2. **Workflow canvas** — `@xyflow/react`-based visual builder for marsys topologies (nodes = agents/users/system/tools, edges = invoke/notify/query/stream); workflow provenance shown inline.
3. **Run inspector** — nested span timeline (Langfuse idiom) with token, latency, and cost chips per span.
4. **Run history** — searchable list of past runs with filtering and re-run.
5. **Triggers** — manual run-now, scheduled cron, webhook + messengers (later releases).
6. **Memory** — view/edit the agent's persistent knowledge (markdown KB editable in `$EDITOR` or via the UI). See [10-memory-architecture.md](./10-memory-architecture.md).
7. **Python file import** — drop a `.py` file written against the marsys framework; Spren parses agent definitions, topology, and configs and materializes them as a workflow record (provenance: `code`).

## Hard product constraints

- **Local-first.** Single user on a laptop is the primary target. A small team self-hosting on a VPS is a valid secondary configuration.
- **BYO API keys.** No gateway service. Keys live on the user's machine (OS keychain preferred).
- **Open-source.** Permissive license. Other users self-host and modify freely.
- **Distinct from MARSYS Cloud and MARSYS Studio (proprietary).** Spren does not implement multi-tenant control plane, organization/billing, or hosted execution.
- **Native distribution.** Spren ships as a native installer per platform (Tauri-bundled), with secondary channels (Homebrew / winget / apt / npm / pipx / Docker) wrapping the same binary or a reduced server-only mode. The marsys framework, in contrast, ships as `pip install marsys` because its audience is Python developers.

## What problem this solves

Today the only way to use the marsys framework is to write Python. That excludes the majority of the audience the framework's capabilities deserve. A visual app:

- Lowers the barrier to building real multi-agent workflows
- Makes runs inspectable for non-developers (AI observability is the primary value prop)
- Provides a deployment story (install + run) that doesn't require the user to set up a Python environment from scratch

## Non-goals (explicit)

- Hosting other users' workflows (that's MARSYS Cloud)
- Multi-tenant team workflows, RBAC, audit logs (out of scope; if needed, see MARSYS Studio)
- Replacing programmatic API for power users (the framework's Python API stays the canonical way to use marsys; Spren is a complementary surface, not a replacement)
- General-purpose personal assistance beyond agent-orchestration scope (no calendar, email, generic Q&A — the meta-agent's role is "your AI ops engineer," not "your Jarvis")

## Critical references

- This repo's CLAUDE.md (project rules)
- [01-system-context.md](./01-system-context.md) — relationships to other systems
- [08-design-principles.md](./08-design-principles.md) — cross-cutting invariants
- [09-meta-agent.md](./09-meta-agent.md) — the meta-agent execution model, agent hierarchy, persona, authority, sandbox
- [10-memory-architecture.md](./10-memory-architecture.md) — memory tiers, write/read paths, consolidation, security
- [`docs/implementation/00-overview.md`](../../implementation/spren/00-overview.md) — version map and rollout
