# spren

Backend for MARSYS Spren — a FastAPI sidecar that runs locally on `127.0.0.1`,
exposes the REST + SSE + POST API consumed by the Tauri desktop GUI, the system
browser tab, and the Textual TUI (per SP-019, the API is the single source of truth).

In Session 01 the package ships with `/healthz` + `/v1/bootstrap` + per-launch
auth; later sessions add workflow CRUD, the run engine, the meta-agent daemon, and
memory.

The Spren product distributes as a native installer (Tauri-bundled). PyPI users
who want to drive it from raw Python scripts use the `spren.telemetry` subpackage
(`SprenTelemetrySink`) — added in v0.4 alongside the framework's `TelemetrySink` PR.
