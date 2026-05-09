# 05 — Packaging & Distribution

Spren is an end-user product. The framework is a Python library. They ship like what they are, under licenses that match: the framework is **Apache-2.0** (permissive, developer adoption), Spren — daemon, web frontend, desktop shell, and TUI — is **FSL-1.1-ALv2** (Functional Source License, source-available with a 2-year non-compete window before it converts to Apache-2.0).

## Distribution channels

| Channel | Audience | What the user does |
|---------|----------|---------------------|
| **Native installer (front door)** | Default — every user we want | `curl spren.dev/install.sh \| sh` (macOS / Linux) or download `.msi` (Windows) / `.dmg` (macOS) / `.deb` / `.AppImage` (Linux). Single binary per platform. Auto-updater built in. |
| **Tauri-packaged desktop app** | Users who want a native window | `.dmg` / `.msi` / `.deb` / `.AppImage`. Same artifact the install script downloads. Tray, autostart, OS notifications (in later releases). |
| **Homebrew tap / winget / apt** | Power users on each OS | `brew install spren` (macOS, Linux), `winget install spren` (Windows), `apt install spren` (Debian/Ubuntu via PPA). Wraps the same Tauri binary. |
| **npm CLI** | Node-comfortable users (Cursor / Claude Code crowd) | `npm install -g @marsys/spren`. Downloads the platform-native binary on `postinstall`, exposes `spren` in PATH. NOT a JS package. |
| **pipx** | Python power users who want Spren as a managed Python service | `pipx install spren`. Server-only mode (no Tauri shell); Python sidecar binds to localhost; user opens a browser tab. Capabilities reduced (no native tray, no auto-update inside the app — `pipx upgrade` is the path). Required for pipx because Python wheels can't ship a Node/Rust runtime cleanly. |
| **Docker** | VPS / home server / NAS | `docker compose up` from a published `marsys/spren` image. Server-only mode (no Tauri shell); access via browser. |

The Tauri-packaged binary is the canonical artifact. The native installer is the front door for every user we want; the secondary channels either wrap the same artifact (brew / winget / apt / npm) or run a reduced server-only mode that uses the same Python sidecar without the Tauri shell (pipx / Docker).

`pip install spren` is deliberately not a channel. Spren's audience treats tools the way they treat Cursor or Claude Code — they install apps, not Python packages — and forcing pip would lock the frontend into "static export served by FastAPI" because Python wheels can't ship a Node runtime cleanly. It would also lock out the Rust Tauri shell.

The marsys framework, by contrast, is shipped only as `pip install marsys` from PyPI. It is an Apache-2.0 Python library with a developer audience; native installers do not apply.

## Architecture inside the Tauri-packaged binary

```
spren (Tauri binary, per platform)
├── Rust shell                                   # Tauri 2 — installer, auto-update, sidecar lifecycle
│                                                # tray + native menus + autostart (later releases)
├── Bundled Python runtime + sidecar             # PyInstaller-built single-file binary
│   └── spren-sidecar-<triple>
│       ├── packages/framework/src/marsys        # marsys framework (in-process)
│       ├── packages/spren/src/spren             # Spren backend (FastAPI + meta-agent daemon)
│       └── all Python deps (vendored)
├── Static web bundle                            # Vite build output of apps/web/
│   └── index.html, assets/                      # Loaded by Tauri webview AND served by Python sidecar
└── (later) bundled TUI binary                   # Textual app, separately invokable as `spren tui`
```

The Tauri shell ships the static web bundle as app resources; it loads them through the system webview and routes API calls to the Python sidecar. The Python sidecar additionally serves the same static bundle on `/` for browser-tab mode (so a user can `spren launch --browser` and use their preferred browser instead of the Tauri window). The bundle lands inside the Spren wheel via setuptools' `package-data` (`spren = ["_webui/**/*"]`); FastAPI conditionally mounts `StaticFiles` only when `_webui/index.html` exists, so dev mode with an empty `_webui/` doesn't shadow Vite.

## CLI surface

A single binary exposes everything:

```
$ spren                              # default — same as `spren launch`
$ spren launch                       # start daemon + open Tauri window
$ spren launch --browser             # start daemon + open default system browser
$ spren launch --headless            # start daemon only; no UI surface
$ spren tui                          # start daemon if not running + open TUI
$ spren memory <subcommand>          # markdown KB CLI (show / edit / remember / forget / why / verify-index) — v0.4+
$ spren expose                       # cloudflared tunnel for webhook channels — v0.5
$ spren --version
```

There is no separate `marsys studio` command. Spren is the single product; manual visual building works alongside the meta-agent, with the meta-agent "armed" or not at runtime.

Every subcommand operates on the same daemon. If the daemon is not running, the CLI starts it and waits for the ready signal (a single stdout line of the form `spren-ready: port=<N> token=<T> data-dir=<P>` that anything spawning the sidecar parses). If it's running, the CLI attaches.

The pipx / Docker channels expose the same `spren` command but skip the Tauri shell: `spren launch` in pipx mode opens a browser tab; in Docker mode the user accesses the published port from outside the container.

## Build pipeline

```
apps/web (pnpm)                       packages/spren (uv)              apps/desktop (cargo)
     │                                          │                              │
     │ pnpm --filter @marsys/spren-web build    │ uv build → wheel             │ cargo build → binary
     │   → apps/web/dist/                       │                              │
     ▼                                          │                              │
copy dist/ → packages/spren/src/spren/_webui/   │                              │
(setuptools package-data ships it in the wheel) │                              │
                                                ▼                              │
                                  PyInstaller --onefile → spren-sidecar        │
                                  (per platform: macos arm64/x86_64,           │
                                   linux x86_64/arm64, windows x86_64)         │
                                                │                              │
                                                └──────────────┬───────────────┘
                                                               ▼
                                                Tauri bundles shell + sidecar +
                                                static webui → installer per platform
                                                               │
                                                ┌──────────────┼──────────────────────┐
                                                ▼              ▼                      ▼
                                           install.sh      OS package           PyPI / Docker /
                                           CDN drop        managers (brew,      npm (different
                                                           winget, apt, AUR)    packaging paths
                                                                                for the same artifact
                                                                                where applicable)
```

Coordinated by `Justfile` recipes. Today: `just build` runs the Vite production build and copies the output into `packages/spren/src/spren/_webui/`; `just install` provisions Python (uv workspace `sync`), JS (pnpm), Rust (cargo fetch), and the Tauri CLI (`cargo install tauri-cli --version "^2"`); `just dev` runs the FastAPI sidecar plus Vite dev server; `just dev-desktop` runs Vite plus the Tauri shell (`cd apps/desktop && cargo tauri dev`); `just test` runs all suites (`uv run --package marsys pytest …`, `uv run --package spren pytest …`, `uv run --package spren-tui pytest …`, `pnpm --filter '@marsys/spren-web' test --run`, `cargo test`).

The release pipeline (Session 10) extends this with PyInstaller + Tauri bundling + signing + per-channel publish recipes (`build-sidecar`, `build-desktop`, `build-pipx`, `build-docker`, `publish-<channel>`).

## Auto-update

| Channel | Update mechanism |
|---------|------------------|
| Native installer | Tauri auto-updater. GitHub Releases as the update server. Silent download in background, install on next launch with notify-and-confirm UX. Signature verification on every update. |
| Homebrew / winget / apt | Channel's native upgrade flow (`brew upgrade spren`, `winget upgrade`, `apt upgrade`). Spren still self-checks and notifies if the Tauri auto-updater is disabled, so users on package managers get nudged. |
| npm | `npm update -g @marsys/spren`. The npm post-install rerun fetches the matching native binary. |
| pipx | `pipx upgrade spren`. App prints "update available" in `spren launch` output. |
| Docker | `docker compose pull && docker compose up -d`. Same nudge. |

Migrations always run on launch and are forward-only (SP-006).

## Code signing and notarization

| Platform | What's required |
|----------|------------------|
| macOS | Apple Developer Program membership ($99/yr). `codesign` + notarization via `notarytool`. CI runs notarization async (minutes-to-hour); release flagged "in notarization" until done. Required for Gatekeeper to allow launching without warnings. |
| Windows | EV code-signing certificate (preferred for SmartScreen reputation) or standard Authenticode. `signtool.exe` invoked in CI. Without signing, Windows shows SmartScreen warnings on first launch. |
| Linux | No platform-mandated signing for the AppImage / .deb. We sign the install.sh script payload with our GPG key; the install script verifies. Distros that ship Spren via apt/AUR follow distro signing conventions. |

Signing keys live in CI secrets. The install script is HTTPS-only (HSTS); it verifies the downloaded binary against a signature served alongside.

## Versioning + release

- Tauri-bundled binary, Spren wheel, Vite bundle, and TUI binary all share one version per release (locked at build time)
- Semver: `<major>.<minor>.<patch>`
- Tags: `spren-vX.Y.Z` (avoid conflict with framework tags `marsys-vX.Y.Z`)
- Release artifacts: native installers (macos arm64, macos x86_64, linux x86_64, linux arm64, windows x86_64), Python wheel on PyPI, Docker image on GHCR, npm package on the npm registry

Each release publishes a manifest at `spren.dev/releases/<version>.json` with checksums + signatures for every channel's artifact. The auto-updater and install script consume this manifest.

## Bundle size budgets

- Tauri installer: target < 80 MB compressed (Tauri shell ~10 MB + PyInstaller sidecar ~50 MB + static web bundle ~10–20 MB)
- pipx-installed wheel: target < 50 MB (no Tauri shell, no bundled Python runtime)
- Docker image: target < 500 MB compressed (slim base + wheel + native deps)
- npm package on disk after postinstall: same as Tauri installer since it downloads that artifact

If we exceed these, debug before shipping.

## Storage paths

Per-user data directory follows platform conventions (resolved via `platformdirs`):

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/spren/` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/spren/` |
| Windows | `%LOCALAPPDATA%/spren/` |

Inside the data directory:

```
spren/
├── data/
│   ├── spren.db                  # SQLite — workflows, runs, schedules, channels, settings, secrets
│   ├── files/{file_id}/...           # uploaded user files
│   └── runs/{run_id}/                # trace.ndjson + workflow.json + artifacts
├── sandbox/
│   ├── shared/memory/                # markdown KB
│   ├── shared/skills/                # skill catalog
│   └── teams/<slug>/                 # team-scoped sandboxes (later releases)
├── logs/spren.log                    # rotated 10 MB × 5
└── runtime/auth-token                # current session's per-launch token (0600)
```

The pipx and Docker channels use the same layout (Docker mounts a volume; the data directory is configurable via `--data-dir`).

## Localhost binding and auth

Per SP-002:
- The Python sidecar binds to `127.0.0.1:<port>`. Default port `8765`; if taken, a random free port is chosen.
- Per-launch auth token (32-byte URL-safe random) generated at startup.
- In Tauri mode: the token is injected into the webview as a window variable.
- In browser-tab mode: the token is passed via URL fragment (`http://127.0.0.1:<port>/#token=...`).
- Every API request carries `Authorization: Bearer <token>`. Constant-time comparison.

## Docker mode

`docker-compose.yml`:

```yaml
services:
  spren:
    image: marsys/spren:{version}
    ports:
      - "127.0.0.1:8765:8765"
    volumes:
      - spren_data:/data
    environment:
      SPREN_DATA_DIR: /data
      SPREN_AUTH_TOKEN: ${SPREN_AUTH_TOKEN:-}
volumes:
  spren_data:
```

The image is built FROM `python:3.12-slim` and installs the Spren wheel. One uvicorn process per container; no multi-process supervisor. The Tauri shell does not run in Docker (no display); the user accesses the bundled web UI via browser to `http://127.0.0.1:8765`.

**Image registry:** `marsys/spren` on Docker Hub / GHCR. Tags follow Spren's version (`marsys/spren:0.3.0`, `marsys/spren:latest`).
