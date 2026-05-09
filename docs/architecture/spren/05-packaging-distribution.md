# 05 — Packaging & Distribution

Spren is an end-user product. The framework is a Python library. They ship like what they are.

## Distribution channels

| Channel | Audience | What the user does |
|---------|----------|---------------------|
| **Native installer (front door)** | Default — every user we want | `curl spren.dev/install.sh \| sh` (macOS / Linux) or download `.msi` (Windows). Single binary per platform. Auto-updater built in. |
| **Tauri-packaged desktop app** | Users who want a native window | `.dmg` / `.msi` / `.deb` / `.AppImage`. Same artifact the install script downloads. Tray, autostart, OS notifications (in later releases). |
| **Homebrew tap / winget / apt PPA** | Power users on each OS | `brew install spren` etc. Wraps the same Tauri binary. |
| **npm CLI** | Node-comfortable users (Cursor / Claude Code crowd) | `npm install -g @marsys/spren`. Downloads the platform-native binary on `postinstall`, exposes `spren` in PATH. |
| **pipx** | Python power users who want Spren as a managed Python service | `pipx install spren`. Headless mode (no Tauri shell); Python sidecar binds to localhost; user opens a browser tab. Capabilities reduced (no native tray, no auto-update inside the app — `pipx upgrade` is the path). |
| **Docker** | VPS / home server / NAS | `docker compose up` from a published `marsys/spren` image. Server mode (no GUI shell); access via browser. |

The Tauri-packaged binary is the canonical artifact. The other channels are different ways of getting it (or, for pipx and Docker, a reduced server-only mode that runs the same Python sidecar without the Tauri shell).

The marsys framework, by contrast, is shipped only as `pip install marsys` from PyPI. It is a Python library with a developer audience; native installers do not apply.

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

The Tauri shell ships the static web bundle as app resources; it loads them through the system webview and routes API calls to the Python sidecar. The Python sidecar additionally serves the same static bundle on `/` for browser-tab mode (so a user can `spren open --browser` and use their preferred browser instead of the Tauri window).

## CLI surface

A single binary exposes everything:

```
$ spren                              # default — same as `spren launch`
$ spren launch                       # start daemon + open Tauri window
$ spren launch --browser             # start daemon + open default system browser
$ spren launch --headless            # start daemon only; no UI surface
$ spren tui                          # start daemon if not running + open TUI
$ spren up                           # alias of `spren launch`, kept for muscle-memory parity
$ spren memory <subcommand>          # markdown KB CLI (show / edit / remember / forget / why / verify-index)
$ spren expose                       # cloudflared tunnel for webhook channels (later releases)
$ spren --version
```

Every subcommand operates on the same daemon. If the daemon is not running, the CLI starts it and waits for the ready signal. If it's running, the CLI attaches.

The pipx / Docker channels expose the same `spren` command but skip the Tauri shell: `spren launch` in pipx mode opens a browser tab; in Docker mode the user accesses the published port from outside the container.

## Build pipeline

```
apps/web (pnpm)                packages/spren (uv)              apps/desktop (cargo)
     │                                  │                                │
     │ pnpm build → dist/               │ uv build → wheel               │ cargo build → binary
     │                                  │                                │
     ▼                                  │                                │
copy dist/ → packages/spren/_webui/     │                                │
                                        ▼                                │
                              PyInstaller --onefile → spren-sidecar      │
                              (per platform: macos arm64/x86_64,         │
                               linux x86_64/arm64, windows x86_64)       │
                                        │                                │
                                        └────────────────┬───────────────┘
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

Coordinated by `Justfile` recipes:
- `just build-frontend` — pnpm build + copy to `packages/spren/_webui/`
- `just build-sidecar` — PyInstaller for the host platform (CI matrix runs all targets)
- `just build-desktop` — Tauri build using the prepared sidecar + webui
- `just build-pipx` — uv build of the Spren wheel without the Tauri shell (sidecar mode only)
- `just build-docker` — Docker image FROM `python:3.13-slim` plus the wheel
- `just publish-<channel>` — sign and publish to the named channel

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

The image is built FROM `python:3.13-slim` and installs the Spren wheel. One uvicorn process per container; no multi-process supervisor. The Tauri shell does not run in Docker (no display); the user accesses the bundled web UI via browser to `http://127.0.0.1:8765`.

**Image registry:** `marsys/spren` on Docker Hub / GHCR. Tags follow Spren's version (`marsys/spren:0.3.0`, `marsys/spren:latest`).
