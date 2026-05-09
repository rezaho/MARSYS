# 07 — Security

Spren is a local-first single-user app distributed as a native installer. The security model has four surfaces: **installer / binary trust** (code signing + notarization + signed install scripts), **localhost API exposure** (per-launch token + 127.0.0.1 binding), **API key storage** (OS keychain + encrypted-SQLite fallback), and **external messenger channels** (allowlists + confirm-before-acting in later releases).

## Threat model in scope

| Threat | Mitigation |
|--------|------------|
| Compromised installer / supply-chain swap of the binary | Code-signed + notarized native installers per platform; install script verifies a GPG signature on the downloaded binary against a published release manifest |
| Auto-updater hijacked to push a malicious update | Tauri auto-updater verifies signature on every update before swap; signing keys live in CI secrets only |
| Other local processes calling our API and exfiltrating data | Per-launch token (SP-002) + 127.0.0.1 binding |
| Browser-based CSRF against our localhost API | Token in `Authorization` header, NOT in cookies → CSRF inapplicable. Strict CORS origin lock |
| API keys leaked from disk or logs | OS keychain primary; encrypted SQLite fallback; never logged or written to traces |
| Spren server reachable from network | Default bind 127.0.0.1; user must explicitly set `--bind` to expose |
| Messenger bot misuse — anyone DMs the bot, gets meta-agent privileges | Per-channel allowlist; confirm-before-acting on write actions (later release) |
| Workflow with shell tool runs malicious command | Document the risk; per-workflow tool capability scoping (later release) |
| Trace files contain user secrets that leak to disk backups | Document; optional secret redaction filter (later release) |

## Threat model OUT of scope

- Multi-user separation: single-user product. If someone has access to the user's machine, they have the user's data. (If multi-user is needed → MARSYS Studio.)
- Network adversaries / TLS: localhost-only.
- Supply-chain attacks on dependencies: we follow SemVer pinning + Renovate-bot updates, but we don't run our own audit.
- Sandboxing of agent code execution beyond what marsys framework provides.

## Localhost API exposure

The HTTP server binds to `127.0.0.1:<port>` by default. This means:

- Other applications running as the same OS user CAN connect to the port (it's localhost — not a separate network namespace)
- They cannot pass auth without the per-launch token

Per-launch token mechanism:
- Generated at server start (32-byte URL-safe random via `secrets.token_urlsafe(32)`)
- Printed to stdout
- Passed to the browser via URL fragment (`http://127.0.0.1:8765/#token=...`); fragment never sent to the server, never logged in browser history beyond the local URL bar
- Frontend reads `window.location.hash`, stores in memory (NOT localStorage — leaks survive process death), strips from URL
- Every request: `Authorization: Bearer <token>`
- Server validates with constant-time comparison

In Tauri-managed launches, the token is injected into the webview as a window-level variable before bundle scripts run; URL fragments are not used. In browser-tab mode (pipx, Docker, `spren launch --browser`), the URL fragment carries the token; if the user opens the URL in a different browser without the fragment, they get a 401 page with instructions to copy the URL from the original `spren launch` output. There is no login.

Token regeneration:
- New token every daemon restart
- API: `POST /v1/auth/rotate` (auth required) regenerates and returns new token; old one invalidated immediately
- UI: button in settings to rotate

## API key storage

Two-tier storage:

### Tier 1: OS keychain (preferred)

Use `keyring` Python library. Maps to:
- macOS Keychain
- Windows Credential Manager
- Linux Secret Service (gnome-keyring / kwallet via libsecret)

Advantages:
- Per-OS-user isolation
- Encrypted at rest by the OS
- Survives reboots, accessible only to authenticated user session

Limitations:
- Requires a desktop session (fails on headless Linux servers)
- macOS prompts the user the first time the spren process accesses the keychain — UX needs to explain this

### Tier 2: Encrypted SQLite (fallback)

When OS keychain is unavailable (Docker, headless server, CI):
- AES-256-GCM encryption
- Master key derived via Argon2id from a user-set passphrase (set on first run, prompted in CLI; cached in memory only for the running session)
- Salt stored in SQLite alongside ciphertext
- On `spren launch`: prompt for passphrase if encrypted-secrets exist; refuse to start without

Implementation:
- `packages/spren/src/spren/storage/secrets.py`
- API: `set_secret(key_name, value)`, `get_secret(key_name)`, `delete_secret(key_name)`, `list_secret_keys()` — never `list_all_secrets`

### Hard rules for handling secrets

- **NEVER log a secret value.** Logging filters at the Python logging layer: any kwarg matching `*_key|*_token|*_secret|password` is replaced with `[REDACTED]`.
- **NEVER include a secret in a trace event.** Trace writer applies the same filter on attrs.
- **NEVER serialize a secret in an API response or error message.**
- API `GET /v1/secrets` returns key NAMES only, never values.
- The frontend NEVER stores secrets — it reads them from server (when needed for display in form fields, the server re-serves and they live in form state only).

## Installer trust and code signing

Native distribution requires a verifiable trust chain.

**Per platform:**
- **macOS:** Apple Developer Program ($99/yr); installers are `codesign`-ed and notarized via `notarytool`. Without notarization, Gatekeeper warns on first launch. Notarization runs async in CI (minutes-to-hour); release flagged "in notarization" until done.
- **Windows:** Authenticode signing (EV preferred for SmartScreen reputation). `signtool.exe` invoked in CI. Without signing, SmartScreen warns on first launch.
- **Linux:** No platform-mandated signing for AppImage / .deb. The install script payload is GPG-signed; the install script verifies the signature before executing the binary. Distros that ship Spren via apt / AUR follow distro signing conventions.

**Auto-updater:** Tauri 2's built-in updater checks GitHub Releases (or our manifest server) for new versions, downloads, verifies signature, and atomically swaps. The signature key is hardcoded into the Rust shell at compile time; an attacker would need both a fresh signed binary AND a way to push it to the manifest server, which lives behind separate credentials.

**Signing keys:** All signing keys live exclusively in CI secrets. No developer workstation has them. CI release jobs are gated on protected branches.

**Install script:** The `install.sh` curl-pipe-sh script is served over HTTPS-only (HSTS preloaded for `spren.dev`); it verifies a GPG signature on the downloaded binary against the public key embedded in the script before executing. Users who don't trust curl-pipe-sh download the binary directly + verify manually with documented `gpg --verify` commands.

## Messenger channel security (later release)

When the user connects an external channel (Telegram bot, Discord gateway, etc.):

1. Channel record stores: bot token (encrypted), channel/server IDs, and an **allowlist** of platform user IDs allowed to interact
2. Inbound messages from non-allowlisted users are silently ignored (no error response — avoids leaking that the bot is connected)
3. Inbound messages that map to write actions (trigger run, modify workflow) require an explicit user confirmation step (channel-side: bot replies with action description + "yes" to confirm)
4. Per-channel scoping: the meta-agent has a reduced toolset when invoked via a channel — write tools require confirmation; read tools work directly
5. Rate limit: per-channel-user max-1-action-per-2-seconds

## Workflow security

A workflow can include agents with shell-execute tools. A workflow author who is also the spren user is trusted with the same shell access they have outside Spren, so this is not a privilege-escalation issue.

A later release adds:
- Per-workflow tool capability scoping (e.g., "this workflow may not use shell")
- Confirmation prompt on first run if the workflow uses dangerous tools
- Audit log of shell commands executed via workflows (in `<data-dir>/data/audit.log`)

For now: documented user-facing risk; warning badge on workflows that use shell/file-write tools; workflows imported from Python files surface the same warning when they bring shell-tool agents.

## CORS

FastAPI middleware:

```python
allow_origins=["http://127.0.0.1:<port>", "http://localhost:<port>"]
allow_methods=["GET","POST","PUT","PATCH","DELETE"]
allow_headers=["Authorization","Content-Type","Idempotency-Key"]
allow_credentials=False
```

In Tauri-managed launches, origin is `tauri://localhost` — added to `allow_origins` by the desktop launcher at sidecar startup.

Strict origin lock prevents browser-based prompt-injection that tricks the user into making API calls from a malicious site.

## Updates that affect security

When the wheel is upgraded and contains migration scripts that touch secrets storage (e.g., re-keying), the migration:
1. Refuses to run unless the user provides the passphrase (for encrypted SQLite path)
2. Re-encrypts in a transaction
3. Logs only "secrets migrated", never values

## Audit / what we log

- Server start/stop with timestamp
- Auth failures (count + endpoint, no token contents)
- Run start/finish with workflow_id and status
- Settings changes (key name only, not value)
- Secrets changes (key name only)
- Channel connect/disconnect
- Migrations run

Log file: `<data-dir>/logs/spren.log` with rotation at 10 MB, 5 backups kept.
