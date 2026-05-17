"""FastAPI app for the Spren sidecar (v0.3 foundation + Sessions 02 + 03).

Endpoints:
- GET /healthz — liveness check, no auth.
- GET /v1/bootstrap — capability + version snapshot, auth-gated.
- /v1/workflows/* — workflow CRUD + Python-file import + lint (auth-gated).
- /v1/tools — framework tool registry surface (auth-gated, Session 03).

CORS is locked to localhost origins (current sidecar port + Vite dev port) plus
`tauri://localhost` for the Tauri webview origin (per docs/architecture/spren/07-security.md).
The static Vite bundle is served from `<package>/_webui/` if present (production);
in dev, that dir is empty and the Vite dev server on 5173 serves the UI instead.

Session 03 also schedules the empty-draft sweeper as a background task on the
lifespan handler (4-hour cadence; deletes empty visual-builder rows older than
24 hours).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__ as spren_version
from .auth import make_auth_dependency
from .models import ErrorEnvelope, ErrorPayload
from .cost import warn_if_rates_stale
from .routes.files import make_files_router
from .routes.lint import make_lint_router
from .routes.runs import make_runs_router
from .routes.tools import make_tools_router
from .routes.workflows import make_workflows_router
from .runs.broker import RunsBroker
from .runs.lifecycle import shutdown_all_active
from .storage import Database, MigrationsRunner
from .storage.idempotency import sweep_expired
from .workers import run_draft_sweeper_forever

_WEBUI_DIR = Path(__file__).parent / "_webui"


class FrameworkInfo(BaseModel):
    version: str


class SprenInfo(BaseModel):
    active: bool
    version: str


class BootstrapResponse(BaseModel):
    framework: FrameworkInfo
    spren: SprenInfo
    surfaces: list[str]
    capabilities: dict[str, bool]
    endpoints: dict[str, str]
    started_at: datetime
    data_dir: str


def _framework_version() -> str:
    try:
        return version("marsys")
    except PackageNotFoundError:
        return "unknown"


def create_app(
    token: str,
    port: int,
    data_dir: Path,
    *,
    started_at: datetime | None = None,
    enable_draft_sweeper: bool = True,
) -> FastAPI:
    del port  # kept in signature for symmetry; CORS uses a regex over all localhost ports.

    db = Database(data_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runner = MigrationsRunner(db.connection)
        runner.run()
        sweep_expired(db.connection)
        warn_if_rates_stale()
        sweeper_task: asyncio.Task | None = None
        if enable_draft_sweeper:
            sweeper_task = asyncio.create_task(
                run_draft_sweeper_forever(lambda: db.connection)
            )
        try:
            yield
        finally:
            # Drain in-flight runs before shutdown (AC-77).
            try:
                await shutdown_all_active(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            if sweeper_task is not None:
                sweeper_task.cancel()
                try:
                    await sweeper_task
                except asyncio.CancelledError:
                    pass
            db.close()

    app = FastAPI(title="Spren", version=spren_version, lifespan=lifespan)
    require_auth = make_auth_dependency(token)
    boot_time = started_at or datetime.now(timezone.utc)
    runs_broker = RunsBroker()

    # CORS: localhost on any port (handles Vite fallback ports + random sidecar
    # ports) plus tauri://localhost (per docs/architecture/spren/07-security.md).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(http://(127\.0\.0\.1|localhost)(:\d+)?|tauri://localhost)$",
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
        allow_credentials=False,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/v1/bootstrap",
        response_model=BootstrapResponse,
        dependencies=[Depends(require_auth)],
    )
    def bootstrap() -> BootstrapResponse:
        return BootstrapResponse(
            framework=FrameworkInfo(version=_framework_version()),
            spren=SprenInfo(active=False, version=spren_version),
            surfaces=["gui"],
            capabilities={},
            endpoints={
                "workflows": "/v1/workflows",
                "tools": "/v1/tools",
                "lint": "/v1/workflows/{id}/lint",
                "runs": "/v1/runs",
                "files": "/v1/files",
            },
            started_at=boot_time,
            data_dir=str(data_dir),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        del request
        envelope = ErrorEnvelope(
            error=ErrorPayload(
                code="VALIDATION_FAILED",
                message="request payload failed validation",
                details={"errors": _flatten_errors(exc.errors())},
            )
        )
        return JSONResponse(status_code=422, content=envelope.model_dump(mode="json"))

    # Resolve the framework tool registry once at app construction. The lint
    # endpoint reads from the same source via ``known_tools_provider``.
    from marsys.environment.tools import AVAILABLE_TOOLS  # type: ignore[import-not-found]

    tool_names: list[str] = sorted(AVAILABLE_TOOLS.keys())

    app.include_router(
        make_workflows_router(require_auth, db_factory=lambda: db.connection),
    )
    app.include_router(
        make_tools_router(require_auth),
    )
    app.include_router(
        make_lint_router(
            require_auth,
            db_factory=lambda: db.connection,
            known_tools_provider=lambda: tool_names,
        ),
    )
    app.include_router(
        make_runs_router(
            require_auth,
            db_factory=lambda: db.connection,
            broker=runs_broker,
            data_dir=data_dir,
        ),
    )
    app.include_router(
        make_files_router(
            require_auth,
            db_factory=lambda: db.connection,
            data_dir=data_dir,
        ),
    )

    if (_WEBUI_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=_WEBUI_DIR, html=True), name="webui")

    return app


def _flatten_errors(errors: list[dict]) -> list[dict]:
    """Strip non-JSON-serializable ``ctx`` values from Pydantic validation errors."""
    flat: list[dict] = []
    for err in errors:
        clean: dict = {}
        for key, value in err.items():
            if key == "ctx" and isinstance(value, dict):
                clean["ctx"] = {k: str(v) for k, v in value.items()}
            else:
                clean[key] = value
        flat.append(clean)
    return flat
