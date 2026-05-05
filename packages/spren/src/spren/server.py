"""FastAPI app for the Spren sidecar (v0.3 foundation).

Endpoints:
- GET /healthz — liveness check, no auth.
- GET /v1/bootstrap — capability + version snapshot, auth-gated.

CORS is locked to localhost origins (current sidecar port + Vite dev port) plus
`tauri://localhost` for the Tauri webview origin (per docs/architecture/spren/07-security.md).
The static Vite bundle is served from `<package>/_webui/` if present (production);
in dev, that dir is empty and the Vite dev server on 5173 serves the UI instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__ as spren_version
from .auth import make_auth_dependency

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
) -> FastAPI:
    del port  # kept in signature for symmetry; CORS uses a regex over all localhost ports.
    app = FastAPI(title="Spren", version=spren_version)
    require_auth = make_auth_dependency(token)
    boot_time = started_at or datetime.now(timezone.utc)

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
            endpoints={},
            started_at=boot_time,
            data_dir=str(data_dir),
        )

    if (_WEBUI_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=_WEBUI_DIR, html=True), name="webui")

    return app
