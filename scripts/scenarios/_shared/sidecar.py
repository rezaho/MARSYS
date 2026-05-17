"""Boot a real spren sidecar for scenario testing.

Mirrors the lifecycle pattern in apps/web/tests/e2e/helpers/sidecar.ts so
Python scenarios talk to the same surface the Playwright suite hits. The
sidecar binds to a random port (``--port 0``) so concurrent runs don't
collide. The ready line carries the per-launch token, which the caller
needs to authenticate every request.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

READY_RE = re.compile(r"spren-ready: port=(\d+) token=(\S+) data-dir=(.+)")
READY_TIMEOUT_S = 15.0
UVICORN_BIND_GRACE_S = 1.0


@dataclass(frozen=True)
class SidecarHandle:
    process: subprocess.Popen[bytes]
    port: int
    token: str
    data_dir: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "packages").is_dir():
            return parent
    raise RuntimeError("could not locate repo root from " + str(here))


def start_sidecar(*, env_overrides: dict[str, str] | None = None) -> SidecarHandle:
    """Spawn a sidecar on a random port and return the handle once ready."""
    cwd = _repo_root()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(
        ["uv", "run", "--package", "spren", "python", "-m", "spren", "--port", "0"],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + READY_TIMEOUT_S
    buf = b""
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        chunk = proc.stdout.readline()
        if not chunk:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"sidecar exited before ready (exit={proc.returncode}); buf={buf!r}"
                )
            continue
        buf += chunk
        match = READY_RE.search(chunk.decode("utf-8", errors="replace"))
        if match:
            port = int(match.group(1))
            token = match.group(2)
            data_dir = Path(match.group(3).strip())
            time.sleep(UVICORN_BIND_GRACE_S)
            return SidecarHandle(process=proc, port=port, token=token, data_dir=data_dir)
    raise RuntimeError(f"sidecar ready timeout after {READY_TIMEOUT_S}s; buf={buf!r}")


def stop_sidecar(handle: SidecarHandle) -> None:
    handle.process.terminate()
    try:
        handle.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        handle.process.wait(timeout=2)


@contextmanager
def sidecar(*, env_overrides: dict[str, str] | None = None) -> Iterator[SidecarHandle]:
    handle = start_sidecar(env_overrides=env_overrides)
    try:
        yield handle
    finally:
        stop_sidecar(handle)


if __name__ == "__main__":
    with sidecar() as h:
        print(f"sidecar up: port={h.port} token={h.token[:8]}... data_dir={h.data_dir}")
        sys.stdout.flush()
        time.sleep(2)
