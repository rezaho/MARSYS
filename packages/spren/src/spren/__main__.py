"""CLI entry point: `python -m spren` or `spren` (when installed).

Boots the FastAPI sidecar, prints the ready signal on stdout for the Tauri shell
to parse, and runs uvicorn until shutdown.

Ready signal format (single line, flushed):
    spren-ready: port=<N> token=<T> data-dir=<path>
"""
from __future__ import annotations

import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import platformdirs
import uvicorn

from . import __version__ as spren_version
from .auth import generate_token
from .server import create_app


def _resolve_port(requested: int) -> int:
    if requested == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", requested))
            return requested
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]


@click.command()
@click.option("--port", default=8765, show_default=True, help="Port to bind 127.0.0.1 to. 0 = random free.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=None, help="Override data dir.")
def main(port: int, data_dir: Path | None) -> None:
    resolved_port = _resolve_port(port)
    token = generate_token()
    data_path = data_dir or Path(platformdirs.user_data_dir("spren"))
    data_path.mkdir(parents=True, exist_ok=True)

    print(
        f"spren-ready: port={resolved_port} token={token} data-dir={data_path}",
        flush=True,
    )
    print(f"spren v{spren_version} starting on 127.0.0.1:{resolved_port}", file=sys.stderr, flush=True)

    app = create_app(
        token=token,
        port=resolved_port,
        data_dir=data_path,
        started_at=datetime.now(timezone.utc),
    )
    uvicorn.run(app, host="127.0.0.1", port=resolved_port, log_level="warning")


if __name__ == "__main__":
    main()
