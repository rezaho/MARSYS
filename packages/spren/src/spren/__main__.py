"""CLI entry point: `python -m spren` or `spren` (when installed).

Boots the FastAPI sidecar, prints the ready signal on stdout for the Tauri shell
to parse, and runs uvicorn until shutdown.

Ready signal format (single line, flushed):
    spren-ready: port=<N> token=<T> data-dir=<path>

Shutdown protocol: a stdin-reader thread consumes lines from sys.stdin and
triggers uvicorn's graceful shutdown when it sees `shutdown\\n`. EOF on stdin
also triggers graceful shutdown — the sidecar should never outlive its parent.
"""
from __future__ import annotations

import os
import socket
import stat as stat_module
import sys
import threading
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


def _stdin_is_pipe() -> bool:
    """True iff stdin is a pipe from a managing parent (vs tty / null / file).

    The Tauri shell spawns the sidecar with `Stdio::piped()` for stdin so it
    can write `shutdown\\n` on window-close; that surfaces as a FIFO on the
    child's fd 0. CI tools that pass `stdio=ignore` (or DEVNULL) hand the
    child a character device pointing at /dev/null, which would otherwise
    look like immediate EOF and shut the sidecar down before uvicorn starts.
    """
    try:
        mode = os.fstat(0).st_mode
    except OSError:
        return False
    return stat_module.S_ISFIFO(mode)


def _watch_stdin_for_shutdown(server: uvicorn.Server) -> None:
    """Trigger graceful shutdown when stdin emits `shutdown\\n` or EOF."""
    try:
        for line in sys.stdin:
            if line.strip() == "shutdown":
                break
    finally:
        # Either the shutdown line arrived, stdin hit EOF, or sys.stdin
        # raised. In all three cases we want uvicorn to exit cleanly.
        server.should_exit = True


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
    config = uvicorn.Config(app, host="127.0.0.1", port=resolved_port, log_level="warning")
    server = uvicorn.Server(config)

    if _stdin_is_pipe():
        threading.Thread(
            target=_watch_stdin_for_shutdown, args=(server,), daemon=True
        ).start()

    server.run()


if __name__ == "__main__":
    main()
