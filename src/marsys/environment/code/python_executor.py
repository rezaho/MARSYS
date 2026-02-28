"""
Python executor with persistent session support and image capture.

This module provides:
- Subprocess-based Python execution
- Persistent session for stateful analysis (maintains variables between calls)
- Display hooks for capturing images (matplotlib, PIL, numpy arrays)
- Resource limits (Linux only)
- Output truncation
"""

import asyncio
import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import CodeExecutionConfig
from .data_models import ExecutionResult
from .validators import (
    build_base_env,
    validate_env,
    validate_python_code,
)

logger = logging.getLogger(__name__)


# Python kernel code for persistent sessions
# This runs as a long-lived subprocess that receives code via stdin
# and sends results via stdout using a simple length-prefixed protocol
#
# Includes display() and display_image() hooks for capturing images
KERNEL_CODE = r'''
import sys, json, traceback, io, contextlib, os, time

ORIG_STDIN = sys.stdin
OUTPUT_DIR = None
IMAGES = []

def _set_output_dir(path):
    global OUTPUT_DIR
    OUTPUT_DIR = path
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def _next_name(ext):
    return f"image_{int(time.time()*1000)}_{len(IMAGES)}.{ext}"

def _save_bytes(data, ext):
    if OUTPUT_DIR is None:
        return None
    name = _next_name(ext)
    path = os.path.join(OUTPUT_DIR, name)
    if isinstance(data, str):
        data = data.encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    IMAGES.append(path)
    return path

def _save_fig(fig):
    if OUTPUT_DIR is None:
        return None
    name = _next_name("png")
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    IMAGES.append(path)
    return path

def display(obj=None):
    """
    Display hook for capturing images.

    Usage:
    - display() - captures all open matplotlib figures
    - display(fig) - captures a specific matplotlib figure
    - display(pil_image) - captures a PIL Image
    - display(numpy_array) - captures a numpy array as image
    - display(obj) - captures objects with _repr_png_/_repr_jpeg_ methods
    """
    if obj is None:
        # Capture all open matplotlib figures
        try:
            import matplotlib.pyplot as plt
            figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                _save_fig(fig)
            if figs:
                plt.close("all")
        except Exception:
            pass
        return None

    # Handle matplotlib figure
    if hasattr(obj, "savefig"):
        try:
            return _save_fig(obj)
        except Exception:
            pass

    # Handle IPython-style repr methods
    for attr, ext in [("_repr_png_", "png"), ("_repr_jpeg_", "jpg"), ("_repr_svg_", "svg")]:
        if hasattr(obj, attr):
            try:
                data = getattr(obj, attr)()
                if data:
                    return _save_bytes(data, ext)
            except Exception:
                pass

    # Handle PIL Image
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            name = _next_name("png")
            path = os.path.join(OUTPUT_DIR, name)
            obj.save(path)
            IMAGES.append(path)
            return path
    except Exception:
        pass

    # Handle numpy array (convert to PIL Image)
    try:
        import numpy as np
        from PIL import Image
        if isinstance(obj, np.ndarray):
            img = Image.fromarray(obj)
            name = _next_name("png")
            path = os.path.join(OUTPUT_DIR, name)
            img.save(path)
            IMAGES.append(path)
            return path
    except Exception:
        pass

    return None

def display_image(path):
    """
    Register an existing image file for return.

    Usage:
        plt.savefig("my_plot.png")
        display_image("my_plot.png")
    """
    if not path:
        return None
    try:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            IMAGES.append(abs_path)
            return abs_path
    except Exception:
        pass
    return None

GLOBALS = {"display": display, "display_image": display_image}

def _read_exactly(n):
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return b""
        data += chunk
    return data

while True:
    len_bytes = _read_exactly(8)
    if not len_bytes:
        break
    size = int.from_bytes(len_bytes, "big")
    payload = _read_exactly(size)
    if not payload:
        break

    try:
        req = json.loads(payload.decode("utf-8"))
    except Exception:
        continue

    code = req.get("code", "")
    working_dir = req.get("working_dir")
    env = req.get("env") or {}
    stdin_text = req.get("stdin")
    output_dir = req.get("output_dir")

    IMAGES = []
    _set_output_dir(output_dir)

    if working_dir:
        try:
            os.chdir(working_dir)
        except Exception:
            pass

    if env:
        os.environ.update(env)

    if stdin_text is not None:
        sys.stdin = io.StringIO(stdin_text)

    out_io, err_io = io.StringIO(), io.StringIO()
    success = True
    exit_code = 0
    start = time.time()

    try:
        with contextlib.redirect_stdout(out_io), contextlib.redirect_stderr(err_io):
            exec(code, GLOBALS)
    except Exception:
        success = False
        exit_code = 1
        err_io.write(traceback.format_exc())
    finally:
        sys.stdin = ORIG_STDIN

    result = {
        "success": success,
        "stdout": out_io.getvalue(),
        "stderr": err_io.getvalue(),
        "exit_code": exit_code,
        "duration_ms": int((time.time() - start) * 1000),
        "images": IMAGES,
    }

    data = json.dumps(result).encode("utf-8")
    sys.stdout.buffer.write(len(data).to_bytes(8, "big"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
'''


def _build_preexec_fn(config: CodeExecutionConfig):
    """
    Build preexec function for subprocess resource limits.

    Only applies limits on Linux where resource module is available.

    Args:
        config: Code execution configuration

    Returns:
        Function to call in child process before exec, or None if not supported
    """
    # Only apply resource limits on Linux
    if platform.system() != "Linux":
        return None

    def _apply_limits():
        try:
            import resource

            # Set CPU time limit
            if config.max_cpu_seconds:
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (config.max_cpu_seconds, config.max_cpu_seconds)
                )

            # Set memory limit (address space)
            if config.max_memory_mb:
                mem_bytes = config.max_memory_mb * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (mem_bytes, mem_bytes)
                )
        except (ImportError, ValueError, OSError):
            # resource module not available or limits not supported
            pass

    return _apply_limits


def _truncate(text: str, max_bytes: int) -> Tuple[str, bool]:
    """
    Truncate text to max_bytes with indicator.

    Args:
        text: Text to truncate
        max_bytes: Maximum bytes allowed

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if len(text) <= max_bytes:
        return text, False
    return text[:max_bytes] + f"\n... (truncated, limit: {max_bytes} bytes)", True


async def _read_exactly(stream: asyncio.StreamReader, n: int) -> bytes:
    """
    Read exactly n bytes from stream.

    Args:
        stream: Async stream reader
        n: Number of bytes to read

    Returns:
        Bytes read, or empty bytes if stream closed
    """
    data = b""
    while len(data) < n:
        chunk = await stream.read(n - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


class PythonSessionManager:
    """
    Manager for persistent Python sessions.

    Maintains a long-running Python subprocess that preserves state
    between executions. Uses a simple length-prefixed protocol for
    communication.
    """

    def __init__(self, config: CodeExecutionConfig):
        """
        Initialize session manager.

        Args:
            config: Code execution configuration
        """
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()

    async def _ensure_process(self) -> None:
        """
        Ensure the persistent Python process is running.

        Creates a new process if one doesn't exist or has terminated.
        """
        if self._process and self._process.returncode is None:
            return

        python_exec = self.config.resolve_python_executable()
        env = build_base_env(self.config)

        self._process = await asyncio.create_subprocess_exec(
            python_exec,
            "-u",
            "-c",
            KERNEL_CODE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(self.config.base_directory),
            env=env,
            preexec_fn=_build_preexec_fn(self.config),
        )

        logger.debug(f"Started persistent Python session (pid={self._process.pid})")

    async def execute(
        self,
        code: str,
        stdin: Optional[str],
        env: Optional[Dict[str, str]],
        timeout: Optional[int],
    ) -> ExecutionResult:
        """
        Execute code in the persistent session.

        Args:
            code: Python code to execute
            stdin: Standard input text
            env: Additional environment variables
            timeout: Timeout in seconds

        Returns:
            ExecutionResult with execution details (virtual paths)
        """
        async with self._lock:
            await self._ensure_process()

            # Resolve working directory via RunFileSystem
            fs = self.config.run_filesystem
            resolved = fs.resolve("/")  # Always use root as cwd
            host_cwd = resolved.host_path
            virtual_cwd = resolved.virtual_path

            # Validate environment
            env_ok, env_reason = validate_env(env, self.config)
            if not env_ok:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    timed_out=False,
                    duration_ms=0,
                    cwd=virtual_cwd,
                    truncated=False,
                    language="python",
                    error=env_reason,
                )

            # Build request payload with output_dir for images (host path)
            output_dir = str(self.config.resolve_output_dir())
            payload = json.dumps({
                "code": code,
                "stdin": stdin,
                "working_dir": str(host_cwd),
                "env": env or {},
                "output_dir": output_dir,
            }).encode("utf-8")

            # Send request
            assert self._process and self._process.stdin and self._process.stdout
            self._process.stdin.write(len(payload).to_bytes(8, "big"))
            self._process.stdin.write(payload)
            await self._process.stdin.drain()

            # Wait for response
            try:
                timeout_val = timeout or self.config.timeout_default
                length_bytes = await asyncio.wait_for(
                    _read_exactly(self._process.stdout, 8),
                    timeout=timeout_val,
                )
                if not length_bytes:
                    raise asyncio.TimeoutError()

                size = int.from_bytes(length_bytes, "big")
                body = await asyncio.wait_for(
                    _read_exactly(self._process.stdout, size),
                    timeout=timeout_val,
                )
                if not body:
                    raise asyncio.TimeoutError()

                result = json.loads(body.decode("utf-8"))

                # Truncate output
                stdout, trunc_out = _truncate(
                    result.get("stdout", ""),
                    self.config.max_output_bytes
                )
                stderr, trunc_err = _truncate(
                    result.get("stderr", ""),
                    self.config.max_output_bytes
                )

                # Limit images and convert to virtual paths
                host_images: List[str] = result.get("images") or []
                if self.config.max_images and len(host_images) > self.config.max_images:
                    host_images = host_images[:self.config.max_images]

                # Convert host paths to virtual paths for agent visibility
                virtual_images: List[str] = []
                for host_path in host_images:
                    try:
                        virtual_path = fs.to_virtual(Path(host_path))
                        virtual_images.append(virtual_path)
                    except ValueError:
                        # If conversion fails, use the output virtual dir + filename
                        filename = Path(host_path).name
                        virtual_images.append(f"{self.config.output_virtual_dir}/{filename}")

                return ExecutionResult(
                    success=result.get("success", False),
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result.get("exit_code", -1),
                    timed_out=False,
                    duration_ms=result.get("duration_ms", 0),
                    cwd=virtual_cwd,
                    truncated=trunc_out or trunc_err,
                    language="python",
                    images=virtual_images,
                    image_host_paths=host_images,  # Keep host paths for internal use
                )

            except asyncio.TimeoutError:
                # Kill the process on timeout
                await self.shutdown()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    timed_out=True,
                    duration_ms=0,
                    cwd=virtual_cwd,
                    truncated=False,
                    language="python",
                    error="Execution timed out",
                )

    async def shutdown(self) -> None:
        """
        Shutdown the persistent Python process.

        Kills the process if it's running and clears the reference.
        """
        if self._process and self._process.returncode is None:
            logger.debug(f"Shutting down persistent Python session (pid={self._process.pid})")
            self._process.kill()
            await self._process.wait()
        self._process = None


class PythonExecutor:
    """
    Python code executor with optional persistent session.

    Provides:
    - Subprocess-based Python execution
    - Optional persistent session for stateful analysis
    - Code validation against security policies
    - Resource limits (Linux only)
    - Output truncation
    - Image capture via display() and display_image() hooks
    """

    def __init__(self, config: CodeExecutionConfig):
        """
        Initialize Python executor.

        Args:
            config: Code execution configuration
        """
        self.config = config
        self._session_manager = PythonSessionManager(config)

    async def execute(
        self,
        code: str,
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute Python code.

        Uses persistent session if config.session_persistent_python=True,
        otherwise uses a temporary session for each execution.

        Args:
            code: Python code to execute
            stdin: Standard input text
            timeout: Timeout in seconds
            env: Additional environment variables

        Returns:
            ExecutionResult with execution details
        """
        # Validate code against policies
        allowed, reason = validate_python_code(code, self.config)
        if not allowed:
            # Get virtual cwd for error response
            fs = self.config.run_filesystem
            virtual_cwd = fs.resolve("/").virtual_path
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=False,
                duration_ms=0,
                cwd=virtual_cwd,
                truncated=False,
                language="python",
                error=reason,
            )

        # Use persistent session if configured
        if self.config.session_persistent_python:
            return await self._session_manager.execute(code, stdin, env, timeout)

        # Otherwise use a temporary session (still uses kernel for consistency)
        temp_session = PythonSessionManager(self.config)
        result = await temp_session.execute(code, stdin, env, timeout)
        await temp_session.shutdown()
        return result

    async def shutdown(self) -> None:
        """
        Shutdown the Python executor.

        Terminates any persistent session if running.
        """
        await self._session_manager.shutdown()
