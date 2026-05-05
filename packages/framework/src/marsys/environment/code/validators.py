"""
Validators for code execution.

This module provides validation functions for:
- Environment variable validation
- Shell command validation
- Python code validation (import checking)
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Set, Tuple

from .config import CodeExecutionConfig


# Regex pattern to extract Python imports
# Matches: import foo, from foo import bar, import foo.bar
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w\.]*)", re.MULTILINE)


def build_base_env(config: CodeExecutionConfig) -> Dict[str, str]:
    """
    Build safe environment variables for subprocess.

    Only includes variables from the allowed list, plus virtual environment
    settings if configured.

    Args:
        config: Code execution configuration

    Returns:
        Dictionary of safe environment variables
    """
    env = {k: v for k, v in os.environ.items() if k in config.allowed_env_vars}

    # Add virtual environment settings if configured
    if config.venv_path:
        env["VIRTUAL_ENV"] = str(config.venv_path)
        env["PATH"] = f"{config.venv_path}/bin:{env.get('PATH', '')}"

    return env


def validate_env(
    env: Optional[Dict[str, str]],
    config: CodeExecutionConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate user-provided environment variables.

    Args:
        env: User-provided environment variables
        config: Code execution configuration

    Returns:
        Tuple of (is_valid, error_message)
    """
    if env is None:
        return True, None

    if not isinstance(env, dict):
        return False, "env must be a dict of string -> string"

    # Check all provided keys are in allowed list
    disallowed = [k for k in env.keys() if k not in config.allowed_env_vars]
    if disallowed:
        return False, f"env var(s) not allowed: {', '.join(disallowed)}"

    return True, None


def validate_shell_command(
    command: str,
    config: CodeExecutionConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate shell command against security policies.

    Checks:
    1. Command is in allowed list (if whitelist configured)
    2. Command doesn't contain blocked patterns
    3. Network commands are blocked if allow_network=False

    Args:
        command: Shell command to validate
        config: Code execution configuration

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Empty command"

    # Check allowed commands whitelist
    if config.allowed_shell_commands:
        cmd_start = command.strip().split()[0] if command.strip() else ""
        if cmd_start not in config.allowed_shell_commands:
            return False, f"Command '{cmd_start}' not in allowed commands list"

    # Check blocked patterns
    for blocked in config.blocked_shell_patterns:
        if blocked in command:
            return False, f"Command contains blocked pattern: '{blocked}'"

    # Check network commands if network disabled
    if not config.allow_network:
        network_markers = ["curl ", "wget ", "nc ", "ssh "]
        for marker in network_markers:
            if marker in command:
                return False, f"Network command blocked: '{marker.strip()}'"

    return True, None


def _extract_imports(code: str) -> Set[str]:
    """
    Extract root module names from Python code.

    Handles:
    - import foo
    - import foo.bar
    - from foo import bar
    - from foo.bar import baz

    Args:
        code: Python source code

    Returns:
        Set of root module names (e.g., 'foo' from 'foo.bar')
    """
    modules = set()
    for match in IMPORT_RE.finditer(code):
        full_module = match.group(1)
        # Extract root module (first part before .)
        root = full_module.split(".")[0]
        modules.add(root)
    return modules


def validate_python_code(
    code: str,
    config: CodeExecutionConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate Python code against security policies.

    Checks:
    1. Imports are not in blocked modules list
    2. Network-related modules are blocked if allow_network=False
    3. All imports are in allowed modules list (if whitelist configured)

    Note: This is a best-effort static check. Runtime imports (importlib,
    __import__) are not detected.

    Args:
        code: Python source code to validate
        config: Code execution configuration

    Returns:
        Tuple of (is_valid, error_message)
    """
    imports = _extract_imports(code)

    # Build effective blocked modules list
    blocked = set(config.blocked_python_modules)

    # Add network modules if network disabled
    if not config.allow_network:
        blocked.update({"socket", "urllib", "http", "requests", "aiohttp"})

    # Check for blocked imports
    for module in imports:
        if module in blocked:
            return False, f"Import blocked: {module}"

    # Check allowed modules whitelist
    if config.allowed_python_modules:
        for module in imports:
            if module not in config.allowed_python_modules:
                return False, f"Import not allowed: {module}"

    return True, None
