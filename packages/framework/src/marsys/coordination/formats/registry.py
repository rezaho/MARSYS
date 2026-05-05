"""
Registry for response format handlers.

This module provides a centralized registry for managing response formats,
allowing runtime selection and configuration of the output format.
"""

from typing import Dict, List, Optional, Type

from .base import BaseResponseFormat


# Global registry of format handlers
_FORMAT_REGISTRY: Dict[str, Type[BaseResponseFormat]] = {}

# Default format
_DEFAULT_FORMAT = "json"


def register_format(name: str, format_class: Type[BaseResponseFormat]) -> None:
    """
    Register a response format handler.

    Args:
        name: Format name (e.g., 'json', 'xml')
        format_class: Format handler class (not instance)

    Example:
        register_format("json", JSONResponseFormat)
    """
    _FORMAT_REGISTRY[name] = format_class


def get_format(name: Optional[str] = None) -> BaseResponseFormat:
    """
    Get a response format handler instance.

    Args:
        name: Format name, or None for default

    Returns:
        Format handler instance

    Raises:
        ValueError: If format is not registered
    """
    format_name = name or _DEFAULT_FORMAT

    if format_name not in _FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown response format: '{format_name}'. "
            f"Available formats: {list(_FORMAT_REGISTRY.keys())}"
        )

    return _FORMAT_REGISTRY[format_name]()


def set_default_format(name: str) -> None:
    """
    Set the default response format.

    Args:
        name: Format name to set as default

    Raises:
        ValueError: If format is not registered
    """
    global _DEFAULT_FORMAT
    if name not in _FORMAT_REGISTRY:
        raise ValueError(f"Cannot set default to unknown format: '{name}'")
    _DEFAULT_FORMAT = name


def list_formats() -> List[str]:
    """
    List all registered format names.

    Returns:
        List of registered format names
    """
    return list(_FORMAT_REGISTRY.keys())


def is_format_registered(name: str) -> bool:
    """
    Check if a format is registered.

    Args:
        name: Format name to check

    Returns:
        True if format is registered
    """
    return name in _FORMAT_REGISTRY


# Register built-in formats
# Import here to avoid circular dependency issues
from .json_format import JSONResponseFormat

register_format("json", JSONResponseFormat)
