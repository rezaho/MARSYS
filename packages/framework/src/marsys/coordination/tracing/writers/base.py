"""
Abstract base for trace writers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TraceTree


class TraceWriter(ABC):
    """
    Base class for trace output backends.

    Implementations persist or export a completed TraceTree.
    """

    @abstractmethod
    async def write(self, trace: 'TraceTree') -> None:
        """Write a completed trace."""

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this writer."""
