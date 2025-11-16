"""
Abstract base classes for file operations toolkit.

This module defines the base interfaces that all file handlers must implement,
providing a consistent API across different file types.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

from .data_models import FileContent, DocumentStructure, Section


def check_character_limit(
    content: str,
    max_chars: int,
    request_description: str,
    solution_hint: str
) -> None:
    """
    Check if content exceeds the absolute character limit.

    Args:
        content: The content to check
        max_chars: Maximum allowed characters
        request_description: Description of what was requested (e.g., "read entire file", "lines 1-100", "pages 5-10")
        solution_hint: Suggestion for how to reduce content (e.g., "Use start_line/end_line", "Request fewer pages")

    Raises:
        ValueError: If content exceeds max_chars
    """
    char_count = len(content)
    if char_count > max_chars:
        raise ValueError(
            f"Content exceeds absolute maximum character limit.\n"
            f"📊 Your request: {request_description}\n"
            f"📏 Result size: {char_count:,} characters\n"
            f"⚠️  Maximum allowed: {max_chars:,} characters\n"
            f"💡 Solution: {solution_hint}"
        )


class FileHandler(ABC):
    """
    Abstract base class for file type handlers.

    Each file type (Python, PDF, Markdown, etc.) should implement this interface
    to provide type-specific functionality while maintaining a consistent API.
    """

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """
        Check if this handler can process the given file.

        Args:
            path: File path to check

        Returns:
            True if this handler can process the file, False otherwise
        """
        pass

    @abstractmethod
    async def read(self, path: Path, **kwargs) -> FileContent:
        """
        Read file content with type-specific logic.

        Args:
            path: File path to read
            **kwargs: Additional arguments specific to file type

        Returns:
            FileContent object with file contents and metadata
        """
        pass

    @abstractmethod
    async def get_structure(self, path: Path, **kwargs) -> DocumentStructure:
        """
        Extract hierarchical structure from the file.

        Args:
            path: File path to analyze
            **kwargs: Additional arguments specific to file type

        Returns:
            DocumentStructure representing the file's hierarchy
        """
        pass

    @abstractmethod
    async def read_section(self, path: Path, section_id: str, **kwargs) -> str:
        """
        Read a specific section of the file.

        Args:
            path: File path
            section_id: Section identifier (e.g., "function:process_payment")
            **kwargs: Additional arguments

        Returns:
            Content of the specified section
        """
        pass

    async def read_partial(
        self,
        path: Path,
        start_line: int,
        end_line: int,
        **kwargs
    ) -> FileContent:
        """
        Read a specific line range from the file.

        Default implementation reads the whole file and slices.
        Subclasses can override for more efficient implementations.

        Args:
            path: File path
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive)
            **kwargs: Additional arguments

        Returns:
            FileContent with the specified line range
        """
        # Default implementation: read and slice
        content = await self.read(path, **kwargs)

        if not content.content:
            return content

        lines = content.content.split('\n')
        # Adjust for 1-indexed lines
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        partial_content = '\n'.join(lines[start_idx:end_idx])

        # Safety check: absolute character limit on the content being returned
        max_chars = kwargs.get('max_characters_absolute', 120_000)
        check_character_limit(
            content=partial_content,
            max_chars=max_chars,
            request_description=f"lines {start_line}-{end_line}",
            solution_hint="Request fewer lines (e.g., reduce the line range)."
        )

        # Mark as explicit range request so usage guide is not added
        metadata = content.metadata.copy() if content.metadata else {}
        metadata['explicit_range_request'] = True

        return FileContent(
            path=path,
            content=partial_content,
            partial=True,
            encoding=content.encoding,
            line_range=(start_line, end_line),
            total_lines=len(lines),
            file_size=content.file_size,
            character_count=len(partial_content),
            estimated_tokens=len(partial_content) // 4,  # Rough estimate: 1 token ≈ 4 chars
            metadata=metadata
        )

    def get_file_extension(self, path: Path) -> str:
        """
        Get file extension (lowercase, without dot).

        Args:
            path: File path

        Returns:
            File extension
        """
        return path.suffix.lower().lstrip('.')

    def get_file_type(self, path: Path) -> str:
        """
        Get file type identifier for this handler.

        Default implementation uses class name minus 'Handler'.
        Subclasses can override.

        Args:
            path: File path

        Returns:
            File type identifier (e.g., "python", "pdf")
        """
        handler_name = self.__class__.__name__
        if handler_name.endswith('Handler'):
            return handler_name[:-7].lower()
        return handler_name.lower()


class ContentExtractor(ABC):
    """
    Abstract base class for content extractors.

    Content extractors parse and extract structured information from files.
    """

    @abstractmethod
    async def extract_structure(self, path: Path, **kwargs) -> DocumentStructure:
        """
        Extract structured content from a file.

        Args:
            path: File path
            **kwargs: Additional arguments

        Returns:
            DocumentStructure with extracted information
        """
        pass

    @abstractmethod
    def extract_metadata(self, path: Path, **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            path: File path
            **kwargs: Additional arguments

        Returns:
            Dictionary of metadata
        """
        pass


class SearchableHandler(ABC):
    """
    Interface for handlers that support structure-based searching.

    Handlers can optionally implement this to enable searching within
    their specific structures (e.g., searching for function names in code).
    """

    @abstractmethod
    async def search_structure(
        self,
        path: Path,
        query: str,
        **kwargs
    ) -> list:
        """
        Search within the file's structure.

        Args:
            path: File path
            query: Search query (may be regex or plain text)
            **kwargs: Additional search parameters

        Returns:
            List of matching sections or elements
        """
        pass


class EditableHandler(ABC):
    """
    Interface for handlers that support advanced editing operations.

    Handlers can optionally implement this to provide specialized editing
    capabilities beyond simple text replacement.
    """

    @abstractmethod
    async def apply_structured_edit(
        self,
        path: Path,
        edit_spec: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Apply a structured edit operation.

        Args:
            path: File path
            edit_spec: Structured edit specification
            **kwargs: Additional parameters

        Returns:
            True if edit succeeded, False otherwise
        """
        pass
