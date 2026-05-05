"""
Intelligent file reading with multiple strategies.

This module implements smart file reading that automatically selects
the best strategy based on file size and type.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config import FileOperationConfig
from .data_models import FileContent, ReadStrategy, DocumentStructure
from .handlers.base import FileHandler

logger = logging.getLogger(__name__)


class IntelligentFileReader:
    """
    Smart file reader with multiple reading strategies.

    Automatically selects the best reading strategy based on file size,
    type, and configuration settings to optimize token usage.
    """

    def __init__(
        self,
        config: FileOperationConfig,
        handler_registry: Dict[str, FileHandler]
    ):
        """
        Initialize intelligent file reader.

        Args:
            config: File operation configuration
            handler_registry: Registry of file type handlers
        """
        self.config = config
        self.handler_registry = handler_registry

    async def read(
        self,
        path: Path,
        strategy: ReadStrategy = ReadStrategy.AUTO,
        **kwargs
    ) -> FileContent:
        """
        Read file with intelligent strategy selection.

        Args:
            path: File path to read
            strategy: Reading strategy (AUTO, FULL, PARTIAL, OVERVIEW, PROGRESSIVE)
            **kwargs: Additional arguments:
                - start_line: Starting line number (for PARTIAL)
                - end_line: Ending line number (for PARTIAL)
                - section: Section ID to read (for section-based reading)
                - max_depth: Maximum depth for structure (for OVERVIEW/PROGRESSIVE)

        Returns:
            FileContent object with file contents and metadata
        """
        # Get appropriate handler
        handler = self._get_handler(path)
        if not handler:
            raise ValueError(f"No handler found for file: {path}")

        # Auto-select strategy if needed
        if strategy == ReadStrategy.AUTO:
            strategy = await self._select_strategy_async(handler, path)
            logger.debug(f"Auto-selected strategy {strategy} for {path}")

        # Execute based on strategy
        if strategy == ReadStrategy.FULL:
            return await self._read_full(handler, path, **kwargs)

        elif strategy == ReadStrategy.PARTIAL:
            return await self._read_partial(handler, path, **kwargs)

        elif strategy == ReadStrategy.OVERVIEW:
            return await self._read_overview(handler, path, **kwargs)

        elif strategy == ReadStrategy.PROGRESSIVE:
            return await self._read_progressive(handler, path, **kwargs)

        else:
            raise ValueError(f"Unknown read strategy: {strategy}")

    async def _read_full(
        self,
        handler: FileHandler,
        path: Path,
        **kwargs
    ) -> FileContent:
        """Read entire file (no limits applied - full strategy)."""
        logger.debug(f"Reading full file: {path}")
        # Pass absolute character limit (safety guardrail)
        kwargs['max_characters_absolute'] = self.config.max_characters_absolute
        content = await handler.read(path, **kwargs)

        # For JSON files: Check character limit and provide overview if exceeded
        if path.suffix.lower() == '.json' and len(content.content) > self.config.max_json_content_chars:
            logger.info(f"JSON file {path} exceeds character limit, creating overview")
            try:
                import json
                original_char_count = len(content.content)
                data = json.loads(content.content)

                # Create smart overview with truncated values
                def truncate_value(value, max_chars=200):
                    """Truncate value to max_chars."""
                    value_str = json.dumps(value) if not isinstance(value, str) else value
                    if len(value_str) > max_chars:
                        return value_str[:max_chars]
                    return value_str

                # Build overview
                if isinstance(data, dict):
                    overview = {k: truncate_value(v) for k, v in data.items()}
                    overview_str = json.dumps(overview, indent=2)
                    content.metadata['total_keys'] = len(data)
                elif isinstance(data, list):
                    overview = [truncate_value(item) for item in data[:20]]  # First 20 items
                    overview_str = json.dumps(overview, indent=2)
                    content.metadata['total_items'] = len(data)
                    content.metadata['items_shown'] = min(20, len(data))
                else:
                    overview_str = truncate_value(data)

                content.content = overview_str
                content.partial = True
                content.metadata['truncated'] = True
                content.metadata['original_character_count'] = original_char_count
                content.metadata['type'] = 'json_overview'

            except json.JSONDecodeError:
                # If not valid JSON, just truncate without adding text
                original_char_count = len(content.content)
                content.content = content.content[:self.config.max_json_content_chars]
                content.partial = True
                content.metadata['truncated'] = True
                content.metadata['original_character_count'] = original_char_count

        return content

    async def _read_partial(
        self,
        handler: FileHandler,
        path: Path,
        **kwargs
    ) -> FileContent:
        """Read specific line/page range (with limits enforced)."""
        start_line = kwargs.pop('start_line', 1)
        end_line = kwargs.pop('end_line', None)

        # Always pass limits to metadata so agents know the constraints
        kwargs['max_pages_per_read'] = self.config.max_pages_per_read
        kwargs['max_lines_per_read'] = self.config.max_lines_per_read
        # Pass absolute character limit (safety guardrail)
        kwargs['max_characters_absolute'] = self.config.max_characters_absolute

        # For PDFs when no end specified: pass max_pages config to handler
        if path.suffix.lower() == '.pdf' and end_line is None:
            kwargs['max_pages'] = self.config.max_pages_per_read
            logger.debug(f"Reading partial PDF: {path} from page {start_line}, max_pages={self.config.max_pages_per_read}")
        elif not end_line:
            # For text files: use config's max_lines_per_read
            end_line = start_line + self.config.max_lines_per_read - 1
            logger.debug(f"Reading partial file: {path} lines {start_line}-{end_line}")

        return await handler.read_partial(path, start_line, end_line, **kwargs)

    def _generate_overview_guide(self, path: Path, structure: DocumentStructure) -> str:
        """Generate concise usage guide for overview content."""
        guide = []
        guide.append("=== FILE OVERVIEW ===")

        if path.suffix.lower() == '.pdf':
            total = structure.metadata.get('total_pages', '?')
            analyzed = structure.metadata.get('pages_processed', 10)
            guide.append(f"Total pages: {total}, analyzed: {analyzed}")
            guide.append("Options: read_file with start_page/end_page, search_files, or read_section")
        else:
            guide.append("Large file - structure overview only")
            guide.append("Options: read_file with start_line/end_line, or search_files")

        guide.append("=" * 50)
        return "\n".join(guide)

    async def _read_overview(
        self,
        handler: FileHandler,
        path: Path,
        **kwargs
    ) -> FileContent:
        """Get structure only, no full content."""
        logger.debug(f"Reading overview: {path}")

        try:
            # For PDFs, only analyze first 10 pages for structure (safety measure)
            if path.suffix.lower() == '.pdf' and 'max_pages' not in kwargs:
                kwargs['max_pages'] = 10

            structure = await handler.get_structure(path, **kwargs)

            # Generate usage guide
            usage_guide = self._generate_overview_guide(path, structure)

            # Generate summary as content
            max_depth = kwargs.get('max_depth', 2)
            summary = structure.to_summary(max_depth=max_depth)

            # Combine guide with summary
            content = usage_guide + "\n\n" + summary

            file_size = path.stat().st_size
            total_lines = self._count_lines(path)

            return FileContent(
                path=path,
                content=content,
                partial=True,
                structure=structure,
                total_lines=total_lines,
                file_size=file_size,
                character_count=len(content),
                estimated_tokens=len(content) // 4,  # Rough estimate: 1 token ≈ 4 chars
                sections_available=True,
                metadata={
                    'strategy': 'overview',
                    'sections_count': len(structure.sections),
                    'type': structure.type
                }
            )

        except Exception as e:
            logger.warning(f"Could not extract structure from {path}: {e}")
            # Fallback to partial read
            return await self._read_partial(handler, path, start_line=1, end_line=50)

    async def _read_progressive(
        self,
        handler: FileHandler,
        path: Path,
        **kwargs
    ) -> FileContent:
        """Return structure with ability to drill down."""
        logger.debug(f"Reading progressive: {path}")

        try:
            structure = await handler.get_structure(path, **kwargs)

            # Generate outline as content
            content = structure.to_outline()

            file_size = path.stat().st_size
            total_lines = self._count_lines(path)

            return FileContent(
                path=path,
                content=content,
                partial=True,
                structure=structure,
                total_lines=total_lines,
                file_size=file_size,
                character_count=len(content),
                estimated_tokens=len(content) // 4,  # Rough estimate: 1 token ≈ 4 chars
                sections_available=True,
                metadata={
                    'strategy': 'progressive',
                    'sections_count': structure._count_sections(),
                    'type': structure.type,
                    'read_section_help': f"Use read_section('{path}', section_id) to read specific sections"
                }
            )

        except Exception as e:
            logger.warning(f"Could not extract structure from {path}: {e}")
            # Fallback to overview (first N lines)
            return await self._read_partial(handler, path, start_line=1, end_line=100)

    async def _select_strategy_async(self, handler: FileHandler, path: Path) -> ReadStrategy:
        """
        Automatically select reading strategy based on file size and content.

        Safety: Always check file size first to prevent loading huge files into memory.
        Then for files that pass the size check, load content and check character count.

        Args:
            handler: File handler to use
            path: File path

        Returns:
            Selected ReadStrategy
        """
        # SAFETY CHECK: Always check file size first to prevent memory exhaustion
        file_size = path.stat().st_size if path.exists() else 0

        # If file is too large on disk, don't even try to load it
        if file_size > self.config.max_file_size_bytes:
            logger.warning(
                f"File {path} size ({file_size} bytes) exceeds max_file_size_bytes "
                f"({self.config.max_file_size_bytes}). Using OVERVIEW strategy."
            )
            return ReadStrategy.OVERVIEW

        # For PDFs: Use PARTIAL to read first few pages by default (not entire document)
        # PDF handler will apply max_pages_per_read limit
        if path.suffix.lower() == '.pdf':
            return ReadStrategy.PARTIAL

        # For images: Load in full (they passed size safety check)
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return ReadStrategy.FULL

        # For text-based files, do a quick character count
        try:
            # Read first portion to estimate total character count
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 10MB to estimate
                sample = f.read(10 * 1024 * 1024)
                sample_chars = len(sample)

                # If we read less than 10MB, this is the total
                if len(sample) < 10 * 1024 * 1024:
                    char_count = sample_chars
                else:
                    # Estimate total based on file size ratio
                    bytes_read = len(sample.encode('utf-8'))
                    total_bytes = path.stat().st_size
                    estimated_char_count = int((total_bytes / bytes_read) * sample_chars)
                    char_count = estimated_char_count

        except Exception as e:
            logger.warning(f"Could not count characters for {path}: {e}, falling back to file size")
            # Fallback to file size heuristic (assume 1 byte ≈ 1 char)
            char_count = path.stat().st_size if path.exists() else 0

        # Select strategy based on character count
        if char_count < self.config.small_file_threshold:
            return ReadStrategy.FULL

        if char_count < self.config.medium_file_threshold:
            return ReadStrategy.PARTIAL

        if char_count < self.config.large_file_threshold:
            return ReadStrategy.PROGRESSIVE

        # Very large files: overview only
        return ReadStrategy.OVERVIEW

    def _get_handler(self, path: Path) -> Optional[FileHandler]:
        """
        Get appropriate handler for file type.

        Args:
            path: File path

        Returns:
            FileHandler instance or None
        """
        # Try each handler until we find one that can handle the file
        for handler in self.handler_registry.values():
            if handler.can_handle(path):
                return handler

        # No specific handler found - use text handler as fallback
        if 'text' in self.handler_registry:
            return self.handler_registry['text']

        return None

    def _count_lines(self, path: Path) -> Optional[int]:
        """
        Quickly count lines in a file.

        Args:
            path: File path

        Returns:
            Number of lines or None if error
        """
        try:
            with open(path, 'rb') as f:
                return sum(1 for _ in f)
        except Exception:
            return None
