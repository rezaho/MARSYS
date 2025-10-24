"""
File search engine with multiple search strategies.

This module implements content search, filename search, and structure search
capabilities with pagination and filtering.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Pattern

from .config import FileOperationConfig
from .data_models import SearchResults, SearchMatch, SearchType
from .handlers.base import FileHandler, SearchableHandler

logger = logging.getLogger(__name__)


class FileSearchEngine:
    """Multi-strategy file search engine."""

    def __init__(
        self,
        config: FileOperationConfig,
        handler_registry: dict
    ):
        """
        Initialize file search engine.

        Args:
            config: File operation configuration
            handler_registry: Registry of file type handlers
        """
        self.config = config
        self.handler_registry = handler_registry

    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.CONTENT,
        path: Optional[Path] = None,
        **kwargs
    ) -> SearchResults:
        """
        Search files with specified strategy.

        Args:
            query: Search query
            search_type: Type of search (CONTENT, FILENAME, STRUCTURE)
            path: Root path to search (default: base_directory)
            **kwargs: Additional search parameters:
                - file_pattern: Glob pattern for files to search (default: **/**)
                - max_results: Maximum results to return
                - case_sensitive: Case-sensitive search (default: False)
                - include_context: Include context lines (default: True)
                - context_lines: Number of context lines (default: 3)

        Returns:
            SearchResults object
        """
        start_time = time.time()

        # Use base directory if no path specified
        if path is None:
            path = self.config.base_directory or Path.cwd()

        # Ensure path exists
        if not path.exists():
            return SearchResults(
                query=query,
                search_type=search_type,
                matches=[],
                total_matches=0,
                truncated=False,
                files_searched=0,
                metadata={'error': f'Path does not exist: {path}'}
            )

        # Dispatch to appropriate search method
        if search_type == SearchType.CONTENT:
            results = await self._content_search(query, path, **kwargs)
        elif search_type == SearchType.FILENAME:
            results = await self._filename_search(query, path, **kwargs)
        elif search_type == SearchType.STRUCTURE:
            results = await self._structure_search(query, path, **kwargs)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        results.search_duration_ms = duration_ms

        return results

    async def _content_search(
        self,
        query: str,
        path: Path,
        **kwargs
    ) -> SearchResults:
        """
        Search within file contents (grep-like).

        Args:
            query: Search query (regex pattern)
            path: Root path to search
            **kwargs: Additional parameters

        Returns:
            SearchResults
        """
        file_pattern = kwargs.get('file_pattern', '**/*')
        max_results = kwargs.get('max_results', self.config.max_search_results)
        case_sensitive = kwargs.get('case_sensitive', False)
        include_context = kwargs.get('include_context', True)
        context_lines = kwargs.get('context_lines', min(kwargs.get('context_lines', 3), self.config.max_context_lines))

        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            return SearchResults(
                query=query,
                search_type=SearchType.CONTENT,
                matches=[],
                total_matches=0,
                metadata={'error': f'Invalid regex pattern: {e}'}
            )

        matches = []
        files_searched = 0

        # Search files
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob(file_pattern))

        for file_path in files:
            if not file_path.is_file():
                continue

            # Skip blocked files
            if self.config.is_file_blocked(file_path):
                continue

            files_searched += 1

            # Check if PDF
            if file_path.suffix.lower() == '.pdf':
                pdf_matches = await self._search_pdf_content(
                    file_path,
                    pattern,
                    include_context,
                    context_lines
                )
                matches.extend(pdf_matches)
            else:
                # Skip binary files for text search
                if self._is_binary_file(file_path):
                    continue

                # Search in text file
                file_matches = self._search_file_content(
                    file_path,
                    pattern,
                    include_context,
                    context_lines
                )
                matches.extend(file_matches)

            # Stop if we have enough results
            if len(matches) >= max_results:
                break

        # Truncate if needed
        total_matches = len(matches)
        truncated = total_matches > max_results
        matches = matches[:max_results]

        return SearchResults(
            query=query,
            search_type=SearchType.CONTENT,
            matches=matches,
            total_matches=total_matches,
            truncated=truncated,
            files_searched=files_searched
        )

    def _search_file_content(
        self,
        file_path: Path,
        pattern: Pattern,
        include_context: bool,
        context_lines: int
    ) -> List[SearchMatch]:
        """Search for pattern in a single file."""
        matches = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                if pattern.search(line):
                    # Get context
                    context_before = []
                    context_after = []

                    if include_context:
                        # Context before
                        start = max(0, line_num - context_lines - 1)
                        context_before = [lines[i].rstrip('\n') for i in range(start, line_num - 1)]

                        # Context after
                        end = min(len(lines), line_num + context_lines)
                        context_after = [lines[i].rstrip('\n') for i in range(line_num, end)]

                    matches.append(SearchMatch(
                        file=file_path,
                        line=line_num,
                        match=line.rstrip('\n'),
                        context_before=context_before,
                        context_after=context_after
                    ))

        except Exception as e:
            logger.warning(f"Error searching {file_path}: {e}")

        return matches

    async def _search_pdf_content(
        self,
        path: Path,
        pattern: Pattern,
        include_context: bool,
        context_lines: int
    ) -> List[SearchMatch]:
        """Search within PDF content with page numbers."""
        matches = []
        handler = self.handler_registry.get('pdf')
        if not handler:
            logger.warning("PDF handler not available, skipping PDF search")
            return matches

        try:
            # Check if PyMuPDF is available
            try:
                import pymupdf
            except ImportError:
                logger.warning("PyMuPDF not available, skipping PDF search")
                return matches

            with pymupdf.open(path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text = page.get_text()
                    if not text:
                        continue

                    lines = text.split('\n')
                    for line_num, line in enumerate(lines, start=1):
                        if pattern.search(line):
                            context_before = []
                            context_after = []

                            if include_context:
                                start_ctx = max(0, line_num - context_lines - 1)
                                end_ctx = min(len(lines), line_num + context_lines)
                                context_before = lines[start_ctx:line_num-1]
                                context_after = lines[line_num:end_ctx]

                            matches.append(SearchMatch(
                                file=path,
                                line=line_num,
                                match=line.strip(),
                                context_before=context_before,
                                context_after=context_after,
                                metadata={'page': page_num + 1, 'file_type': 'pdf'}  # +1 for 1-based page numbers
                            ))

        except Exception as e:
            logger.warning(f"Error searching PDF {path}: {e}")

        return matches

    async def _filename_search(
        self,
        query: str,
        path: Path,
        **kwargs
    ) -> SearchResults:
        """
        Search by filename pattern (glob or substring).

        Args:
            query: Glob pattern or substring to search for in filenames
            path: Root path to search
            **kwargs: Additional parameters
                - recursive: Search subdirectories (default: True)

        Returns:
            SearchResults
        """
        max_results = kwargs.get('max_results', self.config.max_search_results)
        recursive = kwargs.get('recursive', True)

        matches = []
        files_searched = 0

        try:
            # Determine if query is a glob pattern or substring
            is_glob = '*' in query or '?' in query or '[' in query

            if path.is_file():
                # Check if single file matches
                if is_glob:
                    files = [path] if path.match(query) else []
                else:
                    files = [path] if query.lower() in path.name.lower() else []
            else:
                # Search directory
                if is_glob:
                    # Use glob pattern
                    if recursive and not query.startswith('**/'):
                        # Make glob recursive if not already
                        if query.startswith('*'):
                            # Already has wildcard at start, just add **/
                            pattern = f"**/{query}"
                        else:
                            # No wildcard at start
                            pattern = f"**/{query}"
                    else:
                        pattern = query
                    files = list(path.glob(pattern))
                else:
                    # Substring search - find all files containing the query
                    glob_pattern = '**/*' if recursive else '*'
                    all_files = path.glob(glob_pattern)
                    files = [f for f in all_files if query.lower() in f.name.lower()]

            for file_path in files:
                if not file_path.is_file():
                    continue

                # Skip blocked files
                if self.config.is_file_blocked(file_path):
                    continue

                files_searched += 1

                matches.append(SearchMatch(
                    file=file_path,
                    line=1,
                    match=str(file_path)
                ))

                if len(matches) >= max_results:
                    break

        except Exception as e:
            logger.error(f"Error in filename search: {e}")
            return SearchResults(
                query=query,
                search_type=SearchType.FILENAME,
                matches=[],
                total_matches=0,
                metadata={'error': str(e)}
            )

        total_matches = len(matches)
        truncated = total_matches > max_results
        matches = matches[:max_results]

        return SearchResults(
            query=query,
            search_type=SearchType.FILENAME,
            matches=matches,
            total_matches=total_matches,
            truncated=truncated,
            files_searched=files_searched
        )

    async def _structure_search(
        self,
        query: str,
        path: Path,
        **kwargs
    ) -> SearchResults:
        """
        Search within code/document structure.

        Args:
            query: Search query (regex for structure names)
            path: Root path to search
            **kwargs: Additional parameters

        Returns:
            SearchResults
        """
        file_pattern = kwargs.get('file_pattern', '**/*.py')  # Default to Python files
        max_results = kwargs.get('max_results', self.config.max_search_results)
        case_sensitive = kwargs.get('case_sensitive', False)

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            return SearchResults(
                query=query,
                search_type=SearchType.STRUCTURE,
                matches=[],
                total_matches=0,
                metadata={'error': f'Invalid regex pattern: {e}'}
            )

        matches = []
        files_searched = 0

        # Search files
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob(file_pattern))

        for file_path in files:
            if not file_path.is_file():
                continue

            # Skip blocked files
            if self.config.is_file_blocked(file_path):
                continue

            files_searched += 1

            # Get handler
            handler = self._get_handler(file_path)
            if not handler:
                continue

            # Check if handler supports structure search
            if isinstance(handler, SearchableHandler):
                try:
                    handler_matches = await handler.search_structure(file_path, query)
                    matches.extend(handler_matches)
                except Exception as e:
                    logger.warning(f"Error searching structure in {file_path}: {e}")
            else:
                # Fallback: search in structure titles
                try:
                    structure = await handler.get_structure(file_path)
                    for section in structure.sections:
                        if pattern.search(section.title):
                            matches.append(SearchMatch(
                                file=file_path,
                                line=section.start_line,
                                match=section.title,
                                section_id=section.id
                            ))
                except Exception as e:
                    logger.warning(f"Error getting structure from {file_path}: {e}")

            if len(matches) >= max_results:
                break

        total_matches = len(matches)
        truncated = total_matches > max_results
        matches = matches[:max_results]

        return SearchResults(
            query=query,
            search_type=SearchType.STRUCTURE,
            matches=matches,
            total_matches=total_matches,
            truncated=truncated,
            files_searched=files_searched
        )

    def _get_handler(self, path: Path) -> Optional[FileHandler]:
        """Get appropriate handler for file type."""
        for handler in self.handler_registry.values():
            if handler.can_handle(path):
                return handler
        return None

    def _is_binary_file(self, path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                # Check for null bytes (common in binary files)
                return b'\x00' in chunk
        except Exception:
            return True  # Assume binary if we can't read it
