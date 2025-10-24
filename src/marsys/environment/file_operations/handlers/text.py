"""Text file handler for plain text files."""

from pathlib import Path
from typing import Optional

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

from .base import FileHandler, check_character_limit
from ..data_models import FileContent, DocumentStructure, Section


class TextFileHandler(FileHandler):
    """Handler for plain text files (.txt, .log, etc.)."""

    SUPPORTED_EXTENSIONS = ['.txt', '.log', '.text', '.md', '.rst', '.csv', '.tsv']

    def can_handle(self, path: Path) -> bool:
        """Check if this handler can process the file."""
        extension = path.suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS or extension == ''

    async def read(self, path: Path, **kwargs) -> FileContent:
        """
        Read text file content.

        Args:
            path: File path to read
            encoding: Optional encoding (if None, will auto-detect)

        Returns:
            FileContent with file contents
        """
        encoding = kwargs.get('encoding')

        # Auto-detect encoding if not provided
        if not encoding:
            encoding = self._detect_encoding(path)

        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()

            # Safety check: absolute character limit (from config)
            max_chars = kwargs.get('max_characters_absolute', 120_000)
            check_character_limit(
                content=content,
                max_chars=max_chars,
                request_description="read entire file",
                solution_hint="Use read_file with start_line/end_line parameters to read specific sections."
            )

            file_size = path.stat().st_size
            total_lines = content.count('\n') + 1 if content else 0

            return FileContent(
                path=path,
                content=content,
                partial=False,
                encoding=encoding,
                total_lines=total_lines,
                file_size=file_size,
                character_count=len(content),
                estimated_tokens=len(content) // 4,  # Rough estimate: 1 token ≈ 4 chars
                metadata={
                    'type': 'text',
                    'extension': self.get_file_extension(path)
                }
            )

        except UnicodeDecodeError as e:
            # Try with fallback encoding
            fallback_encoding = kwargs.get('fallback_encoding', 'latin-1')
            try:
                with open(path, 'r', encoding=fallback_encoding) as f:
                    content = f.read()

                return FileContent(
                    path=path,
                    content=content,
                    partial=False,
                    encoding=fallback_encoding,
                    character_count=len(content),
                    estimated_tokens=len(content) // 4,  # Rough estimate: 1 token ≈ 4 chars
                    metadata={
                        'type': 'text',
                        'encoding_fallback': True,
                        'original_error': str(e)
                    }
                )
            except Exception as fallback_error:
                raise ValueError(f"Could not read file with encoding {encoding} or {fallback_encoding}: {fallback_error}")

    async def get_structure(self, path: Path, **kwargs) -> DocumentStructure:
        """
        Extract structure from text file.

        For plain text files, structure is based on empty lines (paragraphs).

        Args:
            path: File path

        Returns:
            DocumentStructure with paragraph sections
        """
        content = await self.read(path, **kwargs)
        sections = []

        if not content.content:
            return DocumentStructure(
                type=self.get_file_type(path),
                sections=[],
                metadata={'empty_file': True}
            )

        lines = content.content.split('\n')
        current_paragraph_start = None
        paragraph_count = 0

        for line_num, line in enumerate(lines, 1):
            if line.strip():  # Non-empty line
                if current_paragraph_start is None:
                    current_paragraph_start = line_num
            else:  # Empty line - end of paragraph
                if current_paragraph_start is not None:
                    paragraph_count += 1
                    # Get first line as title
                    first_line = lines[current_paragraph_start - 1].strip()[:50]
                    if len(lines[current_paragraph_start - 1].strip()) > 50:
                        first_line += "..."

                    sections.append(Section(
                        id=f"paragraph:{paragraph_count}",
                        title=first_line,
                        level=1,
                        start_line=current_paragraph_start,
                        end_line=line_num - 1
                    ))
                    current_paragraph_start = None

        # Handle last paragraph if file doesn't end with empty line
        if current_paragraph_start is not None:
            paragraph_count += 1
            first_line = lines[current_paragraph_start - 1].strip()[:50]
            if len(lines[current_paragraph_start - 1].strip()) > 50:
                first_line += "..."

            sections.append(Section(
                id=f"paragraph:{paragraph_count}",
                title=first_line,
                level=1,
                start_line=current_paragraph_start,
                end_line=len(lines)
            ))

        return DocumentStructure(
            type=self.get_file_type(path),
            sections=sections,
            metadata={
                'total_paragraphs': paragraph_count,
                'total_lines': len(lines)
            }
        )

    def get_file_type(self, path: Path) -> str:
        """
        Get file type based on extension.

        Args:
            path: File path

        Returns:
            File type identifier (e.g., "markdown", "python", "json", "text")
        """
        ext = path.suffix.lower()

        # Map extensions to types
        extension_map = {
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.csv': 'csv',
        }

        return extension_map.get(ext, 'text')

    async def read_section(self, path: Path, section_id: str, **kwargs) -> str:
        """
        Read a specific section (paragraph) from the file.

        Args:
            path: File path
            section_id: Section ID (e.g., "paragraph:1")

        Returns:
            Content of the specified section
        """
        structure = await self.get_structure(path, **kwargs)
        section = structure.find_section(section_id)

        if not section:
            raise ValueError(f"Section '{section_id}' not found in {path}")

        # Read the specific line range
        partial_content = await self.read_partial(
            path,
            section.start_line,
            section.end_line or section.start_line,
            **kwargs
        )

        return partial_content.content

    def _detect_encoding(self, path: Path) -> str:
        """
        Detect file encoding using chardet.

        Args:
            path: File path

        Returns:
            Detected encoding name
        """
        try:
            with open(path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection

            result = chardet.detect(raw_data)
            detected_encoding = result.get('encoding', 'utf-8')

            # Normalize encoding names
            if detected_encoding:
                detected_encoding = detected_encoding.lower()

            # Default to utf-8 if confidence is low
            if not detected_encoding or result.get('confidence', 0) < 0.7:
                detected_encoding = 'utf-8'

            return detected_encoding

        except Exception:
            # Fallback to utf-8
            return 'utf-8'
