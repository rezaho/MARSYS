"""PDF file handler using PyMuPDF."""

import logging
from pathlib import Path
from typing import Optional

from .base import FileHandler, check_character_limit
from ..data_models import FileContent, DocumentStructure
from ..parsers.pdf_extractor import PDFStructureExtractor, PYMUPDF_AVAILABLE

logger = logging.getLogger(__name__)


class PDFHandler(FileHandler):
    """Handler for PDF files using PyMuPDF."""

    def __init__(self):
        """Initialize PDF handler."""
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available. PDF support will be disabled.")

        self.extractor = PDFStructureExtractor() if PYMUPDF_AVAILABLE else None

    def can_handle(self, path: Path) -> bool:
        """Check if this handler can process the file."""
        return path.suffix.lower() == '.pdf'

    async def read(self, path: Path, **kwargs) -> FileContent:
        """
        Read PDF content with optional image extraction and overview.

        Args:
            path: PDF file path
            max_pages: Maximum pages to read (optional)
            start_page: Starting page (optional, default: 1)
            end_page: Ending page (optional)
            extract_images: Whether to extract embedded images (default: False)
            include_overview: Whether to include document overview/structure (default: False)
            max_images_per_page: Max images per page (default: 10)
            max_pixels: Max pixels per image (default: None)
            provider: Token estimation provider (default: 'generic')

        Returns:
            FileContent with text and optionally ordered images and overview
        """
        if not self.extractor:
            raise ImportError("PyMuPDF is required for PDF support. Install with: pip install PyMuPDF")

        start_page = kwargs.get('start_page', 1)
        end_page = kwargs.get('end_page', kwargs.get('max_pages'))
        extract_images = kwargs.get('extract_images', False)
        include_overview = kwargs.get('include_overview', False)

        # Auto-include overview on first read (no ranges specified)
        # If start_page==1 and end_page is from max_pages config, it's first read
        is_first_read = (start_page == 1 and kwargs.get('max_pages') is not None and
                        'start_page' not in kwargs and 'end_page' not in kwargs)

        should_include_overview = include_overview or is_first_read

        try:
            if extract_images:
                # Extract ordered content (text + images)
                ordered_chunks, images = self.extractor.extract_ordered_content(
                    path,
                    start_page=start_page,
                    end_page=end_page,
                    extract_images=True,
                    max_images_per_page=kwargs.get('max_images_per_page', 10),
                    max_pixels=kwargs.get('max_pixels'),
                    provider=kwargs.get('provider', 'generic')
                )

                # Build text content with image placeholders
                # Now organized by page: text first, then images per page
                content_lines = []
                current_page = None

                for chunk in ordered_chunks:
                    page_num = chunk.get("page")

                    # Add page header when entering new page
                    if current_page != page_num and page_num is not None:
                        if current_page is not None:
                            content_lines.append("")  # Blank line between pages
                        content_lines.append(f"--- Page {page_num} ---")
                        current_page = page_num

                    if chunk["type"] == "text":
                        content_lines.append(chunk["text"])
                    elif chunk["type"] == "image":
                        content_lines.append(
                            f"[IMAGE {chunk['image_index'] + 1}: "
                            f"{chunk['dimensions']}, ~{chunk['tokens']} tokens]"
                        )

                content_text = "\n".join(content_lines)

                # Safety check: absolute character limit (from config)
                max_chars = kwargs.get('max_characters_absolute', 120_000)
                pages_requested = f"{start_page}-{end_page}" if end_page else f"{start_page}+"
                check_character_limit(
                    content=content_text,
                    max_chars=max_chars,
                    request_description=f"pages {pages_requested}",
                    solution_hint="Request fewer pages at a time (e.g., reduce the page range)."
                )

                # Calculate totals
                total_image_pixels = sum(img.total_pixels for img in images) if images else None
                total_image_tokens = sum(img.estimated_tokens for img in images) if images else None

                # Get total pages
                import pymupdf
                with pymupdf.open(str(path)) as pdf:
                    total_pages = len(pdf)

                file_size = path.stat().st_size

                # Extract overview if requested or first read
                overview = None
                if should_include_overview:
                    try:
                        structure = self.extractor.extract_hierarchy(path, max_pages=total_pages)
                        overview = {
                            'type': structure.type,
                            'total_sections': len(structure.sections),
                            'sections': [s.to_dict() for s in structure.sections],
                            'metadata': structure.metadata
                        }
                    except Exception as e:
                        logger.warning(f"Failed to extract PDF overview: {e}")
                        overview = None

                metadata = {
                    'type': 'pdf',
                    'total_pages': total_pages,
                    'pages_read': f"{start_page}-{end_page}" if end_page else f"{start_page}+",
                    'images_extracted': len(images),
                    'total_image_tokens': total_image_tokens,
                    'ordered_chunks': ordered_chunks  # KEY: Include for FileContent.to_tool_response()
                }

                # Add limits if provided (only in partial strategy)
                if 'max_pages_per_read' in kwargs:
                    metadata['max_pages_per_read'] = kwargs['max_pages_per_read']
                if 'max_lines_per_read' in kwargs:
                    metadata['max_lines_per_read'] = kwargs['max_lines_per_read']

                if overview:
                    metadata['overview'] = overview

                return FileContent(
                    path=path,
                    content=content_text,
                    partial=end_page is not None,
                    encoding='utf-8',
                    file_size=file_size,
                    character_count=len(content_text),
                    estimated_tokens=len(content_text) // 4,
                    images=images,
                    total_image_pixels=total_image_pixels,
                    total_estimated_image_tokens=total_image_tokens,
                    metadata=metadata
                )
            else:
                # Text-only extraction (existing behavior)
                text = self.extractor.extract_text(path, start_page, end_page)

                # Safety check: absolute character limit (from config)
                max_chars = kwargs.get('max_characters_absolute', 120_000)
                pages_requested = f"{start_page}-{end_page}" if end_page else f"{start_page}+"
                check_character_limit(
                    content=text,
                    max_chars=max_chars,
                    request_description=f"pages {pages_requested}",
                    solution_hint="Request fewer pages at a time (e.g., reduce the page range)."
                )

                file_size = path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                # Get total pages for overview
                import pymupdf
                with pymupdf.open(str(path)) as pdf:
                    total_pages = len(pdf)

                # Extract overview if requested or first read
                overview = None
                if should_include_overview:
                    try:
                        structure = self.extractor.extract_hierarchy(path, max_pages=total_pages)
                        overview = {
                            'type': structure.type,
                            'total_sections': len(structure.sections),
                            'sections': [s.to_dict() for s in structure.sections],
                            'metadata': structure.metadata
                        }
                    except Exception as e:
                        logger.warning(f"Failed to extract PDF overview: {e}")
                        overview = None

                metadata = {
                    'type': 'pdf',
                    'size_mb': round(file_size_mb, 2),
                    'pages_read': end_page if end_page else 'all',
                    'total_pages': total_pages
                }

                # Add limits if provided (only in partial strategy)
                if 'max_pages_per_read' in kwargs:
                    metadata['max_pages_per_read'] = kwargs['max_pages_per_read']
                if 'max_lines_per_read' in kwargs:
                    metadata['max_lines_per_read'] = kwargs['max_lines_per_read']

                if overview:
                    metadata['overview'] = overview

                return FileContent(
                    path=path,
                    content=text,
                    partial=end_page is not None,
                    encoding='utf-8',  # PDF text is converted to UTF-8
                    file_size=file_size,
                    character_count=len(text),
                    estimated_tokens=len(text) // 4,  # Rough estimate: 1 token ≈ 4 chars
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}", exc_info=True)
            raise

    async def get_structure(self, path: Path, **kwargs) -> DocumentStructure:
        """
        Extract hierarchical structure from PDF.

        Args:
            path: PDF file path
            max_pages: Maximum pages to analyze (optional, default: all)

        Returns:
            DocumentStructure with sections based on font size analysis
        """
        if not self.extractor:
            raise ImportError("PyMuPDF is required for PDF support. Install with: pip install PyMuPDF")

        max_pages = kwargs.get('max_pages', None)

        try:
            structure = self.extractor.extract_hierarchy(path, max_pages)
            return structure

        except Exception as e:
            logger.error(f"Error extracting PDF structure from {path}: {e}", exc_info=True)
            # Return minimal structure
            return DocumentStructure(
                type="pdf",
                sections=[],
                metadata={
                    'error': str(e),
                    'extraction_failed': True
                }
            )

    async def read_section(self, path: Path, section_id: str, **kwargs) -> str:
        """
        Read a specific section from the PDF.

        Args:
            path: PDF file path
            section_id: Section identifier (e.g., "section:1:1")

        Returns:
            Content of the specified section
        """
        if not self.extractor:
            raise ImportError("PyMuPDF is required for PDF support. Install with: pip install PyMuPDF")

        # Get structure first
        structure = await self.get_structure(path, **kwargs)

        # Find the section
        section = structure.find_section(section_id)

        if not section:
            raise ValueError(f"Section '{section_id}' not found in PDF")

        # Extract text for this section's page range
        text = self.extractor.extract_text(
            path,
            start_page=section.start_page,
            end_page=section.end_page,
            start_line=section.start_line,
            end_line=section.end_line
        )

        return text

    async def read_partial(
        self,
        path: Path,
        start_line: int,
        end_line: Optional[int],
        **kwargs
    ) -> FileContent:
        """
        Read specific page range from PDF.

        For PDFs, we interpret lines as pages for simplicity.
        Supports image extraction via extract_images parameter.

        Args:
            path: PDF file path
            start_line: Starting page number (1-indexed)
            end_line: Ending page number (1-indexed, inclusive). Can be None if max_pages is provided in kwargs.
            **kwargs: Additional parameters including extract_images and max_pages

        Returns:
            FileContent with extracted text and optionally images from page range
        """
        if not self.extractor:
            raise ImportError("PyMuPDF is required for PDF support. Install with: pip install PyMuPDF")

        # For PDFs, use start_line as start_page unless explicitly provided
        start_page = kwargs.get('start_page', start_line)
        end_page = kwargs.get('end_page', end_line)

        # If no end_page specified but max_pages is provided, calculate end_page
        if end_page is None and 'max_pages' in kwargs:
            max_pages = kwargs.pop('max_pages')
            end_page = start_page + max_pages - 1

        # IMPORTANT: Preserve extract_images parameter!
        kwargs['start_page'] = start_page
        kwargs['end_page'] = end_page

        # Use the full read() method which handles extract_images properly
        return await self.read(path, **kwargs)
