"""
PDF extractor using PyMuPDF.

Extracts text, images, and hierarchical structure from PDFs.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not installed. PDF support will be disabled.")

from ..data_models import DocumentStructure, Section

logger = logging.getLogger(__name__)


class PDFStructureExtractor:
    """
    Extracts text, images, and structure from PDF files using PyMuPDF.
    """

    def __init__(self):
        """Initialize PDF extractor."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF support. Install with: pip install PyMuPDF")

    def extract_text(
        self,
        path: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> str:
        """
        Extract text from specific pages/lines in PDF.

        Args:
            path: PDF file path
            start_page: Starting page (1-indexed)
            end_page: Ending page (1-indexed, inclusive)
            start_line: Starting line (1-indexed) - applied after page extraction
            end_line: Ending line (1-indexed, inclusive) - applied after page extraction

        Returns:
            Extracted text
        """
        import pymupdf

        with pymupdf.open(str(path)) as pdf:
            total_pages = len(pdf)

            # Default to all pages
            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = total_pages

            # Validate page numbers
            start_page = max(1, min(start_page, total_pages))
            end_page = max(1, min(end_page, total_pages))

            text_parts = []

            for page_num in range(start_page - 1, end_page):
                page = pdf[page_num]
                page_text = page.get_text()

                if page_text:
                    text_parts.append(page_text)

            # Join all text
            full_text = '\n'.join(text_parts)

            # Apply line filtering if specified
            if start_line is not None or end_line is not None:
                lines = full_text.split('\n')
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                full_text = '\n'.join(lines[start_idx:end_idx])

            return full_text

    def extract_ordered_content(
        self,
        path: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        extract_images: bool = True,
        max_images_per_page: int = 10,
        max_pixels: Optional[int] = None,
        provider: str = 'generic'
    ) -> Tuple[List[Dict[str, Any]], List['ImageData']]:
        """
        Extract PDF content organized by page: text first, then images.

        For each page, returns:
        1. One text block containing all text from that page
        2. Followed by all images from that page

        Args:
            path: PDF file path
            start_page: Starting page (1-indexed)
            end_page: Ending page (1-indexed, None = last page)
            extract_images: Whether to extract embedded images
            max_images_per_page: Maximum images per page
            max_pixels: Maximum pixels per image (will downsample if needed)
            provider: Provider for token estimation

        Returns:
            Tuple of:
            - ordered_chunks: List of dicts organized by page:
              [
                {"type": "text", "text": "Page 1 text...", "page": 1},
                {"type": "image", "image_index": 0, "page": 1, "bbox": [...], "dimensions": "800x600", "tokens": 170},
                {"type": "image", "image_index": 1, "page": 1, "bbox": [...], "dimensions": "600x400", "tokens": 120},
                {"type": "text", "text": "Page 2 text...", "page": 2},
                {"type": "image", "image_index": 2, "page": 2, "bbox": [...], "dimensions": "1024x768", "tokens": 255},
                ...
              ]
            - all_images: List of ImageData objects (ordered by appearance)
        """
        import pymupdf
        from PIL import Image
        import io
        from ..data_models import ImageData
        from ..token_estimation import estimate_image_tokens

        ordered_chunks = []
        all_images = []  # Global list of ImageData
        image_counter = 0

        with pymupdf.open(str(path)) as pdf:
            total_pages = len(pdf)
            end = min(end_page or total_pages, total_pages)

            for page_num in range(start_page - 1, end):
                page = pdf[page_num]
                page_number = page_num + 1  # 1-indexed for user

                # Get page layout: text blocks and image blocks
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])

                # Collect all text and images for this page separately
                page_text_parts = []
                page_images = []  # Images for this page
                images_extracted = 0

                # Process blocks in order
                for block_idx, block in enumerate(blocks):
                    block_type = block.get("type")  # 0 = text, 1 = image
                    bbox = block.get("bbox", [0, 0, 0, 0])  # [x0, y0, x1, y1]

                    if block_type == 0:
                        # TEXT BLOCK - collect text
                        lines = []
                        for line in block.get("lines", []):
                            spans = line.get("spans", [])
                            line_text = "".join(span.get("text", "") for span in spans)
                            if line_text.strip():
                                lines.append(line_text)

                        text = "\n".join(lines)
                        if text.strip():
                            page_text_parts.append(text)

                    elif block_type == 1 and extract_images:
                        # IMAGE BLOCK - collect image info
                        if images_extracted >= max_images_per_page:
                            logger.debug(f"Skipping image on page {page_number}: max_images_per_page ({max_images_per_page}) reached")
                            continue

                        try:
                            # Extract image data from block
                            image_bytes = block.get("image")
                            if not image_bytes:
                                continue

                            # Get image extension and dimensions
                            image_ext = block.get("ext", "png")
                            width = block.get("width", 0)
                            height = block.get("height", 0)

                            # Convert to PIL Image for processing
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            actual_width, actual_height = pil_image.size

                            # Use actual dimensions if block dimensions are missing
                            if width == 0 or height == 0:
                                width, height = actual_width, actual_height

                            # Downsample if exceeds max_pixels
                            if max_pixels and (width * height) > max_pixels:
                                scale = (max_pixels / (width * height)) ** 0.5
                                new_size = (int(width * scale), int(height * scale))
                                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                                width, height = new_size

                                # Re-encode downsampled image
                                buffer = io.BytesIO()
                                save_format = image_ext.upper()
                                if save_format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'WEBP']:
                                    save_format = 'PNG'
                                if save_format == 'JPG':
                                    save_format = 'JPEG'
                                pil_image.save(buffer, format=save_format)
                                image_bytes = buffer.getvalue()

                            # Estimate tokens for vision models
                            estimated_tokens = estimate_image_tokens(width, height, provider, 'high')

                            # Create ImageData object
                            image_data = ImageData(
                                data=image_bytes,
                                format=image_ext,
                                width=width,
                                height=height,
                                page_number=page_number,
                                estimated_tokens=estimated_tokens,
                                metadata={
                                    'source': 'pdf_extraction',
                                    'bbox': list(bbox),
                                    'provider': provider,
                                    'block_num': block_idx
                                }
                            )

                            all_images.append(image_data)

                            # Store image info for this page
                            page_images.append({
                                "image_index": image_counter,
                                "bbox": list(bbox),
                                "dimensions": f"{width}x{height}",
                                "tokens": estimated_tokens
                            })

                            image_counter += 1
                            images_extracted += 1

                        except Exception as e:
                            logger.warning(f"Failed to extract image on page {page_number}, block {block_idx}: {e}")
                            continue

                # Assemble page chunks: text first, then images
                # 1. Add combined text for the page (if any)
                if page_text_parts:
                    combined_text = "\n\n".join(page_text_parts)
                    ordered_chunks.append({
                        "type": "text",
                        "text": combined_text,
                        "page": page_number
                    })

                # 2. Add all images for the page
                for img_info in page_images:
                    ordered_chunks.append({
                        "type": "image",
                        "image_index": img_info["image_index"],
                        "page": page_number,
                        "bbox": img_info["bbox"],
                        "dimensions": img_info["dimensions"],
                        "tokens": img_info["tokens"]
                    })

        return ordered_chunks, all_images

    def extract_hierarchy(
        self,
        path: Path,
        max_pages: Optional[int] = None
    ) -> DocumentStructure:
        """
        Extract hierarchical structure from PDF using PyMuPDF.

        Uses font size analysis to detect headings and build a hierarchy.

        Args:
            path: PDF file path
            max_pages: Maximum number of pages to process (None = all)

        Returns:
            DocumentStructure with sections
        """
        import pymupdf

        logger.debug(f"Extracting structure from PDF: {path}")

        with pymupdf.open(str(path)) as pdf:
            total_pages = len(pdf)

            if max_pages:
                pages_to_process = min(max_pages, total_pages)
            else:
                pages_to_process = total_pages

            # Analyze font sizes across first N pages to identify heading levels
            font_sizes = self._analyze_font_sizes(pdf, pages_to_process)

            # Extract text blocks with font information
            text_blocks = self._extract_text_blocks(pdf, pages_to_process, font_sizes)

            # Build hierarchy from text blocks
            sections = self._build_hierarchy(text_blocks)

            return DocumentStructure(
                type="pdf",
                sections=sections,
                metadata={
                    'total_pages': total_pages,
                    'pages_processed': pages_to_process,
                    'font_levels': len(font_sizes['heading_sizes']),
                    'total_sections': len(sections)
                }
            )

    def _analyze_font_sizes(
        self,
        pdf: Any,
        num_pages: int
    ) -> Dict[str, Any]:
        """
        Analyze font sizes to identify heading levels using PyMuPDF.

        Args:
            pdf: PyMuPDF PDF object
            num_pages: Number of pages to analyze

        Returns:
            Dict with font size analysis
        """
        font_sizes = defaultdict(int)

        # Collect font sizes from first N pages
        for page_num in range(min(num_pages, len(pdf))):
            page = pdf[page_num]

            # Get text with font information
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 12)
                            font_sizes[size] += 1

        if not font_sizes:
            # No font information, use defaults
            return {
                'heading_sizes': [18, 16, 14],
                'body_size': 12,
                'size_distribution': {}
            }

        # Sort by frequency
        sorted_sizes = sorted(font_sizes.items(), key=lambda x: x[1], reverse=True)

        # Most common is likely body text
        body_size = sorted_sizes[0][0]

        # Larger sizes are likely headings
        heading_sizes = sorted([size for size, count in sorted_sizes if size > body_size])

        # Limit to top 3 heading levels
        heading_sizes = heading_sizes[-3:] if len(heading_sizes) > 3 else heading_sizes

        return {
            'heading_sizes': heading_sizes,
            'body_size': body_size,
            'size_distribution': dict(sorted_sizes)
        }

    def _extract_text_blocks(
        self,
        pdf: Any,
        num_pages: int,
        font_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract text blocks with metadata using PyMuPDF.

        Args:
            pdf: PyMuPDF PDF object
            num_pages: Number of pages to process
            font_info: Font size analysis

        Returns:
            List of text blocks
        """
        text_blocks = []
        heading_sizes = set(font_info['heading_sizes'])

        for page_num in range(min(num_pages, len(pdf))):
            page = pdf[page_num]

            # Get text with font information
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_texts = []
                        line_sizes = []

                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            size = span.get("size", font_info['body_size'])
                            if text.strip():
                                line_texts.append(text)
                                line_sizes.append(size)

                        if line_texts:
                            line_text = "".join(line_texts)
                            # Get dominant font size for this line
                            font_size = max(set(line_sizes), key=line_sizes.count) if line_sizes else font_info['body_size']

                            # Determine if heading
                            is_heading = font_size in heading_sizes
                            heading_level = 0

                            if is_heading:
                                # Larger size = higher importance (lower level number)
                                sorted_headings = sorted(heading_sizes, reverse=True)
                                heading_level = sorted_headings.index(font_size) + 1

                            text_blocks.append({
                                'page': page_num + 1,
                                'text': line_text,
                                'font_size': font_size,
                                'is_heading': is_heading,
                                'heading_level': heading_level
                            })

        return text_blocks

    def _build_hierarchy(
        self,
        text_blocks: List[Dict[str, Any]]
    ) -> List[Section]:
        """
        Build hierarchical structure from text blocks.

        Args:
            text_blocks: List of text blocks with metadata

        Returns:
            List of top-level sections
        """
        sections = []
        section_stack = []  # Stack to track nested sections

        current_section = None
        section_counter = defaultdict(int)

        for block in text_blocks:
            if not block['is_heading']:
                # Body text - add to current section's content
                if current_section:
                    # Track content (don't store it all to save memory)
                    if 'content_lines' not in current_section.metadata:
                        current_section.metadata['content_lines'] = 0
                    current_section.metadata['content_lines'] += 1
                continue

            # This is a heading
            level = block['heading_level']
            title = block['text'].strip()

            # Pop sections from stack until we find the parent
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()

            # Create new section
            section_counter[level] += 1
            section_id = f"section:{level}:{section_counter[level]}"

            section = Section(
                id=section_id,
                title=title,
                level=level,
                start_page=block['page'],
                end_page=block['page'],  # Will be updated
                metadata={
                    'font_size': block['font_size'],
                    'heading_level': level
                }
            )

            # Add to parent's children or to top-level
            if section_stack:
                parent = section_stack[-1]
                parent.children.append(section)
                section.parent = parent
                # Update parent's end page
                parent.end_page = block['page']
            else:
                sections.append(section)

            # Push to stack
            section_stack.append(section)
            current_section = section

        return sections
