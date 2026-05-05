"""Image file handler for reading and processing images."""

import base64
import logging
from pathlib import Path
from typing import Optional

from .base import FileHandler
from ..data_models import FileContent, DocumentStructure, Section, ImageData
from ..token_estimation import estimate_image_tokens

logger = logging.getLogger(__name__)


class ImageFileHandler(FileHandler):
    """Handler for image files (.jpg, .png, .webp, .gif, .bmp, etc.)."""

    SUPPORTED_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp',
        '.tiff', '.tif', '.ico', '.svg'
    ]

    def can_handle(self, path: Path) -> bool:
        """Check if this handler can process the file."""
        extension = path.suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS

    async def read(self, path: Path, **kwargs) -> FileContent:
        """
        Read image file.

        Args:
            path: File path to read
            provider: Provider for token estimation ("openai", "anthropic", "google", "xai", "generic")
            detail: Detail level ("high" or "low") for some providers
            include_base64: Whether to include base64-encoded image data (default: True)
            max_pixels: Optional maximum pixels (will downsample if needed)

        Returns:
            FileContent with image data
        """
        try:
            # Import PIL/Pillow for image processing
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow (PIL) is required for image handling. "
                "Install with: pip install Pillow"
            )

        provider = kwargs.get('provider', 'generic')
        detail = kwargs.get('detail', 'high')
        include_base64 = kwargs.get('include_base64', True)
        max_pixels = kwargs.get('max_pixels')

        try:
            # Open image
            with Image.open(path) as img:
                # Get image info
                width, height = img.size
                format_name = img.format or path.suffix[1:].upper()

                # Get additional metadata
                dpi = img.info.get('dpi', (72, 72))
                mode = img.mode  # RGB, RGBA, L, etc.

                # Check if downsampling needed
                should_downsample = False
                target_width, target_height = width, height

                if max_pixels and (width * height) > max_pixels:
                    should_downsample = True
                    # Calculate target dimensions
                    scale = (max_pixels / (width * height)) ** 0.5
                    target_width = int(width * scale)
                    target_height = int(height * scale)

                # Downsample if needed
                if should_downsample:
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    width, height = target_width, target_height

                # Estimate tokens
                estimated_tokens = estimate_image_tokens(width, height, provider, detail)

                # Read image bytes
                if include_base64:
                    import io
                    buffer = io.BytesIO()
                    # Save in original format or default to PNG
                    save_format = format_name if format_name in ['JPEG', 'PNG', 'GIF', 'WEBP'] else 'PNG'
                    img.save(buffer, format=save_format)
                    image_bytes = buffer.getvalue()
                else:
                    # Just read file bytes without processing
                    with open(path, 'rb') as f:
                        image_bytes = f.read()

            # Create ImageData
            image_data = ImageData(
                data=image_bytes,
                format=format_name.lower(),
                width=width,
                height=height,
                estimated_tokens=estimated_tokens,
                metadata={
                    'dpi': dpi,
                    'mode': mode,
                    'downsampled': should_downsample,
                    'provider': provider,
                    'detail': detail
                }
            )

            # Create content description
            content = f"Image: {path.name}\n"
            content += f"Format: {format_name}\n"
            content += f"Dimensions: {width}x{height} pixels\n"
            content += f"Total pixels: {width * height:,}\n"
            content += f"Megapixels: {image_data.megapixels:.2f} MP\n"
            content += f"Estimated tokens ({provider}): {estimated_tokens}\n"
            content += f"Mode: {mode}\n"
            if should_downsample:
                content += f"⚠️  Downsampled from original due to max_pixels limit\n"

            file_size = path.stat().st_size

            return FileContent(
                path=path,
                content=content,
                partial=False,
                file_size=file_size,
                character_count=len(content),
                estimated_tokens=0,  # Text tokens (minimal)
                images=[image_data],
                total_image_pixels=image_data.total_pixels,
                total_estimated_image_tokens=estimated_tokens,
                metadata={
                    'type': 'image',
                    'format': format_name.lower(),
                    'dimensions': (width, height),
                    'dpi': dpi,
                    'mode': mode,
                    'downsampled': should_downsample
                }
            )

        except Exception as e:
            logger.error(f"Failed to read image {path}: {e}")
            raise ValueError(f"Could not read image file: {e}")

    async def get_structure(self, path: Path, **kwargs) -> DocumentStructure:
        """
        Get structure of image file (minimal - just metadata).

        Args:
            path: File path

        Returns:
            DocumentStructure with image metadata
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow (PIL) is required for image handling")

        try:
            with Image.open(path) as img:
                width, height = img.size
                format_name = img.format or path.suffix[1:].upper()

                # Create single section for the image
                section = Section(
                    id="image",
                    title=f"Image: {path.name}",
                    level=1,
                    start_line=1,
                    end_line=1,
                    metadata={
                        'format': format_name,
                        'width': width,
                        'height': height,
                        'total_pixels': width * height,
                        'megapixels': (width * height) / (1024 * 1024)
                    }
                )

                return DocumentStructure(
                    type='image',
                    sections=[section],
                    metadata={
                        'format': format_name.lower(),
                        'dimensions': (width, height),
                        'file_size': path.stat().st_size
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get image structure for {path}: {e}")
            raise ValueError(f"Could not get image structure: {e}")

    async def read_section(self, path: Path, section_id: str, **kwargs) -> str:
        """
        Read a section of the image (not applicable for images).

        For images, this just returns the full image info.
        """
        content = await self.read(path, **kwargs)
        return content.content

    async def read_partial(
        self,
        path: Path,
        start_line: int,
        end_line: Optional[int] = None,
        **kwargs
    ) -> FileContent:
        """
        Partial read not applicable for images, returns full image.
        """
        return await self.read(path, **kwargs)

    def get_file_extension(self, path: Path) -> str:
        """Get clean file extension."""
        return path.suffix.lower()[1:] if path.suffix else ''
