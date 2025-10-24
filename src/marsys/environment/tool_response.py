"""Structured tool response format for complex returns with ordered content blocks."""

from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ToolResponseContent:
    """
    Represents a single content block (text or image) in tool response.

    This enables ordered sequences of text and images that map directly to
    LLM message content arrays.

    VALIDATION RULES:
    - type="text": MUST have 'text' field, CANNOT have image_path or image_data
    - type="image": MUST have exactly ONE of (image_path OR image_data), CANNOT have 'text'

    Examples:
        # Text content (string)
        ToolResponseContent(type="text", text="Hello world")

        # Text content (dict)
        ToolResponseContent(type="text", text={"key": "value"})

        # Image with base64 data
        ToolResponseContent(type="image", image_data="data:image/png;base64,iVBORw...")

        # Image with local path (converted to base64 on demand)
        ToolResponseContent(type="image", image_path="/path/to/image.png")
    """
    type: str  # "text" or "image"

    # For type="text" - string or dict
    text: Optional[Union[str, Dict]] = None

    # For type="image" - exactly ONE of these
    image_path: Optional[Union[str, Path]] = None  # Path to local image
    image_data: Optional[str] = None  # Base64 data URL (data:image/...;base64,...)

    def __post_init__(self):
        """Validate that exactly one content field is provided based on type."""
        if self.type == "text":
            # Text type: must have text, cannot have image fields
            if self.text is None:
                raise ValueError("ToolResponseContent with type='text' must provide 'text' field")
            if self.image_path is not None or self.image_data is not None:
                raise ValueError(
                    "ToolResponseContent with type='text' cannot have 'image_path' or 'image_data' fields. "
                    "Only 'text' field should be provided."
                )

        elif self.type == "image":
            # Image type: must have exactly ONE of image_path or image_data
            if self.text is not None:
                raise ValueError(
                    "ToolResponseContent with type='image' cannot have 'text' field. "
                    "Only 'image_path' or 'image_data' should be provided."
                )

            has_path = self.image_path is not None
            has_data = self.image_data is not None

            if not has_path and not has_data:
                raise ValueError(
                    "ToolResponseContent with type='image' must provide either 'image_path' or 'image_data'. "
                    "Exactly one is required."
                )
            if has_path and has_data:
                raise ValueError(
                    "ToolResponseContent with type='image' cannot have both 'image_path' and 'image_data'. "
                    "Provide only one."
                )

        else:
            raise ValueError(
                f"ToolResponseContent type must be 'text' or 'image', got: '{self.type}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to LLM message content format.

        Returns:
            For text: {"type": "text", "text": "..."}
            For image: {"type": "image_url", "image_url": {"url": "data:image/..."}}
        """
        if self.type == "text":
            if isinstance(self.text, dict):
                # Stringify dict for text content - use compressed JSON to save tokens
                import json
                return {
                    "type": "text",
                    "text": json.dumps(self.text, separators=(',', ':'))
                }
            else:
                return {
                    "type": "text",
                    "text": str(self.text) if self.text is not None else ""
                }

        elif self.type == "image":
            # Convert path to base64 if needed
            if self.image_data:
                image_url = self.image_data
            elif self.image_path:
                image_url = self._load_image_as_base64(self.image_path)
            else:
                # Should never reach here due to __post_init__ validation
                raise ValueError("Image content validation failed - no image data or path provided")

            return {
                "type": "image_url",
                "image_url": {"url": image_url}
            }

        else:
            raise ValueError(f"Unknown content type: {self.type}")

    def _load_image_as_base64(self, path: Union[str, Path]) -> str:
        """
        Load image from path and convert to base64 data URL.

        Args:
            path: Path to image file

        Returns:
            Base64 data URL (data:image/{format};base64,{encoded_data})

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        import base64
        from pathlib import Path

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # Determine format from extension
        ext = path.suffix.lower()[1:]  # Remove dot
        format_map = {
            'jpg': 'jpeg',
            'jpeg': 'jpeg',
            'png': 'png',
            'gif': 'gif',
            'webp': 'webp',
            'bmp': 'bmp',
            'tiff': 'tiff',
            'tif': 'tiff'
        }
        image_format = format_map.get(ext, 'png')

        # Read and encode
        with open(path, 'rb') as f:
            image_bytes = f.read()

        encoded = base64.b64encode(image_bytes).decode('ascii')
        return f"data:image/{image_format};base64,{encoded}"


@dataclass
class ToolResponse:
    """
    Structured response from tools with ordered content blocks.

    Supports three content formats:
    1. Simple string: For basic text responses
    2. Dictionary: For structured data (will be stringified by memory layer)
    3. List[ToolResponseContent]: For ordered text + image sequences (multimodal)

    The content format is preserved and directly maps to LLM message content format.

    Attributes:
        content: Tool response content (str, dict, or list of content blocks)
        metadata: Optional execution metadata (shown in tool message)

    Examples:
        # Simple string
        ToolResponse(content="File read successfully")

        # Dictionary
        ToolResponse(content={"status": "success", "count": 42})

        # Ordered text + images
        ToolResponse(
            content=[
                ToolResponseContent(type="text", text="Chapter 1"),
                ToolResponseContent(type="image", image_data="data:image/png;base64,..."),
                ToolResponseContent(type="text", text="Figure caption")
            ],
            metadata={"pages": "1-5", "images": 3}
        )
    """
    content: Union[str, Dict, List[ToolResponseContent]]
    metadata: Optional[Dict[str, Any]] = None

    def to_content_array(self) -> Union[str, Dict, List[Dict]]:
        """
        Convert content to LLM message format.

        This method prepares the content for insertion into LLM messages:
        - String content: returned as-is
        - Dict content: returned as-is (memory layer will stringify if needed)
        - List of ToolResponseContent: converted to typed array format

        Returns:
            Content in LLM-compatible format
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return self.content
        elif isinstance(self.content, list):
            # Convert list of ToolResponseContent to typed array
            return [block.to_dict() for block in self.content]
        else:
            return str(self.content)

    def has_images(self) -> bool:
        """
        Check if this ToolResponse contains any images.

        Returns:
            True if content list contains at least one image block
        """
        if isinstance(self.content, list):
            return any(block.type == "image" for block in self.content)
        return False

    def get_metadata_str(self) -> str:
        """
        Get metadata as string for tool message.

        The metadata is shown in the role="tool" message to provide context
        about the tool execution (file path, page numbers, image count, etc.).

        Returns:
            Metadata formatted as compressed JSON or default message
        """
        if self.metadata is None:
            return "Tool execution completed successfully."
        if isinstance(self.metadata, dict):
            import json
            # Use compressed JSON to save tokens
            return json.dumps(self.metadata, separators=(',', ':'))
        return str(self.metadata)
