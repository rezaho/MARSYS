"""Structured tool response format for complex returns with ordered content blocks."""

from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, model_validator


class ToolResponseContent(BaseModel):
    """
    Represents a single content block (text or image) in tool response.

    This enables ordered sequences of text and images that map directly to
    LLM message content arrays. The type is automatically inferred from which
    fields are provided.

    Examples:
        # Text content (string)
        ToolResponseContent(text="Hello world")

        # Text content (dict)
        ToolResponseContent(text={"key": "value"})

        # Image with base64 data
        ToolResponseContent(image_data="data:image/png;base64,iVBORw...")

        # Image with local path (converted to base64 on demand)
        ToolResponseContent(image_path="/path/to/image.png")
    """
    # Provide EITHER text OR (image_path/image_data), not both
    text: Optional[Union[str, Dict]] = None  # For text content
    image_path: Optional[Union[str, Path]] = None  # Path to local image
    image_data: Optional[str] = None  # Base64 data URL (data:image/...;base64,...)

    @model_validator(mode='after')
    def validate_content_type(self):
        """Validate that exactly one content type is provided and auto-infer type."""
        has_text = self.text is not None
        has_image_path = self.image_path is not None
        has_image_data = self.image_data is not None

        # Check for conflicting fields
        if has_text and (has_image_path or has_image_data):
            raise ValueError(
                "Cannot provide both 'text' and image fields ('image_path' or 'image_data'). "
                "Provide only text for text content, or only image fields for image content."
            )

        if has_image_path and has_image_data:
            raise ValueError(
                "Cannot provide both 'image_path' and 'image_data'. "
                "Provide only one image source."
            )

        # Check that at least one field is provided
        if not (has_text or has_image_path or has_image_data):
            raise ValueError(
                "Must provide either 'text' for text content, or 'image_path'/'image_data' for image content."
            )

        return self

    @property
    def type(self) -> str:
        """Auto-infer type based on which fields are provided."""
        if self.text is not None:
            return "text"
        else:
            return "image"

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


class ToolResponse(BaseModel):
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
                ToolResponseContent(text="Chapter 1"),
                ToolResponseContent(image_data="data:image/png;base64,..."),
                ToolResponseContent(text="Figure caption")
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

        Raises:
            TypeError: If list content doesn't contain ToolResponseContent objects
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return self.content
        elif isinstance(self.content, list):
            # List must contain ToolResponseContent objects for proper multimodal conversion
            if not self.content:  # Empty list is valid
                return []

            # Validate that all items are ToolResponseContent objects
            for i, item in enumerate(self.content):
                if not isinstance(item, ToolResponseContent):
                    raise TypeError(
                        f"When using list as content, all items must be ToolResponseContent objects. "
                        f"Item at index {i} is of type: {type(item).__name__}. "
                        f"Use ToolResponseContent to wrap your data: "
                        f"ToolResponseContent(text='...') for text, or "
                        f"ToolResponseContent(image_data='data:image/...') for images. "
                        f"Multiple ToolResponseContent objects create multimodal content (text + images)."
                    )

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
        about the tool execution. When no custom metadata is provided,
        generates an informative summary of the content to help the LLM
        understand what data will be provided by the user in subsequent messages.

        Note: Due to API constraints, tool results with multimodal content are
        split into two messages:
        1. role="tool": This metadata message (you are here)
        2. role="user": The actual content (will follow after all tool messages)

        Returns:
            Metadata formatted as compressed JSON or auto-generated summary
        """
        if self.metadata is not None:
            # Custom metadata provided - use it
            if isinstance(self.metadata, dict):
                import json
                return json.dumps(self.metadata, separators=(',', ':'))
            return str(self.metadata)

        # No custom metadata - generate informative summary based on content type
        # to help LLM understand what to expect in the user message
        if isinstance(self.content, str):
            # String content - show size and preview
            char_count = len(self.content)
            if char_count > 150:
                preview = self.content[:100].strip()
                return f"Tool execution successful. Returned {char_count} characters of text data. The user will provide the full content in a following message. Preview: {preview}..."
            else:
                return f"Tool execution successful. Returned {char_count} characters. The user will provide the content in a following message."

        elif isinstance(self.content, dict):
            # Dict content - show structure
            import json
            keys = list(self.content.keys())
            key_preview = ', '.join(f"'{k}'" for k in keys[:5])
            if len(keys) > 5:
                key_preview += f", ... ({len(keys)} total)"
            return f"Tool execution successful. Returned structured data with keys: [{key_preview}]. The user will provide the full data in a following message."

        elif isinstance(self.content, list):
            # List of ToolResponseContent - show detailed counts
            text_count = sum(1 for item in self.content if item.type == "text")
            image_count = sum(1 for item in self.content if item.type == "image")

            parts = []
            if text_count > 0:
                parts.append(f"{text_count} text block{'s' if text_count != 1 else ''}")
            if image_count > 0:
                parts.append(f"{image_count} image{'s' if image_count != 1 else ''}")

            content_desc = ' and '.join(parts)
            return f"Tool execution successful. Returned multimodal content ({content_desc}). The user will provide this content in a following message tagged with the tool_call_id."

        # Fallback
        return "Tool execution successful. The user will provide the results in a following message."
