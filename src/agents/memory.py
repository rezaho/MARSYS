import asyncio
import dataclasses
import json
import logging
import time
import uuid
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from src.models.models import BaseLLM  # Added ModelConfig
from src.models.models import BaseAPIModel, BaseVLM

# Import the new exception classes
from .exceptions import (
    AgentConfigurationError,
    MessageError,
    AgentFrameworkError,
)


# --- Structured Content Data Classes ---

@dataclasses.dataclass
class MessageContent:
    """Represents the structured content of a message."""
    thought: Optional[str] = None
    next_action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate next_action field."""
        if self.next_action is not None:
            valid_actions = {"call_tool", "invoke_agent", "final_response"}
            if self.next_action not in valid_actions:
                raise ValueError(
                    f"next_action must be one of {valid_actions}, got: {self.next_action}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.thought is not None:
            result["thought"] = self.thought
        if self.next_action is not None:
            result["next_action"] = self.next_action
        if self.action_input is not None:
            result["action_input"] = self.action_input
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageContent":
        """Create from dictionary format."""
        return cls(
            thought=data.get("thought"),
            next_action=data.get("next_action"),
            action_input=data.get("action_input")
        )


@dataclasses.dataclass
class ToolCallMsg:
    """Represents a tool call in a message."""
    id: str
    call_id: str
    type: str
    name: str
    arguments: str
    
    def __post_init__(self):
        """Validate tool call fields."""
        if not self.id:
            raise ValueError("Tool call id cannot be empty")
        if not self.call_id:
            raise ValueError("Tool call call_id cannot be empty")
        if not self.type:
            raise ValueError("Tool call type cannot be empty")
        if not self.name:
            raise ValueError("Tool call name cannot be empty")
        if not isinstance(self.arguments, str):
            raise ValueError("Tool call arguments must be a string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with OpenAI API."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallMsg":
        """Create from dictionary format (OpenAI API format)."""
        function_data = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            call_id=data.get("id", ""),  # Use id as call_id for compatibility
            type=data.get("type", "function"),
            name=function_data.get("name", ""),
            arguments=function_data.get("arguments", "{}")
        )


@dataclasses.dataclass
class AgentCallMsg:
    """Represents an agent invocation call in a message."""
    agent_name: str
    request: Any
    
    def __post_init__(self):
        """Validate agent call fields."""
        if not self.agent_name:
            raise ValueError("Agent call agent_name cannot be empty")
        if not isinstance(self.agent_name, str):
            raise ValueError("Agent call agent_name must be a string")
        if self.request is None:
            raise ValueError("Agent call request cannot be None")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "agent_name": self.agent_name,
            "request": self.request
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCallMsg":
        """Create from dictionary format."""
        return cls(
            agent_name=data.get("agent_name", ""),
            request=data.get("request")
        )


# --- Existing Classes ---


# --- Core Data Structures ---


@dataclasses.dataclass
class Message:
    """Represents a single message in a conversation, with a unique ID."""

    role: str
    content: Optional[Union[str, Dict[str, Any]]] = None  # Can be string or any dictionary
    message_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None  # For tool role (tool name) or assistant name (model name)
    tool_calls: Optional[List[ToolCallMsg]] = None  # For assistant requesting tool calls
    agent_calls: Optional[List[AgentCallMsg]] = None  # For assistant requesting agent invocations
    structured_data: Optional[Dict[str, Any]] = None  # For storing structured data when agent returns a dictionary from auto_run
    images: Optional[List[str]] = None  # For storing image paths/URLs for vision models

    def __post_init__(self):
        """Automatically convert dictionaries to proper dataclasses with validation."""
        # Convert tool_calls from List[Dict] to List[ToolCallMsg] if needed
        if self.tool_calls is not None:
            if isinstance(self.tool_calls, list):
                converted_tool_calls = []
                for i, tc in enumerate(self.tool_calls):
                    if isinstance(tc, dict):
                        try:
                            converted_tool_calls.append(ToolCallMsg.from_dict(tc))
                        except (ValueError, KeyError, TypeError) as e:
                            raise MessageError(
                                f"Failed to convert tool_calls[{i}] to ToolCallMsg: {e}",
                                message_id=self.message_id,
                                invalid_data=tc,
                                expected_format="ToolCallMsg with id, call_id, type, name, arguments fields"
                            )
                    elif isinstance(tc, ToolCallMsg):
                        converted_tool_calls.append(tc)
                    else:
                        raise MessageError(
                            f"tool_calls[{i}] must be dict or ToolCallMsg, got {type(tc).__name__}",
                            message_id=self.message_id,
                            invalid_data=tc,
                            expected_format="Dict or ToolCallMsg object"
                        )
                self.tool_calls = converted_tool_calls
            else:
                raise MessageError(
                    f"tool_calls must be a list, got {type(self.tool_calls).__name__}",
                    message_id=self.message_id,
                    invalid_data=self.tool_calls,
                    expected_format="List[ToolCallMsg] or List[Dict]"
                )

        # Convert agent_calls from List[Dict] to List[AgentCallMsg] if needed
        if self.agent_calls is not None:
            if isinstance(self.agent_calls, list):
                converted_agent_calls = []
                for i, ac in enumerate(self.agent_calls):
                    if isinstance(ac, dict):
                        try:
                            converted_agent_calls.append(AgentCallMsg.from_dict(ac))
                        except (ValueError, KeyError, TypeError) as e:
                            raise MessageError(
                                f"Failed to convert agent_calls[{i}] to AgentCallMsg: {e}",
                                message_id=self.message_id,
                                invalid_data=ac,
                                expected_format="AgentCallMsg with agent_name and request fields"
                            )
                    elif isinstance(ac, AgentCallMsg):
                        converted_agent_calls.append(ac)
                    else:
                        raise MessageError(
                            f"agent_calls[{i}] must be dict or AgentCallMsg, got {type(ac).__name__}",
                            message_id=self.message_id,
                            invalid_data=ac,
                            expected_format="Dict or AgentCallMsg object"
                        )
                self.agent_calls = converted_agent_calls
            else:
                raise MessageError(
                    f"agent_calls must be a list, got {type(self.agent_calls).__name__}",
                    message_id=self.message_id,
                    invalid_data=self.agent_calls,
                    expected_format="List[AgentCallMsg] or List[Dict]"
                )

        # Content can now be any dictionary - no automatic conversion to MessageContent
        # This allows specialized agents like InteractiveElementsAgent to return custom formats

        # Validate images field
        if self.images is not None:
            if not isinstance(self.images, list):
                raise MessageError(
                    f"images must be a list, got {type(self.images).__name__}",
                    message_id=self.message_id,
                    invalid_data=self.images,
                    expected_format="List[str] of image paths/URLs"
                )
            for i, img in enumerate(self.images):
                if not isinstance(img, str):
                    raise MessageError(
                        f"images[{i}] must be a string (path/URL), got {type(img).__name__}",
                        message_id=self.message_id,
                        invalid_data=img,
                        expected_format="String path or URL"
                    )

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Helper method to encode an image to base64 format for LLM APIs."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                encoded = base64.b64encode(image_data).decode("utf-8")
                
                # Determine MIME type based on file extension
                extension = Path(image_path).suffix.lower()
                if extension in [".jpg", ".jpeg"]:
                    mime_type = "image/jpeg"
                elif extension == ".png":
                    mime_type = "image/png"
                elif extension == ".gif":
                    mime_type = "image/gif"
                elif extension == ".webp":
                    mime_type = "image/webp"
                else:
                    # Default to PNG if unknown
                    mime_type = "image/png"
                
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            # Return placeholder or empty string if image can't be read
            logging.warning(f"Failed to encode image {image_path}: {e}")
            return ""

    def _is_agent_action_content(self, content_dict: Dict[str, Any]) -> bool:
        """
        Check if a content dictionary represents standard agent action format.
        
        Agent action content has the structure:
        - thought (optional): str
        - next_action (optional): one of "call_tool", "invoke_agent", "final_response"  
        - action_input (optional): dict
        
        Args:
            content_dict: Dictionary to check
            
        Returns:
            True if the dict appears to be agent action content, False otherwise
        """
        if not isinstance(content_dict, dict):
            return False
        
        # Check if it has the standard agent action keys
        agent_action_keys = {"thought", "next_action", "action_input"}
        content_keys = set(content_dict.keys())
        
        # If it only contains agent action keys (or subset), it's likely agent action content
        if content_keys <= agent_action_keys:
            # Additional validation: if next_action is present, it should be valid
            next_action = content_dict.get("next_action")
            if next_action is not None:
                valid_actions = {"call_tool", "invoke_agent", "final_response"}
                return next_action in valid_actions
            return True
        
        # If it has other keys beyond agent action keys, it's specialized content
        return False

    def to_llm_dict(self) -> Dict[str, Any]:
        """Converts the message to a dictionary format suitable for LLM APIs."""
        payload: Dict[str, Any] = {"role": self.role}

        # Content handling based on OpenAI spec
        if self.role == "assistant" and self.tool_calls:
            payload["content"] = None  # Must be null if tool_calls are present
        elif self.content is not None:
            # Handle structured content
            if isinstance(self.content, dict):
                # Check if it's a MessageContent-like dict (has standard agent action keys)
                if self._is_agent_action_content(self.content):
                    # Standard agent action content - serialize as JSON string
                    payload["content"] = json.dumps(self.content, ensure_ascii=False, indent=2)
                else:
                    # Specialized content (like InteractiveElementsAgent response) - serialize as JSON string
                    payload["content"] = json.dumps(self.content, ensure_ascii=False, indent=2)
            else:
                # Check if we have images to create multimodal content
                if self.images and len(self.images) > 0:
                    # Create multimodal content with text and images
                    content_parts = []
                    
                    # Add text content if present
                    if str(self.content).strip():
                        content_parts.append({
                            "type": "text",
                            "text": str(self.content)
                        })
                    
                    # Add image content with proper base64 encoding
                    for image_path in self.images:
                        encoded_image = self._encode_image_to_base64(image_path)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": encoded_image}
                        })
                    
                    payload["content"] = content_parts
                else:
                    payload["content"] = str(self.content)
        elif self.role in [
            "user",
            "system",
            "tool",
        ]:  # These roles generally require content
            # Check if we have images without text content
            if self.images and len(self.images) > 0:
                content_parts = []
                for image_path in self.images:
                    encoded_image = self._encode_image_to_base64(image_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": encoded_image}
                    })
                payload["content"] = content_parts
            else:
                payload["content"] = ""  # Keep as empty string if no content or images
        # else: content can remain None for assistant without tool_calls (though less common)

        # Add optional fields if they exist
        if self.name:
            payload["name"] = self.name
        if self.tool_calls and self.role == "assistant":  # tool_calls are specific to assistant role
            # Convert ToolCallMsg objects back to dicts for LLM API
            payload["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        # Special handling for tool messages - extract tool_call_id from content
        if self.role == "tool" and self.content:
            try:
                # Try to parse content as JSON to extract tool_call_id
                if isinstance(self.content, str):
                    content_data = json.loads(self.content)
                    if isinstance(content_data, dict) and "tool_call_id" in content_data:
                        payload["tool_call_id"] = content_data["tool_call_id"]
                        
                        # Handle content with images
                        output_text = content_data.get("output", "")
                        if self.images and len(self.images) > 0:
                            # Create multimodal content with text and images
                            content_parts = []
                            
                            # Add text content if present
                            if output_text.strip():
                                content_parts.append({
                                    "type": "text",
                                    "text": output_text
                                })
                            
                            # Add image content with proper base64 encoding
                            for image_path in self.images:
                                encoded_image = self._encode_image_to_base64(image_path)
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": encoded_image}
                                })
                            
                            payload["content"] = content_parts
                        else:
                            # No images, just set text content
                            payload["content"] = output_text
                elif isinstance(self.content, dict) and "tool_call_id" in self.content:
                    # Content is already a dict with tool_call_id
                    payload["tool_call_id"] = self.content["tool_call_id"]
                    output_text = self.content.get("output", "")
                    if self.images and len(self.images) > 0:
                        # Create multimodal content with text and images
                        content_parts = []
                        
                        # Add text content if present
                        if output_text.strip():
                            content_parts.append({
                                "type": "text",
                                "text": output_text
                            })
                        
                        # Add image content with proper base64 encoding
                        for image_path in self.images:
                            encoded_image = self._encode_image_to_base64(image_path)
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": encoded_image}
                            })
                        
                        payload["content"] = content_parts
                    else:
                        # No images, just set text content
                        payload["content"] = output_text
            except (json.JSONDecodeError, TypeError):
                # If content is not JSON or doesn't have expected structure, keep as is
                pass
        
        # Note: agent_calls is NOT included in LLM format as it's not part of OpenAI API
        # It should be converted to JSON content by message transformers when needed

        # Filter out keys with None values, except for 'content: None' for assistant with tool_calls
        final_payload = {}
        for k, v in payload.items():
            if k == "content" and self.role == "assistant" and self.tool_calls:
                final_payload[k] = None  # Explicitly keep content: null
            elif v is not None:
                final_payload[k] = v
        return final_payload

    def to_action_dict(self) -> Optional[Dict[str, Any]]:
        """
        Converts the Message to an action dictionary format expected by auto_run.
        
        This method checks the Message attributes in priority order and returns
        the appropriate action dictionary format, or None if no valid action is found.
        
        Priority order:
        1. tool_calls attribute -> call_tool action
        2. agent_calls attribute -> invoke_agent action  
        3. content with next_action='final_response' -> final_response action
        4. None if no valid action found
        
        Returns:
            Dictionary with 'thought', 'next_action', and 'action_input' keys, or None
        """
        # Extract thought from content if it's a string or dict with thought
        thought = None
        if isinstance(self.content, str):
            thought = self.content
        elif isinstance(self.content, dict) and "thought" in self.content:
            thought = self.content["thought"]
        
        # Priority 1: Check for tool_calls attribute
        if self.tool_calls:
            # Convert ToolCallMsg objects to dicts
            tc_dicts = []
            for tc in self.tool_calls:
                if hasattr(tc, 'to_dict'):
                    tc_dicts.append(tc.to_dict())
                else:
                    # Already a dict
                    tc_dicts.append(tc)
            
            return {
                "thought": thought,
                "next_action": "call_tool",
                "action_input": {"tool_calls": tc_dicts}
            }
        
        # Priority 2: Check for agent_calls attribute
        if hasattr(self, 'agent_calls') and self.agent_calls:
            # For now, we only support single agent invocation, so take the first one
            # In the future, this could be extended to support multiple agent calls
            first_agent_call_msg = self.agent_calls[0]
            
            # Convert AgentCallMsg to dict
            if hasattr(first_agent_call_msg, 'to_dict'):
                ac_dict = first_agent_call_msg.to_dict()
            else:
                # Already a dict
                ac_dict = first_agent_call_msg
            
            return {
                "thought": thought,
                "next_action": "invoke_agent",
                "action_input": ac_dict
            }
        
        # Priority 3: Check for final_response in content
        if (isinstance(self.content, dict) and 
            self.content.get('next_action') == 'final_response'):
            
            return self.content  # Return the content dict as-is
        
        # Priority 4: Check if content is agent action format (handles legacy MessageContent)
        if isinstance(self.content, dict) and self._is_agent_action_content(self.content):
            return self.content  # Return the agent action content as-is
        
        # No valid action found
        return None

    @classmethod
    def from_response_dict(
        cls,
        response_dict: Dict[str, Any],
        default_id: Optional[str] = None,
        default_role: str = "assistant",
        default_name: Optional[str] = None,
        processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> "Message":
        """
        Creates a Message instance from a model response dictionary.
        
        This method handles various response formats from different model providers
        and ensures consistent Message creation. The Message.__post_init__ will
        automatically convert any dict-based tool_calls or agent_calls to proper dataclasses.
        
        Args:
            response_dict: Dictionary containing the model response
            default_id: Default message ID if not present in response
            default_role: Default role if not present in response
            default_name: Default name if not present in response
            processor: Optional callable to transform the response dict before processing
            
        Returns:
            Message instance with properly converted structured types
        """
        # Apply processor if provided
        if processor:
            response_dict = processor(response_dict)
            
        # Extract basic fields
        role = response_dict.get("role", default_role)
        content = response_dict.get("content")
        name = response_dict.get("name", default_name)
        message_id = response_dict.get("message_id", default_id or str(uuid.uuid4()))
        
        # Extract tool_calls (will be auto-converted by __post_init__)
        tool_calls = response_dict.get("tool_calls")
        
        # Extract agent_calls (will be auto-converted by __post_init__)
        agent_calls = response_dict.get("agent_calls")
        
        # Extract structured_data if present
        structured_data = response_dict.get("structured_data")
        
        # Create Message - __post_init__ will handle all conversions automatically
        return cls(
            role=role,
            content=content,
            message_id=message_id,
            name=name,
            tool_calls=tool_calls,  # Can be List[Dict] or List[ToolCallMsg]
            agent_calls=agent_calls,  # Can be List[Dict] or List[AgentCallMsg]
            structured_data=structured_data,
            images=response_dict.get("images"),
        )


# --- Memory Classes ---


class BaseMemory(ABC):
    """Abstract base class for agent memory modules."""

    def __init__(self, memory_type: str) -> None:
        """
        Initializes BaseMemory.

        Args:
            memory_type: String identifier for the type of memory (e.g., 'conversation_history').
        """
        self.memory_type = memory_type

    @abstractmethod
    def update_memory(
        self,
        message: Optional[Message] = None,
        *,  # Make subsequent arguments keyword-only if message is not None
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,  # Updated type
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,  # For image attachments
    ) -> None:
        """
        Updates memory with a new message.

        Args:
            message: Optional Message object to add directly.
            role: Message role (user, assistant, tool, etc.).
            content: Message content.
            message_id: Optional message ID (auto-generated if not provided).
            name: Optional name field.
            tool_calls: Optional list of tool calls (can be dicts or ToolCallMsg objects).
            agent_calls: Optional list of agent calls (can be list of dicts or AgentCallMsg objects).
            images: Optional list of image paths/URLs for vision models.
        """
        if message is not None:
            # Use the provided message directly - __post_init__ ensures proper conversion
            self._add_message(message)
        else:
            # Create new message - __post_init__ will handle any dict-to-dataclass conversions
            new_message = Message(
                role=role or "user",
                content=content,
                message_id=message_id or str(uuid.uuid4()),
                name=name,
                tool_calls=tool_calls,  # Can be List[Dict] or List[ToolCallMsg]
                agent_calls=agent_calls,  # Can be List[Dict] or List[AgentCallMsg]
                images=images,  # List of image paths/URLs
            )
            self._add_message(new_message)

    @abstractmethod
    def replace_memory(
        self,
        idx: int,
        message: Optional[Message] = None,
        *,  # Make subsequent arguments keyword-only if message is not None
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,  # Updated type
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,  # For image attachments
    ) -> None:
        """
        Replaces a message at a specific index.

        Args:
            idx: Index of the message to replace.
            message: Optional Message object to use as replacement.
            role: Message role.
            content: Message content.
            message_id: Message ID.
            name: Optional name field.
            tool_calls: Optional list of tool calls.
            agent_calls: Optional list of agent call information.
            images: Optional list of image paths/URLs for vision models.
        """
        raise NotImplementedError("replace_memory must be implemented in subclasses.")

    @abstractmethod
    def delete_memory(self, idx: int) -> None:
        """Deletes information from the memory by index."""
        raise NotImplementedError("delete_memory must be implemented in subclasses.")

    @abstractmethod
    def retrieve_recent(self, n: int = 1) -> List[Message]:
        """Retrieves the 'n' most recent Message objects."""
        raise NotImplementedError("retrieve_recent must be implemented in subclasses.")

    @abstractmethod
    def retrieve_all(self) -> List[Message]:
        """Retrieves all Message objects."""
        raise NotImplementedError("retrieve_all must be implemented in subclasses.")

    @abstractmethod
    def retrieve_by_id(self, message_id: str) -> Optional[Message]:
        """Retrieves a specific Message object by its ID."""
        raise NotImplementedError("retrieve_by_id must be implemented in subclasses.")

    @abstractmethod
    def remove_by_id(self, message_id: str) -> bool:
        """Removes a specific Message object by its ID. Returns True if removed, False if not found."""
        raise NotImplementedError("remove_by_id must be implemented in subclasses.")

    @abstractmethod
    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Message]:
        """Retrieves Message objects filtered by role."""
        raise NotImplementedError("retrieve_by_role must be implemented in subclasses.")

    @abstractmethod
    def reset_memory(self) -> None:
        """Clears all entries from the memory."""
        raise NotImplementedError("reset_memory must be implemented in subclasses.")

    @abstractmethod
    def to_llm_format(
        self, 
        transform_message: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Formats the memory content into a list of dictionaries suitable for LLM input.
        
        Args:
            transform_message: Optional callable to transform each message dict before returning.
                              Takes a message dict and returns a transformed dict.
        """
        raise NotImplementedError("to_llm_format must be implemented in subclasses.")

    @abstractmethod
    def set_message_transformers(
        self,
        from_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        to_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Sets transformation functions for converting messages to/from LLM format.
        
        Args:
            from_llm_transformer: Callable to transform LLM response to Message-compatible format
            to_llm_transformer: Callable to transform Message dict to LLM-compatible format
        """
        raise NotImplementedError("set_message_transformers must be implemented in subclasses.")


class ConversationMemory(BaseMemory):
    """
    Memory module that stores conversation history as a list of Message objects.
    """

    def __init__(
        self, 
        description: Optional[str] = None,
        from_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        to_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initializes ConversationMemory.

        Args:
            description: Optional content for an initial system message.
            from_llm_transformer: Callable to transform LLM response to Message-compatible format
            to_llm_transformer: Callable to transform Message dict to LLM-compatible format
        """
        super().__init__(memory_type="conversation_history")
        self.memory: List[Message] = []
        self.from_llm_transformer = from_llm_transformer
        self.to_llm_transformer = to_llm_transformer
        if description:
            self.memory.append(Message(role="system", content=description))

    def set_message_transformers(
        self,
        from_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        to_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Sets transformation functions for converting messages to/from LLM format."""
        if from_llm_transformer is not None:
            self.from_llm_transformer = from_llm_transformer
        if to_llm_transformer is not None:
            self.to_llm_transformer = to_llm_transformer

    def update_memory_from_llm_response(
        self,
        llm_response_dict: Dict[str, Any],
        message_id: Optional[str] = None,
    ) -> None:
        """
        Adds a message from LLM response, applying transformation if set.
        
        Args:
            llm_response_dict: Raw response dict from LLM
            message_id: Optional message ID to use
        """
        # Apply transformation if available
        if self.from_llm_transformer:
            transformed_dict = self.from_llm_transformer(llm_response_dict)
        else:
            transformed_dict = llm_response_dict
        
        # Create Message from transformed dict
        message = Message.from_response_dict(transformed_dict, default_id=message_id)
        self.memory.append(message)

    def update_memory(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """
        Appends a message to the conversation history.
        If `message` object is provided, it's used directly.
        Otherwise, a new Message is created from `role` and `content`, etc.
        `message_id` can be provided to use an existing ID, otherwise a new one is generated.
        """
        if message:
            # If a full Message object is provided, ensure its ID is unique if not already set
            # or if we want to enforce new IDs for additions. For now, trust the provided ID.
            self.memory.append(message)
        elif (
            role is not None
        ):  # content can be None for assistant messages with tool_calls or agent_calls
            new_message = Message(
                role=role,
                content=content,
                message_id=message_id
                or str(uuid.uuid4()),  # Use provided or generate new
                name=name,
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                images=images,
            )
            self.memory.append(new_message)
        else:
            raise MessageError(
                "Either a Message object or role must be provided to update_memory.",
                agent_name="ConversationMemory",
                validation_path="update_memory.message_or_role_validation"
            )

    def replace_memory(
        self,
        idx: int,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """
        Replaces the message at the specified index.
        """
        if not (0 <= idx < len(self.memory)):
            raise IndexError("Memory index out of range.")

        if message:
            self.memory[idx] = message
        elif role is not None:
            # If message_id is not provided, it defaults to a new UUID,
            # effectively giving the replaced message a new ID.
            # If an ID is provided, it uses that.
            self.memory[idx] = Message(
                role=role,
                content=content,
                message_id=message_id or str(uuid.uuid4()),
                name=name,
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                images=images,
            )
        else:
            raise MessageError(
                "Either a Message object or role must be provided to replace_memory.",
                agent_name="ConversationMemory",
                validation_path="replace_memory.message_or_role_validation"
            )

    def delete_memory(self, idx: int) -> None:
        """
        Deletes the message at the specified index.
        """
        if 0 <= idx < len(self.memory):
            del self.memory[idx]
        else:
            raise IndexError("Memory index out of range.")

    def retrieve_recent(self, n: int = 1) -> List[Message]:
        """
        Retrieves the 'n' most recent Message objects.
        """
        return self.memory[-n:] if n > 0 else []

    def retrieve_all(self) -> List[Message]:
        """Retrieves all Message objects in the history."""
        return list(self.memory)  # Return a copy

    def retrieve_by_id(self, message_id: str) -> Optional[Message]:
        """Retrieves a specific Message object by its ID."""
        for msg in self.memory:
            if msg.message_id == message_id:
                return msg
        return None

    def remove_by_id(self, message_id: str) -> bool:
        """Removes a specific Message object by its ID. Returns True if removed, False if not found."""
        for i, msg in enumerate(self.memory):
            if msg.message_id == message_id:
                del self.memory[i]
                return True
        return False

    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Message]:
        """
        Retrieves Message objects filtered by role, optionally limited to the most recent 'n'.
        """
        filtered = [m for m in self.memory if m.role == role]
        return filtered[-n:] if n else filtered

    def reset_memory(self) -> None:
        """Clears the conversation history, keeping the system prompt Message object if it exists."""
        system_message: Optional[Message] = None
        if self.memory and self.memory[0].role == "system":
            system_message = self.memory[0]
        self.memory.clear()
        if system_message:
            self.memory.append(system_message)

    def to_llm_format(
        self, 
        transform_message: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Converts stored Message objects to a list of dictionaries suitable for LLM input.
        Uses instance transformer if no override provided.
        """
        transformer = transform_message or self.to_llm_transformer
        messages = [msg.to_llm_dict() for msg in self.memory]
        
        if transformer:
            return [transformer(msg) for msg in messages]
        return messages


class KGMemory(BaseMemory):
    """
    Memory module storing knowledge as timestamped (Subject, Predicate, Object) triplets.
    Requires a language model instance to extract facts from text.
    Facts are converted to Message objects upon retrieval.
    """

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        description: Optional[str] = None,  # Renamed system_prompt to description
    ) -> None:
        super().__init__(memory_type="kg")
        self.model = model
        self.kg: List[Dict[str, Any]] = (
            []
        )  # Stores raw fact dicts: {role, subject, predicate, object, timestamp}
        if description:
            # Store description as an initial fact-like entry if desired
            self.add_fact(
                role="system",
                subject="system_configuration",
                predicate="has_initial_description",  # Renamed
                obj=description,
            )

    def add_fact(self, role: str, subject: str, predicate: str, obj: str) -> None:
        """
        Adds a new fact (triplet) to the knowledge graph.
        This is the primary method for adding structured KG data.
        """
        timestamp = time.time()
        self.kg.append(
            {
                "role": role,  # Role associated with the source/reason for this fact
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "timestamp": timestamp,
                "message_id": str(
                    uuid.uuid4()
                ),  # Assign a unique ID to each fact entry
            }
        )

    def update_memory(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,  # Ignored for new facts, new ID generated
        name: Optional[
            str
        ] = None,  # Can be used as part of the fact's role or metadata
        tool_calls: Optional[
            List[ToolCallMsg]
        ] = None,  # Generally not applicable to KG facts
        agent_calls: Optional[List[AgentCallMsg]] = None,  # Generally not applicable to KG facts
        images: Optional[List[str]] = None,  # Generally not applicable to KG facts
    ) -> None:
        """
        Adds information to KG memory. If Message object or role/content is provided,
        it attempts to extract facts from the content.
        """
        text_to_extract_from = None
        origin_role = "user"  # Default role for extracted facts if not specified

        if message:
            text_to_extract_from = message.content
            origin_role = message.role
        elif role and content:
            text_to_extract_from = content
            origin_role = role

        if text_to_extract_from:
            logging.info(
                f"KGMemory attempting to extract facts from text (role: {origin_role}): {text_to_extract_from[:100]}..."
            )
            # This is an async call, but update_memory is sync.
            # For simplicity here, we'll log and skip async extraction.
            # In a real scenario, this might need to be handled differently,
            # e.g., by making update_memory async or queueing extraction.
            # For now, we'll rely on explicit calls to extract_and_update_from_text.
            asyncio.create_task(
                self.extract_and_update_from_text(
                    text_to_extract_from, role=origin_role
                )
            )
            logging.debug("Fact extraction from Message content scheduled.")
        else:
            logging.warning(
                "KGMemory.update_memory called without Message object or role/content for fact extraction."
            )

    def replace_memory(
        self,
        idx: int,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """
        Replaces a fact at a given index. If Message or role/content is provided,
        it attempts to extract new fact(s) and replace the old one.
        This is a simplified implementation; replacing one fact with potentially multiple
        extracted facts is complex. For now, it will log a warning.
        A more robust implementation would delete the old fact and add new ones.
        """
        if not (0 <= idx < len(self.kg)):
            raise IndexError("KG index out of range.")

        logging.warning(
            f"KGMemory.replace_memory called for index {idx}. "
            "This operation is complex for KG. Consider delete and add_fact/extract_and_update."
        )
        # Simplified: delete and then try to update (which might extract)
        del self.kg[idx]
        self.update_memory(message=message, role=role, content=content, name=name)

    def delete_memory(self, idx: int) -> None:
        if 0 <= idx < len(self.kg):
            del self.kg[idx]
        else:
            raise IndexError("KG index out of range.")

    def _fact_to_message(self, fact_dict: Dict[str, Any]) -> Message:
        """Converts a raw KG fact dictionary to a Message object."""
        content = f"Fact ({fact_dict['role']}): {fact_dict['subject']} {fact_dict['predicate']} {fact_dict['object']}."
        return Message(
            role=fact_dict["role"],  # Or a generic "knowledge" role
            content=content,
            message_id=fact_dict.get(
                "message_id", str(uuid.uuid4())
            ),  # Use stored ID or generate
        )

    def retrieve_recent(self, n: int = 1) -> List[Message]:
        if n <= 0:
            return []
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"], reverse=True)
        return [self._fact_to_message(fact) for fact in sorted_kg[:n]]

    def retrieve_all(self) -> List[Message]:
        # Sort by timestamp before converting to ensure order if needed, though not strictly required by BaseMemory
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"])
        return [self._fact_to_message(fact) for fact in sorted_kg]

    def retrieve_by_id(self, message_id: str) -> Optional[Message]:
        """Retrieves a specific KG fact Message by its stored message_id."""
        for fact_dict in self.kg:
            if fact_dict.get("message_id") == message_id:
                return self._fact_to_message(fact_dict)
        logging.debug(f"KGMemory: Fact with message_id '{message_id}' not found.")
        return None

    def remove_by_id(self, message_id: str) -> bool:
        """Removes a specific KG fact by its ID. Returns True if removed, False if not found."""
        for i, fact_dict in enumerate(self.kg):
            if fact_dict.get("message_id") == message_id:
                del self.kg[i]
                return True
        return False

    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Message]:
        filtered = [fact for fact in self.kg if fact["role"] == role]
        filtered = sorted(
            filtered, key=lambda x: x["timestamp"], reverse=True
        )  # Most recent first
        if n is not None and n > 0:
            filtered = filtered[:n]
        return [self._fact_to_message(fact) for fact in filtered]

    def reset_memory(self) -> None:
        self.kg.clear()
        # Re-add system prompt if it was part of the initial setup logic
        # This depends on how system_prompt was handled in __init__
        # For now, simple clear. If system_prompt was stored as a fact, it's gone.

    def to_llm_format(
        self, 
        transform_message: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Formats all facts in the KG into LLM message dictionaries."""
        messages = self.retrieve_all()  # Gets List[Message]
        message_dicts = [msg.to_llm_dict() for msg in messages]
        
        if transform_message:
            return [transform_message(msg) for msg in message_dicts]
        return message_dicts

    async def extract_and_update_from_text(
        self, input_text: str, role: str = "user"
    ) -> List[Dict[str, str]]:  # Returns raw extracted fact dicts
        """
        Uses the associated LLM to extract facts from input text and adds them to the KG.

        Args:
            input_text: The text to extract facts from.
            role: The role to associate with the extracted facts (default: 'user').

        Returns:
            A list of the raw fact dictionaries extracted by the LLM.
        """
        extraction_prompt = (
            "Extract all knowledge graph facts from the following text. "
            "Return a JSON list of triplets, where each triplet is a dict with keys: subject, predicate, object. "
            'Example: [{"subject": "Paris", "predicate": "is the capital of", "object": "France"}, ...]'
            " If no facts are found, return an empty list []."
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": role, "content": input_text},
        ]
        extracted_facts: List[Dict[str, str]] = []
        valid_facts_added = 0
        try:
            result: Union[Dict, List[Dict], str] = self.model.run(
                messages=messages, json_mode=True
            )

            parsed_result: Any
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    logging.warning(
                        f"KG extraction result was string but not valid JSON: {result}"
                    )
                    parsed_result = None
            else:
                parsed_result = result

            if isinstance(parsed_result, list):
                extracted_facts = parsed_result
            elif parsed_result is not None:
                logging.warning(f"KG extraction result was not a list: {parsed_result}")

            for fact in extracted_facts:
                if (
                    isinstance(fact, dict)
                    and "subject" in fact
                    and "predicate" in fact
                    and "object" in fact
                ):
                    self.update_memory(
                        role,
                        str(fact["subject"]),
                        str(fact["predicate"]),
                        str(fact["object"]),
                    )
                    valid_facts_added += 1
                else:
                    logging.warning(
                        f"Skipping invalid fact format during KG extraction: {fact}"
                    )
            logging.info(f"Extracted and added {valid_facts_added} facts to KG memory.")

        except Exception as e:
            logging.error(f"Error during KG fact extraction: {e}")

        return extracted_facts


class MemoryManager:
    """
    Factory and manager for creating and interacting with memory modules.

    Delegates operations to the appropriate underlying memory module instance
    (e.g., ConversationMemory, KGMemory) based on the specified `memory_type`.
    """

    def __init__(
        self,
        memory_type: str = "conversation_history",
        description: Optional[str] = None,
        model: Optional[Union["BaseLLM", "BaseVLM"]] = None,
        input_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_processor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Initialize the MemoryManager with a specific memory strategy.

        Args:
            memory_type: Type of memory to use ("conversation_history" or "kg").
            description: Initial system description/prompt.
            model: Language model instance (required for KG memory).
            input_processor: Optional function to transform messages from LLM format before storage.
            output_processor: Optional function to transform messages to LLM format before sending.
        """
        self.memory_type = memory_type
        self.memory_module: BaseMemory
        self._input_processor = input_processor
        self._output_processor = output_processor

        if memory_type == "conversation_history":
            self.memory_module = ConversationMemory(
                description=description,
                from_llm_transformer=input_processor,
                to_llm_transformer=output_processor,
            )
        elif memory_type == "kg":
            if model is None:
                raise AgentConfigurationError(
                    "KGMemory requires a 'model' instance for fact extraction.",
                    agent_name="MemoryManager",
                    config_key="model",
                    config_value=None
                )
            self.memory_module = KGMemory(model=model, description=description)
        else:
            raise AgentConfigurationError(
                f"Unknown memory_type: {memory_type}",
                agent_name="MemoryManager", 
                config_key="memory_type",
                config_value=memory_type
            )

    def set_message_transformers(
        self,
        from_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        to_llm_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Delegates to the underlying memory module's set_message_transformers."""
        self.memory_module.set_message_transformers(from_llm_transformer, to_llm_transformer)

    def update_from_response(
        self,
        llm_response: Dict[str, Any],
        message_id: Optional[str] = None,
        default_role: str = "assistant",
        default_name: Optional[str] = None,
    ) -> None:
        """
        Updates memory with a response from the LLM, applying input transformation if configured.
        
        This method is specifically designed to handle responses from language models,
        applying any necessary transformations before storing the message.
        
        Args:
            llm_response: Raw response dictionary from the LLM
            message_id: Optional message ID to use
            default_role: Default role if not in response
            default_name: Default name if not in response
        """
        # Apply transformation if processor is configured
        if self._input_processor:
            transformed_response = self._input_processor(llm_response)
        else:
            transformed_response = llm_response
            
        # Create Message from transformed response
        message = Message.from_response_dict(
            transformed_response,
            default_id=message_id,
            default_role=default_role,
            default_name=default_name
        )
        
        # Store in memory
        self.memory_module.update_memory(message=message)

    def update_memory(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """Delegates to the underlying memory module's update_memory."""
        self.memory_module.update_memory(
            message=message,
            role=role,
            content=content,
            message_id=message_id,
            name=name,
            tool_calls=tool_calls,
            agent_calls=agent_calls,
            images=images,
        )

    def replace_memory(
        self,
        idx: int,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCallMsg]] = None,
        agent_calls: Optional[List[AgentCallMsg]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """Delegates to the underlying memory module's replace_memory."""
        self.memory_module.replace_memory(
            idx=idx,
            message=message,
            role=role,
            content=content,
            message_id=message_id,
            name=name,
            tool_calls=tool_calls,
            agent_calls=agent_calls,
            images=images,
        )

    def delete_memory(self, idx: int) -> None:
        """Delegates to the underlying memory module's delete_memory."""
        self.memory_module.delete_memory(idx)

    def retrieve_recent(self, n: int = 1) -> List[Message]:
        """Delegates to the underlying memory module's retrieve_recent."""
        return self.memory_module.retrieve_recent(n)

    def retrieve_all(self) -> List[Message]:
        """Delegates to the underlying memory module's retrieve_all."""
        return self.memory_module.retrieve_all()

    def retrieve_by_id(self, message_id: str) -> Optional[Message]:
        """Delegates to the underlying memory module's retrieve_by_id."""
        return self.memory_module.retrieve_by_id(message_id)

    def remove_by_id(self, message_id: str) -> bool:
        """Delegates to the underlying memory module's remove_by_id."""
        return self.memory_module.remove_by_id(message_id)

    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Message]:
        """Delegates to the underlying memory module's retrieve_by_role."""
        # All BaseMemory implementations should now have retrieve_by_role
        return self.memory_module.retrieve_by_role(role, n)

    def reset_memory(self) -> None:
        """Delegates to the underlying memory module's reset_memory."""
        return self.memory_module.reset_memory()

    def to_llm_format(self) -> List[Dict[str, Any]]:
        """
        Get all messages in LLM-compatible format, applying output transformation if configured.
        
        Returns:
            List of message dictionaries ready for LLM consumption.
        """
        messages = self.memory_module.retrieve_all()
        llm_messages = []
        
        for msg in messages:
            msg_dict = msg.to_llm_dict()
            
            # Apply transformation if processor is configured
            if self._output_processor:
                transformed_dict = self._output_processor(msg_dict)
            else:
                transformed_dict = msg_dict
                
            llm_messages.append(transformed_dict)
            
        return llm_messages

    def extract_and_update_from_text(
        self, *args: Any, **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Delegates to the underlying KGMemory's extract_and_update_from_text.

        Raises:
            NotImplementedError: If the memory type is not 'kg'.
        """
        if isinstance(self.memory_module, KGMemory):
            return self.memory_module.extract_and_update_from_text(*args, **kwargs)
        else:
            raise NotImplementedError(
                "extract_and_update_from_text is only available for KGMemory."
            )

    ### Helper Extract Prompt Method
    def _extract_prompt_and_context(self, prompt: Any) -> tuple[str, List[Message]]:
        """
        Returns (prompt_text, passed_context_messages).

        • If `prompt` is a dict:
            – take 'passed_referenced_context' list if present;
            – use the 'prompt' value (JSON-dump it if it is itself a dict);
            – if 'prompt' missing stringify the dict minus 'passed_referenced_context'.
        • Otherwise cast `prompt` to str and return an empty context list.
        """
        if isinstance(prompt, dict):
            context = prompt.get("passed_referenced_context", []) or []
            raw = prompt.get("prompt")
            if raw is None:
                raw = {
                    k: v for k, v in prompt.items() if k != "passed_referenced_context"
                }
            if isinstance(raw, dict):
                raw = json.dumps(raw, ensure_ascii=False, indent=2)
            return str(raw), context
        return str(prompt), []
