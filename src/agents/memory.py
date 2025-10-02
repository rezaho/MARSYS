import asyncio
import base64
import dataclasses
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from src.models.models import (
    BaseAPIModel,
    BaseLLM,  # Added ModelConfig
    BaseVLM,
)
from src.models.response_models import HarmonizedResponse

# Import the new exception classes
from .exceptions import (
    AgentConfigurationError,
    AgentFrameworkError,
    MessageError,
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
    tool_call_id: Optional[str] = None  # For tool response messages, links to the original tool call

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


    # def to_action_dict(self) -> Optional[Dict[str, Any]]:
    #     """
    #     Converts the Message to an action dictionary format expected by auto_run.
        
    #     This method checks the Message attributes in priority order and returns
    #     the appropriate action dictionary format, or None if no valid action is found.
        
    #     Priority order:
    #     1. tool_calls attribute -> call_tool action
    #     2. agent_calls attribute -> invoke_agent action  
    #     3. content with next_action='final_response' -> final_response action
    #     4. None if no valid action found
        
    #     Returns:
    #         Dictionary with 'thought', 'next_action', and 'action_input' keys, or None
    #     """
    #     # Extract thought from content if it's a string or dict with thought
    #     thought = None
    #     if isinstance(self.content, str):
    #         thought = self.content
    #     elif isinstance(self.content, dict) and "thought" in self.content:
    #         thought = self.content["thought"]
        
    #     # Priority 1: Check for tool_calls attribute
    #     if self.tool_calls:
    #         # Convert ToolCallMsg objects to dicts
    #         tc_dicts = []
    #         for tc in self.tool_calls:
    #             if hasattr(tc, 'to_dict'):
    #                 tc_dicts.append(tc.to_dict())
    #             else:
    #                 # Already a dict
    #                 tc_dicts.append(tc)
            
    #         return {
    #             "thought": thought,
    #             "next_action": "call_tool",
    #             "action_input": {"tool_calls": tc_dicts}
    #         }
        
    #     # Priority 2: Check for agent_calls attribute
    #     if hasattr(self, 'agent_calls') and self.agent_calls:
    #         # For now, we only support single agent invocation, so take the first one
    #         # In the future, this could be extended to support multiple agent calls
    #         first_agent_call_msg = self.agent_calls[0]
            
    #         # Convert AgentCallMsg to dict
    #         if hasattr(first_agent_call_msg, 'to_dict'):
    #             ac_dict = first_agent_call_msg.to_dict()
    #         else:
    #             # Already a dict
    #             ac_dict = first_agent_call_msg
            
    #         return {
    #             "thought": thought,
    #             "next_action": "invoke_agent",
    #             "action_input": ac_dict
    #         }
        
    #     # Priority 3: Check for final_response in content
    #     if (isinstance(self.content, dict) and 
    #         self.content.get('next_action') == 'final_response'):
            
    #         return self.content  # Return the content dict as-is
        
    #     # Priority 4: Check if content is agent action format (handles legacy MessageContent)
    #     if isinstance(self.content, dict) and self._is_agent_action_content(self.content):
    #         return self.content  # Return the agent action content as-is
        
    #     # No valid action found
    #     return None

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
    
    @classmethod
    def from_harmonized_response(
        cls,
        harmonized_response: HarmonizedResponse,
        name: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> "Message":
        """
        Creates a Message instance from a HarmonizedResponse object.
        
        This method provides a clean way to convert from the BaseAPIModel's
        HarmonizedResponse format to the Message format used by agents.
        
        Args:
            harmonized_response: HarmonizedResponse object from BaseAPIModel
            name: Optional name (e.g., model name) to include in the message
            images: Optional list of image paths/URLs
            
        Returns:
            Message instance created from the HarmonizedResponse
        """
        
        # Convert tool_calls from response format to message format
        tool_calls = None
        if harmonized_response.tool_calls:
            tool_calls = []
            for tc in harmonized_response.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                })
        
        # Create Message with fields from HarmonizedResponse
        return cls(
            role=harmonized_response.role,
            content=harmonized_response.content,
            name=name or harmonized_response.metadata.model,
            tool_calls=tool_calls,
            images=images,
        )
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert message to OpenAI API format."""
        msg_dict = {"role": self.role}
        
        # Add content if present
        if self.content is not None:
            if isinstance(self.content, dict):
                # For structured content, convert to string
                msg_dict["content"] = json.dumps(self.content)
            else:
                msg_dict["content"] = self.content
        
        # Add name for tool responses or assistant messages
        if self.name:
            msg_dict["name"] = self.name
        
        # Add tool calls for assistant messages
        if self.tool_calls:
            msg_dict["tool_calls"] = []
            for tc in self.tool_calls:
                if hasattr(tc, 'to_dict'):
                    tc_dict = tc.to_dict()
                elif isinstance(tc, dict):
                    tc_dict = tc.copy()
                else:
                    continue
                
                # Remove internal metadata before sending to API
                tc_dict.pop('_origin', None)
                msg_dict["tool_calls"].append(tc_dict)
        
        # Add tool_call_id for tool response messages
        if self.role == "tool" and self.tool_call_id:
            msg_dict["tool_call_id"] = self.tool_call_id
        
        return msg_dict


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
    def add(
        self,
        message: Optional[Message] = None,
        *,  # Make subsequent arguments keyword-only if message is not None
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """
        Adds a new message to memory and returns the message ID.

        Args:
            message: Optional Message object to add directly.
            role: Message role (user, assistant, tool, etc.).
            content: Message content.
            name: Optional name field.
            tool_calls: Optional list of tool calls (can be dicts or ToolCallMsg objects).
            agent_calls: Optional list of agent calls (can be dicts or AgentCallMsg objects).
            images: Optional list of image paths/URLs for vision models.
            tool_call_id: Optional ID linking tool response to original tool call.
            
        Returns:
            str: The message ID of the added message
        """
        raise NotImplementedError("add must be implemented in subclasses.")

    @abstractmethod
    def update(
        self,
        message_id: str,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """
        Updates an existing message by its ID.

        Args:
            message_id: ID of the message to update.
            role: New role (if provided).
            content: New content (if provided).
            name: New name (if provided).
            tool_calls: New tool calls (if provided).
            agent_calls: New agent calls (if provided).
            images: New images list (if provided).
        """
        raise NotImplementedError("update must be implemented in subclasses.")

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
    def retrieve_recent(self, n: int = 1) -> List[Dict[str, Any]]:
        """Retrieves the 'n' most recent messages as dictionaries."""
        raise NotImplementedError("retrieve_recent must be implemented in subclasses.")

    @abstractmethod
    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieves all messages as dictionaries."""
        raise NotImplementedError("retrieve_all must be implemented in subclasses.")

    @abstractmethod
    def retrieve_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific message as dictionary by its ID."""
        raise NotImplementedError("retrieve_by_id must be implemented in subclasses.")

    @abstractmethod
    def remove_by_id(self, message_id: str) -> bool:
        """Removes a specific Message object by its ID. Returns True if removed, False if not found."""
        raise NotImplementedError("remove_by_id must be implemented in subclasses.")

    @abstractmethod
    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves messages filtered by role as dictionaries."""
        raise NotImplementedError("retrieve_by_role must be implemented in subclasses.")

    @abstractmethod
    def reset_memory(self) -> None:
        """Clears all entries from the memory."""
        raise NotImplementedError("reset_memory must be implemented in subclasses.")


class ConversationMemory(BaseMemory):
    """
    Memory module that stores conversation history as a list of Message objects.
    """

    def __init__(
        self, 
        description: Optional[str] = None,
    ) -> None:
        """
        Initializes ConversationMemory.

        Args:
            description: Optional content for an initial system message.
        """
        super().__init__(memory_type="conversation_history")
        self.memory: List[Message] = []
        if description:
            self.memory.append(Message(role="system", content=description))

    def _message_to_dict(self, msg: Message) -> Dict[str, Any]:
        """Convert Message to standard dict format for AI models"""
        # Handle content with images if present
        if msg.images and len(msg.images) > 0:
            # Create multimodal content with text and images
            content_parts = []
            
            # Add text content if present
            if msg.content is not None:
                content_text = json.dumps(msg.content, ensure_ascii=False, indent=2) if isinstance(msg.content, dict) else str(msg.content)
                if content_text.strip():
                    content_parts.append({
                        "type": "text",
                        "text": content_text
                    })
            
            # Add image content with base64 encoding
            for image_path in msg.images:
                encoded_image = msg._encode_image_to_base64(image_path)
                if encoded_image:  # Only add if encoding succeeded
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": encoded_image}
                    })
            
            result = {
                "role": msg.role,
                "content": content_parts
            }
        else:
            # No images, use simple content
            content = msg.content
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False, indent=2)
            
            result = {
                "role": msg.role,
                "content": content,
            }
        
        # Only include non-None optional fields
        if msg.name:
            result["name"] = msg.name
        if msg.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]
        
        # Add tool_call_id for tool response messages
        if msg.role == "tool" and msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        
        # Note: agent_calls and structured_data not included in standard AI format
        return result

    def add(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """
        Adds a new message to the conversation history and returns the message ID.
        If `message` object is provided, it's used directly.
        Otherwise, a new Message is created from the provided parameters.
        
        Returns:
            str: The message ID of the added message
        """
        if message:
            self.memory.append(message)
            return message.message_id
        elif role is not None:  # content can be None for assistant messages with tool_calls
            new_message = Message(
                role=role,
                content=content,
                name=name,
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                images=images,
                tool_call_id=tool_call_id,
            )
            self.memory.append(new_message)
            return new_message.message_id
        else:
            raise MessageError(
                "Either a Message object or role must be provided to add.",
                agent_name="ConversationMemory",
                validation_path="add.message_or_role_validation"
            )
    
    def update(
        self,
        message_id: str,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """
        Updates an existing message by its ID.
        Only provided fields are updated; others remain unchanged.
        """
        for i, msg in enumerate(self.memory):
            if msg.message_id == message_id:
                # Create a new message with updated fields
                updated_message = Message(
                    role=role if role is not None else msg.role,
                    content=content if content is not None else msg.content,
                    message_id=message_id,  # Keep the same ID
                    name=name if name is not None else msg.name,
                    tool_calls=tool_calls if tool_calls is not None else msg.tool_calls,
                    agent_calls=agent_calls if agent_calls is not None else msg.agent_calls,
                    images=images if images is not None else msg.images,
                )
                self.memory[i] = updated_message
                return
        
        raise MessageError(
            f"Message with ID '{message_id}' not found in memory.",
            agent_name="ConversationMemory",
            message_id=message_id,
            validation_path="update.message_id_not_found"
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

    def retrieve_recent(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves the 'n' most recent messages as dictionaries.
        """
        messages = self.memory[-n:] if n > 0 else []
        return [self._message_to_dict(msg) for msg in messages]

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieves all messages in the history as dictionaries."""
        return [self._message_to_dict(msg) for msg in self.memory]

    def retrieve_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific message as dictionary by its ID."""
        for msg in self.memory:
            if msg.message_id == message_id:
                return self._message_to_dict(msg)
        return None

    def remove_by_id(self, message_id: str) -> bool:
        """Removes a specific Message object by its ID. Returns True if removed, False if not found."""
        for i, msg in enumerate(self.memory):
            if msg.message_id == message_id:
                del self.memory[i]
                return True
        return False

    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves messages filtered by role as dictionaries, optionally limited to the most recent 'n'.
        """
        filtered = [m for m in self.memory if m.role == role]
        if n:
            filtered = filtered[-n:]
        return [self._message_to_dict(msg) for msg in filtered]

    def reset_memory(self) -> None:
        """Clears the conversation history, keeping the system prompt Message object if it exists."""
        system_message: Optional[Message] = None
        if self.memory and self.memory[0].role == "system":
            system_message = self.memory[0]
        self.memory.clear()
        if system_message:
            self.memory.append(system_message)



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

    def add(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """
        Adds a raw content entry to KG memory and schedules fact extraction.
        Similar to ConversationMemory but stores as a fact entry.
        
        Returns:
            str: The message ID of the added fact entry
        """
        if message:
            content_text = str(message.content) if message.content else "empty"
            fact_id = str(uuid.uuid4())
            # Store the raw content as a fact
            self.kg.append({
                "role": message.role,
                "subject": "raw_content",
                "predicate": "contains",
                "object": content_text,
                "timestamp": time.time(),
                "message_id": fact_id,
            })
            # Schedule extraction
            if message.content:
                asyncio.create_task(
                    self.extract_and_update_from_text(content_text, role=message.role)
                )
            return fact_id
        elif role is not None:
            content_text = str(content) if content else "empty"
            fact_id = str(uuid.uuid4())
            # Store the raw content as a fact
            self.kg.append({
                "role": role,
                "subject": "raw_content",
                "predicate": "contains",
                "object": content_text,
                "timestamp": time.time(),
                "message_id": fact_id,
            })
            # Schedule extraction
            if content:
                asyncio.create_task(
                    self.extract_and_update_from_text(content_text, role=role)
                )
            return fact_id
        else:
            raise MessageError(
                "Either a Message object or role must be provided to add.",
                agent_name="KGMemory",
                validation_path="add.message_or_role_validation"
            )
    
    def update(
        self,
        message_id: str,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """
        Updates an existing fact by its message_id.
        For KG, this updates the fact's properties directly.
        """
        for i, fact in enumerate(self.kg):
            if fact.get("message_id") == message_id:
                # Update fact properties
                if role is not None:
                    fact["role"] = role
                if content is not None:
                    # For KG, content should be a structured fact update
                    # We expect content to be a dict with subject/predicate/object
                    if isinstance(content, dict):
                        if "subject" in content:
                            fact["subject"] = content["subject"]
                        if "predicate" in content:
                            fact["predicate"] = content["predicate"]
                        if "object" in content:
                            fact["object"] = content["object"]
                    else:
                        logging.warning(
                            f"KGMemory.update: content should be a dict with subject/predicate/object, got {type(content)}"
                        )
                return
        
        raise MessageError(
            f"Fact with message_id '{message_id}' not found in KG.",
            agent_name="KGMemory",
            message_id=message_id,
            validation_path="update.fact_not_found"
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

    def _fact_to_dict(self, fact_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert KG fact to standard message dict"""
        content = f"Fact ({fact_dict['role']}): {fact_dict['subject']} {fact_dict['predicate']} {fact_dict['object']}."
        return {
            "role": fact_dict["role"],
            "content": content
        }

    def retrieve_recent(self, n: int = 1) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"], reverse=True)
        return [self._fact_to_dict(fact) for fact in sorted_kg[:n]]

    def retrieve_all(self) -> List[Dict[str, Any]]:
        # Sort by timestamp before converting to ensure order if needed, though not strictly required by BaseMemory
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"])
        return [self._fact_to_dict(fact) for fact in sorted_kg]

    def retrieve_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific KG fact as dict by its stored message_id."""
        for fact_dict in self.kg:
            if fact_dict.get("message_id") == message_id:
                return self._fact_to_dict(fact_dict)
        logging.debug(f"KGMemory: Fact with message_id '{message_id}' not found.")
        return None

    def remove_by_id(self, message_id: str) -> bool:
        """Removes a specific KG fact by its ID. Returns True if removed, False if not found."""
        for i, fact_dict in enumerate(self.kg):
            if fact_dict.get("message_id") == message_id:
                del self.kg[i]
                return True
        return False

    def retrieve_by_role(self, role: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
        filtered = [fact for fact in self.kg if fact["role"] == role]
        filtered = sorted(
            filtered, key=lambda x: x["timestamp"], reverse=True
        )  # Most recent first
        if n is not None and n > 0:
            filtered = filtered[:n]
        return [self._fact_to_dict(fact) for fact in filtered]

    def reset_memory(self) -> None:
        self.kg.clear()
        # Re-add system prompt if it was part of the initial setup logic
        # This depends on how system_prompt was handled in __init__
        # For now, simple clear. If system_prompt was stored as a fact, it's gone.


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
    ):
        """
        Initialize the MemoryManager with a specific memory strategy.

        Args:
            memory_type: Type of memory to use ("conversation_history" or "kg").
            description: Initial system description/prompt.
            model: Language model instance (required for KG memory).
        """
        self.memory_type = memory_type
        self.memory_module: BaseMemory

        if memory_type == "conversation_history":
            self.memory_module = ConversationMemory(
                description=description,
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

    def add(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """Delegates to the underlying memory module's add method and returns the message ID."""
        return self.memory_module.add(
            message=message,
            role=role,
            content=content,
            name=name,
            tool_calls=tool_calls,
            agent_calls=agent_calls,
            images=images,
            tool_call_id=tool_call_id,
        )
    
    def update(
        self,
        message_id: str,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict[str, Any], ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict[str, Any], AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """Delegates to the underlying memory module's update method."""
        self.memory_module.update(
            message_id=message_id,
            role=role,
            content=content,
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

    def save_to_file(self, filepath: str) -> None:
        """
        Save the current memory state to a file for persistent storage.
        
        Args:
            filepath: Path to save the memory state
        """
        import json
        from pathlib import Path
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get all messages from memory
        all_messages = self.memory_module.retrieve_all()
        
        # Prepare data for serialization
        data = {
            "memory_type": self.memory_type,
            "messages": all_messages,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Memory saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """
        Load memory state from a file.
        
        Args:
            filepath: Path to load the memory state from
        """
        import json
        from pathlib import Path
        
        if not Path(filepath).exists():
            logging.warning(f"Memory file {filepath} does not exist. Starting with empty memory.")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate memory type matches
            if data.get("memory_type") != self.memory_type:
                logging.warning(
                    f"Memory type mismatch: file has '{data.get('memory_type')}', "
                    f"but MemoryManager is configured for '{self.memory_type}'. "
                    "Loading anyway but this may cause issues."
                )
            
            # Clear existing memory
            self.memory_module.reset_memory()
            
            # Reload messages
            messages = data.get("messages", [])
            for msg_dict in messages:
                # Convert dict back to proper format for adding
                if isinstance(msg_dict, dict):
                    # Extract fields from the dict
                    role = msg_dict.get("role")
                    content = msg_dict.get("content")
                    name = msg_dict.get("name")
                    tool_calls = msg_dict.get("tool_calls")
                    agent_calls = msg_dict.get("agent_calls")
                    images = msg_dict.get("images")
                    
                    # Add message to memory
                    self.memory_module.add(
                        role=role,
                        content=content,
                        name=name,
                        tool_calls=tool_calls,
                        agent_calls=agent_calls,
                        images=images
                    )
            
            logging.info(f"Memory loaded from {filepath} with {len(messages)} messages")
            
        except Exception as e:
            logging.error(f"Failed to load memory from {filepath}: {e}")
            raise AgentFrameworkError(
                f"Failed to load memory from {filepath}: {e}",
                agent_name="MemoryManager",
                file_path=filepath
            )

