from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from marsys.models.models import BaseAPIModel
from marsys.models.response_models import HarmonizedResponse

# Type-only imports for optional local model support (requires marsys[local-models])
if TYPE_CHECKING:
    from marsys.models.models import BaseLocalModel, LocalProviderAdapter
    from marsys.coordination.event_bus import EventBus

# Import the new exception classes
from .exceptions import (
    AgentConfigurationError,
    AgentFrameworkError,
    MessageError,
)

# --- Event Classes ---

@dataclasses.dataclass
class MemoryResetEvent:
    """Event emitted when agent memory is reset."""
    agent_name: str
    timestamp: float = dataclasses.field(default_factory=time.time)


# --- Structured Content Data Classes ---

@dataclasses.dataclass
class ToolTruncationConfig:
    """Configuration for tool response truncation reducer."""
    enabled: bool = True
    max_tool_message_tokens: int = 1200
    grace_recent_messages: int = 1
    append_marker: str = " [truncated]"
    include_tool_payload_user_messages: bool = True


@dataclasses.dataclass
class SlidingWindowConfig:
    """Configuration for sliding window mode."""
    enabled: bool = True
    headroom_percent: float = 0.1


@dataclasses.dataclass
class SummarizationConfig:
    """Configuration for model-based summarization processor."""
    enabled: bool = True
    grace_recent_messages: int = 1
    output_max_tokens: int = 6000
    prompt_mode: str = "generic"
    include_original_instruction: bool = True
    include_image_payload_bytes: bool = False
    max_retained_images: Optional[int] = None


# Backward compatibility alias
CompactionConfig = SummarizationConfig


@dataclasses.dataclass
class BackwardPackingConfig:
    """Configuration for backward packing processor."""
    grace_recent_messages: int = 1


@dataclasses.dataclass
class ActiveContextPolicyConfig:
    """Configuration for active context management policy."""
    enabled: bool = True
    mode: str = "compaction"
    sliding_window: SlidingWindowConfig = dataclasses.field(default_factory=SlidingWindowConfig)
    processor_order: List[str] = dataclasses.field(
        default_factory=lambda: ["tool_truncation", "summarization", "backward_packing"]
    )
    excluded_processors: List[str] = dataclasses.field(default_factory=list)
    min_reduction_ratio: float = 0.4
    tool_truncation: ToolTruncationConfig = dataclasses.field(default_factory=ToolTruncationConfig)
    summarization: SummarizationConfig = dataclasses.field(default_factory=SummarizationConfig)
    backward_packing: BackwardPackingConfig = dataclasses.field(default_factory=BackwardPackingConfig)

    @property
    def compaction(self) -> SummarizationConfig:
        """Backward compatibility alias for summarization config."""
        return self.summarization

    @property
    def reduction_order(self) -> List[str]:
        """Backward compatibility alias for processor_order."""
        return self.processor_order


@dataclasses.dataclass
class ManagedMemoryConfig:
    """Configuration for ManagedMemory with active context management."""

    # Trigger threshold — when to trigger compaction
    threshold_tokens: int = 150_000

    # Image handling
    image_token_estimate: int = 800

    # Strategy selection
    trigger_events: List[str] = dataclasses.field(default_factory=lambda: ["add", "get_messages"])
    cache_invalidation_events: List[str] = dataclasses.field(default_factory=lambda: ["add", "remove_by_id", "delete_memory"])

    # Token counter
    token_counter: Optional[Callable] = None  # Override with custom counter

    # Active context management policy
    active_context: ActiveContextPolicyConfig = dataclasses.field(default_factory=ActiveContextPolicyConfig)

    @property
    def compaction_target_tokens(self) -> int:
        """Derived target: threshold * (1 - min_reduction_ratio)."""
        ratio = self.active_context.min_reduction_ratio
        return int(self.threshold_tokens * (1 - ratio))

# --- Structured Content Data Classes (continued) ---

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
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None  # Can be string, dict, or typed array
    message_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None  # For tool role (tool name) or assistant name (model name)
    tool_calls: Optional[List[ToolCallMsg]] = None  # For assistant requesting tool calls
    agent_calls: Optional[List[AgentCallMsg]] = None  # For assistant requesting agent invocations
    structured_data: Optional[Dict[str, Any]] = None  # For storing structured data when agent returns a dictionary from auto_run
    images: Optional[List[str]] = None  # For storing image paths/URLs for vision models
    tool_call_id: Optional[str] = None  # For tool response messages, links to the original tool call
    reasoning_details: Optional[List[Dict[str, Any]]] = None  # For Gemini 3 thought signatures (must be preserved for multi-turn tool calling)

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

    def to_llm_dict(self) -> Dict[str, Any]:
        """
        Convert Message to dict in LLM API format.

        This method handles all the conversion logic to ensure messages are properly
        formatted for LLM API calls. Handles:
        - Tool role with string content requirement (OpenAI API)
        - Multimodal content (typed arrays, images)
        - Tool calls serialization
        - Base64 image encoding

        Returns:
            Dict in LLM API format
        """
        # Special handling for role="tool": content MUST be string
        if self.role == "tool":
            content = self.content
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            elif content is None:
                content = ""
            else:
                content = str(content)

            result = {"role": "tool", "content": content}

            if self.tool_call_id:
                result["tool_call_id"] = self.tool_call_id
            if self.name:
                result["name"] = self.name

            return result

        # For other roles: handle multimodal and typed arrays

        # If content is already a typed array, use it directly
        if isinstance(self.content, list):
            result = {"role": self.role, "content": self.content}
        # Handle legacy images field (for backward compatibility)
        elif self.images and len(self.images) > 0:
            content_parts = []

            # Add text content if present
            if self.content is not None:
                if isinstance(self.content, dict):
                    content_text = json.dumps(
                        self.content, ensure_ascii=False, separators=(",", ":")
                    )
                elif isinstance(self.content, list):
                    content_text = json.dumps(
                        self.content, ensure_ascii=False, separators=(",", ":")
                    )
                else:
                    content_text = str(self.content)

                if content_text.strip():
                    content_parts.append({"type": "text", "text": content_text})

            # Add image content with base64 encoding
            for image_path in self.images:
                encoded_image = self._encode_image_to_base64(image_path)
                if encoded_image:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": encoded_image}}
                    )

            result = {"role": self.role, "content": content_parts}
        else:
            # No images, simple content
            content = self.content
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))

            result = {"role": self.role, "content": content}

        # Add optional fields (for non-tool roles)
        if self.name and self.role != "tool":
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        # Add reasoning_details for Gemini 3 thought signature preservation
        # This is critical for multi-turn tool calling with Gemini 3 models
        if self.reasoning_details:
            result["reasoning_details"] = self.reasoning_details

        return result


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
            reasoning_details=harmonized_response.reasoning_details,  # Preserve for Gemini 3
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

        # Add reasoning_details for Gemini 3 thought signature preservation
        if self.reasoning_details:
            msg_dict["reasoning_details"] = self.reasoning_details

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
        self._event_bus: Optional['EventBus'] = None
        self._agent_name: Optional[str] = None
        self._session_id: Optional[str] = None

    def set_event_context(
        self,
        agent_name: str,
        event_bus: Optional['EventBus'] = None,
        session_id: Optional[str] = None
    ) -> None:
        """
        Configure event emission for memory reset and compaction notifications.

        Called by the agent during run_step() when EventBus becomes available.

        Args:
            agent_name: Name of the owning agent (used for event filtering)
            event_bus: EventBus instance for emitting events
            session_id: Coordination session ID for status events
        """
        self._agent_name = agent_name
        if event_bus is not None:
            self._event_bus = event_bus
        if session_id is not None:
            self._session_id = session_id

    def _emit_compaction_event(self, **kwargs) -> None:
        """Emit CompactionEvent if event context is configured."""
        if not (self._event_bus and self._session_id):
            return
        from marsys.coordination.status.events import CompactionEvent
        self._event_bus.emit_nowait(
            CompactionEvent(
                session_id=self._session_id,
                agent_name=self._agent_name or "unknown",
                **kwargs
            )
        )

    def _emit_reset_event(self) -> None:
        """Emit MemoryResetEvent if configured."""
        if self._event_bus and self._agent_name:
            self._event_bus.emit_nowait(
                MemoryResetEvent(agent_name=self._agent_name)
            )

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

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Retrieves messages for LLM (primary interface method).

        Default implementation delegates to retrieve_all() for backward compatibility.
        Subclasses with active context management should override this method.

        Returns:
            List of message dictionaries ready for LLM
        """
        return self.retrieve_all()

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

    # === Common Utility Methods (Non-Abstract) ===

    def _message_to_dict(self, msg: Message) -> Dict[str, Any]:
        """
        Convert Message to dict in LLM API format.

        This is a common utility method used across all memory types to ensure
        consistent formatting. It delegates to the Message class's to_llm_dict()
        method to maintain separation of concerns.

        Args:
            msg: Message object to convert

        Returns:
            Dict in LLM API format
        """
        return msg.to_llm_dict()


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
        self._emit_reset_event()



class KGMemory(BaseMemory):
    """
    Memory module storing knowledge as timestamped (Subject, Predicate, Object) triplets.
    Requires a language model instance to extract facts from text.
    Facts are converted to Message objects upon retrieval.
    """

    def __init__(
        self,
        model: Union[BaseLocalModel, LocalProviderAdapter, BaseAPIModel],
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
        self._emit_reset_event()


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


class ManagedConversationMemory(BaseMemory):
    """
    Conversation memory with active context management (ACM).

    This memory implementation automatically manages context size by:
    1. Tracking token usage across all messages
    2. Triggering context management when thresholds are exceeded
    3. Retrieving curated context for LLM using pluggable strategies
    4. Caching curated context to avoid redundant computation

    Features:
    - Automatic token tracking (text + multimodal)
    - Lazy trigger checking (on add/retrieve events)
    - Cached curated context
    - Pluggable strategies (trigger, process, retrieval)
    - Full lossless raw message history (for future strategies)

    The curated context is transparent to agents - they call get_messages()
    and receive a token-budgeted subset without any code changes.
    """

    def __init__(
        self,
        config: Optional[ManagedMemoryConfig] = None,
        trigger_strategy: Optional[Any] = None,  # TriggerStrategy
        process_strategy: Optional[Any] = None,  # ProcessStrategy
        retrieval_strategy: Optional[Any] = None,  # RetrievalStrategy
        description: Optional[str] = None,
    ):
        """
        Initialize ManagedConversationMemory with strategies.

        Args:
            config: Memory configuration (uses defaults if not provided)
            trigger_strategy: When to engage ACM (defaults to SimpleThresholdTrigger)
            process_strategy: How to process messages (defaults to NoOpProcessStrategy)
            retrieval_strategy: How to retrieve curated context (defaults to BackwardPackingRetrieval)
            description: Optional system description (not stored, Agent rebuilds it)
        """
        super().__init__(memory_type="managed_conversation")

        # Import strategies lazily to avoid circular imports
        from marsys.agents.memory_strategies import (
            BackwardPackingRetrieval,
            NoOpProcessStrategy,
            SimpleThresholdTrigger,
        )
        from marsys.utils.tokens import DefaultTokenCounter

        # Configuration
        self.config = config or ManagedMemoryConfig()

        # Strategies
        self.trigger_strategy = trigger_strategy or SimpleThresholdTrigger()
        self.process_strategy = process_strategy or NoOpProcessStrategy()
        self.retrieval_strategy = retrieval_strategy or BackwardPackingRetrieval()

        # Token counter
        self.token_counter = self.config.token_counter or DefaultTokenCounter(
            image_token_estimate=self.config.image_token_estimate
        )

        # Raw message storage (full, lossless history)
        self.raw_messages: List[Message] = []

        # Curated context cache
        self._cached_context: Optional[List[Dict[str, Any]]] = None
        self._cache_valid = False
        self._last_retrieval_index = 0

        # Token tracking
        self._estimated_total_tokens = 0
        self._tokens_since_last_retrieval = 0
        self._messages_since_last_retrieval = 0

        # Metadata
        self.metadata: Dict[str, Any] = {}

        # System message handling: not stored (Agent rebuilds it dynamically)
        if description:
            logging.debug(
                "ManagedConversationMemory: description provided but not stored "
                "(system prompts are rebuilt by Agent)"
            )

    # === Primary Methods (Framework Interface) ===

    def add(
        self,
        message: Optional[Message] = None,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict, ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict, AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """Add message and update token tracking."""
        # Create message
        if message:
            msg = message
        elif role is not None:
            msg = Message(
                role=role,
                content=content,
                name=name,
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                images=images,
                tool_call_id=tool_call_id,
            )
        else:
            raise MessageError("Either message or role required")

        # Add to raw storage
        self.raw_messages.append(msg)

        # Update token estimate
        msg_dict = self._message_to_dict(msg)
        msg_tokens = self.token_counter.count_message(msg_dict)
        self._estimated_total_tokens += msg_tokens
        self._tokens_since_last_retrieval += msg_tokens
        self._messages_since_last_retrieval += 1

        # Check cache invalidation
        if "add" in self.config.cache_invalidation_events:
            self._invalidate_cache()

        # Check trigger (lazy - just set flag)
        if "add" in self.config.trigger_events:
            from marsys.agents.memory_strategies import MemoryState

            state = self._get_memory_state()
            if self.trigger_strategy.should_trigger_on_add(state, self.config):
                self._cache_valid = False

        return msg.message_id

    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieve curated context for LLM (PRIMARY method)."""
        mode = self.config.active_context.mode

        # Handle deprecated mode name
        if mode == "destructive_compaction":
            import warnings
            warnings.warn(
                'mode="destructive_compaction" is deprecated, use mode="compaction"',
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "compaction"

        # In compaction mode, raw_messages have already been
        # compacted by _run_compaction(). Return them directly without
        # applying backward-packing retrieval (which would discard compacted output).
        if mode == "compaction" and self.config.active_context.enabled:
            return [self._message_to_dict(msg) for msg in self.raw_messages]

        # sliding_window mode: use backward-packing retrieval
        # Check trigger
        if "get_messages" in self.config.trigger_events:
            from marsys.agents.memory_strategies import MemoryState

            state = self._get_memory_state()
            if self.trigger_strategy.should_trigger_on_retrieve(state, self.config):
                self._cache_valid = False

        # Return cached if valid
        if self._cache_valid and self._cached_context is not None:
            new_messages = self.raw_messages[self._last_retrieval_index :]
            new_dicts = [self._message_to_dict(msg) for msg in new_messages]
            return self._cached_context + new_dicts

        # Recompute curated context
        return self._retrieve_curated_context()

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieve all messages (delegates to get_messages for ACM)."""
        return self.get_messages()

    def retrieve_recent(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get N most recent messages from RAW storage."""
        recent = self.raw_messages[-n:] if n > 0 else []
        return [self._message_to_dict(msg) for msg in recent]

    def update(
        self,
        message_id: str,
        *,
        role: Optional[str] = None,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[Union[Dict, ToolCallMsg]]] = None,
        agent_calls: Optional[List[Union[Dict, AgentCallMsg]]] = None,
        images: Optional[List[str]] = None,
    ) -> None:
        """Update existing message and recalculate tokens."""
        for i, msg in enumerate(self.raw_messages):
            if msg.message_id == message_id:
                updated = Message(
                    role=role if role is not None else msg.role,
                    content=content if content is not None else msg.content,
                    message_id=message_id,
                    name=name if name is not None else msg.name,
                    tool_calls=tool_calls if tool_calls is not None else msg.tool_calls,
                    agent_calls=(
                        agent_calls if agent_calls is not None else msg.agent_calls
                    ),
                    images=images if images is not None else msg.images,
                )

                old_dict = self._message_to_dict(msg)
                new_dict = self._message_to_dict(updated)
                old_tokens = self.token_counter.count_message(old_dict)
                new_tokens = self.token_counter.count_message(new_dict)

                self._estimated_total_tokens += new_tokens - old_tokens
                self.raw_messages[i] = updated
                self._invalidate_cache()
                return

        raise MessageError(f"Message {message_id} not found")

    def remove_by_id(self, message_id: str) -> bool:
        """Remove message by ID and update token tracking."""
        for i, msg in enumerate(self.raw_messages):
            if msg.message_id == message_id:
                msg_dict = self._message_to_dict(msg)
                msg_tokens = self.token_counter.count_message(msg_dict)
                self._estimated_total_tokens -= msg_tokens

                del self.raw_messages[i]

                if "remove_by_id" in self.config.cache_invalidation_events:
                    self._invalidate_cache()

                return True
        return False

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
        """Replace message at index and update token tracking."""
        if not (0 <= idx < len(self.raw_messages)):
            raise IndexError("Index out of range")

        old_msg = self.raw_messages[idx]

        if message:
            new_msg = message
        elif role is not None:
            new_msg = Message(
                role=role,
                content=content,
                message_id=message_id or str(uuid.uuid4()),
                name=name,
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                images=images,
            )
        else:
            raise MessageError("Either message or role required")

        old_dict = self._message_to_dict(old_msg)
        new_dict = self._message_to_dict(new_msg)
        old_tokens = self.token_counter.count_message(old_dict)
        new_tokens = self.token_counter.count_message(new_dict)
        self._estimated_total_tokens += new_tokens - old_tokens

        self.raw_messages[idx] = new_msg
        self._invalidate_cache()

    def delete_memory(self, idx: int) -> None:
        """Delete message at index and update token tracking."""
        if 0 <= idx < len(self.raw_messages):
            msg = self.raw_messages[idx]
            msg_dict = self._message_to_dict(msg)
            msg_tokens = self.token_counter.count_message(msg_dict)
            self._estimated_total_tokens -= msg_tokens

            del self.raw_messages[idx]

            if "delete_memory" in self.config.cache_invalidation_events:
                self._invalidate_cache()
        else:
            raise IndexError("Index out of range")

    def retrieve_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get message by ID from raw storage."""
        for msg in self.raw_messages:
            if msg.message_id == message_id:
                return self._message_to_dict(msg)
        return None

    def retrieve_by_role(
        self, role: str, n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Filter messages by role from raw storage."""
        filtered = [msg for msg in self.raw_messages if msg.role == role]
        if n:
            filtered = filtered[-n:]
        return [self._message_to_dict(msg) for msg in filtered]

    def reset_memory(self) -> None:
        """Clear all messages and reset state."""
        self.raw_messages.clear()
        self._cached_context = None
        self._cache_valid = False
        self._last_retrieval_index = 0
        self._estimated_total_tokens = 0
        self._tokens_since_last_retrieval = 0
        self._messages_since_last_retrieval = 0
        self.metadata.clear()
        self._emit_reset_event()

    # === Secondary Methods (ACM-specific) ===

    def _get_memory_state(self):
        """Build current memory state snapshot."""
        from marsys.agents.memory_strategies import MemoryState

        return MemoryState(
            raw_messages=self.raw_messages.copy(),
            estimated_tokens=self._estimated_total_tokens,
            messages_since_last_retrieval=self._messages_since_last_retrieval,
            tokens_since_last_retrieval=self._tokens_since_last_retrieval,
            last_retrieval_index=self._last_retrieval_index,
            metadata=self.metadata.copy(),
        )

    def _retrieve_curated_context(self) -> List[Dict[str, Any]]:
        """Execute retrieval strategy and cache result."""
        state = self._get_memory_state()

        curated = self.retrieval_strategy.retrieve(
            state, self.config, self.token_counter
        )

        self._cached_context = curated
        self._cache_valid = True
        self._last_retrieval_index = len(self.raw_messages)
        self._messages_since_last_retrieval = 0
        self._tokens_since_last_retrieval = 0

        return curated

    def _invalidate_cache(self) -> None:
        """Mark cache as invalid."""
        self._cache_valid = False

    # === Active Context Compaction (Async) ===

    async def _run_compaction(
        self, runtime: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Execute the compaction processor chain and rewrite memory.

        Checks whether compaction should trigger based on estimated tokens
        exceeding threshold_tokens, then runs processors in priority order.
        Each processor estimates its reduction capability before being applied.
        The last processor in the chain always runs as a last resort.
        """
        from marsys.agents.memory_strategies import COMPACTION_PROCESSOR_REGISTRY

        policy = self.config.active_context
        if not policy.enabled or policy.mode != "compaction":
            return
        if self._estimated_total_tokens <= self.config.threshold_tokens:
            return

        pre_tokens = self._estimated_total_tokens
        pre_messages = len(self.raw_messages)
        working_messages = [self._message_to_dict(msg) for msg in self.raw_messages]
        original_messages = list(working_messages)

        metadata_log = []
        processors = []
        for name in policy.processor_order:
            if name in policy.excluded_processors:
                logging.info(f"Compaction: processor '{name}' excluded by config")
                continue
            cls = COMPACTION_PROCESSOR_REGISTRY.get(name)
            if cls is None:
                logging.warning(f"Unknown compaction processor: {name}, skipping")
                continue
            processors.append(cls())

        logging.info(
            f"Compaction triggered: {pre_tokens} tokens, "
            f"threshold={self.config.threshold_tokens}, "
            f"processors={[p.name() for p in processors]}"
        )

        # Emit "started" event
        self._emit_compaction_event(
            status="started",
            pre_tokens=pre_tokens,
            pre_messages=pre_messages,
        )
        compaction_start = time.time()

        try:
            for i, processor in enumerate(processors):
                is_last = (i == len(processors) - 1)

                if is_last:
                    logging.info(f"Compaction: running last-resort '{processor.name()}'")
                    try:
                        working_messages, meta = await processor.reduce(
                            working_messages, self.config, self.token_counter, runtime
                        )
                        metadata_log.append(meta)
                    except Exception as e:
                        logging.error(f"Processor '{processor.name()}' failed: {e}")
                        metadata_log.append({"stage": processor.name(), "error": str(e)})
                else:
                    try:
                        estimated_savings = await processor.estimate_reduction(
                            working_messages, self.config, self.token_counter, runtime
                        )
                        ratio = estimated_savings / self.config.threshold_tokens if self.config.threshold_tokens > 0 else 0.0

                        if ratio >= policy.min_reduction_ratio:
                            logging.info(
                                f"Compaction: running '{processor.name()}' "
                                f"(savings={estimated_savings}, ratio={ratio:.3f}, "
                                f"min_ratio={policy.min_reduction_ratio})"
                            )
                            working_messages, meta = await processor.reduce(
                                working_messages, self.config, self.token_counter, runtime
                            )
                            metadata_log.append(meta)
                        else:
                            logging.info(
                                f"Compaction: skipping '{processor.name()}' "
                                f"(savings={estimated_savings}, ratio={ratio:.3f}, "
                                f"min_ratio={policy.min_reduction_ratio})"
                            )
                            metadata_log.append({
                                "stage": processor.name(),
                                "skipped": True,
                                "reason": "insufficient_reduction",
                                "estimated_savings": estimated_savings,
                                "ratio": ratio,
                            })
                    except Exception as e:
                        logging.error(f"Processor '{processor.name()}' failed: {e}")
                        metadata_log.append({"stage": processor.name(), "error": str(e)})

            # Minimum output guarantee: at least 2 messages
            if len(working_messages) < 2:
                logging.warning(
                    f"Compaction produced insufficient output "
                    f"({len(working_messages)} msgs), "
                    f"running forced summarization fallback"
                )
                working_messages = await self._forced_summarization_fallback(
                    original_messages, runtime, policy, metadata_log
                )

            # Safety check: API calls require at least one user message
            has_user = any(m.get("role") == "user" for m in working_messages)
            if not has_user:
                logging.error(
                    "Compaction result has no user message — API call would fail. "
                    "Injecting most recent user message from original context."
                )
                for msg in reversed(original_messages):
                    if msg.get("role") == "user":
                        working_messages.insert(0, msg)
                        break

            # Rewrite memory
            self._rewrite_memory_with_reduced_context(working_messages)
            self._recompute_token_accounting()

            # Store compaction metadata
            self.metadata["last_compaction"] = {
                "timestamp": time.time(),
                "stages": metadata_log,
                "pre_compaction_tokens": pre_tokens,
                "pre_compaction_messages": pre_messages,
                "post_compaction_tokens": self._estimated_total_tokens,
                "post_compaction_messages": len(self.raw_messages),
            }
            self._invalidate_cache()

            # Emit "completed" event
            stages_run = [
                m["stage"] for m in metadata_log
                if "stage" in m and not m.get("skipped")
            ]
            self._emit_compaction_event(
                status="completed",
                pre_tokens=pre_tokens,
                post_tokens=self._estimated_total_tokens,
                pre_messages=pre_messages,
                post_messages=len(self.raw_messages),
                duration=time.time() - compaction_start,
                stages_run=stages_run,
            )

        except Exception as e:
            logging.error(f"Compaction failed: {e}")
            self._emit_compaction_event(
                status="failed",
                pre_tokens=pre_tokens,
                pre_messages=pre_messages,
                duration=time.time() - compaction_start,
            )
            raise

    def _estimate_payload_bytes(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total serialized payload bytes for a list of message dicts."""
        total = 0
        for msg in messages:
            content = msg.get("content")
            if content is None:
                total += 200
            elif isinstance(content, str):
                total += len(content.encode("utf-8")) + 100
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        itype = item.get("type")
                        if itype == "image_url":
                            url = item.get("image_url", {})
                            if isinstance(url, dict):
                                url = url.get("url", "")
                            total += len(url) if isinstance(url, str) else 1000
                        elif itype == "text":
                            total += len(item.get("text", ""))
                        else:
                            total += len(json.dumps(item, separators=(",", ":")))
                total += 200
            elif isinstance(content, dict):
                total += len(json.dumps(content, separators=(",", ":")))
            tc = msg.get("tool_calls")
            if tc:
                total += len(json.dumps(tc, separators=(",", ":")))
            total += 50
        return total

    async def compact_for_payload_error(
        self, runtime: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Compact memory after a payload-too-large error.

        Runs SummarizationProcessor on the prefix (before the last assistant
        round), preserving the tail. Uses the same summarization config as
        regular compaction. Success is measured by serialized payload-byte
        reduction.

        Returns True if payload bytes were reduced, False otherwise.
        """
        from marsys.agents.memory_strategies import (
            COMPACTION_PROCESSOR_REGISTRY,
            _compute_protected_tail_start,
        )

        messages = [self._message_to_dict(msg) for msg in self.raw_messages]
        if len(messages) <= 3:
            return False

        # Boundary: last assistant round
        policy = self.config.active_context
        summ_cfg = policy.summarization
        grace = max(summ_cfg.grace_recent_messages, 1)
        boundary = _compute_protected_tail_start(messages, grace)

        if boundary <= 1:
            return False

        prefix = messages[:boundary]
        protected_tail = messages[boundary:]

        pre_bytes = self._estimate_payload_bytes(messages)
        pre_tokens = self._estimated_total_tokens
        pre_messages = len(messages)

        self._emit_compaction_event(
            status="started",
            pre_tokens=pre_tokens,
            pre_messages=pre_messages,
        )
        compaction_start = time.time()

        try:
            summ_cls = COMPACTION_PROCESSOR_REGISTRY.get("summarization")
            if summ_cls is None:
                logging.error("SummarizationProcessor not found in registry")
                self._emit_compaction_event(
                    status="failed",
                    pre_tokens=pre_tokens,
                    pre_messages=pre_messages,
                    duration=time.time() - compaction_start,
                )
                return False

            # Use existing config with grace=0 so processor summarizes the
            # ENTIRE prefix (we already split at the boundary)
            import copy
            temp_config = copy.deepcopy(self.config)
            temp_config.active_context.summarization.grace_recent_messages = 0
            if temp_config.active_context.summarization.max_retained_images is None:
                temp_config.active_context.summarization.max_retained_images = 5
            temp_config.active_context.summarization.include_image_payload_bytes = False

            working, meta = await summ_cls().reduce(
                prefix, temp_config, self.token_counter, runtime
            )

            # Combine: summarized prefix + protected tail
            new_messages = working + protected_tail

            # Safety guards (lifecycle parity with _run_compaction)
            if len(new_messages) < 2:
                new_messages = await self._forced_summarization_fallback(
                    messages, runtime, policy, [meta]
                )

            has_user = any(m.get("role") == "user" for m in new_messages)
            if not has_user:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        new_messages.insert(0, msg)
                        break

            # Check success: payload byte reduction
            post_bytes = self._estimate_payload_bytes(new_messages)
            if post_bytes >= pre_bytes:
                logging.warning(
                    f"Payload compaction did not reduce bytes: "
                    f"{pre_bytes} -> {post_bytes}"
                )
                self._emit_compaction_event(
                    status="failed",
                    pre_tokens=pre_tokens,
                    pre_messages=pre_messages,
                    duration=time.time() - compaction_start,
                )
                return False

            # Rewrite memory
            self._rewrite_memory_with_reduced_context(new_messages)
            self._recompute_token_accounting()
            self._invalidate_cache()

            self.metadata["last_compaction"] = {
                "timestamp": time.time(),
                "trigger": "payload_error",
                "stages": [meta],
                "pre_compaction_tokens": pre_tokens,
                "pre_compaction_messages": pre_messages,
                "post_compaction_tokens": self._estimated_total_tokens,
                "post_compaction_messages": len(self.raw_messages),
                "pre_payload_bytes": pre_bytes,
                "post_payload_bytes": post_bytes,
            }

            stages_run = ["summarization"] if not meta.get("skipped") else []
            self._emit_compaction_event(
                status="completed",
                pre_tokens=pre_tokens,
                post_tokens=self._estimated_total_tokens,
                pre_messages=pre_messages,
                post_messages=len(self.raw_messages),
                duration=time.time() - compaction_start,
                stages_run=stages_run,
            )

            logging.info(
                f"Payload compaction: {pre_messages}->{len(self.raw_messages)} msgs, "
                f"~{pre_bytes}->~{post_bytes} bytes"
            )
            return True

        except Exception as e:
            logging.error(f"Payload compaction failed: {e}")
            self._emit_compaction_event(
                status="failed",
                pre_tokens=pre_tokens,
                pre_messages=pre_messages,
                duration=time.time() - compaction_start,
            )
            return False

    async def _forced_summarization_fallback(
        self,
        original_messages: List[Dict[str, Any]],
        runtime: Optional[Dict[str, Any]],
        policy: Any,
        metadata_log: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Emergency fallback: produce at least [user_request_summary, summary] when the
        processor chain yields insufficient output.
        """
        from marsys.agents.memory_strategies import (
            SummarizationProcessor,
            _build_compaction_payload,
            _build_user_rollup,
        )

        compact_cfg = policy.summarization
        compaction_model = runtime.get("compaction_model") if runtime else None
        compaction_model_name = runtime.get("compaction_model_name", "unknown") if runtime else "unknown"

        # Build user rollup from all original messages
        user_rollup_text = _build_user_rollup(original_messages)

        if compaction_model is not None:
            agent_instruction = runtime.get("agent_instruction") if runtime else None
            compaction_messages, _ = _build_compaction_payload(
                original_messages,
                agent_instruction=agent_instruction,
                compact_cfg=compact_cfg,
            )

            processor = SummarizationProcessor()
            summary_json = await processor._run_compaction_model(
                compaction_messages, compaction_model, compact_cfg
            )

            if summary_json:
                result = []

                # 1. User request summary (from model or fallback)
                user_req_summary = summary_json.get("user_request_summary", "")
                if user_req_summary:
                    result.append({
                        "role": "user",
                        "content": f"[User Requests Summary]\n{user_req_summary}",
                    })
                elif user_rollup_text:
                    result.append({
                        "role": "user",
                        "content": f"[User Requests Summary]\n{user_rollup_text}",
                    })

                # 2. Conversation summary
                summary_text = summary_json.get("summary", "")
                salient_facts = summary_json.get("salient_facts", [])
                open_threads = summary_json.get("open_threads", [])

                summary_parts = [f"[Compacted Context Summary]\n{summary_text}"]
                if salient_facts:
                    summary_parts.append(
                        "\n[Key Facts]\n" + "\n".join(f"- {f}" for f in salient_facts)
                    )
                if open_threads:
                    summary_parts.append(
                        "\n[Open Threads]\n" + "\n".join(f"- {t}" for t in open_threads)
                    )

                result.append({
                    "role": "assistant",
                    "content": "\n".join(summary_parts),
                    "name": f"compaction_{compaction_model_name}",
                })

                metadata_log.append({
                    "stage": "forced_summarization_fallback",
                    "post_messages": len(result),
                })
                logging.info(
                    f"Forced fallback produced {len(result)} messages "
                    f"via model={compaction_model_name}"
                )
                return result

        # Hard fallback: no model available — create a minimal placeholder
        result = []
        if user_rollup_text:
            result.append({
                "role": "user",
                "content": f"[User Requests Summary]\n{user_rollup_text}",
            })
        result.append({
            "role": "assistant",
            "content": "[Context was compacted. Previous conversation history is no longer available.]",
        })

        metadata_log.append({
            "stage": "forced_summarization_fallback",
            "hard_fallback": True,
            "post_messages": len(result),
        })
        logging.warning("Forced fallback: no compaction model, using hard placeholder")
        return result

    def _rewrite_memory_with_reduced_context(
        self, reduced_messages: List[Dict[str, Any]]
    ) -> None:
        """
        Replace raw_messages with reduced message dicts.

        Converts dict messages back to Message objects for storage.

        Args:
            reduced_messages: List of message dicts after reduction.
        """
        new_raw: List[Message] = []
        for msg_dict in reduced_messages:
            role = msg_dict.get("role", "assistant")
            content = msg_dict.get("content")
            name = msg_dict.get("name")
            tool_calls = msg_dict.get("tool_calls")
            tool_call_id = msg_dict.get("tool_call_id")
            reasoning_details = msg_dict.get("reasoning_details")

            msg = Message(
                role=role,
                content=content,
                name=name,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                reasoning_details=reasoning_details,
            )
            new_raw.append(msg)

        self.raw_messages = new_raw

    def _recompute_token_accounting(self) -> None:
        """Recalculate token tracking from current raw_messages."""
        total = 0
        for msg in self.raw_messages:
            msg_dict = self._message_to_dict(msg)
            total += self.token_counter.count_message(msg_dict)

        self._estimated_total_tokens = total
        self._messages_since_last_retrieval = 0
        self._tokens_since_last_retrieval = 0
        self._last_retrieval_index = len(self.raw_messages)

    # === Utility Methods ===

    def get_raw_messages(self) -> List[Message]:
        """Access full raw message history (for debugging/export)."""
        return self.raw_messages.copy()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring/debugging."""
        return {
            "cache_valid": self._cache_valid,
            "cached_count": len(self._cached_context) if self._cached_context else 0,
            "raw_count": len(self.raw_messages),
            "estimated_total_tokens": self._estimated_total_tokens,
            "messages_since_last_retrieval": self._messages_since_last_retrieval,
            "tokens_since_last_retrieval": self._tokens_since_last_retrieval,
        }


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
        model: Optional[Union[BaseLocalModel, LocalProviderAdapter]] = None,
        memory_config: Optional[ManagedMemoryConfig] = None,
        token_counter: Optional[Callable] = None,
    ):
        """
        Initialize the MemoryManager with a specific memory strategy.

        Args:
            memory_type: Type of memory to use ("conversation_history", "managed_conversation", or "kg")
            description: Initial system description/prompt
            model: Language model instance (required for KG memory)
            memory_config: Configuration for ManagedConversationMemory (optional, only used for "managed_conversation")
            token_counter: Custom token counter (optional, only used for "managed_conversation")
        """
        self.memory_type = memory_type
        self.memory_module: BaseMemory

        if memory_type == "conversation_history":
            self.memory_module = ConversationMemory(
                description=description,
            )
        elif memory_type == "managed_conversation":
            # Create config if not provided
            if memory_config is None:
                memory_config = ManagedMemoryConfig()

            # Override token counter if provided
            if token_counter is not None:
                memory_config.token_counter = token_counter

            self.memory_module = ManagedConversationMemory(
                config=memory_config,
                description=description,
            )
        elif memory_type == "kg":
            if model is None:
                raise AgentConfigurationError(
                    "KGMemory requires a 'model' instance for fact extraction.",
                    agent_name="MemoryManager",
                    config_field="model",
                    config_value=None
                )
            self.memory_module = KGMemory(model=model, description=description)
        else:
            raise AgentConfigurationError(
                f"Unknown memory_type: {memory_type}",
                agent_name="MemoryManager",
                config_field="memory_type",
                config_value=memory_type
            )

    def set_event_context(
        self,
        agent_name: str,
        event_bus: Optional['EventBus'] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Delegate event context setup to the underlying memory module."""
        self.memory_module.set_event_context(agent_name, event_bus, session_id)

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

    def get_messages(self) -> List[Message]:
        """
        Delegates to the underlying memory module's get_messages.

        This is the preferred method for retrieving messages for LLM consumption.
        For ManagedConversationMemory, this returns curated context; for other
        memory types, it delegates to retrieve_all().
        """
        return self.memory_module.get_messages()

    async def compact_if_needed(self, runtime: Optional[Dict[str, Any]] = None) -> None:
        """Run compaction if the underlying memory supports it."""
        if hasattr(self.memory_module, "_run_compaction"):
            await self.memory_module._run_compaction(runtime)

    async def compact_for_payload_error(self, runtime: Optional[Dict[str, Any]] = None) -> bool:
        """Run payload-error compaction if the underlying memory supports it."""
        if hasattr(self.memory_module, "compact_for_payload_error"):
            return await self.memory_module.compact_for_payload_error(runtime)
        return False

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

    def save_to_file(
        self,
        filepath: str,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the current memory state to a file for persistent storage.

        Args:
            filepath: Path to save the memory state
            additional_state: Optional dict of additional state to save (e.g., planning state)
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

        # Include additional state if provided
        if additional_state:
            data["additional_state"] = additional_state

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Memory saved to {filepath}")

    def load_from_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load memory state from a file.

        Args:
            filepath: Path to load the memory state from

        Returns:
            additional_state dict if present in file, None otherwise
        """
        import json
        from pathlib import Path

        if not Path(filepath).exists():
            logging.warning(f"Memory file {filepath} does not exist. Starting with empty memory.")
            return None

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

            # Return additional state if present
            return data.get("additional_state")

        except Exception as e:
            logging.error(f"Failed to load memory from {filepath}: {e}")
            raise AgentFrameworkError(
                f"Failed to load memory from {filepath}: {e}",
                agent_name="MemoryManager",
                file_path=filepath
            )

