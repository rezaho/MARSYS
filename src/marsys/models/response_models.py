"""
Pydantic models for harmonized API responses.
Provides validation and structure for all provider responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


class ToolCall(BaseModel):
    """Represents a tool/function call."""
    id: str
    type: str = "function"
    function: Dict[str, Any] = Field(default_factory=dict)
    # The container the provider loaded this function out of, when the leg names one: a hosted
    # tool search returns the namespace it searched and stamps the call with it. Carried so the
    # item can go back to that provider whole on the next request.
    namespace: Optional[str] = None

    @field_validator('function')
    @classmethod
    def validate_function(cls, v):
        """Ensure function has required fields."""
        if 'name' not in v:
            raise ValueError("Function must have 'name' field")
        if 'arguments' not in v:
            v['arguments'] = {}
        return v

    @model_serializer(mode="wrap")
    def _omit_absent_namespace(self, handler):
        """Drop ``namespace`` from the serialized form when the provider named none.

        Every leg but one sends no namespace, and a tool call's serialized form becomes a durable
        conversation row on the way back: a key that is always there and always null would change
        those bytes on every call of every leg, which is a cache re-write for a field nobody
        reads. Absent means absent."""
        data = handler(self)
        if isinstance(data, dict) and data.get("namespace") is None:
            data.pop("namespace", None)
        return data


class UsageInfo(BaseModel):
    """Token usage information.

    ``prompt_tokens`` is the provider's *uncached* prompt count. With prompt
    caching active it is NOT the whole prompt: the full prompt is
    ``prompt_tokens + cache_creation_input_tokens + cache_read_input_tokens``
    (Anthropic's own definition). A consumer that sizes a conversation, prices a
    call, or bounds prompt growth must read ``full_prompt_tokens`` — reading
    ``prompt_tokens`` alone silently under-measures by up to ~10x once a cached
    prefix exists. Both cache fields are None on providers that report no cache
    figures, which is why ``full_prompt_tokens`` coalesces rather than sums
    blindly.
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None  # For o1 models
    # Prompt-cache accounting. Populated by the providers that report it
    # (Anthropic api-key, Bedrock, Anthropic OAuth); None everywhere else.
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None

    @model_validator(mode='after')
    def calculate_total(self):
        """Calculate total tokens if not provided.

        Deliberately unchanged by the cache fields: ``total_tokens`` keeps its
        established meaning (prompt + completion + reasoning) so a response that
        reports no cache figures harmonizes to the same number it always did.
        The cache-aware reading is ``full_prompt_tokens``.
        """
        if self.total_tokens is None:
            prompt = self.prompt_tokens or 0
            completion = self.completion_tokens or 0
            reasoning = self.reasoning_tokens or 0
            self.total_tokens = prompt + completion + reasoning
        return self

    @property
    def full_prompt_tokens(self) -> int:
        """The whole prompt the provider processed, cached parts included.

        The number a context bound, a spend ledger, and a runaway-growth backstop
        all actually want. Equals ``prompt_tokens`` exactly when no cache was
        involved, so it is a safe unconditional substitute at every such site.
        """
        return (
            (self.prompt_tokens or 0)
            + (self.cache_creation_input_tokens or 0)
            + (self.cache_read_input_tokens or 0)
        )


class ResponseMetadata(BaseModel):
    """Metadata about the API response."""
    provider: str
    model: str
    request_id: Optional[str] = None
    created: Optional[datetime] = None
    usage: Optional[UsageInfo] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    
    # Provider-specific fields
    stop_reason: Optional[str] = None  # Anthropic
    stop_sequence: Optional[str] = None  # Anthropic
    safety_ratings: Optional[List[Dict[str, Any]]] = Field(default_factory=list)  # Google
    candidates_count: Optional[int] = None  # Google
    reasoning_effort: Optional[str] = None  # OpenRouter
    thinking_budget: Optional[int] = None  # OpenRouter/Google
    site_info: Optional[Dict[str, Any]] = None  # OpenRouter

    model_config = ConfigDict(extra="allow")  # Allow extra fields for provider-specific metadata


class HarmonizedResponse(BaseModel):
    """
    Standardized response format for all API providers.
    This is the single format that all adapters must return.
    """
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning: Optional[str] = None  # For o1 models or reasoning traces
    thinking: Optional[str] = None  # For thinking/planning content
    # Opaque provider-emitted blocks that must round-trip VERBATIM on the next
    # request: Gemini 3 thought signatures ({"type": "text"|"function_call",
    # "thought_signature": ...}), Anthropic extended-thinking blocks
    # ({"type": "thinking", "thinking", "signature"} / {"type":
    # "redacted_thinking", "data"}), and the OpenAI Responses hosted tool search's
    # own two output items ({"type": "tool_search_call"} and
    # {"type": "tool_search_output"}, the output carrying the definitions the
    # search loaded). Type-discriminated — each provider's payload builder
    # re-emits only its own block types. The name says reasoning and the channel
    # means provider output the next request owes back; renaming it reaches four
    # adapters, the agent memory and the tracing, and every row already written.
    reasoning_details: Optional[List[Dict[str, Any]]] = None
    metadata: ResponseMetadata
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Ensure role is valid."""
        valid_roles = ['assistant', 'user', 'system', 'tool']
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}, got {v}")
        return v
    
    @model_validator(mode='after')
    def validate_content_or_tool_calls(self):
        """Ensure we have either content or tool_calls."""
        # Allow empty content if the response was truncated due to length limits
        if hasattr(self.metadata, 'finish_reason') and self.metadata.finish_reason == 'length':
            # Response was truncated, allow empty content
            return self
        
        # Check for None specifically, not empty string
        # Empty strings are valid responses from some providers like OpenAI
        if self.content is None and not self.tool_calls:
            raise ValueError("Response must have either content or tool_calls")
        return self
    
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0
    
    def has_reasoning(self) -> bool:
        """Check if response contains reasoning trace."""
        return self.reasoning is not None
    
    def has_thinking(self) -> bool:
        """Check if response contains thinking content."""
        return self.thinking is not None

    def has_reasoning_details(self) -> bool:
        """Check if response contains reasoning details (Gemini 3 thought signatures)."""
        return self.reasoning_details is not None and len(self.reasoning_details) > 0

    def get_text_content(self) -> str:
        """Get all text content combined."""
        parts = []
        if self.thinking:
            parts.append(f"Thinking: {self.thinking}")
        if self.reasoning:
            parts.append(f"Reasoning: {self.reasoning}")
        if self.content:
            parts.append(self.content)
        return "\n\n".join(parts)

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        use_enum_values=True  # Use enum values for validation
    )


class ErrorResponse(BaseModel):
    """Response for API errors."""
    error: str
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    provider: str
    model: Optional[str] = None
    request_id: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None  # New field for error classification

    model_config = ConfigDict(extra="allow") 