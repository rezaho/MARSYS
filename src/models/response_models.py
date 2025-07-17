"""
Pydantic models for harmonized API responses.
Provides validation and structure for all provider responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolCall(BaseModel):
    """Represents a tool/function call."""
    id: str
    type: str = "function"
    function: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('function')
    @classmethod
    def validate_function(cls, v):
        """Ensure function has required fields."""
        if 'name' not in v:
            raise ValueError("Function must have 'name' field")
        if 'arguments' not in v:
            v['arguments'] = {}
        return v


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None  # For o1 models
    
    @model_validator(mode='after')
    def calculate_total(self):
        """Calculate total tokens if not provided."""
        if self.total_tokens is None:
            prompt = self.prompt_tokens or 0
            completion = self.completion_tokens or 0
            reasoning = self.reasoning_tokens or 0
            self.total_tokens = prompt + completion + reasoning
        return self


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
    
    class Config:
        # Allow extra fields for provider-specific metadata
        extra = "allow"


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
        if not self.content and not self.tool_calls:
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
    
    class Config:
        # Allow extra fields for extensibility
        extra = "allow"
        # Use enum values for validation
        use_enum_values = True


class ErrorResponse(BaseModel):
    """Response for API errors."""
    error: str
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    provider: str
    model: Optional[str] = None
    request_id: Optional[str] = None
    
    class Config:
        extra = "allow" 