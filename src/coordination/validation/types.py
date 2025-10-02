"""
Pydantic data classes for agent invocation validation.

This module provides strongly-typed data structures for validating
agent invocation requests, especially for parallel execution scenarios.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional, Literal
import uuid


class AgentInvocation(BaseModel):
    """
    Single agent invocation with request data.
    
    This class represents a single agent invocation in a parallel or sequential
    execution context. It includes validation to ensure agent names are valid
    and request data is properly formatted.
    """
    agent_name: str = Field(..., description="Name of the agent to invoke")
    request: Any = Field(..., description="Request data for the agent")
    instance_id: Optional[str] = Field(default=None, description="Unique instance ID for parallel invocations")
    
    @field_validator('agent_name')
    @classmethod
    def validate_agent_name(cls, v):
        """Ensure agent name is not empty and properly formatted."""
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v.strip()
    
    def __init__(self, **data):
        """Override init to generate instance_id if not provided."""
        if 'instance_id' not in data or data.get('instance_id') is None:
            # Generate a unique instance ID based on agent name
            agent_name = data.get('agent_name', 'unknown')
            data['instance_id'] = f"{agent_name}_{uuid.uuid4().hex[:8]}"
        super().__init__(**data)
    
    def to_request_data(self) -> Any:
        """
        Extract the request data for passing to agent.
        
        Returns:
            The request data that should be passed to the target agent
        """
        return self.request
    
    def to_coordination_dict(self) -> dict:
        """
        Convert to dict for coordination layer storage.
        
        Returns:
            Dictionary representation for coordination layer
        """
        return {
            "agent_name": self.agent_name,
            "request": self.request,
            "instance_id": self.instance_id
        }
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class ValidationError(BaseModel):
    """
    Structured validation error response.
    
    This is returned to agents when their invocation format is invalid,
    allowing them to correct and retry.
    """
    next_action: Literal["validation_error"] = Field(default="validation_error")
    error: str = Field(..., description="Main error message")
    details: List[str] = Field(default_factory=list, description="Detailed error messages")
    retry_suggestion: str = Field(..., description="Suggestion for how to fix the error")
    raw_response: Any = Field(..., description="The original response that failed validation")