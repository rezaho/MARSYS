"""Pydantic mirror of marsys's Agent constructor surface.

The model-config field is named ``agent_model`` because Pydantic v2's
``BaseModel`` reserves ``model_config`` for class-level configuration. The
nested type is ``ModelConfigSpec`` (Spren's storage-boundary mirror), NOT
``marsys.models.ModelConfig`` directly: the framework's ModelConfig enforces
API-key presence at validation time, which fails at storage time when keys
live in the secrets store and are resolved per-provider only at execution
time.

``tools`` stores tool *names* (strings) only; the runtime registry that
resolves names to callables is wired up at workflow-execution time.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .model_config import ModelConfigSpec


MemoryRetention = Literal["single_run", "session", "persistent"]


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_model: ModelConfigSpec
    name: str
    goal: str
    instruction: str
    tools: list[str] = Field(default_factory=list)
    memory_retention: MemoryRetention = "session"
    allowed_peers: list[str] = Field(default_factory=list)
