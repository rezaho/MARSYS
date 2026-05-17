"""Spren consumes the framework's canonical ``AgentSpec``.

SP-005: the framework owns this shape; Spren does not mirror it. The
framework spec already names the model-config field ``agent_model`` (Pydantic
v2 reserves ``model_config``) and omits ``api_key``.
"""
from __future__ import annotations

from marsys.agents.serialize import AgentSpec, MemoryRetention

__all__ = ["AgentSpec", "MemoryRetention"]
