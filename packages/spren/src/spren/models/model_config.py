"""Spren consumes the framework's canonical model-config wire type.

SP-005: the framework owns this shape; Spren does not mirror it. Thin
re-export so existing ``from spren.models[.model_config] import ...`` sites
keep working and the FastAPI/OpenAPI surface is generated from the framework
type. The framework ``ModelConfigSpec`` already omits ``api_key`` (resolved
per-provider at runtime by ``runtime_model_config_from_spec``).
"""
from __future__ import annotations

from marsys.models.serialize import ApiProvider, ModelConfigSpec, ModelType

__all__ = ["ApiProvider", "ModelConfigSpec", "ModelType"]
