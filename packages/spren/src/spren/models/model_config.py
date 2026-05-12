"""Pydantic mirror of ``marsys.models.ModelConfig`` for the storage boundary.

Mirrors every field of the framework's ``ModelConfig`` EXCEPT ``api_key``: secret
credential values live in the per-user secrets store (OS keychain primary,
encrypted SQLite fallback) keyed by provider, not in workflow definitions.
``oauth_profile`` (a profile *name*, not a credential) is preserved.

At workflow-execution time the runtime materializer reads the spec, looks up
the matching credential from the secrets store by ``provider``, and constructs
a runnable ``marsys.models.ModelConfig`` with the resolved key.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ModelType = Literal["local", "api"]
ApiProvider = Literal[
    "openai",
    "openrouter",
    "google",
    "anthropic",
    "xai",
    "openai-oauth",
    "anthropic-oauth",
]
ModelClass = Literal["llm", "vlm"]
LocalBackend = Literal["huggingface", "vllm"]
LocalQuantization = Literal["awq", "gptq", "fp8"]


class ModelConfigSpec(BaseModel):
    """Storage-boundary mirror of marsys's ModelConfig (no api_key)."""

    model_config = ConfigDict(extra="forbid")

    type: ModelType
    name: str
    provider: Optional[ApiProvider] = None
    base_url: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    thinking_budget: Optional[int] = Field(1024, ge=0)
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = "low"
    oauth_profile: Optional[str] = None

    # Local-model fields (mirrored verbatim; ignored when type == "api").
    model_class: Optional[ModelClass] = None
    backend: Optional[LocalBackend] = "huggingface"
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = "auto"
    quantization_config: Optional[dict[str, Any]] = None
    tensor_parallel_size: Optional[int] = 1
    gpu_memory_utilization: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    quantization: Optional[LocalQuantization] = None
