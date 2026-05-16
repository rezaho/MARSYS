"""
Storage-boundary Pydantic mirror of :class:`marsys.models.ModelConfig`.

``ModelConfigSpec`` is the wire shape that workflow definitions persist. It
omits ``api_key`` because secret credential values live in per-user credential
stores (OS keychain, encrypted SQLite, env vars) keyed by provider, NOT in
the workflow definition itself. It also omits the ``_validate_api_key``
``model_validator(mode="after")`` that runtime ``ModelConfig`` runs, since
that validator raises ``ValueError`` whenever an API key is not reachable —
which is exactly the case at storage time, on machines that haven't been
configured yet, and inside community workflow templates shared publicly.

At execution time, callers materialize a runnable ``ModelConfig`` from a spec
via :func:`runtime_model_config_from_spec`, optionally supplying an
``api_key`` resolved from the consumer's secrets store. When no key is
supplied, the framework's ``ModelConfig._validate_api_key`` resolves the env
var as today.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import ModelConfig


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
ModelClassLiteral = Literal["llm", "vlm"]
LocalBackend = Literal["huggingface", "vllm"]
LocalQuantization = Literal["awq", "gptq", "fp8"]


class ModelConfigSpec(BaseModel):
    """Storage-boundary mirror of :class:`marsys.models.ModelConfig` (no api_key)."""

    model_config = ConfigDict(extra="forbid")

    type: ModelType
    name: str
    provider: Optional[ApiProvider] = None
    base_url: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    thinking_budget: Optional[int] = Field(1024, ge=0)
    # Mirrors ModelConfig.reasoning_effort which is Optional[str] at runtime —
    # narrowing here would reject values the framework accepts (e.g., provider-
    # specific values). Validation of the effort string is the runtime's job.
    reasoning_effort: Optional[str] = "low"
    oauth_profile: Optional[str] = None

    model_class: Optional[ModelClassLiteral] = None
    backend: Optional[LocalBackend] = "huggingface"
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = "auto"
    quantization_config: Optional[dict[str, Any]] = None
    tensor_parallel_size: Optional[int] = 1
    gpu_memory_utilization: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    quantization: Optional[LocalQuantization] = None


def model_config_spec_from_runtime(config: ModelConfig) -> ModelConfigSpec:
    """Build a storage-safe :class:`ModelConfigSpec` from a runtime ``ModelConfig``.

    Drops ``api_key`` deterministically (never copied to the spec).
    """
    return ModelConfigSpec(
        type=config.type,
        name=config.name,
        provider=config.provider,
        base_url=config.base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        thinking_budget=config.thinking_budget,
        reasoning_effort=config.reasoning_effort,
        oauth_profile=config.oauth_profile,
        model_class=config.model_class,
        backend=config.backend,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        quantization_config=config.quantization_config,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        quantization=config.quantization,
    )


def runtime_model_config_from_spec(
    spec: ModelConfigSpec,
    api_key: Optional[str] = None,
) -> ModelConfig:
    """Materialize a :class:`ModelConfig` from a storage spec.

    Two paths:

    - **With ``api_key``**: builds a runnable ``ModelConfig`` via the normal
      Pydantic constructor. ``_validate_api_key`` runs, ``_set_base_url_from_provider``
      runs. Suitable for immediate execution.
    - **Without ``api_key``** (the default): builds a ``ModelConfig`` via
      ``model_construct`` so ``_validate_api_key`` is bypassed. Suitable for
      inspection, round-trip, or staged materialization (caller supplies the
      key later from a credential store or env var). The returned object is
      NOT runnable through ``Agent`` until a real key is attached, but it
      preserves every non-secret field.

    OAuth providers (``openai-oauth``, ``anthropic-oauth``) bypass the
    api-key path; ``oauth_profile`` carries the credential reference.
    """
    common_fields = dict(
        type=spec.type,
        name=spec.name,
        provider=spec.provider,
        base_url=spec.base_url,
        max_tokens=spec.max_tokens,
        temperature=spec.temperature,
        thinking_budget=spec.thinking_budget,
        reasoning_effort=spec.reasoning_effort,
        oauth_profile=spec.oauth_profile,
        model_class=spec.model_class,
        backend=spec.backend,
        torch_dtype=spec.torch_dtype,
        device_map=spec.device_map,
        quantization_config=spec.quantization_config,
        tensor_parallel_size=spec.tensor_parallel_size,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        quantization=spec.quantization,
    )
    if api_key is not None:
        return ModelConfig(api_key=api_key, **common_fields)
    return ModelConfig.model_construct(api_key=None, **common_fields)
