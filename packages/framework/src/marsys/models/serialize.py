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

At execution time, callers materialize a **runnable** ``ModelConfig`` from a
spec via :func:`runtime_model_config_from_spec` (its default,
``runnable=True``): validators run, ``base_url`` is derived from
``provider``, and the credential is resolved from an explicitly-supplied
``api_key`` or the provider env var — raising if neither is reachable,
identical to a directly-constructed ``ModelConfig``. To load, validate, or
display a stored spec on a machine without credentials, callers pass
``runnable=False`` for a non-raising, non-runnable config that preserves
every non-secret field. The non-runnable form is an explicit opt-in, not
the default, so a runtime caller cannot silently receive a dead config.
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
    *,
    runnable: bool = True,
) -> ModelConfig:
    """Materialize a :class:`ModelConfig` from a storage spec.

    Two orthogonal axes: ``runnable`` (must this config execute?) and
    ``api_key`` (an explicitly-injected credential, e.g. from a secrets
    store).

    - ``runnable=True`` (**the default**): build via the normal Pydantic
      constructor so every validator runs — ``_set_base_url_from_provider``
      derives ``base_url`` from ``provider``, and ``_validate_api_key``
      resolves the credential. For a standard-API-key provider with no
      ``api_key`` supplied, the provider env var (``OPENAI_API_KEY`` etc.)
      is read exactly as a directly-constructed ``ModelConfig`` does, and a
      ``ValueError`` is raised if it is missing. This is the execution
      contract; ``pydantic_to_agents`` relies on it so canonically-defined
      workflows run identically to string-notation ones.
    - ``runnable=False``: build via ``model_construct`` so all validators
      are bypassed. Never raises, preserves every non-secret field, but the
      result is NOT runnable until a key is attached. This is the
      storage/inspection contract — loading, validating, or displaying a
      stored spec on a machine without credentials (community templates,
      MARSYS Cloud pre-deploy validation, persistence layers). It was the
      historical default; making it an explicit opt-in removes the footgun
      where a runtime caller silently received a dead config.

    OAuth providers (``openai-oauth``, ``anthropic-oauth``) need no
    ``api_key`` and no env var on either path: ``_validate_api_key``'s oauth
    branch is a no-op and the provider adapter resolves both endpoint and
    credential from ``oauth_profile`` at client-init, independent of these
    fields. Routing them through ``runnable=True`` is safe and matches the
    proven string-notation baseline.
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
    if not runnable:
        # Inspection/storage: bypass every validator — never resolve a
        # credential, never derive base_url, never raise. Not runnable.
        return ModelConfig.model_construct(api_key=api_key, **common_fields)
    # Runnable (default): full validation. api_key=None lets
    # _validate_api_key resolve the provider env var, and raise if it is
    # missing — exact parity with a directly-constructed ModelConfig.
    return ModelConfig(api_key=api_key, **common_fields)
