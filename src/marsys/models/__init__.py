"""Models module for the MARS framework."""

from .models import (
    ModelConfig,
    BaseAPIModel,
    BaseLocalModel,
    # Local model adapters
    LocalProviderAdapter,
    LocalAdapterFactory,
    HuggingFaceLLMAdapter,
    HuggingFaceVLMAdapter,
    VLLMAdapter,
)

from .response_models import (
    HarmonizedResponse,
    ResponseMetadata,
    UsageInfo,
    ToolCall,
    ErrorResponse
)

from .credentials import (
    OAuthProfile,
    OAuthCredentialStore,
    OAuthTokenRefresher,
)

__all__ = [
    # Model config
    "ModelConfig",

    # API models
    "BaseAPIModel",

    # Local models (adapter pattern)
    "BaseLocalModel",
    "LocalProviderAdapter",
    "LocalAdapterFactory",
    "HuggingFaceLLMAdapter",
    "HuggingFaceVLMAdapter",
    "VLLMAdapter",

    # Response models
    "HarmonizedResponse",
    "ResponseMetadata",
    "UsageInfo",
    "ToolCall",
    "ErrorResponse",

    # OAuth credentials
    "OAuthProfile",
    "OAuthCredentialStore",
    "OAuthTokenRefresher",
]