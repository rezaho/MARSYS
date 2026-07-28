"""Adapter for Claude on Amazon Bedrock.

Bedrock exposes two shapes for the same models. This adapter targets the
**Messages-API-shaped** endpoint (``bedrock-mantle.<region>.api.aws``), which
speaks the identical request/response body as the first-party Messages API and
streams real SSE. That is what makes this a thin subclass of
:class:`AnthropicAdapter` instead of a parallel implementation: message
conversion, thinking/temperature capability shaping, tool conversion, deferred
tool loading, SSE accumulation and harmonization are all inherited.

The other shape — ``bedrock-runtime``'s ``/invoke`` and
``/invoke-with-response-stream`` — takes ``anthropic_version`` in the body, no
``model`` key, and frames streams as AWS binary ``application/vnd.amazon.eventstream``
rather than SSE. Using it would mean a second stream parser for no gain, so it is
deliberately not used.

Three deltas from the first-party adapter, each measured against the live
endpoint rather than assumed:

* **Auth** — a bearer token (``AWS_BEARER_TOKEN_BEDROCK``), not ``x-api-key``.
* **Model ids** — must carry the ``anthropic.`` prefix (``anthropic.claude-opus-5``).
  A bare id or a cross-region ``us.anthropic.`` id returns 404 here.
* **No structured outputs** — ``output_config.format`` and per-tool ``strict``
  are rejected outright ("Extra inputs are not permitted"), so schema requests
  fall back to the inherited prompt-based JSON path.
"""

import logging
import os
from typing import Dict, Optional

from marsys.models.adapters.anthropic import AnthropicAdapter, AsyncAnthropicAdapter

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_REGION = "us-east-1"


def bedrock_base_url(region: Optional[str] = None) -> str:
    """Messages-API-shaped Bedrock base URL for a region.

    Region resolution order: explicit argument, ``AWS_REGION``,
    ``AWS_DEFAULT_REGION``, then ``us-east-1``.
    """
    resolved = (
        region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or DEFAULT_BEDROCK_REGION
    )
    return f"https://bedrock-mantle.{resolved}.api.aws/anthropic/v1"


def normalize_bedrock_model_id(model_name: str) -> str:
    """Return the id spelling this endpoint accepts.

    Bedrock requires exactly one ``anthropic.`` prefix. Callers legitimately
    arrive with any of three spellings — a bare id from the shared catalog, an
    ``anthropic/``-prefixed id copied from OpenRouter, or a cross-region
    ``us.anthropic.`` id copied from the AWS console — and the two wrong ones
    404. Normalizing here keeps the caller's id portable.
    """
    name = (model_name or "").strip()
    if not name:
        return name
    for prefix in ("us.anthropic.", "eu.anthropic.", "apac.anthropic.", "anthropic/"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    if not name.lower().startswith("anthropic."):
        name = f"anthropic.{name}"
    return name


class BedrockAdapter(AnthropicAdapter):
    """Claude on Amazon Bedrock via the Messages-API-shaped endpoint."""

    # The endpoint rejects `output_config.format` and per-tool `strict`; the
    # inherited prompt-based JSON fallback carries the schema instead.
    supports_structured_output = False

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        region: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=normalize_bedrock_model_id(model_name),
            api_key=api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""),
            base_url=base_url or bedrock_base_url(region),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def _provider_name(self) -> Optional[str]:
        # The base implementation derives this from the class name; keep it
        # pinned so retry/error settings resolve under the provider id the rest
        # of the stack uses.
        return "bedrock"

    def report_model_id(self, echoed: Optional[str]) -> str:
        """Report the ``anthropic.``-prefixed id, not Bedrock's bare echo.

        A request for ``anthropic.claude-sonnet-5`` comes back as
        ``claude-sonnet-5``, and ``anthropic.claude-haiku-4-5`` as
        ``claude-haiku-4-5-20251001`` — ids that exist in the *first-party*
        namespace, so a cost table keyed by Bedrock ids finds no rate and
        silently prices the call at zero. Prefer the echo only when it round-trips
        to the same id we asked for, so a genuine snapshot resolution is still
        visible.
        """
        if echoed and normalize_bedrock_model_id(echoed) == self.model_name:
            return normalize_bedrock_model_id(echoed)
        return self.model_name


class AsyncBedrockAdapter(AsyncAnthropicAdapter, BedrockAdapter):
    """Async Bedrock adapter.

    Inherits SSE streaming from :class:`AsyncAnthropicAdapter` (the endpoint
    returns ``text/event-stream``) and the auth/id/capability deltas from
    :class:`BedrockAdapter`.
    """

    def _provider_name(self) -> Optional[str]:
        return "bedrock"
