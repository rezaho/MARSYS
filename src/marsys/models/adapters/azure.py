"""Adapter for OpenAI models hosted on Azure OpenAI (AI Foundry).

Azure re-hosts OpenAI's own models behind its own resource, and its **v1 surface**
serves the Responses API at the identical relative path with the identical request and
response bodies. That is what makes this a thin subclass of :class:`OpenAIAdapter`
instead of a parallel implementation: message conversion, tool conversion, structured
output, SSE streaming, reasoning effort, usage harmonization and error classification
are all inherited unchanged. ``get_endpoint_url()`` is inherited too — against a
``.../openai/v1`` base it already produces the documented
``POST .../openai/v1/responses``.

The older Azure surface — ``/openai/deployments/<name>/responses`` with a pinned
``api-version`` query parameter — is deliberately not used. On the v1 GA API
``api-version`` is no longer required, and the deployment name travels in the request
body's ``model`` field, which is exactly the shape the inherited payload builder already
emits. Targeting the legacy surface would mean overriding endpoint construction to gain
nothing.

Four deltas from the first-party adapter, each measured against the live resource
rather than assumed:

* **base_url** — per-resource, so it cannot be a compiled-in constant. Resolved from
  the environment at construction, the way :mod:`marsys.models.adapters.bedrock`
  resolves its region-dependent host.
* **Auth** — the resource key travels in an ``api-key`` header. (A resource key is also
  accepted in ``Authorization: Bearer`` on this surface, measured; ``api-key`` is the
  documented spelling for a key, with Bearer reserved for Entra ID tokens, so the
  documented one is what this sends.)
* **Reasoning-effort vocabulary** — this surface serves ``none``, ``low``, ``medium``,
  ``high``, ``xhigh`` and ``max``, and answers ``minimal`` with a 400 that enumerates
  those six. The smallest thinking budget therefore arrives as ``low`` here while the
  first-party leg keeps sending ``minimal``.
* **Model ids** — the ``model`` field carries an operator-chosen *deployment* name, and
  the response echoes that same deployment name back rather than an underlying model
  snapshot. Measured on ``gpt-5.6-sol`` and ``gpt-5.6-terra``: both echo themselves.
  A cost table keyed on the echoed id therefore keys on deployment names.

Two behaviours of this endpoint that are inherited rather than worked around, recorded
because both are silent until they are not:

* **Prompt caching is implicit and takes no request parameter.** ``prompt_cache_options``
  accepts ``implicit`` (the default) and ``explicit``; a ``prompt_cache_breakpoint``
  field is rejected as an unknown parameter in every position, and ``explicit`` mode
  without one caches nothing. So the correct request is one that says nothing about
  caching, and a measured pair of identical large calls reports
  ``cache_write_tokens: 3395`` then ``cached_tokens: 3395`` with no parameter sent.
  The inherited harmonizer reads both figures.
* **Reasoning-capable deployments reject ``temperature``.** The inherited capability
  check is a regex over the model name, which the real deployment names
  (``gpt-5.6-*``) satisfy. A deployment renamed to something not starting ``gpt-5``+
  would send ``temperature`` and take a 400 from this endpoint. Naming a deployment
  after the model it serves keeps that check honest; this adapter cannot know a
  deployment's underlying model from its name alone.
"""

import logging
import os
from typing import Dict, Optional

from marsys.models.adapters.openai import AsyncOpenAIAdapter, OpenAIAdapter

logger = logging.getLogger(__name__)

AZURE_OPENAI_V1_PATH = "/openai/v1"


def azure_openai_base_url(endpoint: Optional[str] = None) -> str:
    """The v1 base URL for an Azure OpenAI resource, or ``""`` if unconfigured.

    Resolution order: explicit argument, ``AZURE_OPENAI_ENDPOINT``, then
    ``FOUNDRY_ENDPOINT``. The second name is read because this stack's voice legs
    already resolve the same resource under it — one resource, one credential, read by
    whichever name the caller's environment happens to carry, rather than a second
    stored copy of the same secret.

    Accepts the several spellings one resource legitimately arrives as and returns the
    one the Responses path hangs off:

    * ``https://<res>.services.ai.azure.com`` — the resource host.
    * ``https://<res>.openai.azure.com`` — the same resource's other documented host.
    * ``https://<res>.services.ai.azure.com/api/projects/<project>`` — the *project*
      endpoint, which is what the Azure portal offers for copying and what an escrowed
      copy of this value turned out to hold. It addresses a project inside the
      resource and does not serve ``/openai/v1``; the resource host does, so the
      project suffix is dropped.
    * an endpoint already ending in ``/openai/v1`` — returned as-is, so a caller who
      configured the full base is not given ``/openai/v1/openai/v1``.
    """
    resolved = (
        endpoint
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("FOUNDRY_ENDPOINT")
        or ""
    ).strip()
    if not resolved:
        # Empty rather than a guess: there is no default Azure resource, and a
        # fabricated host would turn a missing setting into a confusing connection
        # error instead of an obvious unconfigured one.
        return ""
    resolved = resolved.rstrip("/")
    if resolved.endswith(AZURE_OPENAI_V1_PATH):
        return resolved
    marker = "/api/projects/"
    if marker in resolved:
        resolved = resolved[: resolved.index(marker)]
    return f"{resolved}{AZURE_OPENAI_V1_PATH}"


class AzureOpenAIAdapter(OpenAIAdapter):
    """OpenAI models on an Azure OpenAI resource, via the v1 Responses surface."""

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("FOUNDRY_API_KEY", ""),
            # An empty passed value falls through to the environment, which is what
            # lets a caller whose model-construction path has no per-resource endpoint
            # to hand still reach the right host.
            base_url=base_url or azure_openai_base_url(endpoint),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def get_headers(self) -> Dict[str, str]:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _provider_name(self) -> Optional[str]:
        # Pinned rather than derived from the class name, so retry/error settings and
        # the harmonized response's provider field both resolve under the provider id
        # the rest of the stack uses.
        return "azure"

    def _served_effort(self, effort: str, model_lower: str) -> str:
        # This surface serves none / low / medium / high / xhigh / max and answers
        # `minimal` with a 400 (measured on `gpt-5.6-sol`, whose reply enumerates the
        # six it takes). Unconditional rather than keyed on the model name: the name
        # here is an operator-chosen deployment label and cannot be read as evidence
        # about the generation underneath, and the smallest reasoning this endpoint has
        # is what a request for the smallest should get either way.
        if effort == "minimal":
            return "low"
        return super()._served_effort(effort, model_lower)


class AsyncAzureOpenAIAdapter(AsyncOpenAIAdapter, AzureOpenAIAdapter):
    """Async Azure OpenAI adapter.

    Inherits the Responses SSE stream from :class:`AsyncOpenAIAdapter` and the
    endpoint/auth/provider deltas from :class:`AzureOpenAIAdapter`.
    """

    def _provider_name(self) -> Optional[str]:
        return "azure"
