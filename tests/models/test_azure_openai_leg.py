"""The Azure OpenAI leg: what the wire accepts, and what the meter is told it cost.

Every assertion here is the offline half of something measured against a real Azure
OpenAI resource. The three that would be cheap to get wrong and expensive to discover
live:

* an unrecognized per-item key in the ``input`` array is a hard ``400
  unknown_parameter``, so the message sanitize is a connectivity requirement rather
  than hygiene;
* ``cached_tokens`` is a SLICE of ``input_tokens`` on this API and the ADDITIVE
  complement of ``prompt_tokens`` in ``UsageInfo``, so a field-to-field mapping
  double-counts the cached prefix and bills it at the fresh-input rate;
* ``reasoning_tokens`` is a SLICE of ``output_tokens``, so anything that adds the two
  overstates output by up to the whole reasoning budget.
"""

import warnings

import pytest

from marsys.models.adapters.azure import (
    AsyncAzureOpenAIAdapter,
    AzureOpenAIAdapter,
    azure_openai_base_url,
)
from marsys.models.adapters.factory import ProviderAdapterFactory
from marsys.models.adapters.openai import (
    OpenAIAdapter,
    thinking_budget_to_effort,
)
from marsys.models.models import PROVIDER_BASE_URLS
from marsys.models.serialize import ApiProvider

MESSAGES = [{"role": "user", "content": "hi"}]
RESOURCE = "https://marsys-dev-fn-01.services.ai.azure.com"


def _azure(model_name: str = "gpt-5.6-sol", **kwargs) -> AzureOpenAIAdapter:
    return AzureOpenAIAdapter(
        model_name=model_name,
        api_key="not-a-real-key",
        base_url=f"{RESOURCE}/openai/v1",
        max_tokens=4096,
        **kwargs,
    )


# --- the endpoint ------------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        (RESOURCE, f"{RESOURCE}/openai/v1"),
        (f"{RESOURCE}/", f"{RESOURCE}/openai/v1"),
        # The portal's copyable "project endpoint", which is what the operator's
        # escrowed copy of this value actually holds. It does not serve /openai/v1.
        (f"{RESOURCE}/api/projects/marsys-llm-api-01", f"{RESOURCE}/openai/v1"),
        (f"{RESOURCE}/api/projects/marsys-llm-api-01/", f"{RESOURCE}/openai/v1"),
        # Idempotent, so a caller who configured the full base is not doubled up.
        (f"{RESOURCE}/openai/v1", f"{RESOURCE}/openai/v1"),
        (f"{RESOURCE}/openai/v1/", f"{RESOURCE}/openai/v1"),
        ("https://marsys-dev-fn-01.openai.azure.com", "https://marsys-dev-fn-01.openai.azure.com/openai/v1"),
        ("  " + RESOURCE + "  ", f"{RESOURCE}/openai/v1"),
    ],
)
def test_base_url_normalizes_every_form_one_resource_arrives_as(given, expected):
    assert azure_openai_base_url(given) == expected


def test_an_explicit_base_url_is_normalized_like_every_other_spelling(monkeypatch):
    """The ``base_url`` argument and the ``endpoint`` argument are the same fact arriving
    under two names, so one of them cannot skip the normalizer: a caller passing the bare
    resource host would otherwise build a client that POSTs to ``/responses`` off the host
    root and 404s on every call."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    bare = AzureOpenAIAdapter(model_name="gpt-5.6-sol", api_key="k", base_url=RESOURCE)
    assert bare.get_endpoint_url() == f"{RESOURCE}/openai/v1/responses"
    project = AzureOpenAIAdapter(
        model_name="gpt-5.6-sol",
        api_key="k",
        base_url=f"{RESOURCE}/api/projects/marsys-llm-api-01",
    )
    assert project.get_endpoint_url() == f"{RESOURCE}/openai/v1/responses"
    # Still idempotent on the normalized form Spren actually threads in.
    already = AzureOpenAIAdapter(
        model_name="gpt-5.6-sol", api_key="k", base_url=f"{RESOURCE}/openai/v1"
    )
    assert already.get_endpoint_url() == f"{RESOURCE}/openai/v1/responses"


def test_base_url_is_empty_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    assert azure_openai_base_url() == ""


def test_base_url_reads_the_vendor_name_first_then_the_house_name(monkeypatch):
    monkeypatch.setenv("FOUNDRY_ENDPOINT", "https://house.services.ai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert azure_openai_base_url() == "https://house.services.ai.azure.com/openai/v1"

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", RESOURCE)
    assert azure_openai_base_url() == f"{RESOURCE}/openai/v1"


def test_endpoint_is_the_v1_responses_path_with_no_api_version(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", RESOURCE)
    url = AzureOpenAIAdapter(model_name="gpt-5.6-sol", api_key="k").get_endpoint_url()
    assert url == f"{RESOURCE}/openai/v1/responses"
    # The legacy surface's two markers. Both absent: no per-deployment path segment,
    # and no pinned api-version query — the v1 GA API requires neither.
    assert "/deployments/" not in url
    assert "api-version" not in url


# --- auth and identity -------------------------------------------------------


def test_auth_is_the_api_key_header_not_a_bearer():
    headers = _azure().get_headers()
    assert headers["api-key"] == "not-a-real-key"
    assert "Authorization" not in headers
    assert "x-api-key" not in headers


def test_both_adapters_report_the_azure_provider_id():
    assert _azure()._provider_name() == "azure"
    assert (
        AsyncAzureOpenAIAdapter(
            model_name="gpt-5.6-sol", api_key="k", base_url=f"{RESOURCE}/openai/v1"
        )._provider_name()
        == "azure"
    )


def test_the_deployment_name_travels_in_the_model_body_field():
    """Azure addresses a deployment, not a model snapshot; on the v1 surface the
    deployment name is the `model` field's value and the response echoes it back."""
    payload = _azure("gpt-5.6-terra").format_request_payload(MESSAGES)
    assert payload["model"] == "gpt-5.6-terra"


def test_the_registry_rows_agree_that_azure_exists():
    from typing import get_args

    assert "azure" in get_args(ApiProvider)
    assert "azure" in PROVIDER_BASE_URLS
    adapter = ProviderAdapterFactory.create_adapter(
        provider="azure", model_name="gpt-5.6-sol", api_key="k", base_url=f"{RESOURCE}/openai/v1"
    )
    assert isinstance(adapter, AzureOpenAIAdapter)


def test_a_config_for_this_provider_validates(monkeypatch):
    """Also the regression for the import-order trap: this endpoint is not in the
    module's literal table, it is read from the environment, and `PROVIDER_BASE_URLS`
    is built once at import — which happens before a daemon injects its configuration.
    Snapshotted, the entry stays empty for the life of the process and every request
    goes nowhere. The env vars here are deliberately set AFTER the import above."""
    from marsys.models.models import ModelConfig

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "resource-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", RESOURCE)
    assert PROVIDER_BASE_URLS["azure"] == "", "the snapshot is empty — that is the point"
    cfg = ModelConfig(type="api", provider="azure", name="gpt-5.6-sol")
    assert cfg.api_key == "resource-key"
    assert cfg.base_url == f"{RESOURCE}/openai/v1"


def test_an_unresolved_resource_says_the_endpoint_is_missing(monkeypatch):
    """A known provider with no resolvable endpoint is a different failure from a
    misspelled provider name, and the message has to say which: the fix here is to
    supply an endpoint, not to correct a spelling. (The misspelling case never
    reaches this branch — `provider` is a Literal, so Pydantic rejects it first.)"""
    from marsys.models.models import ModelConfig

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "resource-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    with pytest.warns(UserWarning, match="per-resource"):
        ModelConfig(type="api", provider="azure", name="gpt-5.6-sol")


# --- error classification ----------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


def test_an_azure_rate_limit_is_classified_the_way_a_first_party_one_is():
    """The classifier branches on the error *envelope*, and Azure's is OpenAI's. Left
    out of that branch, an Azure 429 matches nothing, classifies as UNKNOWN and comes
    back not retryable — the request is dropped instead of backed off, which on the
    interactive path looks like the model refusing to answer."""
    response = _FakeResponse(
        429,
        {"error": {"code": "429", "message": "Rate limit exceeded", "type": "rate_limit_error"}},
        {"retry-after": "13"},
    )
    error = _azure().handle_api_error(RuntimeError("429 Too Many Requests"), response=response)
    assert error.classification["category"] == "rate_limit"
    assert error.classification["is_retryable"] is True
    assert error.classification["retry_after"] == 13
    # And the vendor is named honestly, rather than reported as first-party OpenAI.
    assert error.provider == "azure"


def test_an_azure_server_error_is_retryable():
    error = _azure().handle_api_error(
        RuntimeError("503"), response=_FakeResponse(503, {"error": {"message": "busy"}})
    )
    assert error.classification["category"] == "service_unavailable"
    assert error.classification["is_retryable"] is True


# --- the wire sanitize -------------------------------------------------------


def test_internal_message_keys_never_reach_the_wire():
    """A caller's message dict is its own working object. This stack's reasoner
    annotates messages with routing and provenance fields, and the Responses API
    answers an unrecognized per-item key with `400 unknown_parameter: input[0].<key>`
    — so forwarding them is not untidy, it is a leg that cannot connect."""
    annotated = [
        {
            "role": "user",
            "content": "hi",
            "kind": "user_message",
            "actor": "principal",
            "cache_exempt": True,
            "name": "someone",
        }
    ]
    payload = _azure().format_request_payload(annotated)
    item = payload["input"][0]
    assert set(item) == {"role", "content"}
    assert item["role"] == "user"
    assert item["content"] == "hi"


def test_the_item_discriminator_survives_the_sanitize():
    """`type` is not an internal key here — the Responses input array uses it to tell
    a message from a function_call, so an allow-list that dropped it would break
    tool round-trips."""
    payload = _azure().format_request_payload(
        [{"type": "function_call_output", "call_id": "c1", "role": "user", "content": "x"}]
    )
    item = payload["input"][0]
    assert item["type"] == "function_call_output"
    assert "call_id" not in item


def test_the_sanitize_is_the_shared_openai_behaviour_not_an_azure_special_case():
    """One wire contract, one implementation. A fix that lived only on the subclass
    would leave the first-party leg rejecting the same annotated messages."""
    first_party = OpenAIAdapter(
        model_name="gpt-5.5", api_key="k", base_url="https://api.openai.com/v1/"
    )
    payload = first_party.format_request_payload(
        [{"role": "user", "content": "hi", "kind": "user_message"}]
    )
    assert set(payload["input"][0]) == {"role", "content"}


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5.4-mini", "gpt-5.5"])
def test_a_deployment_before_the_5_6_generation_is_told_nothing_about_caching(model):
    """Models before the GPT-5.6 family answer either cache field with a 400, so the
    inherited builder gates both on the deployment name. An earlier reading of this
    surface had the fields rejected in every position and concluded that the correct
    request says nothing at all; twenty live requests on a 5.6 deployment refuted the
    general claim, and this is the part of it that survived."""
    payload = _azure(model_name=model).format_request_payload(MESSAGES)
    assert "prompt_cache_options" not in payload
    assert not any("cache" in key for key in payload)
    assert "prompt_cache_breakpoint" not in str(payload)


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_a_5_6_deployment_asks_for_explicit_mode_and_carries_a_breakpoint(model):
    """The control for the gate above, on the deployment names this fleet actually
    runs. The breakpoint rides inside a content block, which is the placement the live
    requests accepted; at item level or request level it is an unknown parameter."""
    payload = _azure(model_name=model).format_request_payload(MESSAGES)
    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert payload["input"][0]["content"] == [{
        "type": "input_text",
        "text": "hi",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }]


# --- the thinking knob -------------------------------------------------------


@pytest.mark.parametrize(
    "budget, expected",
    [
        (None, None),
        (0, None),  # thinking off is not "think as little as possible"
        (-1, None),
        (512, "minimal"),
        (1024, "low"),
        (4095, "low"),
        (4096, "medium"),
        (8192, "medium"),
        (16384, "high"),
        (32768, "high"),
    ],
)
def test_a_thinking_budget_selects_a_reasoning_effort(budget, expected):
    assert thinking_budget_to_effort(budget) == expected


def test_the_configured_thinking_budget_reaches_this_leg():
    """The only deliberation knob this stack exposes is a token budget. Unmapped, a
    caller's setting is inert and every call runs at the provider default (`medium`),
    which is indistinguishable from the knob working."""
    payload = _azure().format_request_payload(MESSAGES, thinking_budget=32768)
    assert payload["reasoning"] == {"effort": "high"}


def test_an_explicit_effort_beats_the_budget():
    payload = _azure().format_request_payload(
        MESSAGES, thinking_budget=32768, reasoning_effort="low"
    )
    assert payload["reasoning"] == {"effort": "low"}


def test_thinking_off_sends_no_reasoning_block():
    payload = _azure().format_request_payload(MESSAGES, thinking_budget=0)
    assert "reasoning" not in payload


def test_the_budget_kwarg_does_not_warn_as_unknown():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _azure().format_request_payload(MESSAGES, thinking_budget=8192)


def test_the_smallest_budget_asks_for_an_effort_this_surface_actually_serves():
    """`minimal` is a 400 here — the endpoint's own reply lists none / low / medium /
    high / xhigh / max — so the smallest bucket has to arrive as the smallest this
    surface has. It must still ask for reasoning: a positive budget is "think a little",
    and `none` would answer a question nobody asked."""
    payload = _azure().format_request_payload(MESSAGES, thinking_budget=512)
    assert payload["reasoning"] == {"effort": "low"}


def test_an_explicit_minimal_is_substituted_too():
    """The caller who names the effort outright is on the same endpoint as the one who
    named a budget, and it rejects the value for both of them."""
    payload = _azure().format_request_payload(MESSAGES, reasoning_effort="minimal")
    assert payload["reasoning"] == {"effort": "low"}


def test_the_first_party_leg_still_sends_minimal():
    """The control, and the scope line: `minimal` is served by OpenAI's own endpoint and
    the substitution above belongs to this re-hosting surface, not to the shared payload
    builder. A run of this file that changed the first-party leg would be a silent change
    to every OpenAI caller in the stack."""
    payload = OpenAIAdapter(
        model_name="gpt-5.6", api_key="k", base_url="https://api.openai.com/v1"
    ).format_request_payload(MESSAGES, thinking_budget=512)
    assert payload["reasoning"] == {"effort": "minimal"}
    codex = OpenAIAdapter(
        model_name="gpt-5.6-codex", api_key="k", base_url="https://api.openai.com/v1"
    ).format_request_payload(MESSAGES, thinking_budget=512)
    assert codex["reasoning"] == {"effort": "low"}  # the pre-existing codex exception


# --- the meter ---------------------------------------------------------------


def _harmonized(usage: dict):
    raw = {
        "id": "resp_1",
        "model": "gpt-5.6-sol",
        "created_at": 1,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": usage,
    }
    return _azure().harmonize_response(raw, request_start_time=0.0)


def test_a_cached_prefix_is_not_counted_twice():
    """Measured on the resource: `input_tokens: 3398` CONTAINING `cached_tokens: 3395`,
    with `3398 + output 5 == total 3403`. `UsageInfo.prompt_tokens` is the inverse
    convention — the uncached remainder, with the cache figures beside it — so the
    slice has to be subtracted once here. Mapped straight across, the cached prefix
    would be billed as fresh input on top of being billed as a cache read."""
    usage = _harmonized(
        {
            "input_tokens": 3398,
            "output_tokens": 5,
            "total_tokens": 3403,
            "input_tokens_details": {"cached_tokens": 3395},
        }
    ).metadata.usage
    assert usage.prompt_tokens == 3
    assert usage.cache_read_input_tokens == 3395
    assert usage.cache_creation_input_tokens is None
    # The whole prompt is recoverable, exactly: nothing lost, nothing doubled.
    assert usage.full_prompt_tokens == 3398


def test_a_cache_write_is_split_out_the_same_way():
    """The first of the measured pair: the same 3395 tokens, reported as a write."""
    usage = _harmonized(
        {
            "input_tokens": 3398,
            "output_tokens": 5,
            "total_tokens": 3403,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3395},
        }
    ).metadata.usage
    assert usage.prompt_tokens == 3
    assert usage.cache_creation_input_tokens == 3395
    assert usage.cache_read_input_tokens is None
    assert usage.full_prompt_tokens == 3398


def test_slices_that_exceed_their_whole_do_not_walk_the_ledger_backwards():
    usage = _harmonized(
        {
            "input_tokens": 100,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 400},
        }
    ).metadata.usage
    assert usage.prompt_tokens == 0


def test_a_provider_that_reports_no_cache_leaves_both_fields_unset():
    usage = _harmonized(
        {"input_tokens": 75, "output_tokens": 1186, "total_tokens": 1261}
    ).metadata.usage
    assert usage.prompt_tokens == 75
    assert usage.cache_read_input_tokens is None
    assert usage.cache_creation_input_tokens is None
    assert usage.full_prompt_tokens == 75


def test_reasoning_tokens_are_recorded_as_a_subset_of_output():
    """The vendor's own arithmetic proves the containment: `input 75 + output 1186 ==
    total 1261`, with `reasoning_tokens: 1024` reported inside the 1186. A meter that
    adds the two charges the reasoning slice twice — here, 86% high."""
    usage = _harmonized(
        {
            "input_tokens": 75,
            "output_tokens": 1186,
            "total_tokens": 1261,
            "output_tokens_details": {"reasoning_tokens": 1024},
        }
    ).metadata.usage
    assert usage.completion_tokens == 1186
    assert usage.reasoning_tokens == 1024
    assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens == 1261


def test_the_harmonized_response_reports_azure_and_the_echoed_deployment():
    """`metadata.model` is what a cost table is keyed on, and a table miss prices at
    zero silently. Azure echoes the DEPLOYMENT name (measured: `gpt-5.6-sol` for a
    request made with `gpt-5.6-sol`), so the echo needs no renormalization — but the
    provider label must be this leg's, not the first-party one's."""
    response = _harmonized({"input_tokens": 3, "output_tokens": 5, "total_tokens": 8})
    assert response.metadata.provider == "azure"
    assert response.metadata.model == "gpt-5.6-sol"


def test_chat_completions_usage_names_still_read():
    """The same harmonizer serves any OpenAI-compatible endpoint; the older field
    names must not silently become zeros."""
    usage = _harmonized(
        {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 47}
    ).metadata.usage
    assert usage.prompt_tokens == 40
    assert usage.completion_tokens == 7
