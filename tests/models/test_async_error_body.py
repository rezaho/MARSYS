"""B2 regression: the async API path must surface a 4xx provider error
body, not the opaque aiohttp reason phrase "Bad Request".

Two coupled roots, both must be fixed or B2 is half-fixed:
  R1 — the async path discarded the live response body (passed
       response=None to handle_api_error). Fixed by reading the body
       in-frame and wrapping it in `_CapturedErrorResponse`, shaped like
       the requests.Response the sync path already passes.
  R2 — the anthropic error mapper had no generic-400 arm, so even with a
       body a 400 classified UNKNOWN. Fixed by an INVALID_REQUEST arm.

No network: the shim is the exact object the async frame builds, so
exercising handle_api_error with it is a faithful unit of the fixed path.
"""

import pytest
from multidict import CIMultiDict

from marsys.models.adapters.anthropic import AnthropicAdapter
from marsys.models.adapters.base import _CapturedErrorResponse

_BODY = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "messages.0.name: Extra inputs are not permitted",
    },
}


class _FakeClientResponseError(Exception):
    """Mimics the aiohttp.ClientResponseError surface from_provider_response
    reads (`.message` = reason phrase, `.status`) — i.e. the body-less
    exception the async path raises via response.raise_for_status()."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _adapter() -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name="claude-haiku-4-5-20251001",
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
    )


def test_captured_error_response_shim_contract():
    shim = _CapturedErrorResponse(
        status_code=400, body=_BODY, headers=CIMultiDict({"Request-Id": "req_x"})
    )
    assert shim.status_code == 400
    # idempotent: handle_api_error reads .json() once in the mapper and
    # again in the anthropic wrapper (INVALID_REQUEST is non-critical).
    assert shim.json() == _BODY
    assert shim.json() == _BODY
    # case-insensitive headers (aiohttp CIMultiDict snapshot): the wrapper
    # reads "request-id" though Anthropic sends "Request-Id".
    assert shim.headers.get("request-id") == "req_x"
    # no body captured -> ValueError, which from_provider_response/the
    # wrapper already swallow into raw_response=None (existing behavior).
    with pytest.raises(ValueError):
        _CapturedErrorResponse(status_code=400, body=None, headers=CIMultiDict()).json()


def test_async_4xx_body_surfaces_with_shim_R1_and_R2():
    """Joint R1+R2 regression guard. Fails on pre-fix code (R1: body lost
    -> message stays 'Bad Request'; R2: no 400 arm -> classification
    'unknown'); passes only with BOTH parts."""
    err = _FakeClientResponseError(status=400, message="Bad Request")
    shim = _CapturedErrorResponse(
        status_code=400, body=_BODY, headers=CIMultiDict({"request-id": "req_x"})
    )

    result = _adapter().handle_api_error(err, response=shim)

    # R1: the provider's descriptive body reached the user (not "Bad Request")
    assert result.error == "messages.0.name: Extra inputs are not permitted"
    assert result.error != "Bad Request"
    # R2: a generic anthropic 400 classifies as invalid_request, terminal
    assert result.classification["category"] == "invalid_request"
    assert result.classification["is_retryable"] is False


def test_body_less_400_classifies_invalid_request():
    """A 400 whose body could not be captured (response=None, status from
    the exception): message falls back to the reason phrase (nothing to
    recover — preserved), and classification is invalid_request. The latter
    is CORRECT, not a regression — a 400 is an invalid request whether or
    not the body was readable; pre-fix it wrongly stayed 'unknown'."""
    err = _FakeClientResponseError(status=400, message="Bad Request")
    result = _adapter().handle_api_error(err, response=None)
    assert result.error == "Bad Request"
    assert result.classification["category"] == "invalid_request"
    assert result.classification["is_retryable"] is False


def test_true_connection_error_no_status_stays_unknown():
    """A genuine connection failure (no HTTP response, no status anywhere):
    classification stays unknown and the message is the exception text.
    This is the real no-regression case for 'no response at all'."""
    result = _adapter().handle_api_error(
        ConnectionError("Cannot connect to host api.anthropic.com"), response=None
    )
    assert "Cannot connect to host" in result.error
    assert result.classification["category"] == "unknown"
