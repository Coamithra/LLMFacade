"""GoogleProvider wire-format conversions that don't need a live client."""

from __future__ import annotations

import pytest

from llmfacade.exceptions import AuthenticationError, ProviderError, RateLimitError
from llmfacade.models import ImageBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from llmfacade.providers.google import GoogleProvider


def _bare_provider() -> GoogleProvider:
    # Skip __init__ so we don't need google-genai installed or an API key.
    return GoogleProvider.__new__(GoogleProvider)


def test_tool_result_uses_function_name_when_set():
    p = _bare_provider()
    m = Message(
        role="tool",
        content=[
            ToolResultBlock(
                tool_use_id="call-abc",
                content="42",
                name="get_weather",
            )
        ],
    )
    out = p._message_to_api(m, {})
    assert out == [
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"content": "42"},
                    }
                }
            ],
        }
    ]


def test_tool_result_falls_back_to_history_lookup():
    p = _bare_provider()
    history_lookup = {"call-abc": "get_weather"}
    m = Message(
        role="tool",
        content=[ToolResultBlock(tool_use_id="call-abc", content="42")],
    )
    out = p._message_to_api(m, history_lookup)
    assert out[0]["parts"][0]["function_response"]["name"] == "get_weather"


def test_build_kwargs_threads_name_through_history():
    p = _bare_provider()
    messages = [
        Message(role="user", content="what's the weather?"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call-abc", name="get_weather", input={"city": "Oslo"})],
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_use_id="call-abc", content="cold")],
        ),
    ]
    from llmfacade.provider import CompletionRequest

    req = CompletionRequest(
        model="gemini-2.5-pro",
        messages=messages,
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"max_tokens": 128},
    )
    api_kwargs = p._build_kwargs(req)
    tool_part = api_kwargs["contents"][-1]["parts"][0]
    assert tool_part["function_response"]["name"] == "get_weather"


def test_assistant_text_round_trip_unchanged():
    p = _bare_provider()
    m = Message(role="assistant", content=[TextBlock("hello")])
    out = p._message_to_api(m, {})
    assert out == [{"role": "model", "parts": [{"text": "hello"}]}]


def test_usage_extracts_thoughts_token_count():
    """Gemini reports thinking tokens in ``thoughts_token_count``; they map to
    ``Usage.reasoning_tokens`` and are folded into the total."""
    from types import SimpleNamespace

    p = _bare_provider()
    raw = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=15,
            cached_content_token_count=0,
            total_token_count=45,
        )
    )
    u = p._usage_from(raw)
    assert u is not None
    assert u.reasoning_tokens == 15
    assert u.total_tokens == 45


def test_tool_result_list_content_is_flattened():
    """List-form ToolResultBlock content reaches function_response as flattened
    text instead of being silently discarded (matches OpenAI/llamacpp)."""
    p = _bare_provider()
    m = Message(
        role="tool",
        content=[
            ToolResultBlock(
                tool_use_id="call-abc",
                content=[TextBlock("part one. "), TextBlock("part two.")],
                name="get_weather",
            )
        ],
    )
    out = p._message_to_api(m, {})
    assert out[0]["parts"][0]["function_response"]["response"]["content"] == (
        "part one. part two."
    )


def test_tool_result_list_content_ignores_non_text_blocks():
    """Non-text blocks in a list-form tool result are skipped, matching the
    other providers' flatten_text_blocks behaviour. (An ImageBlock here is
    normally rejected upstream by the "tool_result_images" request-time gate.)"""
    p = _bare_provider()
    m = Message(
        role="tool",
        content=[
            ToolResultBlock(
                tool_use_id="call-abc",
                content=[
                    TextBlock("see image"),
                    ImageBlock(data=b"\x89PNG", media_type="image/png"),
                ],
                name="screenshot",
            )
        ],
    )
    out = p._message_to_api(m, {})
    assert out[0]["parts"][0]["function_response"]["response"]["content"] == "see image"


def test_build_kwargs_marshals_list_form_tool_result():
    """Convo-level marshaling: a list-form tool result in history produces the
    flattened text in function_response via _build_kwargs."""
    from llmfacade.provider import CompletionRequest

    p = _bare_provider()
    messages = [
        Message(role="user", content="what's the weather?"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call-abc", name="get_weather", input={"city": "Oslo"})],
        ),
        Message(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_use_id="call-abc",
                    content=[TextBlock("cold"), TextBlock(" and windy")],
                )
            ],
        ),
    ]
    req = CompletionRequest(
        model="gemini-2.5-pro",
        messages=messages,
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"max_tokens": 128},
    )
    api_kwargs = p._build_kwargs(req)
    tool_part = api_kwargs["contents"][-1]["parts"][0]
    assert tool_part["function_response"]["response"]["content"] == "cold and windy"


def test_usage_total_falls_back_to_sum_including_thoughts():
    """With no ``total_token_count`` reported, the total includes thoughts."""
    from types import SimpleNamespace

    p = _bare_provider()
    raw = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=15,
        )
    )
    u = p._usage_from(raw)
    assert u is not None
    assert u.reasoning_tokens == 15
    assert u.total_tokens == 45


class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError: structured .code / .status."""

    def __init__(self, code: int | None, status: str | None, message: str = "boom"):
        self.code = code
        self.status = status
        super().__init__(f"{code} {status}. {message}")


@pytest.mark.parametrize(
    ("code", "status", "expected"),
    [
        (401, "UNAUTHENTICATED", AuthenticationError),
        (403, "PERMISSION_DENIED", AuthenticationError),
        (429, "RESOURCE_EXHAUSTED", RateLimitError),
        (500, "INTERNAL", ProviderError),
    ],
)
def test_reraise_classifies_on_structured_fields(code, status, expected):
    p = _bare_provider()
    with pytest.raises(expected):
        p._reraise(FakeAPIError(code, status))


def test_reraise_classifies_on_status_alone():
    """A missing HTTP code still classifies via the gRPC-style status string."""
    p = _bare_provider()
    with pytest.raises(RateLimitError):
        p._reraise(FakeAPIError(None, "RESOURCE_EXHAUSTED"))
    with pytest.raises(AuthenticationError):
        p._reraise(FakeAPIError(None, "PERMISSION_DENIED"))


def test_reraise_plain_exception_maps_to_provider_error_with_original():
    p = _bare_provider()
    original = Exception("something broke")
    with pytest.raises(ProviderError) as exc_info:
        p._reraise(original)
    assert exc_info.value.original is original


def test_reraise_plain_exception_falls_back_to_message_keywords():
    p = _bare_provider()
    with pytest.raises(AuthenticationError):
        p._reraise(Exception("API key not valid. Please pass a valid API key."))
    with pytest.raises(RateLimitError):
        p._reraise(Exception("Quota exceeded, please retry later."))


def test_reraise_does_not_keyword_match_structured_errors():
    """An APIError with a real code/status never falls into the message-keyword
    fallback (a 500 whose message mentions 'quota' stays a ProviderError)."""
    p = _bare_provider()
    with pytest.raises(ProviderError):
        p._reraise(FakeAPIError(500, "INTERNAL", "backend quota service crashed"))
