"""GoogleProvider wire-format conversions that don't need a live client."""

from __future__ import annotations

from llmfacade.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
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
