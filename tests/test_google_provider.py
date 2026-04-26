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
            content=[
                ToolUseBlock(id="call-abc", name="get_weather", input={"city": "Oslo"})
            ],
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_use_id="call-abc", content="cold")],
        ),
    ]
    api_kwargs = p._build_kwargs(
        model="gemini-2.5-pro",
        messages=messages,
        system_blocks=[],
        tools=[],
        tool_choice="auto",
        max_tokens=128,
        temperature=None,
        stop=None,
        provider_settings={},
        model_settings={},
        convo_settings={},
        per_call_overrides={},
    )
    tool_part = api_kwargs["contents"][-1]["parts"][0]
    assert tool_part["function_response"]["name"] == "get_weather"


def test_assistant_text_round_trip_unchanged():
    p = _bare_provider()
    m = Message(role="assistant", content=[TextBlock("hello")])
    out = p._message_to_api(m, {})
    assert out == [{"role": "model", "parts": [{"text": "hello"}]}]
