"""Live Google (Gemini) tool roundtrip. Skips when ``GOOGLE_API_KEY`` is unset.

Specifically exercises bug #1 from review.md: ``function_response.name`` must
match the original ``function_call.name``. If it regresses, Gemini will return
an error or ignore the tool result and the second turn won't reference it."""

from __future__ import annotations

import pytest

from llmfacade import LLM, ToolResultBlock, ToolUseBlock, tool
from llmfacade.helpers import run_to_completion

pytestmark = pytest.mark.integration


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"Weather in {city}: sunny, 72F."


def _content_blocks(convo) -> list:
    out = []
    for m in convo.history:
        if isinstance(m.content, list):
            out.extend(m.content)
    return out


@pytest.mark.usefixtures("google_api_key")
def test_google_tool_roundtrip() -> None:
    llm = LLM()
    provider = llm.new_provider("google")
    model = provider.new_model("gemini-2.5-flash", max_tokens=512)
    convo = model.new_conversation(tools=[get_weather])

    final = run_to_completion(convo, "What's the weather in Paris? Use the tool.")

    assert final.tool_calls == []
    assert final.text.strip()

    blocks = _content_blocks(convo)
    tool_uses = [b for b in blocks if isinstance(b, ToolUseBlock)]
    tool_results = [b for b in blocks if isinstance(b, ToolResultBlock)]
    assert tool_uses, "model never called the tool"
    assert any(b.name == "get_weather" for b in tool_uses)
    assert tool_results, "tool result not appended to history"
    assert any("72F" in (b.content or "") for b in tool_results)
