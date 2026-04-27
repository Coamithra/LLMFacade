"""Live Anthropic tool roundtrip. Skips when ``ANTHROPIC_API_KEY`` is unset."""

from __future__ import annotations

import pytest

from llmfacade import LLM, ToolResultBlock, ToolUseBlock, tool
from llmfacade.helpers import run_to_completion
from llmfacade.providers.anthropic import AnthropicModel, AnthropicProvider

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


@pytest.mark.usefixtures("anthropic_api_key")
def test_anthropic_tool_roundtrip() -> None:
    llm = LLM()
    provider = AnthropicProvider(manager=llm)
    model = provider.new_model(AnthropicModel.HAIKU_4_5, max_tokens=512)
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
