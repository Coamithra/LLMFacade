"""Live Ollama tool roundtrip. Skips when no Ollama server is reachable.

Defaults to ``http://localhost:11434`` and ``qwen3.5:4b``. Override via
``OLLAMA_HOST`` and ``OLLAMA_TEST_MODEL`` env vars (see ``.env.example``)."""

from __future__ import annotations

from urllib.error import URLError
from urllib.request import urlopen

import pytest

from llmfacade import LLM, ToolResultBlock, ToolUseBlock, tool
from llmfacade.helpers import run_to_completion

pytestmark = pytest.mark.integration


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"Weather in {city}: sunny, 72F."


def _server_reachable(host: str) -> bool:
    try:
        with urlopen(host, timeout=2) as _:
            return True
    except (URLError, OSError, ValueError):
        return False


def _content_blocks(convo) -> list:
    out = []
    for m in convo.history:
        if isinstance(m.content, list):
            out.extend(m.content)
    return out


def test_ollama_tool_roundtrip(ollama_host: str, ollama_model: str) -> None:
    if not _server_reachable(ollama_host):
        pytest.skip(f"Ollama not reachable at {ollama_host}")

    llm = LLM()
    provider = llm.new_provider("ollama", base_url=ollama_host)
    model = provider.new_model(ollama_model, max_tokens=512)
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
