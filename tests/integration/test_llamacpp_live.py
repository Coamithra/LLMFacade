"""Live llama-server tool roundtrip + slot-save/restore/erase smoke. Skips
when no llama-server is reachable.

Defaults to ``http://localhost:8080`` and ``qwen2.5-3b-instruct-q4_k_m``.
Override via ``LLAMACPP_HOST`` and ``LLAMACPP_MODEL`` env vars (see
``.env.example``).

The save/restore test assumes the server was launched with
``--slot-save-path <some-dir>``; if that flag is absent llama-server
returns 500 and the test is skipped with an explanatory message."""

from __future__ import annotations

import uuid
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from llmfacade import LLM, ToolResultBlock, ToolUseBlock, tool
from llmfacade.exceptions import ProviderError
from llmfacade.helpers import run_to_completion
from llmfacade.providers.llamacpp import LlamaCppServerProvider

pytestmark = pytest.mark.integration


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"Weather in {city}: sunny, 72F."


def _server_reachable(host: str) -> bool:
    try:
        with urlopen(f"{host.rstrip('/')}/health", timeout=2) as _:
            return True
    except (URLError, OSError, ValueError):
        return False


def _content_blocks(convo) -> list:
    out = []
    for m in convo.history:
        if isinstance(m.content, list):
            out.extend(m.content)
    return out


def _base_url_for(host: str) -> str:
    h = host.rstrip("/")
    return h if h.endswith("/v1") else f"{h}/v1"


def test_llamacpp_tool_roundtrip(llamacpp_host: str, llamacpp_model: str) -> None:
    if not _server_reachable(llamacpp_host):
        pytest.skip(f"llama-server not reachable at {llamacpp_host}")

    llm = LLM()
    provider = llm.new_provider("llamacpp", base_url=_base_url_for(llamacpp_host))
    model = provider.new_model(llamacpp_model, max_tokens=512)
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


def test_llamacpp_save_restore_erase_roundtrip(llamacpp_host: str, llamacpp_model: str) -> None:
    if not _server_reachable(llamacpp_host):
        pytest.skip(f"llama-server not reachable at {llamacpp_host}")

    llm = LLM()
    provider: LlamaCppServerProvider = llm.new_provider(
        "llamacpp", base_url=_base_url_for(llamacpp_host)
    )  # type: ignore[assignment]
    model = provider.new_model(llamacpp_model, max_tokens=64)
    convo = model.new_conversation()
    # Prime slot 0 with some prompt so the saved file is non-empty.
    convo.send("Say hello in three words.")

    filename = f"warmup-{uuid.uuid4().hex[:8]}.bin"
    try:
        save = provider.save_slot(0, filename)
    except ProviderError as e:
        pytest.skip(f"save_slot failed (server likely launched without --slot-save-path): {e}")
    assert isinstance(save, dict)

    restore = provider.restore_slot(0, filename)
    assert isinstance(restore, dict)

    erase = provider.erase_slot(0)
    assert isinstance(erase, dict)
