"""Provider-specific `tool_choice` translation through `_build_kwargs`.

Drives a `CompletionRequest` through each provider's `_build_kwargs` and
asserts the SDK-shaped `tool_choice` it produces. Covers all four call-site
values (`"auto"`, `"required"`, `"none"`, named tool) and the gating rule
that `tool_choice` is only emitted when `req.tools` is non-empty.

Google and Ollama silently drop `tool_choice` today (tracked in review.md
as #33); not asserted here."""

from __future__ import annotations

import pytest

from llmfacade import tool
from llmfacade.provider import CompletionRequest
from llmfacade.providers.anthropic import AnthropicProvider
from llmfacade.providers.openai import OpenAIProvider


@tool
def forge_item(item: str) -> str:
    """Forge an item."""
    return item


def _req(tool_choice: str, *, with_tool: bool = True) -> CompletionRequest:
    return CompletionRequest(
        model="some-model",
        messages=[],
        system_blocks=[],
        tools=[forge_item] if with_tool else [],
        tool_choice=tool_choice,
        stop=None,
        settings={"max_tokens": 100},
        settings_source={"max_tokens": "convo"},
    )


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key")


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


# --- Anthropic ---


def test_anthropic_tool_choice_auto(anthropic_provider: AnthropicProvider):
    kwargs = anthropic_provider._build_kwargs(_req("auto"))
    assert kwargs["tool_choice"] == {"type": "auto"}


def test_anthropic_tool_choice_required(anthropic_provider: AnthropicProvider):
    kwargs = anthropic_provider._build_kwargs(_req("required"))
    assert kwargs["tool_choice"] == {"type": "any"}


def test_anthropic_tool_choice_none(anthropic_provider: AnthropicProvider):
    kwargs = anthropic_provider._build_kwargs(_req("none"))
    assert kwargs["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": False,
    }


def test_anthropic_tool_choice_named_tool(anthropic_provider: AnthropicProvider):
    kwargs = anthropic_provider._build_kwargs(_req("forge_item"))
    assert kwargs["tool_choice"] == {"type": "tool", "name": "forge_item"}


def test_anthropic_tool_choice_omitted_without_tools(
    anthropic_provider: AnthropicProvider,
):
    kwargs = anthropic_provider._build_kwargs(_req("forge_item", with_tool=False))
    assert "tool_choice" not in kwargs


# --- OpenAI ---


def test_openai_tool_choice_auto(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req("auto"))
    assert kwargs["tool_choice"] == "auto"


def test_openai_tool_choice_required(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req("required"))
    assert kwargs["tool_choice"] == "required"


def test_openai_tool_choice_none(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req("none"))
    assert kwargs["tool_choice"] == "none"


def test_openai_tool_choice_named_tool(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req("forge_item"))
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "forge_item"},
    }


def test_openai_tool_choice_omitted_without_tools(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req("forge_item", with_tool=False))
    assert "tool_choice" not in kwargs
