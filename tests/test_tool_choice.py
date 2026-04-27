"""Provider-specific `tool_choice` translation through `_build_kwargs`.

Drives a `CompletionRequest` through each provider's `_build_kwargs` and
asserts the SDK-shaped `tool_choice` it produces. Covers all four call-site
values (`"auto"`, `"required"`, `"none"`, named tool) for the providers that
support forced selection (Anthropic, OpenAI, Google), the gating rule that
`tool_choice` config is only emitted when `req.tools` is non-empty, and the
cascade-level rejection for Ollama (which has no forced-selection API)."""

from __future__ import annotations

import pytest

from llmfacade import tool
from llmfacade.exceptions import UnsupportedFeature
from llmfacade.provider import CompletionRequest
from llmfacade.providers.anthropic import AnthropicProvider
from llmfacade.providers.google import GoogleProvider
from llmfacade.providers.ollama import OllamaProvider
from llmfacade.providers.openai import OpenAIProvider


@tool
def forge_item(item: str) -> str:
    """Forge an item."""
    return item


def _req(tool_choice: str | None, *, with_tool: bool = True) -> CompletionRequest:
    settings: dict[str, object] = {"max_tokens": 100}
    sources: dict[str, str] = {"max_tokens": "convo"}
    if tool_choice is not None:
        settings["tool_choice"] = tool_choice
        sources["tool_choice"] = "convo"
    return CompletionRequest(
        model="some-model",
        messages=[],
        system_blocks=[],
        tools=[forge_item] if with_tool else [],
        stop=None,
        settings=settings,
        settings_source=sources,
    )


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key")


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


@pytest.fixture
def google_provider() -> GoogleProvider:
    return GoogleProvider(api_key="test-key")


@pytest.fixture
def ollama_provider() -> OllamaProvider:
    # Ollama doesn't take an api_key, but constructor still requires init; use
    # a fake host to avoid hitting a real server.
    return OllamaProvider(base_url="http://invalid.local:0")


# --- Anthropic ---


def test_anthropic_tool_choice_auto(anthropic_provider: AnthropicProvider):
    kwargs = anthropic_provider._build_kwargs(_req("auto"))
    assert kwargs["tool_choice"] == {"type": "auto"}


def test_anthropic_tool_choice_default_is_auto(anthropic_provider: AnthropicProvider):
    # No tool_choice set anywhere — provider should fall back to "auto".
    kwargs = anthropic_provider._build_kwargs(_req(None))
    assert kwargs["tool_choice"] == {"type": "auto"}


def test_anthropic_tool_choice_required(anthropic_provider: AnthropicProvider):
    kwargs = anthropic_provider._build_kwargs(_req("required"))
    assert kwargs["tool_choice"] == {"type": "any"}


def test_anthropic_tool_choice_none(anthropic_provider: AnthropicProvider):
    # Anthropic's API has a real "none" mode; we now translate to it instead
    # of the prior no-op {"type": "auto", "disable_parallel_tool_use": False}.
    kwargs = anthropic_provider._build_kwargs(_req("none"))
    assert kwargs["tool_choice"] == {"type": "none"}


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


def test_openai_tool_choice_default_is_auto(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req(None))
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


# --- Google ---


def test_google_tool_choice_auto_omits_config(google_provider: GoogleProvider):
    # AUTO is the SDK default; we omit tool_config to keep the request clean.
    kwargs = google_provider._build_kwargs(_req("auto"))
    assert "tool_config" not in kwargs["config"]


def test_google_tool_choice_default_omits_config(google_provider: GoogleProvider):
    kwargs = google_provider._build_kwargs(_req(None))
    assert "tool_config" not in kwargs["config"]


def test_google_tool_choice_required(google_provider: GoogleProvider):
    kwargs = google_provider._build_kwargs(_req("required"))
    assert kwargs["config"]["tool_config"] == {"function_calling_config": {"mode": "ANY"}}


def test_google_tool_choice_none(google_provider: GoogleProvider):
    kwargs = google_provider._build_kwargs(_req("none"))
    assert kwargs["config"]["tool_config"] == {"function_calling_config": {"mode": "NONE"}}


def test_google_tool_choice_named_tool(google_provider: GoogleProvider):
    kwargs = google_provider._build_kwargs(_req("forge_item"))
    assert kwargs["config"]["tool_config"] == {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": ["forge_item"],
        }
    }


def test_google_tool_choice_omitted_without_tools(google_provider: GoogleProvider):
    # No tools registered → no tool_config emitted regardless of tool_choice.
    kwargs = google_provider._build_kwargs(_req("forge_item", with_tool=False))
    assert "tool_config" not in kwargs["config"]


# --- Ollama: cascade rejects tool_choice entirely ---


def test_ollama_does_not_advertise_tool_choice():
    assert "tool_choice" not in OllamaProvider.SUPPORTS
    # but it does advertise basic tool calling
    assert "tools" in OllamaProvider.SUPPORTS


def test_ollama_tool_choice_at_provider_raises():
    # Setting tool_choice at any layer raises at that layer because the knob
    # isn't in Ollama's SUPPORTS — same mechanism as e.g. `thinking` on Ollama.
    with pytest.raises(UnsupportedFeature):
        OllamaProvider(base_url="http://invalid.local:0", tool_choice="required")


@pytest.mark.parametrize("value", ["auto", "required", "none", "forge_item"])
def test_ollama_tool_choice_at_model_raises(value: str):
    provider = OllamaProvider(base_url="http://invalid.local:0")
    with pytest.raises(UnsupportedFeature):
        provider.new_model("llama3", tool_choice=value)


def test_ollama_tools_can_be_disabled_per_model():
    # Models that don't support tool calling at all use capability_override
    # to drop "tools" from their effective SUPPORTS.
    provider = OllamaProvider(base_url="http://invalid.local:0")
    narrow_model = provider.new_model("llama2", capability_override=provider.SUPPORTS - {"tools"})
    assert "tools" not in narrow_model.get_capabilities()


# --- Conversation-level validation ---


def test_conversation_named_tool_must_match_registered_tool():
    """Validator runs at request-build time when the cascade is fully resolved."""
    from llmfacade.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    model = provider.new_model("gpt-4o-mini")
    convo = model.new_conversation(tools=[forge_item], tool_choice="not_a_real_tool")
    with pytest.raises(ValueError, match="not_a_real_tool"):
        convo._build_request(stop=None, per_call={})


def test_conversation_forced_choice_requires_tools():
    provider = OpenAIProvider(api_key="test-key")
    model = provider.new_model("gpt-4o-mini")
    convo = model.new_conversation(tool_choice="required")
    with pytest.raises(ValueError, match="requires tools"):
        convo._build_request(stop=None, per_call={})


def test_conversation_tools_unsupported_raises_at_construction():
    """Convo construction blocks tools=[...] when the model can't do tool calling."""
    provider = OllamaProvider(base_url="http://invalid.local:0")
    narrow_model = provider.new_model("llama2", capability_override=provider.SUPPORTS - {"tools"})
    with pytest.raises(UnsupportedFeature):
        narrow_model.new_conversation(tools=[forge_item])


def test_conversation_tool_choice_cascades_from_convo_to_request():
    """tool_choice set at convo-level reaches the provider via req.settings."""
    provider = OpenAIProvider(api_key="test-key")
    model = provider.new_model("gpt-4o-mini")
    convo = model.new_conversation(tools=[forge_item], tool_choice="required")
    req = convo._build_request(stop=None, per_call={})
    assert req.settings["tool_choice"] == "required"
    assert req.settings_source["tool_choice"] == "convo"
