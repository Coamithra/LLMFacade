"""Cache tokens in Usage and auto_cache_last_user routing."""

from __future__ import annotations

import pytest

from llmfacade import SystemBlock, UnsupportedFeature, Usage

from .conftest import MockProvider


def test_usage_carries_cache_fields():
    u = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cache_creation_tokens=200,
        cache_read_tokens=80,
    )
    assert u.cache_creation_tokens == 200
    assert u.cache_read_tokens == 80


def test_usage_defaults_to_zero():
    u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert u.cache_creation_tokens == 0
    assert u.cache_read_tokens == 0


def test_response_propagates_cache_tokens(mock_model):
    p: MockProvider = mock_model.provider
    p.canned_usage = Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cache_creation_tokens=42,
        cache_read_tokens=11,
    )
    convo = mock_model.new_conversation()
    resp = convo.send("hi")
    assert resp.usage.cache_creation_tokens == 42
    assert resp.usage.cache_read_tokens == 11


def test_cached_system_block_requires_capability(mock_model):
    # MockProvider supports auto_cache_last_user, so cache=True is fine.
    mock_model.new_conversation(system_blocks=[SystemBlock(text="ok", cache=True)])

    # On a provider without the capability, cache=True should raise.
    from llmfacade.provider import Provider

    class NoCacheProvider(Provider):
        NAME = "nocache"
        SUPPORTS: frozenset[str] = frozenset({"max_tokens"})

        def _init_client(self):
            self._client = object()

    nocache = NoCacheProvider()
    nc_model = nocache.new_model("nc")
    with pytest.raises(UnsupportedFeature):
        nc_model.new_conversation(system_blocks=[SystemBlock(text="x", cache=True)])


def test_auto_cache_last_user_passes_to_provider(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation(auto_cache_last_user=True)
    convo.send("hi")
    last = p.calls[-1].req
    assert last.settings.get("auto_cache_last_user") is True


def test_auto_cache_tools_passes_to_provider(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation(auto_cache_tools=True)
    convo.send("hi")
    last = p.calls[-1].req
    assert last.settings.get("auto_cache_tools") is True


def test_auto_cache_tools_unsupported_on_non_anthropic_provider():
    """auto_cache_tools is Anthropic-only. Setting it on a provider that
    doesn't declare it must raise UnsupportedFeature at the layer it's set."""
    from llmfacade.provider import Provider

    class NoToolCacheProvider(Provider):
        NAME = "noopcache"
        SUPPORTS: frozenset[str] = frozenset({"max_tokens"})

        def _init_client(self):
            self._client = object()

    p = NoToolCacheProvider()
    with pytest.raises(UnsupportedFeature):
        p.new_model("nc", auto_cache_tools=True)


def test_anthropic_auto_cache_tools_marks_last_tool_only():
    """auto_cache_tools=True puts cache_control on the last tools entry only,
    using the resolved cache_ttl. Earlier entries stay untouched."""
    from llmfacade import tool
    from llmfacade.provider import CompletionRequest
    from llmfacade.providers.anthropic import AnthropicProvider
    from llmfacade.settings import EphemeralCacheTTL

    @tool
    def tool_a(x: str) -> str:
        """First tool."""
        return x

    @tool
    def tool_b(y: str) -> str:
        """Second tool."""
        return y

    p = object.__new__(AnthropicProvider)
    req = CompletionRequest(
        model="claude-sonnet-4-6",
        messages=[],
        system_blocks=[],
        tools=[tool_a, tool_b],
        stop=None,
        settings={
            "auto_cache_tools": True,
            "cache_ttl": EphemeralCacheTTL.ONE_HOUR,
            "max_tokens": 1024,
        },
    )
    api_kwargs = p._build_kwargs(req)
    api_tools = api_kwargs["tools"]
    assert "cache_control" not in api_tools[0]
    assert api_tools[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_auto_cache_tools_default_ttl_omits_ttl_field():
    """Without cache_ttl set, cache_control is bare ephemeral (SDK default 5m)."""
    from llmfacade import tool
    from llmfacade.provider import CompletionRequest
    from llmfacade.providers.anthropic import AnthropicProvider

    @tool
    def only_tool(x: str) -> str:
        """Sole tool."""
        return x

    p = object.__new__(AnthropicProvider)
    req = CompletionRequest(
        model="claude-sonnet-4-6",
        messages=[],
        system_blocks=[],
        tools=[only_tool],
        stop=None,
        settings={"auto_cache_tools": True, "max_tokens": 1024},
    )
    api_kwargs = p._build_kwargs(req)
    assert api_kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_auto_cache_tools_off_leaves_tools_unmarked():
    """No auto_cache_tools setting => no cache_control on any tool."""
    from llmfacade import tool
    from llmfacade.provider import CompletionRequest
    from llmfacade.providers.anthropic import AnthropicProvider

    @tool
    def only_tool(x: str) -> str:
        """Sole tool."""
        return x

    p = object.__new__(AnthropicProvider)
    req = CompletionRequest(
        model="claude-sonnet-4-6",
        messages=[],
        system_blocks=[],
        tools=[only_tool],
        stop=None,
        settings={"max_tokens": 1024},
    )
    api_kwargs = p._build_kwargs(req)
    assert all("cache_control" not in t for t in api_kwargs["tools"])
