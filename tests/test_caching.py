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
