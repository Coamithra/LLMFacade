"""Cache tokens in Usage and AutoCacheLastUser routing."""

from __future__ import annotations

import pytest

from llmfacade import ConvoSettings, UnsupportedFeature, Usage

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
    convo = mock_model.NewConversation()
    convo.Start()
    resp = convo.Complete("hi")
    assert resp.usage.cache_creation_tokens == 42
    assert resp.usage.cache_read_tokens == 11


def test_cached_system_block_requires_capability(mock_model):
    convo = mock_model.NewConversation()
    convo.AddSystemBlock("ok", cache=True)  # MockProvider supports AutoCacheLastUser

    # Now test on a provider without the capability
    from llmfacade.provider import Provider
    from llmfacade.settings import AnySetting, Settings

    class NoCacheProvider(Provider):
        NAME = "nocache"
        SUPPORTS: frozenset[AnySetting] = frozenset({Settings.DefaultMaxTokens})

        def _init_client(self):
            self._client = object()

    nocache = NoCacheProvider()
    nc_model = nocache.NewModel("nc")
    nc_convo = nc_model.NewConversation()
    with pytest.raises(UnsupportedFeature):
        nc_convo.AddSystemBlock("x", cache=True)


def test_auto_cache_last_user_passes_to_provider(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.NewConversation()
    convo.settings.set(ConvoSettings.AutoCacheLastUser, True)
    convo.Start()
    convo.Complete("hi")
    last = p.calls[-1].kwargs
    convo_settings = last["convo_settings"]
    assert convo_settings.get(ConvoSettings.AutoCacheLastUser) is True
