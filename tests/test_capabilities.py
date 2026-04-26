"""Capability queries: isAvailable, getCapabilities, UnsupportedFeature, lock-in."""

from __future__ import annotations

import pytest

from llmfacade import (
    ConvoSettings,
    NotStartedError,
    ProviderSettings,
    Settings,
    SettingsLockedError,
    UnsupportedFeature,
)

from .conftest import MockProvider


def test_provider_capabilities_query():
    p = MockProvider()
    assert p.isAvailable(Settings.ContextSize)
    assert ProviderSettings.OrgID not in p.getCapabilities()


def test_set_supported_setting():
    p = MockProvider()
    p.settings.set(Settings.ContextSize, 4096)
    assert p.settings.get(Settings.ContextSize) == 4096


def test_set_unsupported_raises():
    p = MockProvider()
    with pytest.raises(UnsupportedFeature) as exc:
        p.settings.set(ProviderSettings.OrgID, "org-x")
    assert exc.value.provider == "mock"


def test_model_inherits_provider_capabilities(mock_provider, mock_model):
    assert mock_model.isAvailable(Settings.ContextSize)
    assert mock_model.getCapabilities() == mock_provider.getCapabilities()


def test_model_capability_override(mock_provider):
    override = mock_provider.SUPPORTS - {Settings.Thinking}
    from llmfacade import Model

    m = Model(provider=mock_provider, model_id="mock-no-think", capability_override=override)
    assert not m.isAvailable(Settings.Thinking)
    with pytest.raises(UnsupportedFeature):
        m.settings.set(Settings.Thinking, 1024)


def test_convo_inherits_model_capabilities(mock_model):
    convo = mock_model.NewConversation()
    assert convo.isAvailable(ConvoSettings.AutoCacheLastUser)


def test_lock_in_after_start(mock_model):
    convo = mock_model.NewConversation()
    convo.settings.set(ConvoSettings.AutoCacheLastUser, True)
    convo.Start()
    with pytest.raises(SettingsLockedError):
        convo.settings.set(ConvoSettings.AutoCacheLastUser, False)


def test_add_message_before_start_raises(mock_model):
    convo = mock_model.NewConversation()
    with pytest.raises(NotStartedError):
        convo.AddUserMessage("hi")


def test_set_logging_after_start_raises(mock_model, tmp_path):
    convo = mock_model.NewConversation()
    convo.Start()
    with pytest.raises(SettingsLockedError):
        convo.SetLogging(tmp_path / "log.jsonl")


def test_per_call_unsupported_override_raises(mock_model):
    convo = mock_model.NewConversation()
    convo.Start()
    with pytest.raises(UnsupportedFeature):
        convo.Complete("hi", repeat_penalty=1.1)


def test_falsy_override_still_validates(mock_model):
    # MockProvider does not declare Settings.TopK; passing top_k=0 must not be
    # treated the same as top_k=None (skipped) — it should hit the capability
    # check and raise.
    convo = mock_model.NewConversation()
    convo.Start()
    with pytest.raises(UnsupportedFeature):
        convo.Complete("hi", top_k=0)
