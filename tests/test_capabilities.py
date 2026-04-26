"""Capability queries: is_available, get_capabilities, UnsupportedFeature, per-call validation."""

from __future__ import annotations

import pytest

from llmfacade import UnsupportedFeature

from .conftest import MockProvider


def test_provider_capabilities_query():
    p = MockProvider()
    assert p.is_available("context_size")
    assert "output_format" not in p.get_capabilities()


def test_unsupported_default_raises_at_construction():
    with pytest.raises(UnsupportedFeature) as exc:
        MockProvider(output_format="json")
    assert exc.value.provider == "mock"


def test_provider_default_propagates_to_request(mock_provider):
    p = MockProvider(temperature=0.42)
    convo = p.new_model("mock-model").new_conversation()
    convo.send("hi")
    last = p.calls[-1].req
    assert last.settings["temperature"] == 0.42
    assert last.settings_source["temperature"] == "provider"


def test_model_inherits_provider_capabilities(mock_provider, mock_model):
    assert mock_model.is_available("context_size")
    assert mock_model.get_capabilities() == mock_provider.get_capabilities()


def test_model_capability_override(mock_provider):
    override = mock_provider.SUPPORTS - {"thinking"}
    from llmfacade import Model

    m = Model(provider=mock_provider, model_id="mock-no-think", capability_override=override)
    assert not m.is_available("thinking")
    with pytest.raises(UnsupportedFeature):
        m.new_conversation(thinking=1024)


def test_convo_inherits_model_capabilities(mock_model):
    convo = mock_model.new_conversation()
    assert convo.is_available("auto_cache_last_user")


def test_per_call_unsupported_override_raises(mock_model):
    convo = mock_model.new_conversation()
    with pytest.raises(UnsupportedFeature):
        convo.send("hi", repeat_penalty=1.1)


def test_falsy_override_still_validates(mock_model):
    """MockProvider does not declare top_k; passing top_k=0 must hit the
    capability check and raise (not be skipped as if it were None)."""
    convo = mock_model.new_conversation()
    with pytest.raises(UnsupportedFeature):
        convo.send("hi", top_k=0)


def test_unknown_kwarg_raises_typeerror(mock_model):
    convo = mock_model.new_conversation()
    with pytest.raises(TypeError):
        convo.send("hi", made_up_setting=1)  # type: ignore[call-arg]


def test_per_call_overrides_propagate_to_request(mock_model):
    p = mock_model.provider
    convo = mock_model.new_conversation()
    convo.send("hi", temperature=0.3, max_tokens=999)
    last = p.calls[-1].req
    assert last.settings["temperature"] == 0.3
    assert last.settings["max_tokens"] == 999
    assert last.settings_source["temperature"] == "per_call"
    assert last.settings_source["max_tokens"] == "per_call"


def test_cascade_precedence(mock_provider):
    """provider < model < convo < per_call."""
    p = MockProvider(temperature=0.1, max_tokens=10)
    m = p.new_model("mock-model", temperature=0.2, max_tokens=20)
    c = m.new_conversation(temperature=0.3, max_tokens=30)
    c.send("hi", temperature=0.4)

    last = p.calls[-1].req
    assert last.settings["temperature"] == 0.4
    assert last.settings_source["temperature"] == "per_call"
    assert last.settings["max_tokens"] == 30
    assert last.settings_source["max_tokens"] == "convo"
