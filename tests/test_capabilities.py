"""Capability queries: is_available, get_capabilities, UnsupportedFeature, per-call validation."""

from __future__ import annotations

import pytest

from llmfacade import DrySampler, ThinkingMode, UnsupportedFeature

from .conftest import MockProvider


def test_provider_capabilities_query():
    p = MockProvider()
    assert p.is_available("temperature")
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
    assert mock_model.is_available("thinking")
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


def test_anthropic_model_enum_applies_capabilities():
    """Passing an `AnthropicModel` member to `new_model` resolves the canonical
    model id and applies the enum's capability set automatically."""
    from llmfacade.providers.anthropic import AnthropicModel, AnthropicProvider

    p = object.__new__(AnthropicProvider)
    m = p.new_model(AnthropicModel.SONNET_4_6)
    assert m.model_id == "claude-sonnet-4-6"
    assert m.get_capabilities() == set(AnthropicProvider.SUPPORTS)

    m_haiku = p.new_model(AnthropicModel.HAIKU_4_5)
    assert m_haiku.model_id == "claude-haiku-4-5-20251001"
    assert "thinking" in m_haiku.get_capabilities()
    # Bare Haiku underperforms with no reasoning, so the enum bakes in a budget
    # thinking default plus the max_tokens headroom the API requires above it.
    assert m_haiku.defaults["thinking"] == 4096
    assert m_haiku.defaults["max_tokens"] == 8192

    # An explicit kwarg still wins over the enum default. Note `thinking=None`
    # reads as "unset" and keeps the default — turn thinking off with DISABLED.
    from llmfacade.settings import ThinkingMode

    m_fast = p.new_model(AnthropicModel.HAIKU_4_5, thinking=ThinkingMode.DISABLED, max_tokens=512)
    assert m_fast.defaults["thinking"] is ThinkingMode.DISABLED
    assert m_fast.defaults["max_tokens"] == 512


def test_anthropic_opus_4_8_drops_sampling_and_thinking_budget():
    """Opus 4.8 (like 4.7) rejects temperature/top_p/top_k with a 400, so those
    knobs are dropped from the capability set and fail fast as
    `UnsupportedFeature`. It DOES support adaptive thinking, so the `thinking`
    knob is retained; only the budget-based form is rejected, expressed as the
    pure `"thinking_budget"` flag being absent (the budget *value* is gated at
    request time — see test_thinking.py). `effort` is retained.

    https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8
    """
    from llmfacade.providers.anthropic import AnthropicModel, AnthropicProvider

    p = object.__new__(AnthropicProvider)
    m = p.new_model(AnthropicModel.OPUS_4_8)
    assert m.model_id == "claude-opus-4-8"

    caps = m.get_capabilities()
    for dropped in ("temperature", "top_p", "top_k", "thinking_budget"):
        assert dropped not in caps
    # thinking (adaptive) is kept; effort is the depth control.
    for kept in ("thinking", "effort", "max_tokens", "tools", "tool_choice", "vision"):
        assert kept in caps
    assert "tool_result_images" in caps

    # Sampling knobs fail fast at model construction (the knob name is removed).
    with pytest.raises(UnsupportedFeature):
        p.new_model(AnthropicModel.OPUS_4_8, temperature=0.5)
    with pytest.raises(UnsupportedFeature):
        p.new_model(AnthropicModel.OPUS_4_8, top_p=0.9)
    with pytest.raises(UnsupportedFeature):
        p.new_model(AnthropicModel.OPUS_4_8, top_k=40)

    # The `thinking` knob is now accepted at construction (adaptive thinking is
    # supported); a ThinkingMode value or an int budget both type-check here.
    # The budget *value* is rejected later, at request time, not at construction.
    p.new_model(AnthropicModel.OPUS_4_8, thinking=ThinkingMode.ADAPTIVE)
    p.new_model(AnthropicModel.OPUS_4_8, thinking=2048)


def test_anthropic_opus_4_8_applies_xhigh_and_adaptive_defaults():
    """The OPUS_4_8 enum member carries model-scope defaults — effort=xhigh and
    adaptive thinking (its recommended settings). An explicit kwarg overrides
    them; Sonnet/Haiku carry no generation defaults."""
    from llmfacade import EffortLevel
    from llmfacade.providers.anthropic import AnthropicModel, AnthropicProvider

    p = AnthropicProvider(api_key="test-key", log_dir=False)
    m = p.new_model(AnthropicModel.OPUS_4_8)
    req = m.new_conversation()._build_request(stop=None, per_call={})
    assert req.settings["effort"] is EffortLevel.XHIGH
    assert req.settings["thinking"] is ThinkingMode.ADAPTIVE
    assert req.settings_source["effort"] == "model"
    assert req.settings_source["thinking"] == "model"

    # Explicit kwargs win over the enum defaults.
    m2 = p.new_model(
        AnthropicModel.OPUS_4_8, effort=EffortLevel.LOW, thinking=ThinkingMode.DISABLED
    )
    req2 = m2.new_conversation()._build_request(stop=None, per_call={})
    assert req2.settings["effort"] is EffortLevel.LOW
    assert req2.settings["thinking"] is ThinkingMode.DISABLED

    # Sonnet/Haiku carry no generation defaults.
    ms = p.new_model(AnthropicModel.SONNET_4_6)
    reqs = ms.new_conversation()._build_request(stop=None, per_call={})
    assert "effort" not in reqs.settings
    assert "thinking" not in reqs.settings


def test_anthropic_string_falls_back_to_full_supports():
    """A raw string (unknown to the enum) gets the provider's full SUPPORTS
    set; the caller is on their own to narrow via `capability_override=`."""
    from llmfacade.providers.anthropic import AnthropicProvider

    p = object.__new__(AnthropicProvider)
    m = p.new_model("claude-some-future-model-2099")
    assert m.model_id == "claude-some-future-model-2099"
    assert m.get_capabilities() == set(AnthropicProvider.SUPPORTS)


def test_anthropic_explicit_override_beats_enum():
    """If a caller passes both an enum member and an explicit
    `capability_override`, the explicit override wins."""
    from llmfacade.providers.anthropic import AnthropicModel, AnthropicProvider

    p = object.__new__(AnthropicProvider)
    custom = AnthropicProvider.SUPPORTS - {"top_k"}
    m = p.new_model(AnthropicModel.SONNET_4_6, capability_override=custom)
    # Enum default would have given full SUPPORTS; the explicit override
    # narrows further by dropping "top_k", and that narrowed set is what we
    # see (not the enum's full default).
    assert "top_k" not in m.get_capabilities()
    assert m.get_capabilities() == set(custom)


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


def test_dry_supported_by_llamacpp_only():
    """The `dry` knob is llamacpp-specific: llamacpp declares it, the stock
    hosted providers do not."""
    from llmfacade.providers.anthropic import AnthropicProvider
    from llmfacade.providers.llamacpp import LlamaCppServerProvider
    from llmfacade.providers.openai import OpenAIProvider

    assert "dry" in LlamaCppServerProvider.SUPPORTS
    assert "dry" not in AnthropicProvider.SUPPORTS
    assert "dry" not in OpenAIProvider.SUPPORTS


def test_dry_on_unsupporting_model_raises(mock_model):
    """MockProvider does not declare `dry`; setting it per-call hits the
    capability gate and raises (mirrors the repeat_penalty case)."""
    convo = mock_model.new_conversation()
    with pytest.raises(UnsupportedFeature):
        convo.send("hi", dry=DrySampler(multiplier=0.8))


def test_dry_cascades_and_per_call_wins():
    """`dry` cascades like any knob and a per-call value wholly replaces a
    higher-scope one. Driven against llamacpp (the only provider that supports
    it) in external mode so the gate doesn't reject the convo-scope default."""
    from llmfacade.models import Message
    from llmfacade.providers.llamacpp import LlamaCppServerProvider

    p = LlamaCppServerProvider(base_url="http://invalid.local:0/v1")
    convo = p.new_model("qwen2.5").new_conversation(dry=DrySampler(multiplier=0.5))
    # Reach into the merged request without firing a provider call.
    convo._history.append(Message(role="user", content="hi"))
    req = convo._build_request(stop=None, per_call={"dry": DrySampler(multiplier=0.9)})
    assert req.settings["dry"] == DrySampler(multiplier=0.9)
    assert req.settings_source["dry"] == "per_call"
