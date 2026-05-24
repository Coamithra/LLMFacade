"""Vision capability flag + cross-provider ImageBlock marshalling.

The four stock providers all declare a ``"vision"`` capability flag (a pure
capability flag like ``"tools"`` — in SUPPORTS, not RUNTIME_KNOBS). Sending an
``ImageBlock`` against a model narrowed to drop ``"vision"`` raises
``UnsupportedFeature`` at request time. These tests also lock the per-provider
wire shapes (Anthropic ``image``/``source``, OpenAI ``image_url``, Google
``inline_data``); llama-server's ``image_url`` shape stays covered in
``test_llamacpp.py``.
"""

from __future__ import annotations

import base64

import pytest

from llmfacade import UnsupportedFeature
from llmfacade.models import ImageBlock, Message, TextBlock
from llmfacade.providers.anthropic import AnthropicProvider
from llmfacade.providers.google import GoogleProvider
from llmfacade.providers.llamacpp import LlamaCppServerProvider
from llmfacade.providers.openai import OpenAIProvider

from .conftest import MockProvider

_RAW = b"\x89PNG\r\n\x1a\nfake-png-bytes"
_B64 = base64.b64encode(_RAW).decode("ascii")


def test_all_stock_providers_declare_vision():
    for cls in (AnthropicProvider, OpenAIProvider, GoogleProvider, LlamaCppServerProvider):
        assert "vision" in cls.SUPPORTS, cls.__name__


def test_vision_is_not_a_runtime_knob():
    """``"vision"`` is a pure capability flag — never a settable kwarg, like
    ``"tools"``. It must not have leaked into RUNTIME_KNOBS."""
    from llmfacade.settings import RUNTIME_KNOBS

    assert "vision" not in RUNTIME_KNOBS


# ---- per-provider wire-format marshalling ---------------------------------


def test_anthropic_marshals_image_block():
    p = object.__new__(AnthropicProvider)
    img = ImageBlock(data=_RAW, media_type="image/png")
    api = p._content_to_api([TextBlock("look"), img])
    assert isinstance(api, list)
    assert {"type": "text", "text": "look"} in api
    image_parts = [b for b in api if b.get("type") == "image"]
    assert len(image_parts) == 1
    assert image_parts[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": _B64,
    }


def test_openai_marshals_image_on_user():
    p = object.__new__(OpenAIProvider)
    img = ImageBlock(data=_RAW, media_type="image/png")
    msg = Message(role="user", content=[TextBlock("look"), img])
    api = p._message_to_api(msg)
    assert len(api) == 1
    parts = api[0]["content"]
    assert {"type": "text", "text": "look"} in parts
    image_parts = [pt for pt in parts if pt.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"] == {"url": f"data:image/png;base64,{_B64}"}


def test_openai_drops_image_on_assistant_with_warning():
    p = object.__new__(OpenAIProvider)
    img = ImageBlock(data=_RAW, media_type="image/png")
    msg = Message(role="assistant", content=[TextBlock("here"), img])
    with pytest.warns(UserWarning, match="dropping image"):
        api = p._message_to_api(msg)
    assert api[0]["content"] == "here"
    assert "image_url" not in str(api)


def test_google_marshals_image_block():
    p = object.__new__(GoogleProvider)
    img = ImageBlock(data=_RAW, media_type="image/jpeg")
    msg = Message(role="user", content=[TextBlock("look"), img])
    api = p._message_to_api(msg)
    assert len(api) == 1
    parts = api[0]["parts"]
    assert {"text": "look"} in parts
    image_parts = [pt for pt in parts if "inline_data" in pt]
    assert len(image_parts) == 1
    assert image_parts[0]["inline_data"] == {"mime_type": "image/jpeg", "data": _B64}


# ---- request-time capability gate -----------------------------------------


def _no_vision_model():
    provider = MockProvider()
    model = provider.new_model(
        "mock-model", capability_override=MockProvider.SUPPORTS - {"vision"}
    )
    return provider, model


def test_vision_gate_raises_on_non_vision_model():
    provider, model = _no_vision_model()
    convo = model.new_conversation()
    img = ImageBlock(data=_RAW, media_type="image/png")
    with pytest.raises(UnsupportedFeature):
        convo.send([TextBlock("look"), img])
    assert provider.calls == []  # gate fires before the provider is called


def test_vision_gate_allows_when_supported():
    provider = MockProvider()  # declares "vision"
    convo = provider.new_model("mock-model").new_conversation()
    img = ImageBlock(data=_RAW, media_type="image/png")
    convo.send([TextBlock("look"), img])
    assert len(provider.calls) == 1
    sent = provider.calls[-1].req.messages
    blocks = sent[-1].content
    assert any(isinstance(b, ImageBlock) for b in blocks)


def test_vision_gate_raises_on_stream():
    """The gate lives in the shared `_build_request`, so `stream` is covered by
    the same funnel as `send`. Locks that against a refactor that moves the
    cache lookup ahead of the gate on only one path."""
    provider, model = _no_vision_model()
    convo = model.new_conversation()
    img = ImageBlock(data=_RAW, media_type="image/png")
    with pytest.raises(UnsupportedFeature):
        list(convo.stream([TextBlock("look"), img]))
    assert provider.calls == []


def test_vision_gate_checks_history_images():
    """An image added to history (not the current prompt) is still gated."""
    provider, model = _no_vision_model()
    convo = model.new_conversation()
    convo.add_user_message(content=[ImageBlock(data=_RAW, media_type="image/png")])
    with pytest.raises(UnsupportedFeature):
        convo.send("describe it")
    assert provider.calls == []


def test_vision_gate_ignores_text_only():
    provider, model = _no_vision_model()
    convo = model.new_conversation()
    convo.send("just text")
    assert len(provider.calls) == 1


def test_capability_override_drops_vision():
    p = object.__new__(AnthropicProvider)
    override = AnthropicProvider.SUPPORTS - {"vision"}
    m = p.new_model("claude-text-only-2099", capability_override=override)
    assert m.is_available("vision") is False
    assert "vision" not in m.get_capabilities()
