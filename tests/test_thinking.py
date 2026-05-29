"""ThinkingBlock wire-format roundtrip and stream/non-stream parity."""

from __future__ import annotations

import pytest

from llmfacade import ThinkingMode, UnsupportedFeature
from llmfacade.models import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llmfacade.provider import CompletionRequest
from llmfacade.providers.anthropic import AnthropicProvider
from llmfacade.providers.google import GoogleProvider

from .conftest import MockProvider


def test_thinking_block_defaults():
    b = ThinkingBlock(text="reasoning")
    assert b.text == "reasoning"
    assert b.signature is None
    assert b.encrypted is False
    assert b.provider_data is None


def test_thinking_block_in_message_content():
    msg = Message(
        role="assistant",
        content=[
            ThinkingBlock(text="hmm", signature="sig1"),
            TextBlock("answer"),
        ],
    )
    assert isinstance(msg.content, list)
    assert isinstance(msg.content[0], ThinkingBlock)
    assert msg.content[0].signature == "sig1"


def test_send_persists_thinking_in_history(mock_model):
    p = mock_model.provider
    p.canned_thinking_blocks = [ThinkingBlock(text="reasoning bits", signature="abc")]
    p.canned_text = "final answer"
    convo = mock_model.new_conversation()
    convo.send("question")

    # Last entry is the assistant turn we just appended.
    assistant_msg = convo.history[-1]
    assert assistant_msg.role == "assistant"
    assert isinstance(assistant_msg.content, list)
    thinkers = [b for b in assistant_msg.content if isinstance(b, ThinkingBlock)]
    assert len(thinkers) == 1
    assert thinkers[0].text == "reasoning bits"
    assert thinkers[0].signature == "abc"


def test_stream_persists_thinking_in_history(mock_model):
    p = mock_model.provider
    p.canned_thinking_blocks = [ThinkingBlock(text="reasoning bits", signature="abc")]
    p.canned_text = "final answer"
    convo = mock_model.new_conversation()
    list(convo.stream("question"))

    assistant_msg = convo.history[-1]
    assert assistant_msg.role == "assistant"
    assert isinstance(assistant_msg.content, list)
    thinkers = [b for b in assistant_msg.content if isinstance(b, ThinkingBlock)]
    assert len(thinkers) == 1
    assert thinkers[0].text == "reasoning bits"
    assert thinkers[0].signature == "abc"


def test_send_and_stream_produce_identical_history(mock_model):
    """The asymmetry in review item #17: stream and send should yield the same
    Message.content for the same canned response."""
    p = mock_model.provider
    p.canned_thinking_blocks = [ThinkingBlock(text="reason", signature="s")]
    p.canned_text = "answer"

    send_convo = mock_model.new_conversation()
    send_convo.send("q")
    send_blocks = send_convo.history[-1].content

    stream_convo = mock_model.new_conversation()
    list(stream_convo.stream("q"))
    stream_blocks = stream_convo.history[-1].content

    # Both must contain a ThinkingBlock with the same text + signature, plus
    # the assistant text. Order must be thinking-then-text.
    assert isinstance(send_blocks, list)
    assert isinstance(stream_blocks, list)

    def thinking(blocks):
        return [b for b in blocks if isinstance(b, ThinkingBlock)]

    assert thinking(send_blocks) == thinking(stream_blocks)
    # Thinking must precede text in both paths.
    assert isinstance(send_blocks[0], ThinkingBlock)
    assert isinstance(stream_blocks[0], ThinkingBlock)


@pytest.mark.asyncio
async def test_astream_persists_thinking_in_history(mock_model):
    p = mock_model.provider
    p.canned_thinking_blocks = [ThinkingBlock(text="async reason", signature="z")]
    convo = mock_model.new_conversation()
    async for _ in convo.astream("q"):
        pass

    assistant_msg = convo.history[-1]
    thinkers = [b for b in assistant_msg.content if isinstance(b, ThinkingBlock)]
    assert len(thinkers) == 1
    assert thinkers[0].signature == "z"


# ---- Provider converter round-trips ---------------------------------------


def test_anthropic_content_to_api_emits_thinking_with_signature():
    p = AnthropicProvider.__new__(AnthropicProvider)
    out = p._content_to_api(
        [
            ThinkingBlock(text="reason text", signature="sigtoken"),
            TextBlock("answer"),
        ]
    )
    assert out == [
        {"type": "thinking", "thinking": "reason text", "signature": "sigtoken"},
        {"type": "text", "text": "answer"},
    ]


def test_anthropic_content_to_api_emits_thinking_without_signature():
    p = AnthropicProvider.__new__(AnthropicProvider)
    out = p._content_to_api([ThinkingBlock(text="raw thought")])
    assert out == [{"type": "thinking", "thinking": "raw thought"}]


def test_anthropic_content_to_api_emits_redacted_thinking():
    p = AnthropicProvider.__new__(AnthropicProvider)
    out = p._content_to_api(
        [
            ThinkingBlock(
                text="",
                encrypted=True,
                provider_data={"data": "encrypted-payload"},
            ),
        ]
    )
    assert out == [{"type": "redacted_thinking", "data": "encrypted-payload"}]


def test_anthropic_thinking_preserved_alongside_tool_use():
    """Multi-turn extended thinking + tool use is the actual motivating case
    for issue #17: the thinking block (with signature) must be sent back
    alongside the tool_use it preceded."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    out = p._content_to_api(
        [
            ThinkingBlock(text="should I call the tool?", signature="sig1"),
            ToolUseBlock(id="t1", name="lookup", input={"q": "x"}),
        ]
    )
    assert isinstance(out, list)
    assert out[0]["type"] == "thinking"
    assert out[0]["signature"] == "sig1"
    assert out[1]["type"] == "tool_use"


def test_google_message_to_api_emits_thinking_part():
    p = GoogleProvider.__new__(GoogleProvider)
    msg = Message(
        role="assistant",
        content=[
            ThinkingBlock(text="reason", signature="thoughtsig"),
            TextBlock("answer"),
        ],
    )
    out = p._message_to_api(msg)
    assert len(out) == 1
    parts = out[0]["parts"]
    assert parts[0] == {"text": "reason", "thought": True, "thought_signature": "thoughtsig"}
    assert parts[1] == {"text": "answer"}


def test_google_message_to_api_drops_encrypted_thinking():
    """Gemini has no redacted_thinking analog — encrypted blocks (e.g. from
    Anthropic) can't be reconstructed and should be dropped silently."""
    p = GoogleProvider.__new__(GoogleProvider)
    msg = Message(
        role="assistant",
        content=[
            ThinkingBlock(text="", encrypted=True, provider_data={"data": "x"}),
            TextBlock("answer"),
        ],
    )
    out = p._message_to_api(msg)
    parts = out[0]["parts"]
    assert parts == [{"text": "answer"}]


# ---- thinking knob: adaptive modes vs budget ------------------------------


def _thinking_req(value: object) -> CompletionRequest:
    return CompletionRequest(
        model="claude-opus-4-8",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"thinking": value, "max_tokens": 16},
        settings_source={"thinking": "convo", "max_tokens": "convo"},
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (ThinkingMode.ADAPTIVE, {"type": "adaptive"}),
        ("adaptive", {"type": "adaptive"}),
        (ThinkingMode.ADAPTIVE_SUMMARIZED, {"type": "adaptive", "display": "summarized"}),
        (ThinkingMode.DISABLED, {"type": "disabled"}),
        ("disabled", {"type": "disabled"}),
        (4096, {"type": "enabled", "budget_tokens": 4096}),
    ],
)
def test_anthropic_thinking_knob_maps_to_api_shape(value, expected):
    """ThinkingMode (and its string values) select adaptive/disabled modes; an
    int selects legacy budget-based extended thinking."""
    p = AnthropicProvider(api_key="test-key")
    api_kwargs = p._build_kwargs(_thinking_req(value))
    assert api_kwargs["thinking"] == expected


def test_anthropic_thinking_bool_is_rejected():
    """A stray `thinking=True` must fail loudly, not be silently read as
    budget_tokens=1 (bool is an int subclass)."""
    p = AnthropicProvider(api_key="test-key")
    with pytest.raises(TypeError):
        p._build_kwargs(_thinking_req(True))


def test_anthropic_no_thinking_means_no_thinking_kwarg():
    p = AnthropicProvider(api_key="test-key")
    req = CompletionRequest(
        model="claude-opus-4-8",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"max_tokens": 16},
        settings_source={"max_tokens": "convo"},
    )
    assert "thinking" not in p._build_kwargs(req)


# ---- budget-thinking request-time gate ------------------------------------


def test_budget_thinking_gated_when_unsupported(mock_provider):
    """A model that supports adaptive thinking but not the budget form rejects
    an int budget at request time, before the provider is called (mirrors the
    vision gate). Opus 4.7/4.8 are the real-world case."""
    m = mock_provider.new_model(
        "m", capability_override=MockProvider.SUPPORTS - {"thinking_budget"}
    )
    convo = m.new_conversation()
    with pytest.raises(UnsupportedFeature):
        convo.send("q", thinking=2048)
    assert mock_provider.calls == []


def test_adaptive_thinking_not_gated_without_budget(mock_provider):
    """An adaptive ThinkingMode is not the budget form, so it passes the gate
    even when "thinking_budget" is absent — this is how Opus 4.8 enables
    thinking through the facade."""
    m = mock_provider.new_model(
        "m", capability_override=MockProvider.SUPPORTS - {"thinking_budget"}
    )
    convo = m.new_conversation()
    convo.send("q", thinking=ThinkingMode.ADAPTIVE)
    assert len(mock_provider.calls) == 1


def test_budget_thinking_allowed_when_supported(mock_provider):
    """A model that declares "thinking_budget" accepts an int budget."""
    convo = mock_provider.new_model("m").new_conversation()
    convo.send("q", thinking=2048)
    assert len(mock_provider.calls) == 1
