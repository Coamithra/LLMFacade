"""Targeted tests for `providers/_openai_chat.py` — the Chat Completions
marshaling/parsing shared by the OpenAI and llamacpp providers.

The main safety net for the extraction is the existing per-provider suites
(test_openai.py, test_llamacpp.py, test_vision.py, test_tool_choice.py); these
tests pin the shared functions' own contracts, parametrized over both consumer
providers where cheap.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llmfacade.exceptions import ProviderError
from llmfacade.models import ImageBlock, Message, TextBlock, ThinkingBlock
from llmfacade.providers import _openai_chat as oc
from llmfacade.providers.llamacpp import LlamaCppServerProvider
from llmfacade.providers.openai import OpenAIProvider

_RAW = b"\x89PNG\r\n\x1a\nfake-bytes"


def _providers() -> list:
    return [
        pytest.param(lambda: OpenAIProvider(api_key="test-key"), id="openai"),
        pytest.param(
            lambda: LlamaCppServerProvider(base_url="http://localhost:8080/v1"), id="llamacpp"
        ),
    ]


# ---- tool_choice / tool schema mapping --------------------------------------


def test_tool_choice_to_api_mapping():
    assert oc.tool_choice_to_api("auto") == "auto"
    assert oc.tool_choice_to_api("required") == "required"
    assert oc.tool_choice_to_api("none") == "none"
    assert oc.tool_choice_to_api("forge_item") == {
        "type": "function",
        "function": {"name": "forge_item"},
    }


def test_tool_to_api_shape():
    t = SimpleNamespace(name="search", description="find things", schema={"type": "object"})
    assert oc.tool_to_api(t) == {
        "type": "function",
        "function": {
            "name": "search",
            "description": "find things",
            "parameters": {"type": "object"},
        },
    }


# ---- JSON-args recovery ------------------------------------------------------


def test_parse_tool_arguments_valid():
    assert oc.parse_tool_arguments('{"q": "cats"}') == ({"q": "cats"}, None)


def test_parse_tool_arguments_malformed_keeps_raw():
    args, unparsed = oc.parse_tool_arguments('{"q": "ca')
    assert args == {}
    assert unparsed == '{"q": "ca'


# ---- streaming slot accumulation + ordering contract -------------------------


def _tc_delta(index: int, *, id: str | None = None, name: str | None = None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_fragments_then_exactly_one_terminal_per_index():
    """Zero-or-more tool_args_delta fragments precede exactly one terminal
    tool_call_delta per index; concatenated fragments reconstruct the raw
    arguments string (the documented ordering contract)."""
    tool_buf: dict[int, dict[str, Any]] = {}
    events = []
    deltas = [
        SimpleNamespace(tool_calls=[_tc_delta(0, id="c0", name="a", arguments='{"x"')]),
        SimpleNamespace(tool_calls=[_tc_delta(1, id="c1", name="b", arguments='{"y"')]),
        SimpleNamespace(tool_calls=[_tc_delta(0, arguments=": 1}")]),
        SimpleNamespace(tool_calls=[_tc_delta(1, arguments=": 2}")]),
    ]
    for d in deltas:
        events.extend(oc.tool_fragment_events(d, tool_buf))
    terminal = list(oc.flush_tool_call_events(tool_buf))

    frags = [e.tool_args_delta for e in events]
    assert all(f is not None for f in frags)
    by_index: dict[int, list] = {}
    for f in frags:
        by_index.setdefault(f.index, []).append(f)
    assert "".join(f.fragment for f in by_index[0]) == '{"x": 1}'
    assert "".join(f.fragment for f in by_index[1]) == '{"y": 2}'
    assert {f.id for f in by_index[0]} == {"c0"} and {f.name for f in by_index[0]} == {"a"}

    calls = [e.tool_call_delta for e in terminal]
    assert len(calls) == 2
    assert {c.id: c.input for c in calls} == {"c0": {"x": 1}, "c1": {"y": 2}}
    assert tool_buf == {}  # flushed


def test_flush_recovers_malformed_args_as_raw_arguments():
    tool_buf: dict[int, dict[str, Any]] = {}
    frags = list(
        oc.tool_fragment_events(
            SimpleNamespace(tool_calls=[_tc_delta(0, id="c0", name="a", arguments='{"x": "tr')]),
            tool_buf,
        )
    )
    (call,) = (e.tool_call_delta for e in oc.flush_tool_call_events(tool_buf))
    assert call.input == {}
    assert call.raw_arguments == '{"x": "tr'
    assert "".join(f.tool_args_delta.fragment for f in frags) == call.raw_arguments


# ---- usage parsing ------------------------------------------------------------


def _usage(**extra):
    return SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14, **extra)


def test_usage_from_chat_cached_toggle():
    u = _usage(prompt_tokens_details=SimpleNamespace(cached_tokens=7))
    assert oc.usage_from_chat(u, include_cached=True).cache_read_tokens == 7
    # llamacpp never reports per-request cached tokens; the toggle keeps it 0
    # even if an OpenAI-compat build were to send the field.
    assert oc.usage_from_chat(u, include_cached=False).cache_read_tokens == 0


def test_usage_from_chat_reads_reasoning_tokens_for_both():
    u = _usage(completion_tokens_details=SimpleNamespace(reasoning_tokens=3))
    assert oc.usage_from_chat(u, include_cached=True).reasoning_tokens == 3
    assert oc.usage_from_chat(u, include_cached=False).reasoning_tokens == 3
    assert oc.usage_from_chat(_usage(), include_cached=False).reasoning_tokens == 0


# ---- parse_chat_response -------------------------------------------------------


def _raw_response(text="hi", **kwargs):
    msg = SimpleNamespace(content=text, tool_calls=None, **kwargs)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=_usage(),
        model="m1",
    )


def test_parse_chat_response_empty_choices_uses_server_label():
    raw = SimpleNamespace(choices=[], model="m1")
    with pytest.raises(ProviderError, match="llama-server returned a response with no choices"):
        oc.parse_chat_response(raw, server_label="llama-server", include_cached_tokens=False)
    with pytest.raises(ProviderError, match="OpenAI returned a response with no choices"):
        oc.parse_chat_response(raw, server_label="OpenAI", include_cached_tokens=True)


def test_parse_chat_response_reasoning_hook_leads_with_thinking_block():
    raw = _raw_response(text="answer", reasoning_content="let me think")
    resp = oc.parse_chat_response(
        raw,
        server_label="llama-server",
        include_cached_tokens=False,
        reasoning_text=lambda m: getattr(m, "reasoning_content", "") or "",
    )
    assert isinstance(resp.blocks[0], ThinkingBlock)
    assert resp.blocks[0].text == "let me think"
    assert resp.thinking == "let me think"
    assert resp.text == "answer"


def test_parse_chat_response_without_reasoning_hook():
    raw = _raw_response(text="answer", reasoning_content="ignored without hook")
    resp = oc.parse_chat_response(raw, server_label="OpenAI", include_cached_tokens=True)
    assert resp.thinking is None
    assert all(not isinstance(b, ThinkingBlock) for b in resp.blocks)


# ---- both providers delegate to the shared marshaling --------------------------


@pytest.mark.parametrize("make_provider", _providers())
def test_user_image_marshals_to_image_url_part(make_provider):
    p = make_provider()
    msg = Message(
        role="user", content=[TextBlock("look"), ImageBlock(data=_RAW, media_type="image/png")]
    )
    (out,) = p._message_to_api(msg)
    image_parts = [pt for pt in out["content"] if pt.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("make_provider", _providers())
def test_assistant_image_drop_warns_in_both_providers(make_provider):
    """The historical drift — OpenAI warned on the assistant-role image drop
    while llamacpp dropped silently — is unified: both warn via the shared
    message_to_api."""
    p = make_provider()
    msg = Message(
        role="assistant",
        content=[TextBlock("here"), ImageBlock(data=_RAW, media_type="image/png")],
    )
    with pytest.warns(UserWarning, match="dropping image"):
        (out,) = p._message_to_api(msg)
    assert out["content"] == "here"


@pytest.mark.parametrize("make_provider", _providers())
def test_empty_choices_raises_for_both_providers(make_provider):
    p = make_provider()
    with pytest.raises(ProviderError, match="no choices"):
        p._parse_response(SimpleNamespace(choices=None))
