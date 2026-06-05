"""OpenAI provider `_build_kwargs` wire-shape: max_completion_tokens, the
`effort` -> `reasoning_effort` mapping, and structured-output translation.

Drives a `CompletionRequest` through `OpenAIProvider._build_kwargs` (no request
is fired) and asserts the SDK-shaped payload."""

from __future__ import annotations

import pytest

from llmfacade import EffortLevel, OutputFormat
from llmfacade.provider import CompletionRequest
from llmfacade.providers.openai import OpenAIProvider

_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


def _req(**settings: object) -> CompletionRequest:
    settings.setdefault("max_tokens", 100)
    return CompletionRequest(
        model="gpt-5.5",
        messages=[],
        system_blocks=[],
        tools=[],
        stop=None,
        settings=dict(settings),
        settings_source={k: "convo" for k in settings},
    )


def test_openai_emits_max_completion_tokens(openai_provider: OpenAIProvider):
    """GPT-5 series rejects legacy `max_tokens`; the facade's `max_tokens` knob
    must go out as `max_completion_tokens`."""
    kwargs = openai_provider._build_kwargs(_req(max_tokens=256))
    assert kwargs["max_completion_tokens"] == 256
    assert "max_tokens" not in kwargs


def test_openai_declares_effort_capability():
    assert "effort" in OpenAIProvider.SUPPORTS


def test_openai_effort_enum_maps_to_reasoning_effort(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req(effort=EffortLevel.XHIGH))
    assert kwargs["reasoning_effort"] == "xhigh"
    assert "effort" not in kwargs


def test_openai_effort_string_passes_through(openai_provider: OpenAIProvider):
    """OpenAI accepts values Anthropic doesn't (e.g. "minimal"); raw strings
    pass through verbatim."""
    kwargs = openai_provider._build_kwargs(_req(effort="minimal"))
    assert kwargs["reasoning_effort"] == "minimal"


def test_openai_no_effort_means_no_reasoning_effort(openai_provider: OpenAIProvider):
    assert "reasoning_effort" not in openai_provider._build_kwargs(_req())


def test_openai_json_mode(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req(output_format=OutputFormat.JSON))
    assert kwargs["response_format"] == {"type": "json_object"}


def test_openai_structured_output_bare_schema(openai_provider: OpenAIProvider):
    """A bare JSON-Schema dict becomes a strict json_schema (name defaulted)."""
    kwargs = openai_provider._build_kwargs(_req(output_format=_SCHEMA))
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": _SCHEMA, "strict": True},
    }


def test_openai_structured_output_full_config(openai_provider: OpenAIProvider):
    """A {name, schema, strict} dict is passed through with those values."""
    cfg = {"name": "Point", "schema": _SCHEMA, "strict": False}
    kwargs = openai_provider._build_kwargs(_req(output_format=cfg))
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Point", "schema": _SCHEMA, "strict": False},
    }


def test_openai_text_and_unset_omit_response_format(openai_provider: OpenAIProvider):
    assert "response_format" not in openai_provider._build_kwargs(
        _req(output_format=OutputFormat.TEXT)
    )
    assert "response_format" not in openai_provider._build_kwargs(_req())


# ---- reasoning-token extraction -------------------------------------------


def test_openai_reasoning_tokens_extracted():
    """``completion_tokens_details.reasoning_tokens`` is pulled out; absent
    details or absent field → 0."""
    from types import SimpleNamespace

    from llmfacade.providers.openai import _openai_reasoning_tokens

    with_details = SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens=33))
    assert _openai_reasoning_tokens(with_details) == 33
    assert _openai_reasoning_tokens(SimpleNamespace()) == 0
    assert (
        _openai_reasoning_tokens(SimpleNamespace(completion_tokens_details=SimpleNamespace())) == 0
    )


# ---- streaming tool-call argument fragments -------------------------------


def _tc_delta(index, *, id=None, name=None, arguments=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _chunk(*tool_calls, content=None, finish_reason=None, usage=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=list(tool_calls) or None),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def test_openai_stream_emits_tool_args_fragments(monkeypatch, openai_provider: OpenAIProvider):
    """Each ``fn.arguments`` chunk is forwarded as a ``tool_args_delta`` in order;
    concatenating the fragments reconstructs the full args, and the terminal
    ``tool_call_delta`` still carries the parsed input."""
    from types import SimpleNamespace

    chunks = [
        _chunk(_tc_delta(0, id="call_1", name="search", arguments='{"q": ')),
        _chunk(_tc_delta(0, arguments='"cats"')),
        _chunk(_tc_delta(0, arguments="}")),
        _chunk(
            content=None,
            finish_reason="tool_calls",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        ),
    ]
    monkeypatch.setattr(
        openai_provider._client.chat.completions, "create", lambda **_kw: iter(chunks)
    )

    events = list(openai_provider._stream_raw(_req()))

    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    assert [f.fragment for f in frags] == ['{"q": ', '"cats"', "}"]
    assert all(f.index == 0 for f in frags)
    assert frags[0].id == "call_1" and frags[0].name == "search"
    assert "".join(f.fragment for f in frags) == '{"q": "cats"}'

    calls = [e.tool_call_delta for e in events if e.tool_call_delta is not None]
    assert len(calls) == 1
    assert calls[0].input == {"q": "cats"}
    assert calls[0].raw_arguments is None


def test_openai_stream_malformed_tool_args_roundtrip(monkeypatch, openai_provider: OpenAIProvider):
    """Truncated streamed args: the joined fragments equal the terminal call's
    ``raw_arguments`` and ``input`` is empty (ties into raw_arguments)."""
    from types import SimpleNamespace

    chunks = [
        _chunk(_tc_delta(0, id="call_1", name="search", arguments='{"q": "ca')),
        _chunk(
            content=None,
            finish_reason="tool_calls",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        ),
    ]
    monkeypatch.setattr(
        openai_provider._client.chat.completions, "create", lambda **_kw: iter(chunks)
    )

    events = list(openai_provider._stream_raw(_req()))
    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    call = next(e.tool_call_delta for e in events if e.tool_call_delta is not None)

    assert call.input == {}
    assert call.raw_arguments == '{"q": "ca'
    assert "".join(f.fragment for f in frags) == call.raw_arguments
