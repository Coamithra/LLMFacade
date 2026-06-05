"""Anthropic streaming tool-call argument fragments.

Drives ``AnthropicProvider._chunk_to_events`` directly with fake SDK events
(content_block_start / input_json_delta / content_block_stop) so no live stream
is fired. Asserts each ``partial_json`` chunk is forwarded as a
``tool_args_delta`` and the consolidated ``tool_call_delta`` still follows.
"""

from __future__ import annotations

from types import SimpleNamespace

from llmfacade.providers.anthropic import AnthropicProvider


def _start(*, id: str, name: str):
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type="tool_use", id=id, name=name),
    )


def _json_delta(fragment: str):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="input_json_delta", partial_json=fragment),
    )


def _stop():
    return SimpleNamespace(type="content_block_stop")


def _drive(events):
    p = AnthropicProvider(api_key="test-key")
    state: dict[str, object] = {"current_tool": None, "current_thinking": None}
    out = []
    for ev in events:
        out.extend(p._chunk_to_events(ev, state))
    return out


def test_stream_emits_tool_args_fragments():
    events = _drive(
        [
            _start(id="tu_1", name="search"),
            _json_delta('{"q": '),
            _json_delta('"cats"'),
            _json_delta("}"),
            _stop(),
        ]
    )

    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    assert [f.fragment for f in frags] == ['{"q": ', '"cats"', "}"]
    assert all(f.index == 0 for f in frags)
    assert frags[0].id == "tu_1" and frags[0].name == "search"
    assert "".join(f.fragment for f in frags) == '{"q": "cats"}'

    calls = [e.tool_call_delta for e in events if e.tool_call_delta is not None]
    assert len(calls) == 1
    assert calls[0].input == {"q": "cats"}


def test_stream_two_tool_calls_get_distinct_indices():
    """A turn with two tool_use blocks numbers them 0 then 1, regardless of any
    interleaved non-tool content-block indices."""
    events = _drive(
        [
            _start(id="tu_1", name="a"),
            _json_delta('{"x": 1}'),
            _stop(),
            _start(id="tu_2", name="b"),
            _json_delta('{"y": 2}'),
            _stop(),
        ]
    )
    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    assert [(f.index, f.name, f.fragment) for f in frags] == [
        (0, "a", '{"x": 1}'),
        (1, "b", '{"y": 2}'),
    ]


def test_stream_malformed_tool_args_yields_empty_input():
    """Truncated JSON: fragments still arrive, terminal call has empty input."""
    events = _drive(
        [
            _start(id="tu_1", name="search"),
            _json_delta('{"q": "ca'),
            _stop(),
        ]
    )
    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    call = next(e.tool_call_delta for e in events if e.tool_call_delta is not None)
    assert "".join(f.fragment for f in frags) == '{"q": "ca'
    assert call.input == {}
