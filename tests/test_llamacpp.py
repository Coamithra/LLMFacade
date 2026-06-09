"""LlamaCppServerProvider unit tests.

Drives ``CompletionRequest`` through ``_build_kwargs`` and asserts the
SDK-shaped payload, covers tool-call parsing, finish_reason translation, the
extra_body routing for llama.cpp-specific samplers (top_k/min_p/
repeat_penalty), the output_format JSON branch, and the introspection +
``count_tokens`` paths against a mocked httpx transport."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from llmfacade import Message, tool
from llmfacade.exceptions import ProviderError
from llmfacade.models import TextBlock, ThinkingBlock, ToolUseBlock
from llmfacade.provider import CompletionRequest
from llmfacade.providers.llamacpp import LlamaCppServerProvider
from llmfacade.settings import DrySampler, OutputFormat


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"Weather in {city}: sunny."


def _req(
    *,
    settings: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
    messages: list[Message] | None = None,
) -> CompletionRequest:
    s = {"max_tokens": 64}
    if settings:
        s.update(settings)
    return CompletionRequest(
        model="qwen2.5",
        messages=messages or [Message(role="user", content="hi")],
        system_blocks=[],
        tools=tools or [],
        stop=None,
        settings=s,
        settings_source={k: "convo" for k in s},
    )


@pytest.fixture
def provider() -> LlamaCppServerProvider:
    # Constructor builds the OpenAI client + httpx clients but doesn't fire
    # any requests; a fake base_url is fine for unit tests.
    return LlamaCppServerProvider(base_url="http://invalid.local:0/v1")


# ---- _build_kwargs --------------------------------------------------------


def test_build_kwargs_top_k_min_p_repeat_penalty_via_extra_body(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(
        _req(settings={"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1})
    )
    assert kwargs["extra_body"] == {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1}
    # And these keys do NOT leak into the top-level kwargs (the OpenAI SDK
    # would reject them as unknown).
    for k in ("top_k", "min_p", "repeat_penalty"):
        assert k not in kwargs


def test_build_kwargs_extra_body_omitted_when_no_llamacpp_knobs(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(_req(settings={"temperature": 0.5}))
    assert "extra_body" not in kwargs
    assert kwargs["temperature"] == 0.5


def test_build_kwargs_dry_minimal_uses_field_defaults(
    provider: LlamaCppServerProvider,
):
    # multiplier-only DrySampler: the numeric fields carry their defaults and
    # sequence_breakers is omitted (None) so the server keeps its own breakers.
    kwargs = provider._build_kwargs(_req(settings={"dry": DrySampler(multiplier=0.8)}))
    assert kwargs["extra_body"] == {
        "dry_multiplier": 0.8,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": -1,
    }
    assert "dry_sequence_breakers" not in kwargs["extra_body"]
    assert "dry" not in kwargs


def test_build_kwargs_dry_full_maps_all_fields(
    provider: LlamaCppServerProvider,
):
    dry = DrySampler(
        multiplier=0.9,
        base=2.0,
        allowed_length=3,
        penalty_last_n=256,
        sequence_breakers=("\n", "###"),
    )
    kwargs = provider._build_kwargs(_req(settings={"dry": dry}))
    assert kwargs["extra_body"] == {
        "dry_multiplier": 0.9,
        "dry_base": 2.0,
        "dry_allowed_length": 3,
        "dry_penalty_last_n": 256,
        "dry_sequence_breakers": ["\n", "###"],
    }


def test_build_kwargs_dry_coexists_with_other_samplers(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(
        _req(settings={"top_k": 40, "dry": DrySampler(multiplier=0.8)})
    )
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["extra_body"]["dry_multiplier"] == 0.8


def test_build_kwargs_dry_rejects_plain_dict(
    provider: LlamaCppServerProvider,
):
    # DrySampler is the only accepted form; a raw dict is rejected rather than
    # forwarded (an un-prefixed key would silently become a no-op wire param).
    with pytest.raises(TypeError):
        provider._build_kwargs(_req(settings={"dry": {"dry_multiplier": 0.8}}))


def test_drysampler_rejects_nonpositive_or_bool_multiplier():
    # multiplier is the enabling param; a non-positive value (or a stray bool)
    # would build an object that silently disables DRY.
    for bad in (0.0, -1.0, True):
        with pytest.raises(ValueError):
            DrySampler(multiplier=bad)


def test_build_kwargs_output_format_json_sets_response_format(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(_req(settings={"output_format": OutputFormat.JSON}))
    assert kwargs["response_format"] == {"type": "json_object"}


def test_build_kwargs_output_format_text_omits_response_format(
    provider: LlamaCppServerProvider,
):
    kwargs = provider._build_kwargs(_req(settings={"output_format": OutputFormat.TEXT}))
    assert "response_format" not in kwargs


def test_build_kwargs_tools_sets_tool_choice_default(provider: LlamaCppServerProvider):
    kwargs = provider._build_kwargs(_req(tools=[get_weather]))
    assert kwargs["tools"][0]["function"]["name"] == "get_weather"
    assert kwargs["tool_choice"] == "auto"


def test_build_kwargs_stop_passed_through(provider: LlamaCppServerProvider):
    req = CompletionRequest(
        model="qwen2.5",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=["END"],
        settings={"max_tokens": 16},
        settings_source={"max_tokens": "convo"},
    )
    kwargs = provider._build_kwargs(req)
    assert kwargs["stop"] == ["END"]


# ---- response parsing -----------------------------------------------------


class _FakeFn:
    def __init__(self, name: str | None, arguments: str | None):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, *, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = _FakeFn(name, arguments)


class _FakeReasoningDetails:
    def __init__(self, reasoning_tokens: int):
        self.reasoning_tokens = reasoning_tokens


class _FakeMsg:
    def __init__(
        self,
        *,
        content: str = "",
        tool_calls: list[Any] | None = None,
        reasoning_content: str | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        # Only set when present so getattr(..., None) mirrors the SDK, where a
        # non-reasoning response simply lacks the extra attribute.
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, *, message: _FakeMsg, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, *, prompt: int, completion: int, reasoning_tokens: int | None = None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        if reasoning_tokens is not None:
            self.completion_tokens_details = _FakeReasoningDetails(reasoning_tokens)


class _FakeResponse:
    def __init__(
        self,
        *,
        content: str = "ok",
        finish_reason: str = "stop",
        tool_calls: list[Any] | None = None,
        model: str = "qwen2.5",
        prompt: int = 5,
        completion: int = 2,
        reasoning_content: str | None = None,
        reasoning_tokens: int | None = None,
    ):
        self.choices = [
            _FakeChoice(
                message=_FakeMsg(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                ),
                finish_reason=finish_reason,
            )
        ]
        self.usage = _FakeUsage(
            prompt=prompt, completion=completion, reasoning_tokens=reasoning_tokens
        )
        self.model = model


def test_complete_finish_reason_length(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="hello", finish_reason="length"),
    )
    resp = provider._complete_raw(_req())
    assert resp.finish_reason == "length"
    assert resp.text == "hello"


def test_complete_finish_reason_stop(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="hello", finish_reason="stop"),
    )
    resp = provider._complete_raw(_req())
    assert resp.finish_reason == "stop"


def test_complete_tool_call_roundtrip(monkeypatch, provider: LlamaCppServerProvider):
    fake = _FakeResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            _FakeToolCall(
                id="call-1",
                name="get_weather",
                arguments='{"city": "Paris"}',
            )
        ],
    )
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: fake,
    )
    resp = provider._complete_raw(_req(tools=[get_weather]))
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "get_weather"
    assert call.input == {"city": "Paris"}
    assert call.id == "call-1"
    # A successful parse leaves raw_arguments unset.
    assert call.raw_arguments is None


def test_complete_truncated_tool_call_preserves_raw_arguments(
    monkeypatch, provider: LlamaCppServerProvider
):
    """A tool call truncated mid-JSON (e.g. hit the token limit) keeps its raw
    arguments string so the failed call is not silently lost from logs."""
    truncated = '{"city": "Paris", "details": {"foo": "bar' * 50  # never closes
    fake = _FakeResponse(
        content="",
        finish_reason="length",
        tool_calls=[
            _FakeToolCall(id="call-x", name="get_weather", arguments=truncated),
        ],
    )
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: fake,
    )
    resp = provider._complete_raw(_req(tools=[get_weather]))
    call = resp.tool_calls[0]
    assert call.input == {}  # unparseable -> empty parsed input
    assert call.raw_arguments == truncated  # but the raw string survives
    # The same on the matching ToolUseBlock in resp.blocks.
    tool_blocks = [b for b in resp.blocks if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].raw_arguments == truncated


def test_complete_empty_choices_raises_provider_error(
    monkeypatch, provider: LlamaCppServerProvider
):
    """A 200 with an empty choices list (a known OpenAI-compat quirk) must
    surface as ProviderError, not bare IndexError — _parse_response runs
    outside the SDK-error try/except in _complete_raw."""
    fake = _FakeResponse(model="qwen2.5")
    fake.choices = []
    monkeypatch.setattr(provider._client.chat.completions, "create", lambda **_kw: fake)
    with pytest.raises(ProviderError, match="no choices"):
        provider._complete_raw(_req())


def test_parse_response_none_choices_raises_provider_error(
    provider: LlamaCppServerProvider,
):
    from types import SimpleNamespace

    with pytest.raises(ProviderError, match="no choices"):
        provider._parse_response(SimpleNamespace(choices=None))


# ---- reasoning capture ----------------------------------------------------


def test_complete_captures_reasoning_content(monkeypatch, provider: LlamaCppServerProvider):
    """llama-server's ``reasoning_content`` becomes a leading ThinkingBlock and
    populates ``Response.thinking`` (was silently dropped before)."""
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="the answer", reasoning_content="let me think"),
    )
    resp = provider._complete_raw(_req())
    assert resp.thinking == "let me think"
    thinking = [b for b in resp.blocks if isinstance(b, ThinkingBlock)]
    assert len(thinking) == 1
    assert thinking[0].text == "let me think"
    assert thinking[0].encrypted is False
    # Reasoning leads the assistant turn, text follows.
    assert isinstance(resp.blocks[0], ThinkingBlock)
    assert isinstance(resp.blocks[1], TextBlock)


def test_complete_reasoning_tokens_from_details(monkeypatch, provider: LlamaCppServerProvider):
    """A build that reports ``completion_tokens_details.reasoning_tokens`` is
    surfaced verbatim on ``Usage``."""
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="answer", reasoning_content="x", reasoning_tokens=42),
    )
    resp = provider._complete_raw(_req())
    assert resp.usage is not None
    assert resp.usage.reasoning_tokens == 42


def test_complete_without_reasoning_is_unchanged(monkeypatch, provider: LlamaCppServerProvider):
    """No ``reasoning_content`` → no ThinkingBlock, thinking is None, count 0."""
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: _FakeResponse(content="answer"),
    )
    resp = provider._complete_raw(_req())
    assert resp.thinking is None
    assert all(not isinstance(b, ThinkingBlock) for b in resp.blocks)
    assert resp.usage is not None
    assert resp.usage.reasoning_tokens == 0


# ---- streaming ------------------------------------------------------------


class _FakeDelta:
    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        reasoning_content: str | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content


class _FakeStreamChoice:
    def __init__(self, *, delta: _FakeDelta, finish_reason: str | None = None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeStreamChunk:
    def __init__(
        self,
        *,
        choices: list[_FakeStreamChoice] | None = None,
        usage: _FakeUsage | None = None,
    ):
        self.choices = choices or []
        self.usage = usage


def test_stream_finish_reason_length(monkeypatch, provider: LlamaCppServerProvider):
    chunks = [
        _FakeStreamChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content="hi"))]),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="length")],
            usage=_FakeUsage(prompt=5, completion=2),
        ),
    ]
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        lambda **_kw: iter(chunks),
    )
    events = list(provider._stream_raw(_req()))
    final = [e for e in events if e.done]
    assert len(final) == 1
    assert final[0].finish_reason == "length"


def test_stream_captures_reasoning_content(monkeypatch, provider: LlamaCppServerProvider):
    """Streaming ``delta.reasoning_content`` is emitted as thinking deltas and
    consolidated into one ThinkingBlock flushed *before* the first text."""
    chunks = [
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(reasoning_content="think "))]
        ),
        _FakeStreamChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(reasoning_content="more"))]),
        _FakeStreamChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content="answer"))]),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="stop")],
            usage=_FakeUsage(prompt=5, completion=4, reasoning_tokens=3),
        ),
    ]
    monkeypatch.setattr(provider._client.chat.completions, "create", lambda **_kw: iter(chunks))
    events = list(provider._stream_raw(_req()))

    assert [e.thinking_delta for e in events if e.thinking_delta] == ["think ", "more"]
    tblocks = [e.thinking_block for e in events if e.thinking_block is not None]
    assert len(tblocks) == 1
    assert tblocks[0].text == "think more"
    # The consolidated thinking block precedes the text delta.
    block_idx = next(i for i, e in enumerate(events) if e.thinking_block is not None)
    text_idx = next(i for i, e in enumerate(events) if e.text_delta)
    assert block_idx < text_idx
    # reasoning_tokens rides on the terminal usage event.
    final = [e for e in events if e.done]
    assert final[0].usage is not None
    assert final[0].usage.reasoning_tokens == 3


def test_stream_reasoning_only_still_flushes_block(monkeypatch, provider: LlamaCppServerProvider):
    """A reasoning-only turn (no following content) still flushes the
    ThinkingBlock so it lands in history."""
    chunks = [
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(reasoning_content="just thinking"))]
        ),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="stop")],
            usage=_FakeUsage(prompt=5, completion=4),
        ),
    ]
    monkeypatch.setattr(provider._client.chat.completions, "create", lambda **_kw: iter(chunks))
    events = list(provider._stream_raw(_req()))
    tblocks = [e.thinking_block for e in events if e.thinking_block is not None]
    assert len(tblocks) == 1
    assert tblocks[0].text == "just thinking"


# ---- streaming tool-call argument fragments -------------------------------


class _FakeStreamToolCall:
    """A streamed tool-call delta: ``id``/``name`` arrive on the first chunk
    only, ``arguments`` arrives in fragments across chunks (the SDK shape)."""

    def __init__(
        self,
        *,
        index: int,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ):
        self.index = index
        self.id = id
        self.function = _FakeFn(name, arguments) if (name or arguments) else None


def test_stream_emits_tool_args_fragments(monkeypatch, provider: LlamaCppServerProvider):
    """Each ``fn.arguments`` chunk is forwarded as a ``tool_args_delta`` in order;
    joining the fragments reconstructs the full args string, and exactly one
    terminal ``tool_call_delta`` follows with the parsed input."""
    chunks = [
        _FakeStreamChunk(
            choices=[
                _FakeStreamChoice(
                    delta=_FakeDelta(
                        tool_calls=[
                            _FakeStreamToolCall(index=0, id="c1", name="lookup", arguments='{"id"')
                        ]
                    )
                )
            ]
        ),
        _FakeStreamChunk(
            choices=[
                _FakeStreamChoice(
                    delta=_FakeDelta(tool_calls=[_FakeStreamToolCall(index=0, arguments=": 7}")])
                )
            ]
        ),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="tool_calls")],
            usage=_FakeUsage(prompt=5, completion=3),
        ),
    ]
    monkeypatch.setattr(provider._client.chat.completions, "create", lambda **_kw: iter(chunks))
    events = list(provider._stream_raw(_req()))

    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    assert [f.fragment for f in frags] == ['{"id"', ": 7}"]
    assert all(f.index == 0 for f in frags)
    assert frags[0].id == "c1" and frags[0].name == "lookup"
    assert "".join(f.fragment for f in frags) == '{"id": 7}'

    calls = [e.tool_call_delta for e in events if e.tool_call_delta is not None]
    assert len(calls) == 1
    assert calls[0].input == {"id": 7}
    assert calls[0].raw_arguments is None


def test_stream_malformed_tool_args_roundtrip(monkeypatch, provider: LlamaCppServerProvider):
    """Truncated streamed args: joined fragments equal the terminal call's
    ``raw_arguments`` and ``input`` is empty."""
    chunks = [
        _FakeStreamChunk(
            choices=[
                _FakeStreamChoice(
                    delta=_FakeDelta(
                        tool_calls=[
                            _FakeStreamToolCall(
                                index=0, id="c1", name="lookup", arguments='{"id": '
                            )
                        ]
                    )
                )
            ]
        ),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="tool_calls")],
            usage=_FakeUsage(prompt=5, completion=3),
        ),
    ]
    monkeypatch.setattr(provider._client.chat.completions, "create", lambda **_kw: iter(chunks))
    events = list(provider._stream_raw(_req()))

    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    call = next(e.tool_call_delta for e in events if e.tool_call_delta is not None)
    assert call.input == {}
    assert call.raw_arguments == '{"id": '
    assert "".join(f.fragment for f in frags) == call.raw_arguments


def test_stream_two_tool_calls_keep_distinct_indices(
    monkeypatch, provider: LlamaCppServerProvider
):
    """Two concurrently-streamed tool calls keep their own index/id/name on each
    fragment even though id/name arrive only on the first chunk."""
    chunks = [
        _FakeStreamChunk(
            choices=[
                _FakeStreamChoice(
                    delta=_FakeDelta(
                        tool_calls=[
                            _FakeStreamToolCall(index=0, id="c0", name="a", arguments='{"x"'),
                            _FakeStreamToolCall(index=1, id="c1", name="b", arguments='{"y"'),
                        ]
                    )
                )
            ]
        ),
        _FakeStreamChunk(
            choices=[
                _FakeStreamChoice(
                    delta=_FakeDelta(
                        tool_calls=[
                            _FakeStreamToolCall(index=0, arguments=": 1}"),
                            _FakeStreamToolCall(index=1, arguments=": 2}"),
                        ]
                    )
                )
            ]
        ),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="tool_calls")],
            usage=_FakeUsage(prompt=5, completion=4),
        ),
    ]
    monkeypatch.setattr(provider._client.chat.completions, "create", lambda **_kw: iter(chunks))
    events = list(provider._stream_raw(_req()))

    frags = [e.tool_args_delta for e in events if e.tool_args_delta is not None]
    by_index: dict[int, list] = {}
    for f in frags:
        by_index.setdefault(f.index, []).append(f)

    assert "".join(f.fragment for f in by_index[0]) == '{"x": 1}'
    assert "".join(f.fragment for f in by_index[1]) == '{"y": 2}'
    assert {f.id for f in by_index[0]} == {"c0"} and {f.name for f in by_index[0]} == {"a"}
    assert {f.id for f in by_index[1]} == {"c1"} and {f.name for f in by_index[1]} == {"b"}


# ---- introspection (mocked httpx) ----------------------------------------


class _FakeHttpResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_health_returns_parsed_body(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._http, "get", lambda path: _FakeHttpResponse(json_body={"status": "ok"})
    )
    assert provider.health() == {"status": "ok"}


def test_health_raises_on_503(monkeypatch, provider: LlamaCppServerProvider):
    monkeypatch.setattr(
        provider._http,
        "get",
        lambda path: _FakeHttpResponse(status_code=503, text="loading"),
    )
    with pytest.raises(ProviderError):
        provider.health()


def test_slots_returns_list(monkeypatch, provider: LlamaCppServerProvider):
    fake = [{"id": 0, "is_processing": False, "n_ctx": 8192}]
    monkeypatch.setattr(provider._http, "get", lambda path: _FakeHttpResponse(json_body=fake))
    assert provider.slots() == fake


def test_save_slot_posts_with_action_query(monkeypatch, provider: LlamaCppServerProvider):
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen["path"] = path
        seen["params"] = params
        seen["json"] = json
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(provider._http, "post", fake_post)
    out = provider.save_slot(0, "warmup.bin")
    assert out == {"ok": True}
    assert seen["path"] == "/slots/0"
    assert seen["params"] == {"action": "save"}
    assert seen["json"] == {"filename": "warmup.bin"}


def test_erase_slot_no_body(monkeypatch, provider: LlamaCppServerProvider):
    seen: dict[str, Any] = {}

    def fake_post(path, *, params=None, json=None):
        seen["json"] = json
        return _FakeHttpResponse(json_body={"ok": True})

    monkeypatch.setattr(provider._http, "post", fake_post)
    provider.erase_slot(2)
    assert seen["json"] is None


# ---- count_tokens ---------------------------------------------------------


def test_count_tokens_hits_tokenize(monkeypatch, provider: LlamaCppServerProvider):
    seen: dict[str, Any] = {}

    def fake_post(path, *, json=None):
        seen["path"] = path
        seen["json"] = json
        return _FakeHttpResponse(json_body={"tokens": [1, 2, 3, 4, 5]})

    monkeypatch.setattr(provider._http, "post", fake_post)
    assert provider.count_tokens("hello world") == 5
    assert seen["path"] == "/tokenize"
    assert seen["json"] == {"content": "hello world"}


def test_count_tokens_falls_back_on_connection_error(
    monkeypatch, provider: LlamaCppServerProvider
):
    def boom(path, *, json=None):
        raise ConnectionError("server unreachable")

    monkeypatch.setattr(provider._http, "post", boom)
    # chars/4 of "hello world" (11 chars) → 2.
    assert provider.count_tokens("hello world") == 2


def test_tokenizer_name_is_llama_server(provider: LlamaCppServerProvider):
    assert provider.tokenizer_name() == "llama-server /tokenize"


# ---- managed mode (lazy supervisor, launch-knob validation) --------------


from pathlib import Path  # noqa: E402

from llmfacade.exceptions import UnsupportedFeature  # noqa: E402


def test_external_mode_launch_knob_in_init_rejected() -> None:
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        LlamaCppServerProvider(base_url="http://x:0/v1", context_size=8192)


def test_external_mode_new_model_with_launch_knobs_rejected(
    provider: LlamaCppServerProvider,
) -> None:
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        provider.new_model("qwen", context_size=8192)


def test_external_mode_new_model_without_id_raises(
    provider: LlamaCppServerProvider,
) -> None:
    with pytest.raises(ValueError, match="requires a positional model_id"):
        provider.new_model()


def test_managed_mode_constructor_no_supervisor_started(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    assert p._managed
    assert p._supervisor is not None
    assert not p._supervisor.is_started
    # No openai client built yet; only built once supervisor starts.
    assert p._client is None


def test_external_mode_client_keeps_default_retries(
    provider: LlamaCppServerProvider,
) -> None:
    # External mode talks to a real remote server, so SDK retries stay on.
    assert provider._client.max_retries > 0
    assert provider._aclient.max_retries > 0


def test_managed_mode_client_disables_retries(tmp_path: Path) -> None:
    # Managed mode owns a killable local process; retries against the dead port
    # defeat instant interrupt(), so the clients are built with max_retries=0.
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p._build_clients("http://invalid.local:0/v1")
    assert p._client.max_retries == 0
    assert p._aclient.max_retries == 0


def test_managed_mode_new_model_requires_gguf(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(ValueError, match="requires gguf="):
        p.new_model()


def test_managed_mode_new_model_missing_gguf_path(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(FileNotFoundError, match="gguf not found"):
        p.new_model(gguf=str(tmp_path / "nonexistent.gguf"))


def test_managed_mode_new_model_registers_entry(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), context_size=8192)
    entries = p._supervisor.entries  # type: ignore[union-attr]
    assert len(entries) == 1
    assert entries[0].gguf == str(gguf)
    assert entries[0].context_size == 8192
    # Model id is `<stem>-<hash8>` derived from launch config.
    assert model.model_id.startswith("qwen-")
    suffix = model.model_id.rsplit("-", 1)[1]
    assert len(suffix) == 8


def test_managed_mode_new_model_warns_on_no_mmap_plus_mlock(tmp_path: Path) -> None:
    """no_mmap + mlock together can CUDA-OOM at load on a heavy-spill MoE
    (pinned host allocation). new_model() warns rather than blocks — it's a valid
    combo on a model that mostly fits VRAM."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.warns(UserWarning, match="no_mmap=True and mlock=True"):
        p.new_model(gguf=str(gguf), no_mmap=True, mlock=True)


def test_managed_mode_new_model_warns_when_flags_split_across_scopes(tmp_path: Path) -> None:
    """The warning keys off the merged config, so no_mmap at provider scope +
    mlock at model scope still trips it."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", no_mmap=True)
    with pytest.warns(UserWarning, match="no_mmap=True and mlock=True"):
        p.new_model(gguf=str(gguf), mlock=True)


def test_managed_mode_new_model_no_warn_with_only_one_flag(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        p.new_model(gguf=str(gguf), name="only-no-mmap", no_mmap=True)
        p.new_model(gguf=str(gguf), name="only-mlock", mlock=True)


def test_managed_mode_explicit_name_used_as_model_id(tmp_path: Path) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), name="qwen-fast")
    assert model.model_id == "qwen-fast"


def test_managed_mode_provider_defaults_cascade_into_model(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(
        llmfacade_dir=tmp_path / "sess",
        n_gpu_layers=32,
        cache_type_k="q8_0",
    )
    p.new_model(gguf=str(gguf))
    entry = p._supervisor.entries[0]  # type: ignore[union-attr]
    assert entry.n_gpu_layers == 32
    assert entry.cache_type_k == "q8_0"


def test_managed_mode_model_overrides_provider_defaults(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", n_gpu_layers=32)
    p.new_model(gguf=str(gguf), n_gpu_layers=8)
    assert p._supervisor.entries[0].n_gpu_layers == 8  # type: ignore[union-attr]


def test_managed_mode_two_models_register_two_entries(tmp_path: Path) -> None:
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(a))
    p.new_model(gguf=str(b))
    assert len(p._supervisor.entries) == 2  # type: ignore[union-attr]


def test_managed_mode_running_unload_against_dead_supervisor_raises_useful_error(
    tmp_path: Path,
) -> None:
    """Without llama-swap on PATH, calling `running()` triggers ensure_started()
    which raises ProviderNotInstalledError. The user gets the clear install hint
    rather than a confusing AttributeError."""
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"x")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    # We can't reliably assume llama-swap isn't on PATH in CI, but if it is the
    # call would otherwise spawn it. Stub which() to None to force the
    # not-installed path deterministically.
    import llmfacade.providers._swap_lifecycle as ls

    original = ls.shutil.which
    ls.shutil.which = lambda b: None  # type: ignore[assignment]
    try:
        from llmfacade.exceptions import ProviderNotInstalledError

        with pytest.raises(ProviderNotInstalledError):
            p.running()
    finally:
        ls.shutil.which = original  # type: ignore[assignment]


def test_managed_mode_shutdown_no_op_when_never_started(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.shutdown()  # must not raise
    p.shutdown()


def test_external_mode_shutdown_no_op() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    p.shutdown()  # supervisor is None; should just no-op


def test_external_mode_interrupt_returns_false() -> None:
    # We don't own the process in external mode — interrupt is a no-op.
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    assert p.interrupt() is False
    assert p.interrupt() is False  # idempotent


def test_managed_mode_interrupt_no_op_when_never_started(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    assert p.interrupt() is False  # nothing running yet
    assert p.interrupt() is False


def test_managed_mode_interrupt_delegates_to_supervisor(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    calls: list[int] = []

    def fake_interrupt() -> bool:
        calls.append(1)
        return True

    p._supervisor.interrupt = fake_interrupt  # type: ignore[union-attr,method-assign]
    assert p.interrupt() is True
    assert len(calls) == 1


# ---- fit-params estimation + log_metadata --------------------------------


def test_log_metadata_external_mode_returns_none() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    assert p.log_metadata(model_id="anything") is None


def test_log_metadata_returns_none_when_no_estimate(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    assert p.log_metadata(model_id="never-registered") is None


def test_log_metadata_returns_fit_estimate_when_cached(tmp_path: Path) -> None:
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p._fit_estimates["m"] = {"context_size": 4096, "n_gpu_layers": 32}
    assert p.log_metadata(model_id="m") == {
        "fit_estimate": {"context_size": 4096, "n_gpu_layers": 32}
    }


def test_log_metadata_returns_copy_not_aliased(tmp_path: Path) -> None:
    """Caller mutating the returned dict mustn't alter our cached estimate."""
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p._fit_estimates["m"] = {"context_size": 4096}
    out = p.log_metadata(model_id="m")
    assert out is not None
    out["fit_estimate"]["context_size"] = 99999
    assert p._fit_estimates["m"] == {"context_size": 4096}


def test_new_model_silently_skips_when_fit_params_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _b: None)
    model = p.new_model(gguf=str(gguf), name="m")
    assert p._fit_estimates[model.model_id] is None
    # log_metadata no longer returns None for a managed model: it also surfaces
    # the auto-detected thinking_style. The fake gguf has no readable chat
    # template, so the style is UNKNOWN and there's no fit_estimate key.
    assert p.log_metadata(model_id=model.model_id) == {"thinking_style": "unknown"}


def test_new_model_skips_estimate_when_fit_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    called: dict[str, Any] = {}

    def boom(*_a: Any, **_kw: Any) -> Any:  # pragma: no cover - must not run
        called["ran"] = True
        raise AssertionError("subprocess.run should not run when fit=False")

    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", boom)
    model = p.new_model(gguf=str(gguf), name="m", fit=False)
    assert "ran" not in called
    assert p._fit_estimates[model.model_id] is None


def test_new_model_runs_fit_params_and_stores_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeCompleted:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "-c 8192 -ngl 32 -ts 1\n"
            self.stderr = (
                "fit_params: projected memory use [MiB]:\n"
                "fit_params:   - GPU0: 32 layers,  8192 MiB used,  1024 MiB free\n"
            )

    def fake_run(argv: list[str], **kw: Any) -> _FakeCompleted:
        seen_argv["argv"] = argv
        seen_argv["kw"] = kw
        return _FakeCompleted()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    model = p.new_model(gguf=str(gguf), name="m", parallel=2, fit_target=(1024,))
    est = p._fit_estimates[model.model_id]
    assert est == {
        "context_size": 8192,
        "n_gpu_layers": 32,
        "est_vram_mib": 8192,
        "parallel": 2,
    }
    # Sanity-check the spawned argv shape — assert positionally so swapping
    # two flag/value pairs would still be caught.
    argv = seen_argv["argv"]
    assert argv[0].endswith("llama-fit-params")
    assert argv[argv.index("--model") + 1] == str(gguf)
    assert argv[argv.index("--parallel") + 1] == "2"
    assert argv[argv.index("--fit-target") + 1] == "1024"


def test_new_model_translates_sentinel_estimate_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the model fits at defaults, fit-params prints the unset
    sentinels (-c 0 -ngl -1) verbatim. The provider translates those to
    human-readable labels so the JSONL/HTML log doesn't surface "0" / "-1"
    to a user puzzling over what they mean."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    class _FakeOk:
        returncode = 0
        stdout = "-c 0 -ngl -1 -ts 1\n"
        stderr = ""

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", lambda *a, **k: _FakeOk())

    model = p.new_model(gguf=str(gguf), name="m")
    est = p._fit_estimates[model.model_id]
    assert est is not None
    assert est["context_size"] == "model default"
    assert est["n_gpu_layers"] == "all"


def test_new_model_keeps_real_estimate_values_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sentinel translation must not touch real numbers."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 24\n"
        stderr = ""

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", lambda *a, **k: _FakeOk())

    model = p.new_model(gguf=str(gguf), name="m")
    est = p._fit_estimates[model.model_id]
    assert est is not None
    assert est["context_size"] == 4096
    assert est["n_gpu_layers"] == 24


def test_new_model_handles_fit_params_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    class _FakeFailed:
        returncode = 2
        stdout = ""
        stderr = "boom"

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", lambda *a, **k: _FakeFailed())
    model = p.new_model(gguf=str(gguf), name="m")
    assert p._fit_estimates[model.model_id] is None


def test_new_model_handles_fit_params_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)

    def fake_run(*_a: Any, **_kw: Any) -> Any:
        raise _subprocess.TimeoutExpired(cmd="llama-fit-params", timeout=60.0)

    monkeypatch.setattr(_subprocess, "run", fake_run)
    model = p.new_model(gguf=str(gguf), name="m")
    assert p._fit_estimates[model.model_id] is None


def test_new_model_does_not_forward_extra_args_to_fit_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extra_args` are llama-server-specific flags. Forwarding them to
    `llama-fit-params` would make it exit non-zero and silently lose every
    estimate for users with non-empty extra_args."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 32"
        stderr = ""

    def fake_run(argv: list[str], **_kw: Any) -> _FakeOk:
        seen_argv["argv"] = argv
        return _FakeOk()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    p.new_model(gguf=str(gguf), name="m", extra_args=["--mlock", "--flash-attn"])
    argv = seen_argv["argv"]
    assert "--mlock" not in argv
    assert "--flash-attn" not in argv


def test_external_mode_rejects_fit_target_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        LlamaCppServerProvider(base_url="http://x:0/v1", fit_target=(512,))


def test_external_mode_rejects_fit_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        p.new_model("qwen", fit=False)


def test_external_mode_rejects_flash_attn_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        LlamaCppServerProvider(base_url="http://x:0/v1", flash_attn="on")


def test_external_mode_rejects_flash_attn_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="launch knobs"):
        p.new_model("qwen", flash_attn="on")


def test_managed_mode_flash_attn_provider_default_cascades(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", flash_attn="on")
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].flash_attn == "on"  # type: ignore[union-attr]


def test_managed_mode_flash_attn_model_overrides_provider(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", flash_attn="on")
    p.new_model(gguf=str(gguf), flash_attn="off")
    assert p._supervisor.entries[0].flash_attn == "off"  # type: ignore[union-attr]


def test_managed_mode_flash_attn_default_is_none(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].flash_attn is None  # type: ignore[union-attr]


def test_flash_attn_invalid_value_in_init_raises() -> None:
    with pytest.raises(ValueError, match="flash_attn must be one of"):
        LlamaCppServerProvider(flash_attn="yes")


def test_flash_attn_invalid_value_in_new_model_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(ValueError, match="flash_attn must be one of"):
        p.new_model(gguf=str(gguf), flash_attn="enabled")


def test_external_mode_rejects_n_cpu_moe_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="n_cpu_moe"):
        LlamaCppServerProvider(base_url="http://x:0/v1", n_cpu_moe=60)


def test_external_mode_rejects_n_cpu_moe_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="n_cpu_moe"):
        p.new_model("qwen", n_cpu_moe=60)


def test_managed_mode_n_cpu_moe_provider_default_cascades(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", n_cpu_moe=60)
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].n_cpu_moe == 60  # type: ignore[union-attr]


def test_managed_mode_n_cpu_moe_model_overrides_provider(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", n_cpu_moe=60)
    p.new_model(gguf=str(gguf), n_cpu_moe=40)
    assert p._supervisor.entries[0].n_cpu_moe == 40  # type: ignore[union-attr]


def test_managed_mode_n_cpu_moe_default_is_none(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].n_cpu_moe is None  # type: ignore[union-attr]


def test_new_model_forwards_n_cpu_moe_to_fit_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`n_cpu_moe` IS forwarded to llama-fit-params (unlike extra_args). It's a
    common-arg flag and changes the GPU memory footprint, so forwarding it
    produces a more accurate VRAM estimate."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 32"
        stderr = ""

    def fake_run(argv: list[str], **_kw: Any) -> _FakeOk:
        seen_argv["argv"] = argv
        return _FakeOk()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    p.new_model(gguf=str(gguf), name="m", n_cpu_moe=60)
    argv = seen_argv["argv"]
    assert "--n-cpu-moe" in argv
    idx = argv.index("--n-cpu-moe")
    assert argv[idx + 1] == "60"


def test_new_model_forwards_flash_attn_to_fit_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flash_attn IS forwarded to llama-fit-params (unlike extra_args). It's part
    of the same llama.cpp common arg parsing and affects KV cache layout, so
    forwarding it produces a more accurate VRAM estimate."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 32"
        stderr = ""

    def fake_run(argv: list[str], **_kw: Any) -> _FakeOk:
        seen_argv["argv"] = argv
        return _FakeOk()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    p.new_model(gguf=str(gguf), name="m", flash_attn="on")
    argv = seen_argv["argv"]
    assert "--flash-attn" in argv
    idx = argv.index("--flash-attn")
    assert argv[idx + 1] == "on"


# ---- managed-mode introspection routing ----------------------------------
#
# These verify that managed-mode wrappers prepend ``/upstream/<model>/...``
# and that the model resolver picks the right entry. We populate
# ``_supervisor._entries`` directly (bypassing ``new_model``'s file checks)
# and inject a fake httpx client so nothing actually spawns.


class _CapturingHttp:
    """Records the most recent .get/.post call without doing any I/O."""

    def __init__(self, response: _FakeHttpResponse | None = None) -> None:
        self.response = response or _FakeHttpResponse(json_body={"ok": True})
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str) -> _FakeHttpResponse:
        self.calls.append({"method": "GET", "path": path})
        return self.response

    def post(self, path: str, *, params: Any = None, json: Any = None) -> _FakeHttpResponse:
        self.calls.append({"method": "POST", "path": path, "params": params, "json": json})
        return self.response


def _managed_provider_with_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str
) -> LlamaCppServerProvider:
    """Build a managed-mode provider with the named launch entries already
    registered, with `_ensure_supervised` neutralised. Caller still needs to
    set `_http`/`_ahttp` to a fake."""
    from llmfacade.providers._launch import _LaunchEntry

    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    for n in names:
        gguf = tmp_path / f"{n}.gguf"
        gguf.write_bytes(b"x")
        # Go through the public register() API rather than poking _entries
        # directly so tests catch any future validation/side-effect added to
        # registration.
        p._supervisor.register(_LaunchEntry(model_id=n, gguf=str(gguf)))  # type: ignore[union-attr]
    # Neutralise the lazy-spawn so introspection methods don't try to start
    # llama-swap. The wrappers all call self._ensure_supervised() first.
    monkeypatch.setattr(p, "_ensure_supervised", lambda: None)
    return p


def test_managed_resolve_zero_entries_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no models are registered"):
        p._resolve_introspection_target(None)


def test_managed_resolve_single_entry_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "qwen-fast")
    assert p._resolve_introspection_target(None) == "/upstream/qwen-fast"


def test_managed_resolve_multi_entry_requires_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "fast", "quality")
    with pytest.raises(ValueError, match="requires model="):
        p._resolve_introspection_target(None)


def test_managed_resolve_explicit_model_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "fast", "quality")
    # Even with multiple entries, an explicit model just gets used as-is.
    assert p._resolve_introspection_target("explicit") == "/upstream/explicit"


def test_managed_resolve_url_quotes_slash_in_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    # Don't register the entry — pass model= explicitly. urlquote should
    # escape the slash so llama-swap parses the slash as part of the model id
    # rather than as a path separator.
    assert p._resolve_introspection_target("Qwen/Qwen2.5-3B") == "/upstream/Qwen%2FQwen2.5-3B"


def test_external_resolve_returns_empty_prefix() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    # External mode silently ignores the model arg.
    assert p._resolve_introspection_target(None) == ""
    assert p._resolve_introspection_target("anything") == ""


def test_managed_slots_routes_through_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "qwen-fast")
    fake = [{"id": 0, "is_processing": False}]
    p._http = _CapturingHttp(_FakeHttpResponse(json_body=fake))
    out = p.slots()
    assert out == fake
    assert p._http.calls == [{"method": "GET", "path": "/upstream/qwen-fast/slots"}]


def test_managed_slots_explicit_model_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "a", "b")
    p._http = _CapturingHttp(_FakeHttpResponse(json_body=[]))
    p.slots(model="b")
    assert p._http.calls == [{"method": "GET", "path": "/upstream/b/slots"}]


def test_managed_slots_zero_entries_no_model_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp()
    with pytest.raises(ValueError):
        p.slots()


def test_managed_health_no_model_normalises_swap_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "anything")
    # llama-swap's own /health returns the literal text "OK".
    p._http = _CapturingHttp(_FakeHttpResponse(text="OK\n"))
    assert p.health() == {"status": "ok"}
    # No /upstream/ prefix when probing swap-root health.
    assert p._http.calls == [{"method": "GET", "path": "/health"}]


def test_managed_health_no_model_handles_bytes_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive: if a fake or weird httpx state yields bytes for `.text`,
    `_normalise_swap_health` decodes before comparing."""
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "anything")
    p._http = _CapturingHttp(_FakeHttpResponse(text=b"OK"))  # type: ignore[arg-type]
    assert p.health() == {"status": "ok"}


def test_managed_health_with_model_routes_through_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"status": "ok"}))
    out = p.health(model="qwen-fast")
    assert out == {"status": "ok"}
    assert p._http.calls == [{"method": "GET", "path": "/upstream/qwen-fast/health"}]


def test_managed_save_slot_routes_through_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "m")
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    p.save_slot(0, "warmup.bin")
    assert p._http.calls == [
        {
            "method": "POST",
            "path": "/upstream/m/slots/0",
            "params": {"action": "save"},
            "json": {"filename": "warmup.bin"},
        }
    ]


def test_managed_erase_slot_no_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "m")
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    p.erase_slot(0)
    assert p._http.calls[0]["json"] is None
    assert p._http.calls[0]["path"] == "/upstream/m/slots/0"
    assert p._http.calls[0]["params"] == {"action": "erase"}


def test_managed_count_tokens_uses_upstream_tokenize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "m")
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"tokens": [1, 2, 3]}))
    assert p.count_tokens("hello world") == 3
    assert p._http.calls[0]["path"] == "/upstream/m/tokenize"
    assert p._http.calls[0]["json"] == {"content": "hello world"}


def test_managed_count_tokens_no_entries_falls_back_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp()
    # Resolver raises because there are no entries; the wrapper swallows it
    # and returns chars/4 (11 // 4 == 2). No HTTP call should be made.
    assert p.count_tokens("hello world") == 2
    assert p._http.calls == []


def test_managed_count_tokens_with_explicit_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "a", "b")
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"tokens": [9, 9]}))
    assert p.count_tokens("xy", model_id="b") == 2
    assert p._http.calls[0]["path"] == "/upstream/b/tokenize"


# ---- Model-bound mirrors --------------------------------------------------


def test_model_slots_passes_model_id_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "qwen-fast")
    p._http = _CapturingHttp(_FakeHttpResponse(json_body=[{"id": 0}]))
    # Use the entries we registered. Build a Model object the way new_model
    # would, but skip the launch validation by going through the constructor
    # (the entry already exists).
    from llmfacade.model import Model

    m = Model(provider=p, model_id="qwen-fast")
    assert m.slots() == [{"id": 0}]
    assert p._http.calls[0]["path"] == "/upstream/qwen-fast/slots"


def test_model_health_passes_model_id_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"status": "ok"}))
    from llmfacade.model import Model

    m = Model(provider=p, model_id="anything")
    m.health()
    # Should hit /upstream/, not the swap-root /health.
    assert p._http.calls[0]["path"] == "/upstream/anything/health"


def test_model_save_slot_forwards_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    from llmfacade.model import Model

    m = Model(provider=p, model_id="qwen-fast")
    m.save_slot(2, "snap.bin")
    call = p._http.calls[0]
    assert call["path"] == "/upstream/qwen-fast/slots/2"
    assert call["json"] == {"filename": "snap.bin"}
    assert call["params"] == {"action": "save"}


def test_model_count_tokens_already_passes_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing precedent (Model.count_tokens binds self._model_id) — verify
    it still routes correctly through the new managed-mode prefix."""
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"tokens": [1, 2, 3, 4]}))
    from llmfacade.model import Model

    m = Model(provider=p, model_id="some-model")
    assert m.count_tokens("hello") == 4
    assert p._http.calls[0]["path"] == "/upstream/some-model/tokenize"


def test_model_health_against_non_llamacpp_provider_raises_unsupported() -> None:
    """Calling Model.health() when the provider doesn't expose health()
    should raise UnsupportedFeature (codebase convention for capability
    gaps), not AttributeError."""
    from types import SimpleNamespace

    from llmfacade.exceptions import UnsupportedFeature
    from llmfacade.model import Model

    # Model.__init__ reads .NAME and .SUPPORTS off the provider; need both.
    fake_provider = SimpleNamespace(NAME="madeup", SUPPORTS=frozenset())
    m = Model(provider=fake_provider, model_id="x")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedFeature, match="health"):
        m.health()


# ---- async coverage of the new introspection paths ----------------------
#
# The sync wrappers above all have async siblings (`ahealth`, `aslots`,
# `asave_slot`, `arestore_slot`, `aerase_slot`, plus
# `_swap_root_health_async` and the Model mirrors `aslots`/`ahealth`/etc.).
# These exercise the async branch end-to-end, including the bytes-safe
# normalisation path and the `/upstream/` prefix for both the provider
# methods and the Model mirrors.


class _AsyncCapturingHttp:
    """Async sibling of `_CapturingHttp`. Mirrors the same attributes so the
    same assertion helpers work."""

    def __init__(self, response: _FakeHttpResponse | None = None) -> None:
        self.response = response or _FakeHttpResponse(json_body={"ok": True})
        self.calls: list[dict[str, Any]] = []

    async def get(self, path: str) -> _FakeHttpResponse:
        self.calls.append({"method": "GET", "path": path})
        return self.response

    async def post(self, path: str, *, params: Any = None, json: Any = None) -> _FakeHttpResponse:
        self.calls.append({"method": "POST", "path": path, "params": params, "json": json})
        return self.response


@pytest.mark.asyncio
async def test_managed_aslots_routes_through_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "qwen-fast")
    fake = [{"id": 0}]
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body=fake))
    out = await p.aslots()
    assert out == fake
    assert p._ahttp.calls == [{"method": "GET", "path": "/upstream/qwen-fast/slots"}]


@pytest.mark.asyncio
async def test_managed_ahealth_no_model_normalises_swap_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "anything")
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(text="OK"))
    assert (await p.ahealth()) == {"status": "ok"}
    assert p._ahttp.calls == [{"method": "GET", "path": "/health"}]


@pytest.mark.asyncio
async def test_managed_ahealth_with_model_routes_through_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body={"status": "ok"}))
    assert (await p.ahealth(model="x")) == {"status": "ok"}
    assert p._ahttp.calls == [{"method": "GET", "path": "/upstream/x/health"}]


@pytest.mark.asyncio
async def test_managed_aerase_slot_no_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "m")
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    await p.aerase_slot(0)
    call = p._ahttp.calls[0]
    assert call["json"] is None
    assert call["path"] == "/upstream/m/slots/0"
    assert call["params"] == {"action": "erase"}


@pytest.mark.asyncio
async def test_managed_asave_slot_routes_through_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "m")
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    await p.asave_slot(0, "warmup.bin")
    assert p._ahttp.calls == [
        {
            "method": "POST",
            "path": "/upstream/m/slots/0",
            "params": {"action": "save"},
            "json": {"filename": "warmup.bin"},
        }
    ]


def test_unload_url_quotes_slashed_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slashed author/model-style id must be %2F-escaped in the unload URL,
    matching _resolve_introspection_target — unquoted it parses as extra path
    segments, 404s, and masquerades as 'llama-swap not detected'."""
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    p.unload("Qwen/Qwen2.5-3B")
    assert p._http.calls == [
        {
            "method": "POST",
            "path": "/api/models/unload/Qwen%2FQwen2.5-3B",
            "params": None,
            "json": None,
        }
    ]


@pytest.mark.asyncio
async def test_aunload_url_quotes_slashed_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    await p.aunload("Qwen/Qwen2.5-3B")
    assert p._ahttp.calls[0]["path"] == "/api/models/unload/Qwen%2FQwen2.5-3B"


def test_unload_plain_model_id_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch)
    p._http = _CapturingHttp(_FakeHttpResponse(json_body={"ok": True}))
    p.unload("qwen-fast")
    assert p._http.calls[0]["path"] == "/api/models/unload/qwen-fast"


@pytest.mark.asyncio
async def test_model_aslots_passes_model_id_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _managed_provider_with_entries(tmp_path, monkeypatch, "qwen-fast")
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body=[{"id": 0}]))
    from llmfacade.model import Model

    m = Model(provider=p, model_id="qwen-fast")
    assert (await m.aslots()) == [{"id": 0}]
    assert p._ahttp.calls[0]["path"] == "/upstream/qwen-fast/slots"


@pytest.mark.asyncio
async def test_model_ahealth_against_non_llamacpp_provider_raises_unsupported() -> None:
    from types import SimpleNamespace

    from llmfacade.exceptions import UnsupportedFeature
    from llmfacade.model import Model

    fake_provider = SimpleNamespace(NAME="madeup", SUPPORTS=frozenset())
    m = Model(provider=fake_provider, model_id="x")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedFeature, match="ahealth"):
        await m.ahealth()


# ---- async paths offload _ensure_supervised off the event loop ------------
#
# _ensure_supervised is fully synchronous (threading.Lock + subprocess spawn +
# sleep-polling readiness in _swap_lifecycle); the async surfaces must run it
# via asyncio.to_thread so the first managed-mode async call doesn't block the
# event loop for seconds. The internal lock makes the cross-thread call safe.


class _AsyncIterStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _AsyncIterStream:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_acomplete_runs_ensure_supervised_off_the_loop(
    monkeypatch, provider: LlamaCppServerProvider
) -> None:
    import threading

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        provider,
        "_ensure_supervised",
        lambda: seen.setdefault("thread", threading.current_thread()),
    )

    async def fake_create(**_kw: Any) -> _FakeResponse:
        return _FakeResponse(content="hi")

    monkeypatch.setattr(provider._aclient.chat.completions, "create", fake_create)
    resp = await provider._acomplete_raw(_req())
    assert resp.text == "hi"
    assert seen["thread"] is not threading.current_thread()


@pytest.mark.asyncio
async def test_astream_runs_ensure_supervised_off_the_loop(
    monkeypatch, provider: LlamaCppServerProvider
) -> None:
    import threading

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        provider,
        "_ensure_supervised",
        lambda: seen.setdefault("thread", threading.current_thread()),
    )
    chunks = [
        _FakeStreamChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content="hi"))]),
        _FakeStreamChunk(
            choices=[_FakeStreamChoice(delta=_FakeDelta(), finish_reason="stop")],
            usage=_FakeUsage(prompt=5, completion=2),
        ),
    ]

    async def fake_create(**_kw: Any) -> _AsyncIterStream:
        return _AsyncIterStream(chunks)

    monkeypatch.setattr(provider._aclient.chat.completions, "create", fake_create)
    events = [e async for e in provider._astream_raw(_req())]
    assert any(e.text_delta == "hi" for e in events)
    assert seen["thread"] is not threading.current_thread()


@pytest.mark.asyncio
async def test_async_introspection_awaits_ensure_supervised_via_to_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on asyncio.to_thread: the async introspection wrappers must route
    _ensure_supervised through it (delegating so behaviour is unchanged)."""
    import asyncio as _asyncio

    p = _managed_provider_with_entries(tmp_path, monkeypatch, "m")
    p._ahttp = _AsyncCapturingHttp(_FakeHttpResponse(json_body={"status": "ok"}))

    offloaded: list[Any] = []
    orig_to_thread = _asyncio.to_thread

    async def spy(fn: Any, *args: Any, **kwargs: Any) -> Any:
        offloaded.append(fn)
        return await orig_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(_asyncio, "to_thread", spy)
    await p.ahealth(model="m")
    await p.aunload("m")
    assert offloaded.count(p._ensure_supervised) == 2


# ---- mmproj_path (managed-mode vision launch knob) -----------------------


def test_external_mode_rejects_mmproj_path_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="mmproj_path"):
        LlamaCppServerProvider(base_url="http://x:0/v1", mmproj_path="m.gguf")


def test_external_mode_rejects_mmproj_path_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="mmproj_path"):
        p.new_model("qwen", mmproj_path="m.gguf")


def test_managed_mode_mmproj_path_provider_default_cascades(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", mmproj_path=str(mmproj))
    p.new_model(gguf=str(gguf))
    entry = p._supervisor.entries[0]  # type: ignore[union-attr]
    assert entry.mmproj_path == str(mmproj)


def test_managed_mode_mmproj_path_model_overrides_provider(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", mmproj_path=str(a))
    p.new_model(gguf=str(gguf), mmproj_path=str(b))
    assert p._supervisor.entries[0].mmproj_path == str(b)  # type: ignore[union-attr]


def test_managed_mode_mmproj_path_default_is_none(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].mmproj_path is None  # type: ignore[union-attr]


def test_managed_mode_new_model_missing_mmproj_path_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(FileNotFoundError, match="mmproj_path not found"):
        p.new_model(gguf=str(gguf), mmproj_path=str(tmp_path / "nonexistent.gguf"))


def test_new_model_forwards_mmproj_path_to_fit_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mmproj_path IS forwarded to llama-fit-params. The multimodal projector
    occupies VRAM alongside the main model, so an estimate that ignores it
    would under-count VRAM use."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 32"
        stderr = ""

    def fake_run(argv: list[str], **_kw: Any) -> _FakeOk:
        seen_argv["argv"] = argv
        return _FakeOk()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    p.new_model(gguf=str(gguf), name="m", mmproj_path=str(mmproj))
    argv = seen_argv["argv"]
    assert "--mmproj" in argv
    idx = argv.index("--mmproj")
    assert argv[idx + 1] == str(mmproj)


# ---- ImageBlock wire-format marshalling -----------------------------------


def test_message_to_api_routes_image_block_as_openai_image_url(
    provider: LlamaCppServerProvider,
) -> None:
    """ImageBlock on a user message becomes the OpenAI-shaped
    `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}`
    block that llama-server consumes when --mmproj is loaded."""
    import base64

    from llmfacade import Message
    from llmfacade.models import ImageBlock, TextBlock

    raw = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    img = ImageBlock(data=raw, media_type="image/png")
    msg = Message(role="user", content=[TextBlock("look at this"), img])
    api = provider._message_to_api(msg)
    assert len(api) == 1
    parts = api[0]["content"]
    assert isinstance(parts, list)
    assert {"type": "text", "text": "look at this"} in parts
    expected_url = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"] == {"url": expected_url}


def test_message_to_api_drops_image_block_on_assistant_role(
    provider: LlamaCppServerProvider,
) -> None:
    """Assistant messages are flattened to text-only on the way out — the
    OpenAI-compat surface has no role for an assistant-emitted image, and
    propagating one would hand the model a malformed message on the next
    turn. The drop warns (aligned with the OpenAI provider via the shared
    `_openai_chat.message_to_api`; historically llamacpp dropped silently)."""
    from llmfacade import Message
    from llmfacade.models import ImageBlock, TextBlock

    img = ImageBlock(data=b"\x89PNG\r\n\x1a\nignored", media_type="image/png")
    msg = Message(role="assistant", content=[TextBlock("here you go"), img])
    with pytest.warns(UserWarning, match="dropping image"):
        api = provider._message_to_api(msg)
    assert len(api) == 1
    assert api[0]["content"] == "here you go"
    assert "image_url" not in str(api[0])


# ---- thinking knob -> chat_template_kwargs (enable_thinking) ---------------

from llmfacade.providers import llamacpp as _llamacpp_mod  # noqa: E402
from llmfacade.settings import ThinkingMode, ThinkingStyle  # noqa: E402


def _req_for_model(model_id: str, **settings: Any) -> CompletionRequest:
    """A `_req`-style request bound to a specific `model` id, so a test can line
    it up with an entry in `provider._thinking_styles`."""
    return CompletionRequest(
        model=model_id,
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"max_tokens": 16, **settings},
        settings_source={},
    )


def test_thinking_in_supports_but_not_thinking_budget() -> None:
    """The `thinking` knob is settable; the budget *form* is not — so an int
    budget fails fast through the value-level gate rather than being dropped."""
    assert "thinking" in LlamaCppServerProvider.SUPPORTS
    assert "thinking_budget" not in LlamaCppServerProvider.SUPPORTS


def test_thinking_adaptive_sets_enable_thinking_true(provider: LlamaCppServerProvider) -> None:
    kwargs = provider._build_kwargs(_req(settings={"thinking": ThinkingMode.ADAPTIVE}))
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_thinking_disabled_sets_enable_thinking_false(provider: LlamaCppServerProvider) -> None:
    kwargs = provider._build_kwargs(_req(settings={"thinking": ThinkingMode.DISABLED}))
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_summarized_maps_to_enable_thinking_true(
    provider: LlamaCppServerProvider,
) -> None:
    """llama.cpp has no 'summarized' display mode, so ADAPTIVE_SUMMARIZED folds
    to thinking-on (the closest equivalent)."""
    kwargs = provider._build_kwargs(_req(settings={"thinking": ThinkingMode.ADAPTIVE_SUMMARIZED}))
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_thinking_string_value_works(provider: LlamaCppServerProvider) -> None:
    kwargs = provider._build_kwargs(_req(settings={"thinking": "disabled"}))
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_omitted_no_chat_template_kwargs(provider: LlamaCppServerProvider) -> None:
    kwargs = provider._build_kwargs(_req(settings={"temperature": 0.5}))
    assert "extra_body" not in kwargs


def test_thinking_coexists_with_samplers_in_extra_body(provider: LlamaCppServerProvider) -> None:
    """The enable_thinking kwarg rides in the same extra_body dict as the
    llama.cpp samplers, without clobbering them."""
    kwargs = provider._build_kwargs(
        _req(settings={"thinking": ThinkingMode.ADAPTIVE, "top_k": 40, "min_p": 0.05})
    )
    assert kwargs["extra_body"] == {
        "top_k": 40,
        "min_p": 0.05,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_thinking_bool_raises_typeerror(provider: LlamaCppServerProvider) -> None:
    with pytest.raises(TypeError, match="got a bool"):
        provider._build_kwargs(_req(settings={"thinking": True}))


# ---- conversation-level budget gate (reuses the Opus-4.8 mechanism) -------


def test_int_thinking_budget_rejected_by_gate() -> None:
    """An int token budget is the budget *form*, which llamacpp doesn't accept
    (`thinking_budget` not in SUPPORTS). The value-level gate in
    `Conversation._build_request` raises before any provider call."""
    p = LlamaCppServerProvider(base_url="http://invalid.local:0/v1")
    convo = p.new_model("m").new_conversation(log_dir=False)
    with pytest.raises(UnsupportedFeature, match="thinking_budget"):
        convo._build_request(stop=None, per_call={"thinking": 5})


def test_adaptive_thinking_passes_gate() -> None:
    """A ThinkingMode is never the budget form, so it sails through the gate and
    lands in the merged request settings (no network needed to verify)."""
    p = LlamaCppServerProvider(base_url="http://invalid.local:0/v1")
    convo = p.new_model("m").new_conversation(log_dir=False)
    req = convo._build_request(stop=None, per_call={"thinking": ThinkingMode.ADAPTIVE})
    assert req.settings["thinking"] == ThinkingMode.ADAPTIVE


# ---- jinja launch knob (managed mode) -------------------------------------


def test_external_mode_rejects_jinja_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="jinja"):
        LlamaCppServerProvider(base_url="http://x:0/v1", jinja=True)


def test_external_mode_rejects_jinja_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="jinja"):
        p.new_model("qwen", jinja=False)


def test_managed_mode_jinja_default_true(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].jinja is True  # type: ignore[union-attr]


def test_managed_mode_jinja_false_override(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf), jinja=False)
    assert p._supervisor.entries[0].jinja is False  # type: ignore[union-attr]


def test_managed_mode_jinja_provider_default_cascades(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", jinja=False)
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].jinja is False  # type: ignore[union-attr]


def test_managed_mode_jinja_model_overrides_provider(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", jinja=False)
    p.new_model(gguf=str(gguf), jinja=True)
    assert p._supervisor.entries[0].jinja is True  # type: ignore[union-attr]


def test_new_model_does_not_forward_jinja_to_fit_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--jinja` governs chat templating, not VRAM layout, and fit-params would
    reject the flag — so it's intentionally NOT forwarded to the estimate."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 32"
        stderr = ""

    def fake_run(argv: list[str], **_kw: Any) -> _FakeOk:
        seen_argv["argv"] = argv
        return _FakeOk()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    p.new_model(gguf=str(gguf), name="m", jinja=True)
    assert "--jinja" not in seen_argv["argv"]


# ---- no_mmap / mlock memory-residency launch knobs ------------------------


def test_external_mode_rejects_no_mmap_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="no_mmap"):
        LlamaCppServerProvider(base_url="http://x:0/v1", no_mmap=True)


def test_external_mode_rejects_no_mmap_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="no_mmap"):
        p.new_model("qwen", no_mmap=True)


def test_external_mode_rejects_mlock_in_init() -> None:
    with pytest.raises(UnsupportedFeature, match="mlock"):
        LlamaCppServerProvider(base_url="http://x:0/v1", mlock=True)


def test_external_mode_rejects_mlock_in_new_model() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    with pytest.raises(UnsupportedFeature, match="mlock"):
        p.new_model("qwen", mlock=True)


def test_managed_mode_no_mmap_mlock_default_false(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].no_mmap is False  # type: ignore[union-attr]
    assert p._supervisor.entries[0].mlock is False  # type: ignore[union-attr]


@pytest.mark.filterwarnings("ignore:model .* sets both no_mmap=True and mlock=True")
def test_managed_mode_no_mmap_mlock_provider_default_cascades(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", no_mmap=True, mlock=True)
    p.new_model(gguf=str(gguf))
    assert p._supervisor.entries[0].no_mmap is True  # type: ignore[union-attr]
    assert p._supervisor.entries[0].mlock is True  # type: ignore[union-attr]


def test_managed_mode_no_mmap_model_overrides_provider(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess", no_mmap=True)
    p.new_model(gguf=str(gguf), no_mmap=False)
    assert p._supervisor.entries[0].no_mmap is False  # type: ignore[union-attr]


@pytest.mark.filterwarnings("ignore:model .* sets both no_mmap=True and mlock=True")
def test_new_model_does_not_forward_no_mmap_mlock_to_fit_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-mmap`/`--mlock` govern memory residency, not VRAM layout. fit-params
    would reject them, and `--no-mmap` would force the probe to preload the whole
    model into RAM — so neither is forwarded to the estimate."""
    gguf = tmp_path / "qwen.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")

    seen_argv: dict[str, Any] = {}

    class _FakeOk:
        returncode = 0
        stdout = "-c 4096 -ngl 32"
        stderr = ""

    def fake_run(argv: list[str], **_kw: Any) -> _FakeOk:
        seen_argv["argv"] = argv
        return _FakeOk()

    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setattr(_shutil, "which", lambda b: "/usr/local/bin/" + b)
    monkeypatch.setattr(_subprocess, "run", fake_run)

    p.new_model(gguf=str(gguf), name="m", no_mmap=True, mlock=True)
    assert "--no-mmap" not in seen_argv["argv"]
    assert "--mlock" not in seen_argv["argv"]


# ---- thinking_style auto-detect / override / log surfacing / warning ------


def _write_gguf_with_template(path: Path, template: str) -> None:
    """Write a minimal GGUF (header KV metadata only) carrying just the
    `tokenizer.chat_template` string, enough for the reader to extract it."""
    import struct

    kb = b"tokenizer.chat_template"
    vb = template.encode("utf-8")
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)  # version
        + struct.pack("<Q", 0)  # tensor_count
        + struct.pack("<Q", 1)  # kv_count
        + struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", 8)  # value type = STRING
        + struct.pack("<Q", len(vb))
        + vb
    )


def test_managed_new_model_autodetects_template_kwarg_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "gemma.gguf"
    _write_gguf_with_template(gguf, "{% if enable_thinking %}<think>{% endif %}")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    # Skip the fit-params probe so the test is fast and deterministic.
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _b: None)
    model = p.new_model(gguf=str(gguf), name="gemma")
    assert p._thinking_styles[model.model_id] == ThinkingStyle.TEMPLATE_KWARG


def test_managed_new_model_explicit_thinking_style_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit thinking_style= overrides whatever the template would detect."""
    gguf = tmp_path / "gemma.gguf"
    _write_gguf_with_template(gguf, "{% if enable_thinking %}{% endif %}")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _b: None)
    model = p.new_model(gguf=str(gguf), name="g", thinking_style=ThinkingStyle.THINK_TOKEN)
    assert p._thinking_styles[model.model_id] == ThinkingStyle.THINK_TOKEN


def test_new_model_invalid_thinking_style_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    with pytest.raises(ValueError, match="thinking_style must be"):
        p.new_model(gguf=str(gguf), name="m", thinking_style="bogus")


def test_external_mode_explicit_thinking_style_stored() -> None:
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    p.new_model("m", thinking_style="template_kwarg")
    assert p._thinking_styles["m"] == ThinkingStyle.TEMPLATE_KWARG


def test_external_mode_no_thinking_style_absent_from_log_metadata() -> None:
    """External mode can't read a local GGUF, so without an explicit
    thinking_style= nothing is stored and log_metadata stays None."""
    p = LlamaCppServerProvider(base_url="http://x:0/v1")
    p.new_model("m")
    assert "m" not in p._thinking_styles
    assert p.log_metadata(model_id="m") is None


def test_thinking_style_surfaced_in_log_metadata(tmp_path: Path) -> None:
    gguf = tmp_path / "q.gguf"
    gguf.write_bytes(b"fake")  # no readable template -> UNKNOWN
    p = LlamaCppServerProvider(llmfacade_dir=tmp_path / "sess")
    model = p.new_model(gguf=str(gguf), name="m", thinking_style=ThinkingStyle.TEMPLATE_KWARG)
    meta = p.log_metadata(model_id=model.model_id)
    assert meta is not None
    assert meta["thinking_style"] == "template_kwarg"


def test_warns_when_thinking_set_on_non_kwarg_style(
    provider: LlamaCppServerProvider, recwarn: pytest.WarningsRecorder
) -> None:
    _llamacpp_mod._WARNED_THINKING_STYLE.clear()
    provider._thinking_styles["tk-warn"] = ThinkingStyle.THINK_TOKEN
    req = _req_for_model("tk-warn", thinking=ThinkingMode.ADAPTIVE)
    provider._build_kwargs(req)
    provider._build_kwargs(req)  # second call must not re-warn
    msgs = [str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert len(msgs) == 1
    assert "thinking_style" in msgs[0] and "tk-warn" in msgs[0]
    # The kwarg is still emitted ("never silently wrong" warns, doesn't drop).
    assert provider._build_kwargs(req)["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_no_warn_when_thinking_set_on_template_kwarg_style(
    provider: LlamaCppServerProvider, recwarn: pytest.WarningsRecorder
) -> None:
    _llamacpp_mod._WARNED_THINKING_STYLE.clear()
    provider._thinking_styles["tkw"] = ThinkingStyle.TEMPLATE_KWARG
    req = _req_for_model("tkw", thinking=ThinkingMode.ADAPTIVE)
    provider._build_kwargs(req)
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []


def test_no_warn_when_thinking_style_unknown(
    provider: LlamaCppServerProvider, recwarn: pytest.WarningsRecorder
) -> None:
    """UNKNOWN means we couldn't detect — warning then would be a false positive
    (and every external-mode model without an explicit style is UNKNOWN)."""
    _llamacpp_mod._WARNED_THINKING_STYLE.clear()
    provider._thinking_styles["unk"] = ThinkingStyle.UNKNOWN
    req = _req_for_model("unk", thinking=ThinkingMode.ADAPTIVE)
    provider._build_kwargs(req)
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []
