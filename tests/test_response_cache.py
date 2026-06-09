"""Filesystem-backed response cache: hashing, hit/miss, modes, streaming."""

from __future__ import annotations

import asyncio

import pytest

from llmfacade import (
    CacheMissError,
    ImageBlock,
    SystemBlock,
    TextBlock,
    Usage,
    tool,
)
from llmfacade.cache import (
    fingerprint_request,
    hash_fingerprint,
    replay_stream,
)
from llmfacade.models import StreamEvent, ToolCall, ToolUseBlock

from .conftest import MockProvider

# --- fingerprint stability ----------------------------------------------


def _build_req(convo, *, prompt: str | list = "hello"):
    """Drive a convo to the point where we can introspect a CompletionRequest
    without making the provider call. Returns the CompletionRequest the
    convo would send for ``prompt``."""
    from llmfacade.models import Message

    convo._history.append(Message(role="user", content=prompt))
    return convo._build_request(stop=None, per_call={})


def test_hash_stable_across_runs(mock_model):
    convo1 = mock_model.new_conversation(system_blocks=["sys"])
    convo2 = mock_model.new_conversation(system_blocks=["sys"])
    req1 = _build_req(convo1)
    req2 = _build_req(convo2)
    fp1 = fingerprint_request(req1, "mock")
    fp2 = fingerprint_request(req2, "mock")
    assert hash_fingerprint(fp1) == hash_fingerprint(fp2)


def test_hash_changes_with_prompt(mock_model):
    convo = mock_model.new_conversation()
    req_a = _build_req(convo, prompt="hello")
    convo._history.clear()
    req_b = _build_req(convo, prompt="goodbye")
    assert hash_fingerprint(fingerprint_request(req_a, "mock")) != hash_fingerprint(
        fingerprint_request(req_b, "mock")
    )


def test_hash_changes_with_settings(mock_model):
    p: MockProvider = mock_model.provider
    convo_a = mock_model.new_conversation(temperature=0.1)
    convo_b = mock_model.new_conversation(temperature=0.9)
    req_a = _build_req(convo_a)
    req_b = _build_req(convo_b)
    assert hash_fingerprint(fingerprint_request(req_a, p.NAME)) != hash_fingerprint(
        fingerprint_request(req_b, p.NAME)
    )


def _req_with_settings(settings: dict):
    from llmfacade.models import Message
    from llmfacade.provider import CompletionRequest

    return CompletionRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings=settings,
        settings_source={k: "convo" for k in settings},
    )


def test_normalize_unwraps_drysampler_dataclass():
    """A DrySampler knob value normalises to a plain sorted dict (not a repr
    blob), so it hashes structurally and logs readably."""
    from llmfacade import DrySampler
    from llmfacade.cache import _normalize

    norm = _normalize(DrySampler(multiplier=0.8, sequence_breakers=("\n",)))
    assert norm == {
        "allowed_length": 2,
        "base": 1.75,
        "multiplier": 0.8,
        "penalty_last_n": -1,
        "sequence_breakers": ["\n"],
    }


def test_hash_changes_with_dry_config():
    from llmfacade import DrySampler

    a = _req_with_settings({"dry": DrySampler(multiplier=0.8)})
    b = _req_with_settings({"dry": DrySampler(multiplier=0.9)})
    none = _req_with_settings({})
    h_a = hash_fingerprint(fingerprint_request(a, "llamacpp"))
    h_b = hash_fingerprint(fingerprint_request(b, "llamacpp"))
    h_none = hash_fingerprint(fingerprint_request(none, "llamacpp"))
    assert h_a != h_b  # different DRY configs → different keys
    assert h_a != h_none  # DRY on vs off → different keys


def test_hash_stable_for_equal_dry_config():
    from llmfacade import DrySampler

    a = _req_with_settings({"dry": DrySampler(multiplier=0.8, sequence_breakers=("\n",))})
    b = _req_with_settings({"dry": DrySampler(multiplier=0.8, sequence_breakers=("\n",))})
    assert hash_fingerprint(fingerprint_request(a, "llamacpp")) == hash_fingerprint(
        fingerprint_request(b, "llamacpp")
    )


def test_hash_changes_with_system_block_cache_flag(mock_model):
    """Per the design choice: cache=True markers are part of the fingerprint
    even though they don't affect generation, so flipping caching gets fresh
    output."""
    convo_a = mock_model.new_conversation(system_blocks=[SystemBlock("sys", cache=False)])
    convo_b = mock_model.new_conversation(system_blocks=[SystemBlock("sys", cache=True)])
    req_a = _build_req(convo_a)
    req_b = _build_req(convo_b)
    assert hash_fingerprint(fingerprint_request(req_a, "mock")) != hash_fingerprint(
        fingerprint_request(req_b, "mock")
    )


def test_hash_changes_with_provider_name(mock_model):
    convo = mock_model.new_conversation()
    req = _build_req(convo)
    assert hash_fingerprint(fingerprint_request(req, "mock")) != hash_fingerprint(
        fingerprint_request(req, "anthropic")
    )


def test_hash_changes_with_base_url(mock_model):
    convo = mock_model.new_conversation()
    req = _build_req(convo)
    h_a = hash_fingerprint(fingerprint_request(req, "mock", base_url="http://a:8080/v1"))
    h_b = hash_fingerprint(fingerprint_request(req, "mock", base_url="http://b:8080/v1"))
    h_default = hash_fingerprint(fingerprint_request(req, "mock"))
    assert h_a != h_b  # different endpoints → different keys
    assert h_a != h_default  # explicit endpoint vs default → different keys


def test_hash_stable_for_default_base_url(mock_model):
    convo = mock_model.new_conversation()
    req = _build_req(convo)
    assert hash_fingerprint(fingerprint_request(req, "mock")) == hash_fingerprint(
        fingerprint_request(req, "mock", base_url=None)
    )


def test_hash_changes_with_provider_base_url():
    """Two provider instances differing only in base_url fingerprint differently."""
    p_a = MockProvider(base_url="http://a:8080/v1")
    p_b = MockProvider(base_url="http://b:8080/v1")
    req_a = _build_req(p_a.new_model("alias").new_conversation())
    req_b = _build_req(p_b.new_model("alias").new_conversation())
    h_a = hash_fingerprint(fingerprint_request(req_a, p_a.NAME, base_url=p_a._base_url))
    h_b = hash_fingerprint(fingerprint_request(req_b, p_b.NAME, base_url=p_b._base_url))
    assert h_a != h_b


def test_hash_changes_with_image_bytes(mock_model):
    convo_a = mock_model.new_conversation()
    convo_b = mock_model.new_conversation()
    img_a = ImageBlock(data=b"\x00\x01", media_type="image/png")
    img_b = ImageBlock(data=b"\x00\x02", media_type="image/png")
    req_a = _build_req(convo_a, prompt=[TextBlock("look"), img_a])
    req_b = _build_req(convo_b, prompt=[TextBlock("look"), img_b])
    assert hash_fingerprint(fingerprint_request(req_a, "mock")) != hash_fingerprint(
        fingerprint_request(req_b, "mock")
    )


def test_hash_changes_with_tool_schema(mock_model):
    @tool
    def alpha(x: int) -> str:
        """alpha"""
        return str(x)

    @tool
    def beta(x: str) -> str:
        """beta"""
        return x

    convo_a = mock_model.new_conversation(tools=[alpha])
    convo_b = mock_model.new_conversation(tools=[beta])
    req_a = _build_req(convo_a)
    req_b = _build_req(convo_b)
    assert hash_fingerprint(fingerprint_request(req_a, "mock")) != hash_fingerprint(
        fingerprint_request(req_b, "mock")
    )


# --- send: cache hit / miss ---------------------------------------------


def test_send_writes_then_reads(tmp_path):
    p = MockProvider(canned_text="cached-answer")
    model = p.new_model("mock-model")

    # Convo 1: write
    c1 = model.new_conversation(cache_dir=tmp_path)
    r1 = c1.send("question")
    assert r1.text == "cached-answer"
    assert len(p.calls) == 1

    # Convo 2: same model, same prompt -> hit, no second call
    c2 = model.new_conversation(cache_dir=tmp_path)
    r2 = c2.send("question")
    assert r2.text == "cached-answer"
    assert len(p.calls) == 1, "provider should not have been called on cache hit"
    assert c2.history[-1].role == "assistant"


def test_send_miss_when_prompt_changes(tmp_path):
    p = MockProvider(canned_text="answer")
    model = p.new_model("mock-model")
    c1 = model.new_conversation(cache_dir=tmp_path)
    c1.send("first")
    assert len(p.calls) == 1
    c2 = model.new_conversation(cache_dir=tmp_path)
    c2.send("second")
    assert len(p.calls) == 2


def test_send_does_not_replay_across_base_urls(tmp_path):
    """Two same-NAME providers at different endpoints sharing a cache_dir
    must not replay each other's output (e.g. two external llama-servers
    each serving a different GGUF under the same alias)."""
    p_a = MockProvider(base_url="http://a:8080/v1", canned_text="from-a")
    p_b = MockProvider(base_url="http://b:8080/v1", canned_text="from-b")
    c_a = p_a.new_model("alias").new_conversation(cache_dir=tmp_path)
    c_b = p_b.new_model("alias").new_conversation(cache_dir=tmp_path)

    assert c_a.send("q").text == "from-a"
    assert len(p_a.calls) == 1

    # Same prompt, same provider NAME and model alias, different endpoint:
    # must miss and call provider b, not replay a's cached response.
    assert c_b.send("q").text == "from-b"
    assert len(p_b.calls) == 1

    # Same endpoint as a -> still a hit.
    c_a2 = p_a.new_model("alias").new_conversation(cache_dir=tmp_path)
    assert c_a2.send("q").text == "from-a"
    assert len(p_a.calls) == 1


def test_read_only_does_not_write(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="read_only")
    c.send("q")
    assert len(p.calls) == 1
    # Nothing should have been written.
    assert not list(tmp_path.rglob("*.json"))


def test_record_only_writes_but_always_calls_provider(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c1 = model.new_conversation(cache_dir=tmp_path, cache_mode="record_only")
    c1.send("q")
    c2 = model.new_conversation(cache_dir=tmp_path, cache_mode="record_only")
    c2.send("q")
    assert len(p.calls) == 2
    assert list(tmp_path.rglob("*.json"))


def test_replay_only_raises_on_miss(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    with pytest.raises(CacheMissError):
        c.send("never-cached")
    assert len(p.calls) == 0


def test_replay_only_miss_send_leaves_history_unchanged(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    with pytest.raises(CacheMissError):
        c.send("never-cached")
    assert c.history == [], "the dangling prompt must be rolled back on a replay_only miss"
    # A later turn must not trip over a dangling user message.
    with pytest.raises(CacheMissError):
        c.send("still-never-cached")
    assert c.history == []


def test_replay_only_miss_asend_leaves_history_unchanged(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    with pytest.raises(CacheMissError):
        asyncio.run(c.asend("never-cached"))
    assert c.history == []


def test_replay_only_miss_stream_leaves_history_unchanged(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    with pytest.raises(CacheMissError):
        list(c.stream("never-cached"))
    assert c.history == []


def test_replay_only_miss_astream_leaves_history_unchanged(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")

    async def run():
        async for _ev in c.astream("never-cached"):
            pass

    with pytest.raises(CacheMissError):
        asyncio.run(run())
    assert c.history == []


def test_replay_only_miss_preserves_prior_turns(tmp_path):
    p = MockProvider(canned_text="seeded")
    model = p.new_model("mock-model")
    seed = model.new_conversation(cache_dir=tmp_path)
    seed.send("hello")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    c.send("hello")  # hit
    before = c.history
    with pytest.raises(CacheMissError):
        c.send("uncached-follow-up")
    assert c.history == before


def test_replay_only_returns_hit(tmp_path):
    p = MockProvider(canned_text="seeded")
    model = p.new_model("mock-model")
    # Seed via read_write run.
    seed = model.new_conversation(cache_dir=tmp_path)
    seed.send("hello")
    # Now replay-only.
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    r = c.send("hello")
    assert r.text == "seeded"
    assert len(p.calls) == 1


# --- cascade -------------------------------------------------------------


def test_cache_cascade_provider_to_convo(tmp_path):
    p = MockProvider(canned_text="x", cache_dir=tmp_path)
    model = p.new_model("mock-model")
    c = model.new_conversation()
    c.send("q")
    c2 = model.new_conversation()
    c2.send("q")
    assert len(p.calls) == 1


def test_cache_cascade_convo_disable(tmp_path):
    p = MockProvider(canned_text="x", cache_dir=tmp_path)
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=False)
    c.send("q")
    c2 = model.new_conversation(cache_dir=False)
    c2.send("q")
    assert len(p.calls) == 2  # cache disabled per-convo


def test_cache_mode_cascade(tmp_path):
    p = MockProvider(canned_text="x", cache_dir=tmp_path, cache_mode="record_only")
    model = p.new_model("mock-model")
    c1 = model.new_conversation()
    c1.send("q")
    c2 = model.new_conversation()
    c2.send("q")
    # record_only inherits from provider -> always call provider.
    assert len(p.calls) == 2


# --- streaming -----------------------------------------------------------


def test_stream_replay_yields_events(tmp_path):
    p = MockProvider(canned_text="hello world")
    model = p.new_model("mock-model")
    seed = model.new_conversation(cache_dir=tmp_path)
    seed_text = "".join(ev.text_delta for ev in seed.stream("q") if ev.text_delta)
    assert len(p.calls) == 1

    replay = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    events = list(replay.stream("q"))
    assert len(p.calls) == 1, "no provider call expected on replay"

    replay_text = "".join(ev.text_delta for ev in events if ev.text_delta)
    # Replay must reproduce exactly what the seed run assembled — even
    # whitespace artefacts from the mock chunker. Equivalence, not the
    # canned string.
    assert replay_text == seed_text
    assert events[-1].done is True
    # History should still hold the assistant turn.
    assert replay.history[-1].role == "assistant"


def test_stream_records_on_miss(tmp_path):
    p = MockProvider(canned_text="streamed")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path)
    list(c.stream("hi"))
    assert list(tmp_path.rglob("*.json"))


def test_replay_only_stream_raises_on_miss(tmp_path):
    p = MockProvider(canned_text="x")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    with pytest.raises(CacheMissError):
        list(c.stream("never"))


# --- early stream exit must not poison the cache ---------------------------


class _ExplodingStreamProvider(MockProvider):
    """Streams one text delta, then raises mid-stream."""

    def _stream_raw(self, req):
        yield StreamEvent(text_delta="partial ")
        raise RuntimeError("connection dropped")

    async def _astream_raw(self, req):
        yield StreamEvent(text_delta="partial ")
        raise RuntimeError("connection dropped")


def test_stream_consumer_break_does_not_cache(tmp_path):
    p = MockProvider(canned_text="one two three four")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path)
    for _ev in c.stream("hi"):
        break  # consumer bails after the first event
    assert not list(tmp_path.rglob("*.json")), "truncated response must not be cached"
    # The partial assistant turn still lands in history (role alternation).
    assert c.history[-1].role == "assistant"


def test_stream_provider_error_does_not_cache(tmp_path):
    p = _ExplodingStreamProvider()
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="connection dropped"):
        for _ev in c.stream("hi"):
            pass
    assert not list(tmp_path.rglob("*.json")), "truncated response must not be cached"


def test_stream_natural_completion_still_cached(tmp_path):
    p = MockProvider(canned_text="full answer")
    model = p.new_model("mock-model")
    c = model.new_conversation(cache_dir=tmp_path)
    for _ev in c.stream("hi"):
        pass
    assert list(tmp_path.rglob("*.json"))


def test_astream_consumer_break_does_not_cache(tmp_path):
    p = MockProvider(canned_text="one two three four")
    model = p.new_model("mock-model")

    async def run():
        c = model.new_conversation(cache_dir=tmp_path)
        async for _ev in c.astream("hi"):
            break
        return c

    c = asyncio.run(run())
    assert not list(tmp_path.rglob("*.json")), "truncated response must not be cached"
    assert c.history[-1].role == "assistant"


def test_astream_provider_error_does_not_cache(tmp_path):
    p = _ExplodingStreamProvider()
    model = p.new_model("mock-model")

    async def run():
        c = model.new_conversation(cache_dir=tmp_path)
        async for _ev in c.astream("hi"):
            pass

    with pytest.raises(RuntimeError, match="connection dropped"):
        asyncio.run(run())
    assert not list(tmp_path.rglob("*.json")), "truncated response must not be cached"


def test_astream_natural_completion_still_cached(tmp_path):
    p = MockProvider(canned_text="full answer")
    model = p.new_model("mock-model")

    async def run():
        c = model.new_conversation(cache_dir=tmp_path)
        async for _ev in c.astream("hi"):
            pass

    asyncio.run(run())
    assert list(tmp_path.rglob("*.json"))


# --- async ---------------------------------------------------------------


def test_asend_uses_cache(tmp_path):
    p = MockProvider(canned_text="async")
    model = p.new_model("mock-model")

    async def run():
        c1 = model.new_conversation(cache_dir=tmp_path)
        await c1.asend("hi")
        c2 = model.new_conversation(cache_dir=tmp_path)
        await c2.asend("hi")

    asyncio.run(run())
    assert len(p.calls) == 1


def test_astream_replay(tmp_path):
    p = MockProvider(canned_text="hi there")
    model = p.new_model("mock-model")

    async def seed():
        c = model.new_conversation(cache_dir=tmp_path)
        out = []
        async for ev in c.astream("hi"):
            if ev.text_delta:
                out.append(ev.text_delta)
        return "".join(out)

    async def replay():
        c = model.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
        out = []
        async for ev in c.astream("hi"):
            if ev.text_delta:
                out.append(ev.text_delta)
        return "".join(out)

    seed_text = asyncio.run(seed())
    replay_text = asyncio.run(replay())
    assert replay_text == seed_text
    assert len(p.calls) == 1


# --- multi-turn / tool replay -------------------------------------------


def test_tool_replay_round_trip(tmp_path):
    """After a turn that produced tool calls, sending the tool result and
    receiving the model's continuation should hit the cache on a second run
    of the same sequence."""
    # Turn 1: model emits a tool call.
    p = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 1})],
    )
    model = p.new_model("mock-model")

    def drive(p_):
        c = model.new_conversation(cache_dir=tmp_path)
        c.send("go")
        # Append tool result.
        c.add_tool_result("t1", "result-text", name="echo")
        # Turn 2: model now returns a final answer.
        p_.canned_text = "final"
        p_.canned_tool_calls = []
        c.send()
        return c

    c1 = drive(p)
    assert len(p.calls) == 2
    assert c1.history[-1].role == "assistant"

    # Reset canned state and replay the same sequence on a fresh provider.
    p2 = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 1})],
    )
    model2 = p2.new_model("mock-model")
    c2 = model2.new_conversation(cache_dir=tmp_path, cache_mode="replay_only")
    r1 = c2.send("go")
    assert r1.tool_calls and r1.tool_calls[0].name == "echo"
    c2.add_tool_result("t1", "result-text", name="echo")
    r2 = c2.send()
    assert r2.text == "final"
    assert len(p2.calls) == 0


def test_usage_reasoning_tokens_round_trip():
    """reasoning_tokens survives serialise → deserialise; an older cache entry
    written without the field deserialises to the default 0."""
    from llmfacade.cache import _deserialize_response, _serialize_response
    from llmfacade.models import Response

    resp = Response(
        text="answer",
        blocks=[TextBlock(text="answer")],
        tool_calls=[],
        thinking="reasoning",
        usage=Usage(prompt_tokens=5, completion_tokens=20, total_tokens=25, reasoning_tokens=12),
        finish_reason="stop",
        model="mock",
    )
    restored = _deserialize_response(_serialize_response(resp))
    assert restored.usage is not None
    assert restored.usage.reasoning_tokens == 12

    # Legacy entry: usage dict predating the field → default 0, no crash.
    legacy = {
        "text": "x",
        "blocks": [],
        "tool_calls": [],
        "thinking": None,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "finish_reason": "stop",
        "model": "mock",
    }
    legacy_restored = _deserialize_response(legacy)
    assert legacy_restored.usage is not None
    assert legacy_restored.usage.reasoning_tokens == 0


# --- replay_stream helper directly --------------------------------------


def test_replay_stream_emits_thinking_then_text_then_tools():
    from llmfacade.models import Response, ThinkingBlock

    blocks = [
        ThinkingBlock(text="reasoning"),
        TextBlock(text="answer"),
        ToolUseBlock(id="t1", name="f", input={}),
    ]
    resp = Response(
        text="answer",
        blocks=blocks,
        tool_calls=[ToolCall(id="t1", name="f", input={})],
        thinking="reasoning",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="end_turn",
        model="mock",
    )
    events = list(replay_stream(resp))
    # First event family: thinking.
    assert events[0].thinking_delta == "reasoning"
    assert events[1].thinking_block is not None
    # Then a single text_delta carrying the full text.
    text_evs = [e for e in events if e.text_delta]
    assert len(text_evs) == 1 and text_evs[0].text_delta == "answer"
    # Then a tool call event, then done.
    assert any(e.tool_call_delta is not None for e in events)
    assert events[-1].done and events[-1].usage is not None
