"""Tests for the repetition-loop detector, the RepetitionGuard config knob,
the cascade resolution, and the send/stream retry behaviour."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Iterator

import pytest

from llmfacade import RepetitionGuard, RepetitionLoopError
from llmfacade.conversation import (
    _detect_in_tail,
    _detection_text,
    _DetectionTail,
    _StreamBuffers,
)
from llmfacade.models import Response, StreamEvent, ToolArgsDelta, ToolCall, Usage
from llmfacade.provider import CompletionRequest, Provider
from llmfacade.repetition import (
    coerce_repetition_guard,
    detect_repetition_loop,
    resolve_repetition_guard,
)

# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------


def test_period1_hits_at_threshold():
    assert detect_repetition_loop("a" * 20) is not None
    assert detect_repetition_loop("a" * 19) is None


def test_period_bands():
    # period 2: needs 8 reps and >=24 chars -> "ab"*12 = 24 chars
    assert detect_repetition_loop("ab" * 12) is not None
    assert detect_repetition_loop("ab" * 8) is None  # 16 chars < 24 total
    # period 4: needs 8 reps and >=24 chars -> "abcd"*8 = 32 chars
    assert detect_repetition_loop("abcd" * 8) is not None
    # period 5-10 band: needs 5 reps and >=30 chars -> "spam "*6 = 30 chars
    assert detect_repetition_loop("spam " * 6) is not None
    assert detect_repetition_loop("spam " * 4) is None  # 4 reps < 5


def test_smallest_period_canonicalization():
    hit = detect_repetition_loop("the" * 20)
    assert hit is not None
    # Reported as period 3 "the", not 9 or any multiple.
    assert "(len=3)" in hit
    assert "'the'" in hit


def test_alphanumeric_guard_suppresses_ascii_art():
    assert detect_repetition_loop("-" * 200) is None
    assert detect_repetition_loop("|---|" * 50) is None
    assert detect_repetition_loop("_" * 200) is None
    assert detect_repetition_loop(" " * 200) is None
    assert detect_repetition_loop("=-" * 100) is None


def test_real_loop_in_prose_tail_is_caught():
    text = "Here is the answer. " + ("I cannot help with that. " * 40)
    assert detect_repetition_loop(text) is not None


def test_empty_and_short_text():
    assert detect_repetition_loop("") is None
    assert detect_repetition_loop("hello world, this is fine.") is None


def test_min_reps_floor_raises_strictness():
    # "spam "*6 normally hits (period-5 band needs 5 reps); a floor of 10
    # requires more reps, so it no longer fires.
    assert detect_repetition_loop("spam " * 6) is not None
    assert detect_repetition_loop("spam " * 6, min_reps_floor=10) is None
    assert detect_repetition_loop("spam " * 12, min_reps_floor=10) is not None


def test_tail_chars_bounds_scan():
    # A loop buried before the tail window isn't seen.
    text = ("spam " * 6) + ("X clean unique trailing content that does not repeat. " * 3)
    assert detect_repetition_loop(text, tail_chars=40) is None


# ---------------------------------------------------------------------------
# RepetitionGuard + coercion
# ---------------------------------------------------------------------------


def test_coerce_none_and_false_disable():
    assert coerce_repetition_guard(None) is None
    assert coerce_repetition_guard(False) is None


def test_coerce_int_maps_to_min_reps_floor():
    g = coerce_repetition_guard(3)
    assert isinstance(g, RepetitionGuard)
    assert g.min_reps_floor == 3
    assert g.retries == 2  # default


def test_coerce_guard_passthrough():
    g = RepetitionGuard(retries=5)
    assert coerce_repetition_guard(g) is g


def test_coerce_true_and_bad_type_raise():
    with pytest.raises(TypeError):
        coerce_repetition_guard(True)
    with pytest.raises(TypeError):
        coerce_repetition_guard("loud")


def test_guard_validation():
    with pytest.raises(ValueError):
        RepetitionGuard(retries=-1)
    with pytest.raises(ValueError):
        RepetitionGuard(tail_chars=0)
    with pytest.raises(ValueError):
        RepetitionGuard(check_every=0)
    with pytest.raises(ValueError):
        RepetitionGuard(on_exhausted="explode")


# ---------------------------------------------------------------------------
# Fake provider that streams a looping body for the first K attempts
# ---------------------------------------------------------------------------


class LoopingProvider(Provider):
    """Streams a degenerate loop for the first ``loop_attempts`` calls, then a
    clean body. Records each request so escalation can be inspected."""

    NAME = "looping"
    SUPPORTS = frozenset({"max_tokens", "temperature", "repeat_penalty", "dry", "tools"})

    def __init__(
        self,
        *,
        loop_attempts: int = 1,
        loop_reps: int = 100,
        clean_text: str = "All good.",
        thinking_loop_attempts: int = 0,
        thinking_reps: int = 100,
        clean_thinking: str = "",
        **knobs,
    ):
        self.loop_attempts = loop_attempts
        self.loop_reps = loop_reps
        self.clean_text = clean_text
        self.thinking_loop_attempts = thinking_loop_attempts
        self.thinking_reps = thinking_reps
        self.clean_thinking = clean_thinking
        self.stream_count = 0
        self.complete_count = 0
        self.reqs: list[CompletionRequest] = []
        super().__init__(**knobs)

    def _init_client(self) -> None:
        self._client = object()

    def _resolve_key(self, env_var: str) -> str:
        del env_var
        return "key"

    _USAGE = Usage(prompt_tokens=5, completion_tokens=50, total_tokens=55)

    def _body_events(self, attempt: int) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        # Reasoning stream comes first (as real providers emit it), so a
        # looping chain-of-thought is caught before any text is produced.
        if attempt < self.thinking_loop_attempts:
            events += [StreamEvent(thinking_delta="reason ") for _ in range(self.thinking_reps)]
        elif self.clean_thinking:
            events.append(StreamEvent(thinking_delta=self.clean_thinking))
        if attempt < self.loop_attempts:
            events += [StreamEvent(text_delta="spam ") for _ in range(self.loop_reps)]
        else:
            events.append(StreamEvent(text_delta=self.clean_text))
        events.append(StreamEvent(done=True, usage=self._USAGE, finish_reason="stop"))
        return events

    def _complete_raw(self, req: CompletionRequest) -> Response:
        self.complete_count += 1
        self.reqs.append(req)
        return Response(
            text=self.clean_text,
            blocks=[],
            tool_calls=[],
            thinking=None,
            usage=self._USAGE,
            finish_reason="stop",
            model=req.model,
        )

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        return self._complete_raw(req)

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        attempt = self.stream_count
        self.stream_count += 1
        self.reqs.append(req)
        yield from self._body_events(attempt)

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        attempt = self.stream_count
        self.stream_count += 1
        self.reqs.append(req)
        for ev in self._body_events(attempt):
            yield ev


def _convo(provider: LoopingProvider, **convo_kwargs):
    return provider.new_model("loop-model").new_conversation(**convo_kwargs)


# ---------------------------------------------------------------------------
# send / asend retry behaviour
# ---------------------------------------------------------------------------


def test_send_without_guard_uses_complete_raw():
    p = LoopingProvider(loop_attempts=0)
    convo = _convo(p)
    resp = convo.send("hi")
    assert resp.text == "All good."
    assert p.complete_count == 1
    assert p.stream_count == 0  # no guard -> no streaming-under-the-hood


def test_send_retries_then_succeeds():
    p = LoopingProvider(loop_attempts=1)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    resp = convo.send("hi")
    assert resp.text == "All good."
    assert p.stream_count == 2  # attempt 0 looped, attempt 1 clean
    assert convo.history[-1].role == "assistant"
    assert convo.history[-1].content[0].text == "All good."


def test_send_catches_short_loop_below_check_every():
    # A complete degenerate loop shorter than check_every (64) chars must still
    # be caught by the post-stream final flush. "spam "*6 = 30 chars.
    p = LoopingProvider(loop_attempts=1, loop_reps=6)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    resp = convo.send("hi")
    assert resp.text == "All good."
    assert p.stream_count == 2  # short loop on attempt 0 detected, retried clean


def test_stream_catches_short_loop_below_check_every():
    p = LoopingProvider(loop_attempts=99, loop_reps=6)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    with pytest.raises(RepetitionLoopError):
        for _ev in convo.stream("hi"):
            pass
    assert convo.history == []


def test_send_raises_after_retries_exhausted():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    with pytest.raises(RepetitionLoopError) as ei:
        convo.send("hi")
    assert ei.value.attempts == 3  # retries=2 -> 3 attempts
    assert "spam" in ei.value.partial_text
    assert p.stream_count == 3
    # Conversation rolled back to its pre-send state on failure.
    assert convo.history == []


def test_send_return_last_keeps_looping_output():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=1, on_exhausted="return_last"))
    resp = convo.send("hi")
    assert "spam" in resp.text
    assert convo.history[-1].role == "assistant"


def test_escalation_bumps_repeat_penalty_per_retry():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2, escalate_repeat_penalty=0.5))
    with pytest.raises(RepetitionLoopError):
        convo.send("hi")
    penalties = [r.settings.get("repeat_penalty") for r in p.reqs]
    assert penalties == [None, 1.5, 2.0]


def test_escalation_enables_dry_on_retry():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(
        p,
        repetition_detection=RepetitionGuard(
            retries=1, escalate_repeat_penalty=None, escalate_dry=True
        ),
    )
    with pytest.raises(RepetitionLoopError):
        convo.send("hi")
    dry_values = [r.settings.get("dry") for r in p.reqs]
    assert dry_values[0] is None
    assert dry_values[1] is not None
    assert dry_values[1].multiplier == 0.5


def test_asend_retries_then_succeeds():
    p = LoopingProvider(loop_attempts=1)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    resp = asyncio.run(convo.asend("hi"))
    assert resp.text == "All good."
    assert p.stream_count == 2


# ---------------------------------------------------------------------------
# Aborted attempts must not be cached
# ---------------------------------------------------------------------------


def test_aborted_attempts_not_cached(tmp_path):
    p = LoopingProvider(loop_attempts=1)
    convo = _convo(p, cache_dir=tmp_path, repetition_detection=RepetitionGuard(retries=2))
    resp = convo.send("hi")
    assert resp.text == "All good."
    cache_files = list(tmp_path.rglob("*.json"))
    assert len(cache_files) == 1  # only the clean response stored
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["response"]["text"] == "All good."

    # A second identical send hits the cache (no further provider calls).
    streams_before = p.stream_count
    convo2 = _convo(p, cache_dir=tmp_path, repetition_detection=RepetitionGuard(retries=2))
    resp2 = convo2.send("hi")
    assert resp2.text == "All good."
    assert p.stream_count == streams_before  # cache hit, provider untouched


# ---------------------------------------------------------------------------
# Streaming: abort + raise, history rolled back
# ---------------------------------------------------------------------------


def test_stream_aborts_and_raises():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    seen = []
    with pytest.raises(RepetitionLoopError) as ei:
        for ev in convo.stream("hi"):
            if ev.text_delta:
                seen.append(ev.text_delta)
    assert ei.value.attempts == 1  # streaming does not transparently retry
    assert seen  # some looping deltas were yielded before the abort
    assert p.stream_count == 1
    # Convo rolled back: the user prompt + partial assistant turn are gone.
    assert convo.history == []


def test_stream_without_loop_is_unchanged():
    p = LoopingProvider(loop_attempts=0)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    out = "".join(ev.text_delta or "" for ev in convo.stream("hi"))
    assert out == "All good."
    assert convo.history[-1].role == "assistant"


def test_astream_aborts_and_raises():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))

    async def run():
        async for _ev in convo.astream("hi"):
            pass

    with pytest.raises(RepetitionLoopError):
        asyncio.run(run())
    assert convo.history == []


# ---------------------------------------------------------------------------
# Thinking / reasoning loops
# ---------------------------------------------------------------------------


def test_send_catches_thinking_loop():
    # The model loops inside its reasoning stream on attempt 0 (no visible text
    # yet); the guard must catch it mid-thinking and retry to a clean body.
    p = LoopingProvider(loop_attempts=0, thinking_loop_attempts=1)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    resp = convo.send("hi")
    assert resp.text == "All good."
    assert p.stream_count == 2  # attempt 0 looped in thinking, attempt 1 clean


def test_send_thinking_loop_exhausts():
    p = LoopingProvider(loop_attempts=0, thinking_loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    with pytest.raises(RepetitionLoopError) as ei:
        convo.send("hi")
    assert ei.value.attempts == 3
    assert "reason" in ei.value.partial_text  # the looping reasoning is captured
    assert convo.history == []  # rolled back on failure


def test_stream_aborts_on_thinking_loop():
    p = LoopingProvider(loop_attempts=0, thinking_loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    seen = []
    with pytest.raises(RepetitionLoopError):
        for ev in convo.stream("hi"):
            if ev.thinking_delta:
                seen.append(ev.thinking_delta)
    assert seen  # some looping reasoning deltas were yielded before the abort
    assert convo.history == []


def test_clean_thinking_not_flagged():
    # A short, non-repeating reasoning preamble must not false-positive.
    p = LoopingProvider(
        loop_attempts=0,
        thinking_loop_attempts=0,
        clean_thinking="Let me work through this step by step before answering. ",
    )
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    resp = convo.send("hi")
    assert resp.text == "All good."
    assert p.stream_count == 1  # no loop -> no retry


def test_asend_catches_thinking_loop():
    p = LoopingProvider(loop_attempts=0, thinking_loop_attempts=1)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    resp = asyncio.run(convo.asend("hi"))
    assert resp.text == "All good."
    assert p.stream_count == 2


# ---------------------------------------------------------------------------
# Empty stream under the guard: no empty assistant message, no cache entry
# ---------------------------------------------------------------------------


class EmptyStreamProvider(Provider):
    """Streams only a terminal done event — no blocks at all."""

    NAME = "emptystream"
    SUPPORTS = frozenset({"max_tokens", "temperature", "repeat_penalty", "dry", "tools"})

    def _init_client(self) -> None:
        self._client = object()

    def _resolve_key(self, env_var: str) -> str:
        del env_var
        return "key"

    _USAGE = Usage(prompt_tokens=5, completion_tokens=0, total_tokens=5)

    def _complete_raw(self, req: CompletionRequest) -> Response:
        raise AssertionError("guarded send must drive the stream hook")

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        raise AssertionError("guarded send must drive the stream hook")

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        yield StreamEvent(done=True, usage=self._USAGE, finish_reason="stop")

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(done=True, usage=self._USAGE, finish_reason="stop")


def test_guarded_send_empty_stream_appends_nothing(tmp_path):
    p = EmptyStreamProvider()
    convo = p.new_model("empty-model").new_conversation(
        log_dir=False, cache_dir=tmp_path, repetition_detection=RepetitionGuard(retries=0)
    )
    resp = convo.send("hi")
    assert resp.blocks == [] and resp.text == ""
    # No empty assistant message in history (the next Anthropic send would 400)
    # and no empty Response cached.
    assert [m.role for m in convo.history] == ["user"]
    assert not list(tmp_path.rglob("*.json"))


def test_guarded_asend_empty_stream_appends_nothing(tmp_path):
    p = EmptyStreamProvider()
    convo = p.new_model("empty-model").new_conversation(
        log_dir=False, cache_dir=tmp_path, repetition_detection=RepetitionGuard(retries=0)
    )
    resp = asyncio.run(convo.asend("hi"))
    assert resp.blocks == []
    assert [m.role for m in convo.history] == ["user"]
    assert not list(tmp_path.rglob("*.json"))


def test_stream_empty_stream_appends_nothing(tmp_path):
    """The public stream() path already guards via _finalize_stream; pin the
    behaviour the guarded send now mirrors."""
    p = EmptyStreamProvider()
    convo = p.new_model("empty-model").new_conversation(log_dir=False, cache_dir=tmp_path)
    events = list(convo.stream("hi"))
    assert events[-1].done is True
    assert [m.role for m in convo.history] == ["user"]
    assert not list(tmp_path.rglob("*.json"))


# ---------------------------------------------------------------------------
# Streamed tool-call argument fragments
# ---------------------------------------------------------------------------


class ToolArgsLoopingProvider(Provider):
    """Streams a tool call whose arguments loop, fragment by fragment, then a
    terminal tool_call_delta. ``reached_terminal`` records whether the consumer
    let the stream get that far — mid-stream detection must abandon the stream
    before the terminal event (i.e. before the budget is burned)."""

    NAME = "argslooping"
    SUPPORTS = frozenset({"max_tokens", "temperature", "repeat_penalty", "dry", "tools"})

    def __init__(self, *, fragment: str = '{"q": "spam spam ', reps: int = 100, **knobs):
        self.fragment = fragment
        self.reps = reps
        self.reached_terminal = False
        self.stream_count = 0
        super().__init__(**knobs)

    def _init_client(self) -> None:
        self._client = object()

    def _resolve_key(self, env_var: str) -> str:
        del env_var
        return "key"

    _USAGE = Usage(prompt_tokens=5, completion_tokens=50, total_tokens=55)

    def _events(self):
        raw = self.fragment * self.reps
        for _ in range(self.reps):
            yield StreamEvent(
                tool_args_delta=ToolArgsDelta(index=0, fragment=self.fragment, id="t1", name="f")
            )
        self.reached_terminal = True
        yield StreamEvent(tool_call_delta=ToolCall(id="t1", name="f", input={}, raw_arguments=raw))
        yield StreamEvent(done=True, usage=self._USAGE, finish_reason="tool_calls")

    def _complete_raw(self, req: CompletionRequest) -> Response:
        raise AssertionError("guarded send must drive the stream hook")

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        raise AssertionError("guarded send must drive the stream hook")

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        self.stream_count += 1
        yield from self._events()

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_count += 1
        for ev in self._events():
            yield ev


def test_detection_text_uses_fragments_for_in_flight_call():
    from llmfacade.conversation import _detection_text

    frags = {0: ['{"q": "sp', 'am spam"']}
    out = _detection_text([], [], [], frags)
    assert out == '{"q": "spam spam"'


def test_detection_text_counts_streamed_args_once():
    """When the terminal call arrives for an index that already streamed
    fragments, its args must not be added on top of the fragments."""
    from llmfacade.conversation import _detection_text

    tc = ToolCall(id="t1", name="f", input={}, raw_arguments='{"x": 1}')
    frags = {0: ['{"x"', ": 1}"]}
    assert _detection_text([], [], [tc], frags) == '{"x": 1}'


def test_detection_text_terminal_only_keeps_args():
    """Google-style streams emit no fragments; the terminal call's args are
    still scanned."""
    from llmfacade.conversation import _detection_text

    tc = ToolCall(id="t1", name="f", input={}, raw_arguments='{"x": 1}')
    assert _detection_text([], [], [tc], {}) == '{"x": 1}'
    parsed_only = ToolCall(id="t2", name="f", input={"x": 1})
    assert '"x": 1' in _detection_text([], [], [parsed_only], {})


def test_guarded_send_catches_tool_args_loop_mid_stream():
    p = ToolArgsLoopingProvider()
    convo = p.new_model("args-model").new_conversation(
        log_dir=False, repetition_detection=RepetitionGuard(retries=0)
    )
    with pytest.raises(RepetitionLoopError) as ei:
        convo.send("hi")
    assert "spam" in ei.value.partial_text
    assert p.reached_terminal is False, "loop must be caught before the terminal event"
    assert convo.history == []


def test_guarded_asend_catches_tool_args_loop_mid_stream():
    p = ToolArgsLoopingProvider()
    convo = p.new_model("args-model").new_conversation(
        log_dir=False, repetition_detection=RepetitionGuard(retries=0)
    )
    with pytest.raises(RepetitionLoopError):
        asyncio.run(convo.asend("hi"))
    assert p.reached_terminal is False
    assert convo.history == []


def test_stream_aborts_on_tool_args_loop_mid_stream():
    p = ToolArgsLoopingProvider()
    convo = p.new_model("args-model").new_conversation(
        log_dir=False, repetition_detection=RepetitionGuard(retries=2)
    )
    seen = 0
    with pytest.raises(RepetitionLoopError) as ei:
        for ev in convo.stream("hi"):
            if ev.tool_args_delta is not None:
                seen += 1
    assert seen > 0  # some fragments were yielded before the abort
    assert p.reached_terminal is False, "loop must be caught before the terminal event"
    assert "spam" in ei.value.partial_text
    assert convo.history == []


def test_astream_aborts_on_tool_args_loop_mid_stream():
    p = ToolArgsLoopingProvider()
    convo = p.new_model("args-model").new_conversation(
        log_dir=False, repetition_detection=RepetitionGuard(retries=2)
    )

    async def run():
        async for _ev in convo.astream("hi"):
            pass

    with pytest.raises(RepetitionLoopError):
        asyncio.run(run())
    assert p.reached_terminal is False
    assert convo.history == []


def test_clean_tool_args_fragments_not_flagged():
    p = ToolArgsLoopingProvider(
        fragment='{"q": "one unique question about geese migration"}', reps=1
    )
    convo = p.new_model("args-model").new_conversation(
        log_dir=False, repetition_detection=RepetitionGuard(retries=0)
    )
    resp = convo.send("hi")
    assert p.reached_terminal is True
    assert resp.tool_calls and resp.tool_calls[0].id == "t1"
    # The fragments fed detection only — history holds the terminal tool call.
    assert convo.history[-1].role == "assistant"


# ---------------------------------------------------------------------------
# Cascade resolution
# ---------------------------------------------------------------------------


def test_cascade_provider_to_per_call():
    # Provider-level default resolves onto the convo.
    p = LoopingProvider(loop_attempts=0, repetition_detection=RepetitionGuard(retries=4))
    model = p.new_model("loop-model")
    convo = model.new_conversation()
    assert convo._repetition_guard is not None
    assert convo._repetition_guard.retries == 4

    # Convo overrides provider.
    convo2 = model.new_conversation(repetition_detection=RepetitionGuard(retries=1))
    assert convo2._repetition_guard is not None
    assert convo2._repetition_guard.retries == 1

    # Convo disables with False despite the provider default.
    convo3 = model.new_conversation(repetition_detection=False)
    assert convo3._repetition_guard is None


def test_resolve_helper_none_disables():
    p = LoopingProvider(loop_attempts=0)
    model = p.new_model("loop-model")
    assert resolve_repetition_guard(convo_repetition=None, model=model) is None


def test_per_call_override_enables_on_unguarded_convo():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p)  # no convo-level guard
    assert convo._repetition_guard is None
    with pytest.raises(RepetitionLoopError):
        convo.send("hi", repetition_detection=RepetitionGuard(retries=0))
    # retries=0 -> single attempt, immediate raise
    assert p.stream_count == 1


def test_per_call_false_disables_convo_guard():
    p = LoopingProvider(loop_attempts=99)
    convo = _convo(p, repetition_detection=RepetitionGuard(retries=2))
    # Per-call False turns the guard off -> the looping body streams via
    # _complete_raw with no detection and returns normally.
    resp = convo.send("hi", repetition_detection=False)
    assert resp.text == "All good."  # _complete_raw path, clean canned text
    assert p.stream_count == 0
    assert p.complete_count == 1


def test_int_shorthand_enables_guard():
    p = LoopingProvider(loop_attempts=0)
    convo = _convo(p, repetition_detection=3)
    assert convo._repetition_guard is not None
    assert convo._repetition_guard.min_reps_floor == 3


# ---------------------------------------------------------------------------
# _DetectionTail: incremental tail == full _detection_text re-join
# ---------------------------------------------------------------------------


def _full_join(buf: _StreamBuffers) -> str:
    return _detection_text(buf.thinking_text, buf.text, buf.tool_calls, buf.tool_args)


def _assert_tail_matches(buf: _StreamBuffers, tail: _DetectionTail, cap: int) -> None:
    full = _full_join(buf)
    assert tail.text() == full[-cap:]


_LOOP_PHRASES = [
    "I cannot help with that. ",
    "spam ",
    "abcabc",
    '{"q": "again"}',
]


def _random_events(rng: random.Random) -> list[StreamEvent]:
    """A random-ish interleaving of text / thinking / tool-args / terminal
    tool-call events, with occasional degenerate loops in each channel."""
    events: list[StreamEvent] = []
    n_calls = 0
    fluent = ["The sky is blue. ", "Counting: one two three. ", "ok\n", "- item\n"]
    for _ in range(rng.randrange(5, 40)):
        kind = rng.randrange(8)
        if kind in (0, 1):
            chunk = rng.choice(fluent + [rng.choice(_LOOP_PHRASES) * rng.randrange(1, 15)])
            events.append(StreamEvent(text_delta=chunk))
        elif kind in (2, 3):
            chunk = rng.choice(fluent + [rng.choice(_LOOP_PHRASES) * rng.randrange(1, 15)])
            events.append(StreamEvent(thinking_delta=chunk))
        elif kind in (4, 5):
            # Fragments for the in-flight call (index == n_calls), sometimes
            # looping, sometimes empty (key presence still counts).
            frag = rng.choice(['{"city": "Par', 'is"}', "", rng.choice(_LOOP_PHRASES) * 6])
            events.append(StreamEvent(tool_args_delta=ToolArgsDelta(index=n_calls, fragment=frag)))
        elif kind == 6:
            # Terminal call that streamed no fragments (Google-style): parsed
            # input or a raw_arguments string.
            if rng.random() < 0.5:
                call = ToolCall(id=f"t{n_calls}", name="fn", input={"x": rng.randrange(100)})
            else:
                call = ToolCall(id=f"t{n_calls}", name="fn", input={}, raw_arguments='{"broken": ')
            events.append(StreamEvent(tool_call_delta=call))
            n_calls += 1
        else:
            # Terminal call closing an (possibly) in-flight fragment stream.
            call = ToolCall(id=f"t{n_calls}", name="fn", input={"city": "Paris"})
            events.append(StreamEvent(tool_call_delta=call))
            n_calls += 1
    return events


@pytest.mark.parametrize("seed", range(12))
def test_detection_tail_matches_full_join_property(seed):
    rng = random.Random(seed)
    cap = rng.choice([16, 64, 256, 4096])
    guard = RepetitionGuard(tail_chars=cap)
    buf = _StreamBuffers()
    tail = _DetectionTail(cap)
    for ev in _random_events(rng):
        buf.absorb(ev)
        tail.absorb(ev)
        # Invariant at every step: the incremental tail equals the last
        # tail_chars of the full re-join, so the detector sees identical input.
        _assert_tail_matches(buf, tail, cap)
        expected = detect_repetition_loop(
            _full_join(buf),
            tail_chars=guard.tail_chars,
            max_period=guard.max_period,
            min_reps_floor=guard.min_reps_floor,
        )
        assert _detect_in_tail(tail, guard) == expected


def _absorb_all(events) -> tuple[_StreamBuffers, _DetectionTail, RepetitionGuard]:
    guard = RepetitionGuard()
    buf = _StreamBuffers()
    tail = _DetectionTail(guard.tail_chars)
    for ev in events:
        buf.absorb(ev)
        tail.absorb(ev)
    return buf, tail, guard


def test_detection_tail_catches_thinking_loop():
    events = [StreamEvent(thinking_delta="I should reconsider. ")] * 30
    buf, tail, guard = _absorb_all(events)
    _assert_tail_matches(buf, tail, guard.tail_chars)
    assert _detect_in_tail(tail, guard) is not None


def test_detection_tail_catches_tool_args_loop():
    events = [StreamEvent(tool_args_delta=ToolArgsDelta(index=0, fragment='{"q": "again", '))] * 30
    buf, tail, guard = _absorb_all(events)
    _assert_tail_matches(buf, tail, guard.tail_chars)
    assert _detect_in_tail(tail, guard) is not None


def test_detection_tail_loop_outside_window_not_seen():
    # A loop entirely before the tail window must not fire in either impl.
    cap = 40
    guard = RepetitionGuard(tail_chars=cap)
    events = [StreamEvent(text_delta="spam " * 6)] + [
        StreamEvent(text_delta="X clean unique trailing content that does not repeat. ")
    ] * 3
    buf = _StreamBuffers()
    tail = _DetectionTail(cap)
    for ev in events:
        buf.absorb(ev)
        tail.absorb(ev)
    _assert_tail_matches(buf, tail, cap)
    assert _detect_in_tail(tail, guard) is None


def test_detection_tail_terminal_call_without_fragments_uses_args():
    # No fragments + unparsed args: the raw_arguments string is scanned. The
    # loop sits at the very end, so the suffix-anchored detector fires.
    call = ToolCall(id="t0", name="fn", input={}, raw_arguments='{"q": ' + "again " * 30)
    buf, tail, guard = _absorb_all([StreamEvent(tool_call_delta=call)])
    _assert_tail_matches(buf, tail, guard.tail_chars)
    assert _detect_in_tail(tail, guard) is not None


def test_detection_tail_terminal_call_parsed_input_is_scanned():
    # Google-style: no fragments, parsed input only — the json.dumps of the
    # input lands in the scanned buffer (tail mirrors the full join exactly;
    # the trailing '"}' breaks suffix periodicity, so neither impl fires).
    call = ToolCall(id="t0", name="fn", input={"phrase": "again and again " * 10})
    buf, tail, guard = _absorb_all([StreamEvent(tool_call_delta=call)])
    _assert_tail_matches(buf, tail, guard.tail_chars)
    assert "again and again " in tail.text()


def test_detection_tail_fragments_not_double_counted_with_terminal():
    # A call whose fragments streamed uses the fragments INSTEAD OF the
    # terminal call's arguments — same string, counted once.
    frag = '{"city": "Paris"}'
    events = [
        StreamEvent(tool_args_delta=ToolArgsDelta(index=0, fragment=frag)),
        StreamEvent(tool_call_delta=ToolCall(id="t0", name="fn", input={"city": "Paris"})),
    ]
    buf, tail, guard = _absorb_all(events)
    full = _full_join(buf)
    assert full.count(frag) == 1
    _assert_tail_matches(buf, tail, guard.tail_chars)
