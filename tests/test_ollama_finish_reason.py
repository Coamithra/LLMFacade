"""Ollama surfaces the API's done_reason on Response.finish_reason and on
the final StreamEvent.finish_reason so callers can distinguish 'stop' from
'length' / 'load' / 'unload'.
"""

from __future__ import annotations

from llmfacade import Message
from llmfacade.provider import CompletionRequest
from llmfacade.providers.ollama import OllamaProvider


class _FakeMsg:
    def __init__(self, content: str = "ok"):
        self.content = content
        self.tool_calls: list[object] = []


class _FakeNonStreamResponse:
    def __init__(self, *, done_reason: str | None):
        self.message = _FakeMsg("hello")
        self.model = "llama3"
        self.prompt_eval_count = 5
        self.eval_count = 2
        if done_reason is not None:
            self.done_reason = done_reason


class _FakeChunk:
    def __init__(
        self,
        *,
        content: str = "",
        done: bool = False,
        done_reason: str | None = None,
        tool_calls: list[object] | None = None,
    ):
        self.message = _FakeMsg(content)
        self.message.tool_calls = tool_calls or []
        self.done = done
        if done_reason is not None:
            self.done_reason = done_reason
        # On final chunks Ollama also sends the eval counts.
        self.prompt_eval_count = 5
        self.eval_count = 2


def _make_req() -> CompletionRequest:
    return CompletionRequest(
        model="llama3",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"max_tokens": 16},
        settings_source={"max_tokens": "convo"},
    )


# --- non-streaming path ------------------------------------------------------


def test_parse_response_uses_done_reason_length(monkeypatch):
    p = OllamaProvider()
    monkeypatch.setattr(
        p._client, "chat", lambda **_kw: _FakeNonStreamResponse(done_reason="length")
    )
    resp = p._complete_raw(_make_req())
    assert resp.finish_reason == "length"


def test_parse_response_uses_done_reason_stop(monkeypatch):
    p = OllamaProvider()
    monkeypatch.setattr(
        p._client, "chat", lambda **_kw: _FakeNonStreamResponse(done_reason="stop")
    )
    resp = p._complete_raw(_make_req())
    assert resp.finish_reason == "stop"


def test_parse_response_defaults_when_done_reason_missing(monkeypatch):
    """Older Ollama versions or unusual responses may omit done_reason — default
    to 'stop' to preserve the prior contract."""
    p = OllamaProvider()
    monkeypatch.setattr(p._client, "chat", lambda **_kw: _FakeNonStreamResponse(done_reason=None))
    resp = p._complete_raw(_make_req())
    assert resp.finish_reason == "stop"


# --- streaming path ----------------------------------------------------------


def test_stream_carries_done_reason_length(monkeypatch):
    p = OllamaProvider()
    chunks = [
        _FakeChunk(content="hello"),
        _FakeChunk(done=True, done_reason="length"),
    ]
    monkeypatch.setattr(p._client, "chat", lambda **_kw: iter(chunks))
    events = list(p._stream_raw(_make_req()))
    final = [e for e in events if e.done]
    assert len(final) == 1
    assert final[0].finish_reason == "length"


def test_stream_finish_reason_defaults_to_stop(monkeypatch):
    p = OllamaProvider()
    chunks = [
        _FakeChunk(content="hello"),
        _FakeChunk(done=True),  # no done_reason attr
    ]
    monkeypatch.setattr(p._client, "chat", lambda **_kw: iter(chunks))
    events = list(p._stream_raw(_make_req()))
    final = [e for e in events if e.done]
    assert len(final) == 1
    assert final[0].finish_reason == "stop"
