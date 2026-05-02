"""Async mirrors: asend and astream."""

from __future__ import annotations

import asyncio


def test_asend(mock_model):
    convo = mock_model.new_conversation()

    async def run():
        return await convo.asend("hi")

    resp = asyncio.run(run())
    assert resp.text == "ok"
    assert len(convo.history) == 2


def test_astream(mock_model):
    convo = mock_model.new_conversation()

    async def run():
        chunks = []
        async for ev in convo.astream("multi word response"):
            if ev.text_delta:
                chunks.append(ev.text_delta)
        return chunks

    chunks = asyncio.run(run())
    assert "".join(chunks).strip() == "ok"


def test_sync_stream(mock_model):
    convo = mock_model.new_conversation()
    chunks = []
    for ev in convo.stream("hi"):
        if ev.text_delta:
            chunks.append(ev.text_delta)
        if ev.done:
            assert ev.usage is not None
    assert "".join(chunks).strip() == "ok"


def test_stream_early_break_persists_partial_assistant_turn(mock_model):
    """Breaking out of stream() early must still record the partial assistant
    reply so history doesn't end on a dangling user message."""
    mock_model.provider.canned_text = "one two three four five"
    convo = mock_model.new_conversation()

    chunks: list[str] = []
    for ev in convo.stream("hi"):
        if ev.text_delta:
            chunks.append(ev.text_delta)
            if len(chunks) >= 2:
                break

    assert len(convo.history) == 2
    assert convo.history[0].role == "user"
    assert convo.history[1].role == "assistant"
    # Partial text from the first two deltas, not the full canned response.
    assistant_text = convo.history[1].content[0].text
    assert assistant_text == "".join(chunks)
    assert "five" not in assistant_text


def test_stream_close_persists_partial_assistant_turn(mock_model):
    """Explicitly closing the generator mid-stream must still flush partial
    state into history (covers GeneratorExit path)."""
    mock_model.provider.canned_text = "one two three four five"
    convo = mock_model.new_conversation()

    gen = convo.stream("hi")
    seen = 0
    for ev in gen:
        if ev.text_delta:
            seen += 1
            if seen >= 1:
                gen.close()
                break

    assert len(convo.history) == 2
    assert convo.history[1].role == "assistant"


def test_stream_exception_persists_partial_assistant_turn(mock_model):
    """An exception raised by the consumer mid-stream must still leave history
    in a coherent state (user + partial assistant)."""
    mock_model.provider.canned_text = "alpha beta gamma delta"
    convo = mock_model.new_conversation()

    class Boom(Exception):
        pass

    try:
        for ev in convo.stream("hi"):
            if ev.text_delta:
                raise Boom
    except Boom:
        pass

    assert len(convo.history) == 2
    assert convo.history[1].role == "assistant"


def test_astream_early_break_persists_partial_assistant_turn(mock_model):
    mock_model.provider.canned_text = "one two three four five"
    convo = mock_model.new_conversation()

    async def run():
        chunks: list[str] = []
        async for ev in convo.astream("hi"):
            if ev.text_delta:
                chunks.append(ev.text_delta)
                if len(chunks) >= 2:
                    break
        return chunks

    chunks = asyncio.run(run())

    assert len(convo.history) == 2
    assert convo.history[1].role == "assistant"
    assistant_text = convo.history[1].content[0].text
    assert assistant_text == "".join(chunks)
    assert "five" not in assistant_text
