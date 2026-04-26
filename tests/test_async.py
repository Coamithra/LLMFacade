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
