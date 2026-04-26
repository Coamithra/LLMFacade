"""Async mirrors: aComplete and aStream."""

from __future__ import annotations

import asyncio


def test_acomplete(mock_model):
    convo = mock_model.NewConversation()
    convo.Start()

    async def run():
        return await convo.aComplete("hi")

    resp = asyncio.run(run())
    assert resp.text == "ok"
    assert len(convo.history) == 2


def test_astream(mock_model):
    convo = mock_model.NewConversation()
    convo.Start()

    async def run():
        chunks = []
        async for ev in convo.aStream("multi word response"):
            if ev.text_delta:
                chunks.append(ev.text_delta)
        return chunks

    chunks = asyncio.run(run())
    assert "".join(chunks).strip() == "ok"  # MockProvider canned text


def test_sync_stream(mock_model):
    convo = mock_model.NewConversation()
    convo.Start()
    chunks = []
    for ev in convo.Stream("hi"):
        if ev.text_delta:
            chunks.append(ev.text_delta)
        if ev.done:
            assert ev.usage is not None
    assert "".join(chunks).strip() == "ok"
