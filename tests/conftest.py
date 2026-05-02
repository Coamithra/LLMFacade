"""Test fixtures and a MockProvider that records calls and returns canned responses."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest

from llmfacade.facade import LLM
from llmfacade.models import (
    ContentBlock,
    Response,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from llmfacade.provider import CompletionRequest, Provider


@dataclass
class MockCall:
    req: CompletionRequest


class MockProvider(Provider):
    """Test double. Capabilities are configurable per-instance."""

    NAME = "mock"
    SUPPORTS: frozenset[str] = frozenset(
        {
            "context_size",
            "max_tokens",
            "temperature",
            "top_p",
            "thinking",
            "auto_cache_last_user",
            "user_metadata",
            "tools",
            "tool_choice",
        }
    )

    def __init__(
        self,
        *,
        manager=None,
        api_key=None,
        base_url=None,
        canned_text="ok",
        canned_tool_calls=None,
        canned_thinking=None,
        canned_thinking_blocks=None,
        canned_usage=None,
        **knobs,
    ):
        self.canned_text = canned_text
        self.canned_tool_calls = canned_tool_calls or []
        self.canned_thinking = canned_thinking
        self.canned_thinking_blocks: list[ThinkingBlock] = list(canned_thinking_blocks or [])
        self.canned_usage = canned_usage or Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        self.calls: list[MockCall] = []
        super().__init__(manager=manager, api_key=api_key, base_url=base_url, **knobs)

    def _init_client(self) -> None:
        self._client = object()

    def _resolve_key(self, env_var: str) -> str:
        del env_var
        return "mock-key"

    def _make_response(self) -> Response:
        blocks: list[ContentBlock] = list(self.canned_thinking_blocks)
        if self.canned_text:
            blocks.append(TextBlock(self.canned_text))
        for tc in self.canned_tool_calls:
            blocks.append(ToolUseBlock(id=tc.id, name=tc.name, input=tc.input))
        derived_thinking = (
            "".join(b.text for b in self.canned_thinking_blocks if not b.encrypted)
            or self.canned_thinking
        )
        return Response(
            text=self.canned_text,
            blocks=blocks,
            tool_calls=list(self.canned_tool_calls),
            thinking=derived_thinking,
            usage=self.canned_usage,
            finish_reason="end_turn",
            model="mock-model",
            raw=None,
        )

    def _complete_raw(self, req: CompletionRequest) -> Response:
        self.calls.append(MockCall(req=req))
        return self._make_response()

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        self.calls.append(MockCall(req=req))
        return self._make_response()

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        self.calls.append(MockCall(req=req))
        for tb in self.canned_thinking_blocks:
            if tb.text:
                yield StreamEvent(thinking_delta=tb.text)
            yield StreamEvent(thinking_block=tb)
        for chunk in self.canned_text.split():
            yield StreamEvent(text_delta=chunk + " ")
        for tc in self.canned_tool_calls:
            yield StreamEvent(tool_call_delta=tc)
        yield StreamEvent(done=True, usage=self.canned_usage)

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.calls.append(MockCall(req=req))
        for tb in self.canned_thinking_blocks:
            if tb.text:
                yield StreamEvent(thinking_delta=tb.text)
            yield StreamEvent(thinking_block=tb)
        for chunk in self.canned_text.split():
            yield StreamEvent(text_delta=chunk + " ")
        for tc in self.canned_tool_calls:
            yield StreamEvent(tool_call_delta=tc)
        yield StreamEvent(done=True, usage=self.canned_usage)


@pytest.fixture(autouse=True)
def _reset_llm_default():
    """Drop LLM.default() between tests so api_keys mutations don't leak.

    Pre-seed the default with ``log_dir=False`` so any test that touches
    ``LLM.default()`` doesn't materialise a log directory on disk."""
    LLM.reset_default()
    LLM._default = LLM(log_dir=False)
    yield
    LLM.reset_default()


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def mock_model(mock_provider: MockProvider):
    return mock_provider.new_model("mock-model")


@pytest.fixture
def started_convo(mock_model):
    return mock_model.new_conversation()
