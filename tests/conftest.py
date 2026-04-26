"""Test fixtures and a MockProvider that records calls and returns canned responses."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from llmfacade.models import (
    ContentBlock,
    Response,
    StreamEvent,
    TextBlock,
    ToolCall,
    ToolUseBlock,
    Usage,
)
from llmfacade.provider import Provider
from llmfacade.settings import (
    AnySetting,
    ConvoSettings,
    ProviderSettings,
    Settings,
)


@dataclass
class MockCall:
    kwargs: dict[str, Any]


@dataclass
class MockProvider(Provider):
    """Test double. Capabilities are configurable per-instance."""

    NAME = "mock"
    SUPPORTS: frozenset[AnySetting] = frozenset(
        {
            ProviderSettings.BaseURL,
            Settings.ContextSize,
            Settings.DefaultMaxTokens,
            Settings.DefaultTemperature,
            Settings.TopP,
            Settings.Thinking,
            ConvoSettings.AutoCacheLastUser,
            ConvoSettings.UserMetadata,
        }
    )

    canned_text: str = "ok"
    canned_tool_calls: list[ToolCall] = field(default_factory=list)
    canned_thinking: str | None = None
    canned_usage: Usage | None = None
    calls: list[MockCall] = field(default_factory=list)

    def __init__(
        self,
        *,
        manager=None,
        api_key=None,
        base_url=None,
        canned_text="ok",
        canned_tool_calls=None,
        canned_thinking=None,
        canned_usage=None,
    ):
        self.canned_text = canned_text
        self.canned_tool_calls = canned_tool_calls or []
        self.canned_thinking = canned_thinking
        self.canned_usage = canned_usage or Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        self.calls = []
        super().__init__(manager=manager, api_key=api_key, base_url=base_url)

    def _init_client(self) -> None:
        self._client = object()

    def _resolve_key(self, env_var: str) -> str:
        return "mock-key"

    def _make_response(self) -> Response:
        blocks: list[ContentBlock] = []
        if self.canned_text:
            blocks.append(TextBlock(self.canned_text))
        for tc in self.canned_tool_calls:
            blocks.append(ToolUseBlock(id=tc.id, name=tc.name, input=tc.input))
        return Response(
            text=self.canned_text,
            blocks=blocks,
            tool_calls=list(self.canned_tool_calls),
            thinking=self.canned_thinking,
            usage=self.canned_usage,
            finish_reason="end_turn",
            model="mock-model",
            raw=None,
        )

    def _complete_raw(self, **kwargs: Any) -> Response:
        self.calls.append(MockCall(kwargs=dict(kwargs)))
        return self._make_response()

    async def _acomplete_raw(self, **kwargs: Any) -> Response:
        self.calls.append(MockCall(kwargs=dict(kwargs)))
        return self._make_response()

    def _stream_raw(self, **kwargs: Any) -> Iterator[StreamEvent]:
        self.calls.append(MockCall(kwargs=dict(kwargs)))
        for chunk in self.canned_text.split():
            yield StreamEvent(text_delta=chunk + " ")
        for tc in self.canned_tool_calls:
            yield StreamEvent(tool_call_delta=tc)
        yield StreamEvent(done=True, usage=self.canned_usage)

    async def _astream_raw(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.calls.append(MockCall(kwargs=dict(kwargs)))
        for chunk in self.canned_text.split():
            yield StreamEvent(text_delta=chunk + " ")
        for tc in self.canned_tool_calls:
            yield StreamEvent(tool_call_delta=tc)
        yield StreamEvent(done=True, usage=self.canned_usage)


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def mock_model(mock_provider: MockProvider):
    return mock_provider.NewModel("mock-model")


@pytest.fixture
def started_convo(mock_model):
    convo = mock_model.NewConversation()
    convo.Start()
    return convo
