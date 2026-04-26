"""Conversation lifecycle, history, snapshot/rollback, clone."""

from __future__ import annotations

import pytest

from llmfacade import ToolIterationLimitError, tool
from llmfacade.models import ToolCall

from .conftest import MockProvider


def test_complete_appends_user_and_assistant(started_convo):
    resp = started_convo.Complete("hello")
    assert resp.text == "ok"
    history = started_convo.history
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "assistant"


def test_complete_no_arg_uses_existing_history(mock_model):
    convo = mock_model.NewConversation()
    convo.Start()
    convo.AddUserMessage("seeded")
    resp = convo.Complete()
    assert resp.text == "ok"
    assert len(convo.history) == 2


def test_add_assistant_message_replay(started_convo):
    started_convo.AddUserMessage("hi")
    started_convo.AddAssistantMessage("hello adventurer")
    history = started_convo.history
    assert history[-1].role == "assistant"
    assert history[-1].content == "hello adventurer"


def test_add_system_after_start_raises(mock_model):
    convo = mock_model.NewConversation()
    convo.AddSystemMessage("you are X")
    convo.Start()
    from llmfacade import SettingsLockedError

    with pytest.raises(SettingsLockedError):
        convo.AddSystemMessage("more")


def test_snapshot_rollback_restores_history(started_convo):
    started_convo.AddUserMessage("first")
    snap = started_convo.Snapshot()
    started_convo.AddUserMessage("second")
    assert len(started_convo.history) == 2
    started_convo.Rollback(snap)
    assert len(started_convo.history) == 1
    assert started_convo.history[0].content == "first"


def test_clone_isolates_history(mock_model):
    convo = mock_model.NewConversation()
    convo.AddSystemMessage("system")
    convo.Start()
    convo.AddUserMessage("hi")
    clone = convo.Clone()
    assert not clone.started
    clone.Start()
    clone.AddUserMessage("only-in-clone")
    assert len(convo.history) == 1
    assert len(clone.history) == 2


def test_per_call_overrides_pass_to_provider(mock_model):
    p: MockProvider = mock_model.provider  # type: ignore[assignment]
    convo = mock_model.NewConversation()
    convo.Start()
    convo.Complete("x", max_tokens=999, temperature=0.3)
    last = p.calls[-1].kwargs
    assert last["max_tokens"] == 999
    assert last["temperature"] == 0.3


def test_complete_records_tool_calls_in_history():
    p = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 1})],
    )
    model = p.NewModel("mock-model")
    convo = model.NewConversation()
    convo.Start()
    # auto_tools=False so we don't try to dispatch (no tool registered)
    resp = convo.Complete("go", auto_tools=False)
    assert len(resp.tool_calls) == 1
    last = convo.history[-1]
    assert last.role == "assistant"
    assert isinstance(last.content, list)


def test_max_tool_iterations_raises():
    @tool
    def echo(x: int) -> int:
        """Echo x."""
        return x

    p = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 1})],
    )
    convo = p.NewModel("mock-model").NewConversation()
    convo.AddTool(echo)
    convo.Start()
    with pytest.raises(ToolIterationLimitError):
        convo.Complete("go", max_tool_iterations=3)
