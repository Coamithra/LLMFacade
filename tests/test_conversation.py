"""Conversation lifecycle, history, snapshot/rollback, clone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmfacade import (
    ConversationStateError,
    SystemBlock,
    ToolIterationLimitError,
    helpers,
    tool,
)
from llmfacade.models import ToolCall, ToolUseBlock

from .conftest import MockProvider


def test_send_appends_user_and_assistant(started_convo):
    resp = started_convo.send("hello")
    assert resp.text == "ok"
    history = started_convo.history
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "assistant"


def test_send_no_arg_uses_existing_history(mock_model):
    convo = mock_model.new_conversation()
    convo.add_user_message("seeded")
    resp = convo.send()
    assert resp.text == "ok"
    assert len(convo.history) == 2


def test_add_assistant_message_replay(started_convo):
    started_convo.add_user_message("hi")
    started_convo.add_assistant_message("hello adventurer")
    history = started_convo.history
    assert history[-1].role == "assistant"
    assert history[-1].content == "hello adventurer"


def test_system_blocks_at_construction(mock_model):
    convo = mock_model.new_conversation(system_blocks=["you are X", "be brief"])
    assert len(convo.system_blocks) == 2
    assert convo.system_blocks[0].text == "you are X"
    assert convo.system_blocks[0].cache is False


def test_snapshot_rollback_restores_history(started_convo):
    started_convo.add_user_message("first")
    snap = started_convo.snapshot()
    started_convo.add_user_message("second")
    assert len(started_convo.history) == 2
    started_convo.rollback(snap)
    assert len(started_convo.history) == 1
    assert started_convo.history[0].content == "first"


def test_clone_isolates_history(mock_model):
    convo = mock_model.new_conversation(system_blocks=["system"])
    convo.add_user_message("hi")
    clone = convo.clone()
    clone.add_user_message("only-in-clone")
    assert len(convo.history) == 1
    assert len(clone.history) == 2


def test_per_call_overrides_pass_to_provider(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    convo.send("x", max_tokens=999, temperature=0.3)
    last = p.calls[-1].req
    assert last.settings["max_tokens"] == 999
    assert last.settings["temperature"] == 0.3


def test_send_records_tool_calls_in_history():
    p = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 1})],
    )
    model = p.new_model("mock-model")
    convo = model.new_conversation()
    resp = convo.send("go")
    assert len(resp.tool_calls) == 1
    last = convo.history[-1]
    assert last.role == "assistant"
    assert isinstance(last.content, list)


def test_send_with_dangling_tool_use_raises(started_convo):
    started_convo.add_user_message("hi")
    started_convo.add_assistant_message([ToolUseBlock(id="abc", name="echo", input={"x": 1})])
    with pytest.raises(ConversationStateError):
        started_convo.send("again")


def test_helpers_run_bound_tools_dispatches_and_continues():
    @tool
    def echo(x: int) -> int:
        """Echo x."""
        return x

    p = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 7})],
    )
    convo = p.new_model("mock-model").new_conversation(tools=[echo])
    resp = convo.send("go")
    results = helpers.run_bound_tools(convo, resp)
    assert len(results) == 1
    assert results[0].content == "7"
    last = convo.history[-1]
    assert last.role == "tool"


def test_helpers_run_to_completion_caps_iterations():
    @tool
    def echo(x: int) -> int:
        """Echo x."""
        return x

    p = MockProvider(
        canned_text="",
        canned_tool_calls=[ToolCall(id="t1", name="echo", input={"x": 1})],
    )
    convo = p.new_model("mock-model").new_conversation(tools=[echo])
    with pytest.raises(ToolIterationLimitError):
        helpers.run_to_completion(convo, "go", max_iterations=3)


def _records_of(log_path: Path, kind: str) -> list[dict]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("type") == kind
    ]


def _request_records(log_path: Path) -> list[dict]:
    return _records_of(log_path, "request")


def _response_records(log_path: Path) -> list[dict]:
    return _records_of(log_path, "response")


def _settings_records(log_path: Path) -> list[dict]:
    return _records_of(log_path, "settings")


def test_log_starts_with_settings_header(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log, temperature=0.5)
    [header] = _settings_records(log)
    assert header["convo"] == "t"
    assert header["provider"] == "mock"
    assert header["model"] == "mock-model"
    assert header["settings"]["temperature"] == {"value": 0.5, "source": "convo"}
    convo.send("hi")
    # Subsequent request entries should NOT repeat the full settings.
    [req] = _request_records(log)
    assert "settings" not in req


def test_log_first_turn_has_no_prior(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    convo.send("first")

    [req] = _request_records(log)
    assert "prior_history" not in req
    assert req["new_messages"] == [{"role": "user", "content": "first"}]


def test_log_subsequent_turns_only_log_delta(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    convo.send("first")
    convo.send("second")
    convo.send("third")

    reqs = _request_records(log)
    assert [r["new_messages"][0]["content"] for r in reqs] == ["first", "second", "third"]
    assert "prior_history" not in reqs[0]
    assert reqs[1]["prior_history"]["messages"] == 2
    assert reqs[2]["prior_history"]["messages"] == 4


def test_log_size_bounded_per_turn(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    for i in range(30):
        convo.send(f"msg-{i}")

    sizes = [
        len(line.encode("utf-8"))
        for line in log.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("type") == "request"
    ]
    # Quadratic logging would have the last turn at >> 30x the first.
    # Delta + bounded preview keeps per-turn size flat after the preview saturates.
    assert max(sizes) < 4 * sizes[0] + 1024


def test_log_response_advances_msg_count(mock_model, tmp_path):
    """The assistant message appended after _log_response should NOT re-appear
    as 'new' on the next request."""
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    convo.send("a")
    convo.send("b")

    reqs = _request_records(log)
    assert reqs[1]["new_messages"] == [{"role": "user", "content": "b"}]


def test_clone_inherits_history_as_prior(mock_model, tmp_path):
    parent = mock_model.new_conversation(name="parent")
    parent.send("a")
    parent.send("b")

    log = tmp_path / "clone.jsonl"
    clone = parent.clone(log_path=log)
    clone.send("c")

    [req] = _request_records(log)
    # Inherited [u_a, asst_a, u_b, asst_b] should land in prior_history,
    # not be redumped as new_messages.
    assert req["new_messages"] == [{"role": "user", "content": "c"}]
    assert req["prior_history"]["messages"] == 4


def test_rollback_keeps_logged_count_consistent(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    convo.send("a")
    convo.send("b")
    snap = convo.snapshot()
    convo.send("c")
    convo.rollback(snap)
    convo.send("d")

    last = _request_records(log)[-1]
    assert last["new_messages"] == [{"role": "user", "content": "d"}]
    # post-rollback prior is the snapshotted [u_a, asst_a, u_b, asst_b] = 4.
    assert last["prior_history"]["messages"] == 4


def test_log_max_message_lines_abbreviates_long_user_message(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log, log_max_message_lines=10)
    novel = "\n".join(f"line-{i}" for i in range(100))
    convo.send(novel)

    [req] = _request_records(log)
    content = req["new_messages"][0]["content"]
    assert content.startswith("line-0\nline-1\nline-2\nline-3\nline-4\n")
    assert content.endswith("\nline-95\nline-96\nline-97\nline-98\nline-99")
    assert "[90 lines skipped]" in content


def test_log_max_message_lines_unset_logs_full_text(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    novel = "\n".join(f"line-{i}" for i in range(50))
    convo.send(novel)

    [req] = _request_records(log)
    assert req["new_messages"][0]["content"] == novel


def test_log_max_message_lines_does_not_truncate_short_messages(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log, log_max_message_lines=10)
    convo.send("short message")

    [req] = _request_records(log)
    assert req["new_messages"][0]["content"] == "short message"


def test_cache_summary_present_when_usage_has_cache_read(tmp_path):
    from llmfacade.models import Usage

    p = MockProvider(
        canned_usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cache_creation_tokens=0,
            cache_read_tokens=200,
        )
    )
    log = tmp_path / "log.jsonl"
    convo = p.new_model("mock-model").new_conversation(name="t", log_path=log)
    convo.send("hello there how are you doing today")

    [resp] = _response_records(log)
    cs = resp["cache_summary"]
    assert cs["cache_read_tokens"] == 200
    assert cs["hit_ratio"] > 0
    assert cs["tokenizer"] == "chars/4"
    assert "Caching is working" in cs["_note"]


def test_cache_summary_boundary_estimate(tmp_path):
    """approximate_messages_cached should track the cache_read_tokens count
    via the chars/4 estimate."""
    from llmfacade.models import Usage

    big = "x" * 320  # ~80 tokens via chars/4
    p = MockProvider(
        canned_text="y",
        canned_usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cache_creation_tokens=0,
            cache_read_tokens=165,
        ),
    )
    log = tmp_path / "log.jsonl"
    convo = p.new_model("mock-model").new_conversation(name="t", log_path=log)
    convo.send(big)
    convo.send(big)
    convo.send(big)

    last_resp = _response_records(log)[-1]
    boundary = last_resp["cache_summary"]["approximate_messages_cached"]
    assert 2 <= boundary <= 4


def test_cache_summary_zero_cache_no_markers_nags_user(tmp_path):
    """When the provider supports explicit caching but the user hasn't
    enabled auto_cache_last_user, the diagnostic should call that out."""
    from llmfacade.models import Usage

    p = MockProvider(
        canned_usage=Usage(
            prompt_tokens=2000,
            completion_tokens=5,
            total_tokens=2005,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
    )
    log = tmp_path / "log.jsonl"
    convo = p.new_model("mock-model").new_conversation(name="t", log_path=log)
    convo.send("hi")

    [resp] = _response_records(log)
    note = resp["cache_summary"]["_note"]
    assert "auto_cache_last_user" in note


def test_anthropic_cache_ttl_emits_in_request_body():
    """cache_ttl=1h should land in the cache_control body sent to the
    Anthropic SDK."""
    from llmfacade.providers.anthropic import AnthropicProvider
    from llmfacade.settings import EphemeralCacheTTL

    p = object.__new__(AnthropicProvider)
    blocks = p._system_to_api(
        [SystemBlock(text="you are a bot", cache=True)],
        ttl=EphemeralCacheTTL.ONE_HOUR.value,
    )
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    blocks_default = p._system_to_api([SystemBlock(text="you are a bot", cache=True)], ttl=None)
    assert blocks_default[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_cache_ttl_via_convo_setting():
    """End-to-end: cache_ttl plumbs into the wire body."""
    from llmfacade.provider import CompletionRequest
    from llmfacade.providers.anthropic import AnthropicProvider
    from llmfacade.settings import EphemeralCacheTTL

    p = object.__new__(AnthropicProvider)
    req = CompletionRequest(
        model="claude-sonnet-4-6",
        messages=[],
        system_blocks=[SystemBlock(text="hi", cache=True)],
        tools=[],
        stop=None,
        settings={
            "auto_cache_last_user": False,
            "cache_ttl": EphemeralCacheTTL.ONE_HOUR,
            "max_tokens": 1024,
        },
    )
    api_kwargs = p._build_kwargs(req)
    assert api_kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
