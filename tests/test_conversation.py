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


def test_settings_header_includes_provider_log_metadata(tmp_path):
    """Provider.log_metadata() returns extras that get merged into the
    settings header as siblings of `settings`. Lets the llamacpp provider
    surface a `fit_estimate` block without conversation.py needing to know
    about fit semantics."""

    class _ExtraProvider(MockProvider):
        def log_metadata(self, *, model_id):
            del model_id
            return {"fit_estimate": {"context_size": 4096, "n_gpu_layers": 32}}

    p = _ExtraProvider()
    log = tmp_path / "log.jsonl"
    p.new_model("mock-model").new_conversation(name="t", log_path=log)
    [header] = _settings_records(log)
    assert header["fit_estimate"] == {"context_size": 4096, "n_gpu_layers": 32}
    html = (tmp_path / "log.html").read_text(encoding="utf-8")
    assert "Fit estimate" in html
    assert "context_size" in html and "4096" in html


def test_settings_header_omits_extras_when_provider_returns_none(tmp_path):
    """A provider that doesn't override log_metadata mustn't add any keys."""
    p = MockProvider()  # base log_metadata returns None
    log = tmp_path / "log.jsonl"
    p.new_model("mock-model").new_conversation(name="t", log_path=log)
    [header] = _settings_records(log)
    expected = {"type", "convo", "provider", "model", "system_blocks", "tools", "settings"}
    assert set(header) == expected


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


def test_stream_writes_response_record(mock_model, tmp_path):
    """Streaming must write a 'response' log entry on completion, just like send."""
    mock_model.provider.canned_text = "alpha beta gamma"
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    list(convo.stream("hi"))

    [resp] = _response_records(log)
    assert resp["model"] == "mock-model"
    assert resp["text"].strip() == "alpha beta gamma"
    assert resp["usage"]["prompt_tokens"] == 10
    # The next request must see the streamed assistant turn as prior_history,
    # not redump it as new_messages — _logged_msg_count must have been bumped.
    convo.send("again")
    reqs = _request_records(log)
    assert reqs[1]["new_messages"] == [{"role": "user", "content": "again"}]
    assert reqs[1]["prior_history"]["messages"] == 2


def test_astream_writes_response_record(mock_model, tmp_path):
    import asyncio

    mock_model.provider.canned_text = "one two three"
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)

    async def run():
        async for _ in convo.astream("hi"):
            pass

    asyncio.run(run())
    [resp] = _response_records(log)
    assert resp["text"].strip() == "one two three"
    assert resp["finish_reason"] is None or isinstance(resp["finish_reason"], str)


def test_stream_early_break_still_logs_partial_response(mock_model, tmp_path):
    """A consumer that breaks out of the iterator early should still get a
    response record reflecting the partial assistant turn that was flushed
    into history — symmetric with the partial-history flush behavior."""
    mock_model.provider.canned_text = "one two three four five"
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)

    chunks: list[str] = []
    for ev in convo.stream("hi"):
        if ev.text_delta:
            chunks.append(ev.text_delta)
            if len(chunks) >= 2:
                break

    [resp] = _response_records(log)
    assert resp["text"] == "".join(chunks)
    assert "five" not in resp["text"]


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


def test_log_max_message_lines_abbreviates_html_log(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log, log_max_message_lines=10)
    novel = "\n".join(f"line-{i}" for i in range(100))
    convo.send(novel)

    html = (tmp_path / "log.html").read_text(encoding="utf-8")
    assert "[90 lines skipped]" in html
    assert "line-50" not in html
    assert "line-0" in html and "line-99" in html


def test_log_max_message_lines_unset_writes_full_html(mock_model, tmp_path):
    log = tmp_path / "log.jsonl"
    convo = mock_model.new_conversation(name="t", log_path=log)
    novel = "\n".join(f"line-{i}" for i in range(50))
    convo.send(novel)

    html = (tmp_path / "log.html").read_text(encoding="utf-8")
    assert "lines skipped" not in html
    assert "line-25" in html


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


def test_log_dir_cascade_uses_manager_run_dir(tmp_path):
    """A convo built via LLM(log_dir=...).new_provider().new_model().new_conversation()
    auto-logs to ``<run_dir>/<convo.name>.jsonl`` with no per-convo log_path."""
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path)
    p = MockProvider(manager=llm)
    convo = p.new_model("mock-model").new_conversation(name="hello")
    convo.send("hi")
    assert llm.run_dir is not None
    expected = llm.run_dir / "hello.jsonl"
    assert expected.exists()
    assert expected.read_text(encoding="utf-8")  # non-empty


def test_log_dir_default_name_is_convo_hash(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path)
    p = MockProvider(manager=llm)
    convo = p.new_model("mock-model").new_conversation()
    convo.send("hi")
    assert llm.run_dir is not None
    expected = llm.run_dir / f"{convo.name}.jsonl"
    assert expected.exists()
    assert convo.name.startswith("convo-")


def test_log_dir_provider_override_redirects(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path / "default")
    elsewhere = tmp_path / "elsewhere"
    p = MockProvider(manager=llm, log_dir=elsewhere)
    convo = p.new_model("mock-model").new_conversation(name="x")
    convo.send("hi")
    assert (elsewhere / "x.jsonl").exists()
    # Manager run_dir was never materialised because provider redirected.
    assert llm.run_dir is not None and not llm.run_dir.exists()


def test_log_dir_model_override_redirects(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path / "default")
    p = MockProvider(manager=llm)
    model_dir = tmp_path / "modeldir"
    convo = p.new_model("mock-model", log_dir=model_dir).new_conversation(name="m")
    convo.send("hi")
    assert (model_dir / "m.jsonl").exists()


def test_log_dir_convo_override_redirects(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path / "default")
    p = MockProvider(manager=llm)
    convo_dir = tmp_path / "convodir"
    convo = p.new_model("mock-model").new_conversation(name="c", log_dir=convo_dir)
    convo.send("hi")
    assert (convo_dir / "c.jsonl").exists()


def test_log_dir_false_at_convo_disables(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path)
    p = MockProvider(manager=llm)
    convo = p.new_model("mock-model").new_conversation(name="off", log_path=False)
    convo.send("hi")
    assert llm.run_dir is not None
    assert not (llm.run_dir / "off.jsonl").exists()


def test_log_dir_false_at_provider_disables(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path)
    p = MockProvider(manager=llm, log_dir=False)
    convo = p.new_model("mock-model").new_conversation(name="off")
    convo.send("hi")
    assert llm.run_dir is not None
    assert not (llm.run_dir / "off.jsonl").exists()


def test_log_dir_false_at_manager_disables_but_lower_can_reenable(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=False)
    p = MockProvider(manager=llm, log_dir=tmp_path / "p")
    convo = p.new_model("mock-model").new_conversation(name="r")
    convo.send("hi")
    assert (tmp_path / "p" / "r.jsonl").exists()


def test_explicit_log_path_still_overrides_cascade(tmp_path):
    from llmfacade import LLM

    llm = LLM(log_dir=tmp_path / "default")
    p = MockProvider(manager=llm)
    explicit = tmp_path / "explicit" / "log.jsonl"
    convo = p.new_model("mock-model").new_conversation(name="ignored", log_path=explicit)
    convo.send("hi")
    assert explicit.exists()
    # The convo's name file under the manager dir should NOT exist.
    assert llm.run_dir is not None and not (llm.run_dir / "ignored.jsonl").exists()
