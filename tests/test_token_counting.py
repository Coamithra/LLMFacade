"""Public count_tokens / tokenizer_name API and turn-boundary cache lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmfacade import Usage

from .conftest import MockProvider

# --- public count_tokens API --------------------------------------------------


def test_provider_count_tokens_default_chars_over_4(mock_provider):
    # chars/4 with min 1.
    assert mock_provider.count_tokens("a" * 16) == 4
    assert mock_provider.count_tokens("") == 1
    assert mock_provider.tokenizer_name() == "chars/4"


def test_model_count_tokens_delegates_to_provider(mock_model):
    assert mock_model.count_tokens("a" * 8) == mock_model.provider.count_tokens(
        "a" * 8, model_id=mock_model.model_id
    )
    assert mock_model.tokenizer_name() == mock_model.provider.tokenizer_name(
        model_id=mock_model.model_id
    )


def test_subclass_can_override_count_tokens():
    class TwoCharProvider(MockProvider):
        NAME = "twochar"

        def count_tokens(
            self,
            text: str,
            *,
            system: str | None = None,
            model_id: str | None = None,
        ) -> int:
            del model_id
            combined = len(text) + (len(system) if system else 0)
            return max(1, combined // 2)

        def tokenizer_name(self, *, model_id: str | None = None) -> str:
            del model_id
            return "chars/2"

    p = TwoCharProvider()
    assert p.count_tokens("abcd") == 2
    m = p.new_model("m1")
    assert m.count_tokens("abcd") == 2
    assert m.tokenizer_name() == "chars/2"


# --- AnthropicProvider exact_count_tokens override ---------------------------


class _FakeCountResult:
    def __init__(self, input_tokens: int):
        self.input_tokens = input_tokens


def _make_anthropic_provider(*, exact: bool):
    """Build a real AnthropicProvider with a stub _client so no network is hit."""
    from llmfacade.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(api_key="test-key", exact_count_tokens=exact)
    return p


def test_anthropic_count_tokens_default_uses_chars_over_4():
    p = _make_anthropic_provider(exact=False)
    # 16 chars → 4 tokens via chars/4 fallback. tokenizer_name reports "chars/4".
    assert p.count_tokens("a" * 16) == 4
    assert p.tokenizer_name() == "chars/4"


def test_anthropic_count_tokens_exact_calls_sdk(monkeypatch):
    p = _make_anthropic_provider(exact=True)
    captured: dict[str, object] = {}

    def fake_count_tokens(*, model: str, messages: list[dict[str, object]]):
        captured["model"] = model
        captured["messages"] = messages
        return _FakeCountResult(input_tokens=42)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake_count_tokens)
    n = p.count_tokens("hello there", model_id="claude-haiku-4-5-20251001")
    assert n == 42
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["messages"] == [{"role": "user", "content": "hello there"}]
    assert p.tokenizer_name() == "anthropic-server"


def test_anthropic_count_tokens_exact_requires_model_id():
    p = _make_anthropic_provider(exact=True)
    with pytest.raises(ValueError, match="requires a model_id"):
        p.count_tokens("hello")


def test_anthropic_count_tokens_empty_text_skips_network(monkeypatch):
    """Empty text must not hit the network even with exact_count_tokens=True."""
    p = _make_anthropic_provider(exact=True)

    def explode(**_):
        raise AssertionError("network call should not happen for empty text")

    monkeypatch.setattr(p._client.messages, "count_tokens", explode)
    # Empty text: chars/4 fallback returns 1 (the min).
    assert p.count_tokens("", model_id="claude-haiku-4-5-20251001") == 1


def test_anthropic_count_tokens_exact_falls_back_on_api_error(monkeypatch):
    import httpx

    from llmfacade.providers import anthropic as anthropic_module

    p = _make_anthropic_provider(exact=True)
    # Reset the module-level once-per-error-type warning suppression so the
    # warning fires for this test regardless of order.
    anthropic_module._EXACT_COUNT_FALLBACK_WARNED.clear()

    api_error_cls = p._module.APIError
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens")

    def fake_count_tokens(**_):
        raise api_error_cls("boom", request=req, body=None)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake_count_tokens)
    with pytest.warns(UserWarning, match="falling back to chars/4"):
        n = p.count_tokens("a" * 16, model_id="claude-haiku-4-5-20251001")
    # chars/4 fallback for 16 chars = 4.
    assert n == 4


def test_anthropic_count_tokens_exact_warns_only_once_per_error_type(monkeypatch):
    import warnings as _w

    from llmfacade.providers import anthropic as anthropic_module

    p = _make_anthropic_provider(exact=True)
    anthropic_module._EXACT_COUNT_FALLBACK_WARNED.clear()

    def fake_count_tokens(**_):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(p._client.messages, "count_tokens", fake_count_tokens)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        p.count_tokens("a" * 16, model_id="claude-haiku-4-5-20251001")
        p.count_tokens("a" * 16, model_id="claude-haiku-4-5-20251001")
    # Only the first call emits the warning; the second is suppressed by
    # _EXACT_COUNT_FALLBACK_WARNED tracking.
    assert sum("falling back to chars/4" in str(w.message) for w in caught) == 1


def test_anthropic_count_tokens_forwards_system_kwarg(monkeypatch):
    """system= is forwarded to the SDK so role-overhead matches the real send."""
    p = _make_anthropic_provider(exact=True)
    captured: dict[str, object] = {}

    def fake_count_tokens(**kwargs: object):
        captured.update(kwargs)
        return _FakeCountResult(input_tokens=99)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake_count_tokens)
    n = p.count_tokens(
        "user content",
        system="you are a helpful assistant",
        model_id="claude-haiku-4-5-20251001",
    )
    assert n == 99
    assert captured["system"] == "you are a helpful assistant"
    assert captured["messages"] == [{"role": "user", "content": "user content"}]


def test_anthropic_count_tokens_no_system_omits_kwarg(monkeypatch):
    """When system=None we must not pass `system=None` to the SDK."""
    p = _make_anthropic_provider(exact=True)
    captured: dict[str, object] = {}

    def fake_count_tokens(**kwargs: object):
        captured.update(kwargs)
        return _FakeCountResult(input_tokens=5)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake_count_tokens)
    p.count_tokens("hi", model_id="claude-haiku-4-5-20251001")
    assert "system" not in captured


def test_anthropic_count_tokens_system_only_still_calls_sdk(monkeypatch):
    """Even with empty text but non-empty system, hit the SDK so the count
    reflects system role overhead."""
    p = _make_anthropic_provider(exact=True)
    captured: dict[str, object] = {}

    def fake_count_tokens(**kwargs: object):
        captured.update(kwargs)
        return _FakeCountResult(input_tokens=12)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake_count_tokens)
    n = p.count_tokens("", system="be brief.", model_id="claude-haiku-4-5-20251001")
    assert n == 12
    assert captured["system"] == "be brief."


def test_base_count_tokens_includes_system_in_chars_over_4(mock_provider):
    """Base chars/4 sums system + text lengths."""
    # 8 + 8 = 16 chars -> 4 tokens.
    assert mock_provider.count_tokens("a" * 8, system="b" * 8) == 4
    # system alone, empty text: 8 chars -> 2 tokens.
    assert mock_provider.count_tokens("", system="b" * 8) == 2
    # both empty -> still min 1.
    assert mock_provider.count_tokens("", system="") == 1


def test_model_count_tokens_threads_system_through(monkeypatch):
    """Model.count_tokens forwards system= to the underlying provider."""
    p = _make_anthropic_provider(exact=True)
    captured: dict[str, object] = {}

    def fake(**kwargs: object):
        captured.update(kwargs)
        return _FakeCountResult(input_tokens=42)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake)
    m = p.new_model("claude-sonnet-4-6")
    n = m.count_tokens("user blob", system="sys prompt")
    assert n == 42
    assert captured["system"] == "sys prompt"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["messages"] == [{"role": "user", "content": "user blob"}]


def test_anthropic_model_count_tokens_passes_model_id(monkeypatch):
    """Model.count_tokens routes through Provider.count_tokens with the bound
    model_id, so callers don't have to pass it manually."""
    p = _make_anthropic_provider(exact=True)
    seen: dict[str, str] = {}

    def fake(*, model: str, messages):
        del messages
        seen["model"] = model
        return _FakeCountResult(input_tokens=7)

    monkeypatch.setattr(p._client.messages, "count_tokens", fake)
    m = p.new_model("claude-sonnet-4-6")
    assert m.count_tokens("hi") == 7
    assert seen["model"] == "claude-sonnet-4-6"
    assert m.tokenizer_name() == "anthropic-server"


# --- turn-boundary tracking ---------------------------------------------------


def _set_canned_usage(provider: MockProvider, **kwargs):
    provider.canned_usage = Usage(
        prompt_tokens=kwargs.get("prompt_tokens", 0),
        completion_tokens=kwargs.get("completion_tokens", 5),
        total_tokens=kwargs.get("prompt_tokens", 0) + kwargs.get("completion_tokens", 5),
        cache_creation_tokens=kwargs.get("cache_creation_tokens", 0),
        cache_read_tokens=kwargs.get("cache_read_tokens", 0),
    )


def test_turn_boundary_records_total_input_after_send(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    _set_canned_usage(p, prompt_tokens=120, cache_read_tokens=80, cache_creation_tokens=10)
    convo.send("hello")
    # After send: 1 user message at send-time; total_input = 120 + 80 + 10 = 210.
    assert convo._turn_boundaries == [(1, 210)]
    _set_canned_usage(p, prompt_tokens=300)
    convo.send("again")
    # send-time message count was 3 (user, assistant from prev, user).
    assert convo._turn_boundaries == [(1, 210), (3, 300)]


def test_turn_boundary_skipped_when_usage_zero(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    _set_canned_usage(p, prompt_tokens=0)
    convo.send("hi")
    assert convo._turn_boundaries == []


def test_cache_summary_uses_exact_turn_boundary_match(tmp_path: Path, mock_model):
    p: MockProvider = mock_model.provider
    log_path = tmp_path / "convo.jsonl"
    convo = mock_model.new_conversation(log_path=log_path)
    # Turn 1: total input = 500 (uncached).
    _set_canned_usage(p, prompt_tokens=500)
    convo.send("first")
    # Turn 2: cache_read = 500 (matches turn 1's recorded boundary exactly).
    _set_canned_usage(p, prompt_tokens=20, cache_read_tokens=500)
    convo.send("second")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    responses = [r for r in records if r["type"] == "response"]
    assert len(responses) == 2
    summary = responses[1]["cache_summary"]
    # Boundary should be the turn-1 send-time message count (= 1: just the
    # user message; assistant got appended afterwards).
    assert summary["approximate_messages_cached"] == 1
    assert summary["tokenizer"] == "exact (turn-boundary)"
    assert summary["cache_read_tokens"] == 500


def test_cache_summary_falls_back_to_tokenizer_when_no_match(tmp_path: Path, mock_model):
    p: MockProvider = mock_model.provider
    log_path = tmp_path / "convo.jsonl"
    convo = mock_model.new_conversation(log_path=log_path)
    # Turn 1 records boundary at total=500.
    _set_canned_usage(p, prompt_tokens=500)
    convo.send("first")
    # Turn 2: cache_read = 250, does NOT match any recorded boundary.
    _set_canned_usage(p, prompt_tokens=20, cache_read_tokens=250)
    convo.send("second")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    responses = [r for r in records if r["type"] == "response"]
    summary = responses[1]["cache_summary"]
    # Falls back to chars/4 tokenizer label (mock provider has no override).
    assert summary["tokenizer"] == "chars/4"


def test_rollback_restores_turn_boundaries_to_snapshot(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    _set_canned_usage(p, prompt_tokens=100)
    convo.send("first")
    snap = convo.snapshot()
    _set_canned_usage(p, prompt_tokens=200)
    convo.send("second")
    assert len(convo._turn_boundaries) == 2

    convo.rollback(snap)
    assert convo._turn_boundaries == [(1, 100)]


def test_clone_inherits_turn_boundaries(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    _set_canned_usage(p, prompt_tokens=100)
    convo.send("first")

    twin = convo.clone()
    assert twin._turn_boundaries == convo._turn_boundaries
    # Mutating the clone's list doesn't affect the parent.
    twin._turn_boundaries.append((9, 999))
    assert convo._turn_boundaries == [(1, 100)]


def test_turn_boundary_stream_path_records(mock_model):
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    _set_canned_usage(p, prompt_tokens=77)
    list(convo.stream("hello"))
    assert convo._turn_boundaries == [(1, 77)]


def test_estimate_cached_boundary_picks_largest_matching(mock_model):
    """If multiple recorded boundaries share the same total, prefer the most
    recent (largest msg_count)."""
    p: MockProvider = mock_model.provider
    convo = mock_model.new_conversation()
    # Manually seed boundaries to simulate two turns reporting the same total.
    convo._turn_boundaries = [(1, 500), (3, 500)]
    convo.add_user_message("u1")
    convo.add_assistant_message("a1")
    convo.add_user_message("u2")
    convo.add_assistant_message("a2")
    convo.add_user_message("u3")
    # Build a request to feed _estimate_cached_boundary.
    _set_canned_usage(p, prompt_tokens=20, cache_read_tokens=500)
    convo.send()
    # The send mutated history, but the cache_read=500 lookup happens on the
    # request that includes 5 prefix messages, so it should pick (3, 500).
    # Use the recorded boundaries to verify directly:
    boundary, exact = convo._estimate_cached_boundary(
        type("Req", (), {"messages": [None] * 5, "system_blocks": []})(), 500
    )
    assert exact is True
    assert boundary == 3


# --- PROMPT_TOKENS_INCLUDE_CACHED (per-provider usage semantics) ---------------


def test_prompt_tokens_include_cached_flags():
    from llmfacade.provider import Provider
    from llmfacade.providers.anthropic import AnthropicProvider
    from llmfacade.providers.google import GoogleProvider
    from llmfacade.providers.llamacpp import LlamaCppServerProvider
    from llmfacade.providers.openai import OpenAIProvider

    assert Provider.PROMPT_TOKENS_INCLUDE_CACHED is False
    # Anthropic's input_tokens EXCLUDES cache reads/creations (additive).
    assert AnthropicProvider.PROMPT_TOKENS_INCLUDE_CACHED is False
    # OpenAI's prompt_tokens INCLUDES prompt_tokens_details.cached_tokens.
    assert OpenAIProvider.PROMPT_TOKENS_INCLUDE_CACHED is True
    # google-genai: prompt_token_count "also includes the number of tokens in
    # the cached content" when cached_content is set.
    assert GoogleProvider.PROMPT_TOKENS_INCLUDE_CACHED is True
    # OpenAI-compat shape; cache_read_tokens is never populated today.
    assert LlamaCppServerProvider.PROMPT_TOKENS_INCLUDE_CACHED is True


def test_turn_boundary_includes_cached_uses_prompt_as_total(mock_model):
    p: MockProvider = mock_model.provider
    p.PROMPT_TOKENS_INCLUDE_CACHED = True
    convo = mock_model.new_conversation()
    # OpenAI-style: prompt_tokens=120 already contains the 80 cached tokens,
    # so the boundary total is 120, not 200.
    _set_canned_usage(p, prompt_tokens=120, cache_read_tokens=80)
    convo.send("hello")
    assert convo._turn_boundaries == [(1, 120)]


def test_cache_summary_exact_boundary_fires_for_openai_style_usage(tmp_path: Path, mock_model):
    p: MockProvider = mock_model.provider
    p.PROMPT_TOKENS_INCLUDE_CACHED = True
    log_path = tmp_path / "convo.jsonl"
    convo = mock_model.new_conversation(log_path=log_path)
    # Turn 1: 500 input tokens, nothing cached yet.
    _set_canned_usage(p, prompt_tokens=500)
    convo.send("first")
    # Turn 2: the whole turn-1 prefix is a cache hit. prompt_tokens (520)
    # already includes the 500 cached tokens (OpenAI semantics), so the
    # recorded turn-1 boundary (500) matches cache_read exactly.
    _set_canned_usage(p, prompt_tokens=520, cache_read_tokens=500)
    convo.send("second")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    responses = [r for r in records if r["type"] == "response"]
    summary = responses[1]["cache_summary"]
    assert summary["tokenizer"] == "exact (turn-boundary)"
    assert summary["approximate_messages_cached"] == 1
    assert summary["uncached_input_tokens"] == 20
    assert summary["hit_ratio"] == round(500 / 520, 3)


def test_cache_summary_additive_for_excludes_provider(tmp_path: Path, mock_model):
    # Default MockProvider keeps the base flag (Anthropic-style: excludes).
    p: MockProvider = mock_model.provider
    log_path = tmp_path / "convo.jsonl"
    convo = mock_model.new_conversation(log_path=log_path)
    _set_canned_usage(p, prompt_tokens=500)
    convo.send("first")
    # Anthropic-style: prompt_tokens=20 EXCLUDES the 500 cache-read tokens.
    _set_canned_usage(p, prompt_tokens=20, cache_read_tokens=500)
    convo.send("second")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    responses = [r for r in records if r["type"] == "response"]
    summary = responses[1]["cache_summary"]
    assert summary["tokenizer"] == "exact (turn-boundary)"
    assert summary["approximate_messages_cached"] == 1
    assert summary["uncached_input_tokens"] == 20
    assert summary["hit_ratio"] == round(500 / 520, 3)


# --- token-count memoization (cache-boundary fallback walk) --------------------


class _CountingProvider(MockProvider):
    """Records every count_tokens text so tests can assert memoization."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token_calls: list[str] = []

    def count_tokens(self, text, *, system=None, model_id=None):
        self.token_calls.append(text)
        return super().count_tokens(text, system=system, model_id=model_id)


def test_fallback_walk_memoizes_unchanged_prefix(tmp_path: Path):
    p = _CountingProvider()
    model = p.new_model("mock-model")
    convo = model.new_conversation(log_path=tmp_path / "c.jsonl")
    # A cache_read that never matches a recorded boundary forces the
    # tokenizer fallback walk on every logged response.
    _set_canned_usage(p, prompt_tokens=10, cache_read_tokens=10**9)
    convo.send("first message")
    first_walk = list(p.token_calls)
    assert "first message" in first_walk

    convo.send("second message")
    new_calls = p.token_calls[len(first_walk) :]
    # The unchanged turn-1 prefix is served from the memo; only the new
    # assistant reply and user message are tokenized.
    assert "first message" not in new_calls
    assert "second message" in new_calls
    assert "ok" in new_calls  # canned assistant reply


def test_memoized_fallback_returns_identical_result(tmp_path: Path):
    from llmfacade.models import Message

    p = _CountingProvider()
    model = p.new_model("mock-model")
    convo = model.new_conversation()
    msgs = [
        Message(role="user", content="u1 " * 30),
        Message(role="assistant", content="a1 " * 20),
        Message(role="user", content="u2 " * 10),
    ]
    req = type("Req", (), {"messages": msgs, "system_blocks": []})()
    first = convo._estimate_cached_boundary(req, 25)
    calls_after_first = len(p.token_calls)
    assert calls_after_first > 0
    second = convo._estimate_cached_boundary(req, 25)
    assert second == first
    # Second walk is served entirely from the memo.
    assert len(p.token_calls) == calls_after_first


def test_rollback_clears_memo_and_divergent_history_recounts(tmp_path: Path):
    from llmfacade.models import Message

    p = _CountingProvider()
    model = p.new_model("mock-model")
    convo = model.new_conversation()
    snap = convo.snapshot()
    convo.add_user_message("original branch text")
    req = type("Req", (), {"messages": list(convo.history), "system_blocks": []})()
    convo._estimate_cached_boundary(req, 1000)
    assert convo._token_count_memo

    convo.rollback(snap)
    assert convo._token_count_memo == {}

    # Divergent history after rollback is freshly counted — never served a
    # stale value (and content-hash keys couldn't collide anyway).
    convo.add_user_message("divergent branch text!!")
    req2 = type(
        "Req",
        (),
        {
            "messages": [Message(role="user", content="divergent branch text!!")],
            "system_blocks": [],
        },
    )()
    before = len(p.token_calls)
    boundary, exact = convo._estimate_cached_boundary(req2, 1000)
    assert exact is False
    assert boundary == 1  # fully covered by the huge cache_read
    assert p.token_calls[before:] == ["divergent branch text!!"]
