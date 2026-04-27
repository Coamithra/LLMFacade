"""Public count_tokens / tokenizer_name API and turn-boundary cache lookup."""

from __future__ import annotations

import json
from pathlib import Path

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

        def count_tokens(self, text: str, *, model_id: str | None = None) -> int:
            del model_id
            return max(1, len(text) // 2)

        def tokenizer_name(self, *, model_id: str | None = None) -> str:
            del model_id
            return "chars/2"

    p = TwoCharProvider()
    assert p.count_tokens("abcd") == 2
    m = p.new_model("m1")
    assert m.count_tokens("abcd") == 2
    assert m.tokenizer_name() == "chars/2"


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
