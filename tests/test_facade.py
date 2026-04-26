"""LLM manager: default singleton + reset behavior."""

from __future__ import annotations

from llmfacade import LLM


def test_llm_default_is_a_singleton():
    a = LLM.default()
    b = LLM.default()
    assert a is b


def test_default_mutation_phase_a():
    LLM.default().api_keys["leak"] = "abc"
    assert LLM.default().api_keys["leak"] == "abc"


def test_default_mutation_phase_b_clean():
    # If the autouse _reset_llm_default fixture is wired up correctly, the
    # mutation from phase_a must not leak into this test.
    assert "leak" not in LLM.default().api_keys


def test_reset_default_creates_fresh_instance():
    first = LLM.default()
    LLM.reset_default()
    second = LLM.default()
    assert first is not second
