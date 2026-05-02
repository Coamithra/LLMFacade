"""Anthropic effort kwarg shape: must be wrapped under output_config.

Regression test for the bug where llmfacade was passing effort= as a top-level
kwarg to client.messages.create, which the Anthropic SDK rejects with
TypeError. The SDK expects output_config={"effort": "..."} instead.
"""

from __future__ import annotations

from llmfacade import EffortLevel, Message
from llmfacade.provider import CompletionRequest
from llmfacade.providers.anthropic import AnthropicProvider


def _make_req(*, effort: object) -> CompletionRequest:
    return CompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"effort": effort, "max_tokens": 16},
        settings_source={"effort": "convo", "max_tokens": "convo"},
    )


def test_effort_enum_is_wrapped_in_output_config():
    p = AnthropicProvider(api_key="test-key")
    api_kwargs = p._build_kwargs(_make_req(effort=EffortLevel.MAX))
    assert "effort" not in api_kwargs
    assert api_kwargs["output_config"] == {"effort": "max"}


def test_effort_string_is_wrapped_in_output_config():
    """The cascade allows raw strings too; they should be wrapped identically."""
    p = AnthropicProvider(api_key="test-key")
    api_kwargs = p._build_kwargs(_make_req(effort="normal"))
    assert "effort" not in api_kwargs
    assert api_kwargs["output_config"] == {"effort": "normal"}


def test_no_effort_means_no_output_config():
    """When effort is unset we must not emit output_config — that field is for
    effort only and the SDK shouldn't see an empty/None one."""
    p = AnthropicProvider(api_key="test-key")
    req = CompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[Message(role="user", content="hi")],
        system_blocks=[],
        tools=[],
        stop=None,
        settings={"max_tokens": 16},
        settings_source={"max_tokens": "convo"},
    )
    api_kwargs = p._build_kwargs(req)
    assert "effort" not in api_kwargs
    assert "output_config" not in api_kwargs
