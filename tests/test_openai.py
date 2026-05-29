"""OpenAI provider `_build_kwargs` wire-shape: max_completion_tokens, the
`effort` -> `reasoning_effort` mapping, and structured-output translation.

Drives a `CompletionRequest` through `OpenAIProvider._build_kwargs` (no request
is fired) and asserts the SDK-shaped payload."""

from __future__ import annotations

import pytest

from llmfacade import EffortLevel, OutputFormat
from llmfacade.provider import CompletionRequest
from llmfacade.providers.openai import OpenAIProvider

_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


def _req(**settings: object) -> CompletionRequest:
    settings.setdefault("max_tokens", 100)
    return CompletionRequest(
        model="gpt-5.5",
        messages=[],
        system_blocks=[],
        tools=[],
        stop=None,
        settings=dict(settings),
        settings_source={k: "convo" for k in settings},
    )


def test_openai_emits_max_completion_tokens(openai_provider: OpenAIProvider):
    """GPT-5 series rejects legacy `max_tokens`; the facade's `max_tokens` knob
    must go out as `max_completion_tokens`."""
    kwargs = openai_provider._build_kwargs(_req(max_tokens=256))
    assert kwargs["max_completion_tokens"] == 256
    assert "max_tokens" not in kwargs


def test_openai_declares_effort_capability():
    assert "effort" in OpenAIProvider.SUPPORTS


def test_openai_effort_enum_maps_to_reasoning_effort(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req(effort=EffortLevel.XHIGH))
    assert kwargs["reasoning_effort"] == "xhigh"
    assert "effort" not in kwargs


def test_openai_effort_string_passes_through(openai_provider: OpenAIProvider):
    """OpenAI accepts values Anthropic doesn't (e.g. "minimal"); raw strings
    pass through verbatim."""
    kwargs = openai_provider._build_kwargs(_req(effort="minimal"))
    assert kwargs["reasoning_effort"] == "minimal"


def test_openai_no_effort_means_no_reasoning_effort(openai_provider: OpenAIProvider):
    assert "reasoning_effort" not in openai_provider._build_kwargs(_req())


def test_openai_json_mode(openai_provider: OpenAIProvider):
    kwargs = openai_provider._build_kwargs(_req(output_format=OutputFormat.JSON))
    assert kwargs["response_format"] == {"type": "json_object"}


def test_openai_structured_output_bare_schema(openai_provider: OpenAIProvider):
    """A bare JSON-Schema dict becomes a strict json_schema (name defaulted)."""
    kwargs = openai_provider._build_kwargs(_req(output_format=_SCHEMA))
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": _SCHEMA, "strict": True},
    }


def test_openai_structured_output_full_config(openai_provider: OpenAIProvider):
    """A {name, schema, strict} dict is passed through with those values."""
    cfg = {"name": "Point", "schema": _SCHEMA, "strict": False}
    kwargs = openai_provider._build_kwargs(_req(output_format=cfg))
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Point", "schema": _SCHEMA, "strict": False},
    }


def test_openai_text_and_unset_omit_response_format(openai_provider: OpenAIProvider):
    assert "response_format" not in openai_provider._build_kwargs(
        _req(output_format=OutputFormat.TEXT)
    )
    assert "response_format" not in openai_provider._build_kwargs(_req())
