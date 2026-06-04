from __future__ import annotations

from enum import Enum


class EffortLevel(Enum):
    """Reasoning-effort levels — controls thinking depth and overall token spend.

    Maps to Anthropic ``output_config.effort`` and OpenAI ``reasoning_effort``.
    The members cover the Anthropic value set (default ``HIGH``; ``XHIGH``/``MAX``
    are Opus-tier only — Sonnet/Haiku reject them, and Sonnet 4.5 / Haiku 4.5
    reject effort entirely). The provider surfaces don't line up exactly: OpenAI
    accepts ``none``/``minimal`` (not in this enum) but has **no ``MAX``**. Both
    providers also accept a raw string for the knob, so provider-specific values
    pass through; ``EffortLevel`` is the portable convenience for the shared
    levels (``low``/``medium``/``high``/``xhigh``)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ThinkingMode(Enum):
    """Adaptive-thinking modes for the Anthropic ``thinking`` knob.

    On Opus 4.7/4.8 adaptive is the *only* on-mode — budget-based extended
    thinking (passing an ``int`` token budget) returns a 400 there and is gated
    by the ``"thinking_budget"`` capability. ``ADAPTIVE`` lets the model decide
    when/how much to think with reasoning text omitted (the API default);
    ``ADAPTIVE_SUMMARIZED`` surfaces a summary of the reasoning in thinking
    blocks; ``DISABLED`` turns thinking off explicitly. Passing an ``int`` for
    ``thinking`` still selects legacy budget-based extended thinking on models
    that support it (e.g. Sonnet 4.6, older models)."""

    ADAPTIVE = "adaptive"
    ADAPTIVE_SUMMARIZED = "adaptive_summarized"
    DISABLED = "disabled"


_ADAPTIVE_THINKING_VALUES: frozenset[str] = frozenset(m.value for m in ThinkingMode)


def is_budget_thinking(value: object) -> bool:
    """True if a ``thinking`` knob value selects legacy budget-based extended
    thinking (an integer token budget) rather than an adaptive/disabled mode.

    The budget form (``{"type": "enabled", "budget_tokens": N}``) is rejected by
    Opus 4.7/4.8; the request-time gate uses this to raise ``UnsupportedFeature``
    on models that lack the ``"thinking_budget"`` capability. ``ThinkingMode``
    members and their string values are adaptive/disabled modes, never budget.
    This classifies the thinking *form* only — it does not validate a budget's
    value range (a non-positive or too-small budget is the caller's problem and
    surfaces as a provider 400). ``bool`` is never a budget (so a stray
    ``thinking=True`` isn't silently read as ``budget_tokens=1``)."""
    if value is None or isinstance(value, (bool, ThinkingMode)):
        return False
    if isinstance(value, str):
        return value not in _ADAPTIVE_THINKING_VALUES
    return True


class ThinkingStyle(Enum):
    """How a local (llama.cpp) model's chat template gates reasoning output.

    Auto-detected from the GGUF's embedded ``tokenizer.chat_template`` at
    ``new_model()`` time (managed mode), or set explicitly via
    ``new_model(thinking_style=...)``. It records *whether the* ``thinking``
    *knob can actually control reasoning* for that model: only
    ``TEMPLATE_KWARG`` models honor the ``enable_thinking`` template kwarg the
    knob emits, so the provider warns when ``thinking`` is set against a model
    of any other recognised style.

    * ``TEMPLATE_KWARG``   — ``enable_thinking`` template kwarg (Gemma 4, Qwen3).
                             The ``thinking`` knob maps to it cleanly.
    * ``REASONING_BUDGET`` — ``reasoning_effort`` / ``thinking_budget`` template
                             kwarg; ``enable_thinking`` is not the control.
    * ``THINK_TOKEN``      — emits ``<think>``/``<thinking>`` with no template
                             toggle; reasoning is governed by the server's
                             ``--reasoning-format`` parsing, not a kwarg.
    * ``DEFAULT``          — template carries no recognised reasoning machinery.
    * ``UNKNOWN``          — template absent or GGUF unreadable (and external
                             mode, where the GGUF isn't local to inspect).
    """

    TEMPLATE_KWARG = "template_kwarg"
    REASONING_BUDGET = "reasoning_budget"
    THINK_TOKEN = "think_token"
    DEFAULT = "default"
    UNKNOWN = "unknown"


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"


class EphemeralCacheTTL(Enum):
    """TTL for Anthropic ephemeral cache_control. 5m is the API default; 1h
    requires the extended-cache beta on older API versions."""

    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"


# Every knob that a provider may accept as a per-request parameter. Each
# provider's class-level SUPPORTS frozenset declares which subset it accepts.
# Defaults can be set at provider, model, or conversation construction;
# per-call overrides at send/stream. The cascade merges them in that order.
RUNTIME_KNOBS: frozenset[str] = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "effort",
        "thinking",
        "output_format",
        "user_metadata",
        "cache_ttl",
        "auto_cache_last_user",
        "auto_cache_tools",
        "beta_headers",
        "tool_choice",
    }
)


# Server-launch knobs consumed by the llamacpp provider's managed mode (the
# llama-swap-supervised lifecycle). Valid only at provider/model scope on
# providers that opt in — never per-call, never on Conversation. Other
# providers' constructors don't accept these and so reject them via TypeError
# on unknown kwarg, the same mechanism used today for unrecognised RUNTIME
# knob names.
LAUNCH_KNOBS: frozenset[str] = frozenset(
    {
        "gguf",
        "context_size",
        "cache_type_k",
        "cache_type_v",
        "n_gpu_layers",
        "n_cpu_moe",
        "parallel",
        "slot_save_path",
        "ttl",
        "extra_args",
        "fit",
        "fit_target",
        "fit_ctx",
        "flash_attn",
        "mmproj_path",
        "jinja",
        "no_mmap",
        "mlock",
    }
)


__all__ = [
    "EffortLevel",
    "ThinkingMode",
    "ThinkingStyle",
    "is_budget_thinking",
    "OutputFormat",
    "EphemeralCacheTTL",
    "RUNTIME_KNOBS",
    "LAUNCH_KNOBS",
]
