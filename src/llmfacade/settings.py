from __future__ import annotations

from enum import Enum


class EffortLevel(Enum):
    NORMAL = "normal"
    MAX = "max"


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
    }
)


__all__ = [
    "EffortLevel",
    "OutputFormat",
    "EphemeralCacheTTL",
    "RUNTIME_KNOBS",
    "LAUNCH_KNOBS",
]
