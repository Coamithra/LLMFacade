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
        "repeat_penalty",
        "effort",
        "thinking",
        "output_format",
        "user_metadata",
        "cache_ttl",
        "auto_cache_last_user",
        "beta_headers",
        "keep_alive",
        "context_size",
    }
)


__all__ = [
    "EffortLevel",
    "OutputFormat",
    "EphemeralCacheTTL",
    "RUNTIME_KNOBS",
]
