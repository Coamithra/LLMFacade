from __future__ import annotations

from enum import Enum, auto


class EffortLevel(Enum):
    NORMAL = "normal"
    MAX = "max"


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"


class ProviderSettings(Enum):
    """Settings that live on a Provider instance (auth, connection, per-provider knobs)."""

    BaseURL = auto()
    OrgID = auto()
    BetaHeaders = auto()
    KeepAlive = auto()


class Settings(Enum):
    """Settings that live on a Model instance (load-time / per-call generation knobs)."""

    ContextSize = auto()
    DefaultMaxTokens = auto()
    DefaultTemperature = auto()
    TopP = auto()
    TopK = auto()
    RepeatPenalty = auto()
    Effort = auto()
    Thinking = auto()


class ConvoSettings(Enum):
    """Settings that live on a Conversation instance (per-session knobs)."""

    AutoCacheLastUser = auto()
    UserMetadata = auto()
    OutputFormat = auto()


AnySetting = ProviderSettings | Settings | ConvoSettings
