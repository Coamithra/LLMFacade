from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmfacade.settings import AnySetting


class LLMError(Exception):
    """Base for all llmfacade errors."""


class AuthenticationError(LLMError):
    """Invalid or missing API key."""


class RateLimitError(LLMError):
    """Provider rate limit exceeded."""


class ProviderError(LLMError):
    """Generic provider-side error (wraps the original exception)."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class ModelNotFoundError(LLMError):
    """Requested model does not exist."""


class ProviderNotInstalledError(LLMError):
    """The provider's SDK package is not installed."""


class UnsupportedFeature(LLMError):
    """The active provider/model does not support the requested setting or feature."""

    def __init__(
        self,
        setting: AnySetting | str,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.setting = setting
        self.provider = provider
        self.model = model
        name = setting if isinstance(setting, str) else setting.name
        where = []
        if provider:
            where.append(f"provider={provider!r}")
        if model:
            where.append(f"model={model!r}")
        loc = f" on {', '.join(where)}" if where else ""
        super().__init__(f"Setting {name} is not supported{loc}.")


class NotStartedError(LLMError):
    """Operation requires a Conversation that has been Start()ed."""


class SettingsLockedError(LLMError):
    """Conversation settings cannot be changed after Start()."""
