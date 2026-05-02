from __future__ import annotations


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
        setting: str,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.setting = setting
        self.provider = provider
        self.model = model
        where = []
        if provider:
            where.append(f"provider={provider!r}")
        if model:
            where.append(f"model={model!r}")
        loc = f" on {', '.join(where)}" if where else ""
        super().__init__(f"Setting {setting!r} is not supported{loc}.")


class ToolIterationLimitError(LLMError):
    """A tool-dispatch loop exceeded its maximum iteration count.

    Raised by ``helpers.run_to_completion`` (and its async equivalent) when a
    model keeps calling tools without producing a final answer."""


class ConversationStateError(LLMError):
    """Conversation history is in an invalid state for the requested operation.

    Most commonly: the last assistant turn contains tool-use blocks that have
    no matching tool-result blocks, so the wire format is incomplete and the
    next call would be rejected by the provider."""


class CacheMissError(LLMError):
    """Replay-only response cache had no hit for this request.

    Raised by ``Conversation.send`` / ``stream`` (and async variants) when
    ``cache_mode='replay_only'`` is in effect and the request fingerprint is
    not present in the cache directory. No provider call is made."""
