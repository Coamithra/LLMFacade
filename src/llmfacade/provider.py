from __future__ import annotations

import os
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import AuthenticationError, UnsupportedFeature
from llmfacade.settings import RUNTIME_KNOBS, ThinkingMode

if TYPE_CHECKING:
    from llmfacade.facade import LLM
    from llmfacade.model import Model
    from llmfacade.models import Message, Response, StreamEvent
    from llmfacade.tools import Tool


@dataclass(frozen=True, slots=True)
class SystemBlock:
    """A piece of system prompt. ``cache=True`` requests an ephemeral cache
    marker on providers that support prompt caching (currently Anthropic)."""

    text: str
    cache: bool = False


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """Single round-trip request to a provider's raw hook.

    ``settings`` is the merged effective set: provider defaults < model
    defaults < convo defaults < per-call overrides. Providers read knobs
    directly from this dict (e.g. ``req.settings.get("temperature")``).
    ``settings_source`` records where each key came from, for logging.
    """

    model: str
    messages: list[Message]
    system_blocks: list[SystemBlock]
    tools: list[Tool]
    stop: list[str] | None
    settings: dict[str, Any] = field(default_factory=dict)
    settings_source: dict[str, str] = field(default_factory=dict)


# Track (key, source_scope, model_id) tuples we've already warned about so
# cascade-time mismatches don't spam the same message every send().
_WARNED_DROPS: set[tuple[str, str, str]] = set()


def _validate_knobs(
    knobs: dict[str, Any],
    supports: frozenset[str],
    provider: str,
    model: str | None,
) -> dict[str, Any]:
    """Return the non-None subset of ``knobs``, validated against ``supports``.

    Unknown keys raise ``TypeError``. Known but unsupported keys raise
    ``UnsupportedFeature``."""
    out: dict[str, Any] = {}
    for k, v in knobs.items():
        if k not in RUNTIME_KNOBS:
            raise TypeError(f"Unknown setting {k!r}. Valid: {sorted(RUNTIME_KNOBS)}")
        if v is None:
            continue
        if k not in supports:
            raise UnsupportedFeature(k, provider, model)
        out[k] = v
    return out


def _filter_unsupported(
    merged: dict[str, Any],
    sources: dict[str, str],
    supports: frozenset[str],
    provider: str,
    model: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Drop keys not in ``supports``. Warn once per (key, source, model)."""
    out: dict[str, Any] = {}
    out_src: dict[str, str] = {}
    for k, v in merged.items():
        if k not in supports:
            src = sources.get(k, "?")
            tag = (k, src, model)
            if tag not in _WARNED_DROPS:
                _WARNED_DROPS.add(tag)
                warnings.warn(
                    f"Setting {k!r} from {src!r} scope is not supported by "
                    f"model {model!r} (provider {provider!r}); ignoring.",
                    stacklevel=4,
                )
            continue
        out[k] = v
        out_src[k] = sources[k]
    return out, out_src


# Argument list shared by Provider.__init__, Provider.new_model,
# Model.__init__, Model.new_conversation, Conversation.__init__, and the four
# send/stream variants. Centralised so a new knob only has to be added in two
# places (RUNTIME_KNOBS and here).
_KNOB_DEFAULTS: dict[str, Any] = {k: None for k in RUNTIME_KNOBS}


class Provider:
    """Base provider. Identity (api_key, base_url) is constructor-only.
    Generation defaults are accepted as kwargs and apply to every model and
    conversation under this provider unless overridden at a lower scope."""

    SUPPORTS: frozenset[str] = frozenset()
    NAME: str = "provider"
    API_KEY_ENV: str | None = None

    def __init__(
        self,
        *,
        manager: LLM | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        # Generation defaults (subset of RUNTIME_KNOBS). Each is gated by SUPPORTS.
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
        tool_choice: str | None = None,
    ):
        self._manager = manager
        self._api_key_override = api_key
        self._base_url = base_url
        self._log_dir_override = log_dir
        self._cache_dir_override = cache_dir
        self._cache_mode_override = cache_mode
        self._defaults = _validate_knobs(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "effort": effort,
                "thinking": thinking,
                "output_format": output_format,
                "user_metadata": user_metadata,
                "cache_ttl": cache_ttl,
                "auto_cache_last_user": auto_cache_last_user,
                "auto_cache_tools": auto_cache_tools,
                "beta_headers": beta_headers,
                "tool_choice": tool_choice,
            },
            self.SUPPORTS,
            self.NAME,
            None,
        )
        self._init_client()

    @classmethod
    def create(cls, provider_name: str, **kwargs: Any) -> Provider:
        """Build a Provider via the default LLM manager (no explicit manager needed)."""
        from llmfacade.facade import LLM

        return LLM.default().new_provider(provider_name, **kwargs)

    def new_model(
        self,
        model_id: str,
        *,
        capability_override: frozenset[str] | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
        tool_choice: str | None = None,
    ) -> Model:
        from llmfacade.model import Model

        defaults = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "effort": effort,
            "thinking": thinking,
            "output_format": output_format,
            "user_metadata": user_metadata,
            "cache_ttl": cache_ttl,
            "auto_cache_last_user": auto_cache_last_user,
            "auto_cache_tools": auto_cache_tools,
            "beta_headers": beta_headers,
            "tool_choice": tool_choice,
        }
        return Model(
            provider=self,
            model_id=model_id,
            capability_override=capability_override,
            log_dir=log_dir,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            **defaults,
        )

    def is_available(self, setting: str) -> bool:
        return setting in self.SUPPORTS

    def get_capabilities(self) -> set[str]:
        return set(self.SUPPORTS)

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def defaults(self) -> dict[str, Any]:
        """Read-only view of the generation defaults set on this provider."""
        return dict(self._defaults)

    def _resolve_key(self, env_var: str) -> str:
        if self._api_key_override:
            return self._api_key_override
        if self._manager and self.NAME in self._manager.api_keys:
            return self._manager.api_keys[self.NAME]
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val
        raise AuthenticationError(
            f"No API key for {self.NAME!r}. Pass api_key=, set the manager's "
            f"api_keys dict, or set the {env_var} environment variable."
        )

    def _init_client(self) -> None:
        """Subclasses override to construct their SDK client."""

    def count_tokens(
        self,
        text: str,
        *,
        system: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Count tokens in ``text`` using the provider's best available local
        tokenizer. Default base implementation is ``chars / 4`` (coarse,
        English-biased). Subclasses override with a real local tokenizer where
        one is available — see ``OpenAIProvider`` (tiktoken) and
        ``GoogleProvider`` (sentencepiece via ``google-genai[local-tokenizer]``).

        ``system`` is an optional system-prompt string that will be counted
        alongside ``text`` so callers can size budgets without concatenating
        the system prompt into the user message themselves. Providers that
        forward to a server-side counter (Anthropic) pass it through with
        proper role overhead; providers with local tokenizers add it to the
        local token count. The base ``chars/4`` implementation simply sums
        the character lengths.

        Always local — never makes a network call (except Anthropic with
        ``exact_count_tokens=True``)."""
        del model_id
        combined_len = len(text) + (len(system) if system else 0)
        return max(1, combined_len // 4)

    def tokenizer_name(self, *, model_id: str | None = None) -> str:
        """Human-readable label for the tokenizer ``count_tokens`` will use.
        Useful for log-level diagnostics so callers know whether the count is
        exact or an approximation."""
        del model_id
        return "chars/4"

    def log_metadata(self, *, model_id: str) -> dict[str, Any] | None:
        """Optional per-provider extras to merge into the conversation's JSONL
        settings header (and HTML log) as siblings of `settings`. Default
        returns `None`; subclasses override to surface things like a fit
        estimate or backend version."""
        del model_id
        return None

    def _complete_raw(self, req: CompletionRequest) -> Response:
        raise NotImplementedError

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        raise NotImplementedError

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        raise NotImplementedError

    def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


__all__ = [
    "CompletionRequest",
    "Provider",
    "SystemBlock",
]
