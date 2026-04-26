from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import AuthenticationError, SettingsLockedError, UnsupportedFeature
from llmfacade.settings import (
    AnySetting,
    ConvoSettings,
    ProviderSettings,
    Settings,
)

if TYPE_CHECKING:
    from llmfacade.facade import LLM
    from llmfacade.model import Model
    from llmfacade.models import Message, Response, StreamEvent
    from llmfacade.tools import Tool


class _SettingsFacade:
    """Capability-aware settings store. Used at every layer."""

    def __init__(
        self,
        owner: Provider | Model | object,
        supports: frozenset[AnySetting],
        provider_name: str,
        model_id: str | None = None,
    ):
        self._owner = owner
        self._supports = supports
        self._provider_name = provider_name
        self._model_id = model_id
        self._values: dict[AnySetting, Any] = {}
        self._locked = False

    def isAvailable(self, setting: AnySetting) -> bool:
        return setting in self._supports

    def getCapabilities(self) -> set[AnySetting]:
        return set(self._supports)

    def set(self, setting: AnySetting, value: Any) -> None:
        if self._locked:
            raise SettingsLockedError(f"Cannot change {setting.name} after Start().")
        if setting not in self._supports:
            raise UnsupportedFeature(setting, self._provider_name, self._model_id)
        self._values[setting] = value

    def get(self, setting: AnySetting, default: Any = None) -> Any:
        return self._values.get(setting, default)

    def has(self, setting: AnySetting) -> bool:
        return setting in self._values

    def _lock(self) -> None:
        self._locked = True

    def _snapshot(self) -> dict[AnySetting, Any]:
        return dict(self._values)


class Provider:
    """Public Provider base class. Owns auth, connection, SDK client; spawns Models."""

    SUPPORTS: frozenset[AnySetting] = frozenset()
    NAME: str = "provider"
    API_KEY_ENV: str | None = None

    def __init__(
        self,
        *,
        manager: LLM | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._manager = manager
        self._api_key_override = api_key
        self._base_url = base_url
        self.settings = _SettingsFacade(self, self.SUPPORTS, self.NAME)
        if base_url is not None:
            with contextlib.suppress(UnsupportedFeature):
                self.settings.set(ProviderSettings.BaseURL, base_url)
        self._init_client()

    @classmethod
    def create(
        cls,
        provider_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Provider:
        """Build a Provider via the default LLM manager (no explicit manager needed)."""
        from llmfacade.facade import LLM

        return LLM.default().NewProvider(provider_name, api_key=api_key, base_url=base_url)

    def NewModel(
        self,
        model_id: str,
        *,
        capability_override: frozenset[AnySetting] | None = None,
    ) -> Model:
        from llmfacade.model import Model

        return Model(
            provider=self,
            model_id=model_id,
            capability_override=capability_override,
        )

    def isAvailable(self, setting: AnySetting) -> bool:
        return setting in self.SUPPORTS

    def getCapabilities(self) -> set[AnySetting]:
        return set(self.SUPPORTS)

    @property
    def name(self) -> str:
        return self.NAME

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

    def _estimate_tokens(self, text: str, model_id: str) -> int:
        """Approximate token count for `text`. Used by the cache-summary
        diagnostic to map cache_read_tokens back to message indices.

        Default is ``chars / 4``, which is a coarse English-biased fallback.
        Subclasses should override with a real local tokenizer when available
        (e.g., tiktoken for OpenAI). Anthropic and Google ship no offline
        tokenizer, so they stay on the chars/4 fallback."""
        del model_id
        return max(1, len(text) // 4)

    def _complete_raw(
        self,
        *,
        model: str,
        messages: list[Message],
        system_blocks: list[tuple[str, bool]],
        tools: list[Tool],
        tool_choice: str,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        provider_settings: dict[AnySetting, Any],
        model_settings: dict[AnySetting, Any],
        convo_settings: dict[AnySetting, Any],
        per_call_overrides: dict[AnySetting, Any],
    ) -> Response:
        raise NotImplementedError

    async def _acomplete_raw(
        self,
        *,
        model: str,
        messages: list[Message],
        system_blocks: list[tuple[str, bool]],
        tools: list[Tool],
        tool_choice: str,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        provider_settings: dict[AnySetting, Any],
        model_settings: dict[AnySetting, Any],
        convo_settings: dict[AnySetting, Any],
        per_call_overrides: dict[AnySetting, Any],
    ) -> Response:
        raise NotImplementedError

    def _stream_raw(
        self,
        *,
        model: str,
        messages: list[Message],
        system_blocks: list[tuple[str, bool]],
        tools: list[Tool],
        tool_choice: str,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        provider_settings: dict[AnySetting, Any],
        model_settings: dict[AnySetting, Any],
        convo_settings: dict[AnySetting, Any],
        per_call_overrides: dict[AnySetting, Any],
    ) -> Iterator[StreamEvent]:
        raise NotImplementedError

    def _astream_raw(
        self,
        *,
        model: str,
        messages: list[Message],
        system_blocks: list[tuple[str, bool]],
        tools: list[Tool],
        tool_choice: str,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        provider_settings: dict[AnySetting, Any],
        model_settings: dict[AnySetting, Any],
        convo_settings: dict[AnySetting, Any],
        per_call_overrides: dict[AnySetting, Any],
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError


__all__ = [
    "Provider",
    "ProviderSettings",
    "Settings",
    "ConvoSettings",
    "_SettingsFacade",
]
