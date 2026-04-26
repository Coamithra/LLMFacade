from __future__ import annotations

from typing import TYPE_CHECKING

from llmfacade.provider import _SettingsFacade
from llmfacade.settings import AnySetting

if TYPE_CHECKING:
    from llmfacade.conversation import Conversation
    from llmfacade.provider import Provider


class Model:
    """A specific model_id bound to a Provider, with model-level settings."""

    def __init__(
        self,
        *,
        provider: Provider,
        model_id: str,
        capability_override: frozenset[AnySetting] | None = None,
    ):
        self._provider = provider
        self._model_id = model_id
        effective = capability_override if capability_override is not None else provider.SUPPORTS
        self.settings = _SettingsFacade(
            self,
            effective,
            provider.NAME,
            model_id,
        )

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model_id(self) -> str:
        return self._model_id

    def isAvailable(self, setting: AnySetting) -> bool:
        return self.settings.isAvailable(setting)

    def getCapabilities(self) -> set[AnySetting]:
        return self.settings.getCapabilities()

    def NewConversation(self, name: str | None = None) -> Conversation:
        from llmfacade.conversation import Conversation

        return Conversation(model=self, name=name)

    def __repr__(self) -> str:
        return f"Model(provider={self._provider.NAME!r}, model_id={self._model_id!r})"
