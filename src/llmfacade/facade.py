from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from llmfacade.exceptions import LLMError, ProviderNotInstalledError
from llmfacade.providers import PROVIDER_REGISTRY

if TYPE_CHECKING:
    from llmfacade.provider import Provider


class LLM:
    """Cross-provider manager. Holds shared defaults; spawns Providers."""

    _default: LLM | None = None

    def __init__(
        self,
        *,
        api_keys: dict[str, str] | None = None,
    ):
        self.api_keys: dict[str, str] = dict(api_keys or {})

    @classmethod
    def default(cls) -> LLM:
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def reset_default(cls) -> None:
        """Drop the process-wide default LLM. The next default() call rebuilds it.

        Useful in test setup to ensure mutations to LLM.default().api_keys don't
        leak between tests."""
        cls._default = None

    def NewProvider(
        self,
        provider_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Provider:
        name = provider_name.lower()
        if name not in PROVIDER_REGISTRY:
            available = ", ".join(sorted(set(PROVIDER_REGISTRY.keys())))
            raise LLMError(f"Unknown provider {provider_name!r}. Available: {available}")

        module_path, class_name = PROVIDER_REGISTRY[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ProviderNotInstalledError(
                f"Could not import provider module {module_path!r}. "
                f"Install the SDK: pip install llmfacade[{name}]"
            ) from e

        provider_cls = getattr(module, class_name)
        return provider_cls(manager=self, api_key=api_key, base_url=base_url)
