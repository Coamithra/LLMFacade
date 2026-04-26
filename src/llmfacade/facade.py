from __future__ import annotations

import importlib
from pathlib import Path
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
        log_dir: str | Path | None = None,
    ):
        self.api_keys: dict[str, str] = dict(api_keys or {})
        self.log_dir: Path | None = Path(log_dir) if log_dir is not None else None

    @classmethod
    def default(cls) -> LLM:
        if cls._default is None:
            cls._default = cls()
        return cls._default

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
