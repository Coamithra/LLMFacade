from __future__ import annotations

import datetime as _dt
import importlib
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import LLMError, ProviderNotInstalledError
from llmfacade.providers import PROVIDER_REGISTRY

if TYPE_CHECKING:
    from llmfacade.provider import Provider


class LLM:
    """Cross-provider manager. Holds shared API keys and the logging root;
    spawns Providers.

    Logging is on by default. Each LLM instance reserves a session-stamped
    directory ``<log_dir>/llmfacade<YYYYMMDD-HHMMSS>/`` into which every
    Conversation's JSONL/HTML log is written using the convo's ``name`` as
    the filename. The directory is materialised lazily on first write, so
    constructing an ``LLM`` is filesystem-free.

    - ``log_dir=None`` (default): write under ``<cwd>/logs``.
    - ``log_dir=Path | str``: write under that base.
    - ``log_dir=False``: disable logging at the manager level. Lower layers
      (provider/model/convo) can re-enable by supplying their own ``log_dir``.

    ``max_log_folders`` caps how many ``llmfacade*`` session folders are kept
    inside ``log_dir``. Older ones are deleted on first write."""

    _default: LLM | None = None
    _default_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        *,
        api_keys: dict[str, str] | None = None,
        log_dir: Path | str | bool | None = None,
        max_log_folders: int = 10,
    ):
        self.api_keys: dict[str, str] = dict(api_keys or {})
        self._max_log_folders = max(0, int(max_log_folders))
        self._run_dir_materialized = False
        if log_dir is False:
            self._log_dir_base: Path | None = None
            self._run_dir: Path | None = None
        else:
            base = Path.cwd() / "logs" if log_dir is None or log_dir is True else Path(log_dir)
            self._log_dir_base = base
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self._run_dir = base / f"llmfacade{stamp}"

    @property
    def run_dir(self) -> Path | None:
        """Planned per-session log directory, or ``None`` if logging is
        disabled. Reading this does not create the directory."""
        return self._run_dir

    def _ensure_run_dir(self) -> Path | None:
        """Materialise the session log directory, pruning older sibling
        ``llmfacade*`` folders down to ``max_log_folders``. Idempotent."""
        if self._run_dir is None or self._log_dir_base is None:
            return None
        if not self._run_dir_materialized:
            self._run_dir_materialized = True
            self._prune_old_run_dirs()
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def _prune_old_run_dirs(self) -> None:
        if self._log_dir_base is None or not self._log_dir_base.exists():
            return
        existing = sorted(
            (
                p
                for p in self._log_dir_base.iterdir()
                if p.is_dir() and p.name.startswith("llmfacade") and p != self._run_dir
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Keep the newest (max_log_folders - 1) so that, with the new run added,
        # the total stays at max_log_folders.
        to_keep = max(0, self._max_log_folders - 1)
        for old in existing[to_keep:]:
            shutil.rmtree(old, ignore_errors=True)

    @classmethod
    def default(cls) -> LLM:
        if cls._default is None:
            with cls._default_lock:
                if cls._default is None:
                    cls._default = cls()
        return cls._default

    @classmethod
    def reset_default(cls) -> None:
        """Drop the process-wide default LLM. The next ``default()`` call rebuilds it.

        Useful in test setup to ensure mutations to ``LLM.default().api_keys``
        don't leak between tests."""
        cls._default = None

    def new_provider(self, provider_name: str, **kwargs: Any) -> Provider:
        """Build a provider by name. Extra kwargs (api_key, base_url, generation
        defaults) are forwarded to the provider class's constructor."""
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
        return provider_cls(manager=self, **kwargs)
