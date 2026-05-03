"""Launch-knob bookkeeping for the llamacpp provider's managed mode.

Holds the per-model launch configuration (`_LaunchEntry`), the model-id
derivation rule (user-provided name wins, else `<gguf-stem>-<hash8>` over the
canonical-JSON of the launch config), and the hardcoded provider-level
defaults table.

These pieces are intentionally separate from the supervisor and the provider
class so the YAML renderer and the validation paths can be unit-tested
without touching subprocess plumbing."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmfacade.settings import LAUNCH_KNOBS

__all__ = [
    "LAUNCH_KNOBS",
    "_LaunchEntry",
    "derive_model_id",
    "default_provider_launch_defaults",
    "canonical_launch_json",
]


@dataclass(frozen=True, slots=True)
class _LaunchEntry:
    """One model registered with a managed-mode `LlamaCppServerProvider`.

    `model_id` is what the OpenAI-compat `/v1/chat/completions` request will
    send in the `model` field — llama-swap routes off it. Everything else is
    a server-launch knob translated into a llama-server CLI invocation by
    the YAML renderer."""

    model_id: str
    gguf: str
    context_size: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    n_gpu_layers: int | None = None
    parallel: int | None = None
    slot_save_path: str | None = None
    ttl: int | None = None
    extra_args: tuple[str, ...] = ()


def canonical_launch_json(launch_config: dict[str, Any]) -> str:
    """Produce a deterministic JSON string of a launch config so the same
    settings always hash to the same model id. Sorted keys; tuples become
    lists; `None` values are dropped so omitting a knob equals defaulting it.
    `gguf` is normalised via ``Path.resolve()`` so the same file referenced by
    relative vs absolute path produces the same hash."""
    cleaned: dict[str, Any] = {}
    for k in sorted(launch_config):
        v = launch_config[k]
        if v is None:
            continue
        if k == "gguf" and isinstance(v, str):
            with contextlib.suppress(OSError):
                v = str(Path(v).resolve())
        if isinstance(v, tuple):
            v = list(v)
        cleaned[k] = v
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def derive_model_id(launch_config: dict[str, Any], name: str | None) -> str:
    """`name` wins if given. Otherwise `<gguf-stem>-<hash8>` of the canonical
    launch JSON — readable in logs and uniquely identifying the launch."""
    if name is not None:
        return name
    gguf = launch_config.get("gguf")
    if not gguf:
        raise ValueError("launch_config must include 'gguf' to derive a model id")
    stem = Path(gguf).stem
    digest = hashlib.sha256(canonical_launch_json(launch_config).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def default_provider_launch_defaults(llmfacade_dir: Path) -> dict[str, Any]:
    """Hardcoded provider-level launch defaults. Keys present here cascade
    into every model registered on the provider unless the model overrides
    them. `slot_save_path` is rooted at the per-provider session directory
    so multiple providers in the same Python process don't collide."""
    return {
        "context_size": None,
        "cache_type_k": None,
        "cache_type_v": None,
        "n_gpu_layers": None,
        "parallel": 1,
        "slot_save_path": str(llmfacade_dir / "slots"),
        "ttl": 0,
        "extra_args": (),
    }
