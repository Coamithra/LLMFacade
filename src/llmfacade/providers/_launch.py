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
import re
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
    "parse_fit_print",
    "validate_flash_attn",
    "FLASH_ATTN_VALUES",
]


FLASH_ATTN_VALUES: frozenset[str] = frozenset({"on", "off", "auto"})


def validate_flash_attn(value: str | None) -> str | None:
    """Reject bad spellings early with a clear error rather than letting
    llama-server fail at spawn. ``None`` means "don't pass the flag, let
    llama-server's auto heuristic decide"."""
    if value is None:
        return None
    if value not in FLASH_ATTN_VALUES:
        raise ValueError(
            f"flash_attn must be one of {sorted(FLASH_ATTN_VALUES)!r} or None, got {value!r}"
        )
    return value


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
    n_cpu_moe: int | None = None
    parallel: int | None = None
    slot_save_path: str | None = None
    ttl: int | None = None
    extra_args: tuple[str, ...] = ()
    fit: bool = True
    fit_target: tuple[int, ...] | None = None
    fit_ctx: int | None = None
    flash_attn: str | None = None
    mmproj_path: str | None = None
    jinja: bool = True
    no_mmap: bool = False
    mlock: bool = False


_HASH_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {"fit", "fit_target", "fit_ctx", "no_mmap", "mlock"}
)


def canonical_launch_json(launch_config: dict[str, Any]) -> str:
    """Produce a deterministic JSON string of a launch config so the same
    settings always hash to the same model id. Sorted keys; tuples become
    lists; `None` values are dropped so omitting a knob equals defaulting it.
    `gguf` is normalised via ``Path.resolve()`` so the same file referenced by
    relative vs absolute path produces the same hash.

    The `fit*` knobs are excluded: they govern spawn-time VRAM fitting, not
    generation behaviour, so flipping `--fit on/off` mustn't change `model_id`
    (and break slot-cache continuity for users on persisted slots). `no_mmap`
    and `mlock` are excluded for the same reason: they only change *where the
    model's bytes live* (RAM vs mmap'd from disk, pinned vs pageable), never
    *what the model emits*, so toggling them mustn't shift `model_id` either."""
    cleaned: dict[str, Any] = {}
    for k in sorted(launch_config):
        if k in _HASH_EXCLUDED_KEYS:
            continue
        v = launch_config[k]
        if v is None:
            continue
        if k in ("gguf", "mmproj_path") and isinstance(v, str):
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
    so multiple providers in the same Python process don't collide.

    `fit=True` lets llama-server adjust unset launch args to fit available
    VRAM at spawn time; opt out per-entry with `fit=False`.

    `jinja=True` makes llama-server render the GGUF's embedded chat template
    (`--jinja`) instead of its built-in format detection. It's the prerequisite
    for template-kwarg thinking control (`enable_thinking`) and for tool-calling
    on newer Gemma 4 / Qwen3 quants whose embedded template is the only correct
    one; opt out per-entry with `jinja=False`."""
    return {
        "context_size": None,
        "cache_type_k": None,
        "cache_type_v": None,
        "n_gpu_layers": None,
        "n_cpu_moe": None,
        "parallel": 1,
        "slot_save_path": str(llmfacade_dir / "slots"),
        "ttl": 0,
        "extra_args": (),
        "fit": True,
        "fit_target": None,
        "fit_ctx": None,
        "mmproj_path": None,
        "jinja": True,
        "no_mmap": False,
        "mlock": False,
    }


# Defensive parsers for `llama-fit-params` output. The default invocation prints
# the fitted args (`-c N -ngl N -ts ... -ot ...`) to stdout; LOG_INF lines that
# the underlying `common_fit_params` machinery emits go to stderr and contain
# per-device "<N> MiB used" totals we sum into a VRAM estimate. Source:
# llama.cpp/common/fit.cpp + tools/fit-params/fit-params.cpp.
# `_RE_NGL` accepts negatives because `-ngl -1` is the canonical "all layers"
# sentinel; `_RE_CTX` doesn't because llama-server rejects negative ctx sizes.
_RE_CTX = re.compile(r"-c\s+(\d+)")
_RE_NGL = re.compile(r"-ngl\s+(-?\d+)")
_RE_MIB_USED = re.compile(r"(\d+)\s*MiB\s+used")


def parse_fit_print(stdout: str, stderr: str) -> dict[str, Any] | None:
    """Defensive parser for `llama-fit-params` output. Returns a dict with any
    of `{context_size, n_gpu_layers, est_vram_mib}` that could be extracted, or
    `None` if nothing recognisable was found. Each field is independently
    optional; the empirical regexes are tuned for llama.cpp master output and
    fall through silently on shape changes so a future binary version can't
    break `new_model()`."""
    if not stdout and not stderr:
        return None
    out: dict[str, Any] = {}
    if stdout:
        m = _RE_CTX.search(stdout)
        if m:
            out["context_size"] = int(m.group(1))
        m = _RE_NGL.search(stdout)
        if m:
            out["n_gpu_layers"] = int(m.group(1))
    if stderr:
        mibs = [int(m.group(1)) for m in _RE_MIB_USED.finditer(stderr)]
        if mibs:
            out["est_vram_mib"] = sum(mibs)
    return out or None
