"""Launch-knob bookkeeping for the localimage provider's managed mode.

The image analog of ``_launch.py``: holds the per-model launch configuration
(``_SdLaunchEntry``), the model-id derivation rule (user-provided name wins, else
``<model-stem>-<hash8>`` over the canonical-JSON of the launch config), the
provider-level defaults table, and the pure ``sd-server`` argv builder.

Kept separate from the supervisor so the argv builder and the validation paths
are unit-testable without touching subprocess plumbing."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

__all__ = [
    "SD_LAUNCH_KNOBS",
    "_SdLaunchEntry",
    "canonical_sd_launch_json",
    "derive_image_model_id",
    "default_provider_sd_defaults",
    "build_sd_server_argv",
]


# Launch knobs accepted at provider/model scope in managed mode. The file-path
# group selects the model (a single-file ``--model`` checkpoint, or the
# split-pipeline ``--diffusion-model`` + encoders/VAE); the rest are perf knobs.
# Everything sd-server supports that isn't curated here goes through ``extra_args``.
SD_LAUNCH_KNOBS: frozenset[str] = frozenset(
    {
        "model",
        "diffusion_model",
        "vae",
        "clip_l",
        "clip_g",
        "t5xxl",
        "llm",
        "taesd",
        "lora_model_dir",
        "threads",
        "max_vram",
        "offload_to_cpu",
        "fa",
        "diffusion_fa",
        "extra_args",
    }
)

# Launch knobs that name a file on disk (existence-checked at registration) and
# the one that names a directory.
_FILE_PATH_KNOBS: tuple[str, ...] = (
    "model",
    "diffusion_model",
    "vae",
    "clip_l",
    "clip_g",
    "t5xxl",
    "llm",
    "taesd",
)
_DIR_PATH_KNOBS: tuple[str, ...] = ("lora_model_dir",)


@dataclass(frozen=True, slots=True)
class _SdLaunchEntry:
    """One image model registered with a managed-mode ``LocalImageProvider``.

    ``model_id`` is the supervisor's routing key (which ``sd-server`` to spawn);
    sd-server itself is single-model and ignores any wire ``model`` field. The
    rest are server-launch knobs translated into an ``sd-server`` CLI invocation
    by :func:`build_sd_server_argv`."""

    model_id: str
    model: str | None = None
    diffusion_model: str | None = None
    vae: str | None = None
    clip_l: str | None = None
    clip_g: str | None = None
    t5xxl: str | None = None
    llm: str | None = None
    taesd: str | None = None
    lora_model_dir: str | None = None
    threads: int | None = None
    max_vram: float | None = None
    offload_to_cpu: bool = False
    fa: bool = False
    diffusion_fa: bool = False
    extra_args: tuple[str, ...] = ()


def canonical_sd_launch_json(launch_config: dict[str, Any]) -> str:
    """Deterministic JSON of a launch config so identical settings always hash to
    the same model id. Sorted keys; tuples→lists; ``None`` dropped (so omitting a
    knob equals defaulting it); file-path knobs normalised via ``Path.resolve()``
    so the same file by relative vs absolute path hashes identically."""
    cleaned: dict[str, Any] = {}
    for k in sorted(launch_config):
        v = launch_config[k]
        if v is None:
            continue
        if (k in _FILE_PATH_KNOBS or k in _DIR_PATH_KNOBS) and isinstance(v, str):
            with contextlib.suppress(OSError):
                v = str(Path(v).resolve())
        if isinstance(v, tuple):
            v = list(v)
        cleaned[k] = v
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def derive_image_model_id(launch_config: dict[str, Any], name: str | None) -> str:
    """``name`` wins if given. Otherwise ``<model-stem>-<hash8>`` of the canonical
    launch JSON, where the stem is taken from ``model`` (single-file checkpoint)
    or, failing that, ``diffusion_model``. Raises if neither model source is set —
    sd-server needs one to load anything."""
    if name is not None:
        return name
    source = launch_config.get("model") or launch_config.get("diffusion_model")
    if not source:
        raise ValueError(
            "launch_config must include 'model' or 'diffusion_model' to derive a model id"
        )
    stem = Path(source).stem
    digest = hashlib.sha256(canonical_sd_launch_json(launch_config).encode("utf-8")).hexdigest()[
        :8
    ]
    return f"{stem}-{digest}"


def default_provider_sd_defaults() -> dict[str, Any]:
    """Hardcoded provider-level launch defaults. Keys present here cascade into
    every image model registered on the provider unless the model overrides them
    (``provider < model``, mirroring the llamacpp managed cascade). All knobs
    default to their off/unset value so a bare provider imposes nothing."""
    return {
        "model": None,
        "diffusion_model": None,
        "vae": None,
        "clip_l": None,
        "clip_g": None,
        "t5xxl": None,
        "llm": None,
        "taesd": None,
        "lora_model_dir": None,
        "threads": None,
        "max_vram": None,
        "offload_to_cpu": False,
        "fa": False,
        "diffusion_fa": False,
        "extra_args": (),
    }


# Maps a launch-entry field to its sd-server CLI flag. Value-bearing knobs only;
# booleans are handled separately as presence flags.
_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("model", "--model"),
    ("diffusion_model", "--diffusion-model"),
    ("vae", "--vae"),
    ("clip_l", "--clip_l"),
    ("clip_g", "--clip_g"),
    ("t5xxl", "--t5xxl"),
    ("llm", "--llm"),
    ("taesd", "--taesd"),
    ("lora_model_dir", "--lora-model-dir"),
    ("threads", "--threads"),
    ("max_vram", "--max-vram"),
)
_BOOL_FLAGS: tuple[tuple[str, str], ...] = (
    ("offload_to_cpu", "--offload-to-cpu"),
    ("fa", "--fa"),
    ("diffusion_fa", "--diffusion-fa"),
)


def build_sd_server_argv(
    binary: str,
    entry: _SdLaunchEntry,
    *,
    port: int,
    listen_ip: str = "127.0.0.1",
) -> list[str]:
    """Build the ``sd-server`` argv for one entry. Deterministic flag order
    (listen → value flags → bool flags → ``extra_args``) so it's snapshot-stable
    for tests. Unlike the llamacpp path there is no ``${PORT}`` placeholder — we
    spawn sd-server directly, so the allocated ``port`` is baked in."""
    argv: list[str] = [binary, "--listen-ip", listen_ip, "--listen-port", str(port)]
    for field, flag in _VALUE_FLAGS:
        value = getattr(entry, field)
        if value is not None:
            argv += [flag, str(value)]
    for field, flag in _BOOL_FLAGS:
        if getattr(entry, field):
            argv.append(flag)
    argv.extend(entry.extra_args)
    return argv


# Sanity: every launch knob (minus model_id, which is derived) has a home in the
# defaults table, so a new field can't silently bypass the cascade.
def _assert_knob_coverage() -> None:  # pragma: no cover - import-time guard
    entry_fields = {f.name for f in fields(_SdLaunchEntry)} - {"model_id"}
    assert entry_fields == SD_LAUNCH_KNOBS, (
        f"_SdLaunchEntry fields {entry_fields} != SD_LAUNCH_KNOBS {SD_LAUNCH_KNOBS}"
    )
    assert set(default_provider_sd_defaults()) == SD_LAUNCH_KNOBS


_assert_knob_coverage()
