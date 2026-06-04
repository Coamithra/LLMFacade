"""Unit tests for the sd-server launch bookkeeping (``providers/_sd_launch``):
the launch entry, model-id derivation, the provider-default table, and the pure
argv builder. No subprocess or filesystem side-effects."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmfacade.providers._sd_launch import (
    SD_LAUNCH_KNOBS,
    _SdLaunchEntry,
    build_sd_server_argv,
    canonical_sd_launch_json,
    default_provider_sd_defaults,
    derive_image_model_id,
)

# ---- _SdLaunchEntry / defaults --------------------------------------------


def test_entry_defaults_are_off() -> None:
    e = _SdLaunchEntry(model_id="m")
    assert e.model is None
    assert e.diffusion_model is None
    assert e.offload_to_cpu is False
    assert e.fa is False
    assert e.diffusion_fa is False
    assert e.extra_args == ()


def test_provider_defaults_cover_all_knobs() -> None:
    assert set(default_provider_sd_defaults()) == SD_LAUNCH_KNOBS


# ---- derive_image_model_id ------------------------------------------------


def test_derive_name_wins() -> None:
    assert derive_image_model_id({"model": "/x/sd.safetensors"}, "myname") == "myname"


def test_derive_uses_model_stem_and_hash() -> None:
    mid = derive_image_model_id({"model": "/models/flux_dev.safetensors"}, None)
    assert mid.startswith("flux_dev-")
    assert len(mid.split("-")[-1]) == 8


def test_derive_falls_back_to_diffusion_model_stem() -> None:
    mid = derive_image_model_id({"diffusion_model": "/m/z_image_turbo.gguf"}, None)
    assert mid.startswith("z_image_turbo-")


def test_derive_is_deterministic() -> None:
    cfg = {"diffusion_model": "/x/flux.safetensors", "vae": "/x/ae.sft", "diffusion_fa": True}
    assert derive_image_model_id(dict(cfg), None) == derive_image_model_id(dict(cfg), None)


def test_derive_changes_with_config() -> None:
    base = {"diffusion_model": "/x/flux.safetensors"}
    a = derive_image_model_id(dict(base), None)
    b = derive_image_model_id({**base, "diffusion_fa": True}, None)
    assert a != b


def test_derive_requires_a_model_source() -> None:
    with pytest.raises(ValueError, match="model.*or.*diffusion_model"):
        derive_image_model_id({"vae": "/x/ae.sft"}, None)


def test_canonical_drops_none_and_normalises_tuples() -> None:
    j = canonical_sd_launch_json({"model": None, "extra_args": ("--steps", "8"), "fa": True})
    assert "model" not in j
    assert '"extra_args":["--steps","8"]' in j


# ---- build_sd_server_argv -------------------------------------------------


def test_argv_listen_flags_first() -> None:
    e = _SdLaunchEntry(model_id="m", model="/x/sd.safetensors")
    argv = build_sd_server_argv("sd-server", e, port=7000)
    assert argv[:5] == ["sd-server", "--listen-ip", "127.0.0.1", "--listen-port", "7000"]


def test_argv_value_and_bool_flags() -> None:
    e = _SdLaunchEntry(
        model_id="m",
        diffusion_model="/x/flux.safetensors",
        vae="/x/ae.sft",
        clip_l="/x/clip_l.safetensors",
        t5xxl="/x/t5.safetensors",
        llm="/x/qwen.safetensors",
        threads=8,
        max_vram=-0.5,
        offload_to_cpu=True,
        diffusion_fa=True,
        extra_args=("--cfg-scale", "1.0"),
    )
    argv = build_sd_server_argv("sd-server", e, port=7000)
    assert "--diffusion-model" in argv and "/x/flux.safetensors" in argv
    assert "--vae" in argv and "--clip_l" in argv and "--t5xxl" in argv and "--llm" in argv
    assert argv[argv.index("--threads") + 1] == "8"
    assert argv[argv.index("--max-vram") + 1] == "-0.5"
    assert "--offload-to-cpu" in argv
    assert "--diffusion-fa" in argv
    # extra_args land at the tail, verbatim and in order.
    assert argv[-2:] == ["--cfg-scale", "1.0"]


def test_argv_omits_unset() -> None:
    e = _SdLaunchEntry(model_id="m", model="/x/sd.safetensors")
    argv = build_sd_server_argv("sd-server", e, port=7000)
    assert "--vae" not in argv
    assert "--offload-to-cpu" not in argv
    assert "--diffusion-fa" not in argv
    assert "--threads" not in argv


def test_argv_custom_listen_ip_and_binary_path() -> None:
    e = _SdLaunchEntry(model_id="m", model="/x/sd.safetensors")
    argv = build_sd_server_argv(str(Path("/opt/bin/sd-server")), e, port=9, listen_ip="0.0.0.0")
    assert argv[0].endswith("sd-server")
    assert argv[2] == "0.0.0.0"
