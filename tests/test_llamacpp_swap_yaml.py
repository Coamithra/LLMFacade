"""Pure-function tests for the llama-swap YAML renderer.

Covers minimum-viable entries, all-knobs entries, multiple entries, ordering
determinism, escaping of paths with shell-meaningful characters, and
``extra_args`` passthrough. No subprocess or filesystem dependency."""

from __future__ import annotations

import pytest

# pyyaml is the optional `[llamacpp]` extra; skip the whole module if it's
# missing so a plain `pip install -e .[dev]` still has a green test suite.
yaml = pytest.importorskip("yaml")

from llmfacade.providers._launch import _LaunchEntry, derive_model_id  # noqa: E402
from llmfacade.providers._swap_lifecycle import _render_swap_yaml  # noqa: E402


def _parse(rendered: str) -> dict:
    return yaml.safe_load(rendered)


def test_render_minimum_entry() -> None:
    entry = _LaunchEntry(model_id="qwen", gguf="models/qwen.gguf")
    rendered = _render_swap_yaml([entry])
    doc = _parse(rendered)
    assert doc["models"]["qwen"]["cmd"].startswith("llama-server --model models/qwen.gguf")
    assert "--port ${PORT}" in doc["models"]["qwen"]["cmd"]
    assert doc["models"]["qwen"]["ttl"] == 0
    assert doc["healthCheckTimeout"] == 60


def test_render_all_knobs_set() -> None:
    entry = _LaunchEntry(
        model_id="qwen-fast",
        gguf="models/qwen.gguf",
        context_size=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        n_gpu_layers=32,
        parallel=2,
        slot_save_path="/var/cache/slots",
        ttl=300,
        extra_args=("--mlock", "--flash-attn"),
    )
    cmd = _parse(_render_swap_yaml([entry]))["models"]["qwen-fast"]["cmd"]
    assert "--ctx-size 8192" in cmd
    assert "--cache-type-k q8_0" in cmd
    assert "--cache-type-v q8_0" in cmd
    assert "--n-gpu-layers 32" in cmd
    assert "--parallel 2" in cmd
    assert "--slot-save-path /var/cache/slots" in cmd
    assert "--mlock" in cmd
    assert "--flash-attn" in cmd


def test_render_multiple_entries_preserves_order() -> None:
    e1 = _LaunchEntry(model_id="aaa", gguf="a.gguf")
    e2 = _LaunchEntry(model_id="bbb", gguf="b.gguf")
    e3 = _LaunchEntry(model_id="ccc", gguf="c.gguf")
    rendered = _render_swap_yaml([e1, e2, e3])
    # PyYAML's safe_dump preserves insertion order for dicts in py3.7+.
    keys = list(_parse(rendered)["models"].keys())
    assert keys == ["aaa", "bbb", "ccc"]


def test_render_is_deterministic() -> None:
    e1 = _LaunchEntry(model_id="qwen", gguf="models/qwen.gguf", context_size=4096)
    e2 = _LaunchEntry(model_id="qwen", gguf="models/qwen.gguf", context_size=4096)
    assert _render_swap_yaml([e1]) == _render_swap_yaml([e2])


def test_render_escapes_paths_with_spaces() -> None:
    entry = _LaunchEntry(
        model_id="spaced",
        gguf="C:/Path With Spaces/qwen.gguf",
        slot_save_path="/var/with space/slots",
    )
    cmd = _parse(_render_swap_yaml([entry]))["models"]["spaced"]["cmd"]
    # shlex.quote wraps in single quotes when whitespace is present.
    assert "'C:/Path With Spaces/qwen.gguf'" in cmd
    assert "'/var/with space/slots'" in cmd


def test_render_extra_args_passthrough() -> None:
    entry = _LaunchEntry(
        model_id="raw",
        gguf="x.gguf",
        extra_args=("--something-weird", "value with space"),
    )
    cmd = _parse(_render_swap_yaml([entry]))["models"]["raw"]["cmd"]
    assert "--something-weird" in cmd
    assert "'value with space'" in cmd


def test_render_global_ttl_applies_when_entry_unset() -> None:
    entry = _LaunchEntry(model_id="m", gguf="x.gguf")  # ttl is None on the entry
    doc = _parse(_render_swap_yaml([entry], global_ttl=120))
    assert doc["models"]["m"]["ttl"] == 120


def test_render_entry_ttl_overrides_global() -> None:
    entry = _LaunchEntry(model_id="m", gguf="x.gguf", ttl=999)
    doc = _parse(_render_swap_yaml([entry], global_ttl=120))
    assert doc["models"]["m"]["ttl"] == 999


def test_render_health_check_timeout_passes_through() -> None:
    entry = _LaunchEntry(model_id="m", gguf="x.gguf")
    doc = _parse(_render_swap_yaml([entry], health_check_timeout=42))
    assert doc["healthCheckTimeout"] == 42


def test_derive_model_id_uses_name_when_provided() -> None:
    cfg = {"gguf": "models/qwen.gguf", "context_size": 8192}
    assert derive_model_id(cfg, name="qwen-fast") == "qwen-fast"


def test_derive_model_id_falls_back_to_stem_plus_hash() -> None:
    cfg = {"gguf": "models/qwen.gguf", "context_size": 8192}
    out = derive_model_id(cfg, name=None)
    assert out.startswith("qwen-")
    # 8 hex chars after the stem-dash separator
    suffix = out.rsplit("-", 1)[1]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_derive_model_id_is_idempotent_for_same_config() -> None:
    cfg = {"gguf": "models/qwen.gguf", "context_size": 8192}
    assert derive_model_id(cfg, name=None) == derive_model_id(cfg, name=None)


def test_derive_model_id_changes_with_launch_config() -> None:
    a = derive_model_id({"gguf": "models/qwen.gguf", "context_size": 8192}, name=None)
    b = derive_model_id({"gguf": "models/qwen.gguf", "context_size": 4096}, name=None)
    assert a != b
