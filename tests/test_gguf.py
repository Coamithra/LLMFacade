"""Tests for the dependency-free GGUF chat-template reader + thinking-style
classifier (``llmfacade.providers._gguf``).

The reader walks only the GGUF header's key/value metadata, so the fixtures
build minimal valid headers in-memory (magic + version + tensor_count +
kv_count + a few typed KVs) — no real model file, no weights."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from llmfacade.providers._gguf import classify_thinking_style, read_gguf_chat_template
from llmfacade.settings import ThinkingStyle

_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9


def _kv_string(key: str, val: str) -> bytes:
    kb, vb = key.encode("utf-8"), val.encode("utf-8")
    return (
        struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", _T_STRING)
        + struct.pack("<Q", len(vb))
        + vb
    )


def _kv_bool(key: str, val: bool) -> bytes:
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", _T_BOOL) + struct.pack("<?", val)


def _kv_string_array(key: str, vals: list[str]) -> bytes:
    kb = key.encode("utf-8")
    out = (
        struct.pack("<Q", len(kb))
        + kb
        + struct.pack("<I", _T_ARRAY)
        + struct.pack("<I", _T_STRING)
        + struct.pack("<Q", len(vals))
    )
    for v in vals:
        vb = v.encode("utf-8")
        out += struct.pack("<Q", len(vb)) + vb
    return out


def _build_gguf(kvs: list[bytes], *, version: int = 3) -> bytes:
    return (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", 0)  # tensor_count (unused by the reader)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )


# ---- read_gguf_chat_template ----------------------------------------------


def test_read_chat_template_roundtrip(tmp_path: Path) -> None:
    tpl = "{% if enable_thinking %}<think>{% endif %}"
    f = tmp_path / "m.gguf"
    f.write_bytes(_build_gguf([_kv_string("tokenizer.chat_template", tpl)]))
    assert read_gguf_chat_template(f) == tpl


def test_read_skips_other_kv_types_before_template(tmp_path: Path) -> None:
    """A bool, a string array, and a string KV precede the template — exercises
    every ``_skip_value`` branch the reader walks past to find the template."""
    f = tmp_path / "m.gguf"
    f.write_bytes(
        _build_gguf(
            [
                _kv_bool("some.flag", True),
                _kv_string_array("tokenizer.tokens", ["a", "b", "c"]),
                _kv_string("general.name", "x"),
                _kv_string("tokenizer.chat_template", "T:enable_thinking"),
            ]
        )
    )
    assert read_gguf_chat_template(f) == "T:enable_thinking"


def test_read_no_template_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(_build_gguf([_kv_string("general.name", "x")]))
    assert read_gguf_chat_template(f) is None


def test_read_non_gguf_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"not a gguf file at all")
    assert read_gguf_chat_template(f) is None


def test_read_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_gguf_chat_template(tmp_path / "nope.gguf") is None


def test_read_truncated_header_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + struct.pack("<I", 3))  # magic + version, counts missing
    assert read_gguf_chat_template(f) is None


def test_read_v1_unsupported_returns_none(tmp_path: Path) -> None:
    """GGUF v1 used 32-bit counts/lengths; the reader bails (returns None)
    rather than misparse a layout it doesn't support."""
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\x00" * 16)
    assert read_gguf_chat_template(f) is None


# ---- classify_thinking_style ----------------------------------------------


@pytest.mark.parametrize(
    "template,expected",
    [
        ("{% if enable_thinking %}", ThinkingStyle.TEMPLATE_KWARG),
        ("uses reasoning_effort kwarg", ThinkingStyle.REASONING_BUDGET),
        ("uses thinking_budget kwarg", ThinkingStyle.REASONING_BUDGET),
        ("emits a <think> block", ThinkingStyle.THINK_TOKEN),
        ("closes with </think>", ThinkingStyle.THINK_TOKEN),
        ("opens <thinking>", ThinkingStyle.THINK_TOKEN),
        ("a plain template with no reasoning machinery", ThinkingStyle.DEFAULT),
        ("", ThinkingStyle.UNKNOWN),
        (None, ThinkingStyle.UNKNOWN),
    ],
)
def test_classify_thinking_style(template: str | None, expected: ThinkingStyle) -> None:
    assert classify_thinking_style(template) == expected


def test_classify_enable_thinking_wins_over_think_tag() -> None:
    """When a template carries both an `enable_thinking` toggle and `<think>`
    tags, the controllable toggle wins — it's the gate the `thinking` knob can
    actually flip, and it's checked first."""
    tpl = "{% if enable_thinking %}<think>{% endif %}"
    assert classify_thinking_style(tpl) == ThinkingStyle.TEMPLATE_KWARG


def test_read_then_classify_end_to_end(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(_build_gguf([_kv_string("tokenizer.chat_template", "{{ enable_thinking }}")]))
    assert classify_thinking_style(read_gguf_chat_template(f)) == ThinkingStyle.TEMPLATE_KWARG
