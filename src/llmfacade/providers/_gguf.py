"""Dependency-free GGUF metadata reader + chat-template reasoning classifier.

The llamacpp provider's managed mode uses this at ``new_model()`` to peek at a
GGUF's embedded ``tokenizer.chat_template`` and auto-default the model's
``thinking_style`` (see :class:`llmfacade.settings.ThinkingStyle`). Only the
GGUF header's key/value metadata is read — no tensor data, no weights — so it's
cheap (a handful of seeks) and never loads the model.

Everything here is best-effort: a missing file, an unfamiliar GGUF version, a
truncated header, or a template that matches no known pattern resolves to
``None`` / ``ThinkingStyle.UNKNOWN`` rather than raising, so template probing
can never block model registration."""

from __future__ import annotations

import struct
from pathlib import Path

from llmfacade.settings import ThinkingStyle

__all__ = ["read_gguf_chat_template", "classify_thinking_style"]

_GGUF_MAGIC = b"GGUF"
_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

# GGUF metadata value-type tags (gguf spec). ARRAY nests one of the others.
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_SIZE = {
    _T_UINT8: 1,
    _T_INT8: 1,
    _T_BOOL: 1,
    _T_UINT16: 2,
    _T_INT16: 2,
    _T_UINT32: 4,
    _T_INT32: 4,
    _T_FLOAT32: 4,
    _T_UINT64: 8,
    _T_INT64: 8,
    _T_FLOAT64: 8,
}

# Cap how much metadata we'll walk so a malformed header can't spin forever.
_MAX_KV = 1_000_000


def _read(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b


def _read_u32(f) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _read_len(f, file_size: int) -> int:
    """Read a u64 length and bounds-check it against the bytes actually left in
    the file, so a corrupt length can't reach ``f.read()`` (huge values raise
    OverflowError / MemoryError there, escaping the never-raises contract)."""
    length = _read_u64(f)
    if length > file_size - f.tell():
        raise ValueError(f"gguf length field {length} exceeds remaining file size")
    return length


def _read_str(f, file_size: int) -> str:
    return _read(f, _read_len(f, file_size)).decode("utf-8", errors="replace")


def _skip_value(f, vtype: int, file_size: int) -> None:
    if vtype in _SCALAR_SIZE:
        f.seek(_SCALAR_SIZE[vtype], 1)
    elif vtype == _T_STRING:
        f.seek(_read_len(f, file_size), 1)
    elif vtype == _T_ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        if elem_type == _T_STRING:
            # Each element carries at least an 8-byte length prefix; a count
            # beyond that bound is corrupt (and would loop ~forever).
            if count > max(file_size - f.tell(), 0) // 8:
                raise ValueError(f"gguf array count {count} exceeds remaining file size")
            for _ in range(count):
                f.seek(_read_len(f, file_size), 1)
        elif elem_type in _SCALAR_SIZE:
            if _SCALAR_SIZE[elem_type] * count > file_size - f.tell():
                raise ValueError(f"gguf array count {count} exceeds remaining file size")
            f.seek(_SCALAR_SIZE[elem_type] * count, 1)
        else:
            raise ValueError(f"unsupported gguf array element type {elem_type}")
    else:
        raise ValueError(f"unsupported gguf value type {vtype}")


def read_gguf_chat_template(path: str | Path) -> str | None:
    """Return the ``tokenizer.chat_template`` string embedded in a GGUF file, or
    ``None`` if it's absent or the file can't be parsed. Reads only the header
    key/value metadata (skips tensor data and weights)."""
    try:
        with open(path, "rb") as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            if _read(f, 4) != _GGUF_MAGIC:
                return None
            version = _read_u32(f)
            # v1 used 32-bit counts/lengths; v2 switched to 64-bit (current).
            if version < 2:
                return None
            _read_u64(f)  # tensor_count (unused)
            kv_count = _read_u64(f)
            if kv_count > _MAX_KV:
                return None
            for _ in range(kv_count):
                key = _read_str(f, file_size)
                vtype = _read_u32(f)
                if key == _CHAT_TEMPLATE_KEY and vtype == _T_STRING:
                    return _read_str(f, file_size)
                _skip_value(f, vtype, file_size)
    except (OSError, EOFError, ValueError, struct.error, MemoryError, OverflowError):
        return None
    return None


def classify_thinking_style(template: str | None) -> ThinkingStyle:
    """Classify how a chat template gates reasoning output. Best-effort string
    match; order matters — the most specific gate is checked first.

    * ``enable_thinking`` template kwarg (Gemma 4, Qwen3) -> ``TEMPLATE_KWARG``
    * ``reasoning_effort`` / ``thinking_budget`` kwarg -> ``REASONING_BUDGET``
    * ``<think>`` / ``<thinking>`` tags with no toggle -> ``THINK_TOKEN``
    * no recognised reasoning machinery -> ``DEFAULT``
    * empty/unreadable template -> ``UNKNOWN``"""
    if not template:
        return ThinkingStyle.UNKNOWN
    if "enable_thinking" in template:
        return ThinkingStyle.TEMPLATE_KWARG
    if "reasoning_effort" in template or "thinking_budget" in template:
        return ThinkingStyle.REASONING_BUDGET
    if "<think>" in template or "</think>" in template or "<thinking>" in template:
        return ThinkingStyle.THINK_TOKEN
    return ThinkingStyle.DEFAULT
