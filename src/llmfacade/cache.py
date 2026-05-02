"""Response cache for deterministic replay of model output.

Hashes every input that affects a completion — provider name, model id,
system blocks (including ``cache=True`` markers), the full message list
(image bytes hashed), tool schemas (in registration order), the merged
effective settings, and the stop list — and stores the resulting Response
on disk. A subsequent call with an identical fingerprint reads back the
cached Response without invoking the provider.

The cache is opt-in: callers pass ``cache_dir=`` (and optionally
``cache_mode=``) at any of provider / model / conversation scope. The
cascade is convo > model > provider, identical to ``log_dir``.

Cache modes:
    ``read_write``   — read on hit, call provider on miss and write (default)
    ``read_only``    — read on hit, call provider on miss but do not write
    ``record_only``  — always call provider, write the result (overwrites)
    ``replay_only``  — read on hit, raise ``CacheMissError`` on miss

Streaming replay synthesises a small number of ``StreamEvent`` objects
from the cached Response — enough to reconstruct the assistant turn, but
not an attempt at faithful chunk-timing replay.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade.models import (
    ContentBlock,
    ImageBlock,
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from llmfacade.provider import CompletionRequest
    from llmfacade.tools import Tool


CACHE_MODES: frozenset[str] = frozenset({"read_write", "read_only", "record_only", "replay_only"})


def _normalize(v: Any) -> Any:
    """Recursively coerce ``v`` into a canonical, JSON-serialisable form.

    Sorts dict keys, leaves list order intact, unwraps Enums to their
    ``.value``, hex-encodes raw bytes, and falls back to ``repr`` for any
    other type (so a stray non-JSON value still produces a stable hash)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Enum):
        return _normalize(v.value)
    if isinstance(v, dict):
        return {k: _normalize(v[k]) for k in sorted(v.keys(), key=str)}
    if isinstance(v, (list, tuple)):
        return [_normalize(x) for x in v]
    if isinstance(v, (bytes, bytearray)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(v)).hexdigest()}
    if isinstance(v, Path):
        return str(v)
    return repr(v)


def _block_fingerprint(b: ContentBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ImageBlock):
        return {
            "type": "image",
            "media_type": b.media_type,
            "data_sha256": hashlib.sha256(b.data).hexdigest(),
        }
    if isinstance(b, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": b.id,
            "name": b.name,
            "input": _normalize(b.input),
        }
    if isinstance(b, ToolResultBlock):
        if isinstance(b.content, str):
            content: Any = b.content
        else:
            content = [_block_fingerprint(c) for c in b.content]
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": content,
            "is_error": b.is_error,
            "name": b.name,
        }
    if isinstance(b, ThinkingBlock):
        return {
            "type": "thinking",
            "text": b.text,
            "signature": b.signature,
            "encrypted": b.encrypted,
            "provider_data": _normalize(b.provider_data) if b.provider_data else None,
        }
    raise TypeError(f"Cannot fingerprint content block: {type(b).__name__}")  # pragma: no cover


def _message_fingerprint(m: Message) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}
    return {"role": m.role, "content": [_block_fingerprint(b) for b in m.content]}


def _tool_fingerprint(t: Tool) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "schema": _normalize(t.schema),
    }


def fingerprint_request(req: CompletionRequest, provider_name: str) -> dict[str, Any]:
    """Build a canonical dict describing every input that affects output.

    Tool order is preserved (some providers / models bias on tool ordering).
    Settings keys are sorted by ``_normalize``. Image bytes are reduced to a
    sha256 hex digest so the fingerprint stays small but is still uniquely
    determined by the original bytes."""
    return {
        "provider": provider_name,
        "model": req.model,
        "system_blocks": [{"text": sb.text, "cache": sb.cache} for sb in req.system_blocks],
        "messages": [_message_fingerprint(m) for m in req.messages],
        "tools": [_tool_fingerprint(t) for t in req.tools],
        "stop": list(req.stop) if req.stop else None,
        "settings": _normalize(req.settings),
    }


def hash_fingerprint(fp: dict[str, Any]) -> str:
    canonical = _normalize(fp)
    body = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


# --- Response (de)serialisation -----------------------------------------


def _block_to_dict(b: ContentBlock) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ImageBlock):
        return {
            "type": "image",
            "media_type": b.media_type,
            "data_b64": base64.b64encode(b.data).decode("ascii"),
        }
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        if isinstance(b.content, str):
            content: Any = b.content
        else:
            content = [_block_to_dict(c) for c in b.content]
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": content,
            "is_error": b.is_error,
            "name": b.name,
        }
    if isinstance(b, ThinkingBlock):
        return {
            "type": "thinking",
            "text": b.text,
            "signature": b.signature,
            "encrypted": b.encrypted,
            "provider_data": _normalize(b.provider_data) if b.provider_data else None,
        }
    raise TypeError(f"Cannot serialise content block: {type(b).__name__}")


def _dict_to_block(d: dict[str, Any]) -> ContentBlock:
    t = d["type"]
    if t == "text":
        return TextBlock(text=d["text"])
    if t == "image":
        return ImageBlock(
            data=base64.b64decode(d["data_b64"]),
            media_type=d["media_type"],
        )
    if t == "tool_use":
        return ToolUseBlock(id=d["id"], name=d["name"], input=d["input"])
    if t == "tool_result":
        raw = d["content"]
        content: str | list[TextBlock | ImageBlock]
        if isinstance(raw, list):
            decoded: list[TextBlock | ImageBlock] = []
            for c in raw:
                block = _dict_to_block(c)
                if not isinstance(block, (TextBlock, ImageBlock)):
                    raise TypeError(
                        f"ToolResultBlock content may only contain text/image blocks, "
                        f"got {type(block).__name__}"
                    )
                decoded.append(block)
            content = decoded
        else:
            content = raw
        return ToolResultBlock(
            tool_use_id=d["tool_use_id"],
            content=content,
            is_error=d.get("is_error", False),
            name=d.get("name"),
        )
    if t == "thinking":
        return ThinkingBlock(
            text=d["text"],
            signature=d.get("signature"),
            encrypted=d.get("encrypted", False),
            provider_data=d.get("provider_data"),
        )
    raise ValueError(f"Unknown block type {t!r} in cached response")


def _serialize_response(resp: Response) -> dict[str, Any]:
    return {
        "text": resp.text,
        "blocks": [_block_to_dict(b) for b in resp.blocks],
        "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls],
        "thinking": resp.thinking,
        "usage": (
            None
            if resp.usage is None
            else {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
                "cache_creation_tokens": resp.usage.cache_creation_tokens,
                "cache_read_tokens": resp.usage.cache_read_tokens,
            }
        ),
        "finish_reason": resp.finish_reason,
        "model": resp.model,
    }


def _deserialize_response(d: dict[str, Any]) -> Response:
    return Response(
        text=d["text"],
        blocks=[_dict_to_block(b) for b in d["blocks"]],
        tool_calls=[
            ToolCall(id=c["id"], name=c["name"], input=c["input"]) for c in d["tool_calls"]
        ],
        thinking=d.get("thinking"),
        usage=(None if d.get("usage") is None else Usage(**d["usage"])),
        finish_reason=d.get("finish_reason"),
        model=d["model"],
        raw=None,
    )


# --- Stream replay -------------------------------------------------------


def replay_stream(resp: Response) -> Iterator[StreamEvent]:
    """Synthesise a stream from a cached ``Response``.

    Order: thinking blocks first (matches the canonical ordering Anthropic /
    Gemini expect on the wire), then a single ``text_delta`` carrying the
    full text, then one event per tool call, then a terminal ``done`` event
    with ``usage`` and ``finish_reason``."""
    for block in resp.blocks:
        if isinstance(block, ThinkingBlock):
            if block.text:
                yield StreamEvent(thinking_delta=block.text)
            yield StreamEvent(thinking_block=block)
    if resp.text:
        yield StreamEvent(text_delta=resp.text)
    for tc in resp.tool_calls:
        yield StreamEvent(tool_call_delta=tc)
    yield StreamEvent(done=True, usage=resp.usage, finish_reason=resp.finish_reason)


async def areplay_stream(resp: Response) -> AsyncIterator[StreamEvent]:
    for ev in replay_stream(resp):
        yield ev


# --- Cache backend -------------------------------------------------------


def _safe_segment(s: str) -> str:
    """Make ``s`` safe to use as a single filesystem path segment."""
    return (
        s.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


class ResponseCache:
    """Filesystem-backed cache of ``Response`` objects keyed by request fingerprint.

    Layout: ``<root>/<provider>/<safe_model_id>/<sha256>.json``. Each entry
    stores both the canonical fingerprint (for debugging / inspection) and
    the serialised response."""

    def __init__(self, root: Path, mode: str):
        if mode not in CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {sorted(CACHE_MODES)}, got {mode!r}")
        self.root = Path(root)
        self.mode = mode

    @property
    def reads(self) -> bool:
        return self.mode in {"read_write", "read_only", "replay_only"}

    @property
    def writes(self) -> bool:
        return self.mode in {"read_write", "record_only"}

    def path_for(self, provider: str, model_id: str, key: str) -> Path:
        return self.root / _safe_segment(provider) / _safe_segment(model_id) / f"{key}.json"

    def get(self, provider: str, model_id: str, key: str) -> Response | None:
        if not self.reads:
            return None
        path = self.path_for(provider, model_id, key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return _deserialize_response(data["response"])

    def put(
        self,
        provider: str,
        model_id: str,
        key: str,
        resp: Response,
        fingerprint: dict[str, Any],
    ) -> None:
        if not self.writes:
            return
        path = self.path_for(provider, model_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": fingerprint,
            "response": _serialize_response(resp),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp.replace(path)


def resolve_cache(
    *,
    convo_cache_dir: Any,
    convo_cache_mode: Any,
    model: Any,
) -> ResponseCache | None:
    """Apply the convo > model > provider cascade and return a ``ResponseCache``.

    Returns ``None`` if no layer supplies a path or the topmost layer with
    an opinion supplies ``False`` (explicit disable)."""
    dir_layers = (
        convo_cache_dir,
        getattr(model, "_cache_dir_override", None),
        getattr(model.provider, "_cache_dir_override", None),
    )
    cache_dir: Path | None = None
    for v in dir_layers:
        if v is False:
            return None
        if v is not None:
            cache_dir = Path(v)
            break
    if cache_dir is None:
        return None

    mode_layers = (
        convo_cache_mode,
        getattr(model, "_cache_mode_override", None),
        getattr(model.provider, "_cache_mode_override", None),
    )
    mode = "read_write"
    for v in mode_layers:
        if v is not None:
            mode = v
            break
    return ResponseCache(cache_dir, mode)


__all__ = [
    "CACHE_MODES",
    "ResponseCache",
    "fingerprint_request",
    "hash_fingerprint",
    "replay_stream",
    "areplay_stream",
    "resolve_cache",
]
