from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Union

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ImageBlock:
    data: bytes
    media_type: str

    @classmethod
    def from_path(cls, path: str | Path) -> ImageBlock:
        p = Path(path)
        guess, _ = mimetypes.guess_type(p.name)
        media_type = guess or "image/png"
        return cls(data=p.read_bytes(), media_type=media_type)

    @classmethod
    def from_base64(cls, b64: str, media_type: str) -> ImageBlock:
        return cls(data=base64.b64decode(b64), media_type=media_type)

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """A reasoning / chain-of-thought block returned by a model.

    ``text`` is the human-readable reasoning. ``signature`` is an opaque
    integrity token that some providers require be returned verbatim in
    subsequent turns when tools are in use (Anthropic ``signature``, Gemini
    ``thoughtSignature``). ``encrypted=True`` covers Anthropic's
    ``redacted_thinking`` and OpenAI's ``encrypted_content`` — the visible
    ``text`` will be empty and the opaque payload lives in ``provider_data``.
    ``provider_data`` is a passthrough for any other per-provider fields
    (e.g. OpenAI reasoning item id) so the block can be round-tripped
    losslessly."""

    text: str
    signature: str | None = None
    encrypted: bool = False
    provider_data: dict[str, Any] | None = None


ContentBlock = Union[  # noqa: UP007
    TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str | list[ContentBlock]


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    # Tokens spent on reasoning / chain-of-thought, a subset of the output the
    # model produced. Reported separately by providers that expose it (OpenAI
    # ``completion_tokens_details.reasoning_tokens``, Google
    # ``thoughts_token_count``); ``0`` when the provider folds reasoning into
    # ``completion_tokens`` without a breakdown (Anthropic, most llama.cpp
    # builds). The conversation log falls back to a local tokenizer count of
    # the reasoning text in that case.
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    _fn: Callable[..., Any] | None = field(default=None, repr=False, compare=False)

    def invoke(self) -> Any:
        if self._fn is None:
            raise RuntimeError(
                f"ToolCall {self.name!r} has no bound function; "
                "register the tool via Conversation.AddTool() before invoking."
            )
        return self._fn(**self.input)

    async def ainvoke(self) -> Any:
        import inspect

        if self._fn is None:
            raise RuntimeError(
                f"ToolCall {self.name!r} has no bound function; "
                "register the tool via Conversation.AddTool() before invoking."
            )
        result = self._fn(**self.input)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass(frozen=True, slots=True)
class Response:
    text: str
    blocks: list[ContentBlock]
    tool_calls: list[ToolCall]
    thinking: str | None
    usage: Usage | None
    finish_reason: str | None
    model: str
    raw: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    text_delta: str | None = None
    tool_call_delta: ToolCall | None = None
    thinking_delta: str | None = None
    thinking_block: ThinkingBlock | None = None
    done: bool = False
    usage: Usage | None = None
    finish_reason: str | None = None


_EXT_BY_MEDIA_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


@dataclass(frozen=True, slots=True)
class ImageUsage:
    """Usage reported by an image-generation call. ``image_count`` is always
    set; the token fields are populated only where the provider breaks them out
    (OpenAI ``gpt-image-*``, Gemini-native ``usage_metadata``) and are ``0``
    otherwise. No provider returns a dollar figure — derive cost from the token
    counts and the model's published pricing."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    image_count: int = 0


@dataclass(frozen=True, slots=True)
class ImageResult:
    """Result of a ``generate_image`` call. ``images`` carries the generated
    image bytes as :class:`ImageBlock` instances (so they round-trip straight
    back into a vision request). ``paths`` is empty unless ``save_dir=`` was
    passed (or :meth:`save` was called), in which case it lists the written
    files in image order."""

    images: list[ImageBlock]
    usage: ImageUsage | None
    model: str
    provider: str
    paths: list[Path] = field(default_factory=list)
    raw: object = field(default=None, repr=False, compare=False)

    def save(self, dest: str | Path, *, prefix: str = "image") -> list[Path]:
        """Write each image into directory ``dest`` as ``<prefix>_<i><ext>``
        (extension derived from the block's ``media_type``, defaulting to
        ``.png``). Creates ``dest`` if needed. Returns the written paths."""
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, block in enumerate(self.images):
            ext = _EXT_BY_MEDIA_TYPE.get(block.media_type, ".png")
            path = d / f"{prefix}_{i}{ext}"
            path.write_bytes(block.data)
            written.append(path)
        return written


def _apply_save_dir(result: ImageResult, save_dir: str | Path | None) -> ImageResult:
    """If ``save_dir`` is set, write ``result``'s images there and return a copy
    with ``paths`` populated; otherwise return ``result`` unchanged. Shared by
    the image-generating providers so ``save_dir=`` behaves identically."""
    if save_dir is None:
        return result
    written = result.save(save_dir)
    return replace(result, paths=written)
