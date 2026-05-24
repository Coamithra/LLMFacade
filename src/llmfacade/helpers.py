"""Convenience helpers built on top of the public Conversation API.

These functions are not part of the core lifecycle. They exist so common
patterns (running every tool call the model produced, then sending the
results back until the model stops calling tools) don't have to be hand-
rolled by every caller — but they are *just* loops over ``send`` /
``add_tool_result``, with no privileged access to internals."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import ToolIterationLimitError
from llmfacade.models import (
    ContentBlock,
    ImageBlock,
    Message,
    Response,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)

if TYPE_CHECKING:
    from llmfacade.conversation import Conversation


def flatten_text_blocks(blocks: list[Any]) -> str:
    """Concatenate the ``.text`` of every TextBlock in ``blocks``. Non-text blocks ignored."""
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


_DEFERRED_IMAGE_NOTE = (
    "Here is the image output from the tool call(s) above (this model cannot "
    "receive images inside a tool result, so they are attached here):"
)


def _normalize_tool_result(result: Any) -> str | list[ContentBlock]:
    """Coerce a tool function's return into tool-result content.

    A plain ``str`` passes through; any other scalar keeps the historical
    ``_stringify`` (JSON/str) behaviour. An ``ImageBlock`` — or a non-empty list
    in which **every** element is a ``TextBlock``/``ImageBlock`` — is preserved
    so a tool can return an image. A mixed list (blocks plus other values) is
    stringified whole, same as any other non-str return."""
    if isinstance(result, str):
        return result
    if isinstance(result, ImageBlock):
        return [result]
    if (
        isinstance(result, list)
        and result
        and all(isinstance(b, TextBlock | ImageBlock) for b in result)
    ):
        return list(result)
    return _stringify(result)


def _as_result_content(content: str | list[ContentBlock]) -> str | list[TextBlock | ImageBlock]:
    """Narrow normalised content to the block types a ``ToolResultBlock`` holds."""
    if isinstance(content, str):
        return content
    return [b for b in content if isinstance(b, TextBlock | ImageBlock)]


def _append_tool_result(
    convo: Conversation,
    call: ToolCall,
    content: str | list[ContentBlock],
    deferred_images: list[ImageBlock],
) -> ToolResultBlock:
    """Append a tool result for ``call`` and return the block that was stored.

    If the content carries images and the model declares ``"tool_result_images"``
    the image rides in the tool result. Otherwise the result is reduced to its
    text and, when the model has ``"vision"``, the image(s) are queued in
    ``deferred_images`` for a single follow-up user message (emitted after the
    whole batch so every ``tool_use`` stays paired with its ``tool_result``). A
    model with neither capability can't be shown the image at all, so it is
    dropped — the next ``send`` would otherwise raise on the deferred message."""
    result_content = _as_result_content(content)
    if isinstance(result_content, str):
        images: list[ImageBlock] = []
    else:
        images = [b for b in result_content if isinstance(b, ImageBlock)]
    if not images or convo.is_available("tool_result_images"):
        convo.add_tool_result(call.id, content, name=call.name)
        return ToolResultBlock(tool_use_id=call.id, content=result_content, name=call.name)
    base_text = flatten_text_blocks(result_content) if isinstance(result_content, list) else ""
    if convo.is_available("vision"):
        text = base_text or "(tool returned image output; see the next message)"
        deferred_images.extend(images)
    else:
        text = base_text or "(tool returned image output, omitted: model can't receive images)"
    convo.add_tool_result(call.id, text, name=call.name)
    return ToolResultBlock(tool_use_id=call.id, content=text, name=call.name)


def _flush_deferred_images(convo: Conversation, deferred_images: list[ImageBlock]) -> None:
    if deferred_images:
        convo.add_user_message(content=[TextBlock(_DEFERRED_IMAGE_NOTE), *deferred_images])


def run_bound_tools(convo: Conversation, resp: Response) -> list[ToolResultBlock]:
    """Run every tool call in ``resp`` whose name matches a tool registered on ``convo``.

    For each match, calls the function with the model's arguments and appends
    a ToolResultBlock to the conversation via ``add_tool_result``. Tool calls
    whose name is *not* registered are skipped — the caller is then responsible
    for handling them (or letting the next ``send`` raise ``ConversationStateError``).

    A tool may return an ``ImageBlock`` (or a ``[TextBlock, ImageBlock, ...]``
    list). If the model declares ``"tool_result_images"`` the image rides in the
    tool result; otherwise the tool result is reduced to text and the image(s)
    are appended as a single follow-up user message after the batch.

    Returns the list of ToolResultBlocks that were appended (the text-only block
    in the fallback case)."""
    out: list[ToolResultBlock] = []
    deferred_images: list[ImageBlock] = []
    for call in resp.tool_calls:
        tool_def = convo.tool(call.name)
        if tool_def is None:
            continue
        try:
            content = _normalize_tool_result(tool_def.fn(**call.input))
            out.append(_append_tool_result(convo, call, content, deferred_images))
        except Exception as e:
            err = f"Tool error: {e}"
            convo.add_tool_result(call.id, err, is_error=True, name=call.name)
            out.append(
                ToolResultBlock(tool_use_id=call.id, content=err, is_error=True, name=call.name)
            )
    _flush_deferred_images(convo, deferred_images)
    return out


async def arun_bound_tools(convo: Conversation, resp: Response) -> list[ToolResultBlock]:
    """Async equivalent of ``run_bound_tools``. Awaits coroutine-returning tool fns.

    Handles tool-returned images the same way: embedded in the tool result when
    the model declares ``"tool_result_images"``, otherwise reduced to text with
    the image(s) appended as one follow-up user message after the batch."""
    out: list[ToolResultBlock] = []
    deferred_images: list[ImageBlock] = []
    for call in resp.tool_calls:
        tool_def = convo.tool(call.name)
        if tool_def is None:
            continue
        try:
            result = tool_def.fn(**call.input)
            if inspect.isawaitable(result):
                result = await result
            content = _normalize_tool_result(result)
            out.append(_append_tool_result(convo, call, content, deferred_images))
        except Exception as e:
            err = f"Tool error: {e}"
            convo.add_tool_result(call.id, err, is_error=True, name=call.name)
            out.append(
                ToolResultBlock(tool_use_id=call.id, content=err, is_error=True, name=call.name)
            )
    _flush_deferred_images(convo, deferred_images)
    return out


def run_to_completion(
    convo: Conversation,
    prompt: Any = None,
    *,
    max_iterations: int = 16,
    **send_kwargs: Any,
) -> Response:
    """Send ``prompt``, then dispatch any bound tool calls and continue sending
    until the model returns a response with no tool calls (or ``max_iterations``
    is hit). Raises ``ToolIterationLimitError`` if the loop doesn't terminate."""
    resp = convo.send(prompt, **send_kwargs)
    for _ in range(max_iterations):
        if not resp.tool_calls:
            return resp
        run_bound_tools(convo, resp)
        resp = convo.send(**send_kwargs)
    raise ToolIterationLimitError(
        f"run_to_completion exceeded max_iterations={max_iterations}. "
        f"The model kept calling tools without producing a final answer."
    )


async def arun_to_completion(
    convo: Conversation,
    prompt: Any = None,
    *,
    max_iterations: int = 16,
    **send_kwargs: Any,
) -> Response:
    """Async equivalent of ``run_to_completion``."""
    resp = await convo.asend(prompt, **send_kwargs)
    for _ in range(max_iterations):
        if not resp.tool_calls:
            return resp
        await arun_bound_tools(convo, resp)
        resp = await convo.asend(**send_kwargs)
    raise ToolIterationLimitError(
        f"arun_to_completion exceeded max_iterations={max_iterations}. "
        f"The model kept calling tools without producing a final answer."
    )


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


def _abbreviate_text(text: str, max_lines: int | None) -> str:
    if max_lines is None or max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = max_lines // 2
    tail = max_lines - head
    elided = len(lines) - head - tail
    return (
        "\n".join(lines[:head])
        + f"\n... [{elided} lines skipped] ...\n"
        + "\n".join(lines[-tail:])
    )


def _dump_message(m: Message, *, max_lines: int | None = None) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": m.role, "content": _abbreviate_text(m.content, max_lines)}
    return {
        "role": m.role,
        "content": [_dump_block(b, max_lines=max_lines) for b in m.content],
    }


def _dump_block(b: ContentBlock, *, max_lines: int | None = None) -> dict[str, Any]:
    cls = type(b).__name__
    if isinstance(b, TextBlock):
        return {"type": cls, "text": _abbreviate_text(b.text, max_lines)}
    from llmfacade.models import ToolUseBlock

    if isinstance(b, ToolUseBlock):
        return {"type": cls, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {"type": cls, "tool_use_id": b.tool_use_id, "is_error": b.is_error}
    if isinstance(b, ImageBlock):
        return {"type": cls, "media_type": b.media_type, "bytes": len(b.data)}
    if isinstance(b, ThinkingBlock):
        return {
            "type": cls,
            "text": _abbreviate_text(b.text, max_lines),
            "encrypted": b.encrypted,
            "signature": "<present>" if b.signature else None,
        }
    return {"type": cls}


def _dump_usage(u: Usage | None) -> dict[str, int] | None:
    if u is None:
        return None
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "cache_creation_tokens": u.cache_creation_tokens,
        "cache_read_tokens": u.cache_read_tokens,
    }


def _log_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return str(obj)
