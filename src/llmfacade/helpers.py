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
    ToolResultBlock,
    Usage,
)

if TYPE_CHECKING:
    from llmfacade.conversation import Conversation


def flatten_text_blocks(blocks: list[Any]) -> str:
    """Concatenate the ``.text`` of every TextBlock in ``blocks``. Non-text blocks ignored."""
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def run_bound_tools(convo: Conversation, resp: Response) -> list[ToolResultBlock]:
    """Run every tool call in ``resp`` whose name matches a tool registered on ``convo``.

    For each match, calls the function with the model's arguments and appends
    a ToolResultBlock to the conversation via ``add_tool_result``. Tool calls
    whose name is *not* registered are skipped — the caller is then responsible
    for handling them (or letting the next ``send`` raise ``ConversationStateError``).

    Returns the list of ToolResultBlocks that were appended."""
    out: list[ToolResultBlock] = []
    for call in resp.tool_calls:
        tool_def = convo.tool(call.name)
        if tool_def is None:
            continue
        try:
            result = tool_def.fn(**call.input)
            content = _stringify(result)
            convo.add_tool_result(call.id, content, name=call.name)
            out.append(ToolResultBlock(tool_use_id=call.id, content=content, name=call.name))
        except Exception as e:
            content = f"Tool error: {e}"
            convo.add_tool_result(call.id, content, is_error=True, name=call.name)
            out.append(
                ToolResultBlock(
                    tool_use_id=call.id, content=content, is_error=True, name=call.name
                )
            )
    return out


async def arun_bound_tools(convo: Conversation, resp: Response) -> list[ToolResultBlock]:
    """Async equivalent of ``run_bound_tools``. Awaits coroutine-returning tool fns."""
    out: list[ToolResultBlock] = []
    for call in resp.tool_calls:
        tool_def = convo.tool(call.name)
        if tool_def is None:
            continue
        try:
            result = tool_def.fn(**call.input)
            if inspect.isawaitable(result):
                result = await result
            content = _stringify(result)
            convo.add_tool_result(call.id, content, name=call.name)
            out.append(ToolResultBlock(tool_use_id=call.id, content=content, name=call.name))
        except Exception as e:
            content = f"Tool error: {e}"
            convo.add_tool_result(call.id, content, is_error=True, name=call.name)
            out.append(
                ToolResultBlock(
                    tool_use_id=call.id, content=content, is_error=True, name=call.name
                )
            )
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
