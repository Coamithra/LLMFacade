"""Shared helpers for the OpenAI *Chat Completions* surface.

Both the hosted OpenAI provider and the llamacpp provider speak the OpenAI
Chat Completions wire format via the ``openai`` Python SDK — the same reuse
trick ``_openai_images.py`` plays for the images surface. These functions
marshal facade messages/tools into SDK kwargs and parse responses / stream
chunks back into facade types, so the per-provider modules only own what
genuinely differs: llamacpp's ``reasoning_content`` capture (leading
``ThinkingBlock`` / ``thinking_delta`` events) and ``extra_body`` sampler
routing; OpenAI's ``max_completion_tokens`` / ``effort`` / structured-output
mapping and cached-prompt-token accounting.
"""

from __future__ import annotations

import json as _json
import warnings
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import ProviderError
from llmfacade.helpers import flatten_text_blocks
from llmfacade.models import (
    ContentBlock,
    ImageBlock,
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolArgsDelta,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from llmfacade.provider import CompletionRequest


def cached_prompt_tokens(usage: Any) -> int:
    """Pull cached prompt-token count from OpenAI-shaped usage. Lives in
    ``prompt_tokens_details.cached_tokens`` on chat-completion responses."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "cached_tokens", 0) or 0


def reasoning_tokens_from_details(usage: Any) -> int:
    """Pull reasoning-token count from OpenAI-shaped usage. Lives in
    ``completion_tokens_details.reasoning_tokens`` — populated by OpenAI's
    reasoning models (o-series, GPT-5) and by llama-server builds that break
    reasoning out; absent (→ 0) otherwise."""
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "reasoning_tokens", 0) or 0


def usage_from_chat(usage: Any, *, include_cached: bool) -> Usage:
    """Build a facade :class:`Usage` from OpenAI-shaped usage. ``include_cached``
    reads ``prompt_tokens_details.cached_tokens`` into ``cache_read_tokens``
    (hosted OpenAI); llamacpp passes ``False`` — llama-server's prompt cache is
    internal and never reported per-request."""
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cache_read_tokens=cached_prompt_tokens(usage) if include_cached else 0,
        reasoning_tokens=reasoning_tokens_from_details(usage),
    )


def empty_choices_detail(raw: Any) -> str:
    """Describe a 200 response that carried no choices — a real occurrence on
    OpenAI-compat proxies and content-filter paths. Finish/filter info lives on
    the (absent) choices, so surface what the top-level object still carries:
    the model id and any prompt-level filter results (Azure convention)."""
    parts = []
    model = getattr(raw, "model", None)
    if model:
        parts.append(f"model={model!r}")
    filter_results = getattr(raw, "prompt_filter_results", None)
    if filter_results:
        parts.append(f"prompt_filter_results={filter_results!r}")
    return f" ({', '.join(parts)})" if parts else ""


def parse_tool_arguments(raw_args: Any) -> tuple[dict[str, Any], str | None]:
    """Parse a tool call's JSON arguments string into ``(input, raw_arguments)``.
    On a truncated/malformed string (e.g. the model hit the token limit
    mid-JSON), ``input`` is ``{}`` and ``raw_arguments`` keeps the verbatim
    string so the failed call is still visible in logs instead of collapsing
    to an empty dict."""
    try:
        return _json.loads(raw_args), None
    except _json.JSONDecodeError:
        return {}, raw_args


def message_to_api(m: Message, *, provider_label: str) -> list[dict[str, Any]]:
    """Marshal one facade :class:`Message` into Chat Completions message dicts.

    ``ThinkingBlock``s are dropped on the way out — Chat Completions can't
    round-trip reasoning content (OpenAI's Responses API can but isn't wired
    up; llama-server has no canonical thinking-block input format).
    ``ImageBlock``s are emitted as data-URL ``image_url`` parts on user
    messages and dropped with a warning on any other role (the API only
    accepts images on user messages)."""
    if m.role == "tool":
        results: list[dict[str, Any]] = []
        blocks = m.content if isinstance(m.content, list) else []
        for b in blocks:
            if isinstance(b, ToolResultBlock):
                text = b.content if isinstance(b.content, str) else flatten_text_blocks(b.content)
                results.append(
                    {
                        "role": "tool",
                        "content": text,
                        "tool_call_id": b.tool_use_id,
                    }
                )
        return results

    if isinstance(m.content, str):
        return [{"role": m.role, "content": m.content}]

    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    has_image = False
    for b in m.content:
        if isinstance(b, TextBlock):
            parts.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            has_image = True
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{b.media_type};base64,{b.to_base64()}",
                    },
                }
            )
        elif isinstance(b, ToolUseBlock):
            tool_calls.append(
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": _json.dumps(b.input)},
                }
            )
        elif isinstance(b, ThinkingBlock):
            continue
    out: dict[str, Any] = {"role": m.role}
    if parts:
        if m.role == "user":
            out["content"] = parts
        else:
            if has_image:
                # stacklevel walks out of the shared helper to the provider's
                # request-entry frame: message_to_api -> provider._message_to_api
                # -> chat_messages -> _build_kwargs -> _complete_raw/_stream_raw.
                warnings.warn(
                    f"{provider_label}: dropping image block(s) on {m.role!r} "
                    f"message; the Chat Completions API only accepts images on "
                    f"user messages.",
                    stacklevel=6,
                )
            out["content"] = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    else:
        out["content"] = None if tool_calls else ""
    if tool_calls:
        out["tool_calls"] = tool_calls
    return [out]


def chat_messages(
    req: CompletionRequest,
    message_to_api_fn: Callable[[Message], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build the full ``messages`` array: system blocks joined into a single
    leading system message, then each history message marshaled via the
    provider's ``_message_to_api`` (kept injectable so per-provider overrides
    and the directly-tested method surface stay authoritative)."""
    api_msgs: list[dict[str, Any]] = []
    if req.system_blocks:
        api_msgs.append(
            {
                "role": "system",
                "content": "\n\n".join(sb.text for sb in req.system_blocks),
            }
        )
    for m in req.messages:
        api_msgs.extend(message_to_api_fn(m))
    return api_msgs


def tool_to_api(t: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema,
        },
    }


def tool_choice_to_api(tc: str) -> str | dict[str, Any]:
    if tc in ("auto", "required", "none"):
        return tc
    return {"type": "function", "function": {"name": tc}}


def apply_tools(api_kwargs: dict[str, Any], req: CompletionRequest) -> None:
    """Attach ``tools`` + ``tool_choice`` to the request kwargs (omitted
    entirely when the request carries no tools)."""
    if req.tools:
        api_kwargs["tools"] = [tool_to_api(t) for t in req.tools]
        api_kwargs["tool_choice"] = tool_choice_to_api(req.settings.get("tool_choice", "auto"))


def tool_fragment_events(
    delta: Any,
    tool_buf: dict[int, dict[str, Any]],
) -> Iterator[StreamEvent]:
    """Accumulate streamed tool-call deltas into per-index slots and forward
    each raw arguments fragment as a ``tool_args_delta`` event the moment it
    arrives (before the JSON is complete or valid). Ordering contract: for a
    given tool call, zero-or-more of these fragments precede exactly one
    terminal ``tool_call_delta`` (from :func:`flush_tool_call_events`), and
    concatenating the fragments reconstructs the exact raw arguments string."""
    for tc in getattr(delta, "tool_calls", None) or []:
        idx = getattr(tc, "index", 0)
        slot = tool_buf.setdefault(idx, {"id": None, "name": None, "args": ""})
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["args"] += fn.arguments
                yield StreamEvent(
                    tool_args_delta=ToolArgsDelta(
                        index=idx,
                        fragment=fn.arguments,
                        id=slot["id"],
                        name=slot["name"],
                    )
                )


def flush_tool_call_events(tool_buf: dict[int, dict[str, Any]]) -> Iterator[StreamEvent]:
    """Emit the terminal ``tool_call_delta`` for every accumulated slot (with
    JSON-args recovery: a malformed final string lands in ``raw_arguments``)
    and clear the buffer. Called on ``finish_reason``."""
    for slot in tool_buf.values():
        if slot["id"] is None:
            continue
        parsed, unparsed = parse_tool_arguments(slot["args"] or "{}")
        yield StreamEvent(
            tool_call_delta=ToolCall(
                id=slot["id"],
                name=slot["name"] or "",
                input=parsed,
                raw_arguments=unparsed,
            )
        )
    tool_buf.clear()


def parse_chat_response(
    raw: Any,
    *,
    server_label: str,
    include_cached_tokens: bool,
    reasoning_text: Callable[[Any], str] | None = None,
) -> Response:
    """Parse a non-streaming Chat Completions response into a facade
    :class:`Response`.

    ``server_label`` names the responding party in the empty-choices error
    (``"OpenAI"`` / ``"llama-server"``). ``reasoning_text`` is llamacpp's hook
    for extracting ``reasoning_content`` off the message — when it yields text,
    a ``ThinkingBlock`` leads the assistant turn (the canonical thinking-then-
    text ordering the rest of the facade assumes) and ``Response.thinking`` is
    set; OpenAI passes ``None`` (Chat Completions never carries reasoning)."""
    choices = getattr(raw, "choices", None) or []
    if not choices:
        raise ProviderError(
            f"{server_label} returned a response with no choices"
            f"{empty_choices_detail(raw)}; nothing to parse. This can "
            "happen on OpenAI-compat proxies and content-filter paths."
        )
    choice = choices[0]
    msg = choice.message
    reasoning = reasoning_text(msg) if reasoning_text is not None else ""
    text = getattr(msg, "content", "") or ""
    blocks: list[ContentBlock] = []
    if reasoning:
        blocks.append(ThinkingBlock(text=reasoning))
    if text:
        blocks.append(TextBlock(text))
    tool_calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        args, unparsed = parse_tool_arguments(tc.function.arguments)
        blocks.append(
            ToolUseBlock(id=tc.id, name=tc.function.name, input=args, raw_arguments=unparsed)
        )
        tool_calls.append(
            ToolCall(id=tc.id, name=tc.function.name, input=args, raw_arguments=unparsed)
        )

    usage = None
    if raw.usage:
        usage = usage_from_chat(raw.usage, include_cached=include_cached_tokens)

    return Response(
        text=text,
        blocks=blocks,
        tool_calls=tool_calls,
        thinking=reasoning or None,
        usage=usage,
        finish_reason=choice.finish_reason,
        model=raw.model,
        raw=raw,
    )
