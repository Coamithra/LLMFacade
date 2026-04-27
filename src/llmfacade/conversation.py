from __future__ import annotations

import copy
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade._html_log import HtmlLogger
from llmfacade.exceptions import ConversationStateError, UnsupportedFeature
from llmfacade.helpers import _abbreviate_text, _dump_message, _dump_usage, _log_default
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
)
from llmfacade.provider import (
    CompletionRequest,
    SystemBlock,
    _filter_unsupported,
    _validate_knobs,
)
from llmfacade.tools import Tool

if TYPE_CHECKING:
    from llmfacade.model import Model


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Opaque snapshot token for ``Conversation.rollback()``."""

    history: tuple[Message, ...]
    turn_boundaries: tuple[tuple[int, int], ...] = ()


def _render_message_oneline(m: Message) -> str:
    if isinstance(m.content, str):
        return f"[{m.role}] {m.content}"
    parts: list[str] = []
    for block in m.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(f"<image {block.media_type} {len(block.data)}B>")
        elif isinstance(block, ToolUseBlock):
            parts.append(f"<tool_use {block.name} id={block.id}>")
        elif isinstance(block, ToolResultBlock):
            parts.append(f"<tool_result for={block.tool_use_id}>")
        elif isinstance(block, ThinkingBlock):
            tag = "redacted_thinking" if block.encrypted else "thinking"
            sig = "+sig" if block.signature else ""
            parts.append(f"<{tag} {len(block.text)}c{sig}>")
        else:
            parts.append(f"<{type(block).__name__}>")
    return f"[{m.role}] " + " ".join(parts)


def _message_to_text(m: Message) -> str:
    if isinstance(m.content, str):
        return m.content
    out: list[str] = []
    for block in m.content:
        if isinstance(block, TextBlock):
            out.append(block.text)
        elif isinstance(block, ThinkingBlock):
            # Thinking content is sent back over the wire and counts toward
            # input tokens, so include it in the cache-boundary estimate.
            out.append(block.text)
        elif isinstance(block, ToolUseBlock):
            out.append(block.name)
            out.append(json.dumps(block.input, default=str))
        elif isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                out.append(block.content)
            else:
                out.extend(b.text for b in block.content if isinstance(b, TextBlock))
    return "\n".join(out)


def _tokenizer_label(provider: Any, model_id: str) -> str:
    return provider.tokenizer_name(model_id=model_id)


def _abbreviate_lines(text: str, *, head: int = 3, tail: int = 3) -> str:
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text
    elided = len(lines) - head - tail
    elided_chars = sum(len(line) + 1 for line in lines[head:-tail])
    return (
        "\n".join(lines[:head])
        + f"\n... ({elided} lines, ~{elided_chars} chars elided) ...\n"
        + "\n".join(lines[-tail:])
    )


def _coerce_system_blocks(
    raw: list[SystemBlock | str] | None,
    supports_cache: bool,
    provider: str,
    model: str,
) -> list[SystemBlock]:
    if not raw:
        return []
    out: list[SystemBlock] = []
    for sb in raw:
        if isinstance(sb, str):
            out.append(SystemBlock(text=sb, cache=False))
        else:
            if sb.cache and not supports_cache:
                raise UnsupportedFeature("system_block_cache", provider, model)
            out.append(sb)
    return out


class Conversation:
    """A stateful chat session against one Model.

    Identity, system blocks, tools, logging path, and generation defaults are
    all set at construction. There is no ``Start()`` step — the conversation
    is usable immediately. Mutating state (history) lives on the conversation;
    configuration is immutable post-construction. To get a fresh configuration,
    build a new conversation."""

    def __init__(
        self,
        *,
        model: Model,
        name: str | None = None,
        system_blocks: list[SystemBlock | str] | None = None,
        tools: list[Tool] | None = None,
        log_path: str | Path | None = None,
        log_max_message_lines: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
        tool_choice: str | None = None,
    ):
        self._model = model
        self.name = name or f"convo-{uuid.uuid4().hex[:8]}"

        self._system_blocks = _coerce_system_blocks(
            system_blocks,
            supports_cache=model.is_available("auto_cache_last_user"),
            provider=model.provider.NAME,
            model=model.model_id,
        )
        if tools and not model.is_available("tools"):
            raise UnsupportedFeature("tools", model.provider.NAME, model.model_id)
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

        self._defaults = _validate_knobs(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "effort": effort,
                "thinking": thinking,
                "output_format": output_format,
                "user_metadata": user_metadata,
                "cache_ttl": cache_ttl,
                "auto_cache_last_user": auto_cache_last_user,
                "beta_headers": beta_headers,
                "keep_alive": keep_alive,
                "context_size": context_size,
                "tool_choice": tool_choice,
            },
            model._supports,
            model.provider.NAME,
            model.model_id,
        )

        self._history: list[Message] = []
        self._log_path: Path | None = Path(log_path) if log_path is not None else None
        self._log_max_message_lines = log_max_message_lines
        self._logged_msg_count: int = 0
        # (msg_count_at_send, total_input_tokens) per completed send/stream.
        # Used by _estimate_cached_boundary to short-circuit the tokenizer
        # walk when a later turn's cache_read matches a recorded total.
        self._turn_boundaries: list[tuple[int, int]] = []
        self._html_logger: HtmlLogger | None = _make_html_logger(self._log_path)

        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_settings_header()

    @property
    def model(self) -> Model:
        return self._model

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self._defaults)

    @property
    def system_blocks(self) -> list[SystemBlock]:
        return list(self._system_blocks)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def is_available(self, setting: str) -> bool:
        return self._model.is_available(setting)

    def get_capabilities(self) -> set[str]:
        return self._model.get_capabilities()

    def add_user_message(
        self,
        content: str | list[ContentBlock] | None = None,
        *,
        text: str | None = None,
    ) -> None:
        if content is None:
            if text is None:
                raise ValueError("add_user_message needs content= or text=.")
            body: str | list[ContentBlock] = text
        else:
            body = content
        self._history.append(Message(role="user", content=body))

    def add_assistant_message(self, content: str | list[ContentBlock]) -> None:
        self._history.append(Message(role="assistant", content=content))

    def add_tool_result(
        self,
        tool_use_id: str,
        result: str | list[ContentBlock],
        *,
        is_error: bool = False,
        name: str | None = None,
    ) -> None:
        block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=result if isinstance(result, str) else self._only_text_image(result),
            is_error=is_error,
            name=name,
        )
        self._history.append(Message(role="tool", content=[block]))

    def snapshot(self) -> Snapshot:
        return Snapshot(
            history=tuple(self._history),
            turn_boundaries=tuple(self._turn_boundaries),
        )

    def rollback(self, snap: Snapshot) -> None:
        self._history = list(snap.history)
        if self._logged_msg_count > len(self._history):
            self._logged_msg_count = len(self._history)
        # Restore the boundaries captured at snapshot time. Any boundary
        # recorded after the snapshot referred to a longer prefix than the
        # rolled-back history and is now invalid.
        self._turn_boundaries = list(snap.turn_boundaries)

    def clone(
        self,
        *,
        name: str | None = None,
        log_path: str | Path | None = None,
        log_max_message_lines: int | None = None,
    ) -> Conversation:
        """Deep-copy history, system blocks, tools, and defaults into a fresh
        conversation. The clone may have its own log path; if omitted, the
        clone has no logging (parent's log isn't shared)."""
        clone = Conversation.__new__(Conversation)
        clone._model = self._model
        clone.name = name or f"{self.name}-clone"
        clone._system_blocks = copy.deepcopy(self._system_blocks)
        clone._history = copy.deepcopy(self._history)
        clone._tools = dict(self._tools)
        clone._defaults = dict(self._defaults)
        # Boundaries reference cumulative token counts of a strict prefix of
        # history. Cloning preserves that prefix verbatim, so boundaries stay
        # valid for the clone's first turn.
        clone._turn_boundaries = list(self._turn_boundaries)
        clone._log_path = Path(log_path) if log_path is not None else None
        clone._log_max_message_lines = (
            log_max_message_lines
            if log_max_message_lines is not None
            else self._log_max_message_lines
        )
        clone._html_logger = _make_html_logger(clone._log_path)
        # Inherited history was already part of the parent; treat it as already
        # logged so the clone's first send shows it under prior_history rather
        # than dumping all of it into new_messages.
        clone._logged_msg_count = len(clone._history)
        if clone._log_path is not None:
            clone._log_path.parent.mkdir(parents=True, exist_ok=True)
            clone._write_settings_header()
        return clone

    def send(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        tool_choice: str | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
    ) -> Response:
        """Send one request to the model and return the response.

        If the response includes tool calls, the caller is responsible for
        executing them and appending results via ``add_tool_result`` before
        the next ``send`` / ``stream`` call. The convenience helpers in
        ``llmfacade.helpers`` automate that loop for ``@tool``-bound funcs."""
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            beta_headers=beta_headers,
            keep_alive=keep_alive,
            context_size=context_size,
            tool_choice=tool_choice,
        )
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)
        resp = self._model.provider._complete_raw(req)
        self._record_turn_boundary(resp.usage, len(req.messages))
        self._log_response(req, resp)
        self._history.append(Message(role="assistant", content=list(resp.blocks)))
        return resp

    async def asend(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        tool_choice: str | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
    ) -> Response:
        """Async equivalent of ``send``."""
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            beta_headers=beta_headers,
            keep_alive=keep_alive,
            context_size=context_size,
            tool_choice=tool_choice,
        )
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)
        resp = await self._model.provider._acomplete_raw(req)
        self._record_turn_boundary(resp.usage, len(req.messages))
        self._log_response(req, resp)
        self._history.append(Message(role="assistant", content=list(resp.blocks)))
        return resp

    def stream(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        tool_choice: str | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
    ) -> Iterator[StreamEvent]:
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            beta_headers=beta_headers,
            keep_alive=keep_alive,
            context_size=context_size,
            tool_choice=tool_choice,
        )
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)

        text_buf: list[str] = []
        thinking_blocks: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []
        last_usage = None
        msg_count_at_send = len(req.messages)
        for ev in self._model.provider._stream_raw(req):
            if ev.text_delta:
                text_buf.append(ev.text_delta)
            if ev.thinking_block is not None:
                thinking_blocks.append(ev.thinking_block)
            if ev.tool_call_delta:
                tool_calls.append(ev.tool_call_delta)
            if ev.usage is not None:
                last_usage = ev.usage
            yield ev

        self._record_turn_boundary(last_usage, msg_count_at_send)
        self._finalize_stream(text_buf, thinking_blocks, tool_calls, last_usage)

    async def astream(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        tool_choice: str | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            beta_headers=beta_headers,
            keep_alive=keep_alive,
            context_size=context_size,
            tool_choice=tool_choice,
        )
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)

        text_buf: list[str] = []
        thinking_blocks: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []
        last_usage = None
        msg_count_at_send = len(req.messages)
        async for ev in self._model.provider._astream_raw(req):
            if ev.text_delta:
                text_buf.append(ev.text_delta)
            if ev.thinking_block is not None:
                thinking_blocks.append(ev.thinking_block)
            if ev.tool_call_delta:
                tool_calls.append(ev.tool_call_delta)
            if ev.usage is not None:
                last_usage = ev.usage
            yield ev

        self._record_turn_boundary(last_usage, msg_count_at_send)
        self._finalize_stream(text_buf, thinking_blocks, tool_calls, last_usage)

    def _finalize_stream(
        self,
        text_buf: list[str],
        thinking_blocks: list[ThinkingBlock],
        tool_calls: list[ToolCall],
        usage: Any,
    ) -> None:
        del usage
        # Order matters: Anthropic and Gemini both expect thinking blocks
        # before any text or tool_use in the assistant turn when sent back.
        blocks: list[ContentBlock] = list(thinking_blocks)
        if text_buf:
            blocks.append(TextBlock("".join(text_buf)))
        for call in tool_calls:
            blocks.append(ToolUseBlock(id=call.id, name=call.name, input=call.input))
        if blocks:
            self._history.append(Message(role="assistant", content=blocks))

    def _collect_per_call(self, **kwargs: Any) -> dict[str, Any]:
        return _validate_knobs(
            kwargs,
            self._model._supports,
            self._model.provider.NAME,
            self._model.model_id,
        )

    def _build_request(
        self,
        *,
        stop: list[str] | None,
        per_call: dict[str, Any],
    ) -> CompletionRequest:
        provider = self._model.provider
        merged: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for k, v in provider._defaults.items():
            merged[k] = v
            sources[k] = "provider"
        for k, v in self._model._defaults.items():
            merged[k] = v
            sources[k] = "model"
        for k, v in self._defaults.items():
            merged[k] = v
            sources[k] = "convo"
        for k, v in per_call.items():
            merged[k] = v
            sources[k] = "per_call"

        merged, sources = _filter_unsupported(
            merged, sources, self._model._supports, provider.NAME, self._model.model_id
        )

        # Most provider APIs require ``max_tokens``. Supply a reasonable default
        # if no scope set one.
        if "max_tokens" not in merged and "max_tokens" in self._model._supports:
            merged["max_tokens"] = 1024
            sources["max_tokens"] = "default"

        # Validate forced-tool selection: a named tool_choice must match a
        # registered tool, and any non-"auto" tool_choice requires tools to be
        # registered. "auto" / "required" / "none" / "<name>" are the four
        # canonical values; unknown reserved-word lookalikes ("any", typo'd
        # "requiered") fall into the named branch and are caught below.
        tc = merged.get("tool_choice")
        if tc is not None and tc != "auto":
            if not self._tools:
                raise ValueError(
                    f"tool_choice={tc!r} requires tools to be registered on the "
                    "conversation, but tools is empty."
                )
            if tc not in {"required", "none"} and tc not in self._tools:
                raise ValueError(
                    f"tool_choice={tc!r} is not 'auto'/'required'/'none' and does "
                    f"not match any registered tool. Known: {sorted(self._tools)}."
                )

        return CompletionRequest(
            model=self._model.model_id,
            messages=list(self._history),
            system_blocks=list(self._system_blocks),
            tools=list(self._tools.values()),
            stop=stop,
            settings=merged,
            settings_source=sources,
        )

    def _check_no_dangling_tool_use(self) -> None:
        used: set[str] = set()
        resolved: set[str] = set()
        for msg in self._history:
            if isinstance(msg.content, str):
                continue
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    used.add(block.id)
                elif isinstance(block, ToolResultBlock):
                    resolved.add(block.tool_use_id)
        unresolved = used - resolved
        if unresolved:
            raise ConversationStateError(
                f"Conversation has unresolved tool calls: {sorted(unresolved)}. "
                f"Append a ToolResult for each via add_tool_result() (or use "
                f"llmfacade.helpers.run_bound_tools) before the next send/stream."
            )

    def _only_text_image(self, blocks: list[ContentBlock]) -> list[Any]:
        return [b for b in blocks if isinstance(b, (TextBlock, ImageBlock))]

    # ---- logging ----------------------------------------------------------

    def _write_settings_header(self) -> None:
        """Emit a one-shot settings record at the start of the log file.

        Captures provider/model/convo defaults and their source, plus system
        blocks and tool names. Subsequent request entries only carry per-call
        overrides and the message delta."""
        provider = self._model.provider
        merged: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for k, v in provider._defaults.items():
            merged[k] = v
            sources[k] = "provider"
        for k, v in self._model._defaults.items():
            merged[k] = v
            sources[k] = "model"
        for k, v in self._defaults.items():
            merged[k] = v
            sources[k] = "convo"
        merged, sources = _filter_unsupported(
            merged, sources, self._model._supports, provider.NAME, self._model.model_id
        )

        settings_block = {
            k: {"value": _logsafe(v), "source": sources[k]} for k, v in merged.items()
        }
        record: dict[str, Any] = {
            "type": "settings",
            "convo": self.name,
            "provider": provider.NAME,
            "model": self._model.model_id,
            "system_blocks": [{"text": sb.text, "cache": sb.cache} for sb in self._system_blocks],
            "tools": [t.name for t in self._tools.values()],
            "settings": settings_block,
        }
        self._append_log(record)
        if self._html_logger is not None:
            self._html_logger.write_header(
                convo_name=self.name,
                provider=provider.NAME,
                model_id=self._model.model_id,
                system_blocks=list(self._system_blocks),
                tools=[t.name for t in self._tools.values()],
                settings=settings_block,
            )

    def _log_request(self, req: CompletionRequest, per_call: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        messages = list(req.messages)
        prior = messages[: self._logged_msg_count]
        new = messages[self._logged_msg_count :]

        record: dict[str, Any] = {
            "type": "request",
            "convo": self.name,
            "tool_choice": req.settings.get("tool_choice", "auto"),
            "stop": req.stop,
            "overrides": {k: _logsafe(v) for k, v in per_call.items()},
            "new_messages": [_dump_message(m, max_lines=self._log_max_message_lines) for m in new],
        }
        if prior:
            rendered = "\n".join(_render_message_oneline(m) for m in prior)
            record["prior_history"] = {
                "messages": len(prior),
                "preview": _abbreviate_lines(rendered),
            }
        self._append_log(record)
        if self._html_logger is not None:
            self._html_logger.write_request(
                new_messages=new,
                overrides={k: _logsafe(v) for k, v in per_call.items()},
                tool_choice=req.settings.get("tool_choice"),
                stop=req.stop,
            )
        self._logged_msg_count = len(messages)

    def _log_response(self, req: CompletionRequest, resp: Response) -> None:
        if self._log_path is None:
            self._logged_msg_count += 1
            return
        max_lines = self._log_max_message_lines
        record: dict[str, Any] = {
            "type": "response",
            "convo": self.name,
            "model": resp.model,
            "text": _abbreviate_text(resp.text, max_lines),
            "tool_calls": [
                {"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls
            ],
            "thinking": (
                _abbreviate_text(resp.thinking, max_lines) if resp.thinking else resp.thinking
            ),
            "usage": _dump_usage(resp.usage),
            "finish_reason": resp.finish_reason,
        }
        summary = self._cache_summary(req, resp.usage)
        if summary is not None:
            record["cache_summary"] = summary
        self._append_log(record)
        if self._html_logger is not None:
            self._html_logger.write_response(
                blocks=list(resp.blocks),
                text=resp.text,
                usage=_dump_usage(resp.usage),
                cache_summary=summary,
                finish_reason=resp.finish_reason,
                model_id=resp.model,
            )
        self._logged_msg_count += 1

    def _cache_summary(self, req: CompletionRequest, usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        cache_read = usage.cache_read_tokens or 0
        cache_creation = usage.cache_creation_tokens or 0
        prompt_uncached = usage.prompt_tokens or 0
        total_input = max(prompt_uncached + cache_creation + cache_read, prompt_uncached)
        if total_input == 0:
            return None

        boundary, exact_boundary = self._estimate_cached_boundary(req, cache_read)
        provider = self._model.provider
        explicit = self._model.is_available("auto_cache_last_user")
        auto = bool(explicit and req.settings.get("auto_cache_last_user", False))

        if cache_read > 0:
            note = (
                f"Provider cache hit ~{cache_read} tokens "
                f"(~{boundary} of {len(req.messages)} prefix messages). "
                "Caching is working."
            )
        elif cache_creation > 0:
            note = (
                f"Provider cached ~{cache_creation} new tokens this turn but "
                "had no prefix to read from (first cacheable turn or cache TTL "
                "expired). Subsequent turns within TTL should hit."
            )
        elif explicit and not auto:
            note = (
                "No cache hit and no cache creation. Explicit cache markers "
                "are off — set auto_cache_last_user=True (or pass a "
                "SystemBlock(..., cache=True)) to enable caching on this "
                "provider."
            )
        elif auto and total_input > 1024:
            note = (
                "No cache hit despite explicit markers. Likely causes: cache "
                "TTL expired, prefix divergence from previous turn (mid-prefix "
                "mutation), beta header missing, or first turn."
            )
        else:
            note = (
                "No cache activity reported. This provider may not expose "
                "cache stats, or the prompt was below the cache threshold."
            )

        if exact_boundary:
            tokenizer_label = "exact (turn-boundary)"
        else:
            tokenizer_label = _tokenizer_label(provider, self._model.model_id)

        return {
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "uncached_input_tokens": prompt_uncached,
            "hit_ratio": round(cache_read / total_input, 3) if total_input else 0.0,
            "approximate_messages_cached": boundary,
            "tokenizer": tokenizer_label,
            "_note": note,
        }

    def _estimate_cached_boundary(
        self, req: CompletionRequest, cache_read_tokens: int
    ) -> tuple[int, bool]:
        """Map ``cache_read_tokens`` back to a message index in ``req.messages``.

        Returns ``(boundary, exact)`` where ``exact=True`` means we matched
        ``cache_read_tokens`` against a previously recorded turn boundary
        (no tokenizer estimation needed). Provider cache markers always sit
        at turn boundaries (system blocks + a previous turn's last user
        message), so a hit reported in this turn typically equals the total
        input-token count of some prior send — which we already get for free
        in ``Usage`` and stash in ``self._turn_boundaries``.

        Falls back to a per-message tokenizer estimate via
        ``provider.count_tokens`` when no recorded boundary matches (e.g.
        first-turn caching, system-block-only markers, mid-prefix divergence
        after rollback)."""
        if cache_read_tokens <= 0:
            return 0, False

        # Fast path: exact match against a recorded turn boundary.
        msg_count_now = len(req.messages)
        best_match: int | None = None
        for msg_count, total in self._turn_boundaries:
            if msg_count > msg_count_now:
                continue
            if total == cache_read_tokens and (best_match is None or msg_count > best_match):
                best_match = msg_count
        if best_match is not None:
            return best_match, True

        # Fallback: walk messages with the provider's local tokenizer.
        provider = self._model.provider
        model_id = self._model.model_id
        accumulated = 0
        for sb in req.system_blocks:
            accumulated += provider.count_tokens(sb.text, model_id=model_id)
            if accumulated > cache_read_tokens:
                return 0, False
        msgs = list(req.messages)
        fully_covered = 0
        for i, msg in enumerate(msgs):
            text = _message_to_text(msg)
            tokens = provider.count_tokens(text, model_id=model_id) if text else 0
            if accumulated + tokens > cache_read_tokens:
                return i, False
            accumulated += tokens
            fully_covered = i + 1
        return fully_covered, False

    def _record_turn_boundary(self, usage: Any, msg_count_at_send: int) -> None:
        """Persist (msg_count_at_send, total_input_tokens) so a later turn's
        cache_read can be mapped back to an exact message index without a
        tokenizer call. Called after every successful send/stream."""
        if usage is None:
            return
        prompt = usage.prompt_tokens or 0
        cache_read = usage.cache_read_tokens or 0
        cache_creation = usage.cache_creation_tokens or 0
        total = prompt + cache_read + cache_creation
        if total <= 0:
            return
        self._turn_boundaries.append((msg_count_at_send, total))

    def _append_log(self, record: dict[str, Any]) -> None:
        assert self._log_path is not None
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_log_default) + "\n")


def _logsafe(v: Any) -> Any:
    """Render Enum values as their .value for compact JSON; passthrough others."""
    from enum import Enum

    if isinstance(v, Enum):
        return v.value
    return v


def _make_html_logger(log_path: Path | None) -> HtmlLogger | None:
    """Pair an HTML log alongside the JSONL log unless the JSONL itself
    already lives at the .html path (in which case writing both would
    clobber the JSONL)."""
    if log_path is None:
        return None
    html_path = log_path.with_suffix(".html")
    if html_path == log_path:
        return None
    return HtmlLogger(html_path)
