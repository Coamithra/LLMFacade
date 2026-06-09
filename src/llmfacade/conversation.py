from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade._html_log import HtmlLogger
from llmfacade.cache import (
    ResponseCache,
    fingerprint_request,
    hash_fingerprint,
    replay_stream,
    resolve_cache,
)
from llmfacade.exceptions import (
    CacheMissError,
    ConversationStateError,
    ProviderError,
    RepetitionLoopError,
    UnsupportedFeature,
)
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
from llmfacade.repetition import (
    RepetitionGuard,
    coerce_repetition_guard,
    detect_repetition_loop,
    resolve_repetition_guard,
)
from llmfacade.settings import DrySampler, ThinkingMode, is_budget_thinking
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


def _message_has_toplevel_image(m: Message) -> bool:
    """True if an ``ImageBlock`` sits directly in the message content (a user-
    or assistant-supplied image). Gated by the ``"vision"`` capability."""
    if isinstance(m.content, str):
        return False
    return any(isinstance(block, ImageBlock) for block in m.content)


def _message_has_tool_result_image(m: Message) -> bool:
    """True if an ``ImageBlock`` is nested inside a ``ToolResultBlock`` (an image
    a tool returned). Gated by the ``"tool_result_images"`` capability — only
    Anthropic marshals images in this position."""
    if isinstance(m.content, str):
        return False
    return any(
        isinstance(block, ToolResultBlock)
        and not isinstance(block.content, str)
        and any(isinstance(inner, ImageBlock) for inner in block.content)
        for block in m.content
    )


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


@dataclass
class _StreamBuffers:
    """Mutable accumulator for a single streamed attempt, used by the
    repetition guard. ``absorb`` folds in one ``StreamEvent``; the fields
    mirror what ``_finalize_stream`` reads off the inline stream loops."""

    text: list[str] = field(default_factory=list)
    thinking_text: list[str] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Any = None
    finish_reason: str | None = None

    def absorb(self, ev: StreamEvent) -> None:
        if ev.text_delta:
            self.text.append(ev.text_delta)
        if ev.thinking_delta:
            self.thinking_text.append(ev.thinking_delta)
        if ev.thinking_block is not None:
            self.thinking_blocks.append(ev.thinking_block)
        if ev.tool_call_delta:
            self.tool_calls.append(ev.tool_call_delta)
        if ev.usage is not None:
            self.usage = ev.usage
        if ev.finish_reason is not None:
            self.finish_reason = ev.finish_reason


def _detection_text(
    thinking_buf: list[str], text_buf: list[str], tool_calls: list[ToolCall]
) -> str:
    """Build the buffer the repetition detector scans: the model's reasoning /
    thinking stream, then assistant text, then any streamed tool-call arguments
    — so a loop in any of the three is caught the same way as looping prose.

    The detector is a *suffix* scan, so during a thinking-only phase the suffix
    is the reasoning stream (a looping chain-of-thought is caught live, before
    it burns the whole budget); once text starts the suffix is the text and the
    reasoning sits as a harmless prefix. Reasoning is accumulated from the
    token-by-token ``thinking_delta`` events, never the consolidated
    ``thinking_block`` (which arrives all-at-once only after the loop has
    already happened and would double-count the deltas)."""
    parts = list(thinking_buf) + list(text_buf)
    for tc in tool_calls:
        if tc.raw_arguments:
            parts.append(tc.raw_arguments)
        elif tc.input:
            parts.append(json.dumps(tc.input, default=str, sort_keys=True))
    return "".join(parts)


# DRY-multiplier added per retry when ``RepetitionGuard.escalate_dry`` is on and
# no ``dry`` is already set (so attempt 1 enables DRY at this strength).
_DRY_ESCALATION_STEP = 0.5


def _detect_in_buffers(
    thinking_buf: list[str],
    text_buf: list[str],
    tool_calls: list[ToolCall],
    guard: RepetitionGuard,
) -> str | None:
    return detect_repetition_loop(
        _detection_text(thinking_buf, text_buf, tool_calls),
        tail_chars=guard.tail_chars,
        max_period=guard.max_period,
        min_reps_floor=guard.min_reps_floor,
    )


def _close_stream(stream_iter: Any) -> None:
    """Eagerly close a provider stream iterator so the underlying HTTP
    connection is released on an early (repetition) break. Best-effort: a
    plain iterator without ``close`` is fine to leave to GC."""
    close = getattr(stream_iter, "close", None)
    if close is not None:
        with contextlib.suppress(Exception):
            close()


async def _aclose_stream(stream_iter: Any) -> None:
    """Async equivalent of ``_close_stream`` (prefers ``aclose``)."""
    aclose = getattr(stream_iter, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()
        return
    _close_stream(stream_iter)


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
        log_dir: Any | None = None,
        log_path: Any | None = None,
        log_max_message_lines: int | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        repetition_detection: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
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
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "dry": dry,
                "effort": effort,
                "thinking": thinking,
                "output_format": output_format,
                "user_metadata": user_metadata,
                "cache_ttl": cache_ttl,
                "auto_cache_last_user": auto_cache_last_user,
                "auto_cache_tools": auto_cache_tools,
                "beta_headers": beta_headers,
                "tool_choice": tool_choice,
            },
            model._supports,
            model.provider.NAME,
            model.model_id,
        )

        self._history: list[Message] = []
        self._log_dir_override = log_dir
        self._log_path_override = log_path
        self._log_path: Path | None = _resolve_log_path(
            convo_name=self.name,
            convo_log_path=log_path,
            convo_log_dir=log_dir,
            model=model,
        )
        self._log_max_message_lines = log_max_message_lines
        self._logged_msg_count: int = 0
        # (msg_count_at_send, total_input_tokens) per completed send/stream.
        # Used by _estimate_cached_boundary to short-circuit the tokenizer
        # walk when a later turn's cache_read matches a recorded total.
        self._turn_boundaries: list[tuple[int, int]] = []
        self._html_logger: HtmlLogger | None = _make_html_logger(
            self._log_path, max_lines=self._log_max_message_lines
        )

        self._cache_dir_override = cache_dir
        self._cache_mode_override = cache_mode
        self._cache: ResponseCache | None = resolve_cache(
            convo_cache_dir=cache_dir,
            convo_cache_mode=cache_mode,
            model=model,
        )

        self._repetition_override = repetition_detection
        self._repetition_guard: RepetitionGuard | None = resolve_repetition_guard(
            convo_repetition=repetition_detection,
            model=model,
        )

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
        log_dir: Any | None = None,
        log_path: Any | None = None,
        log_max_message_lines: int | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        repetition_detection: Any | None = None,
    ) -> Conversation:
        """Deep-copy history, system blocks, tools, and defaults into a fresh
        conversation. The clone resolves its own log path through the same
        cascade as a fresh ``new_conversation`` call: pass ``log_dir=False``
        or ``log_path=False`` to disable logging on the clone. The cache
        cascade is re-resolved the same way; pass ``cache_dir=`` /
        ``cache_mode=`` to override what was on the source. ``repetition_detection``
        likewise re-resolves through the cascade; pass it to override the
        source's convo-scope guard (``False`` disables)."""
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
        clone._log_dir_override = log_dir
        clone._log_path_override = log_path
        clone._log_path = _resolve_log_path(
            convo_name=clone.name,
            convo_log_path=log_path,
            convo_log_dir=log_dir,
            model=self._model,
        )
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
        clone._cache_dir_override = cache_dir
        clone._cache_mode_override = cache_mode
        clone._cache = resolve_cache(
            convo_cache_dir=cache_dir,
            convo_cache_mode=cache_mode,
            model=self._model,
        )
        clone._repetition_override = repetition_detection
        clone._repetition_guard = resolve_repetition_guard(
            convo_repetition=repetition_detection,
            model=self._model,
        )
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
        repetition_detection: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
    ) -> Response:
        """Send one request to the model and return the response.

        If the response includes tool calls, the caller is responsible for
        executing them and appending results via ``add_tool_result`` before
        the next ``send`` / ``stream`` call. The convenience helpers in
        ``llmfacade.helpers`` automate that loop for ``@tool``-bound funcs.

        When a ``RepetitionGuard`` is in effect (via ``repetition_detection``
        here or any cascade scope) the round-trip runs under the hood as a
        stream so a degenerate repetition loop is caught mid-generation and the
        whole call is transparently restarted; see ``RepetitionGuard``."""
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            dry=dry,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            auto_cache_tools=auto_cache_tools,
            beta_headers=beta_headers,
            tool_choice=tool_choice,
        )
        guard = self._effective_guard(repetition_detection)
        self._check_no_dangling_tool_use()
        guard_snap = self.snapshot() if guard is not None else None
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)

        cache_key, cache_fp, cached = self._cache_lookup(req)
        if cached is not None:
            self._record_turn_boundary(cached.usage, len(req.messages))
            self._log_response(req, cached)
            self._history.append(Message(role="assistant", content=list(cached.blocks)))
            return cached

        if guard is None:
            resp = self._model.provider._complete_raw(req)
        else:
            try:
                resp = self._guarded_complete(req, guard)
            except RepetitionLoopError:
                if guard_snap is not None:
                    self.rollback(guard_snap)
                raise
        self._cache_store(cache_key, cache_fp, resp)
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
        repetition_detection: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
    ) -> Response:
        """Async equivalent of ``send``."""
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            dry=dry,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            auto_cache_tools=auto_cache_tools,
            beta_headers=beta_headers,
            tool_choice=tool_choice,
        )
        guard = self._effective_guard(repetition_detection)
        self._check_no_dangling_tool_use()
        guard_snap = self.snapshot() if guard is not None else None
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)

        cache_key, cache_fp, cached = self._cache_lookup(req)
        if cached is not None:
            self._record_turn_boundary(cached.usage, len(req.messages))
            self._log_response(req, cached)
            self._history.append(Message(role="assistant", content=list(cached.blocks)))
            return cached

        if guard is None:
            resp = await self._model.provider._acomplete_raw(req)
        else:
            try:
                resp = await self._aguarded_complete(req, guard)
            except RepetitionLoopError:
                if guard_snap is not None:
                    self.rollback(guard_snap)
                raise
        self._cache_store(cache_key, cache_fp, resp)
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
        repetition_detection: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
    ) -> Iterator[StreamEvent]:
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            dry=dry,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            auto_cache_tools=auto_cache_tools,
            beta_headers=beta_headers,
            tool_choice=tool_choice,
        )
        guard = self._effective_guard(repetition_detection)
        self._check_no_dangling_tool_use()
        guard_snap = self.snapshot() if guard is not None else None
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)

        cache_key, cache_fp, cached = self._cache_lookup(req)
        msg_count_at_send = len(req.messages)
        if cached is not None:
            try:
                yield from replay_stream(cached)
            finally:
                self._record_turn_boundary(cached.usage, msg_count_at_send)
                self._history.append(Message(role="assistant", content=list(cached.blocks)))
                self._log_response(req, cached)
            return

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        thinking_blocks: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []
        last_usage = None
        last_finish_reason: str | None = None
        repetition_detail: str | None = None
        chars_since_check = 0
        stream_iter = self._model.provider._stream_raw(req)
        # Use try/finally so a consumer that breaks out of the iterator early
        # (break, exception, generator close) still gets the partial assistant
        # turn appended to history. Otherwise the user message recorded above
        # would be left dangling with no reply, breaking strict role
        # alternation on the next call. The repetition-abort branch is the
        # exception: it discards the partial and rolls the convo back, then
        # raises after the finally.
        try:
            for ev in stream_iter:
                if ev.text_delta:
                    text_buf.append(ev.text_delta)
                if ev.thinking_delta:
                    thinking_buf.append(ev.thinking_delta)
                if ev.thinking_block is not None:
                    thinking_blocks.append(ev.thinking_block)
                if ev.tool_call_delta:
                    tool_calls.append(ev.tool_call_delta)
                if ev.usage is not None:
                    last_usage = ev.usage
                if ev.finish_reason is not None:
                    last_finish_reason = ev.finish_reason
                yield ev
                if guard is not None:
                    if ev.text_delta:
                        chars_since_check += len(ev.text_delta)
                    if ev.thinking_delta:
                        chars_since_check += len(ev.thinking_delta)
                    if ev.tool_call_delta:
                        chars_since_check += guard.check_every
                    if chars_since_check >= guard.check_every:
                        chars_since_check = 0
                        repetition_detail = _detect_in_buffers(
                            thinking_buf, text_buf, tool_calls, guard
                        )
                        if repetition_detail:
                            break
            else:
                # Natural completion: flush a final check so a short-but-complete
                # loop that never crossed the cadence is still caught.
                if guard is not None and chars_since_check > 0:
                    repetition_detail = _detect_in_buffers(
                        thinking_buf, text_buf, tool_calls, guard
                    )
        finally:
            if repetition_detail is not None:
                _close_stream(stream_iter)
                if guard_snap is not None:
                    self.rollback(guard_snap)
            else:
                self._record_turn_boundary(last_usage, msg_count_at_send)
                resp = self._finalize_stream(
                    req, text_buf, thinking_blocks, tool_calls, last_usage, last_finish_reason
                )
                if resp is not None:
                    self._cache_store(cache_key, cache_fp, resp)
                _close_stream(stream_iter)
        if repetition_detail is not None:
            raise RepetitionLoopError(
                repetition_detail,
                attempts=1,
                partial_text=_detection_text(thinking_buf, text_buf, tool_calls),
            )

    async def astream(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        tool_choice: str | None = None,
        stop: list[str] | None = None,
        repetition_detection: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        per_call = self._collect_per_call(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            dry=dry,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            auto_cache_tools=auto_cache_tools,
            beta_headers=beta_headers,
            tool_choice=tool_choice,
        )
        guard = self._effective_guard(repetition_detection)
        self._check_no_dangling_tool_use()
        guard_snap = self.snapshot() if guard is not None else None
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        req = self._build_request(stop=stop, per_call=per_call)
        self._log_request(req, per_call)

        cache_key, cache_fp, cached = self._cache_lookup(req)
        msg_count_at_send = len(req.messages)
        if cached is not None:
            try:
                for ev in replay_stream(cached):
                    yield ev
            finally:
                self._record_turn_boundary(cached.usage, msg_count_at_send)
                self._history.append(Message(role="assistant", content=list(cached.blocks)))
                self._log_response(req, cached)
            return

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        thinking_blocks: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []
        last_usage = None
        last_finish_reason: str | None = None
        repetition_detail: str | None = None
        chars_since_check = 0
        stream_iter = self._model.provider._astream_raw(req)
        try:
            async for ev in stream_iter:
                if ev.text_delta:
                    text_buf.append(ev.text_delta)
                if ev.thinking_delta:
                    thinking_buf.append(ev.thinking_delta)
                if ev.thinking_block is not None:
                    thinking_blocks.append(ev.thinking_block)
                if ev.tool_call_delta:
                    tool_calls.append(ev.tool_call_delta)
                if ev.usage is not None:
                    last_usage = ev.usage
                if ev.finish_reason is not None:
                    last_finish_reason = ev.finish_reason
                yield ev
                if guard is not None:
                    if ev.text_delta:
                        chars_since_check += len(ev.text_delta)
                    if ev.thinking_delta:
                        chars_since_check += len(ev.thinking_delta)
                    if ev.tool_call_delta:
                        chars_since_check += guard.check_every
                    if chars_since_check >= guard.check_every:
                        chars_since_check = 0
                        repetition_detail = _detect_in_buffers(
                            thinking_buf, text_buf, tool_calls, guard
                        )
                        if repetition_detail:
                            break
            else:
                # Natural completion: flush a final check so a short-but-complete
                # loop that never crossed the cadence is still caught.
                if guard is not None and chars_since_check > 0:
                    repetition_detail = _detect_in_buffers(
                        thinking_buf, text_buf, tool_calls, guard
                    )
        finally:
            if repetition_detail is not None:
                await _aclose_stream(stream_iter)
                if guard_snap is not None:
                    self.rollback(guard_snap)
            else:
                self._record_turn_boundary(last_usage, msg_count_at_send)
                resp = self._finalize_stream(
                    req, text_buf, thinking_blocks, tool_calls, last_usage, last_finish_reason
                )
                if resp is not None:
                    self._cache_store(cache_key, cache_fp, resp)
                await _aclose_stream(stream_iter)
        if repetition_detail is not None:
            raise RepetitionLoopError(
                repetition_detail,
                attempts=1,
                partial_text=_detection_text(thinking_buf, text_buf, tool_calls),
            )

    # ---- llama-server slot save/restore (external mode only) -------------

    def _slot_provider(self) -> Any:
        """Capability-check + return ``self._model.provider`` typed as ``Any``.
        Slot methods (``save_slot``, ``arestore_slot``, …) live only on
        ``LlamaCppServerProvider``; the ``Any`` cast lets the slot wrappers
        below call them without Pyright complaining about the absent
        attributes on the base ``Provider``."""
        self._check_slot_capable()
        return self._model.provider

    def save_slot(self, filename: str) -> dict[str, Any]:
        """Persist the current slot's KV state to disk under ``filename``.

        ``filename`` is interpreted relative to llama-server's
        ``--slot-save-path`` directory; the server must have been launched
        with that flag. Allowed characters: ``[a-zA-Z0-9._-]``; no path
        separators, no leading ``.``, no ``..`` substring (stricter than
        "no ``..`` segments" — any pair of dots anywhere is rejected).
        ``Conversation.save_slot``,
        ``restore_slot``, and ``erase_slot`` operate on slot ``0`` (the only
        slot under ``--parallel 1``); multi-slot selection is out of scope
        for v1.

        For atomicity against concurrent slot mutations, wrap a save +
        restore + send sequence in ``with provider.slot_lock(): ...`` —
        these methods do not acquire the lock internally so the caller can
        compose them.

        Raises:
            UnsupportedFeature: if the conversation's provider is not
                ``llamacpp``.
            NotImplementedError: in llamacpp managed mode (deferred to v2;
                slot routing across llama-swap model loads is undefined).
            ValueError: if ``filename`` is not a safe relative basename.
            ProviderError: if the server was started without
                ``--slot-save-path`` (the original 500 body is wrapped with
                a hint pointing at that flag)."""
        provider = self._slot_provider()
        sanitized = _sanitize_slot_filename(filename)
        try:
            return provider.save_slot(0, sanitized)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="save_slot") from e

    async def asave_slot(self, filename: str) -> dict[str, Any]:
        """Async equivalent of ``save_slot``."""
        provider = self._slot_provider()
        sanitized = _sanitize_slot_filename(filename)
        try:
            return await provider.asave_slot(0, sanitized)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="asave_slot") from e

    def restore_slot(self, filename: str) -> dict[str, Any]:
        """Load a previously-saved KV state from ``filename`` into slot ``0``.

        The next ``send`` / ``stream`` against this conversation will
        prefix-match the restored KV; matching tokens skip prefill.
        Mismatched-prefix sends fall through to a normal cold prefill —
        restore is never destructive. See ``save_slot`` for filename rules
        and capability gating."""
        provider = self._slot_provider()
        sanitized = _sanitize_slot_filename(filename)
        try:
            return provider.restore_slot(0, sanitized)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="restore_slot") from e

    async def arestore_slot(self, filename: str) -> dict[str, Any]:
        """Async equivalent of ``restore_slot``."""
        provider = self._slot_provider()
        sanitized = _sanitize_slot_filename(filename)
        try:
            return await provider.arestore_slot(0, sanitized)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="arestore_slot") from e

    def erase_slot(self) -> dict[str, Any]:
        """Wipe slot ``0``'s in-memory KV state on the server.

        This is llama-server's ``?action=erase`` — it clears the slot's
        live cache only. It does NOT delete any ``--slot-save-path`` files
        on disk; previously saved KVs remain restorable. See ``save_slot``
        for capability gating."""
        provider = self._slot_provider()
        try:
            return provider.erase_slot(0)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="erase_slot") from e

    async def aerase_slot(self) -> dict[str, Any]:
        """Async equivalent of ``erase_slot``."""
        provider = self._slot_provider()
        try:
            return await provider.aerase_slot(0)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="aerase_slot") from e

    def warm_and_save(self, filename: str, *, max_warmup_tokens: int = 1) -> dict[str, Any]:
        """Drive a one-token completion against the current system block,
        then save the resulting slot to disk under ``filename``.

        The conversation's history must be empty when called — this is
        intended to run once, right after construction, to materialise the
        static-prefix KV. The warmup user/assistant turns are rolled back
        so subsequent ``send`` calls behave as a fresh first turn that
        prefix-matches the saved KV. Like the other slot methods, this is
        not internally atomic; wrap with ``with provider.slot_lock(): ...``
        if other tasks may mutate the slot concurrently.

        Returns the dict that ``save_slot`` returns. Raises
        ``ConversationStateError`` if history is non-empty; otherwise
        propagates the same errors as ``save_slot``."""
        provider = self._slot_provider()
        sanitized = _sanitize_slot_filename(filename)
        if self._history:
            raise ConversationStateError(
                "warm_and_save requires an empty conversation history; got "
                f"{len(self._history)} prior message(s). Call this once right "
                "after new_conversation(), or clone the convo first."
            )
        snap = self.snapshot()
        try:
            self.send(".", max_tokens=max_warmup_tokens)
        finally:
            self.rollback(snap)
        try:
            return provider.save_slot(0, sanitized)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="warm_and_save") from e

    async def awarm_and_save(self, filename: str, *, max_warmup_tokens: int = 1) -> dict[str, Any]:
        """Async equivalent of ``warm_and_save``."""
        provider = self._slot_provider()
        sanitized = _sanitize_slot_filename(filename)
        if self._history:
            raise ConversationStateError(
                "awarm_and_save requires an empty conversation history; got "
                f"{len(self._history)} prior message(s). Call this once right "
                "after new_conversation(), or clone the convo first."
            )
        snap = self.snapshot()
        try:
            await self.asend(".", max_tokens=max_warmup_tokens)
        finally:
            self.rollback(snap)
        try:
            return await provider.asave_slot(0, sanitized)
        except ProviderError as e:
            raise _wrap_slot_provider_error(e, op="awarm_and_save") from e

    def _check_slot_capable(self) -> None:
        provider = self._model.provider
        if provider.NAME != "llamacpp":
            raise UnsupportedFeature("slot_save_restore", provider.NAME, self._model.model_id)
        # Managed mode is deferred to v2 — slot routing across llama-swap
        # model loads is undefined (KV state is per-architecture and the
        # swap may evict the model between calls). Callers that really
        # want managed-mode slot ops can still hit `provider.save_slot(...)`
        # directly with model=<id>; this conversation-level surface stays
        # opinionated.
        if getattr(provider, "_managed", False):
            raise NotImplementedError(
                "Conversation slot save/restore is external-mode only in v1. "
                "In managed mode (llama-swap), call provider.save_slot/"
                "restore_slot/erase_slot directly with model=<id>."
            )

    def _finalize_stream(
        self,
        req: CompletionRequest,
        text_buf: list[str],
        thinking_blocks: list[ThinkingBlock],
        tool_calls: list[ToolCall],
        usage: Any,
        finish_reason: str | None,
    ) -> Response | None:
        """Assemble a ``Response`` from streaming buffers, append the assistant
        turn to history, and log it. Returns the assembled ``Response``, or
        ``None`` if the stream produced no blocks at all (e.g. consumer broke
        before any deltas arrived) — in which case we leave history alone."""
        # Order matters: Anthropic and Gemini both expect thinking blocks
        # before any text or tool_use in the assistant turn when sent back.
        blocks: list[ContentBlock] = list(thinking_blocks)
        text = "".join(text_buf)
        if text_buf:
            blocks.append(TextBlock(text))
        for call in tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=call.id,
                    name=call.name,
                    input=call.input,
                    raw_arguments=call.raw_arguments,
                )
            )
        if not blocks:
            return None
        self._history.append(Message(role="assistant", content=blocks))
        thinking_text = "".join(b.text for b in thinking_blocks if not b.encrypted) or None
        resp = Response(
            text=text,
            blocks=blocks,
            tool_calls=list(tool_calls),
            thinking=thinking_text,
            usage=usage,
            finish_reason=finish_reason,
            model=req.model,
        )
        self._log_response(req, resp)
        return resp

    # ---- repetition-loop guard -------------------------------------------

    def _effective_guard(self, per_call: Any) -> RepetitionGuard | None:
        """Resolve the active ``RepetitionGuard`` for a send/stream call.

        A per-call ``repetition_detection`` value (anything but ``None``)
        overrides the convo-resolved default; ``None`` defers to it."""
        if per_call is None:
            return self._repetition_guard
        return coerce_repetition_guard(per_call)

    def _guarded_complete(self, req: CompletionRequest, guard: RepetitionGuard) -> Response:
        """Run ``req`` under the repetition guard and return the first attempt
        that does not loop. Drives the provider's stream hook so a loop is
        caught mid-generation; on a hit, discards the attempt and restarts with
        escalated anti-repetition samplers, up to ``guard.retries`` times. Does
        not touch history — the caller appends the returned ``Response``."""
        attempts = guard.retries + 1
        last_detail: str | None = None
        last_buffers: _StreamBuffers | None = None
        for attempt in range(attempts):
            attempt_req = self._escalated_request(req, guard, attempt)
            detail, buffers = self._drive_guarded_stream(attempt_req, guard)
            if detail is None:
                return self._build_response_from_stream(req, buffers)
            last_detail, last_buffers = detail, buffers
        if guard.on_exhausted == "return_last" and last_buffers is not None:
            return self._build_response_from_stream(req, last_buffers)
        raise RepetitionLoopError(
            last_detail or "repetition loop",
            attempts=attempts,
            partial_text=(
                _detection_text(
                    last_buffers.thinking_text, last_buffers.text, last_buffers.tool_calls
                )
                if last_buffers is not None
                else ""
            ),
        )

    async def _aguarded_complete(self, req: CompletionRequest, guard: RepetitionGuard) -> Response:
        """Async equivalent of ``_guarded_complete``."""
        attempts = guard.retries + 1
        last_detail: str | None = None
        last_buffers: _StreamBuffers | None = None
        for attempt in range(attempts):
            attempt_req = self._escalated_request(req, guard, attempt)
            detail, buffers = await self._adrive_guarded_stream(attempt_req, guard)
            if detail is None:
                return self._build_response_from_stream(req, buffers)
            last_detail, last_buffers = detail, buffers
        if guard.on_exhausted == "return_last" and last_buffers is not None:
            return self._build_response_from_stream(req, last_buffers)
        raise RepetitionLoopError(
            last_detail or "repetition loop",
            attempts=attempts,
            partial_text=(
                _detection_text(
                    last_buffers.thinking_text, last_buffers.text, last_buffers.tool_calls
                )
                if last_buffers is not None
                else ""
            ),
        )

    def _drive_guarded_stream(
        self, req: CompletionRequest, guard: RepetitionGuard
    ) -> tuple[str | None, _StreamBuffers]:
        """Consume one provider stream, running the detector every
        ``guard.check_every`` chars. Returns ``(hit_detail_or_None, buffers)``;
        a non-None detail means a loop was detected and the stream was aborted."""
        buf = _StreamBuffers()
        chars_since_check = 0
        detail: str | None = None
        stream_iter = self._model.provider._stream_raw(req)
        try:
            for ev in stream_iter:
                buf.absorb(ev)
                if ev.text_delta:
                    chars_since_check += len(ev.text_delta)
                if ev.thinking_delta:
                    chars_since_check += len(ev.thinking_delta)
                if ev.tool_call_delta:
                    chars_since_check += guard.check_every
                if chars_since_check >= guard.check_every:
                    chars_since_check = 0
                    detail = _detect_in_buffers(buf.thinking_text, buf.text, buf.tool_calls, guard)
                    if detail:
                        break
            else:
                # Stream ended naturally without crossing the cadence on its
                # last bytes: flush one final check so a short-but-complete
                # loop (e.g. a model that repeats a brief phrase then hits a
                # low max_tokens) isn't missed.
                if chars_since_check > 0:
                    detail = _detect_in_buffers(buf.thinking_text, buf.text, buf.tool_calls, guard)
        finally:
            _close_stream(stream_iter)
        return detail, buf

    async def _adrive_guarded_stream(
        self, req: CompletionRequest, guard: RepetitionGuard
    ) -> tuple[str | None, _StreamBuffers]:
        """Async equivalent of ``_drive_guarded_stream``."""
        buf = _StreamBuffers()
        chars_since_check = 0
        detail: str | None = None
        stream_iter = self._model.provider._astream_raw(req)
        try:
            async for ev in stream_iter:
                buf.absorb(ev)
                if ev.text_delta:
                    chars_since_check += len(ev.text_delta)
                if ev.thinking_delta:
                    chars_since_check += len(ev.thinking_delta)
                if ev.tool_call_delta:
                    chars_since_check += guard.check_every
                if chars_since_check >= guard.check_every:
                    chars_since_check = 0
                    detail = _detect_in_buffers(buf.thinking_text, buf.text, buf.tool_calls, guard)
                    if detail:
                        break
            else:
                if chars_since_check > 0:
                    detail = _detect_in_buffers(buf.thinking_text, buf.text, buf.tool_calls, guard)
        finally:
            await _aclose_stream(stream_iter)
        return detail, buf

    def _build_response_from_stream(self, req: CompletionRequest, buf: _StreamBuffers) -> Response:
        """Assemble a ``Response`` from stream buffers without touching history
        (the guarded send appends it itself, after caching). Mirrors the block
        ordering of ``_finalize_stream``."""
        blocks: list[ContentBlock] = list(buf.thinking_blocks)
        text = "".join(buf.text)
        if buf.text:
            blocks.append(TextBlock(text))
        for call in buf.tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=call.id,
                    name=call.name,
                    input=call.input,
                    raw_arguments=call.raw_arguments,
                )
            )
        thinking_text = "".join(b.text for b in buf.thinking_blocks if not b.encrypted) or None
        return Response(
            text=text,
            blocks=blocks,
            tool_calls=list(buf.tool_calls),
            thinking=thinking_text,
            usage=buf.usage,
            finish_reason=buf.finish_reason,
            model=req.model,
        )

    def _escalated_request(
        self, req: CompletionRequest, guard: RepetitionGuard, attempt: int
    ) -> CompletionRequest:
        """Return ``req`` for the first attempt, or a copy with anti-repetition
        samplers nudged up for a retry. Escalation only touches knobs the model
        actually supports, so it is a no-op on providers without
        ``repeat_penalty`` / ``dry`` (e.g. hosted APIs)."""
        if attempt == 0:
            return req
        new_settings = self._escalate_settings(req.settings, guard, attempt)
        if new_settings is req.settings:
            return req
        return dataclasses.replace(req, settings=new_settings)

    def _escalate_settings(
        self, settings: dict[str, Any], guard: RepetitionGuard, attempt: int
    ) -> dict[str, Any]:
        supports = self._model._supports
        out = dict(settings)
        changed = False
        if guard.escalate_repeat_penalty is not None and "repeat_penalty" in supports:
            base = out.get("repeat_penalty")
            # 1.0 is llama.cpp's neutral repeat_penalty (no penalty); escalate
            # upward from there when no numeric value is already in effect.
            base = base if isinstance(base, (int, float)) and not isinstance(base, bool) else 1.0
            out["repeat_penalty"] = round(base + guard.escalate_repeat_penalty * attempt, 4)
            changed = True
        if guard.escalate_dry and "dry" in supports:
            existing = out.get("dry")
            base_mult = existing.multiplier if isinstance(existing, DrySampler) else 0.0
            new_mult = round(base_mult + _DRY_ESCALATION_STEP * attempt, 4)
            if new_mult > 0:
                out["dry"] = (
                    dataclasses.replace(existing, multiplier=new_mult)
                    if isinstance(existing, DrySampler)
                    else DrySampler(multiplier=new_mult)
                )
                changed = True
        return out if changed else settings

    # ---- response cache --------------------------------------------------

    def _cache_lookup(
        self, req: CompletionRequest
    ) -> tuple[str | None, dict[str, Any] | None, Response | None]:
        """Resolve the cache key for ``req`` and return any hit.

        Returns ``(key, fingerprint, cached_or_None)``. If the cache is in
        ``replay_only`` mode and there is no hit, raises ``CacheMissError``
        — no provider call is made by the caller in that case. If no cache
        is configured, returns ``(None, None, None)`` and the caller proceeds
        as normal."""
        if self._cache is None:
            return None, None, None
        provider = self._model.provider
        provider_name = provider.NAME
        fp = fingerprint_request(req, provider_name, base_url=provider._base_url)
        key = hash_fingerprint(fp)
        cached = self._cache.get(provider_name, self._model.model_id, key)
        if cached is not None:
            return key, fp, cached
        if self._cache.mode == "replay_only":
            raise CacheMissError(
                f"replay_only cache miss: no entry for hash {key} "
                f"(provider={provider_name!r}, model={self._model.model_id!r}). "
                f"Looked under {self._cache.path_for(provider_name, self._model.model_id, key)}."
            )
        return key, fp, None

    def _cache_store(
        self,
        key: str | None,
        fingerprint: dict[str, Any] | None,
        resp: Response,
    ) -> None:
        """Write ``resp`` to the cache if one is configured and writes are
        enabled in the active mode. No-op when ``key`` is ``None`` (no cache
        configured) or ``fingerprint`` is ``None`` (lookup didn't run, e.g.
        because the provider call happened on a path that bypassed the
        cache)."""
        if self._cache is None or key is None or fingerprint is None:
            return
        self._cache.put(
            self._model.provider.NAME,
            self._model.model_id,
            key,
            resp,
            fingerprint,
        )

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
        if not self._model.is_available("vision") and any(
            _message_has_toplevel_image(m) for m in self._history
        ):
            raise UnsupportedFeature("vision", provider.NAME, self._model.model_id)
        if not self._model.is_available("tool_result_images") and any(
            _message_has_tool_result_image(m) for m in self._history
        ):
            raise UnsupportedFeature("tool_result_images", provider.NAME, self._model.model_id)
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

        # Budget-based extended thinking (an int token budget) is a distinct
        # capability from the adaptive `thinking` modes: Opus 4.7/4.8 accept
        # adaptive thinking but 400 on a budget. The `thinking` knob name is
        # gated by SUPPORTS like any other; this is the value-level gate that
        # rejects the budget *form* on models without "thinking_budget" (e.g.
        # Opus 4.8), so it fails fast here instead of as a provider 400. A
        # ThinkingMode value is never a budget and passes through.
        thinking_val = merged.get("thinking")
        if is_budget_thinking(thinking_val) and not self._model.is_available("thinking_budget"):
            raise UnsupportedFeature("thinking_budget", provider.NAME, self._model.model_id)

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
        extra = provider.log_metadata(model_id=self._model.model_id)
        if extra:
            # Drop any keys that would shadow the base header fields so a
            # misbehaving provider can't silently corrupt log readers.
            reserved = set(record)
            extra = {k: v for k, v in extra.items() if k not in reserved}
            record.update(extra)
        self._append_log(record)
        if self._html_logger is not None:
            self._html_logger.write_header(
                convo_name=self.name,
                provider=provider.NAME,
                model_id=self._model.model_id,
                system_blocks=list(self._system_blocks),
                tools=[t.name for t in self._tools.values()],
                settings=settings_block,
                extra=extra,
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
                {
                    "id": c.id,
                    "name": c.name,
                    "input": c.input,
                    **(
                        {"raw_arguments": _abbreviate_text(c.raw_arguments, max_lines)}
                        if c.raw_arguments is not None
                        else {}
                    ),
                }
                for c in resp.tool_calls
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
        reasoning = self._reasoning_summary(resp)
        if reasoning is not None:
            record["reasoning"] = reasoning
        self._append_log(record)
        if self._html_logger is not None:
            self._html_logger.write_response(
                blocks=list(resp.blocks),
                text=resp.text,
                usage=_dump_usage(resp.usage),
                cache_summary=summary,
                reasoning=reasoning,
                finish_reason=resp.finish_reason,
                model_id=resp.model,
            )
        self._logged_msg_count += 1

    def _reasoning_summary(self, resp: Response) -> dict[str, Any] | None:
        """Resolve a reasoning-token count for the log.

        Prefers the provider-reported ``usage.reasoning_tokens``. When the API
        doesn't break reasoning out — llama.cpp folds it into
        ``completion_tokens``; Anthropic doesn't report it separately — falls
        back to counting the reasoning text (``resp.thinking``) with the
        provider's local tokenizer: exact for llama.cpp (``/tokenize``), a
        labelled ``chars/4`` estimate for Anthropic. Returns ``None`` when the
        turn produced no visible reasoning (no text and no reported count)."""
        reported = resp.usage.reasoning_tokens if resp.usage else 0
        text = resp.thinking or ""
        if not reported and not text:
            return None
        if reported:
            return {"reasoning_tokens": reported, "source": "provider", "estimated": False}
        provider = self._model.provider
        model_id = self._model.model_id
        out: dict[str, Any] = {
            "source": provider.tokenizer_name(model_id=model_id),
            "estimated": True,
            "chars": len(text),
        }
        with contextlib.suppress(Exception):
            out["reasoning_tokens"] = provider.count_tokens(text, model_id=model_id)
        return out

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
    """Render Enum values as their .value and dataclass instances (e.g. a
    ``DrySampler`` knob value) as a plain dict for compact JSON; passthrough
    others."""
    import dataclasses
    from enum import Enum

    if isinstance(v, Enum):
        return v.value
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        # asdict does not run nested values back through _logsafe, so this
        # assumes the dataclass's fields are JSON primitives (true for
        # DrySampler). A future dataclass knob with an Enum field would log the
        # raw member here — recurse if that ever happens.
        return dataclasses.asdict(v)
    return v


_SLOT_FILENAME_ALLOWED = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _sanitize_slot_filename(filename: str) -> str:
    """Validate ``filename`` is a safe basename for llama-server's
    ``--slot-save-path`` directory. Mirrors server-side ``fs_validate_filename``
    so we surface a clear ValueError up front rather than letting a 400/500
    leak through. Allowed: ``[a-zA-Z0-9._-]+``; rejected: empty, leading
    dot, ``..`` segment, path separators, anything else."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("slot filename must be a non-empty string")
    if filename.startswith("."):
        raise ValueError(f"slot filename {filename!r} must not start with '.' (server-side rule)")
    if ".." in filename:
        raise ValueError(f"slot filename {filename!r} must not contain '..' (path traversal)")
    bad = sorted({c for c in filename if c not in _SLOT_FILENAME_ALLOWED})
    if bad:
        raise ValueError(
            f"slot filename {filename!r} contains disallowed characters {bad!r}; "
            "allowed: [a-zA-Z0-9._-]"
        )
    return filename


def _wrap_slot_provider_error(exc: ProviderError, *, op: str) -> ProviderError:
    """Re-wrap a ProviderError from a slot endpoint with a hint pointing at
    ``--slot-save-path`` when the body indicates the server was launched
    without it. Otherwise the original error passes through. The original
    exception is preserved as ``__cause__`` via the caller's ``raise ...
    from e``."""
    msg = str(exc)
    haystack = msg.lower()
    if "slots action" in haystack or "slot-save-path" in haystack:
        return ProviderError(
            f"{op} failed: {msg} — start llama-server with "
            "--slot-save-path <dir> to enable slot save/restore.",
            original=exc.original,
        )
    return exc


def _resolve_log_path(
    *,
    convo_name: str,
    convo_log_path: Any,
    convo_log_dir: Any,
    model: Model,
) -> Path | None:
    """Resolve a Conversation's effective JSONL log path from the cascade
    (convo > model > provider > manager) plus any explicit ``log_path``
    override on the convo. Any layer can pass ``False`` to disable logging
    at its scope. Returns ``None`` when logging resolves to disabled."""
    # Explicit per-convo log_path wins.
    if convo_log_path is False:
        return None
    if convo_log_path is not None and convo_log_path is not True:
        return Path(convo_log_path)

    # Otherwise resolve a directory from the cascade and compose <dir>/<name>.jsonl.
    layer_overrides = (
        convo_log_dir,
        getattr(model, "_log_dir_override", None),
        getattr(model.provider, "_log_dir_override", None),
    )
    for v in layer_overrides:
        if v is False:
            return None
        if v is not None and v is not True:
            return Path(v) / f"{convo_name}.jsonl"

    manager = getattr(model.provider, "_manager", None)
    if manager is None:
        return None
    run_dir = manager._ensure_run_dir()
    if run_dir is None:
        return None
    return run_dir / f"{convo_name}.jsonl"


def _make_html_logger(log_path: Path | None, *, max_lines: int | None = None) -> HtmlLogger | None:
    """Pair an HTML log alongside the JSONL log unless the JSONL itself
    already lives at the .html path (in which case writing both would
    clobber the JSONL)."""
    if log_path is None:
        return None
    html_path = log_path.with_suffix(".html")
    if html_path == log_path:
        return None
    return HtmlLogger(html_path, max_lines=max_lines)
