from __future__ import annotations

import copy
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import (
    ConversationStateError,
    NotStartedError,
    SettingsLockedError,
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
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from llmfacade.provider import _SettingsFacade
from llmfacade.settings import AnySetting, ConvoSettings, Settings
from llmfacade.tools import Tool

if TYPE_CHECKING:
    from llmfacade.model import Model


_PER_CALL_OVERRIDE_KEYS: dict[str, AnySetting] = {
    "max_tokens": Settings.DefaultMaxTokens,
    "temperature": Settings.DefaultTemperature,
    "top_p": Settings.TopP,
    "top_k": Settings.TopK,
    "effort": Settings.Effort,
    "repeat_penalty": Settings.RepeatPenalty,
}


@dataclass(frozen=True, slots=True)
class _SystemBlock:
    text: str
    cache: bool


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Opaque snapshot token for Conversation.Rollback()."""

    history: tuple[Message, ...]
    system_blocks: tuple[_SystemBlock, ...]


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
        else:
            parts.append(f"<{type(block).__name__}>")
    return f"[{m.role}] " + " ".join(parts)


def _message_to_text(m: Message) -> str:
    """Concatenate the textual content of a message for token estimation.
    Image blocks contribute a fixed estimate per provider (handled at the
    summary level, not here)."""
    if isinstance(m.content, str):
        return m.content
    out: list[str] = []
    for block in m.content:
        if isinstance(block, TextBlock):
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
    """Best-effort label of the tokenizer used for cache_summary estimates."""
    if provider.NAME == "openai":
        try:
            import tiktoken  # noqa: F401

            return "tiktoken"
        except ImportError:
            return "chars/4 (tiktoken not installed)"
    return "chars/4"


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


class Conversation:
    """A stateful chat session against one Model. Holds history, settings, tools."""

    def __init__(
        self,
        *,
        model: Model,
        name: str | None = None,
    ):
        self._model = model
        self.name = name or f"convo-{uuid.uuid4().hex[:8]}"
        self.settings = _SettingsFacade(
            self,
            model.settings.getCapabilities(),
            model.provider.NAME,
            model.model_id,
        )
        self._system_blocks: list[_SystemBlock] = []
        self._history: list[Message] = []
        self._tools: dict[str, Tool] = {}
        self._started = False
        self._log_path: Path | None = None
        self._log_max_message_lines: int | None = None
        self._logged_msg_count: int = 0

    @property
    def model(self) -> Model:
        return self._model

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def started(self) -> bool:
        return self._started

    def isAvailable(self, setting: AnySetting) -> bool:
        return self.settings.isAvailable(setting)

    def getCapabilities(self) -> set[AnySetting]:
        return self.settings.getCapabilities()

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool(self, name: str) -> Tool | None:
        """Look up a registered tool by name (returns None if not registered)."""
        return self._tools.get(name)

    def AddTool(self, t: Tool) -> None:
        self._require_not_started("AddTool")
        self._tools[t.name] = t

    def AddSystemBlock(self, text: str, *, cache: bool = False) -> None:
        self._require_not_started("AddSystemBlock")
        if cache and not self.settings.isAvailable(ConvoSettings.AutoCacheLastUser):
            # Caching capability is gated via AutoCacheLastUser setting availability;
            # if a provider doesn't expose that, it can't honor cache_control either.
            raise UnsupportedFeature(
                ConvoSettings.AutoCacheLastUser,
                self._model.provider.NAME,
                self._model.model_id,
            )
        self._system_blocks.append(_SystemBlock(text=text, cache=cache))

    def AddSystemMessage(self, text: str) -> None:
        self.AddSystemBlock(text, cache=False)

    def AddUserMessage(
        self,
        content: str | list[ContentBlock] | None = None,
        *,
        text: str | None = None,
    ) -> None:
        self._require_started("AddUserMessage")
        body: str | list[ContentBlock]
        if content is None:
            if text is None:
                raise ValueError("AddUserMessage needs content= or text=.")
            body = text
        else:
            body = content
        self._history.append(Message(role="user", content=body))

    def AddAssistantMessage(self, content: str | list[ContentBlock]) -> None:
        self._require_started("AddAssistantMessage")
        self._history.append(Message(role="assistant", content=content))

    def AddToolResult(
        self,
        tool_use_id: str,
        result: str | list[ContentBlock],
        *,
        is_error: bool = False,
        name: str | None = None,
    ) -> None:
        self._require_started("AddToolResult")
        block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=result if isinstance(result, str) else self._only_text_image(result),
            is_error=is_error,
            name=name,
        )
        self._history.append(Message(role="tool", content=[block]))

    def SetLogging(
        self,
        path: str | Path | None,
        *,
        max_message_lines: int | None = None,
    ) -> None:
        """Enable JSONL logging to ``path`` (or disable with ``None``).

        ``max_message_lines`` caps how many lines of any single text payload
        appear in the log. If a message body exceeds the cap, the first half
        and last half of the lines are kept with a ``[N lines skipped]``
        marker between them. ``None`` (default) logs full text."""
        self._require_not_started("SetLogging")
        self._log_path = Path(path) if path is not None else None
        self._log_max_message_lines = max_message_lines
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def Start(self) -> None:
        if self._started:
            return
        self._started = True
        self.settings._lock()

    def Snapshot(self) -> Snapshot:
        return Snapshot(
            history=tuple(self._history),
            system_blocks=tuple(self._system_blocks),
        )

    def Rollback(self, snap: Snapshot) -> None:
        self._history = list(snap.history)
        self._system_blocks = list(snap.system_blocks)
        if self._logged_msg_count > len(self._history):
            self._logged_msg_count = len(self._history)

    def Clone(self) -> Conversation:
        clone = Conversation(model=self._model, name=f"{self.name}-clone")
        clone._system_blocks = copy.deepcopy(self._system_blocks)
        clone._history = copy.deepcopy(self._history)
        clone._tools = dict(self._tools)
        clone.settings._values = dict(self.settings._values)
        clone._log_path = self._log_path
        clone._log_max_message_lines = self._log_max_message_lines
        # Inherited history was already sent by the parent; mark it as logged
        # so the clone's first Send shows it under `prior_history` rather than
        # dumping all of it into `new_messages`.
        clone._logged_msg_count = len(clone._history)
        return clone

    def Send(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        stop: list[str] | None = None,
        tool_choice: str = "auto",
        effort: Any | None = None,
    ) -> Response:
        """Send one request to the model and return the response.

        If the response includes tool calls, the caller is responsible for
        executing them and appending results via `AddToolResult` before the
        next `Send`/`Stream` call. The convenience helpers in
        `llmfacade.helpers` automate that loop for `@tool`-bound functions."""
        self._require_started("Send")
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        overrides = self._collect_overrides(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
        )
        kwargs = self._call_kwargs(
            tool_choice=tool_choice,
            stop=stop,
            overrides=overrides,
        )
        self._log_request(kwargs)
        resp = self._model.provider._complete_raw(**kwargs)
        self._log_response(kwargs, resp)
        self._history.append(Message(role="assistant", content=list(resp.blocks)))
        return resp

    async def aSend(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        stop: list[str] | None = None,
        tool_choice: str = "auto",
        effort: Any | None = None,
    ) -> Response:
        """Async equivalent of `Send`."""
        self._require_started("aSend")
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        overrides = self._collect_overrides(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
        )
        kwargs = self._call_kwargs(
            tool_choice=tool_choice,
            stop=stop,
            overrides=overrides,
        )
        self._log_request(kwargs)
        resp = await self._model.provider._acomplete_raw(**kwargs)
        self._log_response(kwargs, resp)
        self._history.append(Message(role="assistant", content=list(resp.blocks)))
        return resp

    def Stream(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        stop: list[str] | None = None,
        tool_choice: str = "auto",
        effort: Any | None = None,
    ) -> Iterator[StreamEvent]:
        self._require_started("Stream")
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        overrides = self._collect_overrides(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
        )
        kwargs = self._call_kwargs(
            tool_choice=tool_choice,
            stop=stop,
            overrides=overrides,
        )
        self._log_request(kwargs)

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        tool_calls: list[ToolCall] = []
        last_usage = None
        for ev in self._model.provider._stream_raw(**kwargs):
            if ev.text_delta:
                text_buf.append(ev.text_delta)
            if ev.thinking_delta:
                thinking_buf.append(ev.thinking_delta)
            if ev.tool_call_delta:
                tool_calls.append(ev.tool_call_delta)
            if ev.usage is not None:
                last_usage = ev.usage
            yield ev

        self._finalize_stream(text_buf, thinking_buf, tool_calls, last_usage)

    async def aStream(
        self,
        prompt: str | list[ContentBlock] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        stop: list[str] | None = None,
        tool_choice: str = "auto",
        effort: Any | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._require_started("aStream")
        self._check_no_dangling_tool_use()
        if prompt is not None:
            self._history.append(Message(role="user", content=prompt))

        overrides = self._collect_overrides(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
        )
        kwargs = self._call_kwargs(
            tool_choice=tool_choice,
            stop=stop,
            overrides=overrides,
        )
        self._log_request(kwargs)

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        tool_calls: list[ToolCall] = []
        last_usage = None
        async for ev in self._model.provider._astream_raw(**kwargs):
            if ev.text_delta:
                text_buf.append(ev.text_delta)
            if ev.thinking_delta:
                thinking_buf.append(ev.thinking_delta)
            if ev.tool_call_delta:
                tool_calls.append(ev.tool_call_delta)
            if ev.usage is not None:
                last_usage = ev.usage
            yield ev

        self._finalize_stream(text_buf, thinking_buf, tool_calls, last_usage)

    def _finalize_stream(
        self,
        text_buf: list[str],
        thinking_buf: list[str],
        tool_calls: list[ToolCall],
        usage: Any,
    ) -> None:
        del thinking_buf, usage
        blocks: list[ContentBlock] = []
        if text_buf:
            blocks.append(TextBlock("".join(text_buf)))
        for call in tool_calls:
            blocks.append(ToolUseBlock(id=call.id, name=call.name, input=call.input))
        if blocks:
            self._history.append(Message(role="assistant", content=blocks))

    def _collect_overrides(self, **named: Any) -> dict[AnySetting, Any]:
        out: dict[AnySetting, Any] = {}
        for kw, value in named.items():
            if value is None:
                continue
            setting = _PER_CALL_OVERRIDE_KEYS[kw]
            if not self._model.isAvailable(setting):
                raise UnsupportedFeature(setting, self._model.provider.NAME, self._model.model_id)
            out[setting] = value
        return out

    def _call_kwargs(
        self,
        *,
        tool_choice: str,
        stop: list[str] | None,
        overrides: dict[AnySetting, Any],
    ) -> dict[str, Any]:
        provider = self._model.provider
        max_tokens = overrides.get(
            Settings.DefaultMaxTokens,
            self._model.settings.get(Settings.DefaultMaxTokens, 1024),
        )
        temperature = overrides.get(
            Settings.DefaultTemperature,
            self._model.settings.get(Settings.DefaultTemperature),
        )
        return {
            "model": self._model.model_id,
            "messages": list(self._history),
            "system_blocks": [(b.text, b.cache) for b in self._system_blocks],
            "tools": list(self._tools.values()),
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "provider_settings": provider.settings._snapshot(),
            "model_settings": self._model.settings._snapshot(),
            "convo_settings": self.settings._snapshot(),
            "per_call_overrides": overrides,
        }

    def _check_no_dangling_tool_use(self) -> None:
        """Raise if any assistant tool_use in history lacks a matching tool_result.

        Wire format requires each ToolUseBlock to be answered by a ToolResultBlock
        before the next request. Sending without that produces a 400 from most
        providers, so we fail loudly here instead."""
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
                f"Append a ToolResult for each via AddToolResult() (or use "
                f"llmfacade.helpers.run_bound_tools) before the next Send/Stream."
            )

    def _require_started(self, op: str) -> None:
        if not self._started:
            raise NotStartedError(
                f"Conversation.{op}() requires a Started Conversation. Call Start() first."
            )

    def _require_not_started(self, op: str) -> None:
        if self._started:
            raise SettingsLockedError(f"Conversation.{op}() is not allowed after Start().")

    def _only_text_image(self, blocks: list[ContentBlock]) -> list[Any]:
        return [b for b in blocks if isinstance(b, (TextBlock, ImageBlock))]

    def _log_request(self, kwargs: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        messages: list[Message] = list(kwargs.get("messages", []))
        prior = messages[: self._logged_msg_count]
        new = messages[self._logged_msg_count :]

        record: dict[str, Any] = {
            "type": "request",
            "convo": self.name,
            "model": kwargs.get("model"),
            "system_blocks": kwargs.get("system_blocks"),
            "tools": [t.name for t in kwargs.get("tools", [])],
            "tool_choice": kwargs.get("tool_choice"),
            "new_messages": [
                _dump_message(m, max_lines=self._log_max_message_lines) for m in new
            ],
        }
        if prior:
            rendered = "\n".join(_render_message_oneline(m) for m in prior)
            record["prior_history"] = {
                "messages": len(prior),
                "preview": _abbreviate_lines(rendered),
            }
        self._append_log(record)
        self._logged_msg_count = len(messages)

    def _log_response(self, kwargs: dict[str, Any], resp: Response) -> None:
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
        summary = self._cache_summary(kwargs, resp.usage)
        if summary is not None:
            record["cache_summary"] = summary
        self._append_log(record)
        self._logged_msg_count += 1

    def _cache_summary(self, kwargs: dict[str, Any], usage: Any) -> dict[str, Any] | None:
        """Compute a human-readable cache breakdown from response usage.

        Uses provider-specific token estimates (chars/4 fallback) to map the
        ``cache_read_tokens`` count back to an approximate message-index
        boundary so users can see which part of the prefix was a cache hit."""
        if usage is None:
            return None
        cache_read = usage.cache_read_tokens or 0
        cache_creation = usage.cache_creation_tokens or 0
        prompt_uncached = usage.prompt_tokens or 0
        # For Anthropic, prompt_tokens excludes both cache reads and creations
        # (input_tokens). For OpenAI/Google, prompt_tokens is the total input
        # (cache read counted within it). We standardise as: total_input is
        # the maximum reasonable interpretation either way.
        total_input = max(prompt_uncached + cache_creation + cache_read, prompt_uncached)
        if total_input == 0:
            return None

        boundary = self._estimate_cached_boundary(kwargs, cache_read)
        provider = self._model.provider
        explicit = self._model.isAvailable(ConvoSettings.AutoCacheLastUser)
        auto = bool(
            explicit and self.settings.get(ConvoSettings.AutoCacheLastUser, False)
        )

        if cache_read > 0:
            note = (
                f"Provider cache hit ~{cache_read} tokens "
                f"(~{boundary} of {len(kwargs.get('messages', []))} prefix messages). "
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
                "are off — set ConvoSettings.AutoCacheLastUser=True (or "
                "AddSystemBlock(..., cache=True)) to enable caching on this "
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

        return {
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "uncached_input_tokens": prompt_uncached,
            "hit_ratio": round(cache_read / total_input, 3) if total_input else 0.0,
            "approximate_messages_cached": boundary,
            "tokenizer": _tokenizer_label(provider, self._model.model_id),
            "_note": note,
        }

    def _estimate_cached_boundary(
        self, kwargs: dict[str, Any], cache_read_tokens: int
    ) -> int:
        """Walk system blocks then messages, counting how many messages are
        FULLY covered by the cache_read_tokens count (the boundary may fall
        partway through the next message; we round down). 0 = system-only or
        none; len(msgs) = entire prefix was a cache hit."""
        if cache_read_tokens <= 0:
            return 0
        provider = self._model.provider
        model_id = self._model.model_id
        accumulated = 0
        for text, _cache in kwargs.get("system_blocks") or []:
            accumulated += provider._estimate_tokens(text, model_id)
            if accumulated > cache_read_tokens:
                return 0
        msgs: list[Message] = list(kwargs.get("messages") or [])
        fully_covered = 0
        for i, msg in enumerate(msgs):
            text = _message_to_text(msg)
            tokens = provider._estimate_tokens(text, model_id) if text else 0
            if accumulated + tokens > cache_read_tokens:
                return i
            accumulated += tokens
            fully_covered = i + 1
        return fully_covered

    def _append_log(self, record: dict[str, Any]) -> None:
        assert self._log_path is not None
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_log_default) + "\n")
