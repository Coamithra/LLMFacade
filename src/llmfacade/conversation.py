from __future__ import annotations

import copy
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import (
    NotStartedError,
    SettingsLockedError,
    ToolIterationLimitError,
    UnsupportedFeature,
)
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

    def SetLogging(self, path: str | Path | None) -> None:
        self._require_not_started("SetLogging")
        self._log_path = Path(path) if path is not None else None
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

    def Clone(self) -> Conversation:
        clone = Conversation(model=self._model, name=f"{self.name}-clone")
        clone._system_blocks = copy.deepcopy(self._system_blocks)
        clone._history = copy.deepcopy(self._history)
        clone._tools = dict(self._tools)
        clone.settings._values = dict(self.settings._values)
        clone._log_path = self._log_path
        return clone

    def Complete(
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
        auto_tools: bool = True,
        max_tool_iterations: int = 16,
    ) -> Response:
        self._require_started("Complete")
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

        for _ in range(max_tool_iterations):
            kwargs = self._call_kwargs(
                tool_choice=tool_choice,
                stop=stop,
                overrides=overrides,
            )
            self._log_request(kwargs)
            resp = self._model.provider._complete_raw(**kwargs)
            self._log_response(resp)
            self._history.append(Message(role="assistant", content=list(resp.blocks)))
            self._bind_tool_fns(resp.tool_calls)

            if not auto_tools or not resp.tool_calls:
                return resp

            for call in resp.tool_calls:
                self._dispatch_tool_call(call)

        raise ToolIterationLimitError(
            f"Auto-tool dispatch exceeded max_tool_iterations={max_tool_iterations}. "
            f"The model kept calling tools without producing a final answer."
        )

    async def aComplete(
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
        auto_tools: bool = True,
        max_tool_iterations: int = 16,
    ) -> Response:
        self._require_started("aComplete")
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

        for _ in range(max_tool_iterations):
            kwargs = self._call_kwargs(
                tool_choice=tool_choice,
                stop=stop,
                overrides=overrides,
            )
            self._log_request(kwargs)
            resp = await self._model.provider._acomplete_raw(**kwargs)
            self._log_response(resp)
            self._history.append(Message(role="assistant", content=list(resp.blocks)))
            self._bind_tool_fns(resp.tool_calls)

            if not auto_tools or not resp.tool_calls:
                return resp

            for call in resp.tool_calls:
                await self._adispatch_tool_call(call)

        raise ToolIterationLimitError(
            f"Auto-tool dispatch exceeded max_tool_iterations={max_tool_iterations}. "
            f"The model kept calling tools without producing a final answer."
        )

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

    def _bind_tool_fns(self, calls: list[ToolCall]) -> None:
        for i, call in enumerate(calls):
            tool_def = self._tools.get(call.name)
            if tool_def is None:
                continue
            calls[i] = ToolCall(id=call.id, name=call.name, input=call.input, _fn=tool_def.fn)

    def _dispatch_tool_call(self, call: ToolCall) -> None:
        try:
            result = call.invoke()
            self.AddToolResult(call.id, _stringify_tool_result(result), name=call.name)
        except Exception as e:
            self.AddToolResult(call.id, f"Tool error: {e}", is_error=True, name=call.name)

    async def _adispatch_tool_call(self, call: ToolCall) -> None:
        try:
            result = await call.ainvoke()
            self.AddToolResult(call.id, _stringify_tool_result(result), name=call.name)
        except Exception as e:
            self.AddToolResult(call.id, f"Tool error: {e}", is_error=True, name=call.name)

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
        record = {
            "type": "request",
            "convo": self.name,
            "model": kwargs.get("model"),
            "system_blocks": kwargs.get("system_blocks"),
            "messages": [_dump_message(m) for m in kwargs.get("messages", [])],
            "tools": [t.name for t in kwargs.get("tools", [])],
            "tool_choice": kwargs.get("tool_choice"),
        }
        self._append_log(record)

    def _log_response(self, resp: Response) -> None:
        if self._log_path is None:
            return
        record = {
            "type": "response",
            "convo": self.name,
            "model": resp.model,
            "text": resp.text,
            "tool_calls": [
                {"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls
            ],
            "thinking": resp.thinking,
            "usage": _dump_usage(resp.usage),
            "finish_reason": resp.finish_reason,
        }
        self._append_log(record)

    def _append_log(self, record: dict[str, Any]) -> None:
        assert self._log_path is not None
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_log_default) + "\n")


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


def _dump_message(m: Message) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}
    return {
        "role": m.role,
        "content": [_dump_block(b) for b in m.content],
    }


def _dump_block(b: ContentBlock) -> dict[str, Any]:
    cls = type(b).__name__
    if isinstance(b, TextBlock):
        return {"type": cls, "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"type": cls, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {"type": cls, "tool_use_id": b.tool_use_id, "is_error": b.is_error}
    if isinstance(b, ImageBlock):
        return {"type": cls, "media_type": b.media_type, "bytes": len(b.data)}
    return {"type": cls}


def _dump_usage(u: Any) -> dict[str, int] | None:
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
