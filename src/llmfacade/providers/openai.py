from __future__ import annotations

import json as _json
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from llmfacade.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
)
from llmfacade.helpers import flatten_text_blocks
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
from llmfacade.provider import CompletionRequest, Provider
from llmfacade.settings import OutputFormat


def _openai_cached_tokens(usage: Any) -> int:
    """Pull cached prompt-token count from OpenAI usage. Lives in
    ``prompt_tokens_details.cached_tokens`` on chat-completion responses."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "cached_tokens", 0) or 0


class OpenAIProvider(Provider):
    NAME = "openai"
    API_KEY_ENV = "OPENAI_API_KEY"
    SUPPORTS: frozenset[str] = frozenset(
        {
            "max_tokens",
            "temperature",
            "top_p",
            "output_format",
            "tools",
            "tool_choice",
        }
    )

    def __init__(self, *, org_id: str | None = None, **kwargs: Any):
        self._org_id = org_id
        super().__init__(**kwargs)

    def _init_client(self) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ProviderNotInstalledError(
                "OpenAI SDK not installed. Run: pip install llmfacade[openai]"
            ) from e

        key = self._resolve_key(self.API_KEY_ENV or "OPENAI_API_KEY")
        client_kwargs: dict[str, Any] = {"api_key": key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self._org_id:
            client_kwargs["organization"] = self._org_id
        self._client = _openai.OpenAI(**client_kwargs)
        self._aclient = _openai.AsyncOpenAI(**client_kwargs)
        self._module = _openai

    _tiktoken_cache: dict[str, Any] = {}

    def count_tokens(
        self,
        text: str,
        *,
        system: str | None = None,
        model_id: str | None = None,
    ) -> int:
        try:
            import tiktoken
        except ImportError:
            return super().count_tokens(text, system=system, model_id=model_id)
        key = model_id or "__default__"
        enc = self._tiktoken_cache.get(key)
        if enc is None:
            if model_id:
                try:
                    enc = tiktoken.encoding_for_model(model_id)
                except KeyError:
                    enc = tiktoken.get_encoding("o200k_base")
            else:
                enc = tiktoken.get_encoding("o200k_base")
            self._tiktoken_cache[key] = enc
        n = len(enc.encode(text))
        if system:
            n += len(enc.encode(system))
        return n

    def tokenizer_name(self, *, model_id: str | None = None) -> str:
        del model_id
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            return "chars/4 (tiktoken not installed)"
        return "tiktoken"

    def _build_kwargs(self, req: CompletionRequest) -> dict[str, Any]:
        api_msgs: list[dict[str, Any]] = []
        if req.system_blocks:
            api_msgs.append(
                {
                    "role": "system",
                    "content": "\n\n".join(sb.text for sb in req.system_blocks),
                }
            )
        for m in req.messages:
            api_msgs.extend(self._message_to_api(m))

        api_kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": api_msgs,
            "max_tokens": req.settings.get("max_tokens", 1024),
        }
        temperature = req.settings.get("temperature")
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if req.stop:
            api_kwargs["stop"] = req.stop
        top_p = req.settings.get("top_p")
        if top_p is not None:
            api_kwargs["top_p"] = top_p

        if req.tools:
            api_kwargs["tools"] = [self._tool_to_api(t) for t in req.tools]
            api_kwargs["tool_choice"] = self._tool_choice_to_api(
                req.settings.get("tool_choice", "auto")
            )

        out_format = req.settings.get("output_format")
        if out_format is not None:
            value = out_format.value if isinstance(out_format, OutputFormat) else out_format
            if value == "json":
                api_kwargs["response_format"] = {"type": "json_object"}

        return api_kwargs

    def _complete_raw(self, req: CompletionRequest) -> Response:
        api_kwargs = self._build_kwargs(req)
        try:
            raw = self._client.chat.completions.create(**api_kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw)

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        api_kwargs = self._build_kwargs(req)
        try:
            raw = await self._aclient.chat.completions.create(**api_kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw)

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        api_kwargs["stream"] = True
        api_kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = self._client.chat.completions.create(**api_kwargs)
            tool_buf: dict[int, dict[str, Any]] = {}
            state: dict[str, Any] = {"finish_reason": None}
            for chunk in stream:
                yield from self._chunk_to_events(chunk, tool_buf, state)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        api_kwargs["stream"] = True
        api_kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await self._aclient.chat.completions.create(**api_kwargs)
            tool_buf: dict[int, dict[str, Any]] = {}
            state: dict[str, Any] = {"finish_reason": None}
            async for chunk in stream:
                for ev in self._chunk_to_events(chunk, tool_buf, state):
                    yield ev
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    def _chunk_to_events(
        self,
        chunk: Any,
        tool_buf: dict[int, dict[str, Any]],
        state: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                yield StreamEvent(text_delta=text)
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
            choice_finish = getattr(choice, "finish_reason", None)
            if choice_finish is not None:
                state["finish_reason"] = choice_finish
                for slot in tool_buf.values():
                    if slot["id"] is None:
                        continue
                    try:
                        parsed = _json.loads(slot["args"] or "{}")
                    except _json.JSONDecodeError:
                        parsed = {}
                    yield StreamEvent(
                        tool_call_delta=ToolCall(
                            id=slot["id"], name=slot["name"] or "", input=parsed
                        )
                    )
                tool_buf.clear()
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield StreamEvent(
                done=True,
                usage=Usage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    cache_read_tokens=_openai_cached_tokens(usage),
                ),
                finish_reason=state.get("finish_reason"),
            )

    def _message_to_api(self, m: Message) -> list[dict[str, Any]]:
        if m.role == "tool":
            results: list[dict[str, Any]] = []
            blocks = m.content if isinstance(m.content, list) else []
            for b in blocks:
                if isinstance(b, ToolResultBlock):
                    text = (
                        b.content if isinstance(b.content, str) else flatten_text_blocks(b.content)
                    )
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
                # OpenAI Chat Completions can't round-trip reasoning content;
                # the Responses API can but isn't wired up here. Drop on the
                # way out so we don't send invalid payloads.
                continue
        out: dict[str, Any] = {"role": m.role}
        if parts:
            if m.role == "user":
                out["content"] = parts
            else:
                if has_image:
                    warnings.warn(
                        f"OpenAI: dropping image block(s) on {m.role!r} message; "
                        f"the OpenAI Chat Completions API only accepts images on "
                        f"user messages.",
                        stacklevel=3,
                    )
                out["content"] = "".join(
                    p.get("text", "") for p in parts if p.get("type") == "text"
                )
        else:
            out["content"] = None if tool_calls else ""
        if tool_calls:
            out["tool_calls"] = tool_calls
        return [out]

    def _tool_to_api(self, t: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        }

    def _tool_choice_to_api(self, tc: str) -> str | dict[str, Any]:
        if tc == "auto":
            return "auto"
        if tc == "required":
            return "required"
        if tc == "none":
            return "none"
        return {"type": "function", "function": {"name": tc}}

    def _parse_response(self, raw: Any) -> Response:
        choice = raw.choices[0]
        msg = choice.message
        text = getattr(msg, "content", "") or ""
        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text))
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = _json.loads(tc.function.arguments)
            except _json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
                cache_read_tokens=_openai_cached_tokens(raw.usage),
            )

        return Response(
            text=text,
            blocks=blocks,
            tool_calls=tool_calls,
            thinking=None,
            usage=usage,
            finish_reason=choice.finish_reason,
            model=raw.model,
            raw=raw,
        )
