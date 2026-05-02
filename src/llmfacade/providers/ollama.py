from __future__ import annotations

import json as _json
import uuid
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from llmfacade.exceptions import ProviderError, ProviderNotInstalledError
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


class OllamaProvider(Provider):
    """Ollama provider - no API key, talks to a local Ollama server.

    See class-level notes for num_ctx and num_predict caveats."""

    NAME = "ollama"
    API_KEY_ENV = None
    SUPPORTS: frozenset[str] = frozenset(
        {
            "context_size",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repeat_penalty",
            "output_format",
            "keep_alive",
            # "tools" is included by default; for a model that doesn't support
            # tool calling at all (e.g. some smaller local models), pass
            # capability_override=provider.SUPPORTS - {"tools"} to new_model().
            "tools",
            # Note: "tool_choice" is intentionally absent. The Ollama chat API
            # has no equivalent of forced tool selection, so any non-default
            # tool_choice is rejected at the cascade as UnsupportedFeature.
        }
    )

    def _init_client(self) -> None:
        try:
            import ollama as _ollama
        except ImportError as e:
            raise ProviderNotInstalledError(
                "Ollama SDK not installed. Run: pip install llmfacade[ollama]"
            ) from e

        client_kwargs: dict[str, Any] = {}
        if self._base_url:
            client_kwargs["host"] = self._base_url
        self._client = _ollama.Client(**client_kwargs)
        self._aclient = _ollama.AsyncClient(**client_kwargs)
        self._module = _ollama

    def _build_chat_kwargs(self, req: CompletionRequest) -> tuple[dict[str, Any], int | None]:
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

        options: dict[str, Any] = {
            "num_predict": req.settings.get("max_tokens", 1024),
        }
        temperature = req.settings.get("temperature")
        if temperature is not None:
            options["temperature"] = temperature
        ctx = req.settings.get("context_size")
        if ctx is not None:
            options["num_ctx"] = ctx
        if req.stop:
            options["stop"] = req.stop
        for setting, key in (
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("repeat_penalty", "repeat_penalty"),
        ):
            value = req.settings.get(setting)
            if value is not None:
                options[key] = value

        chat_kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": api_msgs,
            "options": options,
        }
        keep_alive = req.settings.get("keep_alive")
        if keep_alive is not None:
            chat_kwargs["keep_alive"] = keep_alive

        out_format = req.settings.get("output_format")
        if out_format is not None:
            value = out_format.value if isinstance(out_format, OutputFormat) else out_format
            if value == "json":
                chat_kwargs["format"] = "json"

        if req.tools:
            chat_kwargs["tools"] = [self._tool_to_api(t) for t in req.tools]

        return chat_kwargs, ctx

    def _complete_raw(self, req: CompletionRequest) -> Response:
        chat_kwargs, ctx = self._build_chat_kwargs(req)
        try:
            raw = self._client.chat(**chat_kwargs)
        except Exception as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw, ctx)

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        chat_kwargs, ctx = self._build_chat_kwargs(req)
        try:
            raw = await self._aclient.chat(**chat_kwargs)
        except Exception as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw, ctx)

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        chat_kwargs, ctx = self._build_chat_kwargs(req)
        chat_kwargs["stream"] = True
        try:
            stream = self._client.chat(**chat_kwargs)
            for chunk in stream:
                yield from self._chunk_to_events(chunk, ctx, final=False)
                if getattr(chunk, "done", False):
                    yield from self._chunk_to_events(chunk, ctx, final=True)
        except Exception as e:
            raise ProviderError(str(e), original=e) from e

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        chat_kwargs, ctx = self._build_chat_kwargs(req)
        chat_kwargs["stream"] = True
        try:
            stream = await self._aclient.chat(**chat_kwargs)
            async for chunk in stream:
                for ev in self._chunk_to_events(chunk, ctx, final=False):
                    yield ev
                if getattr(chunk, "done", False):
                    for ev in self._chunk_to_events(chunk, ctx, final=True):
                        yield ev
        except Exception as e:
            raise ProviderError(str(e), original=e) from e

    def _chunk_to_events(
        self, chunk: Any, ctx: int | None, *, final: bool
    ) -> Iterator[StreamEvent]:
        msg = getattr(chunk, "message", None)
        if msg is not None and not final:
            text = getattr(msg, "content", "") or ""
            if text:
                yield StreamEvent(text_delta=text)
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                yield StreamEvent(
                    tool_call_delta=ToolCall(
                        id=f"call-{uuid.uuid4().hex}",
                        name=getattr(fn, "name", ""),
                        input=getattr(fn, "arguments", {}) or {},
                    )
                )
        if final:
            reason = getattr(chunk, "done_reason", None) or "stop"
            yield StreamEvent(
                done=True,
                usage=self._usage_from(chunk, ctx),
                finish_reason=reason,
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

        text_parts: list[str] = []
        images: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for b in m.content:
            if isinstance(b, TextBlock):
                text_parts.append(b.text)
            elif isinstance(b, ImageBlock):
                images.append(b.to_base64())
            elif isinstance(b, ToolUseBlock):
                tool_calls.append(
                    {
                        "function": {"name": b.name, "arguments": b.input},
                    }
                )
            elif isinstance(b, ThinkingBlock):
                # Ollama's chat API has no canonical thinking-block format;
                # local thinking models (deepseek-r1, qwen-thinking) emit
                # reasoning inline and don't require round-trip. Drop.
                continue
        out: dict[str, Any] = {"role": m.role, "content": "".join(text_parts)}
        if images:
            out["images"] = images
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

    def _parse_response(self, raw: Any, ctx: int | None) -> Response:
        msg = getattr(raw, "message", None)
        text = (getattr(msg, "content", "") or "") if msg is not None else ""
        tool_calls: list[ToolCall] = []
        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text))
        if msg is not None:
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args = getattr(fn, "arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        args = {"_raw": args}
                use_id = f"call-{uuid.uuid4().hex}"
                blocks.append(ToolUseBlock(id=use_id, name=getattr(fn, "name", ""), input=args))
                tool_calls.append(ToolCall(id=use_id, name=getattr(fn, "name", ""), input=args))

        return Response(
            text=text,
            blocks=blocks,
            tool_calls=tool_calls,
            thinking=None,
            usage=self._usage_from(raw, ctx),
            finish_reason=getattr(raw, "done_reason", None) or "stop",
            model=getattr(raw, "model", ""),
            raw=raw,
        )

    def _usage_from(self, raw: Any, ctx: int | None) -> Usage | None:
        prompt = getattr(raw, "prompt_eval_count", None)
        completion = getattr(raw, "eval_count", None)
        if prompt is None and completion is None:
            return None
        prompt = prompt or 0
        completion = completion or 0
        if ctx is not None and prompt >= ctx * 0.95:
            warnings.warn(
                f"Ollama evaluated {prompt} prompt tokens against a num_ctx of {ctx} - "
                "input was likely truncated from the front. Increase context_size or shorten "
                "your prompt.",
                stacklevel=2,
            )
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
