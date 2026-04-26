from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from llmfacade.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
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
    Usage,
)
from llmfacade.provider import CompletionRequest, Provider
from llmfacade.settings import (
    AnySetting,
    ConvoSettings,
    OutputFormat,
    ProviderSettings,
    Settings,
)


class GoogleProvider(Provider):
    NAME = "google"
    API_KEY_ENV = "GOOGLE_API_KEY"
    SUPPORTS: frozenset[AnySetting] = frozenset(
        {
            ProviderSettings.BaseURL,
            Settings.ContextSize,
            Settings.DefaultMaxTokens,
            Settings.DefaultTemperature,
            Settings.TopP,
            Settings.TopK,
            ConvoSettings.OutputFormat,
        }
    )

    def _init_client(self) -> None:
        try:
            from google import genai as _genai
        except ImportError as e:
            raise ProviderNotInstalledError(
                "Google GenAI SDK not installed. Run: pip install llmfacade[google]"
            ) from e

        key = self._resolve_key(self.API_KEY_ENV or "GOOGLE_API_KEY")
        self._client = _genai.Client(api_key=key)
        self._module = _genai

    def _build_kwargs(self, req: CompletionRequest) -> dict[str, Any]:
        tool_id_to_name: dict[str, str] = {}
        for m in req.messages:
            if isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, ToolUseBlock):
                        tool_id_to_name[b.id] = b.name
        contents: list[dict[str, Any]] = []
        for m in req.messages:
            contents.extend(self._message_to_api(m, tool_id_to_name))

        config: dict[str, Any] = {
            "max_output_tokens": req.max_tokens,
        }
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.stop:
            config["stop_sequences"] = req.stop
        for setting, key in (
            (Settings.TopP, "top_p"),
            (Settings.TopK, "top_k"),
        ):
            value = req.per_call_overrides.get(setting, req.model_settings.get(setting))
            if value is not None:
                config[key] = value

        if req.system_blocks:
            config["system_instruction"] = "\n\n".join(text for text, _cache in req.system_blocks)

        if req.tools:
            config["tools"] = [
                {"function_declarations": [self._tool_to_api(t) for t in req.tools]}
            ]

        out_format = req.convo_settings.get(ConvoSettings.OutputFormat)
        if out_format is not None:
            value = out_format.value if isinstance(out_format, OutputFormat) else out_format
            if value == "json":
                config["response_mime_type"] = "application/json"

        return {"model": req.model, "contents": contents, "config": config}

    def _complete_raw(self, req: CompletionRequest) -> Response:
        api_kwargs = self._build_kwargs(req)
        try:
            raw = self._client.models.generate_content(**api_kwargs)
        except Exception as e:
            self._reraise(e)
            raise
        return self._parse_response(raw, api_kwargs["model"])

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        api_kwargs = self._build_kwargs(req)
        try:
            raw = await self._client.aio.models.generate_content(**api_kwargs)
        except Exception as e:
            self._reraise(e)
            raise
        return self._parse_response(raw, api_kwargs["model"])

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        try:
            stream = self._client.models.generate_content_stream(**api_kwargs)
            last_usage: Usage | None = None
            for chunk in stream:
                events, usage = self._chunk_to_events(chunk, api_kwargs["model"])
                yield from events
                if usage is not None:
                    last_usage = usage
            yield StreamEvent(done=True, usage=last_usage)
        except Exception as e:
            self._reraise(e)
            raise

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        try:
            stream = await self._client.aio.models.generate_content_stream(**api_kwargs)
            last_usage: Usage | None = None
            async for chunk in stream:
                events, usage = self._chunk_to_events(chunk, api_kwargs["model"])
                for ev in events:
                    yield ev
                if usage is not None:
                    last_usage = usage
            yield StreamEvent(done=True, usage=last_usage)
        except Exception as e:
            self._reraise(e)
            raise

    def _chunk_to_events(self, chunk: Any, _model: str) -> tuple[list[StreamEvent], Usage | None]:
        events: list[StreamEvent] = []
        text = getattr(chunk, "text", None)
        if text:
            events.append(StreamEvent(text_delta=text))
        for cand in getattr(chunk, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    events.append(
                        StreamEvent(
                            tool_call_delta=ToolCall(
                                id=f"call-{uuid.uuid4().hex}",
                                name=getattr(fn_call, "name", ""),
                                input=dict(getattr(fn_call, "args", {}) or {}),
                            )
                        )
                    )
        usage = self._usage_from(chunk)
        return events, usage

    def _message_to_api(
        self, m: Message, tool_id_to_name: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        role = "model" if m.role == "assistant" else "user"
        if m.role == "tool":
            parts = []
            blocks = m.content if isinstance(m.content, list) else []
            for b in blocks:
                if isinstance(b, ToolResultBlock):
                    text = b.content if isinstance(b.content, str) else ""
                    fn_name = b.name or (
                        (tool_id_to_name or {}).get(b.tool_use_id) or b.tool_use_id
                    )
                    parts.append(
                        {
                            "function_response": {
                                "name": fn_name,
                                "response": {"content": text},
                            }
                        }
                    )
            return [{"role": "user", "parts": parts}] if parts else []

        if isinstance(m.content, str):
            return [{"role": role, "parts": [{"text": m.content}]}]

        parts: list[dict[str, Any]] = []
        for b in m.content:
            if isinstance(b, TextBlock):
                parts.append({"text": b.text})
            elif isinstance(b, ImageBlock):
                parts.append(
                    {
                        "inline_data": {"mime_type": b.media_type, "data": b.to_base64()},
                    }
                )
            elif isinstance(b, ToolUseBlock):
                parts.append(
                    {
                        "function_call": {"name": b.name, "args": b.input},
                    }
                )
        return [{"role": role, "parts": parts}]

    def _tool_to_api(self, t: Any) -> dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema,
        }

    def _parse_response(self, raw: Any, model: str) -> Response:
        text = getattr(raw, "text", "") or ""
        blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []
        if text:
            blocks.append(TextBlock(text))
        for cand in getattr(raw, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    name = getattr(fn_call, "name", "")
                    args = dict(getattr(fn_call, "args", {}) or {})
                    use_id = f"call-{uuid.uuid4().hex}"
                    blocks.append(ToolUseBlock(id=use_id, name=name, input=args))
                    tool_calls.append(ToolCall(id=use_id, name=name, input=args))

        return Response(
            text=text,
            blocks=blocks,
            tool_calls=tool_calls,
            thinking=None,
            usage=self._usage_from(raw),
            finish_reason=None,
            model=model,
            raw=raw,
        )

    def _usage_from(self, raw: Any) -> Usage | None:
        um = getattr(raw, "usage_metadata", None)
        if um is None:
            return None
        prompt = getattr(um, "prompt_token_count", 0) or 0
        completion = getattr(um, "candidates_token_count", 0) or 0
        cached = getattr(um, "cached_content_token_count", 0) or 0
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cache_read_tokens=cached,
        )

    def _reraise(self, e: Exception) -> None:
        err_name = type(e).__name__.lower()
        if "authentication" in err_name or "permission" in err_name:
            raise AuthenticationError(str(e)) from e
        if "resource_exhausted" in err_name or "rate" in err_name:
            raise RateLimitError(str(e)) from e
        raise ProviderError(str(e), original=e) from e
