from __future__ import annotations

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
    EffortLevel,
    EphemeralCacheTTL,
    ProviderSettings,
    Settings,
)


class AnthropicProvider(Provider):
    NAME = "anthropic"
    API_KEY_ENV = "ANTHROPIC_API_KEY"
    SUPPORTS: frozenset[AnySetting] = frozenset(
        {
            ProviderSettings.BaseURL,
            ProviderSettings.BetaHeaders,
            Settings.ContextSize,
            Settings.DefaultMaxTokens,
            Settings.DefaultTemperature,
            Settings.TopP,
            Settings.TopK,
            Settings.Effort,
            Settings.Thinking,
            ConvoSettings.AutoCacheLastUser,
            ConvoSettings.UserMetadata,
            ConvoSettings.CacheTTL,
        }
    )

    # Models that don't support extended thinking get this override.
    _NO_THINKING_MODELS = {"claude-haiku-3-5", "claude-3-5-haiku-20241022"}

    def _init_client(self) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ProviderNotInstalledError(
                "Anthropic SDK not installed. Run: pip install llmfacade[anthropic]"
            ) from e

        key = self._resolve_key(self.API_KEY_ENV or "ANTHROPIC_API_KEY")
        client_kwargs: dict[str, Any] = {"api_key": key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        self._client = _anthropic.Anthropic(**client_kwargs)
        self._aclient = _anthropic.AsyncAnthropic(**client_kwargs)
        self._module = _anthropic

    def NewModel(self, model_id: str, **kwargs: Any):
        from llmfacade.model import Model

        override = None
        if any(m in model_id for m in self._NO_THINKING_MODELS):
            override = self.SUPPORTS - {Settings.Thinking}
        return Model(
            provider=self,
            model_id=model_id,
            capability_override=override,
            **kwargs,
        )

    def _build_kwargs(self, req: CompletionRequest) -> dict[str, Any]:
        ttl = req.convo_settings.get(ConvoSettings.CacheTTL)
        if isinstance(ttl, EphemeralCacheTTL):
            ttl_value = ttl.value
        elif isinstance(ttl, str):
            ttl_value = ttl
        else:
            ttl_value = None  # SDK default (5m)

        api_msgs = self._messages_to_api(
            req.messages,
            auto_cache_last=bool(req.convo_settings.get(ConvoSettings.AutoCacheLastUser)),
            ttl=ttl_value,
        )
        api_kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": api_msgs,
        }
        if req.temperature is not None:
            api_kwargs["temperature"] = req.temperature
        if req.stop:
            api_kwargs["stop_sequences"] = req.stop

        sys_blocks = self._system_to_api(req.system_blocks, ttl=ttl_value)
        if sys_blocks:
            api_kwargs["system"] = sys_blocks

        if req.tools:
            api_kwargs["tools"] = [self._tool_to_api(t) for t in req.tools]
            api_kwargs["tool_choice"] = self._tool_choice_to_api(req.tool_choice)

        thinking_val = req.per_call_overrides.get(
            Settings.Thinking, req.model_settings.get(Settings.Thinking)
        )
        if thinking_val is not None:
            budget = thinking_val if isinstance(thinking_val, int) else int(thinking_val)
            api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        effort = req.per_call_overrides.get(
            Settings.Effort, req.model_settings.get(Settings.Effort)
        )
        if effort is not None:
            api_kwargs["effort"] = effort.value if isinstance(effort, EffortLevel) else effort

        for key in ("TopP", "TopK"):
            setting = getattr(Settings, key)
            value = req.per_call_overrides.get(setting, req.model_settings.get(setting))
            if value is not None:
                api_kwargs["top_p" if key == "TopP" else "top_k"] = value

        metadata = req.convo_settings.get(ConvoSettings.UserMetadata)
        if metadata:
            api_kwargs["metadata"] = metadata

        beta_headers = req.provider_settings.get(ProviderSettings.BetaHeaders)
        if beta_headers:
            api_kwargs["extra_headers"] = {"anthropic-beta": ",".join(beta_headers)}

        return api_kwargs

    def _complete_raw(self, req: CompletionRequest) -> Response:
        api_kwargs = self._build_kwargs(req)
        try:
            raw = self._client.messages.create(**api_kwargs)
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
            raw = await self._aclient.messages.create(**api_kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw)

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        try:
            with self._client.messages.stream(**api_kwargs) as stream:
                yield from self._iter_stream_events(stream)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        try:
            async with self._aclient.messages.stream(**api_kwargs) as stream:
                async for ev in self._aiter_stream_events(stream):
                    yield ev
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    def _iter_stream_events(self, stream: Any) -> Iterator[StreamEvent]:
        current_tool: dict[str, Any] | None = None
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block is not None and getattr(block, "type", None) == "tool_use":
                    current_tool = {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input_json": "",
                    }
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                d_type = getattr(delta, "type", None)
                if d_type == "text_delta":
                    yield StreamEvent(text_delta=getattr(delta, "text", ""))
                elif d_type == "thinking_delta":
                    yield StreamEvent(thinking_delta=getattr(delta, "thinking", ""))
                elif d_type == "input_json_delta" and current_tool is not None:
                    current_tool["input_json"] += getattr(delta, "partial_json", "")
            elif event_type == "content_block_stop" and current_tool is not None:
                import json as _json

                try:
                    parsed = _json.loads(current_tool["input_json"] or "{}")
                except _json.JSONDecodeError:
                    parsed = {}
                yield StreamEvent(
                    tool_call_delta=ToolCall(
                        id=current_tool["id"],
                        name=current_tool["name"],
                        input=parsed,
                    )
                )
                current_tool = None
            elif event_type == "message_stop":
                msg = stream.get_final_message()
                yield StreamEvent(done=True, usage=self._usage_from(msg))

    async def _aiter_stream_events(self, stream: Any) -> AsyncIterator[StreamEvent]:
        current_tool: dict[str, Any] | None = None
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block is not None and getattr(block, "type", None) == "tool_use":
                    current_tool = {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input_json": "",
                    }
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                d_type = getattr(delta, "type", None)
                if d_type == "text_delta":
                    yield StreamEvent(text_delta=getattr(delta, "text", ""))
                elif d_type == "thinking_delta":
                    yield StreamEvent(thinking_delta=getattr(delta, "thinking", ""))
                elif d_type == "input_json_delta" and current_tool is not None:
                    current_tool["input_json"] += getattr(delta, "partial_json", "")
            elif event_type == "content_block_stop" and current_tool is not None:
                import json as _json

                try:
                    parsed = _json.loads(current_tool["input_json"] or "{}")
                except _json.JSONDecodeError:
                    parsed = {}
                yield StreamEvent(
                    tool_call_delta=ToolCall(
                        id=current_tool["id"],
                        name=current_tool["name"],
                        input=parsed,
                    )
                )
                current_tool = None
            elif event_type == "message_stop":
                msg = await stream.get_final_message()
                yield StreamEvent(done=True, usage=self._usage_from(msg))

    def _system_to_api(
        self, blocks: list[tuple[str, bool]], *, ttl: str | None = None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for text, cache in blocks:
            entry: dict[str, Any] = {"type": "text", "text": text}
            if cache:
                cc: dict[str, Any] = {"type": "ephemeral"}
                if ttl:
                    cc["ttl"] = ttl
                entry["cache_control"] = cc
            out.append(entry)
        return out

    def _messages_to_api(
        self,
        messages: list[Message],
        *,
        auto_cache_last: bool,
        ttl: str | None = None,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for m in messages:
            api_role = "user" if m.role in ("user", "tool") else "assistant"
            content = self._content_to_api(m.content)
            if (
                merged
                and merged[-1]["role"] == api_role
                and isinstance(merged[-1]["content"], list)
                and isinstance(content, list)
            ):
                merged[-1]["content"].extend(content)
            else:
                merged.append({"role": api_role, "content": content})

        if auto_cache_last and merged:
            last = merged[-1]
            if last["role"] == "user" and isinstance(last["content"], list) and last["content"]:
                last_block = last["content"][-1]
                if isinstance(last_block, dict):
                    cc: dict[str, Any] = {"type": "ephemeral"}
                    if ttl:
                        cc["ttl"] = ttl
                    last_block["cache_control"] = cc

        return merged

    def _content_to_api(self, content: str | list[ContentBlock]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        out: list[dict[str, Any]] = []
        for b in content:
            if isinstance(b, TextBlock):
                out.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageBlock):
                out.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": b.media_type,
                            "data": b.to_base64(),
                        },
                    }
                )
            elif isinstance(b, ToolUseBlock):
                out.append(
                    {
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input,
                    }
                )
            elif isinstance(b, ToolResultBlock):
                inner: Any
                if isinstance(b.content, str):
                    inner = b.content
                else:
                    inner = []
                    for inner_b in b.content:
                        if isinstance(inner_b, TextBlock):
                            inner.append({"type": "text", "text": inner_b.text})
                        elif isinstance(inner_b, ImageBlock):
                            inner.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": inner_b.media_type,
                                        "data": inner_b.to_base64(),
                                    },
                                }
                            )
                entry: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": inner,
                }
                if b.is_error:
                    entry["is_error"] = True
                out.append(entry)
        return out

    def _tool_to_api(self, t: Any) -> dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema,
        }

    def _tool_choice_to_api(self, tc: str) -> dict[str, Any]:
        if tc == "auto":
            return {"type": "auto"}
        if tc == "required":
            return {"type": "any"}
        if tc == "none":
            return {"type": "auto", "disable_parallel_tool_use": False}
        return {"type": "tool", "name": tc}

    def _parse_response(self, raw: Any) -> Response:
        blocks: list[ContentBlock] = []
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for b in getattr(raw, "content", []):
            b_type = getattr(b, "type", None)
            if b_type == "text":
                text = getattr(b, "text", "")
                blocks.append(TextBlock(text))
                text_parts.append(text)
            elif b_type == "thinking":
                t = getattr(b, "thinking", "")
                thinking_parts.append(t)
            elif b_type == "tool_use":
                use_id = getattr(b, "id", "")
                name = getattr(b, "name", "")
                inp = getattr(b, "input", {}) or {}
                blocks.append(ToolUseBlock(id=use_id, name=name, input=inp))
                tool_calls.append(ToolCall(id=use_id, name=name, input=inp))

        return Response(
            text="".join(text_parts),
            blocks=blocks,
            tool_calls=tool_calls,
            thinking="".join(thinking_parts) or None,
            usage=self._usage_from(raw),
            finish_reason=getattr(raw, "stop_reason", None),
            model=getattr(raw, "model", ""),
            raw=raw,
        )

    def _usage_from(self, raw: Any) -> Usage | None:
        u = getattr(raw, "usage", None)
        if u is None:
            return None
        prompt = getattr(u, "input_tokens", 0) or 0
        completion = getattr(u, "output_tokens", 0) or 0
        cache_creation = getattr(u, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )
