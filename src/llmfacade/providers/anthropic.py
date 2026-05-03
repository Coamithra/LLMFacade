from __future__ import annotations

import json as _json
import warnings
from collections.abc import AsyncIterator, Iterator
from enum import Enum
from typing import TYPE_CHECKING, Any

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
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from llmfacade.provider import CompletionRequest, Provider, SystemBlock
from llmfacade.settings import EffortLevel, EphemeralCacheTTL

if TYPE_CHECKING:
    from llmfacade.model import Model

_SUPPORTS: frozenset[str] = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "effort",
        "thinking",
        "auto_cache_last_user",
        "auto_cache_tools",
        "user_metadata",
        "cache_ttl",
        "beta_headers",
        "tools",
        "tool_choice",
    }
)


class AnthropicModel(Enum):
    """Known Anthropic model ids with capability metadata.

    Pass an enum member to `provider.new_model()` and both the canonical model
    id and a matching `capability_override` are applied automatically. Pass a
    raw string instead to opt out — the provider's full SUPPORTS set is used
    and the caller is responsible for narrowing via `capability_override=` if
    the model needs it.

    This enum is a snapshot of what the library knows about as of its release.
    Use a string for any model not listed here (new releases between library
    versions, fine-tunes, custom deployments)."""

    OPUS_4_7 = ("claude-opus-4-7", _SUPPORTS)
    SONNET_4_6 = ("claude-sonnet-4-6", _SUPPORTS)
    HAIKU_4_5 = ("claude-haiku-4-5-20251001", _SUPPORTS)

    def __init__(self, model_id: str, capabilities: frozenset[str]):
        self.model_id = model_id
        self.capabilities = capabilities


_EXACT_COUNT_FALLBACK_WARNED: set[str] = set()


class AnthropicProvider(Provider):
    NAME = "anthropic"
    API_KEY_ENV = "ANTHROPIC_API_KEY"
    SUPPORTS: frozenset[str] = _SUPPORTS

    def __init__(self, *, exact_count_tokens: bool = False, **kwargs: Any):
        """``exact_count_tokens=True`` makes ``count_tokens`` call the
        Anthropic SDK's free server-side ``messages.count_tokens`` endpoint
        for exact counts under the model's actual tokenizer. Default is
        ``False``: the base ``chars/4`` approximation is used and no network
        call is made. Enable this for callers that need accurate counts
        (e.g. chunk planning over long inputs); leave it off for strict
        offline behaviour. Network/SDK errors fall back to ``chars/4`` with
        a one-time warning per error type."""
        self._exact_count_tokens = exact_count_tokens
        super().__init__(**kwargs)

    def count_tokens(
        self,
        text: str,
        *,
        system: str | None = None,
        model_id: str | None = None,
    ) -> int:
        if not self._exact_count_tokens or (not text and not system):
            return super().count_tokens(text, system=system, model_id=model_id)
        if model_id is None:
            raise ValueError(
                "AnthropicProvider.count_tokens with exact_count_tokens=True "
                "requires a model_id; use Model.count_tokens(text) or pass "
                "model_id= explicitly."
            )
        api_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": text}],
        }
        if system:
            api_kwargs["system"] = system
        try:
            result = self._client.messages.count_tokens(**api_kwargs)
            return int(result.input_tokens)
        except self._module.AuthenticationError as e:
            self._warn_exact_count_fallback("AuthenticationError", e)
            return super().count_tokens(text, system=system, model_id=model_id)
        except self._module.RateLimitError as e:
            self._warn_exact_count_fallback("RateLimitError", e)
            return super().count_tokens(text, system=system, model_id=model_id)
        except self._module.APIError as e:
            self._warn_exact_count_fallback("APIError", e)
            return super().count_tokens(text, system=system, model_id=model_id)
        except Exception as e:
            self._warn_exact_count_fallback(type(e).__name__, e)
            return super().count_tokens(text, system=system, model_id=model_id)

    def tokenizer_name(self, *, model_id: str | None = None) -> str:
        if self._exact_count_tokens:
            return "anthropic-server"
        return super().tokenizer_name(model_id=model_id)

    @staticmethod
    def _warn_exact_count_fallback(error_type: str, exc: Exception) -> None:
        if error_type in _EXACT_COUNT_FALLBACK_WARNED:
            return
        _EXACT_COUNT_FALLBACK_WARNED.add(error_type)
        warnings.warn(
            f"AnthropicProvider.count_tokens server-side call failed "
            f"({error_type}: {exc}); falling back to chars/4. Subsequent "
            f"failures of this type will be silent.",
            stacklevel=3,
        )

    def new_model(
        self,
        model_id: AnthropicModel | str,
        *,
        capability_override: frozenset[str] | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
        tool_choice: str | None = None,
    ) -> Model:
        """Bind a model id (or `AnthropicModel` member) to this provider.

        If `model_id` is an `AnthropicModel` enum member, its `.capabilities`
        are applied as `capability_override` automatically. If `model_id` is a
        raw string, the provider's full SUPPORTS set is used — pass
        `capability_override=` if the model needs narrowing. An explicit
        `capability_override=` always wins, even when an enum member is
        passed."""
        if isinstance(model_id, AnthropicModel):
            if capability_override is None:
                capability_override = model_id.capabilities
            model_id = model_id.model_id
        return super().new_model(
            model_id,
            capability_override=capability_override,
            log_dir=log_dir,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
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

    def _build_kwargs(self, req: CompletionRequest) -> dict[str, Any]:
        ttl_raw = req.settings.get("cache_ttl")
        if isinstance(ttl_raw, EphemeralCacheTTL):
            ttl_value: str | None = ttl_raw.value
        elif isinstance(ttl_raw, str):
            ttl_value = ttl_raw
        else:
            ttl_value = None  # SDK default (5m)

        max_tokens = req.settings.get("max_tokens", 1024)
        api_msgs = self._messages_to_api(
            req.messages,
            auto_cache_last=bool(req.settings.get("auto_cache_last_user")),
            ttl=ttl_value,
        )
        api_kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": max_tokens,
            "messages": api_msgs,
        }
        temperature = req.settings.get("temperature")
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if req.stop:
            api_kwargs["stop_sequences"] = req.stop

        sys_blocks = self._system_to_api(req.system_blocks, ttl=ttl_value)
        if sys_blocks:
            api_kwargs["system"] = sys_blocks

        if req.tools:
            api_tools = [self._tool_to_api(t) for t in req.tools]
            if req.settings.get("auto_cache_tools") and api_tools:
                cc: dict[str, Any] = {"type": "ephemeral"}
                if ttl_value:
                    cc["ttl"] = ttl_value
                api_tools[-1]["cache_control"] = cc
            api_kwargs["tools"] = api_tools
            api_kwargs["tool_choice"] = self._tool_choice_to_api(
                req.settings.get("tool_choice", "auto")
            )

        thinking_val = req.settings.get("thinking")
        if thinking_val is not None:
            budget = thinking_val if isinstance(thinking_val, int) else int(thinking_val)
            api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        effort = req.settings.get("effort")
        if effort is not None:
            value = effort.value if isinstance(effort, EffortLevel) else effort
            # Anthropic SDK rejects a top-level effort= kwarg; the API expects
            # output_config={"effort": "..."} on messages.create.
            api_kwargs["output_config"] = {"effort": value}

        for key in ("top_p", "top_k"):
            value = req.settings.get(key)
            if value is not None:
                api_kwargs[key] = value

        metadata = req.settings.get("user_metadata")
        if metadata:
            api_kwargs["metadata"] = metadata

        beta_headers = req.settings.get("beta_headers")
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
                state: dict[str, Any] = {"current_tool": None, "current_thinking": None}
                for event in stream:
                    if getattr(event, "type", None) == "message_stop":
                        msg = stream.get_final_message()
                        yield StreamEvent(
                            done=True,
                            usage=self._usage_from(msg),
                            finish_reason=getattr(msg, "stop_reason", None),
                        )
                    else:
                        yield from self._chunk_to_events(event, state)
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
                state: dict[str, Any] = {"current_tool": None, "current_thinking": None}
                async for event in stream:
                    if getattr(event, "type", None) == "message_stop":
                        msg = await stream.get_final_message()
                        yield StreamEvent(
                            done=True,
                            usage=self._usage_from(msg),
                            finish_reason=getattr(msg, "stop_reason", None),
                        )
                    else:
                        for ev in self._chunk_to_events(event, state):
                            yield ev
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    def _chunk_to_events(self, event: Any, state: dict[str, Any]) -> Iterator[StreamEvent]:
        event_type = getattr(event, "type", None)
        if event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            block_type = getattr(block, "type", None) if block is not None else None
            if block_type == "tool_use":
                state["current_tool"] = {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input_json": "",
                }
            elif block_type == "thinking":
                state["current_thinking"] = {"text": "", "signature": None}
            elif block_type == "redacted_thinking":
                data = getattr(block, "data", "") or ""
                yield StreamEvent(
                    thinking_block=ThinkingBlock(
                        text="", encrypted=True, provider_data={"data": data}
                    )
                )
        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            d_type = getattr(delta, "type", None)
            if d_type == "text_delta":
                yield StreamEvent(text_delta=getattr(delta, "text", ""))
            elif d_type == "thinking_delta":
                t = getattr(delta, "thinking", "")
                if state["current_thinking"] is not None:
                    state["current_thinking"]["text"] += t
                yield StreamEvent(thinking_delta=t)
            elif d_type == "signature_delta" and state["current_thinking"] is not None:
                state["current_thinking"]["signature"] = (
                    state["current_thinking"]["signature"] or ""
                ) + (getattr(delta, "signature", "") or "")
            elif d_type == "input_json_delta" and state["current_tool"] is not None:
                state["current_tool"]["input_json"] += getattr(delta, "partial_json", "")
        elif event_type == "content_block_stop":
            if state["current_tool"] is not None:
                try:
                    parsed = _json.loads(state["current_tool"]["input_json"] or "{}")
                except _json.JSONDecodeError:
                    parsed = {}
                yield StreamEvent(
                    tool_call_delta=ToolCall(
                        id=state["current_tool"]["id"],
                        name=state["current_tool"]["name"],
                        input=parsed,
                    )
                )
                state["current_tool"] = None
            elif state["current_thinking"] is not None:
                yield StreamEvent(
                    thinking_block=ThinkingBlock(
                        text=state["current_thinking"]["text"],
                        signature=state["current_thinking"]["signature"],
                    )
                )
                state["current_thinking"] = None

    def _system_to_api(
        self, blocks: list[SystemBlock], *, ttl: str | None = None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sb in blocks:
            entry: dict[str, Any] = {"type": "text", "text": sb.text}
            if sb.cache:
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
            elif isinstance(b, ThinkingBlock):
                if b.encrypted:
                    out.append(
                        {
                            "type": "redacted_thinking",
                            "data": (b.provider_data or {}).get("data", ""),
                        }
                    )
                else:
                    entry: dict[str, Any] = {"type": "thinking", "thinking": b.text}
                    if b.signature:
                        entry["signature"] = b.signature
                    out.append(entry)
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
                tr_entry: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": inner,
                }
                if b.is_error:
                    tr_entry["is_error"] = True
                out.append(tr_entry)
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
            return {"type": "none"}
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
                sig = getattr(b, "signature", None)
                blocks.append(ThinkingBlock(text=t, signature=sig))
                thinking_parts.append(t)
            elif b_type == "redacted_thinking":
                data = getattr(b, "data", "") or ""
                blocks.append(ThinkingBlock(text="", encrypted=True, provider_data={"data": data}))
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
