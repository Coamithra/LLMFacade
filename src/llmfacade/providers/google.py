from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
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
    ImageResult,
    ImageUsage,
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    apply_save_dir,
)
from llmfacade.provider import CompletionRequest, Provider
from llmfacade.settings import OutputFormat


class GoogleProvider(Provider):
    NAME = "google"
    API_KEY_ENV = "GOOGLE_API_KEY"
    SUPPORTS: frozenset[str] = frozenset(
        {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "output_format",
            "tools",
            "tool_choice",
            "vision",
            "image_generation",
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

    _local_tokenizer_cache: dict[str, Any] = {}

    def _get_local_tokenizer(self, model_id: str | None) -> Any | None:
        # google.genai.LocalTokenizer wraps a sentencepiece model. The
        # gemma3 model file is fetched from GitHub on first use and cached
        # to a temp dir; subsequent calls are pure-local. Currently covers
        # Gemini 2.0/2.5 plus the 3.0 preview — all on the gemma3 tokenizer.
        # 1.x is not supported by google-genai's loader.
        target = model_id or "gemini-2.5-flash"
        cached = self._local_tokenizer_cache.get(target)
        if cached is not None:
            return cached if cached is not False else None
        try:
            from google import genai as _genai
        except ImportError:
            self._local_tokenizer_cache[target] = False
            return None
        local_tokenizer_cls = getattr(_genai, "LocalTokenizer", None)
        if local_tokenizer_cls is None:
            self._local_tokenizer_cache[target] = False
            return None
        try:
            tok = local_tokenizer_cls(model_name=target)
        except Exception:
            self._local_tokenizer_cache[target] = False
            return None
        self._local_tokenizer_cache[target] = tok
        return tok

    def count_tokens(
        self,
        text: str,
        *,
        system: str | None = None,
        model_id: str | None = None,
    ) -> int:
        tok = self._get_local_tokenizer(model_id)
        if tok is None:
            return super().count_tokens(text, system=system, model_id=model_id)
        try:
            n = int(tok.count_tokens(text).total_tokens)
            if system:
                n += int(tok.count_tokens(system).total_tokens)
            return n
        except Exception:
            return super().count_tokens(text, system=system, model_id=model_id)

    def tokenizer_name(self, *, model_id: str | None = None) -> str:
        if self._get_local_tokenizer(model_id) is None:
            return "chars/4 (google-genai[local-tokenizer] not installed)"
        return "sentencepiece (gemma3)"

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
            "max_output_tokens": req.settings.get("max_tokens", 1024),
        }
        temperature = req.settings.get("temperature")
        if temperature is not None:
            config["temperature"] = temperature
        if req.stop:
            config["stop_sequences"] = req.stop
        for key in ("top_p", "top_k"):
            value = req.settings.get(key)
            if value is not None:
                config[key] = value

        if req.system_blocks:
            config["system_instruction"] = "\n\n".join(sb.text for sb in req.system_blocks)

        if req.tools:
            config["tools"] = [
                {"function_declarations": [self._tool_to_api(t) for t in req.tools]}
            ]
            tc_cfg = self._tool_choice_to_api(req.settings.get("tool_choice", "auto"))
            if tc_cfg is not None:
                config["tool_config"] = tc_cfg

        out_format = req.settings.get("output_format")
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
            state: dict[str, Any] = {"thought_text": [], "thought_sig": None, "in_thought": False}
            last_usage: Usage | None = None
            for chunk in stream:
                events, usage = self._chunk_to_events(chunk, state)
                yield from events
                if usage is not None:
                    last_usage = usage
            flushed = self._flush_thought(state)
            if flushed is not None:
                yield flushed
            yield StreamEvent(done=True, usage=last_usage)
        except Exception as e:
            self._reraise(e)
            raise

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        api_kwargs = self._build_kwargs(req)
        try:
            stream = await self._client.aio.models.generate_content_stream(**api_kwargs)
            state: dict[str, Any] = {"thought_text": [], "thought_sig": None, "in_thought": False}
            last_usage: Usage | None = None
            async for chunk in stream:
                events, usage = self._chunk_to_events(chunk, state)
                for ev in events:
                    yield ev
                if usage is not None:
                    last_usage = usage
            flushed = self._flush_thought(state)
            if flushed is not None:
                yield flushed
            yield StreamEvent(done=True, usage=last_usage)
        except Exception as e:
            self._reraise(e)
            raise

    def _flush_thought(self, state: dict[str, Any]) -> StreamEvent | None:
        if not state["in_thought"]:
            return None
        ev = StreamEvent(
            thinking_block=ThinkingBlock(
                text="".join(state["thought_text"]),
                signature=state["thought_sig"],
            )
        )
        state["thought_text"] = []
        state["thought_sig"] = None
        state["in_thought"] = False
        return ev

    def _chunk_to_events(
        self, chunk: Any, state: dict[str, Any]
    ) -> tuple[list[StreamEvent], Usage | None]:
        events: list[StreamEvent] = []
        for cand in getattr(chunk, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "thought", False):
                    state["in_thought"] = True
                    t = getattr(part, "text", "") or ""
                    sig = getattr(part, "thought_signature", None)
                    if t:
                        state["thought_text"].append(t)
                        events.append(StreamEvent(thinking_delta=t))
                    if sig:
                        state["thought_sig"] = sig
                    continue
                # Non-thought part: flush any pending thought first.
                flushed = self._flush_thought(state)
                if flushed is not None:
                    events.append(flushed)
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
                    continue
                t = getattr(part, "text", None)
                if t:
                    events.append(StreamEvent(text_delta=t))
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
            elif isinstance(b, ThinkingBlock):
                # Gemini has no equivalent of Anthropic's redacted_thinking;
                # encrypted blocks from another provider can't be reconstructed,
                # so drop them. Plain thoughts round-trip with their signature.
                if b.encrypted:
                    continue
                tp: dict[str, Any] = {"text": b.text, "thought": True}
                if b.signature:
                    tp["thought_signature"] = b.signature
                parts.append(tp)
        return [{"role": role, "parts": parts}]

    def _tool_to_api(self, t: Any) -> dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema,
        }

    def _tool_choice_to_api(self, tc: str) -> dict[str, Any] | None:
        # Gemini's tool_config.function_calling_config has three modes:
        # AUTO (model decides), ANY (must call a tool — optionally restricted to
        # an allow-list), and NONE (no tool calls). AUTO is the SDK default and
        # we omit tool_config entirely in that case to keep the request lean.
        if tc == "auto":
            return None
        if tc == "required":
            return {"function_calling_config": {"mode": "ANY"}}
        if tc == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        return {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tc],
            }
        }

    def _parse_response(self, raw: Any, model: str) -> Response:
        blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for cand in getattr(raw, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "thought", False):
                    t = getattr(part, "text", "") or ""
                    sig = getattr(part, "thought_signature", None)
                    blocks.append(ThinkingBlock(text=t, signature=sig))
                    thinking_parts.append(t)
                    continue
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    name = getattr(fn_call, "name", "")
                    args = dict(getattr(fn_call, "args", {}) or {})
                    use_id = f"call-{uuid.uuid4().hex}"
                    blocks.append(ToolUseBlock(id=use_id, name=name, input=args))
                    tool_calls.append(ToolCall(id=use_id, name=name, input=args))
                    continue
                t = getattr(part, "text", None)
                if t:
                    blocks.append(TextBlock(t))
                    text_parts.append(t)

        return Response(
            text="".join(text_parts),
            blocks=blocks,
            tool_calls=tool_calls,
            thinking="".join(thinking_parts) or None,
            usage=self._usage_from(raw),
            finish_reason=None,
            model=model,
            raw=raw,
        )

    # ---- Image generation (Gemini-native, "Nano Banana") -------------------
    # Imagen is being shut down per Google's Gemini API deprecations, so the
    # only image path is gemini-2.5-flash-image via generate_content with
    # response_modalities=["IMAGE"]. Reference images are passed as inline_data
    # parts in `contents` (the same wire shape as vision input), which is what
    # makes image-conditioned generation / editing work on the Developer API.

    _DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"

    def _image_contents(
        self, prompt: str, reference_images: list[ImageBlock] | None
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for b in reference_images or []:
            parts.append({"inline_data": {"mime_type": b.media_type, "data": b.to_base64()}})
        return [{"role": "user", "parts": parts}]

    def _image_config(
        self, aspect_ratio: str | None, extra: dict[str, Any] | None
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"response_modalities": ["IMAGE"]}
        if aspect_ratio is not None:
            config["image_config"] = {"aspect_ratio": aspect_ratio}
        if extra:
            config.update(extra)
        return config

    def _parse_image_response(self, raw: Any, model: str) -> ImageResult:
        images: list[ImageBlock] = []
        for cand in getattr(raw, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue
                data = getattr(inline, "data", None)
                if not data:
                    continue
                mime = getattr(inline, "mime_type", None) or "image/png"
                block = (
                    ImageBlock(data=data, media_type=mime)
                    if isinstance(data, bytes)
                    else ImageBlock.from_base64(data, media_type=mime)
                )
                images.append(block)
        return ImageResult(
            images=images,
            usage=self._image_usage_from(raw, len(images)),
            model=model,
            provider=self.NAME,
            raw=raw,
        )

    def _image_usage_from(self, raw: Any, image_count: int) -> ImageUsage:
        um = getattr(raw, "usage_metadata", None)
        if um is None:
            return ImageUsage(image_count=image_count)
        prompt = getattr(um, "prompt_token_count", 0) or 0
        candidates = getattr(um, "candidates_token_count", 0) or 0
        total = getattr(um, "total_token_count", 0) or 0
        return ImageUsage(
            input_tokens=prompt,
            output_tokens=candidates,
            total_tokens=total or (prompt + candidates),
            image_count=image_count,
        )

    def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        reference_images: list[ImageBlock] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        model = model or self._DEFAULT_IMAGE_MODEL
        api_kwargs: dict[str, Any] = {
            "model": model,
            "contents": self._image_contents(prompt, reference_images),
            "config": self._image_config(aspect_ratio, extra),
        }
        try:
            raw = self._client.models.generate_content(**api_kwargs)
        except Exception as e:
            self._reraise(e)
            raise
        return apply_save_dir(self._parse_image_response(raw, model), save_dir)

    async def agenerate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        reference_images: list[ImageBlock] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        model = model or self._DEFAULT_IMAGE_MODEL
        api_kwargs: dict[str, Any] = {
            "model": model,
            "contents": self._image_contents(prompt, reference_images),
            "config": self._image_config(aspect_ratio, extra),
        }
        try:
            raw = await self._client.aio.models.generate_content(**api_kwargs)
        except Exception as e:
            self._reraise(e)
            raise
        return apply_save_dir(self._parse_image_response(raw, model), save_dir)

    def _usage_from(self, raw: Any) -> Usage | None:
        um = getattr(raw, "usage_metadata", None)
        if um is None:
            return None
        prompt = getattr(um, "prompt_token_count", 0) or 0
        completion = getattr(um, "candidates_token_count", 0) or 0
        cached = getattr(um, "cached_content_token_count", 0) or 0
        # Gemini reports thinking tokens separately; candidates_token_count is
        # the visible output and excludes them, so add them into the total.
        thoughts = getattr(um, "thoughts_token_count", 0) or 0
        total = getattr(um, "total_token_count", None)
        if not total:
            total = prompt + completion + thoughts
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cache_read_tokens=cached,
            reasoning_tokens=thoughts,
        )

    def _reraise(self, e: Exception) -> None:
        err_name = type(e).__name__.lower()
        if "authentication" in err_name or "permission" in err_name:
            raise AuthenticationError(str(e)) from e
        if "resource_exhausted" in err_name or "rate" in err_name:
            raise RateLimitError(str(e)) from e
        raise ProviderError(str(e), original=e) from e
