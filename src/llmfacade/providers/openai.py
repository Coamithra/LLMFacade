from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from llmfacade.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
)
from llmfacade.models import (
    ImageResult,
    Message,
    ReferenceImage,
    Response,
    StreamEvent,
    _apply_save_dir,
)
from llmfacade.provider import CompletionRequest, Provider
from llmfacade.providers import _openai_chat as _chat
from llmfacade.providers._openai_images import (
    build_edit_kwargs,
    build_generate_kwargs,
    media_type_for,
    parse_images_response,
)
from llmfacade.settings import EffortLevel, OutputFormat

# Back-compat aliases for the helpers that moved to the shared
# `_openai_chat` module (kept importable under their historical names).
_openai_cached_tokens = _chat.cached_prompt_tokens
_empty_choices_detail = _chat.empty_choices_detail
_openai_reasoning_tokens = _chat.reasoning_tokens_from_details


class OpenAIProvider(Provider):
    NAME = "openai"
    API_KEY_ENV = "OPENAI_API_KEY"
    SUPPORTS: frozenset[str] = frozenset(
        {
            "max_tokens",
            "temperature",
            "top_p",
            "effort",
            "output_format",
            "tools",
            "tool_choice",
            "vision",
            "image_generation",
        }
    )
    # OpenAI's ``usage.prompt_tokens`` already includes
    # ``prompt_tokens_details.cached_tokens`` (the cached subset we surface as
    # ``cache_read_tokens``), so total input must not re-add it.
    PROMPT_TOKENS_INCLUDE_CACHED = True

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
        api_kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": _chat.chat_messages(req, self._message_to_api),
            # The GPT-5 series and o-series reasoning models reject the legacy
            # `max_tokens` (400) and require `max_completion_tokens`; the
            # gpt-4o-class models accept it too, so always emit it. The facade
            # knob stays named `max_tokens`.
            "max_completion_tokens": req.settings.get("max_tokens", 1024),
        }
        temperature = req.settings.get("temperature")
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if req.stop:
            api_kwargs["stop"] = req.stop
        top_p = req.settings.get("top_p")
        if top_p is not None:
            api_kwargs["top_p"] = top_p

        # `effort` maps to OpenAI's `reasoning_effort` (reasoning models only;
        # OpenAI accepts none/minimal/low/medium/high/xhigh — note it has no
        # "max", unlike Anthropic). Passed verbatim; an unsupported value or a
        # non-reasoning model is the caller's responsibility (the API 400s).
        effort = req.settings.get("effort")
        if effort is not None:
            api_kwargs["reasoning_effort"] = (
                effort.value if isinstance(effort, EffortLevel) else effort
            )

        _chat.apply_tools(api_kwargs, req)

        self._apply_output_format(api_kwargs, req.settings.get("output_format"))
        return api_kwargs

    @staticmethod
    def _apply_output_format(api_kwargs: dict[str, Any], out_format: Any) -> None:
        """Translate the `output_format` knob to OpenAI's `response_format`.

        A `dict` is a JSON Schema for strict Structured Outputs — emitted as
        `{"type": "json_schema", ...}`. Accepts either a full
        `{name, schema, strict}` config or a bare schema (wrapped with
        `name="response"`, `strict=True`). `OutputFormat.JSON` / `"json"` emits
        the looser `{"type": "json_object"}` mode; `"text"` / `None` omits it."""
        if out_format is None:
            return
        if isinstance(out_format, dict):
            # Disambiguate by a top-level "schema" key: a {name, schema, strict}
            # config has one; a bare JSON Schema does not (the JSON Schema root
            # vocabulary has no "schema" keyword).
            if "schema" in out_format:
                cfg = {
                    "name": out_format.get("name", "response"),
                    "schema": out_format["schema"],
                    "strict": out_format.get("strict", True),
                }
            else:
                cfg = {"name": "response", "schema": out_format, "strict": True}
            api_kwargs["response_format"] = {"type": "json_schema", "json_schema": cfg}
            return
        value = out_format.value if isinstance(out_format, OutputFormat) else out_format
        if value == "json":
            api_kwargs["response_format"] = {"type": "json_object"}

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
            yield from _chat.tool_fragment_events(delta, tool_buf)
            choice_finish = getattr(choice, "finish_reason", None)
            if choice_finish is not None:
                state["finish_reason"] = choice_finish
                yield from _chat.flush_tool_call_events(tool_buf)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield StreamEvent(
                done=True,
                usage=_chat.usage_from_chat(usage, include_cached=True),
                finish_reason=state.get("finish_reason"),
            )

    def _message_to_api(self, m: Message) -> list[dict[str, Any]]:
        return _chat.message_to_api(m, provider_label="OpenAI")

    def _parse_response(self, raw: Any) -> Response:
        return _chat.parse_chat_response(raw, server_label="OpenAI", include_cached_tokens=True)

    # ---- Image generation --------------------------------------------------

    def _image_kwargs(
        self,
        prompt: str,
        model: str,
        n: int,
        size: str | None,
        quality: str | None,
        background: str | None,
        output_format: str | None,
        reference_images: Sequence[ReferenceImage] | None,
        extra: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``("edit"|"generate", kwargs)``. Reference images route to the
        edits endpoint, which for ``gpt-image-*`` accepts the same
        ``quality`` / ``background`` / ``output_format`` params as generate.
        ``request_b64=False``: ``gpt-image-*`` always returns base64 and
        rejects ``response_format``."""
        if reference_images:
            return "edit", build_edit_kwargs(
                model=model,
                prompt=prompt,
                reference_images=reference_images,
                n=n,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                extra=extra,
                request_b64=False,
            )
        return "generate", build_generate_kwargs(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            extra=extra,
            request_b64=False,
        )

    def _generate_image_raw(
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
        reference_images: Sequence[ReferenceImage] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        model = model or "gpt-image-1"
        endpoint, kwargs = self._image_kwargs(
            prompt, model, n, size, quality, background, output_format, reference_images, extra
        )
        try:
            if endpoint == "edit":
                raw = self._client.images.edit(**kwargs)
            else:
                raw = self._client.images.generate(**kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        result = parse_images_response(
            raw, model=model, provider="openai", fallback_media_type=media_type_for(output_format)
        )
        return _apply_save_dir(result, save_dir)

    async def _agenerate_image_raw(
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
        reference_images: Sequence[ReferenceImage] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        model = model or "gpt-image-1"
        endpoint, kwargs = self._image_kwargs(
            prompt, model, n, size, quality, background, output_format, reference_images, extra
        )
        try:
            if endpoint == "edit":
                raw = await self._aclient.images.edit(**kwargs)
            else:
                raw = await self._aclient.images.generate(**kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        result = parse_images_response(
            raw, model=model, provider="openai", fallback_media_type=media_type_for(output_format)
        )
        return _apply_save_dir(result, save_dir)
