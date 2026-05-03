from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import (
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
    UnsupportedFeature,
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
from llmfacade.providers._launch import (
    _LaunchEntry,
    default_provider_launch_defaults,
    derive_model_id,
)
from llmfacade.providers._swap_lifecycle import _LlamaSwapSupervisor
from llmfacade.settings import OutputFormat

if TYPE_CHECKING:
    from llmfacade.facade import LLM
    from llmfacade.model import Model


class LlamaCppServerProvider(Provider):
    """Talks to a llama.cpp ``llama-server`` over its OpenAI-compatible
    HTTP endpoint at ``<base_url>/v1/chat/completions`` plus a small set of
    server-native introspection paths (``/health``, ``/slots``,
    ``/slots/{id}?action=...``, ``/tokenize``).

    **Two modes**, mutually exclusive:

    * **External** (`base_url=...`): talk to a `llama-server` (or `llama-swap`)
      the user is already running. No process management. `LAUNCH_KNOBS`
      passed in this mode raise ``UnsupportedFeature``.
    * **Managed** (`base_url=None`): the provider owns a `llama-swap`
      subprocess and the YAML it consumes. `new_model(gguf=..., ...)`
      registers an entry; the first `convo.send()` lazily spawns the
      supervisor. Use `provider.shutdown()` for explicit teardown
      (atexit + signal handlers also call it).
    """

    NAME = "llamacpp"
    API_KEY_ENV = None
    SUPPORTS: frozenset[str] = frozenset(
        {
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "output_format",
            "tools",
            "tool_choice",
        }
    )

    def __init__(
        self,
        *,
        manager: LLM | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        # Managed-mode-only knob
        llmfacade_dir: str | Path | None = None,
        # LAUNCH_KNOBS as provider-level defaults
        gguf: str | None = None,
        context_size: int | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        n_gpu_layers: int | None = None,
        parallel: int | None = None,
        slot_save_path: str | None = None,
        ttl: int | None = None,
        extra_args: list[str] | tuple[str, ...] | None = None,
        # RUNTIME_KNOBS passthrough
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
    ) -> None:
        # Mode is decided here and never changes.
        self._managed = base_url is None

        # Session dir is needed in managed mode for slot_save_path default,
        # YAML / pidfile / log location. Resolve eagerly so a later cwd change
        # doesn't move it.
        sess_dir = (Path(llmfacade_dir) if llmfacade_dir else Path.cwd() / ".llmfacade").resolve()
        self._llmfacade_dir = sess_dir

        # Merge provider-level launch defaults: hardcoded < explicit kwargs.
        # Only non-None explicit values override the defaults.
        baseline = default_provider_launch_defaults(sess_dir)
        explicit_launch: dict[str, Any] = {
            "gguf": gguf,
            "context_size": context_size,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "n_gpu_layers": n_gpu_layers,
            "parallel": parallel,
            "slot_save_path": slot_save_path,
            "ttl": ttl,
            "extra_args": tuple(extra_args) if extra_args is not None else None,
        }
        if not self._managed:
            # External mode: launch knobs are nonsensical (the server is
            # already running). Reject early with a clear error.
            offending = sorted(k for k, v in explicit_launch.items() if v is not None)
            if offending:
                raise UnsupportedFeature(
                    f"launch knobs {offending!r} require managed mode (omit base_url= to enable)",
                    self.NAME,
                    None,
                )
            self._launch_defaults: dict[str, Any] = {}
            self._supervisor: _LlamaSwapSupervisor | None = None
        else:
            merged = dict(baseline)
            for k, v in explicit_launch.items():
                if v is not None:
                    merged[k] = v
            self._launch_defaults = merged
            # global_ttl is the fallback for entries that don't set ttl. The
            # baseline already supplies ttl=0, so this is mostly belt-and-braces.
            self._supervisor = _LlamaSwapSupervisor(
                llmfacade_dir=sess_dir, global_ttl=int(merged.get("ttl") or 0)
            )

        super().__init__(
            manager=manager,
            api_key=api_key,
            base_url=base_url,
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

    # ---- client construction ---------------------------------------------

    def _init_client(self) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ProviderNotInstalledError(
                "OpenAI SDK not installed (required for the llamacpp provider). "
                "Run: pip install llmfacade[llamacpp]"
            ) from e

        try:
            import httpx as _httpx
        except ImportError as e:
            raise ProviderNotInstalledError(
                "httpx not installed (required for the llamacpp provider). "
                "Run: pip install llmfacade[llamacpp]"
            ) from e

        self._module = _openai
        self._httpx = _httpx
        self._client: Any = None
        self._aclient: Any = None
        self._http: Any = None
        self._ahttp: Any = None
        self._http_base: str | None = None
        # Serialises the build-clients step in `_ensure_supervised` so two
        # concurrent first-calls don't race and stomp each other's clients.
        import threading as _threading

        self._client_lock = _threading.Lock()

        if not self._managed:
            self._build_clients(self._base_url or "http://localhost:8080/v1")

    def _build_clients(self, openai_base: str) -> None:
        """Build (or rebuild) the openai/httpx clients pointed at ``openai_base``.
        ``openai_base`` is the OpenAI-compat root including any ``/v1`` suffix.
        Also derives ``self._http_base`` (server root for /health, /slots, etc.)
        by stripping a trailing ``/v1``."""
        client_kwargs: dict[str, Any] = {"api_key": "sk-noop", "base_url": openai_base}
        self._client = self._module.OpenAI(**client_kwargs)
        self._aclient = self._module.AsyncOpenAI(**client_kwargs)

        root = openai_base.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._http_base = root
        self._http = self._httpx.Client(base_url=root, timeout=30.0)
        self._ahttp = self._httpx.AsyncClient(base_url=root, timeout=30.0)

    def _ensure_supervised(self) -> None:
        """Managed mode: lazily spawn llama-swap and (re)build the openai/httpx
        clients pointed at it. No-op in external mode or after the first call."""
        if not self._managed or self._supervisor is None:
            return
        with self._client_lock:
            self._ensure_supervised_locked()

    def _ensure_supervised_locked(self) -> None:
        # Caller holds self._client_lock so the build-clients step doesn't race
        # against a concurrent first call.
        assert self._supervisor is not None
        if self._client is not None and self._supervisor.is_started:
            return
        self._supervisor.ensure_started()
        base = self._supervisor.base_url
        if base is None:
            raise ProviderError("llama-swap supervisor reported started but has no base URL")
        self._build_clients(base.rstrip("/") + "/v1")

    # ---- model factory ----------------------------------------------------

    def new_model(
        self,
        model_id: str | None = None,
        *,
        # Managed-mode-only kwargs
        name: str | None = None,
        gguf: str | None = None,
        context_size: int | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        n_gpu_layers: int | None = None,
        parallel: int | None = None,
        slot_save_path: str | None = None,
        ttl: int | None = None,
        extra_args: list[str] | tuple[str, ...] | None = None,
        # Existing args (subset of base Provider.new_model)
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
        from llmfacade.model import Model

        # Build the per-model launch overrides dict from explicit kwargs only.
        explicit_launch: dict[str, Any] = {
            "gguf": gguf,
            "context_size": context_size,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "n_gpu_layers": n_gpu_layers,
            "parallel": parallel,
            "slot_save_path": slot_save_path,
            "ttl": ttl,
            "extra_args": tuple(extra_args) if extra_args is not None else None,
        }
        nonnull_launch_keys = sorted(k for k, v in explicit_launch.items() if v is not None)

        if not self._managed:
            if nonnull_launch_keys:
                raise UnsupportedFeature(
                    f"launch knobs {nonnull_launch_keys!r} require managed mode "
                    "(omit base_url= on the provider to enable)",
                    self.NAME,
                    model_id,
                )
            if name is not None:
                raise UnsupportedFeature(
                    "name= is a managed-mode kwarg (omit base_url= on the provider "
                    "to enable). In external mode pass the model id positionally.",
                    self.NAME,
                    model_id,
                )
            if model_id is None:
                raise ValueError(
                    "external-mode new_model() requires a positional model_id "
                    "(the model name your llama-server is configured to expose)."
                )
            return Model(
                provider=self,
                model_id=model_id,
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

        # Managed mode: cascade provider-level launch defaults < model overrides.
        merged = dict(self._launch_defaults)
        for k, v in explicit_launch.items():
            if v is not None:
                merged[k] = v
        # In managed mode the positional arg is the model name (the YAML key
        # llama-swap routes off). Reject conflict with explicit name= rather
        # than silently picking one.
        if model_id is not None:
            if name is not None and name != model_id:
                raise ValueError(
                    f"new_model() got conflicting names: positional={model_id!r} "
                    f"vs name={name!r}. Pass one or the other."
                )
            name = model_id
        if not merged.get("gguf"):
            raise ValueError(
                "managed-mode new_model() requires gguf= (set at provider or model scope)"
            )

        gguf_path = Path(merged["gguf"])
        if not gguf_path.exists():
            raise FileNotFoundError(f"gguf not found: {gguf_path}")

        derived = derive_model_id(merged, name)
        entry = _LaunchEntry(
            model_id=derived,
            gguf=str(gguf_path),
            context_size=merged.get("context_size"),
            cache_type_k=merged.get("cache_type_k"),
            cache_type_v=merged.get("cache_type_v"),
            n_gpu_layers=merged.get("n_gpu_layers"),
            parallel=merged.get("parallel"),
            slot_save_path=merged.get("slot_save_path"),
            ttl=merged.get("ttl"),
            extra_args=tuple(merged.get("extra_args") or ()),
        )
        assert self._supervisor is not None
        self._supervisor.register(entry)

        return Model(
            provider=self,
            model_id=derived,
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

    # ---- chat completions ------------------------------------------------

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

        # llama.cpp accepts these on /v1/chat/completions but the OpenAI SDK
        # doesn't expose them as typed kwargs; route through extra_body so the
        # SDK forwards them verbatim onto the wire.
        extra: dict[str, Any] = {}
        for key in ("top_k", "min_p", "repeat_penalty"):
            value = req.settings.get(key)
            if value is not None:
                extra[key] = value
        if extra:
            api_kwargs["extra_body"] = extra

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
        self._ensure_supervised()
        api_kwargs = self._build_kwargs(req)
        try:
            raw = self._client.chat.completions.create(**api_kwargs)
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw)

    async def _acomplete_raw(self, req: CompletionRequest) -> Response:
        self._ensure_supervised()
        api_kwargs = self._build_kwargs(req)
        try:
            raw = await self._aclient.chat.completions.create(**api_kwargs)
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        return self._parse_response(raw)

    def _stream_raw(self, req: CompletionRequest) -> Iterator[StreamEvent]:
        self._ensure_supervised()
        api_kwargs = self._build_kwargs(req)
        api_kwargs["stream"] = True
        api_kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = self._client.chat.completions.create(**api_kwargs)
            tool_buf: dict[int, dict[str, Any]] = {}
            state: dict[str, Any] = {"finish_reason": None}
            for chunk in stream:
                yield from self._chunk_to_events(chunk, tool_buf, state)
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    async def _astream_raw(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self._ensure_supervised()
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
                ),
                finish_reason=state.get("finish_reason"),
            )

    # ---- message / tool / response shaping --------------------------------

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
        for b in m.content:
            if isinstance(b, TextBlock):
                parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageBlock):
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
                # llama-server has no canonical thinking-block format; local
                # thinking models emit reasoning inline. Drop on the way out.
                continue
        out: dict[str, Any] = {"role": m.role}
        if parts:
            if m.role == "user":
                out["content"] = parts
            else:
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

    # ---- introspection ----------------------------------------------------

    def health(self) -> dict[str, Any]:
        """``GET /health``. Returns the parsed JSON body. Raises
        ``ProviderError`` on a 4xx or 5xx response."""
        self._ensure_supervised()
        return self._http_get("/health")

    async def ahealth(self) -> dict[str, Any]:
        self._ensure_supervised()
        return await self._ahttp_get("/health")

    def slots(self) -> list[dict[str, Any]]:
        """``GET /slots``. Per-slot processing state, sampling params, token
        counts, and generation speed."""
        self._ensure_supervised()
        data = self._http_get("/slots")
        return data if isinstance(data, list) else []

    async def aslots(self) -> list[dict[str, Any]]:
        self._ensure_supervised()
        data = await self._ahttp_get("/slots")
        return data if isinstance(data, list) else []

    def save_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        """``POST /slots/{id_slot}?action=save`` body ``{"filename": filename}``.
        ``filename`` is interpreted relative to the server's
        ``--slot-save-path`` directory."""
        self._ensure_supervised()
        return self._http_post(
            f"/slots/{id_slot}", params={"action": "save"}, json={"filename": filename}
        )

    async def asave_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        self._ensure_supervised()
        return await self._ahttp_post(
            f"/slots/{id_slot}", params={"action": "save"}, json={"filename": filename}
        )

    def restore_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        self._ensure_supervised()
        return self._http_post(
            f"/slots/{id_slot}", params={"action": "restore"}, json={"filename": filename}
        )

    async def arestore_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        self._ensure_supervised()
        return await self._ahttp_post(
            f"/slots/{id_slot}", params={"action": "restore"}, json={"filename": filename}
        )

    def erase_slot(self, id_slot: int) -> dict[str, Any]:
        self._ensure_supervised()
        return self._http_post(f"/slots/{id_slot}", params={"action": "erase"})

    async def aerase_slot(self, id_slot: int) -> dict[str, Any]:
        self._ensure_supervised()
        return await self._ahttp_post(f"/slots/{id_slot}", params={"action": "erase"})

    # ---- llama-swap-native introspection ---------------------------------

    def running(self) -> list[dict[str, Any]]:
        """``GET /running`` on the llama-swap supervisor — list of currently
        loaded models. In external mode this hits the user-supplied URL; if
        what's there is bare llama-server the request 404s and we raise
        ``UnsupportedFeature``."""
        self._ensure_supervised()
        try:
            data = self._http_get("/running")
        except ProviderError as e:
            if "404" in str(e):
                raise UnsupportedFeature(
                    "llama-swap not detected at base_url (no /running endpoint)",
                    self.NAME,
                    None,
                ) from e
            raise
        return data if isinstance(data, list) else []

    async def arunning(self) -> list[dict[str, Any]]:
        self._ensure_supervised()
        try:
            data = await self._ahttp_get("/running")
        except ProviderError as e:
            if "404" in str(e):
                raise UnsupportedFeature(
                    "llama-swap not detected at base_url (no /running endpoint)",
                    self.NAME,
                    None,
                ) from e
            raise
        return data if isinstance(data, list) else []

    def unload(self, model_id: str) -> None:
        """``POST /api/models/unload/{model_id}`` — ask llama-swap to evict
        ``model_id``. Raises ``UnsupportedFeature`` against bare llama-server."""
        self._ensure_supervised()
        try:
            self._http_post(f"/api/models/unload/{model_id}")
        except ProviderError as e:
            if "404" in str(e):
                raise UnsupportedFeature(
                    "llama-swap not detected at base_url (no /api/models/unload endpoint)",
                    self.NAME,
                    model_id,
                ) from e
            raise

    async def aunload(self, model_id: str) -> None:
        self._ensure_supervised()
        try:
            await self._ahttp_post(f"/api/models/unload/{model_id}")
        except ProviderError as e:
            if "404" in str(e):
                raise UnsupportedFeature(
                    "llama-swap not detected at base_url (no /api/models/unload endpoint)",
                    self.NAME,
                    model_id,
                ) from e
            raise

    def unload_all(self) -> None:
        """``POST /api/models/unload`` — ask llama-swap to evict every loaded
        model. Raises ``UnsupportedFeature`` against bare llama-server."""
        self._ensure_supervised()
        try:
            self._http_post("/api/models/unload")
        except ProviderError as e:
            if "404" in str(e):
                raise UnsupportedFeature(
                    "llama-swap not detected at base_url (no /api/models/unload endpoint)",
                    self.NAME,
                    None,
                ) from e
            raise

    async def aunload_all(self) -> None:
        self._ensure_supervised()
        try:
            await self._ahttp_post("/api/models/unload")
        except ProviderError as e:
            if "404" in str(e):
                raise UnsupportedFeature(
                    "llama-swap not detected at base_url (no /api/models/unload endpoint)",
                    self.NAME,
                    None,
                ) from e
            raise

    # ---- explicit lifecycle ----------------------------------------------

    def shutdown(self) -> None:
        """Tear down any managed-mode subprocess. Idempotent. atexit and signal
        handlers also call this; explicit invocation is for tests and for
        callers that don't want to wait for process exit."""
        if self._supervisor is not None:
            self._supervisor.shutdown()

    # ---- token counting ---------------------------------------------------

    def count_tokens(
        self,
        text: str,
        *,
        system: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Count tokens by calling the running llama-server's ``/tokenize``
        endpoint. Falls back to ``chars/4`` if the server is unreachable so
        logging never blocks on a transient network error.

        Note: in managed mode this routes through llama-swap, which may forward
        ``/tokenize`` to the active backend or 404 — either way the fall-back
        keeps logging working."""
        del model_id
        combined = text + (system or "")
        if self._http is None:
            return super().count_tokens(text, system=system)
        try:
            resp = self._http.post("/tokenize", json={"content": combined})
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return super().count_tokens(text, system=system)
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, list):
            return len(tokens)
        return super().count_tokens(text, system=system)

    def tokenizer_name(self, *, model_id: str | None = None) -> str:
        del model_id
        return "llama-server /tokenize"

    # ---- httpx helpers ----------------------------------------------------

    def _http_get(self, path: str) -> Any:
        if self._http is None:
            raise ProviderError(
                "HTTP client not initialised (managed-mode supervisor not started yet)"
            )
        try:
            resp = self._http.get(path)
        except Exception as e:
            raise ProviderError(f"GET {path} failed: {e}", original=e) from e
        return self._parse_http(resp, f"GET {path}")

    def _http_post(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if self._http is None:
            raise ProviderError(
                "HTTP client not initialised (managed-mode supervisor not started yet)"
            )
        try:
            resp = self._http.post(path, params=params, json=json)
        except Exception as e:
            raise ProviderError(f"POST {path} failed: {e}", original=e) from e
        return self._parse_http(resp, f"POST {path}")

    async def _ahttp_get(self, path: str) -> Any:
        if self._ahttp is None:
            raise ProviderError(
                "HTTP client not initialised (managed-mode supervisor not started yet)"
            )
        try:
            resp = await self._ahttp.get(path)
        except Exception as e:
            raise ProviderError(f"GET {path} failed: {e}", original=e) from e
        return self._parse_http(resp, f"GET {path}")

    async def _ahttp_post(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if self._ahttp is None:
            raise ProviderError(
                "HTTP client not initialised (managed-mode supervisor not started yet)"
            )
        try:
            resp = await self._ahttp.post(path, params=params, json=json)
        except Exception as e:
            raise ProviderError(f"POST {path} failed: {e}", original=e) from e
        return self._parse_http(resp, f"POST {path}")

    def _parse_http(self, resp: Any, label: str) -> Any:
        import contextlib

        status = getattr(resp, "status_code", None)
        if status is None or status >= 400:
            body = ""
            with contextlib.suppress(Exception):
                # httpx streaming responses need .read() before .text is safe;
                # try it first, fall through silently on error.
                if hasattr(resp, "read"):
                    resp.read()
                body = resp.text
            raise ProviderError(f"{label} returned status {status}: {body}")
        try:
            return resp.json()
        except Exception as e:
            raise ProviderError(f"{label} returned non-JSON body", original=e) from e
