from __future__ import annotations

import json as _json
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote as _urlquote

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
from llmfacade.providers._gguf import classify_thinking_style, read_gguf_chat_template
from llmfacade.providers._launch import (
    _LaunchEntry,
    default_provider_launch_defaults,
    derive_model_id,
    parse_fit_print,
    validate_flash_attn,
)
from llmfacade.providers._swap_lifecycle import _LlamaSwapSupervisor
from llmfacade.settings import DrySampler, OutputFormat, ThinkingMode, ThinkingStyle

if TYPE_CHECKING:
    from llmfacade.facade import LLM
    from llmfacade.model import Model


def _llama_reasoning(obj: Any) -> str:
    """Read reasoning text off a llama-server message or streaming delta.

    llama.cpp surfaces extracted reasoning as ``reasoning_content`` (DeepSeek
    convention, the default when ``--reasoning-format`` parsing is on); some
    OpenAI-compat builds use ``reasoning``. The OpenAI SDK preserves these as
    extra attributes on the parsed model. Returns ``""`` when neither is set."""
    return getattr(obj, "reasoning_content", None) or getattr(obj, "reasoning", None) or ""


def _llama_reasoning_tokens(usage: Any) -> int:
    """Pull a reasoning-token count from llama-server usage if the build
    reports one in ``completion_tokens_details.reasoning_tokens``. Most builds
    fold reasoning into ``completion_tokens`` with no breakdown (→ 0); the
    conversation log then counts the reasoning text locally via ``/tokenize``."""
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "reasoning_tokens", 0) or 0


def _coerce_thinking_style(value: ThinkingStyle | str) -> ThinkingStyle:
    """Validate an explicit ``thinking_style=`` value into a ``ThinkingStyle``.
    Raises ``ValueError`` on an unrecognised string rather than silently
    defaulting, so a typo surfaces at ``new_model()`` instead of as a missing
    warning later."""
    if isinstance(value, ThinkingStyle):
        return value
    try:
        return ThinkingStyle(value)
    except ValueError as e:
        valid = [s.value for s in ThinkingStyle]
        raise ValueError(
            f"thinking_style must be a ThinkingStyle or one of {valid!r}, got {value!r}"
        ) from e


def _resolve_thinking_style(
    *, gguf_path: str | None, explicit: ThinkingStyle | str | None
) -> ThinkingStyle:
    """Resolve a model's thinking style. An explicit override always wins;
    otherwise auto-detect by reading the GGUF's embedded ``tokenizer.chat_template``
    (only possible when the file is local, i.e. managed mode). Best-effort: an
    absent ``gguf_path`` or an unreadable/template-less GGUF resolves to
    ``ThinkingStyle.UNKNOWN`` and never raises, so detection can't block
    ``new_model()``."""
    if explicit is not None:
        return _coerce_thinking_style(explicit)
    if gguf_path is None:
        return ThinkingStyle.UNKNOWN
    return classify_thinking_style(read_gguf_chat_template(gguf_path))


# Track (model_id, style) tuples we've warned about so a thinking-knob-vs-style
# mismatch warning fires once per model rather than on every send().
_WARNED_THINKING_STYLE: set[tuple[str, str]] = set()


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
            "dry",
            "thinking",
            "output_format",
            "tools",
            "tool_choice",
            "vision",
        }
    )
    # Note: "thinking_budget" is intentionally absent. The `thinking` knob maps
    # to llama.cpp's `enable_thinking` template kwarg (a ThinkingMode), which
    # has no budget form — so an int token budget fails fast through the
    # value-level gate in `Conversation._build_request` (the same mechanism that
    # rejects a budget on Opus 4.8), instead of being silently dropped here.
    # Wall-clock cap on the synchronous `llama-fit-params` probe in
    # `_maybe_estimate_fit`. Sub-second in the normal case; capped low so
    # `new_model()` never hangs on a missing or stuck binary.
    _FIT_PARAMS_TIMEOUT_SECONDS: float = 15.0

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
        n_cpu_moe: int | None = None,
        parallel: int | None = None,
        slot_save_path: str | None = None,
        ttl: int | None = None,
        extra_args: list[str] | tuple[str, ...] | None = None,
        fit: bool | None = None,
        fit_target: list[int] | tuple[int, ...] | None = None,
        fit_ctx: int | None = None,
        flash_attn: str | None = None,
        mmproj_path: str | None = None,
        jinja: bool | None = None,
        # RUNTIME_KNOBS passthrough
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
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

        # Best-effort estimates from `llama-fit-params`, populated lazily from
        # `new_model()`. Keyed by model_id; `None` means we tried and failed
        # (binary missing, non-zero exit, unparseable output) — surfacing the
        # absence is intentional and lets `log_metadata` skip the field.
        self._fit_estimates: dict[str, dict[str, Any] | None] = {}

        # Auto-detected (or explicitly set) thinking style per model_id, keyed
        # like `_fit_estimates`. Drives the thinking-knob-vs-style warning in
        # `_build_kwargs` and surfaces in `log_metadata`.
        self._thinking_styles: dict[str, ThinkingStyle] = {}

        # Merge provider-level launch defaults: hardcoded < explicit kwargs.
        # Only non-None explicit values override the defaults.
        baseline = default_provider_launch_defaults(sess_dir)
        explicit_launch: dict[str, Any] = {
            "gguf": gguf,
            "context_size": context_size,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "n_gpu_layers": n_gpu_layers,
            "n_cpu_moe": n_cpu_moe,
            "parallel": parallel,
            "slot_save_path": slot_save_path,
            "ttl": ttl,
            "extra_args": tuple(extra_args) if extra_args is not None else None,
            "fit": fit,
            "fit_target": tuple(fit_target) if fit_target is not None else None,
            "fit_ctx": fit_ctx,
            "flash_attn": flash_attn,
            "mmproj_path": mmproj_path,
            "jinja": jinja,
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
            validate_flash_attn(flash_attn)
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
            dry=dry,
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
        import asyncio as _asyncio
        import threading as _threading

        self._client_lock = _threading.Lock()
        # Per-provider slot locks. Used by callers that want to make a
        # restore + send pair (or warm_and_save's send + save pair) atomic
        # against concurrent slot mutations; see `slot_lock` / `aslot_lock`.
        # The two locks do not synchronise against each other — a process
        # mixing sync and async slot ops on the same provider is not safe.
        # `asyncio.Lock()` is built here without a running loop on purpose:
        # Python 3.10+ lazy-binds the loop on first acquire, so this is
        # safe as long as a single event loop owns the provider.
        self._slot_lock = _threading.Lock()
        self._slot_alock = _asyncio.Lock()

        if not self._managed:
            self._build_clients(self._base_url or "http://localhost:8080/v1")

    def _build_clients(self, openai_base: str) -> None:
        """Build (or rebuild) the openai/httpx clients pointed at ``openai_base``.
        ``openai_base`` is the OpenAI-compat root including any ``/v1`` suffix.
        Also derives ``self._http_base`` (server root for /health, /slots, etc.)
        by stripping a trailing ``/v1``."""
        client_kwargs: dict[str, Any] = {"api_key": "sk-noop", "base_url": openai_base}
        if self._managed:
            # We own the llama-swap process and may hard-kill it via interrupt().
            # The SDK default (max_retries=2) would retry twice against the
            # now-dead local port (backoff + connect timeout each) before
            # raising, so a worker blocked in send() takes seconds to unblock
            # after the kill -- defeating instant cancel. We deliberately trade
            # the SDK's transient-error resilience (e.g. a mid-swap 5xx during
            # TTL eviction) for that responsiveness on a process we control.
            # External mode talks to a real remote server, so it keeps the default.
            client_kwargs["max_retries"] = 0
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
        n_cpu_moe: int | None = None,
        parallel: int | None = None,
        slot_save_path: str | None = None,
        ttl: int | None = None,
        extra_args: list[str] | tuple[str, ...] | None = None,
        fit: bool | None = None,
        fit_target: list[int] | tuple[int, ...] | None = None,
        fit_ctx: int | None = None,
        flash_attn: str | None = None,
        mmproj_path: str | None = None,
        jinja: bool | None = None,
        # Existing args (subset of base Provider.new_model)
        capability_override: frozenset[str] | None = None,
        thinking_style: ThinkingStyle | str | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        dry: DrySampler | None = None,
        effort: Any | None = None,
        thinking: int | ThinkingMode | str | None = None,
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
            "n_cpu_moe": n_cpu_moe,
            "parallel": parallel,
            "slot_save_path": slot_save_path,
            "ttl": ttl,
            "extra_args": tuple(extra_args) if extra_args is not None else None,
            "fit": fit,
            "fit_target": tuple(fit_target) if fit_target is not None else None,
            "fit_ctx": fit_ctx,
            "flash_attn": flash_attn,
            "mmproj_path": mmproj_path,
            "jinja": jinja,
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
            # No local GGUF to inspect in external mode, so style is UNKNOWN
            # unless the caller states it explicitly. Only store when given so
            # `log_metadata` doesn't surface a noisy UNKNOWN for every model.
            if thinking_style is not None:
                self._thinking_styles[model_id] = _coerce_thinking_style(thinking_style)
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
                dry=dry,
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
        validate_flash_attn(flash_attn)
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

        mmproj_value = merged.get("mmproj_path")
        if mmproj_value is not None:
            mmproj_p = Path(mmproj_value)
            if not mmproj_p.exists():
                raise FileNotFoundError(f"mmproj_path not found: {mmproj_p}")
            mmproj_value = str(mmproj_p)

        derived = derive_model_id(merged, name)
        fit_target = merged.get("fit_target")
        entry = _LaunchEntry(
            model_id=derived,
            gguf=str(gguf_path),
            context_size=merged.get("context_size"),
            cache_type_k=merged.get("cache_type_k"),
            cache_type_v=merged.get("cache_type_v"),
            n_gpu_layers=merged.get("n_gpu_layers"),
            n_cpu_moe=merged.get("n_cpu_moe"),
            parallel=merged.get("parallel"),
            slot_save_path=merged.get("slot_save_path"),
            ttl=merged.get("ttl"),
            extra_args=tuple(merged.get("extra_args") or ()),
            fit=bool(merged.get("fit", True)),
            fit_target=tuple(fit_target) if fit_target is not None else None,
            fit_ctx=merged.get("fit_ctx"),
            flash_attn=merged.get("flash_attn"),
            mmproj_path=mmproj_value,
            jinja=bool(merged.get("jinja", True)),
        )
        assert self._supervisor is not None
        self._supervisor.register(entry)
        self._maybe_estimate_fit(entry)
        self._thinking_styles[derived] = _resolve_thinking_style(
            gguf_path=str(gguf_path), explicit=thinking_style
        )

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
            dry=dry,
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

        # DRY ("Don't Repeat Yourself") sampler: a DrySampler is unpacked into
        # llama.cpp's dry_* wire params, also via extra_body. n-gram-level loop
        # escape that token-level repeat_penalty can't break.
        dry = req.settings.get("dry")
        if dry is not None:
            extra.update(self._dry_to_extra_body(dry))

        # Thinking control: a ThinkingMode maps to llama.cpp's
        # `chat_template_kwargs={"enable_thinking": bool}`, routed through
        # extra_body like the samplers above. The embedded chat template only
        # honors this under --jinja (managed-mode default-on); on a model whose
        # template doesn't gate thinking that way, `_warn_thinking_style_mismatch`
        # flags it once. An int budget never reaches here (gated upstream).
        thinking_val = req.settings.get("thinking")
        template_kwargs = self._thinking_to_template_kwargs(thinking_val)
        if template_kwargs is not None:
            extra["chat_template_kwargs"] = template_kwargs
            self._warn_thinking_style_mismatch(req.model, thinking_val)

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

    @staticmethod
    def _dry_to_extra_body(value: Any) -> dict[str, Any]:
        """Map the ``dry`` knob value to llama.cpp's DRY wire params for
        ``extra_body``. A ``DrySampler`` is unpacked into ``dry_multiplier`` /
        ``dry_base`` / ``dry_allowed_length`` / ``dry_penalty_last_n`` /
        ``dry_sequence_breakers``, omitting any field left ``None`` so the server
        keeps its own default (only ``sequence_breakers`` is omittable; the
        numeric fields always carry a value). A plain mapping is accepted
        defensively — assumed already in the ``dry_*`` wire shape — and passed
        through for its non-``None`` entries. Anything else is a usage error."""
        if isinstance(value, DrySampler):
            fields: dict[str, Any] = {
                "dry_multiplier": value.multiplier,
                "dry_base": value.base,
                "dry_allowed_length": value.allowed_length,
                "dry_penalty_last_n": value.penalty_last_n,
                "dry_sequence_breakers": (
                    list(value.sequence_breakers) if value.sequence_breakers is not None else None
                ),
            }
            return {k: v for k, v in fields.items() if v is not None}
        if isinstance(value, Mapping):
            return {k: v for k, v in value.items() if v is not None}
        raise TypeError(
            "dry= expects a DrySampler (or a dry_* mapping) for the llamacpp "
            f"provider, got {type(value).__name__}."
        )

    @staticmethod
    def _thinking_to_template_kwargs(value: Any) -> dict[str, Any] | None:
        """Map the ``thinking`` knob to llama.cpp's ``chat_template_kwargs``.

        ``ThinkingMode.ADAPTIVE`` / ``ADAPTIVE_SUMMARIZED`` →
        ``{"enable_thinking": True}`` (llama.cpp has no "summarized" display mode,
        so both map to thinking-on); ``DISABLED`` → ``{"enable_thinking": False}``.
        Returns ``None`` when thinking is unset. An int token budget never
        reaches here — llamacpp doesn't declare ``"thinking_budget"``, so the
        request-time gate in ``Conversation._build_request`` rejects it first;
        any other non-mode value is treated defensively as "no kwarg" rather
        than forwarded as garbage. Mirrors Anthropic's ``_thinking_to_api``:
        a uniform knob value, a provider-specific wire form."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(
                "thinking= expects a ThinkingMode or a mode string "
                "('adaptive'/'adaptive_summarized'/'disabled') for the llamacpp "
                "provider — got a bool."
            )
        if isinstance(value, ThinkingMode):
            value = value.value
        if value == ThinkingMode.DISABLED.value:
            return {"enable_thinking": False}
        if value in (ThinkingMode.ADAPTIVE.value, ThinkingMode.ADAPTIVE_SUMMARIZED.value):
            return {"enable_thinking": True}
        return None

    def _warn_thinking_style_mismatch(self, model_id: str, thinking_val: Any) -> None:
        """Warn once per (model_id, style) when ``thinking`` is set against a
        model whose detected ``thinking_style`` won't honor the ``enable_thinking``
        kwarg (any recognised style other than ``TEMPLATE_KWARG``). ``UNKNOWN``
        is skipped — we couldn't detect the style, so a warning would be a false
        positive (and every external-mode model without an explicit
        ``thinking_style=`` is ``UNKNOWN``). "Never silently wrong": the kwarg is
        still emitted (harmless on a template that ignores it), but the caller is
        told it probably did nothing."""
        style = self._thinking_styles.get(model_id)
        if style is None or style in (ThinkingStyle.TEMPLATE_KWARG, ThinkingStyle.UNKNOWN):
            return
        tag = (model_id, style.value)
        if tag in _WARNED_THINKING_STYLE:
            return
        _WARNED_THINKING_STYLE.add(tag)
        warnings.warn(
            f"thinking={thinking_val!r} is set, but model {model_id!r}'s chat "
            f"template (thinking_style={style.value!r}) does not gate reasoning "
            f"via the enable_thinking kwarg, so the thinking knob likely has no "
            f"effect. Pass thinking_style= to new_model() to override detection "
            f"if this is wrong.",
            stacklevel=2,
        )

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
            state: dict[str, Any] = {
                "finish_reason": None,
                "reasoning": [],
                "reasoning_emitted": False,
            }
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
            state: dict[str, Any] = {
                "finish_reason": None,
                "reasoning": [],
                "reasoning_emitted": False,
            }
            async for chunk in stream:
                for ev in self._chunk_to_events(chunk, tool_buf, state):
                    yield ev
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e

    @staticmethod
    def _flush_reasoning(state: dict[str, Any]) -> StreamEvent | None:
        """Emit the accumulated reasoning as a single ``ThinkingBlock`` event,
        once, before the first non-reasoning content. Returns ``None`` if there
        is nothing to flush or it was already flushed."""
        if state.get("reasoning") and not state.get("reasoning_emitted"):
            state["reasoning_emitted"] = True
            return StreamEvent(thinking_block=ThinkingBlock(text="".join(state["reasoning"])))
        return None

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
            reasoning = _llama_reasoning(delta)
            if reasoning:
                state["reasoning"].append(reasoning)
                yield StreamEvent(thinking_delta=reasoning)
            text = getattr(delta, "content", None)
            if text:
                flushed = self._flush_reasoning(state)
                if flushed is not None:
                    yield flushed
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
                # Flush a reasoning-only turn (reasoning with no following text)
                # before any tool calls so it still lands in history.
                flushed = self._flush_reasoning(state)
                if flushed is not None:
                    yield flushed
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
            # Safety net: a usage chunk with no prior finish_reason still gets
            # any pending reasoning flushed before the terminal event.
            flushed = self._flush_reasoning(state)
            if flushed is not None:
                yield flushed
            yield StreamEvent(
                done=True,
                usage=Usage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    reasoning_tokens=_llama_reasoning_tokens(usage),
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
        reasoning = _llama_reasoning(msg)
        text = getattr(msg, "content", "") or ""
        # Reasoning leads the assistant turn (matches the canonical thinking-
        # then-text ordering the rest of the facade assumes).
        blocks: list[ContentBlock] = []
        if reasoning:
            blocks.append(ThinkingBlock(text=reasoning))
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
                reasoning_tokens=_llama_reasoning_tokens(raw.usage),
            )

        return Response(
            text=text,
            blocks=blocks,
            tool_calls=tool_calls,
            thinking=reasoning or None,
            usage=usage,
            finish_reason=choice.finish_reason,
            model=raw.model,
            raw=raw,
        )

    # ---- introspection ----------------------------------------------------

    def _resolve_introspection_target(self, model: str | None) -> str:
        """Return the path prefix to prepend to a llama-server-native endpoint.

        External mode: returns ``""`` and silently ignores ``model`` (bare
        llama-server has no model routing; passing ``model=`` against external
        mode is documented-harmless and parallel to how
        ``count_tokens(model_id=...)`` is ignored in external mode — keeps
        ``Model``-bound mirrors working uniformly across modes).

        Managed mode: returns ``"/upstream/<urlquoted-model>"``. An explicit
        ``model`` is passed through verbatim without checking the supervisor's
        registered entries — llama-swap loads on demand and the user may
        legitimately reference an entry from a hand-edited ``swap.yaml`` that
        ``-watch-config`` picked up. Resolves ``None`` by inferring iff
        exactly one entry is registered, else raises ``ValueError`` listing
        the registered ids."""
        if not self._managed:
            return ""
        if model is None:
            # Hold the supervisor's lock for the read so a concurrent
            # register() can't make the "exactly one" check stale by the time
            # we pick the inferred id. The lock is reentrant.
            if self._supervisor is None:
                entries: list[_LaunchEntry] = []
            else:
                with self._supervisor._lock:
                    entries = list(self._supervisor._entries)
            if not entries:
                raise ValueError(
                    "managed-mode introspection requires a model=<id> argument; "
                    "no models are registered on this provider yet."
                )
            if len(entries) > 1:
                names = [e.model_id for e in entries]
                raise ValueError(
                    "managed-mode introspection on a multi-model provider "
                    f"requires model=<id>; registered: {names!r}"
                )
            model = entries[0].model_id
        # safe="" so author/model-style ids get %2F-escaped and llama-swap
        # parses the slash inside the model id, not as another path segment.
        return f"/upstream/{_urlquote(model, safe='')}"

    def health(self, *, model: str | None = None) -> dict[str, Any]:
        """``GET /health``.

        With no ``model`` arg in managed mode, hits llama-swap's own
        ``/health`` (returns plain text ``"OK"``) and normalises the result to
        ``{"status": "ok"}``. With a ``model`` arg (or via ``Model.health()``),
        forwards through ``/upstream/<model>/health`` and returns the
        backend's JSON body. In external mode hits the bare server's
        ``/health`` directly."""
        self._ensure_supervised()
        if self._managed and model is None:
            return self._swap_root_health_sync()
        prefix = self._resolve_introspection_target(model)
        return self._http_get(f"{prefix}/health")

    async def ahealth(self, *, model: str | None = None) -> dict[str, Any]:
        self._ensure_supervised()
        if self._managed and model is None:
            return await self._swap_root_health_async()
        prefix = self._resolve_introspection_target(model)
        return await self._ahttp_get(f"{prefix}/health")

    def _swap_root_health_sync(self) -> dict[str, Any]:
        # llama-swap's own /health returns plain text "OK", not JSON. Don't
        # route through _http_get (which insists on JSON) — handle it inline.
        if self._http is None:
            raise ProviderError(
                "HTTP client not initialised (managed-mode supervisor not started yet)"
            )
        try:
            resp = self._http.get("/health")
        except Exception as e:
            raise ProviderError(f"GET /health failed: {e}", original=e) from e
        return self._normalise_swap_health(resp)

    async def _swap_root_health_async(self) -> dict[str, Any]:
        if self._ahttp is None:
            raise ProviderError(
                "HTTP client not initialised (managed-mode supervisor not started yet)"
            )
        try:
            resp = await self._ahttp.get("/health")
        except Exception as e:
            raise ProviderError(f"GET /health failed: {e}", original=e) from e
        return self._normalise_swap_health(resp)

    def _normalise_swap_health(self, resp: Any) -> dict[str, Any]:
        import contextlib

        status = getattr(resp, "status_code", None)
        if status is None or status >= 400:
            body = ""
            with contextlib.suppress(Exception):
                body = resp.text
            raise ProviderError(f"GET /health returned status {status}: {body}")
        body = getattr(resp, "text", "") or ""
        # httpx.Response.text is always str, but be defensive — a streaming
        # response that hasn't been .read() can yield bytes through .content
        # paths and a fake might too; equality against "OK" silently fails on
        # bytes and would leak `{"status": b"OK"}` to the caller.
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        body = body.strip()
        return {"status": "ok"} if body.upper() == "OK" else {"status": body}

    def slots(self, *, model: str | None = None) -> list[dict[str, Any]]:
        """``GET /slots``. Per-slot processing state, sampling params, token
        counts, and generation speed. In managed mode pass ``model=`` (or use
        ``Model.slots()``) to pick the backend."""
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        data = self._http_get(f"{prefix}/slots")
        return data if isinstance(data, list) else []

    async def aslots(self, *, model: str | None = None) -> list[dict[str, Any]]:
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        data = await self._ahttp_get(f"{prefix}/slots")
        return data if isinstance(data, list) else []

    def slot_lock(self) -> Any:
        """Per-provider ``threading.Lock`` for span-locking sync slot ops.

        Use as ``with provider.slot_lock(): convo.restore_slot(...); convo.send(...)``
        to make a restore + send pair atomic against concurrent slot
        mutations on the same provider. Returns the underlying Lock so the
        caller can ``with``-acquire it; non-reentrant. The conversation-level
        slot methods do NOT acquire this lock internally — that would deadlock
        any caller already holding it. Pair only with the sync API; sync and
        async locks do not synchronise against each other."""
        return self._slot_lock

    def aslot_lock(self) -> Any:
        """Per-provider ``asyncio.Lock`` for span-locking async slot ops.

        Use as ``async with provider.aslot_lock(): await convo.arestore_slot(...);
        await convo.asend(...)`` to make a restore + send pair atomic. See
        ``slot_lock`` for the contract; pair only with the async API."""
        return self._slot_alock

    def save_slot(
        self, id_slot: int, filename: str, *, model: str | None = None
    ) -> dict[str, Any]:
        """``POST /slots/{id_slot}?action=save`` body ``{"filename": filename}``.
        ``filename`` is interpreted relative to that backend's
        ``--slot-save-path`` directory (which differs across registered
        managed-mode models)."""
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        return self._http_post(
            f"{prefix}/slots/{id_slot}", params={"action": "save"}, json={"filename": filename}
        )

    async def asave_slot(
        self, id_slot: int, filename: str, *, model: str | None = None
    ) -> dict[str, Any]:
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        return await self._ahttp_post(
            f"{prefix}/slots/{id_slot}", params={"action": "save"}, json={"filename": filename}
        )

    def restore_slot(
        self, id_slot: int, filename: str, *, model: str | None = None
    ) -> dict[str, Any]:
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        return self._http_post(
            f"{prefix}/slots/{id_slot}",
            params={"action": "restore"},
            json={"filename": filename},
        )

    async def arestore_slot(
        self, id_slot: int, filename: str, *, model: str | None = None
    ) -> dict[str, Any]:
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        return await self._ahttp_post(
            f"{prefix}/slots/{id_slot}",
            params={"action": "restore"},
            json={"filename": filename},
        )

    def erase_slot(self, id_slot: int, *, model: str | None = None) -> dict[str, Any]:
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        return self._http_post(f"{prefix}/slots/{id_slot}", params={"action": "erase"})

    async def aerase_slot(self, id_slot: int, *, model: str | None = None) -> dict[str, Any]:
        self._ensure_supervised()
        prefix = self._resolve_introspection_target(model)
        return await self._ahttp_post(f"{prefix}/slots/{id_slot}", params={"action": "erase"})

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

    def interrupt(self) -> bool:
        """Instantly abort an in-flight local generation by hard-killing the
        managed llama-swap backend process tree.

        Designed to be called from a *different thread* than the one blocked
        inside ``send()`` / ``stream()`` (i.e. parked in the OpenAI SDK's HTTP
        call): terminating the backend out from under the blocked call makes it
        raise a transport error promptly, which a consumer can treat as a clean
        cancel. This is NOT a graceful drain — it does not wait for the current
        request or token batch to finish (a graceful unload that blocks until
        the request completes would defeat the purpose).

        Recoverable: leaves the provider in the lazy-spawn state, so the next
        ``send()`` / ``stream()`` respawns llama-swap and reloads the model just
        like the first call did (``_ensure_supervised`` rebuilds the clients
        against the freshly allocated port). Idempotent and safe to call
        repeatedly.

        Returns ``True`` if a backend was actually killed, ``False`` if nothing
        was running (idle, no model loaded yet) or in **external mode** (we
        don't own the process there). Never raises for the nothing-to-kill
        case."""
        if not self._managed or self._supervisor is None:
            return False
        return self._supervisor.interrupt()

    # ---- fit estimation + metadata ---------------------------------------

    def _maybe_estimate_fit(self, entry: _LaunchEntry) -> None:
        """Best-effort: spawn `llama-fit-params` with this entry's launch flags
        and stash the parsed estimate keyed by `entry.model_id`. Records `None`
        on any failure (binary missing, non-zero exit, timeout, unparseable
        output) so `log_metadata` can skip the field for that model. Never
        raises — `new_model()` callers see at most a brief synchronous wait
        (capped by `_FIT_PARAMS_TIMEOUT_SECONDS`, currently 15s; the probe is
        normally sub-second once GPU is warm). `entry.extra_args` is
        intentionally NOT forwarded: those are llama-server-specific flags
        that `llama-fit-params` will reject and exit non-zero."""
        import shutil as _shutil
        import subprocess as _subprocess

        if not entry.fit:
            self._fit_estimates[entry.model_id] = None
            return
        binary = _shutil.which("llama-fit-params")
        if binary is None:
            self._fit_estimates[entry.model_id] = None
            return

        argv: list[str] = [binary, "--model", entry.gguf]
        if entry.context_size is not None:
            argv += ["--ctx-size", str(entry.context_size)]
        if entry.cache_type_k is not None:
            argv += ["--cache-type-k", entry.cache_type_k]
        if entry.cache_type_v is not None:
            argv += ["--cache-type-v", entry.cache_type_v]
        if entry.n_gpu_layers is not None:
            argv += ["--n-gpu-layers", str(entry.n_gpu_layers)]
        if entry.n_cpu_moe is not None:
            argv += ["--n-cpu-moe", str(entry.n_cpu_moe)]
        if entry.parallel is not None:
            argv += ["--parallel", str(entry.parallel)]
        if entry.fit_target is not None:
            argv += ["--fit-target", ",".join(str(v) for v in entry.fit_target)]
        if entry.fit_ctx is not None:
            argv += ["--fit-ctx", str(entry.fit_ctx)]
        if entry.flash_attn is not None:
            argv += ["--flash-attn", entry.flash_attn]
        if entry.mmproj_path is not None:
            argv += ["--mmproj", entry.mmproj_path]

        try:
            result = _subprocess.run(
                argv,
                capture_output=True,
                timeout=self._FIT_PARAMS_TIMEOUT_SECONDS,
                check=False,
                text=True,
                stdin=_subprocess.DEVNULL,
            )
        except (OSError, _subprocess.SubprocessError):
            self._fit_estimates[entry.model_id] = None
            return
        if result.returncode != 0:
            self._fit_estimates[entry.model_id] = None
            return
        parsed = parse_fit_print(result.stdout or "", result.stderr or "")
        if parsed is None:
            self._fit_estimates[entry.model_id] = None
            return
        # Translate llama.cpp's CLI sentinels to readable labels so the log
        # block doesn't leave a user puzzling over `context_size: 0`. fit-params
        # prints `-c N -ngl N` verbatim even when the model fits at defaults
        # without any reduction; in that case N is the unset default
        # (0 = "use the model's trained context", -1 = "all layers on GPU").
        if parsed.get("context_size") == 0:
            parsed["context_size"] = "model default"
        if parsed.get("n_gpu_layers") == -1:
            parsed["n_gpu_layers"] = "all"
        if entry.parallel is not None:
            parsed.setdefault("parallel", entry.parallel)
        self._fit_estimates[entry.model_id] = parsed

    def log_metadata(self, *, model_id: str) -> dict[str, Any] | None:
        """Surface per-model extras for the conversation log header: the cached
        `fit_estimate` (managed mode only) and the detected/declared
        `thinking_style` (either mode, when known). Returns `None` when neither
        is known. The inner `fit_estimate` is a shallow copy — currently safe
        because every value is a primitive; if it ever holds nested containers,
        switch to `copy.deepcopy`."""
        out: dict[str, Any] = {}
        if self._managed:
            est = self._fit_estimates.get(model_id)
            if est is not None:
                out["fit_estimate"] = dict(est)
        style = self._thinking_styles.get(model_id)
        if style is not None:
            out["thinking_style"] = style.value
        return out or None

    # ---- token counting ---------------------------------------------------

    def count_tokens(
        self,
        text: str,
        *,
        system: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Count tokens by calling the running llama-server's ``/tokenize``
        endpoint. Falls back to ``chars/4`` if the server is unreachable, the
        endpoint 404s, or (in managed mode) no model is registered/specified
        so the routing helper raises — logging never blocks on these."""
        combined = text + (system or "")
        if self._http is None:
            return super().count_tokens(text, system=system)
        try:
            prefix = self._resolve_introspection_target(model_id)
            resp = self._http.post(f"{prefix}/tokenize", json={"content": combined})
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
