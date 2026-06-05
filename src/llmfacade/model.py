from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import UnsupportedFeature
from llmfacade.provider import _validate_knobs
from llmfacade.settings import DrySampler, ThinkingMode

if TYPE_CHECKING:
    from llmfacade.conversation import Conversation
    from llmfacade.provider import Provider, SystemBlock
    from llmfacade.tools import Tool


class Model:
    """A specific model_id bound to a Provider, with optional model-level
    generation defaults. Identity (provider, model_id) is constructor-only.

    ``capability_override`` narrows the set of supported settings below the
    provider's full SUPPORTS — used for models that don't honor a feature
    the provider generally implements (e.g. extended thinking)."""

    def __init__(
        self,
        *,
        provider: Provider,
        model_id: str,
        capability_override: frozenset[str] | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        repetition_detection: Any | None = None,
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
    ):
        self._provider = provider
        self._model_id = model_id
        self._log_dir_override = log_dir
        self._cache_dir_override = cache_dir
        self._cache_mode_override = cache_mode
        self._repetition_override = repetition_detection
        self._supports: frozenset[str] = (
            capability_override if capability_override is not None else provider.SUPPORTS
        )
        self._defaults = _validate_knobs(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "dry": dry,
                "effort": effort,
                "thinking": thinking,
                "output_format": output_format,
                "user_metadata": user_metadata,
                "cache_ttl": cache_ttl,
                "auto_cache_last_user": auto_cache_last_user,
                "auto_cache_tools": auto_cache_tools,
                "beta_headers": beta_headers,
                "tool_choice": tool_choice,
            },
            self._supports,
            provider.NAME,
            model_id,
        )

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self._defaults)

    def is_available(self, setting: str) -> bool:
        return setting in self._supports

    def get_capabilities(self) -> set[str]:
        return set(self._supports)

    def count_tokens(self, text: str, *, system: str | None = None) -> int:
        """Count tokens in ``text`` using the provider's local tokenizer for
        this model. Convenience wrapper around ``provider.count_tokens(text,
        system=..., model_id=self.model_id)``.

        Pass ``system=`` to count a system prompt alongside ``text`` (the
        Anthropic provider with ``exact_count_tokens=True`` forwards it as
        the SDK's ``system=`` kwarg so role overhead matches the actual
        generation call; other providers add the local count of ``system``
        to ``text``)."""
        return self._provider.count_tokens(text, system=system, model_id=self._model_id)

    def tokenizer_name(self) -> str:
        """Label of the tokenizer ``count_tokens`` will use for this model."""
        return self._provider.tokenizer_name(model_id=self._model_id)

    # ---- llamacpp-only introspection passthroughs --------------------------
    # Each binds self._model_id and forwards to the provider, mirroring the
    # count_tokens pattern above. The provider must expose the corresponding
    # method (only LlamaCppServerProvider does today); duck-typed via
    # _require_provider_method to avoid the circular import that an isinstance
    # check would need, and surfaces UnsupportedFeature on miss to match the
    # codebase's cross-provider capability-gating idiom.

    def _require_provider_method(self, name: str) -> Any:
        method = getattr(self._provider, name, None)
        if method is None:
            raise UnsupportedFeature(name, self._provider.NAME, self._model_id)
        return method

    def health(self) -> dict[str, Any]:
        """Backend health for this specific model. llamacpp-only."""
        return self._require_provider_method("health")(model=self._model_id)

    async def ahealth(self) -> dict[str, Any]:
        return await self._require_provider_method("ahealth")(model=self._model_id)

    def slots(self) -> list[dict[str, Any]]:
        """Per-slot processing state for this model's backend. llamacpp-only."""
        return self._require_provider_method("slots")(model=self._model_id)

    async def aslots(self) -> list[dict[str, Any]]:
        return await self._require_provider_method("aslots")(model=self._model_id)

    def save_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        """Save the KV cache for ``id_slot`` to ``filename`` (relative to
        this model's ``--slot-save-path``). llamacpp-only."""
        return self._require_provider_method("save_slot")(id_slot, filename, model=self._model_id)

    async def asave_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        return await self._require_provider_method("asave_slot")(
            id_slot, filename, model=self._model_id
        )

    def restore_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        return self._require_provider_method("restore_slot")(
            id_slot, filename, model=self._model_id
        )

    async def arestore_slot(self, id_slot: int, filename: str) -> dict[str, Any]:
        return await self._require_provider_method("arestore_slot")(
            id_slot, filename, model=self._model_id
        )

    def erase_slot(self, id_slot: int) -> dict[str, Any]:
        return self._require_provider_method("erase_slot")(id_slot, model=self._model_id)

    async def aerase_slot(self, id_slot: int) -> dict[str, Any]:
        return await self._require_provider_method("aerase_slot")(id_slot, model=self._model_id)

    def new_conversation(
        self,
        *,
        name: str | None = None,
        system_blocks: list[SystemBlock | str] | None = None,
        tools: list[Tool] | None = None,
        log_dir: Any | None = None,
        log_path: Any | None = None,
        log_max_message_lines: int | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        repetition_detection: Any | None = None,
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
    ) -> Conversation:
        from llmfacade.conversation import Conversation

        return Conversation(
            model=self,
            name=name,
            system_blocks=system_blocks,
            tools=tools,
            log_dir=log_dir,
            log_path=log_path,
            log_max_message_lines=log_max_message_lines,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            repetition_detection=repetition_detection,
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

    def __repr__(self) -> str:
        return f"Model(provider={self._provider.NAME!r}, model_id={self._model_id!r})"
