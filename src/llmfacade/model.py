from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmfacade.provider import _validate_knobs

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
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
        tool_choice: str | None = None,
    ):
        self._provider = provider
        self._model_id = model_id
        self._log_dir_override = log_dir
        self._supports: frozenset[str] = (
            capability_override if capability_override is not None else provider.SUPPORTS
        )
        self._defaults = _validate_knobs(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "effort": effort,
                "thinking": thinking,
                "output_format": output_format,
                "user_metadata": user_metadata,
                "cache_ttl": cache_ttl,
                "auto_cache_last_user": auto_cache_last_user,
                "auto_cache_tools": auto_cache_tools,
                "beta_headers": beta_headers,
                "keep_alive": keep_alive,
                "context_size": context_size,
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

    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the provider's local tokenizer for
        this model. Convenience wrapper around ``provider.count_tokens(text,
        model_id=self.model_id)``."""
        return self._provider.count_tokens(text, model_id=self._model_id)

    def tokenizer_name(self) -> str:
        """Label of the tokenizer ``count_tokens`` will use for this model."""
        return self._provider.tokenizer_name(model_id=self._model_id)

    def new_conversation(
        self,
        *,
        name: str | None = None,
        system_blocks: list[SystemBlock | str] | None = None,
        tools: list[Tool] | None = None,
        log_dir: Any | None = None,
        log_path: Any | None = None,
        log_max_message_lines: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        effort: Any | None = None,
        thinking: int | None = None,
        output_format: Any | None = None,
        user_metadata: dict[str, str] | None = None,
        cache_ttl: Any | None = None,
        auto_cache_last_user: bool | None = None,
        auto_cache_tools: bool | None = None,
        beta_headers: list[str] | None = None,
        keep_alive: str | int | None = None,
        context_size: int | None = None,
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
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            effort=effort,
            thinking=thinking,
            output_format=output_format,
            user_metadata=user_metadata,
            cache_ttl=cache_ttl,
            auto_cache_last_user=auto_cache_last_user,
            auto_cache_tools=auto_cache_tools,
            beta_headers=beta_headers,
            keep_alive=keep_alive,
            context_size=context_size,
            tool_choice=tool_choice,
        )

    def __repr__(self) -> str:
        return f"Model(provider={self._provider.NAME!r}, model_id={self._model_id!r})"
