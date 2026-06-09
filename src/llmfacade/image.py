from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from llmfacade.models import ImageResult, ReferenceImage
    from llmfacade.provider import Provider

_IMAGE_DEFAULT_KEYS = (
    "n",
    "size",
    "aspect_ratio",
    "quality",
    "background",
    "output_format",
)


class ImageModel:
    """An image-generation ``model_id`` bound to a Provider, with optional
    per-model defaults. The image analog of :class:`~llmfacade.model.Model`.

    Defaults set here fill any argument left unset on ``generate`` /
    ``agenerate``; an explicit per-call value always wins. ``capability_override``
    narrows the provider's SUPPORTS for this model, same as ``Model``."""

    def __init__(
        self,
        *,
        provider: Provider,
        model_id: str,
        capability_override: frozenset[str] | None = None,
        n: int | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
    ):
        self._provider = provider
        self._model_id = model_id
        self._supports: frozenset[str] = (
            capability_override if capability_override is not None else provider.SUPPORTS
        )
        candidates = {
            "n": n,
            "size": size,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
        self._defaults: dict[str, Any] = {k: v for k, v in candidates.items() if v is not None}

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

    def _merge(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Per-call value wins; else this model's default; else omitted (so the
        provider method's own default applies)."""
        out: dict[str, Any] = {}
        for key in _IMAGE_DEFAULT_KEYS:
            value = overrides.get(key)
            if value is None:
                value = self._defaults.get(key)
            if value is not None:
                out[key] = value
        return out

    def generate(
        self,
        prompt: str,
        *,
        n: int | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        reference_images: Sequence[ReferenceImage] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        merged = self._merge(
            {
                "n": n,
                "size": size,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "background": background,
                "output_format": output_format,
            }
        )
        return self._provider.generate_image(
            prompt,
            model=self._model_id,
            reference_images=reference_images,
            save_dir=save_dir,
            extra=extra,
            **merged,
        )

    async def agenerate(
        self,
        prompt: str,
        *,
        n: int | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        reference_images: Sequence[ReferenceImage] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        merged = self._merge(
            {
                "n": n,
                "size": size,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "background": background,
                "output_format": output_format,
            }
        )
        return await self._provider.agenerate_image(
            prompt,
            model=self._model_id,
            reference_images=reference_images,
            save_dir=save_dir,
            extra=extra,
            **merged,
        )

    def __repr__(self) -> str:
        return f"ImageModel(provider={self._provider.NAME!r}, model_id={self._model_id!r})"
