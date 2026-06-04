from __future__ import annotations

from pathlib import Path
from typing import Any

from llmfacade.exceptions import (
    AuthenticationError,
    LLMError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
)
from llmfacade.models import ImageBlock, ImageResult, _apply_save_dir
from llmfacade.provider import Provider
from llmfacade.providers._openai_images import (
    build_edit_kwargs,
    build_generate_kwargs,
    media_type_for,
    parse_images_response,
)


class LocalImageProvider(Provider):
    """Local image generation against an OpenAI-compatible image server.

    Targets any server that speaks the OpenAI Images API — stable-diffusion.cpp's
    ``sd-server`` (the image analog of ``llama-server``; Flux / SD / SDXL /
    Qwen-Image), LocalAI, etc. — reusing the ``openai`` SDK as transport, exactly
    as the llamacpp provider reuses it for chat.

    **External mode only** (this release): pass ``base_url`` pointing at a running
    server, e.g. ``LLM().new_provider("localimage",
    base_url="http://127.0.0.1:1234/v1")``. No auth is required (``api_key`` is
    optional; a placeholder is sent to satisfy the SDK). A **managed mode** that
    spawns and supervises ``sd-server`` (mirroring the llamacpp ``llama-swap``
    supervisor) is a planned follow-up.

    Server-specific knobs that have no OpenAI equivalent (steps, cfg, sampler,
    seed, ...) go through ``extra=`` — forwarded as the SDK's ``extra_body`` for
    servers that read JSON fields (LocalAI). For ``sd-server`` specifically, those
    are instead embedded in the prompt via its
    ``<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`` syntax.
    """

    NAME = "localimage"
    API_KEY_ENV = None
    SUPPORTS: frozenset[str] = frozenset({"image_generation"})

    def _init_client(self) -> None:
        if not self._base_url:
            raise LLMError(
                "localimage requires base_url= pointing at a running "
                "OpenAI-compatible image server (e.g. sd-server "
                "'http://127.0.0.1:1234/v1' or LocalAI). Managed mode (spawning "
                "sd-server) is not yet implemented."
            )
        try:
            import openai as _openai
        except ImportError as e:
            raise ProviderNotInstalledError(
                "OpenAI SDK not installed. Run: pip install llmfacade[localimage]"
            ) from e
        # Local servers need no key; the SDK requires a non-empty value.
        api_key = self._api_key_override or "-"
        self._client = _openai.OpenAI(base_url=self._base_url, api_key=api_key)
        self._aclient = _openai.AsyncOpenAI(base_url=self._base_url, api_key=api_key)
        self._module = _openai

    def _image_kwargs(
        self,
        prompt: str,
        model: str | None,
        n: int,
        size: str | None,
        quality: str | None,
        background: str | None,
        output_format: str | None,
        reference_images: list[ImageBlock] | None,
        extra: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        # request_b64=True: OpenAI-compatible local servers default to returning
        # URLs; we want the bytes inline.
        if reference_images:
            return "edit", build_edit_kwargs(
                model=model,
                prompt=prompt,
                reference_images=reference_images,
                n=n,
                size=size,
                extra=extra,
                request_b64=True,
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
            request_b64=True,
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
        reference_images: list[ImageBlock] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
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
            raw,
            model=model or "localimage",
            provider=self.NAME,
            fallback_media_type=media_type_for(output_format),
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
        reference_images: list[ImageBlock] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
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
            raw,
            model=model or "localimage",
            provider=self.NAME,
            fallback_media_type=media_type_for(output_format),
        )
        return _apply_save_dir(result, save_dir)
