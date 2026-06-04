"""Shared helpers for the OpenAI *Images* API surface.

Both the hosted OpenAI provider and the local-image provider talk to an
OpenAI-compatible images endpoint (``/v1/images/generations`` and
``/v1/images/edits``) via the ``openai`` Python SDK — the same reuse trick the
llamacpp provider uses for the chat surface. These functions build the request
kwargs and parse the response into an :class:`ImageResult`, so the per-provider
methods only own the actual (sync/async) SDK call and error mapping.
"""

from __future__ import annotations

from typing import Any

from llmfacade.exceptions import ProviderError
from llmfacade.models import ImageBlock, ImageResult, ImageUsage


def media_type_for(output_format: str | None) -> str:
    """Map an OpenAI ``output_format`` (png/jpeg/webp) to a MIME type. The
    images response carries no per-image MIME, so this drives the
    :class:`ImageBlock` ``media_type`` for generated images. Defaults to PNG."""
    return {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        output_format or "png", "image/png"
    )


def build_generate_kwargs(
    *,
    model: str | None,
    prompt: str,
    n: int,
    size: str | None,
    quality: str | None,
    background: str | None,
    output_format: str | None,
    extra: dict[str, Any] | None,
    request_b64: bool,
) -> dict[str, Any]:
    """Kwargs for ``client.images.generate``. ``request_b64`` emits
    ``response_format="b64_json"`` (needed for local servers; omitted for
    ``gpt-image-*`` which always returns base64 and rejects the param). A
    ``model`` of ``None`` is omitted so the server uses its loaded default."""
    kwargs: dict[str, Any] = {"prompt": prompt, "n": n}
    if model is not None:
        kwargs["model"] = model
    if size is not None:
        kwargs["size"] = size
    if quality is not None:
        kwargs["quality"] = quality
    if background is not None:
        kwargs["background"] = background
    if output_format is not None:
        kwargs["output_format"] = output_format
    if request_b64:
        kwargs["response_format"] = "b64_json"
    if extra:
        # `mask` is an edits-only concept (build_edit_kwargs pops it); on the
        # plain generate path it would just ride along as an unused extra_body
        # field. Callers pass it only with reference_images, so it never lands here.
        kwargs["extra_body"] = dict(extra)
    return kwargs


def build_edit_kwargs(
    *,
    model: str | None,
    prompt: str,
    reference_images: list[ImageBlock] | None,
    n: int,
    size: str | None,
    extra: dict[str, Any] | None,
    request_b64: bool,
) -> dict[str, Any]:
    """Kwargs for ``client.images.edit``. Reference images become the
    ``(filename, bytes, mimetype)`` tuples the SDK uploads as multipart; a
    ``mask`` (if present in ``extra``) is forwarded as the edit mask. A ``model``
    of ``None`` is omitted so the server uses its loaded default."""
    images = [
        (f"ref_{i}{_ext(b.media_type)}", b.data, b.media_type)
        for i, b in enumerate(reference_images or [])
    ]
    kwargs: dict[str, Any] = {"prompt": prompt, "image": images, "n": n}
    if model is not None:
        kwargs["model"] = model
    if size is not None:
        kwargs["size"] = size
    if request_b64:
        kwargs["response_format"] = "b64_json"
    rest = dict(extra or {})
    mask = rest.pop("mask", None)
    if mask is not None:
        kwargs["mask"] = mask
    if rest:
        kwargs["extra_body"] = rest
    return kwargs


def parse_images_response(
    raw: Any,
    *,
    model: str,
    provider: str,
    fallback_media_type: str = "image/png",
) -> ImageResult:
    """Turn an OpenAI-shaped images response into an :class:`ImageResult`,
    reading ``data[i].b64_json``. The per-image MIME type is not carried on the
    response, so ``fallback_media_type`` (derived from the requested
    ``output_format``) is used."""
    data = getattr(raw, "data", None) or []
    images: list[ImageBlock] = []
    for d in data:
        b64 = getattr(d, "b64_json", None)
        if not b64:
            if getattr(d, "url", None):
                raise ProviderError(
                    f"{provider}: image was returned as a URL, not base64. Request "
                    f"response_format='b64_json' or use a base64-returning model "
                    f"(e.g. gpt-image-1)."
                )
            raise ProviderError(f"{provider}: image response carried no b64_json data.")
        images.append(ImageBlock.from_base64(b64, media_type=fallback_media_type))
    return ImageResult(
        images=images,
        usage=_usage_from(raw, len(images)),
        model=model,
        provider=provider,
        raw=raw,
    )


def _usage_from(raw: Any, image_count: int) -> ImageUsage:
    u = getattr(raw, "usage", None)
    if u is None:
        return ImageUsage(image_count=image_count)
    return ImageUsage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        total_tokens=getattr(u, "total_tokens", 0) or 0,
        image_count=image_count,
    )


def _ext(media_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(media_type, ".png")
