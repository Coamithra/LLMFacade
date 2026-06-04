"""Image generation across the OpenAI, Google (Gemini-native) and local
providers. The SDK client is mocked on each provider — no real API call is
fired — so these assert the request shape the facade builds and the
``ImageResult`` it parses back.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmfacade import (
    LLM,
    ImageBlock,
    ImageModel,
    ImageResult,
    ImageUsage,
)
from llmfacade.exceptions import LLMError, ProviderError, UnsupportedFeature
from llmfacade.providers.anthropic import AnthropicProvider
from llmfacade.providers.google import GoogleProvider
from llmfacade.providers.llamacpp import LlamaCppServerProvider
from llmfacade.providers.localimage import LocalImageProvider
from llmfacade.providers.openai import OpenAIProvider

# ---- fakes -----------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _openai_image_response(*datas: bytes, usage: object | None = None) -> SimpleNamespace:
    data = [SimpleNamespace(b64_json=_b64(d), url=None) for d in datas]
    return SimpleNamespace(data=data, usage=usage)


def _openai_usage(inp: int = 12, out: int = 34) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=inp, output_tokens=out, total_tokens=inp + out)


def _gemini_image_response(*datas: bytes, usage: object | None = None) -> SimpleNamespace:
    parts = [
        SimpleNamespace(inline_data=SimpleNamespace(data=d, mime_type="image/png")) for d in datas
    ]
    cand = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[cand], usage_metadata=usage)


def _gemini_usage(prompt: int = 5, candidates: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=prompt + candidates,
    )


# ---- OpenAI ----------------------------------------------------------------


def test_openai_generate_returns_image_result():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(
        b"PNGDATA", usage=_openai_usage()
    )

    result = provider.generate_image("a cat", model="gpt-image-1", size="1024x1024")

    assert isinstance(result, ImageResult)
    assert result.provider == "openai"
    assert result.model == "gpt-image-1"
    assert [b.data for b in result.images] == [b"PNGDATA"]
    assert result.images[0].media_type == "image/png"
    assert result.usage == ImageUsage(
        input_tokens=12, output_tokens=34, total_tokens=46, image_count=1
    )

    kwargs = provider._client.images.generate.call_args.kwargs
    assert kwargs["prompt"] == "a cat"
    assert kwargs["model"] == "gpt-image-1"
    assert kwargs["size"] == "1024x1024"
    assert kwargs["n"] == 1
    # gpt-image-* always returns base64 and rejects response_format.
    assert "response_format" not in kwargs
    provider._client.images.edit.assert_not_called()


def test_openai_reference_images_route_to_edit():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.edit.return_value = _openai_image_response(b"EDITED")

    result = provider.generate_image(
        "make it night",
        model="gpt-image-1",
        reference_images=[ImageBlock(data=b"REF", media_type="image/png")],
    )

    assert result.images[0].data == b"EDITED"
    provider._client.images.generate.assert_not_called()
    kwargs = provider._client.images.edit.call_args.kwargs
    assert isinstance(kwargs["image"], list)
    fname, content, mime = kwargs["image"][0]
    assert content == b"REF"
    assert mime == "image/png"


def test_openai_output_format_drives_media_type():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"JPG")

    result = provider.generate_image("x", model="gpt-image-1", output_format="jpeg")

    assert result.images[0].media_type == "image/jpeg"
    assert provider._client.images.generate.call_args.kwargs["output_format"] == "jpeg"


def test_openai_edit_mask_forwarded():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.edit.return_value = _openai_image_response(b"M")

    provider.generate_image(
        "inpaint",
        model="gpt-image-1",
        reference_images=[ImageBlock(data=b"REF", media_type="image/png")],
        extra={"mask": b"MASKBYTES"},
    )

    kwargs = provider._client.images.edit.call_args.kwargs
    assert kwargs["mask"] == b"MASKBYTES"
    assert "extra_body" not in kwargs  # mask is popped out of extra


def test_openai_agenerate_image():
    provider = OpenAIProvider(api_key="test-key")
    provider._aclient = MagicMock()
    provider._aclient.images.generate = AsyncMock(return_value=_openai_image_response(b"ASYNC"))

    result = asyncio.run(provider.agenerate_image("a cat", model="gpt-image-1"))

    assert result.images[0].data == b"ASYNC"


# ---- Google (Gemini-native) ------------------------------------------------


def test_google_generate_uses_image_modality():
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(
        b"DRAGON", usage=_gemini_usage()
    )

    result = provider.generate_image("a dragon", aspect_ratio="16:9")

    assert result.provider == "google"
    assert result.model == "gemini-2.5-flash-image"
    assert result.images[0].data == b"DRAGON"
    assert result.usage == ImageUsage(
        input_tokens=5, output_tokens=100, total_tokens=105, image_count=1
    )

    kwargs = provider._client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash-image"
    assert kwargs["config"]["response_modalities"] == ["IMAGE"]
    assert kwargs["config"]["image_config"] == {"aspect_ratio": "16:9"}
    assert kwargs["contents"][0]["parts"][0]["text"] == "a dragon"


def test_google_reference_images_embedded_in_contents():
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(b"OUT")

    provider.generate_image(
        "same character, new pose",
        reference_images=[ImageBlock(data=b"REF", media_type="image/png")],
    )

    parts = provider._client.models.generate_content.call_args.kwargs["contents"][0]["parts"]
    inline = [p for p in parts if "inline_data" in p]
    assert len(inline) == 1
    assert inline[0]["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(inline[0]["inline_data"]["data"]) == b"REF"


def test_google_agenerate_image():
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = AsyncMock(
        return_value=_gemini_image_response(b"AIMG")
    )

    result = asyncio.run(provider.agenerate_image("a fox"))
    assert result.images[0].data == b"AIMG"


def test_google_warns_and_drops_n():
    """Gemini-native emits one image per call; n>1 must warn and never leak into
    the request (it has no such param). Pins the silent-drop so a future wiring
    of n is a deliberate change."""
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(b"X")

    with pytest.warns(UserWarning, match="n=4 is ignored"):
        provider.generate_image("a dragon", n=4)

    kwargs = provider._client.models.generate_content.call_args.kwargs
    assert "n" not in kwargs
    assert "n" not in kwargs["config"]


# ---- local image -----------------------------------------------------------


def test_localimage_requires_base_url():
    with pytest.raises(LLMError, match="base_url"):
        LocalImageProvider()


def test_localimage_requests_b64_and_parses():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"FLUX")

    result = provider.generate_image("a fox in the snow", model="flux")

    assert result.provider == "localimage"
    assert result.model == "flux"
    assert result.images[0].data == b"FLUX"
    assert provider._client.images.generate.call_args.kwargs["response_format"] == "b64_json"


def test_localimage_extra_forwarded_as_extra_body():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"X")

    provider.generate_image("x", model="flux", extra={"steps": 20})

    assert provider._client.images.generate.call_args.kwargs["extra_body"] == {"steps": 20}


def test_localimage_url_only_response_raises():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = SimpleNamespace(
        data=[SimpleNamespace(b64_json=None, url="http://server/img.png")], usage=None
    )

    with pytest.raises(ProviderError, match="URL"):
        provider.generate_image("x", model="flux")


def test_localimage_model_omitted_when_none():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"X")

    provider.generate_image("x")

    assert "model" not in provider._client.images.generate.call_args.kwargs


# ---- ImageResult.save ------------------------------------------------------


def test_image_result_save(tmp_path):
    result = ImageResult(
        images=[
            ImageBlock(data=b"AAA", media_type="image/png"),
            ImageBlock(data=b"BBB", media_type="image/jpeg"),
        ],
        usage=None,
        model="m",
        provider="p",
    )

    paths = result.save(tmp_path)

    assert [p.name for p in paths] == ["image_0.png", "image_1.jpg"]
    assert paths[0].read_bytes() == b"AAA"
    assert paths[1].read_bytes() == b"BBB"


def test_save_dir_populates_paths(tmp_path):
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"PNGDATA")

    result = provider.generate_image("x", model="gpt-image-1", save_dir=tmp_path)

    assert len(result.paths) == 1
    assert result.paths[0].read_bytes() == b"PNGDATA"


# ---- capability gating ------------------------------------------------------


def test_image_generation_capability_flags():
    assert "image_generation" in OpenAIProvider.SUPPORTS
    assert "image_generation" in GoogleProvider.SUPPORTS
    assert "image_generation" in LocalImageProvider.SUPPORTS
    assert "image_generation" not in AnthropicProvider.SUPPORTS
    assert "image_generation" not in LlamaCppServerProvider.SUPPORTS


def test_base_provider_generate_image_raises():
    provider = AnthropicProvider(api_key="test-key")
    assert not provider.is_available("image_generation")
    with pytest.raises(UnsupportedFeature):
        provider.generate_image("x", model="claude")


# ---- ImageModel ------------------------------------------------------------


def test_image_model_binds_and_applies_defaults():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"X")

    im = provider.new_image_model("gpt-image-1", size="1024x1024", quality="high")
    assert isinstance(im, ImageModel)
    assert im.model_id == "gpt-image-1"

    im.generate("a cat")
    kwargs = provider._client.images.generate.call_args.kwargs
    assert kwargs["model"] == "gpt-image-1"
    assert kwargs["size"] == "1024x1024"
    assert kwargs["quality"] == "high"


def test_image_model_per_call_overrides_default():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"X")

    im = provider.new_image_model("gpt-image-1", size="1024x1024")
    im.generate("a cat", size="512x512")

    assert provider._client.images.generate.call_args.kwargs["size"] == "512x512"


# ---- LLM.generate_image convenience ----------------------------------------


def test_llm_generate_image_resolves_and_caches():
    llm = LLM(api_keys={"openai": "test-key"}, log_dir=False)
    fake = MagicMock()
    fake.generate_image.return_value = "RESULT"
    built: list[tuple[str, dict]] = []

    def fake_new_provider(name, **kw):
        built.append((name, kw))
        return fake

    llm.new_provider = fake_new_provider  # type: ignore[method-assign]

    out = llm.generate_image("a cat", provider="openai", model="gpt-image-1", size="1024x1024")
    assert out == "RESULT"
    fake.generate_image.assert_called_once()
    call = fake.generate_image.call_args
    assert call.args == ("a cat",)
    assert call.kwargs["model"] == "gpt-image-1"
    assert call.kwargs["size"] == "1024x1024"

    # Second call with the same (provider, base_url) reuses the cached provider.
    llm.generate_image("a dog", provider="openai", model="gpt-image-1")
    assert len(built) == 1


def test_llm_generate_image_local_passes_base_url():
    llm = LLM(log_dir=False)
    fake = MagicMock()
    fake.generate_image.return_value = "R"
    built: list[tuple[str, dict]] = []

    def fake_new_provider(name, **kw):
        built.append((name, kw))
        return fake

    llm.new_provider = fake_new_provider  # type: ignore[method-assign]

    llm.generate_image(
        "flux cat", provider="localimage", model="flux", base_url="http://127.0.0.1:1234/v1"
    )
    assert built == [("localimage", {"base_url": "http://127.0.0.1:1234/v1"})]
