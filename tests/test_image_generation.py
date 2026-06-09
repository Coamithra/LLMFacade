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
    LabeledImage,
)
from llmfacade.exceptions import ProviderError, UnsupportedFeature
from llmfacade.models import normalize_reference_images
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


def test_localimage_no_base_url_is_managed():
    # Omitting base_url selects managed mode (it used to raise); a supervisor is
    # created and external-only behaviour no longer applies.
    provider = LocalImageProvider()
    assert provider._managed is True
    assert provider._supervisor is not None


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


# ---- localimage managed mode ----------------------------------------------


def _managed_provider_with_model(tmp_path, name="flux", **launch):
    """Build a managed LocalImageProvider with one registered model whose
    diffusion_model file actually exists (existence is checked at registration)."""
    model_file = tmp_path / f"{name}.safetensors"
    model_file.write_bytes(b"weights")
    provider = LocalImageProvider(llmfacade_dir=tmp_path / ".llmfacade")
    provider.new_image_model(diffusion_model=str(model_file), name=name, **launch)
    return provider


def test_localimage_external_provider_rejects_launch_knobs():
    with pytest.raises(UnsupportedFeature, match="managed mode"):
        LocalImageProvider(base_url="http://127.0.0.1:1234/v1", diffusion_fa=True)


def test_localimage_external_new_image_model_rejects_launch_knobs():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    with pytest.raises(UnsupportedFeature, match="managed mode"):
        provider.new_image_model("flux", diffusion_model="/x/flux.safetensors")


def test_localimage_external_new_image_model_rejects_name():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    with pytest.raises(UnsupportedFeature, match="managed-mode kwarg"):
        provider.new_image_model("flux", name="y")


def test_localimage_managed_new_image_model_registers(tmp_path):
    provider = _managed_provider_with_model(tmp_path, name="flux")
    entries = provider._supervisor.entries
    assert len(entries) == 1
    assert entries[0].model_id == "flux"
    assert entries[0].diffusion_model.endswith("flux.safetensors")


def test_localimage_managed_requires_model_source(tmp_path):
    provider = LocalImageProvider(llmfacade_dir=tmp_path / ".llmfacade")
    with pytest.raises(ValueError, match="requires model"):
        provider.new_image_model(name="x")


def test_localimage_managed_missing_file_raises(tmp_path):
    provider = LocalImageProvider(llmfacade_dir=tmp_path / ".llmfacade")
    with pytest.raises(FileNotFoundError, match="diffusion_model"):
        provider.new_image_model(diffusion_model=str(tmp_path / "missing.safetensors"))


def test_localimage_managed_generate_routes_through_supervisor(tmp_path, monkeypatch):
    provider = _managed_provider_with_model(tmp_path, name="flux")

    calls: list[str] = []
    monkeypatch.setattr(
        provider._supervisor,
        "ensure_model",
        lambda mid: calls.append(mid) or "http://127.0.0.1:9000",
    )
    # Pre-seed a stub client at the URL the supervisor will return so no rebuild
    # (and no real openai client / network) happens.
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"FLUX")
    provider._client_base = "http://127.0.0.1:9000/v1"

    result = provider.generate_image("a fox", model="flux")

    assert calls == ["flux"]
    assert result.images[0].data == b"FLUX"
    assert result.model == "flux"


def test_localimage_managed_infers_single_model(tmp_path, monkeypatch):
    provider = _managed_provider_with_model(tmp_path, name="flux")
    calls: list[str] = []
    monkeypatch.setattr(
        provider._supervisor,
        "ensure_model",
        lambda mid: calls.append(mid) or "http://127.0.0.1:9000",
    )
    provider._client = MagicMock()
    provider._client.images.generate.return_value = _openai_image_response(b"X")
    provider._client_base = "http://127.0.0.1:9000/v1"

    provider.generate_image("x")  # no model= → inferred from the single registration

    assert calls == ["flux"]


def test_localimage_managed_multi_model_requires_model(tmp_path):
    provider = _managed_provider_with_model(tmp_path, name="flux")
    second = tmp_path / "sdxl.safetensors"
    second.write_bytes(b"w")
    provider.new_image_model(diffusion_model=str(second), name="sdxl")

    with pytest.raises(ValueError, match="multi-model provider requires"):
        provider.generate_image("x")


def test_localimage_managed_rebuilds_client_on_new_port(tmp_path, monkeypatch):
    provider = _managed_provider_with_model(tmp_path, name="flux")
    monkeypatch.setattr(provider._supervisor, "ensure_model", lambda mid: "http://127.0.0.1:9000")

    built: list[str] = []

    def fake_build(openai_base):
        built.append(openai_base)
        provider._client = MagicMock()
        provider._client.images.generate.return_value = _openai_image_response(b"X")
        provider._client_base = openai_base

    monkeypatch.setattr(provider, "_build_image_clients", fake_build)
    provider.generate_image("x", model="flux")

    assert built == ["http://127.0.0.1:9000/v1"]


def test_localimage_shutdown_calls_supervisor(tmp_path, monkeypatch):
    provider = _managed_provider_with_model(tmp_path, name="flux")
    called: list[bool] = []
    monkeypatch.setattr(provider._supervisor, "shutdown", lambda: called.append(True))
    provider.shutdown()
    assert called == [True]


def test_localimage_external_shutdown_is_noop():
    provider = LocalImageProvider(base_url="http://127.0.0.1:1234/v1")
    provider.shutdown()  # must not raise (no supervisor in external mode)


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


def test_llm_generate_image_distinct_api_keys_build_distinct_providers():
    llm = LLM(log_dir=False)
    built: list[dict] = []

    def fake_new_provider(name, **kw):
        built.append(kw)
        fake = MagicMock()
        fake.generate_image.return_value = f"R{len(built)}"
        return fake

    llm.new_provider = fake_new_provider  # type: ignore[method-assign]

    out1 = llm.generate_image("a cat", provider="openai", model="gpt-image-1", api_key="key-one")
    out2 = llm.generate_image("a cat", provider="openai", model="gpt-image-1", api_key="key-two")
    assert (out1, out2) == ("R1", "R2")
    assert [kw["api_key"] for kw in built] == ["key-one", "key-two"]

    # Repeating either key reuses its cached provider — no third build.
    out3 = llm.generate_image("a dog", provider="openai", model="gpt-image-1", api_key="key-one")
    assert out3 == "R1"
    assert len(built) == 2

    # The raw secret is never a member of the cache-key tuples (digest only).
    for key in llm._image_providers:
        assert "key-one" not in key
        assert "key-two" not in key


def test_llm_generate_image_explicit_key_not_shadowed_by_default_key_provider():
    llm = LLM(api_keys={"openai": "env-key"}, log_dir=False)
    built: list[dict] = []

    def fake_new_provider(name, **kw):
        built.append(kw)
        fake = MagicMock()
        fake.generate_image.return_value = "R"
        return fake

    llm.new_provider = fake_new_provider  # type: ignore[method-assign]

    llm.generate_image("x", provider="openai", model="gpt-image-1")  # manager/env key
    llm.generate_image("x", provider="openai", model="gpt-image-1", api_key="explicit")
    assert len(built) == 2
    assert "api_key" not in built[0]
    assert built[1]["api_key"] == "explicit"


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


# ---- Labeled / interleaved reference images --------------------------------


def test_normalize_reference_images_coerces_each_form():
    a = ImageBlock(data=b"A", media_type="image/png")
    b = ImageBlock(data=b"B", media_type="image/jpeg")
    c = ImageBlock(data=b"C", media_type="image/png")

    pairs = normalize_reference_images([a, LabeledImage("Bert", b), ("Cara", c)])

    assert pairs == [(None, a), ("Bert", b), ("Cara", c)]


def test_normalize_reference_images_none_and_empty():
    assert normalize_reference_images(None) == []
    assert normalize_reference_images([]) == []


def test_normalize_reference_images_rejects_bad_item():
    with pytest.raises(TypeError, match="ImageBlock, LabeledImage"):
        normalize_reference_images(["not an image"])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        # a (label, block) tuple with a non-str label is not accepted
        normalize_reference_images([(1, ImageBlock(data=b"X", media_type="image/png"))])  # type: ignore[list-item]


def test_google_labeled_references_interleaved():
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(b"OUT")

    adam = ImageBlock(data=b"ADAM", media_type="image/png")
    bert = ImageBlock(data=b"BERT", media_type="image/png")
    provider.generate_image(
        "draw Adam waving at Bert",
        reference_images=[LabeledImage("This is Adam:", adam), ("This is Bert:", bert)],
    )

    parts = provider._client.models.generate_content.call_args.kwargs["contents"][0]["parts"]
    # Each label text part precedes its image; the prompt is the final part.
    assert parts[0] == {"text": "This is Adam:"}
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"ADAM"
    assert parts[2] == {"text": "This is Bert:"}
    assert base64.b64decode(parts[3]["inline_data"]["data"]) == b"BERT"
    assert parts[4] == {"text": "draw Adam waving at Bert"}


def test_google_mixed_labeled_unlabeled():
    """An unlabeled item in a labeled list emits its image with no preceding
    text part, keeping its position."""
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(b"OUT")

    adam = ImageBlock(data=b"ADAM", media_type="image/png")
    anon = ImageBlock(data=b"ANON", media_type="image/png")
    cara = ImageBlock(data=b"CARA", media_type="image/png")
    provider.generate_image(
        "scene",
        reference_images=[LabeledImage("Adam", adam), anon, ("Cara", cara)],
    )

    parts = provider._client.models.generate_content.call_args.kwargs["contents"][0]["parts"]
    assert parts[0] == {"text": "Adam"}
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"ADAM"
    assert base64.b64decode(parts[2]["inline_data"]["data"]) == b"ANON"  # no label text
    assert parts[3] == {"text": "Cara"}
    assert base64.b64decode(parts[4]["inline_data"]["data"]) == b"CARA"
    assert parts[5] == {"text": "scene"}


def test_google_empty_label_treated_as_unlabeled():
    """An empty-string label carries no identity, so it is dropped like an
    unlabeled reference (no text part)."""
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(b"OUT")

    provider.generate_image(
        "pose",
        reference_images=[LabeledImage("", ImageBlock(data=b"REF", media_type="image/png"))],
    )

    parts = provider._client.models.generate_content.call_args.kwargs["contents"][0]["parts"]
    # No-label path == unlabeled bag: prompt first, then the image, no "" text part.
    assert parts[0] == {"text": "pose"}
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"REF"
    assert len(parts) == 2


def test_google_unlabeled_references_unchanged():
    """Back-compat: an unlabeled bag keeps the prompt-first shape."""
    provider = GoogleProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.models.generate_content.return_value = _gemini_image_response(b"OUT")

    provider.generate_image(
        "same character, new pose",
        reference_images=[ImageBlock(data=b"REF", media_type="image/png")],
    )

    parts = provider._client.models.generate_content.call_args.kwargs["contents"][0]["parts"]
    assert parts[0] == {"text": "same character, new pose"}
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"REF"


def test_openai_labeled_references_preamble():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.edit.return_value = _openai_image_response(b"EDITED")

    adam = ImageBlock(data=b"ADAM", media_type="image/png")
    bert = ImageBlock(data=b"BERT", media_type="image/png")
    provider.generate_image(
        "draw Adam waving at Bert",
        model="gpt-image-1",
        reference_images=[("Adam", adam), LabeledImage("Bert", bert)],
    )

    kwargs = provider._client.images.edit.call_args.kwargs
    # Edit endpoint cannot interleave: labels degrade to an order-binding preamble.
    assert kwargs["prompt"] == (
        "Reference image 1 is Adam. Reference image 2 is Bert.\n\ndraw Adam waving at Bert"
    )
    # Images still upload in list order.
    assert [content for _, content, _ in kwargs["image"]] == [b"ADAM", b"BERT"]


def test_openai_mixed_labeled_unlabeled_preamble():
    """The preamble numbers over ALL images (1-based) but only names labeled
    ones, preserving the order->identity map."""
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.edit.return_value = _openai_image_response(b"E")

    anon = ImageBlock(data=b"ANON", media_type="image/png")
    bert = ImageBlock(data=b"BERT", media_type="image/png")
    provider.generate_image(
        "draw them",
        model="gpt-image-1",
        reference_images=[anon, LabeledImage("Bert", bert)],
    )

    kwargs = provider._client.images.edit.call_args.kwargs
    assert kwargs["prompt"] == "Reference image 2 is Bert.\n\ndraw them"
    assert [content for _, content, _ in kwargs["image"]] == [b"ANON", b"BERT"]


def test_openai_tuple_shorthand_accepted():
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.images.edit.return_value = _openai_image_response(b"E")

    provider.generate_image(
        "x",
        model="gpt-image-1",
        reference_images=[("Solo", ImageBlock(data=b"REF", media_type="image/png"))],
    )

    kwargs = provider._client.images.edit.call_args.kwargs
    assert kwargs["prompt"].startswith("Reference image 1 is Solo.")
    assert kwargs["image"][0][1] == b"REF"
