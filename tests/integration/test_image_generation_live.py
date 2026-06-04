"""Live image generation. Gated behind ``-m integration`` and individually
skipped when the relevant key / server is absent. Costs money for the hosted
providers; the local one needs a running OpenAI-compatible image server.

Run explicitly, e.g.:
    pytest -m integration tests/integration/test_image_generation_live.py
"""

from __future__ import annotations

import os

import pytest

from llmfacade import LLM, ImageBlock

pytestmark = pytest.mark.integration


@pytest.mark.usefixtures("openai_api_key")
def test_openai_image_generation() -> None:
    llm = LLM()
    result = llm.generate_image(
        "a single red maple leaf on a white background",
        provider="openai",
        model="gpt-image-1",
        size="1024x1024",
    )
    assert result.images and result.images[0].data
    assert result.images[0].media_type.startswith("image/")


@pytest.mark.usefixtures("google_api_key")
def test_gemini_native_reference_edit() -> None:
    llm = LLM()
    # First generate a base image, then condition a second generation on it.
    base = llm.generate_image(
        "a plain blue circle on a white background",
        provider="gemini",
        model="gemini-2.5-flash-image",
    )
    assert base.images
    ref = ImageBlock(data=base.images[0].data, media_type=base.images[0].media_type)
    edited = llm.generate_image(
        "the same circle, now with a thick yellow border",
        provider="gemini",
        model="gemini-2.5-flash-image",
        reference_images=[ref],
    )
    assert edited.images and edited.images[0].data


def test_localimage_generation() -> None:
    base_url = os.getenv("LOCALIMAGE_BASE_URL")
    if not base_url:
        pytest.skip("LOCALIMAGE_BASE_URL not set; skipping local image server test")
    llm = LLM()
    result = llm.generate_image(
        "a fox in the snow",
        provider="localimage",
        model=os.getenv("LOCALIMAGE_MODEL", "default"),
        base_url=base_url,
        size="512x512",
    )
    assert result.images and result.images[0].data
