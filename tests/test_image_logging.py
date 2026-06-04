"""Audit logging for image generation.

A fake image provider (declares ``image_generation`` and returns a canned
``ImageResult`` from ``_generate_image_raw``) runs under a real ``LLM`` manager,
so these exercise the template-method logging in ``Provider.generate_image`` /
``agenerate_image`` without any network. The shared ledger lands in the
manager's run dir as ``images.jsonl`` + ``images.html``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from llmfacade import LLM, ImageBlock, ImageResult, ImageUsage
from llmfacade.models import _apply_save_dir
from llmfacade.provider import Provider


class FakeImageProvider(Provider):
    NAME = "fakeimage"
    SUPPORTS = frozenset({"image_generation"})

    def _init_client(self) -> None:
        self._client = None

    def _result(self, model: str | None) -> ImageResult:
        return ImageResult(
            images=[ImageBlock(data=b"PNGDATA", media_type="image/png")],
            usage=ImageUsage(input_tokens=12, output_tokens=34, total_tokens=46, image_count=1),
            model=model or "fake-image-1",
            provider=self.NAME,
        )

    def _generate_image_raw(self, prompt, *, model=None, save_dir=None, **kw) -> ImageResult:
        return _apply_save_dir(self._result(model), save_dir)

    async def _agenerate_image_raw(
        self, prompt, *, model=None, save_dir=None, **kw
    ) -> ImageResult:
        return _apply_save_dir(self._result(model), save_dir)


def _provider(tmp_path, **mgr_kwargs) -> FakeImageProvider:
    llm = LLM(log_dir=tmp_path, **mgr_kwargs)
    return FakeImageProvider(manager=llm)


def _ledger(tmp_path):
    runs = list(tmp_path.glob("llmfacade*"))
    assert len(runs) == 1, runs
    return runs[0] / "images.jsonl"


def _records(tmp_path) -> list[dict]:
    jsonl = _ledger(tmp_path)
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]


def test_image_log_writes_jsonl_record(tmp_path):
    provider = _provider(tmp_path)

    provider.generate_image("a cat", model="fake-image-1", size="1024x1024", n=2)

    records = _records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["provider"] == "fakeimage"
    assert rec["model"] == "fake-image-1"
    assert rec["prompt"] == "a cat"
    assert rec["n"] == 2
    assert rec["size"] == "1024x1024"
    assert rec["usage"] == {
        "input_tokens": 12,
        "output_tokens": 34,
        "total_tokens": 46,
        "image_count": 1,
    }
    assert rec["image_count"] == 1
    assert rec["images"] == [{"media_type": "image/png", "bytes": len(b"PNGDATA")}]
    assert rec["paths"] == []


def test_image_log_appends_per_call(tmp_path):
    provider = _provider(tmp_path)

    provider.generate_image("one", model="fake-image-1")
    provider.generate_image("two", model="fake-image-1")

    records = _records(tmp_path)
    assert [r["prompt"] for r in records] == ["one", "two"]


def test_image_log_writes_html_sibling(tmp_path):
    provider = _provider(tmp_path)

    provider.generate_image("a dragon", model="fake-image-1")
    provider.generate_image("a fox", model="fake-image-1")

    html_path = _ledger(tmp_path).with_suffix(".html")
    html = html_path.read_text(encoding="utf-8")
    assert html.count("<!DOCTYPE html>") == 1  # header written once
    assert html.count('<section class="gen">') == 2
    assert "a dragon" in html and "a fox" in html
    # The CSS is emitted verbatim (not str.format'd), so it must use single
    # braces — doubled braces would render as literal text and break styling.
    assert "{{" not in html and "}}" not in html


def test_image_log_records_save_dir_paths(tmp_path):
    provider = _provider(tmp_path)
    save_dir = tmp_path / "out"

    result = provider.generate_image("a cat", model="fake-image-1", save_dir=save_dir)

    assert len(result.paths) == 1
    rec = _records(tmp_path)[0]
    assert rec["paths"] == [str(result.paths[0])]


def test_image_log_reference_image_count(tmp_path):
    provider = _provider(tmp_path)

    provider.generate_image(
        "edit",
        model="fake-image-1",
        reference_images=[
            ImageBlock(data=b"REF1", media_type="image/png"),
            ImageBlock(data=b"REF2", media_type="image/png"),
        ],
    )

    rec = _records(tmp_path)[0]
    assert rec["reference_images"] == 2  # a count, not the bytes


def test_image_log_async(tmp_path):
    provider = _provider(tmp_path)

    asyncio.run(provider.agenerate_image("async cat", model="fake-image-1"))

    rec = _records(tmp_path)[0]
    assert rec["prompt"] == "async cat"


def test_image_log_disabled_when_log_dir_false(tmp_path):
    llm = LLM(log_dir=False)
    provider = FakeImageProvider(manager=llm)

    result = provider.generate_image("a cat", model="fake-image-1")

    assert result.images[0].data == b"PNGDATA"  # still returned
    assert not list(tmp_path.glob("llmfacade*"))


def test_image_log_disabled_when_provider_log_dir_false(tmp_path):
    llm = LLM(log_dir=tmp_path)
    provider = FakeImageProvider(manager=llm, log_dir=False)

    result = provider.generate_image("a cat", model="fake-image-1")

    assert result.images[0].data == b"PNGDATA"  # still returned
    # Provider opted out — no ledger written anywhere under the log root.
    assert not list(tmp_path.rglob("images.jsonl"))


def test_image_log_write_failure_is_swallowed(tmp_path, monkeypatch):
    """A failure inside the best-effort writer warns but never breaks the call —
    a paid-for image must always be returned."""
    provider = _provider(tmp_path)

    import llmfacade._image_log as image_log

    def boom(html_path, record):
        raise OSError("disk full")

    monkeypatch.setattr(image_log, "_append_html", boom)

    with pytest.warns(UserWarning, match="image-generation logging failed"):
        result = provider.generate_image("a cat", model="fake-image-1")

    assert result.images[0].data == b"PNGDATA"
