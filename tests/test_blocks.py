"""Content block construction and round-trip behaviors."""

from __future__ import annotations

from llmfacade.models import (
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_text_block_immutable():
    b = TextBlock("hi")
    assert b.text == "hi"


def test_image_from_base64_roundtrip():
    raw = b"\x89PNG fake bytes"
    import base64

    b64 = base64.b64encode(raw).decode("ascii")
    img = ImageBlock.from_base64(b64, "image/png")
    assert img.data == raw
    assert img.media_type == "image/png"
    assert img.to_base64() == b64


def test_message_with_block_list():
    msg = Message(
        role="user",
        content=[TextBlock("look:"), ImageBlock(data=b"x", media_type="image/jpeg")],
    )
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2


def test_tool_use_block():
    b = ToolUseBlock(id="1", name="add", input={"a": 1})
    assert b.input["a"] == 1


def test_tool_result_block_with_text_only():
    r = ToolResultBlock(tool_use_id="1", content="3")
    assert r.content == "3"
    assert not r.is_error


def test_image_from_path(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"PNG-bytes")
    img = ImageBlock.from_path(p)
    assert img.data == b"PNG-bytes"
    assert img.media_type == "image/png"
