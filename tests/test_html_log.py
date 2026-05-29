"""HTML sibling log: exists, escapes hostile content, mirrors clones, appends."""

from __future__ import annotations

from pathlib import Path

from llmfacade import SystemBlock, tool
from llmfacade.models import ThinkingBlock, ToolCall, Usage

from .conftest import MockProvider


def test_html_sibling_created(mock_model, tmp_path: Path):
    log = tmp_path / "convo.jsonl"
    convo = mock_model.new_conversation(log_path=log, system_blocks=["you are X"])
    convo.send("hi")

    html = log.with_suffix(".html")
    assert html.exists(), "expected sibling .html log next to .jsonl"
    text = html.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "<style>" in text
    # The convo and provider name are visible in the header.
    assert "mock" in text
    assert "mock-model" in text
    # The user message and assistant reply made it into the body.
    assert "hi" in text
    assert "ok" in text


def test_html_escapes_hostile_model_output(tmp_path: Path):
    """Model output containing <script> tags must be escaped as text, not
    embedded as live markup."""
    p = MockProvider(canned_text="<script>alert('xss')</script>")
    model = p.new_model("mock-model")
    log = tmp_path / "x.jsonl"
    convo = model.new_conversation(log_path=log)
    convo.send("ping")

    html = log.with_suffix(".html").read_text(encoding="utf-8")
    # The literal tag must not appear unescaped — escaping turns < into &lt;
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html


def test_html_escapes_hostile_user_input(mock_model, tmp_path: Path):
    log = tmp_path / "u.jsonl"
    convo = mock_model.new_conversation(log_path=log)
    convo.send("</section><script>boom()</script>")

    html = log.with_suffix(".html").read_text(encoding="utf-8")
    assert "<script>boom" not in html
    assert "&lt;/section&gt;" in html
    assert "&lt;script&gt;boom" in html


def test_html_appends_across_multiple_turns(mock_model, tmp_path: Path):
    log = tmp_path / "multi.jsonl"
    convo = mock_model.new_conversation(log_path=log)
    convo.send("first")
    size_after_one = log.with_suffix(".html").stat().st_size
    convo.send("second")
    size_after_two = log.with_suffix(".html").stat().st_size

    assert size_after_two > size_after_one
    text = log.with_suffix(".html").read_text(encoding="utf-8")
    assert text.count("first") >= 1
    assert text.count("second") >= 1
    # Both assistant turns rendered.
    assert text.count('class="msg msg-assistant"') == 2


def test_html_renders_tool_use_block(tmp_path: Path):
    """A model that returns a tool_use block should produce a styled tool-use
    panel in the HTML log (collapsed details for the result, visible name)."""
    call = ToolCall(id="call_1", name="add", input={"a": 1, "b": 2})
    p = MockProvider(canned_text="ok", canned_tool_calls=[call])
    model = p.new_model("mock-model")
    log = tmp_path / "tool.jsonl"

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    convo = model.new_conversation(log_path=log, tools=[add])
    convo.send("compute")

    html = log.with_suffix(".html").read_text(encoding="utf-8")
    assert 'class="tool-use"' in html
    assert ">add<" in html  # tool name rendered
    assert "id=call_1" in html


def test_html_renders_reasoning_block_and_estimated_badge(tmp_path: Path):
    """A ThinkingBlock with no provider-reported reasoning_tokens renders a
    collapsible reasoning block plus a usage badge with a locally-counted (~)
    estimate. This is the llama.cpp / Gemma 4 case."""
    p = MockProvider(
        canned_text="the answer",
        canned_thinking_blocks=[ThinkingBlock(text="a long stretch of reasoning text")],
    )
    model = p.new_model("mock-model")
    log = tmp_path / "reason.jsonl"
    convo = model.new_conversation(log_path=log)
    convo.send("q")

    html = log.with_suffix(".html").read_text(encoding="utf-8")
    # Collapsible, default-collapsed reasoning block (no `open` attribute).
    assert 'class="thinking-block"' in html
    assert "<summary>thinking" in html
    assert "a long stretch of reasoning text" in html
    # Estimated token badge sits in the usage line, prefixed with ~.
    assert "reasoning ~" in html


def test_html_reasoning_badge_uses_provider_reported_count(tmp_path: Path):
    """When the provider reports reasoning_tokens, the badge shows the exact
    count with no ~ prefix."""
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, reasoning_tokens=7)
    p = MockProvider(
        canned_text="answer",
        canned_thinking_blocks=[ThinkingBlock(text="reason")],
        canned_usage=usage,
    )
    model = p.new_model("mock-model")
    log = tmp_path / "reason2.jsonl"
    convo = model.new_conversation(log_path=log)
    convo.send("q")

    html = log.with_suffix(".html").read_text(encoding="utf-8")
    assert "reasoning 7" in html
    assert "reasoning ~" not in html


def test_html_no_reasoning_badge_without_reasoning(mock_model, tmp_path: Path):
    """A plain turn with no reasoning produces no reasoning badge."""
    log = tmp_path / "plain.jsonl"
    convo = mock_model.new_conversation(log_path=log)
    convo.send("q")
    html = log.with_suffix(".html").read_text(encoding="utf-8")
    assert "reasoning " not in html


def test_html_collapsibles_present(mock_model, tmp_path: Path):
    """Settings header, system blocks, and tools each get their own <details>
    summary so the file opens scannable."""
    log = tmp_path / "fold.jsonl"
    convo = mock_model.new_conversation(
        log_path=log,
        system_blocks=[SystemBlock(text="you are X", cache=True)],
    )
    convo.send("hello")

    html = log.with_suffix(".html").read_text(encoding="utf-8")
    assert "<summary>Settings</summary>" in html
    assert "<summary>System blocks (1)</summary>" in html


def test_html_clone_writes_independent_file(mock_model, tmp_path: Path):
    parent_log = tmp_path / "parent.jsonl"
    convo = mock_model.new_conversation(log_path=parent_log)
    convo.send("hi")

    clone_log = tmp_path / "clone.jsonl"
    clone = convo.clone(log_path=clone_log)
    clone.send("from-clone")

    parent_html = parent_log.with_suffix(".html").read_text(encoding="utf-8")
    clone_html = clone_log.with_suffix(".html").read_text(encoding="utf-8")

    # Clone's HTML carries the convo header but only the post-clone turn —
    # parent history is treated as already-logged at clone time.
    assert "from-clone" in clone_html
    assert "hi" in parent_html
    assert "from-clone" not in parent_html


def test_html_skipped_when_log_path_is_html(tmp_path: Path):
    """If the JSONL path itself ends in .html, skip HTML logging — otherwise
    we'd clobber the JSONL with HTML writes."""
    p = MockProvider()
    model = p.new_model("mock-model")
    log = tmp_path / "weird.html"
    convo = model.new_conversation(log_path=log)
    convo.send("hi")

    # The .html file should still hold the JSONL contents (one record per line),
    # not HTML markup. Settings header is the first JSONL record.
    text = log.read_text(encoding="utf-8")
    assert text.startswith("{")
    assert "<!DOCTYPE html>" not in text
