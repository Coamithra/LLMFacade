"""Sibling HTML log for human-readable inspection.

When ``log_path=foo.jsonl`` is set on a Conversation, this writes ``foo.html``
alongside it, with every turn rendered as a boxed section the browser can open
without any tooling. The JSONL stays the source of truth for programmatic use
(replay, analysis, diffs); the HTML is a derivative artifact for humans.

Append-only writes; HTML5 lets us omit ``</body></html>`` so we never need to
seek-and-rewrite — the browser renders partial files gracefully. Model output
is ``html.escape``'d before insertion, so the format-collision worry that
motivates this format (markdown-in-markdown, html-in-html) dissolves: model
text is *data inside* a structural container, not sibling markup that can
break out.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmfacade.helpers import _abbreviate_text
from llmfacade.models import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# Doubled braces are escaped for str.format(); single braces are placeholders.
_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{
  --bg: #fafafa;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --user: #2a6df4;
  --assistant: #1a8870;
  --tool: #b35f1f;
  --thinking: #888;
  --border: #e0e0e0;
  --code-bg: #f4f4f4;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0 auto;
  max-width: 940px;
  padding: 1.2rem;
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: var(--fg);
  background: var(--bg);
}}
h1 {{ font-size: 1.3rem; margin: 0 0 .25rem; }}
.subtitle {{ color: var(--muted); margin: 0 0 1rem; font-size: .9rem; }}
.subtitle code {{ background: var(--code-bg); padding: 0 .25rem; border-radius: 3px; }}
details {{
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: .25rem .6rem;
  margin: .25rem 0;
  background: white;
}}
details > summary {{
  cursor: pointer;
  user-select: none;
  font-weight: 500;
  color: var(--muted);
}}
details > summary:hover {{ color: var(--fg); }}
details[open] > summary {{ margin-bottom: .35rem; }}
section.msg {{
  border: 1px solid var(--border);
  border-left-width: 4px;
  border-radius: 6px;
  margin: .65rem 0;
  padding: .55rem .85rem;
  background: white;
}}
section.msg-user      {{ border-left-color: var(--user); }}
section.msg-assistant {{ border-left-color: var(--assistant); }}
section.msg-tool      {{ border-left-color: var(--tool); }}
section.msg > header {{
  font-size: .8rem;
  color: var(--muted);
  margin-bottom: .35rem;
  display: flex;
  gap: .55rem;
  align-items: center;
  flex-wrap: wrap;
}}
.role-badge {{
  display: inline-block;
  padding: .05rem .45rem;
  border-radius: 3px;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: white;
}}
.role-user      {{ background: var(--user); }}
.role-assistant {{ background: var(--assistant); }}
.role-tool      {{ background: var(--tool); }}
.msg-text {{
  white-space: pre-wrap;
  word-wrap: break-word;
  margin: .25rem 0;
}}
.tool-use {{
  border-left: 3px solid var(--tool);
  padding: .3rem .6rem;
  margin: .35rem 0;
  background: #fff8f0;
  border-radius: 0 4px 4px 0;
}}
.tool-use .name {{
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-weight: 600;
  color: var(--tool);
}}
.tool-use small {{ color: var(--muted); }}
pre {{
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: .82rem;
  background: var(--code-bg);
  padding: .5rem .65rem;
  border-radius: 4px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  margin: .25rem 0;
}}
dl.kv {{
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: .15rem .85rem;
  font-size: .85rem;
  margin: .25rem 0;
}}
dl.kv dt {{ color: var(--muted); font-family: ui-monospace, monospace; }}
dl.kv dt small {{ color: #aaa; font-weight: normal; }}
dl.kv dd {{ margin: 0; font-family: ui-monospace, monospace; word-break: break-word; }}
.usage-line {{ font-size: .8rem; color: var(--muted); margin: .4rem 0 .1rem; }}
.badge {{
  display: inline-block;
  background: var(--code-bg);
  padding: .05rem .45rem;
  border-radius: 3px;
  margin-right: .35rem;
  font-family: ui-monospace, monospace;
  font-size: .78rem;
}}
.thinking-block {{
  border-left: 3px solid var(--thinking);
  background: #f8f8f8;
  color: #555;
}}
.thinking-block summary {{ font-style: italic; }}
.error {{ color: #c0392b; }}
hr {{ border: 0; border-top: 1px dashed var(--border); margin: 1rem 0; }}
</style>
<h1>{header}</h1>
<p class="subtitle">{subtitle}</p>
"""


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _repr_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, str)):
        return str(v)
    try:
        return json.dumps(v, default=str)
    except Exception:
        return repr(v)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_block(block: ContentBlock, *, max_lines: int | None = None) -> str:
    if isinstance(block, TextBlock):
        return f'<div class="msg-text">{_escape(_abbreviate_text(block.text, max_lines))}</div>\n'
    if isinstance(block, ImageBlock):
        return (
            f'<div class="msg-text"><em>[image: {_escape(block.media_type)}, '
            f"{len(block.data)} bytes]</em></div>\n"
        )
    if isinstance(block, ToolUseBlock):
        try:
            args = json.dumps(block.input, indent=2, default=str)
        except Exception:
            args = repr(block.input)
        raw_html = ""
        if block.raw_arguments is not None:
            raw_html = (
                '  <div class="tool-raw"><em>unparsed arguments '
                "(tool call failed to parse — likely truncated):</em>\n"
                f"  <pre>{_escape(_abbreviate_text(block.raw_arguments, max_lines))}</pre></div>\n"
            )
        return (
            '<div class="tool-use">\n'
            f'  <span class="name">{_escape(block.name)}</span>'
            f"  <small>id={_escape(block.id)}</small>\n"
            f"  <pre>{_escape(_abbreviate_text(args, max_lines))}</pre>\n"
            f"{raw_html}"
            "</div>\n"
        )
    if isinstance(block, ToolResultBlock):
        if isinstance(block.content, str):
            body = _escape(_abbreviate_text(block.content, max_lines))
        else:
            parts: list[str] = []
            for b in block.content:
                if isinstance(b, TextBlock):
                    parts.append(b.text)
                elif isinstance(b, ImageBlock):
                    parts.append(f"[image {b.media_type}, {len(b.data)} bytes]")
            body = _escape(_abbreviate_text("\n".join(parts), max_lines))
        err_cls = ' class="error"' if block.is_error else ""
        name = f" ({_escape(block.name)})" if block.name else ""
        return (
            f"<details><summary{err_cls}>tool_result{name}"
            f" <small>id={_escape(block.tool_use_id)}</small></summary>\n"
            f"<pre>{body}</pre></details>\n"
        )
    # ContentBlock is a closed Union; ThinkingBlock is the last branch.
    assert isinstance(block, ThinkingBlock)
    tag = "redacted_thinking" if block.encrypted else "thinking"
    body = _escape(_abbreviate_text(block.text, max_lines)) if block.text else "<em>(opaque)</em>"
    return (
        f'<details class="thinking-block"><summary>{tag}'
        f" ({len(block.text)} chars)</summary>\n"
        f"<pre>{body}</pre></details>\n"
    )


def _render_message_body(msg: Message, *, max_lines: int | None = None) -> str:
    if isinstance(msg.content, str):
        return f'<div class="msg-text">{_escape(_abbreviate_text(msg.content, max_lines))}</div>\n'
    return "".join(_render_block(b, max_lines=max_lines) for b in msg.content)


class HtmlLogger:
    """Append-only HTML sibling for the JSONL log.

    The settings header is written once at construction time. Each
    request appends a section per new history message (and a collapsed
    per-call settings block); each response appends one assistant section
    with collapsibles for thinking, tool use, usage, and cache summary.
    """

    def __init__(self, path: Path, *, max_lines: int | None = None):
        self.path = path
        self.max_lines = max_lines

    def write_header(
        self,
        *,
        convo_name: str,
        provider: str,
        model_id: str,
        system_blocks: list[Any],
        tools: list[str],
        settings: dict[str, dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ) -> None:
        title = f"{convo_name} · {provider}:{model_id}"
        started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subtitle = (
            f"<code>{_escape(provider)}</code> · "
            f"<code>{_escape(model_id)}</code> · "
            f"{len(system_blocks)} system block(s) · "
            f"{len(tools)} tool(s) · "
            f"started {started}"
        )
        out: list[str] = [
            _HEAD.format(
                title=_escape(title),
                header=_escape(convo_name),
                subtitle=subtitle,
            )
        ]

        out.append('<details><summary>Settings</summary>\n<dl class="kv">\n')
        for k in sorted(settings):
            info = settings[k]
            if isinstance(info, dict):
                v = info.get("value")
                src = info.get("source", "")
            else:
                v, src = info, ""
            src_html = f" <small>({_escape(src)})</small>" if src else ""
            out.append(f"  <dt>{_escape(k)}{src_html}</dt><dd>{_escape(_repr_value(v))}</dd>\n")
        out.append("</dl></details>\n")

        if extra:
            for top_key, payload in extra.items():
                summary = "Fit estimate" if top_key == "fit_estimate" else top_key
                out.append(
                    f'<details open><summary>{_escape(summary)}</summary>\n<dl class="kv">\n'
                )
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        out.append(
                            f"  <dt>{_escape(str(k))}</dt><dd>{_escape(_repr_value(v))}</dd>\n"
                        )
                else:
                    out.append(f"  <dt>value</dt><dd>{_escape(_repr_value(payload))}</dd>\n")
                out.append("</dl></details>\n")

        if system_blocks:
            out.append(f"<details><summary>System blocks ({len(system_blocks)})</summary>\n")
            for sb in system_blocks:
                cache = " <small>(cache=True)</small>" if getattr(sb, "cache", False) else ""
                out.append(f"<pre>{_escape(sb.text)}</pre>{cache}\n")
            out.append("</details>\n")

        if tools:
            out.append(f"<details><summary>Tools ({len(tools)})</summary><ul>\n")
            for t in tools:
                out.append(f"<li><code>{_escape(t)}</code></li>\n")
            out.append("</ul></details>\n")

        out.append("<hr>\n")
        self.path.write_text("".join(out), encoding="utf-8")

    def write_request(
        self,
        *,
        new_messages: list[Message],
        overrides: dict[str, Any],
        tool_choice: str | None,
        stop: list[str] | None,
    ) -> None:
        out: list[str] = []
        for msg in new_messages:
            role = msg.role
            role_class = f"msg-{role}" if role in {"user", "assistant", "tool"} else "msg-user"
            badge_class = f"role-{role}" if role in {"user", "assistant", "tool"} else "role-user"
            out.append(
                f'<section class="msg {role_class}">\n'
                f'  <header><span class="role-badge {badge_class}">'
                f"{_escape(role)}</span></header>\n"
            )
            out.append(_render_message_body(msg, max_lines=self.max_lines))
            out.append("</section>\n")

        notable = {k: v for k, v in overrides.items() if v is not None}
        forced_choice = tool_choice and tool_choice != "auto"
        if notable or forced_choice or stop:
            out.append('<details><summary>Per-call request settings</summary>\n<dl class="kv">\n')
            if forced_choice:
                out.append(f"  <dt>tool_choice</dt><dd>{_escape(_repr_value(tool_choice))}</dd>\n")
            if stop:
                out.append(f"  <dt>stop</dt><dd>{_escape(_repr_value(stop))}</dd>\n")
            for k in sorted(notable):
                out.append(f"  <dt>{_escape(k)}</dt><dd>{_escape(_repr_value(notable[k]))}</dd>\n")
            out.append("</dl></details>\n")

        if out:
            self._append("".join(out))

    def write_response(
        self,
        *,
        blocks: list[ContentBlock],
        text: str,
        usage: dict[str, int] | None,
        cache_summary: dict[str, Any] | None,
        reasoning: dict[str, Any] | None = None,
        finish_reason: str | None,
        model_id: str,
    ) -> None:
        out: list[str] = []
        finish_html = f"<small>finish: {_escape(finish_reason)}</small>" if finish_reason else ""
        out.append(
            '<section class="msg msg-assistant">\n'
            "  <header>"
            '<span class="role-badge role-assistant">assistant</span>'
            f"<small>{_escape(model_id)}</small>"
            f"{finish_html}"
            "</header>\n"
        )
        # Prefer rendering the structured blocks (they preserve thinking and
        # tool_use ordering). Fall back to plain text if blocks is empty.
        if blocks:
            for b in blocks:
                out.append(_render_block(b, max_lines=self.max_lines))
        elif text:
            body = _escape(_abbreviate_text(text, self.max_lines))
            out.append(f'<div class="msg-text">{body}</div>\n')

        badges: list[str] = []
        if usage:
            for key, label in (
                ("prompt_tokens", "in"),
                ("completion_tokens", "out"),
                ("cache_read_tokens", "cache_read"),
                ("cache_creation_tokens", "cache_create"),
            ):
                v = usage.get(key) or 0
                if v:
                    badges.append(f'<span class="badge">{label} {v}</span>')
        reasoning_badge = self._reasoning_badge(reasoning)
        if reasoning_badge:
            badges.append(reasoning_badge)
        if badges:
            out.append(f'<div class="usage-line">{"".join(badges)}</div>\n')

        if cache_summary:
            note = cache_summary.get("_note", "")
            note_html = f" — {_escape(_truncate(note, 80))}" if note else ""
            out.append(f'<details><summary>cache summary{note_html}</summary>\n<dl class="kv">\n')
            for k, v in cache_summary.items():
                out.append(f"  <dt>{_escape(k)}</dt><dd>{_escape(_repr_value(v))}</dd>\n")
            out.append("</dl></details>\n")

        out.append("</section>\n")
        self._append("".join(out))

    @staticmethod
    def _reasoning_badge(reasoning: dict[str, Any] | None) -> str:
        """Render the reasoning-token count as a usage badge, sitting next to
        ``out``. ``~`` prefixes a locally-counted estimate (the provider didn't
        report a breakdown); a ``title`` tooltip names the source tokenizer."""
        if not reasoning or reasoning.get("reasoning_tokens") is None:
            return ""
        n = reasoning["reasoning_tokens"]
        estimated = reasoning.get("estimated")
        source = reasoning.get("source", "")
        prefix = "~" if estimated else ""
        title = f"counted locally via {source}" if estimated else "reported by provider usage"
        return f'<span class="badge" title="{_escape(str(title))}">reasoning {prefix}{n}</span>'

    def _append(self, content: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(content)
