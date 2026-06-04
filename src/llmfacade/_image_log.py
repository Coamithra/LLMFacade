"""Audit ledger for image generation.

Image generation is a one-shot, not a stateful conversation, so it does not fit
the conversation JSONL/HTML turn format. Instead every successful
``provider.generate_image`` / ``agenerate_image`` call appends one record to a
shared ledger in the manager's run directory:

- ``<run_dir>/images.jsonl`` — one JSON object per generation (the source of
  truth for programmatic audit).
- ``<run_dir>/images.html`` — a human-readable sibling; the header is written
  once on the first record, then each call appends a ``<section>``.

The motivating concern is *spend*: hosted image generation (``gpt-image-1``,
Gemini-native) costs real money, and without this a caller has no record of
prompt, model, count, token usage, or where images were written.

Writes are **best-effort**: a failure to log must never prevent returning the
(paid-for) image to the caller, so :func:`log_image_generation` swallows and
warns rather than propagating. Image bytes are never written to the log — only
each image's media type, byte size, and any saved path.
"""

from __future__ import annotations

import html
import json
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfacade.models import ImageBlock, ImageResult
    from llmfacade.provider import Provider

_LEDGER_NAME = "images.jsonl"

# The ledger is shared across every image call in a run dir, so concurrent
# generations from multiple threads can race on the (multi-write) HTML append.
# A process-wide lock keeps each record's JSONL line and HTML section atomic.
_WRITE_LOCK = threading.Lock()

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<title>Image generation ledger</title>
<style>
:root {
  --bg: #fafafa; --fg: #1a1a1a; --muted: #6b6b6b; --accent: #8a4fff;
  --border: #e0e0e0; --code-bg: #f4f4f4;
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 940px; padding: 1.2rem;
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: var(--fg); background: var(--bg);
}
h1 { font-size: 1.3rem; margin: 0 0 .25rem; }
.subtitle { color: var(--muted); margin: 0 0 1rem; font-size: .9rem; }
section.gen {
  border: 1px solid var(--border); border-left: 4px solid var(--accent);
  border-radius: 6px; margin: .65rem 0; padding: .55rem .85rem; background: white;
}
section.gen > header {
  font-size: .8rem; color: var(--muted); margin-bottom: .35rem;
  display: flex; gap: .55rem; align-items: center; flex-wrap: wrap;
}
.badge {
  display: inline-block; background: var(--code-bg); padding: .05rem .45rem;
  border-radius: 3px; margin-right: .35rem; font-family: ui-monospace, monospace;
  font-size: .78rem;
}
.prompt { white-space: pre-wrap; word-wrap: break-word; margin: .25rem 0; }
dl.kv {
  display: grid; grid-template-columns: max-content 1fr; gap: .15rem .85rem;
  font-size: .85rem; margin: .25rem 0;
}
dl.kv dt { color: var(--muted); font-family: ui-monospace, monospace; }
dl.kv dd { margin: 0; font-family: ui-monospace, monospace; word-break: break-word; }
.usage-line { font-size: .8rem; color: var(--muted); margin: .4rem 0 .1rem; }
</style>
<h1>Image generation ledger</h1>
<p class="subtitle">Audit trail of image API spend for this run.</p>
"""


def resolve_image_log_path(provider: Provider) -> Path | None:
    """Resolve the image ledger path from the ``log_dir`` switch.

    Cascade (no model/convo layer exists for one-shot image generation):
    the provider's ``log_dir`` override wins, else the manager's run directory.
    Returns ``None`` when logging resolves to disabled (``log_dir=False`` at any
    layer) or there is no manager (e.g. a bare provider in a unit test)."""
    override = getattr(provider, "_log_dir_override", None)
    if override is False:
        return None
    if override is not None and override is not True:
        return Path(override) / _LEDGER_NAME

    manager = getattr(provider, "_manager", None)
    if manager is None:
        return None
    run_dir = manager._ensure_run_dir()
    if run_dir is None:
        return None
    return run_dir / _LEDGER_NAME


def build_image_record(
    *,
    prompt: str,
    model: str,
    provider: str,
    n: int,
    size: str | None,
    aspect_ratio: str | None,
    quality: str | None,
    background: str | None,
    output_format: str | None,
    reference_images: list[ImageBlock] | None,
    result: ImageResult,
) -> dict[str, Any]:
    """Build the JSONL record for one generation. The full prompt is kept (audit);
    reference images are recorded as a count, and output images as
    ``{media_type, bytes}`` — never the bytes themselves."""
    usage = result.usage
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "reference_images": len(reference_images) if reference_images else 0,
        "usage": (
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "image_count": usage.image_count,
            }
            if usage is not None
            else None
        ),
        "image_count": len(result.images),
        "images": [{"media_type": b.media_type, "bytes": len(b.data)} for b in result.images],
        "paths": [str(p) for p in result.paths],
    }


def log_image_generation(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` to the JSONL ledger and a rendered section to the HTML
    sibling. Best-effort: any failure is swallowed with a warning so a logging
    problem never prevents returning the generated image."""
    try:
        with _WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            _append_html(path.with_suffix(".html"), record)
    except Exception as e:  # noqa: BLE001 - logging must never break generation
        warnings.warn(f"image-generation logging failed: {e}", stacklevel=2)


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _append_html(html_path: Path, record: dict[str, Any]) -> None:
    if not html_path.exists():
        html_path.write_text(_HTML_HEAD, encoding="utf-8")

    out: list[str] = []
    out.append(
        '<section class="gen">\n  <header>'
        f'<span class="badge">{_escape(str(record["provider"]))}:'
        f"{_escape(str(record['model']))}</span>"
        f"<small>{_escape(str(record['ts']))}</small>"
        "</header>\n"
    )
    out.append(f'<div class="prompt">{_escape(str(record["prompt"]))}</div>\n')

    out.append('<dl class="kv">\n')
    for key in ("n", "size", "aspect_ratio", "quality", "background", "output_format"):
        v = record.get(key)
        if v is not None:
            out.append(f"  <dt>{key}</dt><dd>{_escape(str(v))}</dd>\n")
    if record.get("reference_images"):
        out.append(f"  <dt>reference_images</dt><dd>{record['reference_images']}</dd>\n")
    out.append(f"  <dt>image_count</dt><dd>{record['image_count']}</dd>\n")
    for i, img in enumerate(record.get("images", [])):
        out.append(
            f"  <dt>image[{i}]</dt>"
            f"<dd>{_escape(str(img['media_type']))}, {img['bytes']} bytes</dd>\n"
        )
    for i, p in enumerate(record.get("paths", [])):
        out.append(f"  <dt>path[{i}]</dt><dd>{_escape(str(p))}</dd>\n")
    out.append("</dl>\n")

    usage = record.get("usage")
    if usage:
        badges = "".join(
            f'<span class="badge">{label} {usage[key]}</span>'
            for key, label in (
                ("input_tokens", "in"),
                ("output_tokens", "out"),
                ("total_tokens", "total"),
                ("image_count", "images"),
            )
            if usage.get(key)
        )
        if badges:
            out.append(f'<div class="usage-line">{badges}</div>\n')

    out.append("</section>\n")
    with html_path.open("a", encoding="utf-8") as fh:
        fh.write("".join(out))
