# Logging / audit trail for image generation

Trello: https://trello.com/c/C7tSQz5X (board `69f86428`). Follow-up from the
image-generation card (cold /review finding #6).

## Context

`generate_image` / `LLM.generate_image` produce **no log artifact** — even though
hosted image generation (`gpt-image-1`, Gemini-native "Nano Banana") costs real
money. A production caller has zero audit trail of image API spend: no record of
prompt, model, count, token usage, or where images were written.

Chat goes through the manager's JSONL/HTML logging; image generation was
explicitly scoped out of the initial image PR to keep it reviewable. This card
closes that gap.

## Design

### Where the hook lives — template method on `Provider`

All three image entry points funnel through `provider.generate_image` /
`agenerate_image` (`ImageModel.generate` forwards to it; `LLM.generate_image`
resolves a provider and forwards to it). So the provider method is the single
chokepoint — log there once, cover everything.

Apply the same template-method split the chat side uses (`send` orchestrates,
`_complete_raw` does the work):

- `Provider.generate_image` / `agenerate_image` become **concrete** in the base:
  `result = self._generate_image_raw(...)` → `_log_image_generation(...)` →
  `return result`.
- Add base `Provider._generate_image_raw` / `_agenerate_image_raw` that raise
  `UnsupportedFeature("image_generation", ...)` — this *moves* the existing gate
  down one level, so Anthropic/llamacpp still fail fast (they don't override the
  raw hook), and `test_base_provider_generate_image_raises` still passes.
- Rename the three implementers' methods `generate_image` → `_generate_image_raw`
  and `agenerate_image` → `_agenerate_image_raw` (signatures unchanged; they keep
  their existing `_apply_save_dir(...)` call, so the returned `result.paths` is
  already populated when the base logs it). No other change to provider bodies.

### Log format — dedicated image ledger, not the conversation format

Image generation is a one-shot: no turns, no history, no system blocks, no tools.
The conversation header+turn structure doesn't fit. Instead write a **dedicated
append-only ledger** per manager run-dir:

- `<run_dir>/images.jsonl` — one JSON record per successful generation.
- `<run_dir>/images.html` — human-readable sibling (header written once on
  first record, then append-only, same HTML5 partial-file trick as `_html_log`).

A single shared ledger (rather than one file per call) is exactly what "audit
trail of image API spend" wants: a running record of every image generated in
the session, across all providers/models.

### New module `src/llmfacade/_image_log.py`

- `resolve_image_log_path(provider) -> Path | None` — reuses the existing
  `log_dir` switch: provider `_log_dir_override` (an explicit dir → write there;
  `False` → disabled → `None`) else the manager's `_ensure_run_dir()`. Returns
  `None` when logging resolves to disabled or there's no manager (e.g. a provider
  built bare in a unit test) — so logging is a no-op in those cases.
- `build_image_record(*, prompt, model, provider, n, size, aspect_ratio, quality,
  background, output_format, reference_images, result) -> dict` — the JSONL
  record (timestamp via `datetime.now(timezone.utc)`, full prompt, all request
  params, `reference_images` *count*, `usage` dict, per-image `{media_type, bytes}`,
  and `paths`). No image bytes in the log — byte sizes + saved paths only.
- `log_image_generation(path, record)` — append one JSONL line + one HTML
  `<section>` (writing the HTML header first if the file is new). **Best-effort**:
  wrapped in try/except that warns once and never propagates — a paid-for image
  must always be returned to the caller even if the log write fails. (Mirrors the
  fit-params probe's "logging never blocks the call" philosophy.)

### Config / cascade

Reuses the existing `log_dir` cascade — no new knob:
- On by default (provider built under a manager with logging on → ledger written).
- `LLM(log_dir=False)` or `new_provider(..., log_dir=False)` → disabled.
- `new_provider(..., log_dir="/some/dir")` → ledger written under that dir.

There is no model/convo layer for images, so the cascade is just
provider-then-manager. The `LLM.generate_image` path builds providers with no
`log_dir`, so they fall through to the manager run-dir (on by default).

## Tests — `tests/test_image_logging.py` (new)

A fake image provider (subclass declaring `image_generation`, `_generate_image_raw`
returns a canned `ImageResult`) under a real `LLM(log_dir=tmp_path)` manager — no
network.

- `test_image_log_writes_jsonl_record` — record has prompt/model/provider/n/usage/
  images/paths; one line per call; second call appends.
- `test_image_log_writes_html_sibling` — `images.html` created, header once, a
  `<section>` per record.
- `test_image_log_records_save_dir_paths` — `save_dir=` paths land in the record.
- `test_image_log_disabled_when_log_dir_false` — `LLM(log_dir=False)` writes nothing.
- `test_image_log_async` — `agenerate_image` logs identically.
- `test_image_log_reference_image_count` — references logged as a count, not bytes.
- `test_image_log_write_failure_is_swallowed` — a logging error doesn't break the
  returned `ImageResult` (monkeypatch the writer to raise; assert result returned + warns).
- Existing `test_image_generation.py` stays green (bare providers → no manager → no log).

## Out of scope

- Inline image thumbnails in the HTML (just metadata + paths). Future nice-to-have.
- Logging *failed* generations (exceptions propagate; a failed call generally
  isn't billed). Success = the spend event.
- Per-model/per-convo image-log overrides (no such layer for one-shot images).
- A dollar-cost figure (no provider returns one; derive from token counts).

## Verification

- `ruff check src/` + `ruff format src/` clean.
- `python -c "import llmfacade"`.
- `pytest` (unit only).
- Manual: run a real `LLM().generate_image(...)` against a provider, inspect
  `<run_dir>/images.jsonl` + `images.html`. (Needs an API key — flag for the user.)
