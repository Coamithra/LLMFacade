# `tool_result_images` capability + auto-loop image handling

No Trello card (card creation is currently blocked by permissions; tracked
conversationally). Follow-up to the `vision` capability work.

## Context

Images in a `ToolResultBlock` (a tool *returns* an image) only reach the model
on Anthropic — its `_content_to_api` marshals nested `ImageBlock`s. OpenAI,
Google, and llamacpp flatten tool-result content to text and silently drop the
image. The new `"vision"` gate passes on those three (they have vision), so the
drop is silent. This closes that hole and teaches the auto-loop helpers to do
the right thing.

## Design

### 1. New capability flag `"tool_result_images"`

A pure capability flag (in `SUPPORTS`, not `RUNTIME_KNOBS`), declared **only by
Anthropic** (added to `_SUPPORTS`, so it flows to `AnthropicModel` members).
OpenAI/Google/llamacpp do not declare it. Queryable via `is_available` /
`get_capabilities`; narrowable via `capability_override`.

### 2. Gate split in `Conversation._build_request`

Split `_message_has_image` into:
- `_message_has_toplevel_image(m)` — `ImageBlock` directly in message content → gated by `"vision"` (unchanged behaviour).
- `_message_has_tool_result_image(m)` — `ImageBlock` nested in a `ToolResultBlock` → gated by `"tool_result_images"`.

Each raises its own `UnsupportedFeature`. This converts the silent drop into a
clean, queryable error.

### 3. Auto-loop helpers (`helpers.py`)

`run_bound_tools` / `arun_bound_tools` learn to carry images a tool returns:
- `_normalize_tool_result(result)` — `str` passes through; `ImageBlock` →
  `[ImageBlock]`; a non-empty list of `TextBlock`/`ImageBlock` is preserved;
  everything else keeps the old `_stringify` behaviour (backward compatible —
  `echo()->7` still yields `"7"`).
- For each call, `_append_tool_result(...)`:
  - no images, or model `is_available("tool_result_images")` → append the tool
    result as-is (image embedded on Anthropic).
  - otherwise → reduce the tool result to its text and queue the image(s) in a
    `deferred_images` list.
- After the whole batch, if any images were deferred, append **one** follow-up
  user message `[TextBlock(note), *deferred_images]`. Deferring until after all
  tool results keeps every `tool_use` paired with its `tool_result` (a user
  message between two tool results would dangle the second call).

`run_to_completion` / `arun_to_completion` inherit this for free (their next
`send()` ships the appended history).

## Tests (`tests/test_vision.py`)

- `test_only_anthropic_declares_tool_result_images`.
- `test_tool_result_image_gate_raises_without_flag` — narrowed mock (vision but
  no tool_result_images), manual `add_tool_result` with image, `send` raises
  `UnsupportedFeature`; no provider call.
- `test_tool_result_image_allowed_with_flag` — mock with the flag; image rides
  in the tool result; provider called once.
- `test_run_bound_tools_embeds_image_when_supported` — tool returns `ImageBlock`,
  model has the flag → tool-result content is a block list with the image, no
  extra user message.
- `test_run_bound_tools_defers_image_when_unsupported` — model lacks the flag →
  tool-result content is text, a trailing user message carries the image.
- `test_run_bound_tools_string_return_unchanged` — `echo()->7` still `"7"`.

## Docs

- CLAUDE.md: bump the pure-capability-flag note to four; expand **Vision** under
  Capability gating with the tool-result split + helper fallback; update the
  `helpers.py` key-file line.
- README.md: one sentence on `tool_result_images` + the helper fallback.

## Out of scope

- Actually marshalling tool-result images on OpenAI/Google/llamacpp (uncertain
  upstream-API support; the flag makes the absence explicit instead).
- The granularity question (one day a generic block-capability model). Accepted.

## Verification

ruff, `import llmfacade`, `pytest` (worktree `PYTHONPATH`), `/review`.
