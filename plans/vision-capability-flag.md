# Vision capability flag + cross-provider image gating

Trello card: **Support mmproj_path + vision-block marshalling in managed llama.cpp provider**
(`6a134b8fa3decdd9ff22361d`, https://trello.com/c/V1xiSOK3)

## Context

The card's two explicit "Required (llmfacade-side)" items are already shipped on `main`:

1. `mmproj_path` on managed-mode `provider.new_model(...)` — merged (`a2d96f4`, `19f0130`); documented in CLAUDE.md.
2. Vision-block marshalling into the OpenAI `image_url` shape for llama-server — `llamacpp.py` `_message_to_api`, covered by two tests in `test_llamacpp.py`.

The "Wider Requirement" on the card also asks for a generic image API across providers *"…and the usual availability checks etc."* All four providers already marshal `ImageBlock` on user messages (Anthropic `image`/`source`, OpenAI/llamacpp `image_url`, Google `inline_data`). The remaining gap is the **availability check**: there is no `"vision"` capability flag, so callers cannot query `model.is_available("vision")` and a non-vision model silently produces a malformed request instead of a clean `UnsupportedFeature`.

This card closes that gap.

## Design

Mirror the existing `"tools"` pure-capability-flag precedent exactly (a flag in `SUPPORTS`, **not** in `RUNTIME_KNOBS`, queried via `is_available`/`get_capabilities`, narrowable via `capability_override`).

### File-by-file

- `src/llmfacade/providers/anthropic.py` — add `"vision"` to module-level `_SUPPORTS` (flows to `AnthropicProvider.SUPPORTS` and every `AnthropicModel` enum member, all of which are multimodal).
- `src/llmfacade/providers/openai.py` — add `"vision"` to `SUPPORTS`.
- `src/llmfacade/providers/google.py` — add `"vision"` to `SUPPORTS`.
- `src/llmfacade/providers/llamacpp.py` — add `"vision"` to `SUPPORTS` (the wire format can carry images in both modes; whether the loaded model actually uses them is a runtime/`mmproj` concern, same shape as "the API accepts images" for the cloud providers).
- `src/llmfacade/conversation.py` — request-time gate in `_build_request` (the single funnel for `send`/`asend`/`stream`/`astream`, and where the wire-format invariant is already enforced). Add a module-level `_message_has_image(m) -> bool` helper that scans top-level content blocks and `ToolResultBlock.content` for `ImageBlock`. If any history message carries an image and `not self._model.is_available("vision")`, raise `UnsupportedFeature("vision", provider.NAME, model_id)`.
- `tests/conftest.py` — add `"vision"` to `MockProvider.SUPPORTS` so the generic test double is vision-capable by default (keeps the existing `test_hash_changes_with_image_bytes` request build green; the negative-gate test narrows via `capability_override`).

### Public API / behaviour

- `model.is_available("vision")` / `convo.is_available("vision")` → `True` for all four stock providers.
- `get_capabilities()` includes `"vision"`.
- Narrow a text-only model with `capability_override=provider.SUPPORTS - {"vision"}`; sending an `ImageBlock` against it raises `UnsupportedFeature` at request time.
- No change to `RUNTIME_KNOBS`; `"vision"` is never a settable knob (no `Conversation(vision=...)` kwarg), exactly like `"tools"`.

## Tests

New file `tests/test_vision.py`:

- `test_all_stock_providers_declare_vision` — `"vision"` in each provider class's `SUPPORTS`.
- `test_anthropic_marshals_image_block` — `_content_to_api([...])` → `image`/`source`/`base64` shape.
- `test_openai_marshals_image_on_user` / `test_openai_drops_image_on_assistant_warns`.
- `test_google_marshals_image_block` — `inline_data`/`mime_type`.
- `test_vision_gate_raises_on_non_vision_model` — `MockProvider` narrowed to drop `"vision"`, `send([ImageBlock])` raises `UnsupportedFeature`; assert no provider call recorded.
- `test_vision_gate_allows_when_supported` — default `MockProvider` (vision) sends image; `req.messages` carries the `ImageBlock`.
- `test_vision_gate_checks_history_image` — `add_user_message([ImageBlock])` then `send("text")` on a narrowed model raises.
- `test_capability_override_drops_vision` — narrowed real-provider model reports `is_available("vision") is False`.

(llamacpp marshalling stays covered by the two existing `test_llamacpp.py` tests.)

## Docs

- CLAUDE.md: note `"vision"` in the "Capability gating" section and the "Adding a new provider" pure-capability-flag note (alongside `"tools"`/`"tool_choice"`); document the request-time gate.

## Out of scope

- Tool-result image marshalling for OpenAI/Google/llamacpp (only Anthropic marshals images inside `ToolResultBlock` today). The gate detects them, but emitting them is a separate enhancement.
- Vision-projector autodetection (already tracked as future work in CLAUDE.md).
- Any MTGAI-side wiring (downstream card).

## Verification

- `ruff check src/ tests/`, `ruff format --check`.
- `python -c "import llmfacade"`.
- `pytest` (default markers, no integration). Run from the worktree with `PYTHONPATH=<worktree>/src` and confirm `llmfacade.__file__` resolves to the worktree so the edits are exercised.
