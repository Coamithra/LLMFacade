# Extend repetition-loop detection to thinking tokens

## Context

The repetition-loop guard (card "Repetition-loop detection + auto-retry (port from
MTGAI)", shipped in #7) currently scans **assistant text** and **streamed tool-call
arguments** for degenerate tandem repeats, but **not the model's reasoning /
thinking stream**. This is a real gap: a local reasoning quant (Gemma 4 / Qwen3,
low-bit especially) can fall into a degenerate loop *inside its chain-of-thought*
— burning the whole token budget on repeated reasoning before it ever emits visible
text — and the guard would not catch it. Looping-in-reasoning is precisely the
failure mode this feature exists for.

The plumbing is half-present: `_StreamBuffers` already collects the consolidated
`thinking_blocks` (for history/`Response` building), but the detector
(`_detect_in_buffers` → `_detection_text`) is only ever handed `text_buf` +
`tool_calls`. Thinking deltas also do not bump the `chars_since_check` cadence
counter, so even the periodic check never fires during a thinking-only phase.

## Design

Scan the **live thinking stream** (`StreamEvent.thinking_delta`) the same way text
and tool-args are scanned. Thinking is accumulated from the per-token
`thinking_delta` events (not the consolidated `thinking_block`, which would
double-count and which arrives all-at-once only *after* the loop has already burned
the budget).

All changes in `src/llmfacade/conversation.py`:

1. **`_detection_text(thinking_buf, text_buf, tool_calls)`** — prepend a thinking
   buffer. Detection string is `thinking + text + tool-args`. The MTGAI detector is
   a *suffix* scan, so during a thinking-only phase the suffix is the thinking
   stream (loop caught live); once text starts, the suffix is text and thinking
   sits as a harmless prefix. Net effect: thinking loops are caught *while thinking
   streams*, text loops while text streams — exactly the desired live behaviour.

2. **`_StreamBuffers`** — add `thinking_text: list[str]`, fed in `absorb` from
   `ev.thinking_delta`. Keep the existing `thinking_blocks` accumulation untouched
   (it still drives history/`Response`).

3. **`_detect_in_buffers(thinking_buf, text_buf, tool_calls, guard)`** — thread the
   thinking buffer through to `_detection_text`.

4. **`stream` / `astream` inline loops** — add a `thinking_buf: list[str]`
   accumulator fed from `ev.thinking_delta`; bump `chars_since_check` by
   `len(ev.thinking_delta)`; pass `thinking_buf` to both the mid-stream and the
   final-flush `_detect_in_buffers` calls. `RepetitionLoopError.partial_text` then
   naturally includes the looping reasoning.

5. **`_drive_guarded_stream` / `_adrive_guarded_stream`** — use `buf.thinking_text`
   in the `_detect_in_buffers` calls (mid-stream and final flush).

No public API change, no new knob, no capability flag. `RepetitionGuard` is
unchanged. The behaviour shift is purely "the detector now also sees reasoning".

### Why deltas, not the consolidated block

Providers emit `thinking_delta` token-by-token, then flush one consolidated
`thinking_block` before the first text (llama.cpp, Anthropic). Scanning deltas
catches the loop *mid-generation*; scanning the consolidated block would only fire
after the entire (already-wasted) thinking turn completed. Detection uses deltas;
history-building keeps using the block. The two buffers never overlap, so no
double-count.

## Tests

`tests/test_repetition.py`:

- Extend `LoopingProvider._body_events` (or add a sibling knob) to stream a
  degenerate **thinking** loop (`StreamEvent(thinking_delta="reason ")` ×N) before
  the body, gated by a flag so existing text-loop tests are unaffected.
- `test_send_catches_thinking_loop` — a thinking-only loop on attempt 0 is detected
  and the call retried to a clean body (`stream_count == 2`).
- `test_send_thinking_loop_exhausts` — a permanent thinking loop raises
  `RepetitionLoopError` with the reasoning text in `partial_text`.
- `test_stream_aborts_on_thinking_loop` — `stream` aborts mid-thinking and rolls
  back (`history == []`).
- `test_clean_thinking_not_flagged` — a short, non-repeating thinking preamble
  followed by a clean body does not false-positive.

## Out of scope

- Tool **result** text (external, not model-generated, never streamed) — correctly
  remains out of scope.
- The non-streaming `_complete_raw` path — the guard already routes `send` through
  the stream hook, so there is no separate non-stream thinking path to cover.
- External-mode providers that emit a consolidated `thinking_block` with *no*
  preceding deltas would not be scanned mid-stream; none of the stock providers do
  this, so it is a documented non-goal, not a regression.

## Verification

- `ruff check src/` + `ruff format src/` clean.
- `python -c "import llmfacade"`.
- `pytest tests/test_repetition.py` green (new + existing).
- Update `CLAUDE.md`: the streaming/repetition sections that say the detector runs
  "on the accumulating assistant text **and** any streamed tool-call arguments"
  gain "and the reasoning/thinking stream".
