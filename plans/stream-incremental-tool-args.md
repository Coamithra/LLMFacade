# Plan: stream tool-call arguments incrementally (live, pre-parse)

_Spec'd 2026-06-05. Goal: let a caller iterating `convo.stream()` / `astream()` read tool-call
argument **fragments as they arrive**, before the JSON is complete or valid — the same live
experience text and thinking already get._

## Motivation

Today `stream()` surfaces tool calls as a **single terminal event**: the providers buffer every
argument fragment internally and only emit one `StreamEvent(tool_call_delta=ToolCall(...))` once the
arguments JSON has closed and parsed. So a consumer cannot watch the tool call being written — they
get nothing, then the whole parsed call at once.

Text (`text_delta`) and reasoning (`thinking_delta`) already stream token-by-token. Tool calls are
the odd one out. We want parity: a caller should be able to **see the response as it's being written,
even when it's a tool call and the JSON isn't valid yet.**

What the caller does with partial args is **explicitly their problem**, by design:
- Maybe they just render the raw fragments to a user live ("the model is calling `search(...)`…") and
  don't care about structure until it's done.
- Maybe they run heuristic / defensive partial-JSON parsing to make a best-guess preview.
- Maybe they ignore the deltas and only act on the final parsed `ToolCall`.

llmfacade's job is only to **forward the fragments**, not to interpret them.

## Current behavior (where the buffering lives)

All stream paths accumulate then emit-once at block close:

- **Anthropic** — `input_json_delta` chunks accumulate into `state["current_tool"]["input_json"]`;
  one `tool_call_delta` is yielded at `content_block_stop`.
  See `providers/anthropic.py` (`content_block_delta` / `content_block_stop` handling).
- **OpenAI** — `fn.arguments` chunks accumulate into `slot["args"]`; one `tool_call_delta` at
  `finish_reason`. See `providers/openai.py` stream path.
- **llama.cpp** — same as OpenAI (`slot["args"]` → emit at finish). See `providers/llamacpp.py`.

The public `Conversation.stream` / `astream` already yield each `StreamEvent` straight through
(`conversation.py`, the `for ev in ... _stream_raw(req): ... yield ev` loop) and accumulate
`tool_call_delta`s into the final assistant turn. So the consumer-facing change is purely **additive
new events** — the finalization logic is unchanged.

Note: the recently-merged `raw_arguments` field already preserves the full unparsed string on a
**failed** terminal parse (PR #4). This plan is the *streaming* analog — the fragments as they arrive,
regardless of whether the final parse succeeds.

## Design (additive, backward-compatible)

Mirror the thinking pattern: stream **deltas** during the call, keep the **consolidated** event at the
end. The existing terminal `tool_call_delta` (full parsed `ToolCall`, with `raw_arguments` on failure)
stays exactly as-is, so every existing consumer is unaffected.

1. **New type** in `models.py`:

   ```python
   @dataclass(frozen=True, slots=True)
   class ToolArgsDelta:
       index: int            # position of this tool call within the turn (0-based)
       fragment: str         # the raw arguments-string chunk, verbatim from the provider
       id: str | None = None    # tool-call id, once the provider has emitted it
       name: str | None = None  # tool name, once known (usually on the first fragment)
   ```

2. **New optional field** on `StreamEvent`:

   ```python
   tool_args_delta: ToolArgsDelta | None = None
   ```

   Default `None` → no breakage. Consumers opt in by reading it.

3. **Provider stream paths** — emit a `StreamEvent(tool_args_delta=ToolArgsDelta(...))` for each raw
   fragment, *in addition to* still accumulating it for the terminal `tool_call_delta`:
   - Anthropic: in the `input_json_delta` branch, yield a delta carrying `partial_json` (id/name come
     from the `content_block_start` for the `tool_use` block; emit name on the first fragment or on a
     dedicated start event).
   - OpenAI + llama.cpp: in the `fn.arguments` accumulation branch, yield a delta carrying the chunk
     (`tc.index`, `slot["id"]`, `slot["name"]`).
   - Apply to **both sync and async** stream methods in all three providers.
   - Anthropic/Google note: Google's function-call args arrive structured (not a JSON string stream),
     so it has no fragments to forward — it keeps emitting only the terminal `tool_call_delta`. Document
     this asymmetry (same reason Google was skipped for `raw_arguments`).

4. **`Conversation.stream` / `astream`** — already pass `ev` through verbatim, so no change needed for
   forwarding. Confirm the accumulation loop ignores `tool_args_delta` for history-building (the final
   `tool_call_delta` remains the source of truth for the assistant turn). Add an explicit no-op branch
   only if clarity demands it.

5. **Ordering guarantee** to document: for a given tool call the consumer sees zero-or-more
   `tool_args_delta` events (in arrival order) followed by exactly one terminal `tool_call_delta` with
   the parsed `input` (or `raw_arguments` if parsing failed). Concatenating all `fragment`s for one
   `index` reconstructs the exact raw arguments string.

## Out of scope / non-goals

- **No partial-JSON parsing in llmfacade.** We forward raw fragments only. Defensive/heuristic parsing
  is the caller's choice (mention `partial-json-parser`-style libraries in docs, but don't depend on one).
- No change to the non-streaming (`send`/`asend`) surface.
- No new cache behavior (streaming hits replay from the cached `Response`; partial fragments are not
  re-synthesized — `replay_stream` continues to emit the consolidated tool call, matching its existing
  thinking/text replay fidelity. Document that live-arg streaming is a cache-miss-only experience).

## Tests

- New streaming-tool tests per provider (extend `tests/test_*` stream fakes that feed `fn.arguments` /
  `input_json_delta` in multiple chunks):
  - Assert N `tool_args_delta` events arrive in order, and `"".join(fragments)` == the full args string.
  - Assert `id`/`name` are populated once known.
  - Assert exactly one terminal `tool_call_delta` still follows, with the correct parsed `input`.
  - Failure path: malformed/truncated streamed args → terminal call has `input == {}` and
    `raw_arguments == "".join(fragments)` (ties into PR #4).
- Backward-compat: existing stream tests that only read `text_delta` / `tool_call_delta` still pass
  unchanged.

## Touch list

- `src/llmfacade/models.py` — `ToolArgsDelta` type + `StreamEvent.tool_args_delta` field.
- `src/llmfacade/providers/anthropic.py` — sync + async stream: emit fragment deltas.
- `src/llmfacade/providers/openai.py` — sync + async stream: emit fragment deltas.
- `src/llmfacade/providers/llamacpp.py` — sync + async stream: emit fragment deltas.
- `src/llmfacade/conversation.py` — confirm pass-through; doc the ordering contract.
- `CLAUDE.md` — document the streaming contract under the streaming/`StreamEvent` discussion.
- `tests/` — per-provider streamed-args tests + backward-compat assertions.

## Done when

A caller can iterate `convo.stream(prompt)` and, for a turn that calls a tool, read the arguments text
live via `ev.tool_args_delta.fragment` before the call completes — then still receive the final parsed
`ToolCall` (or `raw_arguments` on failure) via `ev.tool_call_delta`. Card moved to Done and this file
deleted per the repo convention.
