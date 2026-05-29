# llamacpp: facade-level `thinking` control (+ reasoning-with-tools)

## Problem

The llamacpp provider captures reasoning *inbound* (`reasoning_content` → `ThinkingBlock`) but offers **no outbound control** — `"thinking"` is not in `LlamacppProvider.SUPPORTS`. Managed mode also never passes `--jinja`, so llama-server bypasses the GGUF's embedded chat template entirely (built-in format detection, e.g. `peg-gemma4`), meaning the model-specific thinking machinery (Gemma 4 / Qwen3 `enable_thinking` template kwarg) never runs. Net effect: thinking can't be toggled, and on tool-using turns reasoning often doesn't appear at all.

Surfaced while debugging MTGAI: every review turn passes a tool, and the bound model emitted no reasoning, so the new "reasoning in logs" feature looked broken when it wasn't.

## Findings (validated empirically + against llama.cpp upstream, May 2026)

- Gemma 4 (and Qwen3) gate thinking via the **`enable_thinking` chat-template kwarg**, not a magic prompt token. It lives in the GGUF's embedded `tokenizer.chat_template`.
- **Auto-detecting the strategy is feasible and cheap.** Parsing the embedded template out of the GGUF header is a ~90-line stdlib job (no weights read; the `gguf` pip pkg would be optional sugar). Confirmed by reading vlad's vs unsloth's templates: classifier tagged both `TEMPLATE_KWARG (enable_thinking)`. Hook point: `new_model()` in managed mode (has the gguf path; same seam as the `llama-fit-params` probe). External mode: read from llama-server `/props`.
- **Reasoning ↔ tool-calling tension is real and documented upstream — not our bug:**
  - `--jinja` + `enable_thinking=True` + `tool_choice=auto` → the model *can* emit reasoning **and** a tool call in one turn, but under `auto` it sometimes answers in text instead (genuine model choice; the widely-noted "CoT vs tools" tradeoff). Needs a retry-if-no-toolcall guard for reliable structured output.
  - **Forcing `tool_choice`** (required / specific fn) with reasoning on is *worse* — a known llama.cpp bug: the tool-call XML is misrouted into `reasoning_content` and never parsed as `tool_calls` (wrong `finish_reason`, empty `tool_calls`). Refs: ggml-org/llama.cpp#20809, discussion#12204. Maintainer guidance: don't grammar-constrain a reasoning model.
  - `enable_thinking=False` reliably yields clean tool calls (no reasoning).

## Proposed work

1. **Managed mode: `--jinja` support.** Add a launch knob (lean toward default-on) so llama-server renders the embedded template — prerequisite for template-kwarg thinking control, and on newer Gemma 4 quants for tool-calling at all. Keep `--reasoning-format` at its default (`auto`); never `none`.
2. **`thinking` knob for llamacpp.** Add `"thinking"` to `SUPPORTS`. Translate the existing `ThinkingMode`: `ADAPTIVE` → `chat_template_kwargs={"enable_thinking": true}`, `DISABLED` → `{"enable_thinking": false}`, routed via the provider's existing `extra_body`. Mirrors Anthropic's `_thinking_to_api`: uniform knob value, provider-specific wire form.
3. **Thinking-strategy auto-detect (convenience layer).** At `new_model()`, parse the GGUF template, classify into `{TEMPLATE_KWARG, THINK_TOKEN, REASONING_BUDGET, DEFAULT}`, set a default `thinking_style`; an explicit `thinking_style=` / override always wins. Best-effort, never silently wrong.

Suggested order: **#1 + #2 first** (unblocks reasoning-with-tools for real workloads), then **#3** as the ergonomics pass.

## Caveats / non-goals

- Don't force `tool_choice` when thinking is on (triggers the misrouting bug) — document it.
- `enable_thinking` is honored by Gemma 4 / Qwen3 but **ignored by some models** (e.g. MiniMax — upstream #20196); auto-detect can't fix model-side non-compliance.
- These parser interactions are **version-sensitive** — recommend/pin a known-good llama.cpp build.
- MTGAI implication: a vlad→unsloth swap needs the `--jinja`+`enable_thinking` config or tool calls break; reasoning-with-tools needs a retry guard.
