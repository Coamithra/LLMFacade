# llama.cpp: reasoning vs tool-calling (and why thinking vanishes from logs)

*Recorded 2026-05-29. Surfaced debugging MTGAI: gemma4 review turns logged no reasoning even though the "reasoning in logs" feature was working. Models: `vlad-gemma4-26b-dynamic`, `unsloth/gemma-4-26B-A4B-it` (UD-IQ4_XS).*

## TL;DR

If a llama.cpp model emits **no reasoning on tool-using turns**, it's usually not our bug. Three things stack up:

1. **Managed mode doesn't pass `--jinja`**, so llama-server ignores the GGUF's embedded chat template and uses built-in format detection (`peg-gemma4`). The model-specific thinking machinery never runs.
2. **Gemma 4 / Qwen3 gate thinking via the `enable_thinking` chat-template kwarg**, not a prompt token. No `--jinja` ⇒ no `enable_thinking` ⇒ default behavior only.
3. **Reasoning and tool-calling are in tension** (see below). With thinking on, the model often answers in text instead of calling the tool; if you *force* the tool, llama.cpp misroutes the call into `reasoning_content`.

Our capture pipeline is fine — verified end to end (raw HTTP → OpenAI SDK → external mode → managed mode). When the model produces reasoning, we log it.

## The reasoning ↔ tool-calling tension

Empirically, on the gemma4 quants with `--jinja --reasoning-format auto`:

| thinking | tool_choice | reasoning | tool call | note |
|---|---|---|---|---|
| on  | `auto`   | yes | **sometimes** | model decides each turn (the CoT-vs-tools tradeoff) |
| on  | **forced** | huge | **lost** | llama.cpp misroutes the tool XML into `reasoning_content` |
| off | any      | none | yes | clean tool call, reasoning parser disabled |

- The tradeoff is acknowledged upstream — Fireworks/Qwen: *"open-source LLMs forced a choice between showing chain of thought or calling tools deterministically."* Newer architectures (Qwen 3) explicitly do both in one pass; this gemma4 quant is inconsistent under `auto`.
- **Forcing `tool_choice` with thinking on is a known bug, not randomness.** When the model is in thinking mode, a forced/grammar-constrained tool call gets parsed into `reasoning_content` instead of `tool_calls` — wrong `finish_reason`, empty `tool_calls`. Observed directly: forced + thinking-on → 23k chars of "reasoning", `finish=stop`, no tool call. Refs: [ggml-org/llama.cpp#20809](https://github.com/ggml-org/llama.cpp/issues/20809), [discussion#12204](https://github.com/ggml-org/llama.cpp/discussions/12204). Maintainer guidance: don't grammar-constrain a reasoning model.

## What actually works

- **Recipe for reasoning + structured output:** `--jinja` + `chat_template_kwargs={"enable_thinking": true}` + `tool_choice=auto` (never forced) + a **retry-if-no-toolcall guard** for the turns it answers in text.
- **`enable_thinking` is respected** by gemma4 (`false` → clean tool call, no reasoning). Some models ignore it (e.g. MiniMax — [#20196](https://github.com/ggml-org/llama.cpp/issues/20196)); auto-detect can't fix model-side non-compliance.
- These parser interactions are **version-sensitive** — pin / recommend a known-good llama.cpp build. **llama.cpp does not auto-update** — `llama-server` is a hand-built / hand-downloaded binary (build from source or grab a GitHub release). Periodically `git pull` + rebuild (or re-download) and re-run the reasoning+tool matrix above; bugs like #20809 get fixed upstream over time, so a stale binary keeps a stale bug. (Switching from ollama to llama.cpp doesn't escape this class of bug; it's inherent to the reasoning+tool parsing layer both share.)

## Other notes

- **GGUF chat template is cheap to read** (stdlib header parse, no weights) — the basis for thinking-strategy auto-detection. The vlad quant carried an *outdated* gemma4 template (llama-server warns "applying compatibility workarounds"); unsloth's Apr-11+ template fixed tool-calling. Same `enable_thinking` support in both; the fix was in tool-call handling.
- **In llama.cpp, vision is already separate** — the text GGUF holds no vision weights; you opt in with `--mmproj`. There's no "vision-stripped" variant to chase.
- **Reasoning only reaches us with `--jinja` for the *template-kwarg* control**, but `reasoning_content` itself is emitted in the legacy path too — the bug is specifically the tool-turn interaction, not reasoning capture in general.

## Follow-up

Folded into the facade (Trello card `6a19f630`, shipped): the llamacpp provider now takes a `thinking` knob (`ThinkingMode` → `chat_template_kwargs={"enable_thinking": …}`), a managed-mode `jinja` launch knob (default-on), and auto-detects each GGUF's `thinking_style` to warn when the knob won't bite. See the **llama.cpp** provider quirks in `CLAUDE.md` (**Thinking control** and **Thinking-style auto-detect**). Still caller-owned: the retry-if-no-toolcall guard for reasoning + structured output under `tool_choice="auto"`.
