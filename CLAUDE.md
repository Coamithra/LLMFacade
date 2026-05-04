# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMFacade is a Python library providing a unified interface to multiple LLM providers (Anthropic, OpenAI, Google Gemini, llama.cpp via `llama-server`). Zero required runtime dependencies - provider SDKs are lazy-loaded only when used. Python 3.10+.

## Project tracker

Trello board: **LLMFacade** — https://trello.com/b/WIanVfPx (board id `69f86428`). Manage from the CLI with `trello --board 69f86428 ...` or `trello use 69f86428` to make it active.

`plans/*.md` files are the long-form source of truth for open work; each file has a corresponding Trello card whose description points back to it. When a card's work is completed (and merged), delete the matching `plans/<file>.md` and archive (or move to `Done`) the card. The convention is: a card with no `plans/` file behind it is either already done or hasn't been spec'd yet.

**Picking up a new card:** read `CONTRIBUTING.md` first — it covers the dev-loop expectations (style, tests, integration-test gating, commit/PR conventions) that every card depends on.

## Commands

```bash
# Install for development (all providers + tooling)
pip install -e ".[dev,all]"

# Install with specific provider(s)
pip install -e ".[anthropic,openai]"

# Lint
ruff check src/
ruff format src/

# Test
pytest
pytest tests/test_conversation.py::test_send_appends_user_and_assistant   # single test
```

**Integration tests cost money. Do not run them without express permission.**

`tests/integration/` hits real provider APIs (Anthropic, OpenAI, Google) and burns credits on every run. They are gated behind `-m integration` and skipped by default — keep it that way. Never invoke `pytest -m integration`, `pytest -m "integration or not integration"`, `pytest tests/integration/`, or any variant that includes them, unless the user has explicitly asked for that specific run in the current turn. Past authorization in earlier turns does not carry over. The llamacpp integration test is local and free, but still requires explicit permission so the rule stays simple: never auto-run anything under `tests/integration/`.

## Architecture

Four-level hierarchy. Each level owns its own concerns and spawns the next:

```
LLM            manager: shared api_keys; LLM.default() is a process-wide singleton
 -> Provider   identity (api_key, base_url) + SDK client + generation defaults
   -> Model    a model_id bound to a provider, with optional model-level defaults
     -> Conversation   stateful session: history, system blocks, tools, convo-level defaults
```

**Configuration is constructor-only.** Identity (api_key, base_url, model_id, system_blocks, tools, log_dir, log_path) is supplied at construction and never changes. Generation knobs (temperature, max_tokens, etc.) are accepted as kwargs at every layer and form a cascade (`provider < model < convo < per_call`); they are also immutable post-construction. There is no `Start()` step — conversations are usable immediately after construction.

Key files:

- `src/llmfacade/facade.py` — `LLM` manager; `new_provider(name, **kwargs)` dynamically imports a provider module via `PROVIDER_REGISTRY` and forwards kwargs.
- `src/llmfacade/provider.py` — `Provider` base class. Owns API-key resolution (override > manager dict > env var), `_init_client()`, the merged `CompletionRequest`, and the `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw` hooks. Also defines the `SystemBlock` dataclass and the cascade helpers `_validate_knobs` / `_filter_unsupported`.
- `src/llmfacade/model.py` — `Model` binds a `model_id` to a `Provider` and stores its own generation defaults. Supports a `capability_override` to narrow the provider's `SUPPORTS` set for a specific model (e.g. when calling a deprecated model that doesn't honor a knob the provider generally supports).
- `src/llmfacade/conversation.py` — `Conversation` is the stateful API: history, system blocks (with optional `cache=True`), tools, JSONL logging, `snapshot`/`rollback`/`clone`. `send`/`asend`/`stream`/`astream` are strict single round-trips that accept the full set of generation-knob kwargs as per-call overrides. The wire-format invariant (every `tool_use` matched by a `tool_result`) is enforced at request time and raises `ConversationStateError` if violated.
- `src/llmfacade/helpers.py` — Optional convenience built on the public Conversation API. `run_bound_tools(convo, resp)` dispatches every tool call whose name matches a `@tool` registered on the conversation and appends results. `run_to_completion(convo, prompt)` is the agent loop (`send` -> dispatch -> `send` -> ...) with a `max_iterations` cap, raising `ToolIterationLimitError` on overflow. Async equivalents: `arun_bound_tools`, `arun_to_completion`. Helpers touch only the public surface — no underscore attributes.
- `src/llmfacade/settings.py` — `RUNTIME_KNOBS` frozenset (the 15 string knob names) plus the value enums `EffortLevel`, `OutputFormat`, `EphemeralCacheTTL` (`FIVE_MINUTES`, `ONE_HOUR`).
- `src/llmfacade/models.py` — Frozen dataclasses for the wire format: `Message`, `TextBlock`, `ImageBlock`, `ToolUseBlock`, `ToolResultBlock`, `ToolCall`, `Response`, `Usage`, `StreamEvent`. `ImageBlock` has `from_path` / `from_base64`.
- `src/llmfacade/tools.py` — `@tool` decorator that builds a JSON schema from a function's signature + type hints + docstring. Handles primitives, `Literal`, `Optional`/`Union`, `list[T]`, `dict`.
- `src/llmfacade/exceptions.py` — Hierarchy rooted at `LLMError`: `AuthenticationError`, `RateLimitError`, `ProviderError`, `ModelNotFoundError`, `ProviderNotInstalledError`, `UnsupportedFeature`, `ToolIterationLimitError`, `ConversationStateError`, `CacheMissError`.
- `src/llmfacade/cache.py` — `ResponseCache` (filesystem backend), `fingerprint_request` / `hash_fingerprint` (canonical hash inputs), `replay_stream` / `areplay_stream` (synthesise a stream from a cached `Response`), and `resolve_cache` (cascade resolver). See the **Response cache** section below for behaviour.
- `src/llmfacade/providers/__init__.py` — `PROVIDER_REGISTRY` mapping names to `(module_path, class_name)`. `"google"` and `"gemini"` both resolve to `GoogleProvider`.
- `src/llmfacade/providers/{anthropic,openai,google,llamacpp}.py` — Provider implementations.

### Settings cascade

Every generation knob can be defaulted at any of four scopes:

1. `Provider(... temperature=0.7)` — applies to every model and convo under this provider.
2. `provider.new_model("...", temperature=0.7)` — overrides the provider default for this model.
3. `model.new_conversation(temperature=0.7)` — overrides for this convo.
4. `convo.send("...", temperature=0.7)` — overrides for this single call.

Precedence on read is `provider < model < convo < per_call` (later wins). Unknown kwarg names raise `TypeError`. Knobs not in the relevant layer's effective `SUPPORTS` raise `UnsupportedFeature` at the layer they're set. If a higher-scope default is for something a downstream model doesn't honor (e.g. `thinking` set on the Anthropic provider but the convo binds to a non-thinking model), the cascade silently drops the value at request time and warns once per `(key, source, model)` tuple.

### Logging

Logging is **on by default**. `LLM(log_dir=..., max_log_folders=10)` configures the manager-level root and retention. Each `LLM` instance reserves a session-stamped subfolder `<log_dir>/llmfacade<YYYYMMDD-HHMMSS>/` (default base: `<cwd>/logs`). On first write, the manager prunes older sibling `llmfacade*` directories down to `max_log_folders` and materialises the new one. Each `Conversation`'s log file is `<run_dir>/<convo.name>.jsonl` with an HTML sibling. The convo's `name` is auto-generated (`convo-<8hex>`) unless you pass `name=` to `new_conversation`.

`log_dir` cascades: convo > model > provider > manager. Any layer can pass `log_dir=False` to disable logging for its scope; a lower layer can re-enable by supplying its own `log_dir`. `Conversation(log_path=Path(...))` is an explicit-file override that bypasses the cascade entirely; `log_path=False` disables logging for that one convo.

The JSONL log starts with a single `settings` header record listing every effective knob, its value, and which scope (`provider`/`model`/`convo`) supplied it — plus the system blocks and tool names. Subsequent entries are tight: `request` records carry only `overrides` (per-call kwargs) and `new_messages` (delta since last log), and `response` records carry the assistant content and a `cache_summary` block.

`cache_summary.approximate_messages_cached` maps `cache_read_tokens` back to a message index. The lookup uses **turn-boundary tracking**: each successful send/stream records `(msg_count_at_send, total_input_tokens)` from `usage`, and a later turn's `cache_read_tokens` is matched exactly against that list (cache markers always sit at turn boundaries, so a hit typically equals some prior turn's recorded total). When matched, `tokenizer` reports `"exact (turn-boundary)"`. If no recorded boundary matches (first-turn caching, system-block-only markers, mid-prefix divergence after rollback), it falls back to a per-message walk via `Provider.count_tokens` — exact for OpenAI (tiktoken), Google (sentencepiece via `google-genai[local-tokenizer]`), and llamacpp (server `/tokenize`); `chars/4` for Anthropic.

### Token counting

`Provider.count_tokens(text, *, model_id=None)` and `Provider.tokenizer_name(model_id=None)` are the public local-tokenizer API. Convenience wrappers `Model.count_tokens(text)` and `Model.tokenizer_name()` bind the model id automatically. Always local — never makes an external network call (the llamacpp provider calls its own running `llama-server`'s `/tokenize`, which is local by definition). Install the optional `tokenizers` extra (`pip install llmfacade[tokenizers]`) to enable tiktoken (OpenAI) and sentencepiece (Google). Anthropic has no offline tokenizer and returns `chars/4`; for exact counts call `client.messages.count_tokens` via the SDK directly. llamacpp uses the running server's `/tokenize` and falls back to `chars/4` on connection error.

### Response cache (deterministic replay)

Off by default. Set `cache_dir=<path>` at any of provider, model, or conversation scope to enable a filesystem-backed cache of `Response` objects. On a hit, no provider call is made — the stored response is returned (or replayed as a stream). The hash key covers every input that affects output: provider name, model id, system blocks (including `cache=True` markers — flipping caching gets fresh output, by design), the full message list (image bytes hashed via SHA-256), tool schemas in registration order, the merged effective settings (with `Enum`s normalised to `.value`), and the `stop` list. Storage layout: `<cache_dir>/<provider>/<model_id>/<sha256>.json`, where each file holds the canonical fingerprint (for inspection) plus the serialised response.

`cache_dir` cascades convo > model > provider exactly like `log_dir`. Pass `cache_dir=False` at any scope to disable for that scope; a lower scope can re-enable with its own `cache_dir`. `cache_mode` cascades the same way (default `"read_write"`):

- `"read_write"` — read on hit, call provider on miss and write the result.
- `"read_only"` — read on hit, call provider on miss but do not write.
- `"record_only"` — always call provider, write the result (overwrites any existing entry).
- `"replay_only"` — read on hit, raise `CacheMissError` on miss. No provider call. Use this in CI to guarantee no accidental API spend.

Streaming hits are reconstructed by `cache.replay_stream(resp)`: thinking blocks (and their deltas) come first, then a single `text_delta` carrying the full text, then one event per tool call, then a terminal `done` event with the cached `usage` and `finish_reason`. This is enough to drive any consumer that handles real streams; faithful chunk-timing replay is intentionally not attempted. Hits skip `_finalize_stream` entirely so the cached blocks land in history in their original order.

### Capability gating

Every provider declares `SUPPORTS: frozenset[str]` listing the knob names it accepts. Setting an unsupported knob at any layer raises `UnsupportedFeature`. Use `provider.is_available("temperature")` / `model.is_available(...)` / `convo.is_available(...)` and `get_capabilities()` to query (returns plain string sets). Never catch `UnsupportedFeature` to branch — query first.

### Conversation lifecycle

- Construction: `model.new_conversation(name=..., system_blocks=..., tools=..., log_dir=..., log_path=..., **defaults)`. Everything is set here, immutably. `name` defaults to `convo-<8hex>` and doubles as the log filename.
- `add_user_message` / `add_assistant_message` / `add_tool_result` mutate history; `send` / `asend` / `stream` / `astream` are strict single round-trips.
- Tool calls in a response are returned to the caller. Dispatch them yourself (or via `helpers.run_bound_tools`) and append results before the next call.
- `snapshot()` returns an opaque token capturing history; `rollback(snap)` restores it. `clone(*, name=None, log_dir=None, log_path=None, cache_dir=None, cache_mode=None)` deep-copies everything into a fresh conversation that resolves its log path and response-cache settings through the same cascade as a fresh `new_conversation`.

### Provider quirks

- **Anthropic**: System messages are extracted into the separate `system` parameter; `cache=True` on a `SystemBlock` emits `cache_control: ephemeral`. `auto_cache_last_user=True` adds the same marker to the last user block. `cache_ttl` (an `EphemeralCacheTTL` value or `"5m"` / `"1h"`) controls the TTL on every emitted cache_control block; default is the API default (5m). Uses `stop_sequences` (not `stop`). Exports an `AnthropicModel` enum (`from llmfacade.providers.anthropic import AnthropicModel`) listing the current generation only — `OPUS_4_7`, `SONNET_4_6`, `HAIKU_4_5` — each carrying canonical model id and capability metadata. Passing a member to `new_model` auto-applies both. Passing a raw string opts out: full SUPPORTS is used and the caller is responsible for `capability_override=` if needed (e.g. for deprecated 3.x models that lack `thinking`). An explicit `capability_override=` always wins over an enum's default. The enum is a per-release snapshot — bump it when adding new generations.
- **OpenAI**: Accepts `org_id` as a constructor identity arg (passed to the SDK client). Supports `output_format` (JSON mode). Does not currently expose `thinking` / `top_k`.
- **Google (Gemini)**: Converts `"assistant"` role to `"model"`; system messages go to `system_instruction`. Registered under both `"google"` and `"gemini"`. Tool result `function_response.name` is taken from the stored `name` on the `ToolResultBlock` (or looked up via the prior `ToolUseBlock.id` map).
- **llama.cpp (`llamacpp`)**: Talks to a `llama-server` (or `llama-swap`) over its OpenAI-compat HTTP endpoint. No auth. Two modes, decided by the presence of `base_url` at provider construction:
  - **External** (`base_url=...`): identical to phase-1 behaviour. `base_url` points at the OpenAI-compat root, e.g. `http://localhost:8080/v1`; the introspection side channel strips a trailing `/v1` to derive the bare server URL. Server-launch knobs (KV cache quantization via `--cache-type-k/v`, slot persistence via `--slot-save-path`, context size via `--ctx-size`, GPU offload) are CLI flags on `llama-server`, not facade settings — to vary them, run multiple servers (or put llama-swap in front). Passing any `LAUNCH_KNOBS` value (or `name=`) here raises `UnsupportedFeature`.
  - **Managed** (no `base_url`): the provider owns a `llama-swap` subprocess (lazily spawned on first `send`) and the YAML it consumes. `new_model(gguf=..., name=...)` registers a launch entry; per-model launch knobs cascade `provider < model`. `LAUNCH_KNOBS` (defined in `settings.py` alongside `RUNTIME_KNOBS`): `gguf`, `context_size`, `cache_type_k`, `cache_type_v`, `n_gpu_layers`, `parallel`, `slot_save_path`, `ttl`, `extra_args`. Provider-level constructor accepts `llmfacade_dir=` (default `./.llmfacade/`) and `default_ttl=`. Model id is `<gguf-stem>-<hash8>` (canonical-JSON of the launch config) or the explicit `name=`. Adds llama-swap-native methods (sync + async): `running()`, `unload(model_id)`, `unload_all()`, plus `shutdown()` for explicit teardown. Shutdown defense in depth: OS kill-on-parent-death (Win32 Job Object / Linux `prctl(PR_SET_PDEATHSIG)`), `atexit`+signal handlers, and a PID-file sweep on the next start. The `<llmfacade_dir>` layout is `swap.yaml` (regenerated when entries change), `swap.pid` (PID + port + session UUID), `logs/llamacpp-swap.log`.
  - **Both modes**: uses the OpenAI Python SDK for chat transport (so `tool_choice` works the same as on OpenAI), routing the llama.cpp-specific samplers `top_k` / `min_p` / `repeat_penalty` through the SDK's `extra_body=` argument verbatim. Introspection methods (sync + async): `health()`, `slots()`, `save_slot(id_slot, filename)`, `restore_slot(id_slot, filename)`, `erase_slot(id_slot)`. `count_tokens()` calls `POST /tokenize` and falls back to `chars/4` on connection error. Supports first-class `min_p`. Does NOT advertise `context_size` or `keep_alive` as runtime knobs — both are server-launch concerns and live in `LAUNCH_KNOBS` (managed mode only).
  - **Introspection routing in managed mode**: llama-swap does not proxy llama-server-native paths via the bare URL, but it does expose `/upstream/<model_id>/<arbitrary-path>` (with on-demand model load), so we route through that. Every per-backend introspection method on the provider (`health`, `slots`, `save_slot`, `restore_slot`, `erase_slot`, plus `count_tokens` via its existing `model_id=`) takes an optional `model: str | None = None` kwarg. In managed mode: `model` is required when more than one entry is registered; with exactly one entry the resolver infers it; URL-quotes slashes (e.g. `Qwen/Qwen2.5-3B`) so llama-swap parses the model id correctly. In external mode the `model` kwarg is silently ignored — bare `llama-server` has no model routing. `Model` exposes mirror methods (`model.health()`, `model.slots()`, `model.save_slot(...)`, etc., sync + async) that auto-bind `self._model_id` — same precedent as `Model.count_tokens()`. Special case: `provider.health()` with no `model=` in managed mode hits the swap's own `/health` (which returns plain text `"OK"`) and normalises the result to `{"status": "ok"}`; pass `model=` (or use `Model.health()`) to get the per-backend JSON. `count_tokens` silently falls back to chars/4 if the routing helper raises (no models registered) so cache-summary logging never blocks. `running()` / `unload()` / `unload_all()` are llama-swap-native and don't take `model=`.

## Adding a new provider

1. Create `src/llmfacade/providers/<name>.py` with a `class <Name>Provider(Provider)`.
2. Set `NAME`, `API_KEY_ENV`, and a `SUPPORTS: frozenset[str]` of knob names.
3. Implement `_init_client()` and `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw`. Each receives a `CompletionRequest` (model, messages, system_blocks, tools, stop, plus a single merged `settings: dict[str, Any]` and `settings_source: dict[str, str]`). Generation knobs — including `tool_choice` — live in `settings`; read them via `req.settings.get("tool_choice", "auto")`.
4. Register `(module_path, class_name)` in `PROVIDER_REGISTRY`.
5. Add an optional dependency block to `pyproject.toml`.
6. Map SDK errors to facade exceptions (`AuthenticationError`, `RateLimitError`, `ProviderError`).

If your provider needs extra identity args (e.g. `org_id`), add them to a `__init__` override that pops them before calling `super().__init__(**kwargs)`. The base accepts `manager`, `api_key`, `base_url`, plus the 15 generation kwargs (gated by `SUPPORTS`).

`SUPPORTS` carries two pure capability flags alongside the runtime knobs: `"tools"` (the provider can route a tool list — without it, `Conversation(tools=[...])` raises `UnsupportedFeature` at construction) and `"tool_choice"` (forced selection beyond `"auto"` is supported). The two are orthogonal: a provider may declare `"tools"` without `"tool_choice"` if its API has no forced-selection mode. For models that don't support tool calling at all, callers narrow with `capability_override=provider.SUPPORTS - {"tools"}` on `new_model()`.

## Code Style

- Ruff with rules `E, F, I, UP, B, SIM`; line length 99.
- All wire-format models use `@dataclass(frozen=True, slots=True)`.
- Full type annotations throughout.
- src/ layout - imports as `from llmfacade import ...`.
- Naming: snake_case throughout. Public lifecycle methods (`new_provider`, `new_model`, `new_conversation`, `send`, `stream`, `snapshot`, `rollback`, `clone`, `add_user_message`, etc.) and helper module functions are all snake_case. Class names and dataclass field names follow the usual Python conventions.

## Future work

- **Google `cachedContents` integration.** Gemini supports prompt caching but via a separate resource API: `client.caches.create(...)` returns a named cache handle that subsequent `generate_content` calls reference via `cached_content=<handle>`. This is a caller-managed lifecycle (create/list/delete + TTL) rather than a per-request `cache_control` marker, so the Anthropic-style `auto_cache_tools` / `auto_cache_last_user` knobs don't translate. Designing a facade-level surface for it (probably a `Provider.create_cache(...)` returning a handle, plus a `cached_content=` argument on `Conversation`) is its own piece of work and is deferred. Until it lands, callers running tool-heavy workloads against Gemini cannot get the equivalent of Anthropic's tool-array caching.
- **OpenAI / llamacpp tool caching — intentionally not implemented.** OpenAI auto-caches prompt prefixes above ~1024 tokens transparently (no per-request marker, no knob to flip), and llama-server's prompt cache is fully internal (the server reuses overlapping prefixes per slot and there's no client-facing `auto_cache_tools` semantic to expose). Adding `auto_cache_tools` to either would be a misleading no-op. This is documented here so the absence is understood as deliberate, not an oversight.
