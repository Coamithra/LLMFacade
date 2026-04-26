# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMFacade is a Python library providing a unified interface to multiple LLM providers (Anthropic, OpenAI, Google Gemini, Ollama). Zero required runtime dependencies - provider SDKs are lazy-loaded only when used. Python 3.10+.

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

## Architecture

Four-level hierarchy. Each level owns its own concerns and spawns the next:

```
LLM            manager: shared api_keys; LLM.default() is a process-wide singleton
 -> Provider   identity (api_key, base_url) + SDK client + generation defaults
   -> Model    a model_id bound to a provider, with optional model-level defaults
     -> Conversation   stateful session: history, system blocks, tools, convo-level defaults
```

**Configuration is constructor-only.** Identity (api_key, base_url, model_id, system_blocks, tools, log_path) is supplied at construction and never changes. Generation knobs (temperature, max_tokens, etc.) are accepted as kwargs at every layer and form a cascade (`provider < model < convo < per_call`); they are also immutable post-construction. There is no `Start()` step — conversations are usable immediately after construction.

Key files:

- `src/llmfacade/facade.py` — `LLM` manager; `new_provider(name, **kwargs)` dynamically imports a provider module via `PROVIDER_REGISTRY` and forwards kwargs.
- `src/llmfacade/provider.py` — `Provider` base class. Owns API-key resolution (override > manager dict > env var), `_init_client()`, the merged `CompletionRequest`, and the `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw` hooks. Also defines the `SystemBlock` dataclass and the cascade helpers `_validate_knobs` / `_filter_unsupported`.
- `src/llmfacade/model.py` — `Model` binds a `model_id` to a `Provider` and stores its own generation defaults. Supports a `capability_override` (used by Anthropic to drop `thinking` for non-thinking models).
- `src/llmfacade/conversation.py` — `Conversation` is the stateful API: history, system blocks (with optional `cache=True`), tools, JSONL logging, `snapshot`/`rollback`/`clone`. `send`/`asend`/`stream`/`astream` are strict single round-trips that accept the full set of generation-knob kwargs as per-call overrides. The wire-format invariant (every `tool_use` matched by a `tool_result`) is enforced at request time and raises `ConversationStateError` if violated.
- `src/llmfacade/helpers.py` — Optional convenience built on the public Conversation API. `run_bound_tools(convo, resp)` dispatches every tool call whose name matches a `@tool` registered on the conversation and appends results. `run_to_completion(convo, prompt)` is the agent loop (`send` -> dispatch -> `send` -> ...) with a `max_iterations` cap, raising `ToolIterationLimitError` on overflow. Async equivalents: `arun_bound_tools`, `arun_to_completion`. Helpers touch only the public surface — no underscore attributes.
- `src/llmfacade/settings.py` — `RUNTIME_KNOBS` frozenset (the 14 string knob names) plus the value enums `EffortLevel`, `OutputFormat`, `EphemeralCacheTTL` (`FIVE_MINUTES`, `ONE_HOUR`).
- `src/llmfacade/models.py` — Frozen dataclasses for the wire format: `Message`, `TextBlock`, `ImageBlock`, `ToolUseBlock`, `ToolResultBlock`, `ToolCall`, `Response`, `Usage`, `StreamEvent`. `ImageBlock` has `from_path` / `from_base64`.
- `src/llmfacade/tools.py` — `@tool` decorator that builds a JSON schema from a function's signature + type hints + docstring. Handles primitives, `Literal`, `Optional`/`Union`, `list[T]`, `dict`.
- `src/llmfacade/exceptions.py` — Hierarchy rooted at `LLMError`: `AuthenticationError`, `RateLimitError`, `ProviderError`, `ModelNotFoundError`, `ProviderNotInstalledError`, `UnsupportedFeature`, `ToolIterationLimitError`, `ConversationStateError`.
- `src/llmfacade/providers/__init__.py` — `PROVIDER_REGISTRY` mapping names to `(module_path, class_name)`. `"google"` and `"gemini"` both resolve to `GoogleProvider`.
- `src/llmfacade/providers/{anthropic,openai,google,ollama}.py` — Provider implementations.

### Settings cascade

Every generation knob can be defaulted at any of four scopes:

1. `Provider(... temperature=0.7)` — applies to every model and convo under this provider.
2. `provider.new_model("...", temperature=0.7)` — overrides the provider default for this model.
3. `model.new_conversation(temperature=0.7)` — overrides for this convo.
4. `convo.send("...", temperature=0.7)` — overrides for this single call.

Precedence on read is `provider < model < convo < per_call` (later wins). Unknown kwarg names raise `TypeError`. Knobs not in the relevant layer's effective `SUPPORTS` raise `UnsupportedFeature` at the layer they're set. If a higher-scope default is for something a downstream model doesn't honor (e.g. `thinking` set on the Anthropic provider but the convo binds to a non-thinking model), the cascade silently drops the value at request time and warns once per `(key, source, model)` tuple.

### Logging

When `log_path=` is passed at convo construction, the JSONL log starts with a single `settings` header record listing every effective knob, its value, and which scope (`provider`/`model`/`convo`) supplied it — plus the system blocks and tool names. Subsequent entries are tight: `request` records carry only `overrides` (per-call kwargs) and `new_messages` (delta since last log), and `response` records carry the assistant content and a `cache_summary` block.

### Capability gating

Every provider declares `SUPPORTS: frozenset[str]` listing the knob names it accepts. Setting an unsupported knob at any layer raises `UnsupportedFeature`. Use `provider.is_available("temperature")` / `model.is_available(...)` / `convo.is_available(...)` and `get_capabilities()` to query (returns plain string sets). Never catch `UnsupportedFeature` to branch — query first.

### Conversation lifecycle

- Construction: `model.new_conversation(system_blocks=..., tools=..., log_path=..., **defaults)`. Everything is set here, immutably.
- `add_user_message` / `add_assistant_message` / `add_tool_result` mutate history; `send` / `asend` / `stream` / `astream` are strict single round-trips.
- Tool calls in a response are returned to the caller. Dispatch them yourself (or via `helpers.run_bound_tools`) and append results before the next call.
- `snapshot()` returns an opaque token capturing history; `rollback(snap)` restores it. `clone(*, name=None, log_path=None)` deep-copies everything into a fresh conversation that may have its own log path.

### Provider quirks

- **Anthropic**: System messages are extracted into the separate `system` parameter; `cache=True` on a `SystemBlock` emits `cache_control: ephemeral`. `auto_cache_last_user=True` adds the same marker to the last user block. `cache_ttl` (an `EphemeralCacheTTL` value or `"5m"` / `"1h"`) controls the TTL on every emitted cache_control block; default is the API default (5m). Uses `stop_sequences` (not `stop`). Models matching `_NO_THINKING_MODELS` get a capability override that drops `thinking`.
- **OpenAI**: Accepts `org_id` as a constructor identity arg (passed to the SDK client). Supports `output_format` (JSON mode). Does not currently expose `thinking` / `top_k`.
- **Google (Gemini)**: Converts `"assistant"` role to `"model"`; system messages go to `system_instruction`. Registered under both `"google"` and `"gemini"`. Tool result `function_response.name` is taken from the stored `name` on the `ToolResultBlock` (or looked up via the prior `ToolUseBlock.id` map).
- **Ollama**: No auth needed. `context_size` maps to `num_ctx`; `max_tokens` maps to `num_predict` (Ollama default 128, facade default 1024). Warns on silent context truncation at 95% of `num_ctx`. `base_url` for remote hosts; `keep_alive` controls model unload.

## Adding a new provider

1. Create `src/llmfacade/providers/<name>.py` with a `class <Name>Provider(Provider)`.
2. Set `NAME`, `API_KEY_ENV`, and a `SUPPORTS: frozenset[str]` of knob names.
3. Implement `_init_client()` and `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw`. Each receives a `CompletionRequest` (model, messages, system_blocks, tools, tool_choice, stop, plus a single merged `settings: dict[str, Any]` and `settings_source: dict[str, str]`).
4. Register `(module_path, class_name)` in `PROVIDER_REGISTRY`.
5. Add an optional dependency block to `pyproject.toml`.
6. Map SDK errors to facade exceptions (`AuthenticationError`, `RateLimitError`, `ProviderError`).

If your provider needs extra identity args (e.g. `org_id`), add them to a `__init__` override that pops them before calling `super().__init__(**kwargs)`. The base accepts `manager`, `api_key`, `base_url`, plus the 14 generation kwargs (gated by `SUPPORTS`).

## Code Style

- Ruff with rules `E, F, I, UP, B, SIM`; line length 99.
- All wire-format models use `@dataclass(frozen=True, slots=True)`.
- Full type annotations throughout.
- src/ layout - imports as `from llmfacade import ...`.
- Naming: snake_case throughout. Public lifecycle methods (`new_provider`, `new_model`, `new_conversation`, `send`, `stream`, `snapshot`, `rollback`, `clone`, `add_user_message`, etc.) and helper module functions are all snake_case. Class names and dataclass field names follow the usual Python conventions.
