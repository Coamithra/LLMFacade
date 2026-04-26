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
pytest tests/test_conversation.py::test_complete_round_trip   # single test
```

## Architecture

Four-level hierarchy. Each level owns its own concerns and spawns the next:

```
LLM            manager: shared api_keys, log_dir; LLM.default() is a process-wide singleton
 -> Provider   auth + SDK client + per-provider knobs
   -> Model    a model_id bound to a provider, with model-level settings
     -> Conversation   stateful session: history, system blocks, tools, per-call settings
```

Key files:

- `src/llmfacade/facade.py` - `LLM` manager; `NewProvider()` dynamically imports a provider module via `PROVIDER_REGISTRY` and instantiates it.
- `src/llmfacade/provider.py` - `Provider` base class. Owns API-key resolution (override > manager dict > env var), `_init_client()`, and the `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw` hooks subclasses implement. Also defines `_SettingsFacade`, the capability-aware settings store reused at every level.
- `src/llmfacade/model.py` - `Model` binds a `model_id` to a `Provider` and gets its own settings facade. Supports a `capability_override` (used by Anthropic to drop `Settings.Thinking` for non-thinking models).
- `src/llmfacade/conversation.py` - `Conversation` is the stateful API: history, system blocks (with optional `cache=True`), tools, JSONL logging, `Snapshot`/`Rollback`/`Clone`. Lifecycle: configure -> `Start()` (locks settings) -> `Complete`/`aComplete`/`Stream`/`aStream`. `Complete` runs the full tool-use loop by default; `auto_tools=False` returns control to the caller.
- `src/llmfacade/settings.py` - Three enums: `ProviderSettings` (BaseURL, OrgID, BetaHeaders, KeepAlive), `Settings` (ContextSize, DefaultMaxTokens, DefaultTemperature, TopP, TopK, RepeatPenalty, Effort, Thinking), `ConvoSettings` (AutoCacheLastUser, UserMetadata, OutputFormat). Plus `EffortLevel` and `OutputFormat` value enums. `AnySetting = ProviderSettings | Settings | ConvoSettings`.
- `src/llmfacade/models.py` - Frozen dataclasses for the wire format: `Message`, `TextBlock`, `ImageBlock`, `ToolUseBlock`, `ToolResultBlock`, `ToolCall`, `Response`, `Usage`, `StreamEvent`. `ImageBlock` has `from_path` / `from_base64`.
- `src/llmfacade/tools.py` - `@tool` decorator that builds a JSON schema from a function's signature + type hints + docstring. Handles primitives, `Literal`, `Optional`/`Union`, `list[T]`, `dict`.
- `src/llmfacade/exceptions.py` - Hierarchy rooted at `LLMError`: `AuthenticationError`, `RateLimitError`, `ProviderError`, `ModelNotFoundError`, `ProviderNotInstalledError`, `UnsupportedFeature`, `NotStartedError`, `SettingsLockedError`.
- `src/llmfacade/providers/__init__.py` - `PROVIDER_REGISTRY` mapping names to `(module_path, class_name)`. `"google"` and `"gemini"` both resolve to `GoogleProvider`.
- `src/llmfacade/providers/{anthropic,openai,google,ollama}.py` - Provider implementations.

### Capability gating

Every settings facade enforces a `SUPPORTS: frozenset[AnySetting]` declared on the provider class. Setting an unsupported value raises `UnsupportedFeature`. Per-call kwargs on `Complete`/`Stream` (`max_tokens`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `effort`) are validated against `Model.isAvailable` before the call. Always check `isAvailable` / `getCapabilities` rather than catching the exception when branching is the goal.

### Conversation lifecycle

- Pre-`Start()`: add system blocks, tools, set `SetLogging`, mutate settings.
- `Start()`: locks settings; `AddUserMessage` / `AddAssistantMessage` / `AddToolResult` / `Complete` / `Stream` become legal.
- `Snapshot()` returns an opaque token capturing history + system blocks; `Rollback(snap)` restores them. `Clone()` deep-copies everything into a fresh, unstarted conversation.

### Provider quirks

- **Anthropic**: System messages are extracted into the separate `system` parameter; `cache=True` on a system block emits `cache_control: ephemeral`. `ConvoSettings.AutoCacheLastUser` adds the same marker to the last user block. Uses `stop_sequences` (not `stop`). Models matching `_NO_THINKING_MODELS` get a capability override that drops `Settings.Thinking`.
- **OpenAI**: Supports `ProviderSettings.OrgID` and `ConvoSettings.OutputFormat` (JSON mode). Does not currently expose `Thinking` / `TopK`.
- **Google (Gemini)**: Converts `"assistant"` role to `"model"`; system messages go to `system_instruction`. Registered under both `"google"` and `"gemini"`.
- **Ollama**: No auth needed. `Settings.ContextSize` maps to `num_ctx`; `max_tokens` maps to `num_predict` (Ollama default 128, facade default 1024). Warns on silent context truncation at 95% of `num_ctx`. `ProviderSettings.BaseURL` for remote hosts; `KeepAlive` controls model unload.

## Adding a new provider

1. Create `src/llmfacade/providers/<name>.py` with a `class <Name>Provider(Provider)`.
2. Set `NAME`, `API_KEY_ENV`, and a `SUPPORTS` frozenset.
3. Implement `_init_client()` and `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw`. Each receives the full call kwargs (model, messages, system_blocks, tools, tool_choice, max_tokens, temperature, stop, plus the four settings dicts and `per_call_overrides`).
4. Register `(module_path, class_name)` in `PROVIDER_REGISTRY`.
5. Add an optional dependency block to `pyproject.toml`.
6. Map SDK errors to facade exceptions (`AuthenticationError`, `RateLimitError`, `ProviderError`).

## Code Style

- Ruff with rules `E, F, I, UP, B, SIM`; line length 99.
- All wire-format models use `@dataclass(frozen=True, slots=True)`.
- Full type annotations throughout.
- src/ layout - imports as `from llmfacade import ...`.
- Naming: public lifecycle methods on the main classes use PascalCase (`NewProvider`, `NewModel`, `NewConversation`, `Start`, `Complete`, `Stream`, `Snapshot`, `Rollback`, `Clone`, `AddTool`, `AddSystemBlock`, `SetLogging`). Internal helpers use snake_case.
