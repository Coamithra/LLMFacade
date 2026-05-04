# llmfacade

A lean, unified Python interface to multiple LLM providers (Anthropic, OpenAI, Google Gemini, llama.cpp).

> **Just want to see code?** → [**EXAMPLES.md**](EXAMPLES.md) — runnable samples covering multi-provider, tools, streaming, snapshot/rollback, deterministic replay, and local models.

- **Zero required Python runtime dependencies.** Provider SDKs are optional extras, lazy-loaded only when used. (Local models via the `llamacpp` provider additionally need the native `llama-server` and `llama-swap` binaries on `PATH` — see [Installing the llama.cpp binaries](#installing-the-llamacpp-binaries).)
- **Capability-aware settings.** Each provider/model declares what it supports; unsupported knobs raise a clear error instead of being silently dropped.
- **Same API for sync, async, and streaming.** Plus first-class tool use, multimodal input, prompt caching, and conversation snapshot/rollback.
- **Built-in JSONL+HTML logging** of every request/response, with token-aware cache summaries.
- **Deterministic response cache** for replaying recorded sessions in tests with zero API spend.
- **Local token counting** for OpenAI, Google, and llama.cpp.
- Python 3.10+.

## Install

```bash
pip install llmfacade[anthropic]              # one provider
pip install llmfacade[anthropic,openai]       # several
pip install llmfacade[all]                    # everything

pip install llmfacade[all,tokenizers]         # adds tiktoken + sentencepiece for local token counting
```

## Quickstart

```python
from llmfacade import LLM

provider = LLM.default().new_provider("anthropic")    # reads ANTHROPIC_API_KEY
model    = provider.new_model("claude-sonnet-4-6")
chat     = model.new_conversation(system_blocks="You are a terse assistant.")

resp = chat.send("What is 2 + 2?")
print(resp.text)
```

## Architecture

The library has a four-level hierarchy. Each level owns its own concerns and spawns the next:

```
LLM            manager: shared api_keys, log root, cache root; LLM.default() is a process-wide singleton
 -> Provider   identity (api_key, base_url) + SDK client + generation defaults
   -> Model    a model_id bound to a provider, with optional model-level defaults
     -> Conversation   stateful session: history, system blocks, tools, convo-level defaults
```

```python
from llmfacade import LLM

mgr      = LLM(api_keys={"anthropic": "sk-..."})
provider = mgr.new_provider("anthropic", temperature=0.7)
model    = provider.new_model("claude-sonnet-4-6", max_tokens=2048)
chat     = model.new_conversation()
```

Every level exposes `is_available(knob)` and `get_capabilities()` so you can branch on what the current provider/model actually supports. Capabilities also include the pure flags `"tools"` (the provider can route a tool list at all) and `"tool_choice"` (forced selection beyond `"auto"` is supported); the two are orthogonal.

## Settings cascade

All generation knobs are plain string kwargs — currently `temperature`, `max_tokens`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `effort`, `thinking`, `output_format`, `auto_cache_last_user`, `auto_cache_tools`, `cache_ttl`, `user_metadata`, `beta_headers`, and `tool_choice` (the canonical list lives in `llmfacade.RUNTIME_KNOBS`). Set defaults at any of four scopes:

```python
provider = mgr.new_provider("anthropic", temperature=0.7)        # provider-wide default
model    = provider.new_model("claude-opus-4-7", thinking=2048)  # narrows for this model
chat     = model.new_conversation(temperature=0.3)               # narrows for this convo
resp     = chat.send("Hello", max_tokens=128)                    # one-shot override
```

Precedence is `provider < model < convo < per_call` (later wins). Unknown kwarg names raise `TypeError`. Knobs not in the relevant layer's effective `SUPPORTS` raise `UnsupportedFeature` at construction:

```python
from llmfacade import UnsupportedFeature

if chat.is_available("auto_cache_last_user"):
    chat = model.new_conversation(auto_cache_last_user=True)

try:
    chat = model.new_conversation(thinking=2048)   # not on every model
except UnsupportedFeature as e:
    print(e)
```

Configuration is constructor-only: identity (api_key, base_url, model_id, system_blocks, tools, log_dir/log_path, cache_dir) and defaults are supplied at construction and never change after.

### Anthropic model enum

For Anthropic, you can pass an `AnthropicModel` enum member to `new_model` — it auto-applies the canonical model id and the matching capability metadata:

```python
from llmfacade.providers.anthropic import AnthropicModel

opus = provider.new_model(AnthropicModel.OPUS_4_7, max_tokens=4096)
```

The enum is a per-release snapshot of the current generation (`OPUS_4_7`, `SONNET_4_6`, `HAIKU_4_5`). Passing a raw string opts out — full `SUPPORTS` is used and you're responsible for `capability_override=` if needed (e.g. for older 3.x models that lack `thinking`).

## Tools

Decorate any function with `@tool`. The schema is generated from its signature, type hints, and docstring.

```python
from llmfacade import tool

@tool
def forge_item(item: str, material: str = "iron") -> str:
    """Forge an item out of a material. Returns a description string."""
    return f"You receive a {material} {item}."

chat = model.new_conversation(tools=[forge_item])

# One round-trip: model may return tool_calls.
resp = chat.send("Make me a sword.")
for call in resp.tool_calls:
    chat.add_tool_result(call.id, str(forge_item(**call.input)), name=call.name)
resp = chat.send()                # continue with the tool results
print(resp.text)
```

`send`/`stream` are exactly one provider round-trip. The library never auto-executes user code. For the common case — run every tool the model calls, send results back, repeat until done — use `llmfacade.helpers.run_to_completion`:

```python
from llmfacade import helpers

resp = helpers.run_to_completion(chat, "Make me a sword.")
print(resp.text)
```

`helpers.run_bound_tools(chat, resp)` is the lower-level building block: it dispatches tool calls whose name matches a `@tool` registered on the conversation. Because the helpers only use the public API, you can write your own (e.g. with approval prompts or parallel dispatch) without subclassing anything. Async equivalents: `arun_bound_tools`, `arun_to_completion`.

## Streaming, async, multimodal

```python
# Streaming — text, thinking (when supported), and tool-call deltas all flow through StreamEvent.
for ev in chat.stream("Tell me a story."):
    if ev.thinking_delta:
        pass                                     # extended-thinking chunk, if the model emits one
    if ev.text_delta:
        print(ev.text_delta, end="", flush=True)
    if ev.tool_call_delta:
        ...                                      # a fully-formed ToolCall

# Async
import asyncio
resp = asyncio.run(chat.asend("Briefly?"))

# Multimodal
from llmfacade import ImageBlock, TextBlock
chat.add_user_message(content=[
    TextBlock("What's in this image?"),
    ImageBlock.from_path("photo.png"),
])
resp = chat.send()
```

`stream` and `send` are both strict single round-trips with the same wire-format guard: if history contains a `tool_use` without a matching `tool_result`, both raise `ConversationStateError`.

## Snapshot / Rollback / Clone

```python
snap = chat.snapshot()
chat.send("[experiment]")
chat.rollback(snap)               # back to pre-experiment state

alt = chat.clone()                # independent copy with the same history & tools
alt2 = chat.clone(                # override any of these for the clone
    name="branch-b",
    log_dir="./logs/alt",
    cache_dir="./cache/alt",
    cache_mode="replay_only",
)
```

## Logging

Logging is **on by default**. Each `LLM` instance reserves a session-stamped subfolder `<log_dir>/llmfacade<YYYYMMDD-HHMMSS>/` (default base: `<cwd>/logs`); the directory is materialised lazily on first write. Each `Conversation`'s log file is `<run_dir>/<convo.name>.jsonl` plus an HTML sibling for at-a-glance review.

```python
mgr = LLM(log_dir="./logs", max_log_folders=10)   # retention: keeps the 10 most recent sessions
chat = model.new_conversation()                    # auto-named "convo-<8hex>"; log file uses that name
chat = model.new_conversation(name="planning")     # named explicitly
```

`log_dir` cascades convo > model > provider > manager. Pass `log_dir=False` at any layer to disable logging for that scope; a lower layer can re-enable by supplying its own `log_dir`. For one-off explicit-file control, `Conversation(log_path=Path(...))` bypasses the cascade entirely; `log_path=False` disables logging for that one convo.

The JSONL log starts with a single `settings` header listing every effective knob, its value, and which scope (`provider`/`model`/`convo`) supplied it — plus the system blocks and tool names. Subsequent entries are tight: `request` records carry only `overrides` (per-call kwargs) and `new_messages` (delta since last log); `response` records carry the assistant content and a `cache_summary` block (cache_read_tokens, cache_creation_tokens, hit_ratio, and an `approximate_messages_cached` index that maps cache reads back to a turn boundary).

## Response cache (deterministic replay)

Off by default. Set `cache_dir=<path>` at provider, model, or conversation scope to enable a filesystem-backed cache of `Response` objects. On a hit, no provider call is made — the stored response is returned (or replayed as a stream).

```python
provider = mgr.new_provider("anthropic", cache_dir="./cache")
chat = provider.new_model("claude-sonnet-4-6").new_conversation()

chat.send("Hello")                               # miss: provider call, response written
chat.send("Hello")                               # hit: replayed from disk, no API spend
```

The hash key covers every input that affects output: provider name, model id, system blocks (including `cache=True` markers — flipping caching gets fresh output, by design), the full message list (image bytes hashed via SHA-256), tool schemas in registration order, the merged effective settings, and the `stop` list. Files live under `<cache_dir>/<provider>/<model_id>/<sha256>.json`.

`cache_dir` cascades convo > model > provider; pass `cache_dir=False` at any scope to disable for that scope. `cache_mode` cascades the same way (default `"read_write"`):

| Mode | Hit | Miss |
|---|---|---|
| `"read_write"` (default) | replay | call provider, write |
| `"read_only"` | replay | call provider, do not write |
| `"record_only"` | always call provider, overwrite | always call provider, write |
| `"replay_only"` | replay | raise `CacheMissError` |

Use `replay_only` in CI to guarantee no accidental API spend. Streams are reconstructed from cached responses: thinking blocks first, then a single `text_delta` carrying the full text, then one event per tool call, then a terminal `done` event with the cached usage and finish reason.

## Token counting

Every provider implements `count_tokens(text, *, model_id=None)` and `tokenizer_name(model_id=None)`; `Model` exposes the same methods auto-bound to its model id.

```python
model.tokenizer_name()              # e.g. "tiktoken/cl100k_base", "sentencepiece/gemini-2.0", ...
model.count_tokens("hello world")   # local; never makes an external network call
```

Install the optional `tokenizers` extra (`pip install llmfacade[tokenizers]`) to enable tiktoken (OpenAI) and sentencepiece (Google). Anthropic has no offline tokenizer and returns `chars/4`; for exact counts, call `client.messages.count_tokens` via the SDK directly. llama.cpp uses the running server's `/tokenize` endpoint and falls back to `chars/4` on connection error (the call stays local — your own llama-server).

The same machinery powers the `cache_summary.approximate_messages_cached` field in the JSONL log: a turn-boundary table maps `cache_read_tokens` back to a message index exactly when a prior turn's recorded total matches, and falls back to a per-message walk via `count_tokens` otherwise.

## Providers

| Provider | Install extra | API key env | Notes |
|---|---|---|---|
| Anthropic | `[anthropic]` | `ANTHROPIC_API_KEY` | Extended thinking, prompt caching, system blocks with `cache=True`, `cache_ttl` (`EphemeralCacheTTL.FIVE_MINUTES` / `ONE_HOUR`), `auto_cache_last_user`, `auto_cache_tools`. Exports `AnthropicModel` enum (`OPUS_4_7`, `SONNET_4_6`, `HAIKU_4_5`). |
| OpenAI    | `[openai]`    | `OPENAI_API_KEY`    | `output_format` (JSON mode); `org_id` constructor arg. |
| Google Gemini | `[google]` | `GOOGLE_API_KEY`   | Registered as both `"google"` and `"gemini"`. |
| llama.cpp | `[llamacpp]`  | (none)              | Two modes (see below). First-class `min_p`; `top_k`/`min_p`/`repeat_penalty` ride the SDK's `extra_body=`. Introspection: `health()`, `slots()`, `save_slot()`, `restore_slot()`, `erase_slot()`. Managed-mode-only: `running()`, `unload()`, `unload_all()`, `shutdown()`. `count_tokens()` calls the server's `/tokenize`. |

### Installing the llama.cpp binaries

The `llamacpp` provider needs `llama-server` (always) and `llama-swap` (only for managed mode) on `PATH`. They are not pulled in by `pip install llmfacade[llamacpp]`, since they are native binaries.

- **`llama-server`** ships in the [llama.cpp release ZIPs](https://github.com/ggml-org/llama.cpp/releases). Pick the build matching your hardware:
  - NVIDIA → `llama-*-bin-win-cuda-13.1-x64.zip` plus the matching `cudart-llama-bin-win-cuda-13.1-x64.zip` (extract both into the same folder so the CUDA runtime DLLs sit next to `llama-server.exe`).
  - AMD / Intel / iGPU / cross-vendor → `llama-*-bin-win-vulkan-x64.zip`. On Windows, `winget install llama.cpp` ships the Vulkan build.
  - macOS → `brew install llama.cpp`.
  - Linux → build from source or use a distro package.
- **`llama-swap`** ships in the [llama-swap release ZIPs](https://github.com/mostlygeek/llama-swap/releases) (one binary per platform). Or `go install github.com/mostlygeek/llama-swap@latest` if you have Go.

Verify with `llama-server --version` and `llama-swap --version`. On the CUDA build the version banner will list the detected GPU; if you see `CPU` instead, the cudart DLLs aren't being found.

### llama.cpp — external mode

Pass `base_url=` to point the provider at a `llama-server` (or `llama-swap`) you run yourself:

```
llama-server -m models/qwen2.5-3b-q4.gguf --host 0.0.0.0 --port 8080 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --slot-save-path ./slot_cache --metrics
```

```python
provider = llm.new_provider("llamacpp", base_url="http://localhost:8080/v1")
model    = provider.new_model("qwen2.5-3b-instruct-q4_k_m", max_tokens=512)
```

Knobs that affect the loaded model — context size, KV-cache quantization, GPU offload — live on the `llama-server` CLI, not in the LLMFacade settings cascade. To run multiple configurations side-by-side, launch one `llama-server` per config on different ports and instantiate one provider per port.

### llama.cpp — managed mode (zero-YAML llama-swap supervision)

Omit `base_url=` and the provider owns a `llama-swap` subprocess that supervises one or more `llama-server` instances on demand. You never edit YAML — pass launch knobs at `new_model` and the supervisor materialises everything on the first `convo.send()`.

```python
provider = llm.new_provider("llamacpp", n_gpu_layers=32)            # provider-level launch default
fast    = provider.new_model(name="fast",    gguf="models/qwen-3b-q4.gguf",
                             context_size=8192,  cache_type_k="f16")
quality = provider.new_model(name="quality", gguf="models/qwen-3b-q4.gguf",
                             context_size=32768, cache_type_k="q8_0")

convo = quality.new_conversation()
print(convo.send("Hello").text)          # spawns llama-swap, loads `quality`

provider.running()                        # native llama-swap endpoint
provider.unload("quality")                # ditto
provider.shutdown()                       # explicit teardown (atexit also wired)
```

Launch knobs (the `llmfacade.settings.LAUNCH_KNOBS` set, valid only at provider/model scope in managed mode): `gguf`, `context_size`, `cache_type_k`, `cache_type_v`, `n_gpu_layers`, `parallel`, `slot_save_path`, `ttl`, `extra_args`. Provider-level constructor also accepts `llmfacade_dir=` (default `./.llmfacade/`) and `default_ttl=`.

Lifecycle:

* The supervisor lives under `<llmfacade_dir>/`. It contains `swap.yaml`, `swap.pid`, and `logs/llamacpp-swap.log`.
* First `send()` triggers a lazy spawn; subsequent `new_model` calls rewrite `swap.yaml` and llama-swap's `-watch-config` picks them up.
* Shutdown defense in depth: OS-level kill-on-parent-death (Windows Job Object / Linux `prctl(PR_SET_PDEATHSIG)`), Python `atexit` + `SIGINT`/`SIGTERM` handlers, and a PID-file sweep on the next start that reaps any orphan that survived a hard kill.
* macOS has no kernel-level kill-on-death, so a hard-killed Python may leave llama-swap alive briefly; the next `new_provider("llamacpp")` cleans it up via the PID-file sweep.

## Exceptions

All errors derive from `LLMError`:

- `AuthenticationError`, `RateLimitError`, `ProviderError`, `ModelNotFoundError`
- `ProviderNotInstalledError` — the SDK extra wasn't installed
- `UnsupportedFeature` — knob not supported by this provider/model
- `ConversationStateError` — history has unresolved `tool_use` blocks; append `tool_result`s before sending again
- `ToolIterationLimitError` — `helpers.run_to_completion` exceeded `max_iterations`
- `CacheMissError` — `cache_mode="replay_only"` and there was no cache hit

## Development

```bash
pip install -e ".[dev,all,tokenizers]"

ruff check src/
ruff format src/
pytest
```

Integration tests under `tests/integration/` hit real provider APIs and burn credits; they're gated behind `pytest -m integration` and skipped by default. See `CONTRIBUTING.md` for the full dev-loop expectations.
