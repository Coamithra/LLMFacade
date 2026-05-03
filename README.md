# llmfacade

A lean, unified Python interface to multiple LLM providers (Anthropic, OpenAI, Google Gemini, llama.cpp).

- **Zero required runtime dependencies.** Provider SDKs are lazy-loaded only when used.
- **Capability-aware settings.** Each provider/model declares what it supports; unsupported knobs raise a clear error instead of being silently dropped.
- **Same API for sync, async, and streaming.** Plus first-class tool use, multimodal input, prompt caching, and conversation snapshot/rollback.
- Python 3.10+.

## Install

```bash
pip install llmfacade[anthropic]              # one provider
pip install llmfacade[anthropic,openai]       # several
pip install llmfacade[all]                    # everything
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

There is no `Start()` step — conversations are usable immediately after construction.

## Architecture

The library has a four-level hierarchy. Each level owns its own concerns and spawns the next:

```
LLM            manager: shared api_keys; LLM.default() is a process-wide singleton
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

Every level exposes `is_available(knob)` and `get_capabilities()` so you can branch on what the current provider/model actually supports.

## Settings cascade

All generation knobs are plain string kwargs (`temperature`, `max_tokens`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `effort`, `thinking`, `output_format`, `auto_cache_last_user`, `cache_ttl`, `user_metadata`, `beta_headers`, `tool_choice`). Set defaults at any of four scopes:

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

Configuration is constructor-only: identity (api_key, base_url, model_id, system_blocks, tools, log_path) and defaults are supplied at construction and never change after.

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
# Streaming
for ev in chat.stream("Tell me a story."):
    if ev.text_delta:
        print(ev.text_delta, end="", flush=True)

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
```

## Logging

Pass `log_path=` at conversation construction to write a JSONL log of every request and response. The first record is a `settings` header listing every effective knob, its value, and which scope (`provider`/`model`/`convo`) supplied it. Subsequent records are tight: `request` carries only `overrides` and `new_messages` (delta since last log); `response` carries the assistant content and a `cache_summary` block (cache_read_tokens, cache_creation_tokens, hit_ratio, etc.).

```python
chat = model.new_conversation(log_path="./logs/session.jsonl")
```

## Providers

| Provider | Install extra | API key env | Notes |
|---|---|---|---|
| Anthropic | `[anthropic]` | `ANTHROPIC_API_KEY` | Extended thinking, prompt caching, system blocks with `cache=True`, `cache_ttl`. Exports `AnthropicModel` enum (`OPUS_4_7`, `SONNET_4_6`, `HAIKU_4_5`) — passing a member to `new_model` auto-applies model id and capability metadata. |
| OpenAI    | `[openai]`    | `OPENAI_API_KEY`    | `output_format` (JSON mode); `org_id` constructor arg. |
| Google Gemini | `[google]` | `GOOGLE_API_KEY`   | Registered as both `"google"` and `"gemini"`. |
| llama.cpp | `[llamacpp]`  | (none)              | Two modes. **External**: `base_url=` points at a `llama-server` (or `llama-swap`) you run yourself — ctx-size, KV quantization, slot-save dir are server-launch flags. **Managed**: omit `base_url=` and the provider owns a `llama-swap` subprocess; pass launch knobs (`gguf=`, `context_size=`, `cache_type_k=`, etc.) at `new_model` and the YAML is generated for you. First-class `min_p`; `top_k`/`min_p`/`repeat_penalty` ride the SDK's `extra_body=`. Introspection: `health()`, `slots()`, `save_slot()`, `restore_slot()`, `erase_slot()`. Managed-mode-only: `running()`, `unload()`, `unload_all()`, `shutdown()`. `count_tokens()` calls the server's `/tokenize`. |

### Installing the binaries

The `llamacpp` provider needs `llama-server` (always) and `llama-swap` (only for managed mode) on `PATH`. They are not pulled in by `pip install llmfacade[llamacpp]`, since they are native binaries.

- **`llama-server`** ships in the [llama.cpp release ZIPs](https://github.com/ggml-org/llama.cpp/releases). Pick the build matching your hardware:
  - NVIDIA → `llama-*-bin-win-cuda-13.1-x64.zip` plus the matching `cudart-llama-bin-win-cuda-13.1-x64.zip` (extract both into the same folder so the CUDA runtime DLLs sit next to `llama-server.exe`).
  - AMD / Intel / iGPU / cross-vendor → `llama-*-bin-win-vulkan-x64.zip`. On Windows, `winget install llama.cpp` ships the Vulkan build.
  - macOS → `brew install llama.cpp`.
  - Linux → build from source or use a distro package.
- **`llama-swap`** ships in the [llama-swap release ZIPs](https://github.com/mostlygeek/llama-swap/releases) (one binary per platform). Or `go install github.com/mostlygeek/llama-swap@latest` if you have Go.

Verify with `llama-server --version` and `llama-swap --version`. On the CUDA build the version banner will list the detected GPU; if you see `CPU` instead, the cudart DLLs aren't being found.

### Running llama-server (external mode)

The `llamacpp` provider works against a `llama-server` (from llama.cpp) that you run yourself. A canonical invocation:

```
llama-server -m models/qwen2.5-3b-q4.gguf --host 0.0.0.0 --port 8080 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --slot-save-path ./slot_cache --metrics
```

```python
provider = llm.new_provider("llamacpp", base_url="http://localhost:8080/v1")
model = provider.new_model("qwen2.5-3b-instruct-q4_k_m", max_tokens=512)
```

Knobs that affect the loaded model — context size, KV-cache quantization, GPU offload — live on the `llama-server` CLI, not in the LLMFacade settings cascade. To run multiple configurations side-by-side, launch one `llama-server` per config on different ports and instantiate one `LlamaCppServerProvider` per port.

### Managed mode (zero-YAML llama-swap supervision)

Omit `base_url=` and the provider owns a `llama-swap` subprocess that supervises one or more `llama-server` instances on demand. You never edit YAML — pass launch knobs at `new_model` and the supervisor materialises everything on the first `convo.send()`.

```python
provider = llm.new_provider(
    "llamacpp",
    n_gpu_layers=32,                     # provider-level launch default
)
fast    = provider.new_model(name="fast",    gguf="models/qwen-3b-q4.gguf",
                             context_size=8192,  cache_type_k="f16")
quality = provider.new_model(name="quality", gguf="models/qwen-3b-q4.gguf",
                             context_size=32768, cache_type_k="q8_0")

convo = quality.new_conversation()
print(convo.send("Hello").text)          # ← spawns llama-swap, loads `quality`

provider.running()                        # native llama-swap endpoint
provider.unload("quality")                # ditto
provider.shutdown()                       # explicit teardown (atexit also wired)
```

Prerequisites: both `llama-swap` and `llama-server` on `PATH` (see [Installing the binaries](#installing-the-binaries) above).

Lifecycle:

* The supervisor lives under `<llmfacade_dir>/` (default `./.llmfacade/`). It contains `swap.yaml`, `swap.pid`, and `logs/llamacpp-swap.log`.
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

## Development

```bash
pip install -e ".[dev,all]"

ruff check src/
ruff format src/
pytest
```
