# llmfacade

A lean, unified Python interface to multiple LLM providers (Anthropic, OpenAI, Google Gemini, Ollama).

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

provider = LLM.default().NewProvider("anthropic")    # reads ANTHROPIC_API_KEY
model    = provider.NewModel("claude-sonnet-4-6")
chat     = model.NewConversation()

chat.AddSystemBlock("You are a terse assistant.")
chat.Start()

resp = chat.Complete("What is 2 + 2?")
print(resp.text)
```

## Architecture

The library has a four-level hierarchy. Each level owns its own concerns and spawns the next:

```
LLM           manager: shared api_keys, log_dir
 -> Provider  auth + SDK client + per-provider knobs (BaseURL, BetaHeaders, ...)
   -> Model   a model_id bound to a Provider + per-model knobs (TopP, Thinking, ...)
     -> Conversation   stateful session: history, system blocks, tools, per-call settings
```

```python
from llmfacade import LLM

mgr      = LLM(api_keys={"anthropic": "sk-..."}, log_dir="./logs")
provider = mgr.NewProvider("anthropic")
model    = provider.NewModel("claude-sonnet-4-6")
chat     = model.NewConversation(name="dnd-session")
```

Every level exposes `isAvailable(setting)` and `getCapabilities()` so you can branch on what the current provider/model actually supports.

## Settings

Settings are enums grouped by where they live:

| Enum | Lives on | Examples |
|---|---|---|
| `ProviderSettings` | Provider | `BaseURL`, `OrgID`, `BetaHeaders`, `KeepAlive` |
| `Settings`         | Model    | `ContextSize`, `DefaultMaxTokens`, `DefaultTemperature`, `TopP`, `TopK`, `RepeatPenalty`, `Effort`, `Thinking` |
| `ConvoSettings`    | Conversation | `AutoCacheLastUser`, `UserMetadata`, `OutputFormat` |

Set a value via the `.settings` facade. Unsupported settings raise `UnsupportedFeature`:

```python
from llmfacade import Settings, ConvoSettings, UnsupportedFeature

model.settings.set(Settings.DefaultMaxTokens, 2048)

if chat.isAvailable(ConvoSettings.AutoCacheLastUser):
    chat.settings.set(ConvoSettings.AutoCacheLastUser, True)

try:
    chat.settings.set(Settings.Thinking, 2048)   # not on every model
except UnsupportedFeature as e:
    print(e)
```

Conversation settings lock when you call `Start()`. After that, only per-call overrides on `Complete`/`Stream` are allowed.

## Tools

Decorate any function with `@tool`. The schema is generated from its signature and docstring.

```python
from llmfacade import tool

@tool
def forge_item(item: str, material: str = "iron") -> str:
    """Forge an item out of a material. Returns a description string."""
    return f"You receive a {material} {item}."

chat.AddTool(forge_item)
chat.Start()

resp = chat.Complete("Make me a sword.")     # tool runs automatically
print(resp.text)
```

By default `Complete` runs the full tool loop: model -> tool calls -> dispatch -> tool results -> model, repeating until the model returns no more tool calls. Pass `auto_tools=False` to inspect `resp.tool_calls` and run them yourself.

## Streaming, async, multimodal

```python
# Streaming
for ev in chat.Stream("Tell me a story."):
    if ev.text_delta:
        print(ev.text_delta, end="", flush=True)

# Async
import asyncio
resp = asyncio.run(chat.aComplete("Briefly?"))

# Multimodal
from llmfacade import ImageBlock, TextBlock
chat.AddUserMessage(content=[
    TextBlock("What's in this image?"),
    ImageBlock.from_path("photo.png"),
])
resp = chat.Complete()
```

## Snapshot / Rollback / Clone

```python
snap = chat.Snapshot()
chat.Complete("[experiment]")
chat.Rollback(snap)               # back to pre-experiment state

alt = chat.Clone()                # independent copy with the same history & tools
alt.Start()
```

## Logging

```python
chat.SetLogging("./logs/session.jsonl")   # JSONL of every request and response
```

The manager's `log_dir` provides a default root if you prefer to set it once.

## Providers

| Provider | Install extra | API key env | Notes |
|---|---|---|---|
| Anthropic | `[anthropic]` | `ANTHROPIC_API_KEY` | Extended thinking, prompt caching, system blocks with `cache=True` |
| OpenAI    | `[openai]`    | `OPENAI_API_KEY`    | JSON `OutputFormat`, `OrgID` |
| Google Gemini | `[google]` | `GOOGLE_API_KEY`   | Registered as both `"google"` and `"gemini"` |
| Ollama    | `[ollama]`    | (none)              | `BaseURL` for remote, `ContextSize` -> `num_ctx`, warns on silent context truncation |

## Exceptions

All errors derive from `LLMError`:

- `AuthenticationError`, `RateLimitError`, `ProviderError`, `ModelNotFoundError`
- `ProviderNotInstalledError` - the SDK extra wasn't installed
- `UnsupportedFeature` - setting not supported by this provider/model
- `NotStartedError` - operation needs `Conversation.Start()`
- `SettingsLockedError` - tried to mutate settings after `Start()`

## Development

```bash
pip install -e ".[dev,all]"

ruff check src/
ruff format src/
pytest
```
