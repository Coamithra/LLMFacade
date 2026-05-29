# llmfacade examples

Runnable samples that exercise the headline features. Each one stands alone — copy, fill in an API key, run.

## One API, three providers

```python
from llmfacade import LLM

llm = LLM()
prompt = "Reply with a single word: rabbit, hare, or jackrabbit?"
for name, model_id in [
    ("anthropic", "claude-sonnet-4-6"),
    ("openai",    "gpt-4o-mini"),
    ("google",    "gemini-2.0-flash"),
]:
    chat = llm.new_provider(name).new_model(model_id).new_conversation()
    print(f"{name:10s} -> {chat.send(prompt).text}")
```

## Tools and agent loop

`@tool` builds the JSON schema from your function's signature + docstring. `helpers.run_to_completion` keeps calling the model and dispatching tools until the model is done.

```python
from llmfacade import LLM, tool, helpers

@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It's 22C and sunny in {city}."

@tool
def book_table(restaurant: str, party_size: int) -> str:
    """Book a table at the given restaurant."""
    return f"Booked {restaurant} for {party_size}."

chat = (LLM.default()
        .new_provider("anthropic")
        .new_model("claude-sonnet-4-6")
        .new_conversation(tools=[get_weather, book_table]))

resp = helpers.run_to_completion(
    chat, "What's the weather in Lisbon? If it's nice, book Cantina for 2."
)
print(resp.text)
```

## Streaming with extended thinking

`thinking=<int>` requests legacy budget-based extended thinking — valid on Sonnet 4.6 (and older models), but **not** Opus 4.7/4.8, which only accept adaptive thinking:

```python
from llmfacade import LLM

chat = (LLM.default()
        .new_provider("anthropic")
        .new_model("claude-sonnet-4-6", thinking=4096)
        .new_conversation())

for ev in chat.stream("Prove there are infinitely many primes."):
    if ev.thinking_delta:                    # dim — model's scratch reasoning
        print(f"\033[2m{ev.thinking_delta}\033[0m", end="", flush=True)
    if ev.text_delta:                        # bright — final answer
        print(ev.text_delta, end="", flush=True)
```

On Opus 4.8, use **adaptive** thinking instead — the model decides when and how
much to reason, and `effort` controls depth. `ADAPTIVE_SUMMARIZED` surfaces a
summary of the reasoning as `thinking_delta`s (plain `ADAPTIVE` keeps them
empty, the API default):

```python
from llmfacade import LLM, EffortLevel, ThinkingMode

chat = (LLM.default()
        .new_provider("anthropic")
        .new_model("claude-opus-4-8")
        .new_conversation(thinking=ThinkingMode.ADAPTIVE_SUMMARIZED, effort=EffortLevel.HIGH))

for ev in chat.stream("Prove there are infinitely many primes."):
    if ev.thinking_delta:
        print(f"\033[2m{ev.thinking_delta}\033[0m", end="", flush=True)
    if ev.text_delta:
        print(ev.text_delta, end="", flush=True)
```

## Snapshot / rollback for branching exploration

Try several continuations from the same point without rebuilding history.

```python
chat = model.new_conversation(system_blocks="You are a brand strategist.")
chat.send("Our product is a self-watering plant pot. Suggest a working name.")

snap = chat.snapshot()

chat.send("Rewrite with a futuristic angle.")
print("futuristic:", chat.send("Give me one more variant.").text)

chat.rollback(snap)
chat.send("Rewrite with a cottagecore angle.")
print("cottagecore:", chat.send("Give me one more variant.").text)
```

## Deterministic replay in CI (zero API spend)

Record once during local development; replay forever in CI.

```python
# Step 1 — record (locally, with credentials):
chat = model.new_conversation(
    cache_dir="./tests/fixtures/cache",
    cache_mode="record_only",
)
chat.send("Greet me in exactly three words.")

# Step 2 — replay (in CI, no credentials needed):
chat = model.new_conversation(
    cache_dir="./tests/fixtures/cache",
    cache_mode="replay_only",                # CacheMissError if anything drifts
)
assert chat.send("Greet me in exactly three words.").text  # served from disk
```

The cache key covers everything that affects output (provider, model, system blocks, full message list, tool schemas, settings, stop list), so any change you didn't mean to make will surface as a `CacheMissError` instead of a silent regression.

## Local model in five lines (managed mode)

No YAML to edit. The provider supervises a `llama-swap` subprocess that lazily spawns `llama-server` on first send.

```python
from llmfacade import LLM

provider = LLM().new_provider("llamacpp", n_gpu_layers=99)
model    = provider.new_model(
    name="qwen-3b",
    gguf="models/qwen2.5-3b-instruct-q4_k_m.gguf",
    context_size=8192,
)
print(model.new_conversation().send("Hi!").text)   # spawns the server, loads the model
provider.shutdown()                                  # explicit teardown (atexit also wired)
```

Prerequisite: `llama-server` and `llama-swap` on `PATH` — see the README's [Installing the llama.cpp binaries](README.md#installing-the-llamacpp-binaries) section.
