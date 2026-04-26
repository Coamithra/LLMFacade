"""End-to-end demo of the v1 API.

Runs against local Ollama by default. If ANTHROPIC_API_KEY is set, also runs the
Anthropic-only sections (caching, thinking, multimodal).

  python examples/blacksmith.py
  python examples/blacksmith.py --provider anthropic --model claude-sonnet-4-6
  python examples/blacksmith.py --provider ollama    --model gemma2:2b
"""

from __future__ import annotations

import argparse
import asyncio
import os

from llmfacade import (
    LLM,
    ConvoSettings,
    Settings,
    UnsupportedFeature,
    tool,
)


@tool
def forge_item(item: str, material: str = "iron") -> str:
    """Forge an item out of a material. Returns a description string."""
    return f"You receive a {material} {item}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="gemma2:2b")
    args = parser.parse_args()

    api_keys = {}
    if "ANTHROPIC_API_KEY" in os.environ:
        api_keys["anthropic"] = os.environ["ANTHROPIC_API_KEY"]

    mgr = LLM(api_keys=api_keys, log_dir="./logs")
    provider = mgr.NewProvider(args.provider)
    model = provider.NewModel(args.model)

    print(f"== {args.provider}:{args.model} ==")
    print("Capabilities:", sorted(s.name for s in model.getCapabilities()))

    blacksmith = model.NewConversation(name="blacksmith")
    blacksmith.AddSystemBlock(
        "You are Garrick, a gruff but fair blacksmith in a medieval village.",
        cache=blacksmith.isAvailable(ConvoSettings.AutoCacheLastUser),
    )
    if blacksmith.isAvailable(Settings.ContextSize):
        blacksmith.settings.set(Settings.ContextSize, 4096)
    blacksmith.AddTool(forge_item)
    blacksmith.SetLogging(f"./logs/{args.provider}-blacksmith.jsonl")
    blacksmith.Start()

    print("\n[sync Complete]")
    resp = blacksmith.Complete("Hi there.", max_tokens=200)
    print("response:", resp.text[:200])
    if resp.usage:
        print(f"usage: prompt={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")

    print("\n[tool use]")
    resp = blacksmith.Complete(
        "I need a sword. Use the forge_item tool to make me one.",
        max_tokens=300,
    )
    print("text:", resp.text[:200])
    print("tool calls used:", len(resp.tool_calls))

    print("\n[Stream]")
    for ev in blacksmith.Stream("Now describe the sword poetically.", max_tokens=200):
        if ev.text_delta:
            print(ev.text_delta, end="", flush=True)
    print()

    print("\n[aComplete]")

    async def go():
        return await blacksmith.aComplete("Anything else, smith?", max_tokens=100)

    resp = asyncio.run(go())
    print("async response:", resp.text[:200])

    print("\n[Snapshot/Rollback]")
    snap = blacksmith.Snapshot()
    blacksmith.Complete("[ignore me]", max_tokens=20)
    print("history len before rollback:", len(blacksmith.history))
    blacksmith.Rollback(snap)
    print("history len after rollback:", len(blacksmith.history))

    print("\n[Clone]")
    alt = blacksmith.Clone()
    alt.Start()
    alt.Complete("Now refuse rudely.", max_tokens=80)
    print("original history:", len(blacksmith.history), "clone history:", len(alt.history))

    print("\n[capability gating]")
    fresh = model.NewConversation(name="probe")
    if fresh.isAvailable(Settings.Thinking):
        fresh.settings.set(Settings.Thinking, 1024)
        print(f"Thinking is supported on {args.provider}/{args.model}; budget=1024 set.")
    else:
        try:
            fresh.settings.set(Settings.Thinking, 1024)
        except UnsupportedFeature as e:
            print(f"As expected, Thinking is not available on {args.provider}: {e}")


if __name__ == "__main__":
    main()
