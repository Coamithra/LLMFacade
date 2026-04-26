"""MTGAI pattern: forced tool_choice for structured output, multimodal art selection.

Demonstrates how MTGAI's card generation (forced tool_use) and art selector (vision)
map onto the new API.
"""

from __future__ import annotations

import os
from pathlib import Path

from llmfacade import LLM, ImageBlock, TextBlock, tool


@tool
def card_design(
    name: str,
    mana_cost: str,
    type_line: str,
    rules_text: str,
    power: int = 0,
    toughness: int = 0,
) -> str:
    """Submit a fully-designed Magic card."""
    return f"{name} | {mana_cost} | {type_line} | {rules_text} | {power}/{toughness}"


@tool
def select_art(index: int, justification: str) -> str:
    """Select the best art among the candidates by index (1-based)."""
    return f"selected #{index}: {justification[:80]}"


def gen_card():
    """Card generation with forced tool_choice."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        return None
    mgr = LLM(api_keys={"anthropic": os.environ["ANTHROPIC_API_KEY"]})
    sonnet = mgr.NewProvider("anthropic").NewModel("claude-sonnet-4-6")

    convo = sonnet.NewConversation(name="card-gen")
    convo.AddSystemBlock(
        "You design balanced Magic: the Gathering cards. <long set brief...>",
        cache=True,
    )
    convo.AddTool(card_design)
    convo.Start()

    resp = convo.Send(
        "Design a 2-mana green creature with vigilance.",
        max_tokens=2048,
        tool_choice="card_design",
    )
    print("forced tool call:", resp.tool_calls[0].input)
    return resp


def select_art_demo(art_paths: list[Path]):
    """Multi-image vision: pick the best art from candidates."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        return None
    mgr = LLM(api_keys={"anthropic": os.environ["ANTHROPIC_API_KEY"]})
    sonnet = mgr.NewProvider("anthropic").NewModel("claude-sonnet-4-6")

    convo = sonnet.NewConversation(name="art-select")
    convo.AddSystemMessage("You select Magic card art that best fits the theme.")
    convo.AddTool(select_art)
    convo.Start()

    blocks: list = [TextBlock("Pick the best art for 'haunted forest':")]
    for i, p in enumerate(art_paths, 1):
        blocks.append(TextBlock(f"\nCandidate #{i}:"))
        blocks.append(ImageBlock.from_path(p))

    resp = convo.Send(
        prompt=blocks,
        max_tokens=500,
        tool_choice="select_art",
    )
    print("art pick:", resp.tool_calls[0].input if resp.tool_calls else "none")
    return resp


if __name__ == "__main__":
    print("=== Card generation ===")
    gen_card()
    print("\n=== Art selection (needs PNGs in arts/) ===")
    arts = sorted(Path("arts").glob("*.png")) if Path("arts").exists() else []
    if arts:
        select_art_demo(arts[:3])
    else:
        print("(skipped - no arts/*.png found)")
