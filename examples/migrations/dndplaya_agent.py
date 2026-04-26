"""DnDPlaya pattern: stateful multi-agent tool-use with snapshot/rollback for failure recovery.

Demonstrates how a DM-style agent (with thinking budget, tool dispatch, history snapshots)
maps onto the new API.
"""

from __future__ import annotations

import os

from llmfacade import LLM, ConvoSettings, Settings, tool


@tool
def narrate(scene: str) -> str:
    """Narrate a scene to the players."""
    return f"[narrated: {scene[:60]}...]"


@tool
def ask_skill_check(skill: str, dc: int, player: str) -> str:
    """Request a skill check from a player."""
    return f"[{player} rolls {skill} DC {dc}: passes]"


def build_dm_agent():
    api_keys = {}
    if "ANTHROPIC_API_KEY" in os.environ:
        api_keys["anthropic"] = os.environ["ANTHROPIC_API_KEY"]
    mgr = LLM(api_keys=api_keys)
    anth = mgr.NewProvider("anthropic")
    sonnet = anth.NewModel("claude-sonnet-4-6")

    dm = sonnet.NewConversation(name="dm")
    dm.AddSystemBlock(
        "You are the Dungeon Master. Lore: <long campaign setting...>",
        cache=True,
    )
    dm.AddSystemBlock("Current scene: tavern brawl in the Dragon's Tail Inn.", cache=False)
    dm.settings.set(ConvoSettings.AutoCacheLastUser, True)
    dm.settings.set(Settings.Thinking, 500)
    dm.AddTool(narrate)
    dm.AddTool(ask_skill_check)
    dm.SetLogging("./logs/dm.jsonl")
    dm.Start()
    return dm


def turn(dm, player_action: str):
    """One DM turn with snapshot/rollback for failure recovery (DnDPlaya pattern)."""
    snap = dm.Snapshot()
    try:
        resp = dm.Complete(player_action, max_tokens=2048, tool_choice="auto")
        return resp
    except Exception:
        dm.Rollback(snap)
        raise


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return
    dm = build_dm_agent()
    resp = turn(dm, "Karthos the warrior tries to leap onto the table and intimidate.")
    print("DM response:", resp.text[:300])
    print(f"cache_read={resp.usage.cache_read_tokens} thinking={bool(resp.thinking)}")


if __name__ == "__main__":
    main()
