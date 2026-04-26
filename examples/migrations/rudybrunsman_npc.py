"""RudyBrunsman pattern: async NPC chat with cached static system + dynamic context.

Demonstrates the async-first multi-turn NPC chat with prompt caching that
RudyBrunsman uses for in-game NPC interactions.
"""

from __future__ import annotations

import asyncio
import os

from llmfacade import LLM, ConvoSettings

# Static lore - rarely changes, gets cached
NPC_LORE = """\
You are an NPC in a 2D adventure game. Respond as the character.
World rules: medieval fantasy, no anachronisms.
Output format: a single tag in [BRACKETS] followed by 1-2 sentences of dialogue.
Tags: [FRIENDLY], [NEUTRAL], [ANGRY], [GIVE_ITEM], [CALL_GUARDS].
Strip any roleplay actions or emotes.
"""


def build_npc(player_name: str, friendship: int):
    if "ANTHROPIC_API_KEY" not in os.environ:
        return None
    mgr = LLM(api_keys={"anthropic": os.environ["ANTHROPIC_API_KEY"]})
    haiku = mgr.NewProvider("anthropic").NewModel("claude-haiku-4-5-20251001")

    npc = haiku.NewConversation(name=f"npc-for-{player_name}")
    npc.AddSystemBlock(NPC_LORE, cache=True)
    npc.AddSystemBlock(
        f"Player: {player_name}. Friendship: {friendship}.",
        cache=False,
    )
    npc.settings.set(ConvoSettings.AutoCacheLastUser, True)
    npc.settings.set(
        ConvoSettings.UserMetadata,
        {"user_id": f"notzelda-npc-{player_name}"},
    )
    npc.Start()
    return npc


async def chat_loop(npc, lines: list[str]):
    history_cap = 4  # RudyBrunsman trims to MAX_HISTORY=4 turns
    for line in lines:
        # Trim history before sending (keep system blocks + last few turns)
        if len(npc.history) > history_cap * 2:
            npc.Rollback(npc.Snapshot())  # no-op rollback to current; in practice
            # you'd snapshot before each turn and roll back to a trimmed point.
        resp = await npc.aComplete(line, max_tokens=100)
        print(f"  player: {line}")
        print(f"  npc:    {resp.text.strip()}")
        if resp.usage:
            print(
                f"  tokens: in={resp.usage.prompt_tokens} "
                f"cache_read={resp.usage.cache_read_tokens} "
                f"cache_write={resp.usage.cache_creation_tokens}"
            )


def main():
    npc = build_npc("Karthos", friendship=2)
    if npc is None:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return
    asyncio.run(
        chat_loop(
            npc,
            [
                "Greetings, friend.",
                "Do you have anything to trade?",
                "What about news from the capital?",
            ],
        )
    )


if __name__ == "__main__":
    main()
