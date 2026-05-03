# Feature request: managed slot save/restore for KV-cache reuse

**Component**: `llmfacade` — `llamacpp` provider (both external and managed modes), `Conversation` API
**Type**: Feature / new high-level API on top of llama-server's `/slots/{id}` endpoint
**Motivated by**: NotZelda NPC chat on Hetzner CX22 (CPU-only, gemma-2-2b-it Q4_K_M), 2026-05-03 — every NPC carries a 200–250-token static system prompt. With ~25 NPCs, a 32-slot in-memory checkpoint LRU isn't enough headroom; we want disk-backed prefilled-prefix caches so every NPC's first chat after a server restart skips the static-prompt prefill.

## Summary

`llama-server` exposes a slot-cache REST API:

- `GET  /slots`                                     — list slot state (KV occupancy, current tokens)
- `POST /slots/{id_slot}?action=save`              with body `{"filename": "..."}`
- `POST /slots/{id_slot}?action=restore`           with body `{"filename": "..."}`
- `POST /slots/{id_slot}?action=erase`             with body `{"filename": "..."}` (delete on disk)

The `--slot-save-path <dir>` server flag enables it; without that flag the endpoint returns 500 "This server does not support slots action".

Combined with llama-server's built-in cache-prompt prefix matching, this lets a caller:
1. Prefill a per-conversation static prefix once (the system prompt of an NPC, the role brief of an agent, the schema header of a tool-use model, …).
2. Persist that slot's KV state to disk.
3. Restore it before each subsequent chat completion. The completion's tokens prefix-match the restored KV, so prefill skips the static portion entirely.

llmfacade today has all the **plumbing** to drive this — `self._http` / `self._ahttp` httpx clients are already built against `<base_url>/v1` with `/v1` stripped to get the server root (the `_build_clients` docstring even says "server root for /health, /slots, etc."). What it doesn't have is a **high-level API** that lets callers say "warm and persist this conversation" / "restore this conversation's prefix before sending". Today every consumer that wants this has to bypass llmfacade and hit `/slots/{id}` directly with raw httpx, which means re-discovering each undocumented quirk of the endpoint.

## Why this matters

### NotZelda's situation — why this isn't optional

NotZelda's NPC chat sends a 200–250 token static system prompt per (room, npc) plus a smaller dynamic block (player name, gift state, situation). On the CX22 (2 vCPU, no GPU) prefill runs at ~26 tok/s, so a cold full prefill costs **~9 seconds** just for the static portion before any generation begins. Measured on production:

| Turn | Tokens prefilled | Time | Reused from cache |
|---|---|---|---|
| 1st Guard chat (cold) | 246 / 246 | **9.4s** | 0 |
| 2nd Guard (next msg) | 73 / 312 | **3.0s** | 239 |
| Switch to Smith (cold) | 209 / 215 | **7.8s** | 6 |
| Back to Guard (auto-checkpoint hit) | 7 / 246 | **0.7s** | 239 |
| Back to Smith (auto-checkpoint hit) | 7 / 215 | **0.5s** | 208 |

The auto-checkpoint hits are llama-server's built-in 32-slot in-memory LCP-similarity matcher doing its thing. With only 2 NPCs in rotation it's enough; at NotZelda's actual NPC count (~25) the LRU starts evicting prefixes you still need, and `systemctl restart notzelda-llama` always loses the entire warm cache. Disk-backed slot save/restore solves both — it survives restart, and when paired with explicit `restore_*` calls it makes the right prefix deterministically present in slot 0 regardless of LRU pressure.

### What NotZelda ended up writing — verbatim, as a survey of pain points

Three round-trip-and-fix iterations on production were required just to land working save/restore:

1. **First attempt** put `action=save` and `filename=...` both in query params with no body → llama-server's body-parser blew up: `parse error at line 1, column 1: attempting to parse an empty input`. (`/slots/{id}` *requires* a JSON body even though the action is in the query string.)
2. **Second attempt** moved both into the JSON body → `{"error":"Invalid action"}`. The endpoint reads `action` via `req.get_param(...)` (query string only), not from the body.
3. **Working call** is the awkward split: `?action=save` in the URL, `{"filename": "..."}` in the body. This isn't documented in the README; we only found it by reading `tools/server/server-context.cpp` at tag b9010 (`post_slots` lambda + `handle_slots_save` body parse).

Other quirks the NotZelda code had to encode:

- `--slot-save-path` has to be a launch flag, not a runtime knob — the systemd unit had to grow `--slot-save-path /var/lib/llama-cache` and the deploy script had to `mkdir` the directory. Setting it after launch is impossible without a restart.
- Slot ID is required in the URL path. With `--parallel 1` it's unambiguously `0`. With `--parallel >1` the caller has to either know which slot llama-server will pick (LCP similarity) or use `GET /slots` first and reason about it, which is ugly.
- Saving captures whatever tokens are currently in the slot's KV, including any junk from a prior unrelated request. Callers have to **first** drive a chat completion that processes only the prefix they want to cache, then save. NotZelda does this by sending `messages=[{system: static}, {user: "."}]` with `max_tokens=1` and `cache_prompt: true` so prefill happens on the static block, then the trailing `"."` user-turn tokens become a small irrelevant tail that future requests' prefix-matching cleanly truncates.
- `cache_prompt` defaults to `true` in recent llama.cpp builds, but should be set explicitly on the warmup call so the persisted KV doesn't depend on a server-side default that might flip.
- Concurrent chats will race on a slot — restore + chat-completion has to be **one atomic critical section** under `--parallel 1`, otherwise NPC B's `restore` clobbers slot 0 while NPC A's chat is still queued. NotZelda holds an asyncio `Lock` for the duration of `prepare() + asend()`. This is *very* easy to get wrong.

Anyone outside NotZelda who wants the same speedup will hit the same five quirks. They belong in llmfacade once, not in every consumer.

### Beyond NotZelda

Same primitive serves several adjacent use cases that show up in MTGAI / agentic workflows:

- **Long static system prompts + many short turns** (the canonical "snappy chat" case): one persisted KV per conversation lineage, restored on each turn.
- **Tool-use agents** with a large schema/tool-definition prefix that's stable across runs: persist once, restore at the start of each new conversation.
- **Few-shot exemplars baked into the system block**: same pattern, different content.

The CPU-only deploy is where it's most felt, but on a 4070 Ti even a 2k-token prefix is meaningful overhead at long-context settings.

## Proposed change to llmfacade

### High-level API on `Conversation`

The natural place for save/restore is on `Conversation` — a conversation already has a system block + an ongoing token sequence; "save the slot that holds my current state" is a one-liner.

```python
class Conversation:
    ...

    async def asave_slot(self, name: str) -> SlotInfo:
        """Persist this conversation's current KV state to disk under `name`.

        Tokens already processed by llama-server during prior sends are written
        as a single .bin (filename derived from `name`, sanitized). Returns
        SlotInfo(name, n_tokens, bytes_written, save_ms).

        Backend: external-mode llamacpp only (managed-mode raises NotImplemented
        until llama-swap-aware slot routing lands — see Open questions below).
        Other providers raise CapabilityError so callers can introspect.
        """
        ...

    async def arestore_slot(self, name: str) -> SlotInfo:
        """Load a previously-saved KV state into the slot this conversation
        will use on its next send. Subsequent sends prefix-match the restored
        KV so prefill skips the cached portion. Mismatched-prefix sends fall
        through to a normal cold prefill — restore is never destructive.

        Raises FileNotFoundError if `name` was never saved on this server.
        """
        ...

    async def aerase_slot(self, name: str) -> None:
        """Remove the on-disk .bin for `name`. Best-effort — silently OK if absent."""
        ...
```

Sync mirrors (`save_slot` / `restore_slot` / `erase_slot`) follow the existing `send` / `asend` pattern.

### High-level "warm and save" helper

The full `warmup → drive a completion → save` recipe is brittle enough that callers shouldn't have to reproduce it. Wrap it:

```python
class Conversation:
    async def awarm_and_save(self, name: str, *, max_warmup_tokens: int = 1) -> SlotInfo:
        """Drive a one-token completion against the current system block,
        then save the resulting slot to disk under `name`. The conversation's
        history must be empty when called — this is intended to be the first
        thing you do after `provider.new_conversation(system_blocks=[...])`.

        Equivalent to:
            await self.asend(".", max_tokens=max_warmup_tokens)
            self._messages.pop()  # discard the warmup turn
            return await self.asave_slot(name)
        """
        ...
```

Caller code becomes:

```python
convo = model.new_conversation(system_blocks=[static_prompt])
await convo.awarm_and_save("npc/town_square/Guard")
# ...later, possibly after a server restart:
convo = model.new_conversation(system_blocks=[static_prompt])
await convo.arestore_slot("npc/town_square/Guard")
resp = await convo.asend(player_message + dynamic_context)  # prefill skips static
```

That's it from the consumer side. No raw `/slots` calls, no query/body split, no slot-id arithmetic.

### Provider-level launch knob (managed mode)

Already exists: `provider.new_model(slot_save_path=...)` is plumbed to the YAML `--slot-save-path` flag. So managed-mode users get the launch flag for free. The new bit is the *runtime* save/restore methods above, which work in **both** modes.

For external mode, llmfacade should `GET /props` on first save/restore to verify `--slot-save-path` is set; if not, raise a clear `ProviderError("server was started without --slot-save-path; slot save/restore is unavailable")` rather than letting the 500 leak through.

### Concurrency contract

`Conversation.asend` / `asave_slot` / `arestore_slot` already need to interleave correctly with each other under `--parallel 1`. The natural answer is a **per-provider asyncio lock** held for the duration of any operation that mutates slot state (restore + send, save). NotZelda has to roll this lock by hand today. llmfacade owning it removes the most subtle footgun in the API.

The lock is **per-provider**, not per-conversation, because slots are a shared server resource. With `--parallel N` we'd want a slot-id-aware refinement, but `--parallel 1` is the realistic CPU-deploy default and where this matters most.

### Endpoint quirks that should be documented inside the impl

Once the impl exists, the following should live as code comments next to the HTTP calls so the next person doesn't repeat the discovery:

- `POST /slots/{id}?action=save` requires **both** `?action=` in query and `{"filename": "..."}` in JSON body. Other shapes return 500 / "Invalid action".
- The `filename` is relative to `--slot-save-path` and validated via `fs_validate_filename` server-side; no slashes, no `..`. We should sanitize on our side too rather than surface server errors.
- Saving captures *current* slot tokens, so callers must drive a completion that processed exactly the prefix they want before saving (`awarm_and_save` enforces this).
- `cache_prompt: true` should be explicit on the warmup call.

## Open questions / out of scope

- **Managed mode + llama-swap routing**: When llama-swap is in front, `/slots/{id}` on the swap port refers to whichever model is currently loaded. Save/restore semantics across model swaps are undefined (KV state is per-architecture). v1 of this feature should `raise NotImplementedError` in managed mode and revisit once llama-swap exposes a per-model slot path. NotZelda is fine with external mode only.
- **`--parallel >1` slot selection**: out of scope for v1. Fall back to slot 0 for now; emit a warning if `len(slots) > 1`.
- **Auto-eviction policy**: `--slot-save-path` accumulates files indefinitely. v1 should expose `aerase_slot` and let the caller manage retention; v2 might add `provider.list_slots()` / a global LRU policy.
- **Programmatic check of `cache_prompt` default**: not blocking, but `GET /props` could verify and warn if a server build has flipped it back to `false`.

## Acceptance for NotZelda

With this feature shipped, NotZelda's `server/npc_kv_cache.py` (~150 LoC of bespoke httpx + locking + endpoint quirk handling) collapses to:

```python
# server/npc_kv_cache.py — after llmfacade ships
async def npc_chat(convo, npc_key, static_prompt, message):
    try:
        await convo.arestore_slot(npc_key)
    except FileNotFoundError:
        await convo.awarm_and_save(npc_key)
    return await convo.asend(message)
```

And the on-disk state at `/var/lib/llama-cache/npc_*.bin` survives `systemctl restart notzelda-llama` instead of forcing one ~9s cold prefill per NPC the first time someone talks to them after every restart.

## References

- llama.cpp slot endpoint source (b9010): `tools/server/server-context.cpp` — `post_slots` lambda (line ~3584), `handle_slots_save` (line ~4153), `handle_slots_restore` (line ~4189). `req.get_param("action")` for the action verb; `json::parse(req.body).at("filename")` for the filename.
- llama.cpp `--slot-save-path` flag: enables the slot endpoint; without it `post_slots` returns ERROR_TYPE_NOT_SUPPORTED.
- llama-server built-in checkpoint LRU (32 slots, LCP-similarity match): visible in journalctl as `slot create_check: ...checkpoint X of 32` / `restored context checkpoint`. Complements but does not replace disk-backed save (no restart resilience, evicts under churn).
- NotZelda spike: branch `spike/npc-kv-cache`, file `server/npc_kv_cache.py`, commits 4368e39 / 87d62e5 / 57d4557 / 94bb37a — all four surface different facets of the endpoint contract that llmfacade should hide.
