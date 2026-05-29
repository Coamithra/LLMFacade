# Learnings

Durable, hard-won knowledge — external quirks, upstream bugs, non-obvious gotchas, plus research/comparison notes (which model or quant to pick, and why), that cost real time and would be painful to rediscover.

**This is not `plans/`.** `plans/*.md` describe *open work* and get deleted when the work merges. Learnings *persist* — they're the institutional memory of "why is it like this" and "what bit us." Add an entry whenever a debugging session ends in an insight you'd otherwise forget.

One file per topic. Date each entry and link any relevant issues/PRs/plans.

| Entry | What it covers |
|---|---|
| [llamacpp-reasoning-tool-calling.md](llamacpp-reasoning-tool-calling.md) | Why reasoning vanishes on tool-using turns; the reasoning↔tool-calling tension; `--jinja` / `enable_thinking`; forced-tool_choice misrouting bug |
| [qwen3.6-27b-12gb.md](qwen3.6-27b-12gb.md) | A 12 GB-friendly Qwen3.6-27B quant vs Gemma 4 `UD-IQ4_XS` — dense vs MoE, why `n_cpu_moe` doesn't apply, the quant-size ladder |
