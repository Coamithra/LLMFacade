# Learnings

Durable, hard-won knowledge — external quirks, upstream bugs, non-obvious gotchas, plus research/comparison notes (which model or quant to pick, and why), that cost real time and would be painful to rediscover.

**This is not `plans/`.** `plans/*.md` describe *open work* and get deleted when the work merges. Learnings *persist* — they're the institutional memory of "why is it like this" and "what bit us." Add an entry whenever a debugging session ends in an insight you'd otherwise forget.

One file per topic. Date each entry and link any relevant issues/PRs/plans.

| Entry | What it covers |
|---|---|
| [llamacpp-reasoning-tool-calling.md](llamacpp-reasoning-tool-calling.md) | Why reasoning vanishes on tool-using turns; the reasoning↔tool-calling tension; `--jinja` / `enable_thinking`; forced-tool_choice misrouting bug |
| [qwen3.6-27b-12gb.md](qwen3.6-27b-12gb.md) | A 12 GB-friendly Qwen3.6-27B quant vs Gemma 4 `UD-IQ4_XS` — dense vs MoE, why `n_cpu_moe` doesn't apply, the quant-size ladder |
| [gemma4-12b-12gb.md](gemma4-12b-12gb.md) | Gemma 4 12B (dense, multimodal) fits 12 GB at Q5/Q6 fully GPU-resident — quant ladder + facade config. **Benchmarked 2026-06-05: does NOT replace the 26B-A4B** — the MoE decodes faster (61 vs 40 tok/s, 4B active) and matched/beat it on quality; the 12B wins only on prompt-eval throughput. Retires the 26B homebrew-IQ2 plan |
| [llamacpp-low-vram-moe-flags.md](llamacpp-low-vram-moe-flags.md) | The 5-flag ladder for fast low-VRAM MoE (3→17 t/s): `n_cpu_moe`, `--no-mmap`, GPU-fill, TurboQuant KV, `--mlock`; the Docker `IPC_LOCK` gotcha; why speculative decoding backfires on MoE/SSM |
