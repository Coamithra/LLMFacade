# Research: A 12 GB-friendly Qwen3.6-27B quant comparable to Gemma 4 `UD-IQ4_XS`

_Researched 2026-05-29. Baseline at time of writing: `gemma-4-26B-A4B-it-UD-IQ4_XS` (13.6 GB) on a 12 GB VRAM card._

## Questions
- Q1: Does "qwen3-6-27b" exist, with an Unsloth Dynamic GGUF quant in the IQ4_XS tier?
- Q2: Is it comparable to the Gemma 4 `UD-IQ4_XS`, and will it run on 12 GB VRAM?

## Bottom line
Yes, **Qwen3.6-27B exists** and Unsloth ships a full GGUF quant ladder including IQ4_XS (15.4 GB) and several sub-12 GB options. **But it is not an apples-to-apples swap for Gemma 4**, and the difference is architectural, not just file size: Gemma is MoE with ~4B active params; Qwen3.6-27B is **dense** (all 27B active every token). That changes everything about how it behaves on a 12 GB card.

## Finding 1 — The model is real and is a *dense* 27B (this is the crux)

- "Qwen3.6-27B has 27 billion parameters... released on Hugging Face Hub and ModelScope on April 22, 2026, under Apache 2.0." [S3]
- Model card: *"Number of Parameters: 27B... Number of Layers: 64... Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))"* — and: *"Qwen3.6-27B is a dense model, not a mixture-of-experts (MoE) model... all parameters are active per token."* [S4]

The Gemma baseline is the opposite: `gemma-4-26B-A4B` — the `A4B` means **~4B active** (sparse MoE), exactly the case the LLMFacade CLAUDE.md describes the `n_cpu_moe` trick for ("keeps only the chosen experts moving across PCIe per token"). [S0]

**Why this matters for 12 GB:** the way a 13.6 GB Gemma file fits on a 12 GB card is the MoE expert-offload trick — keep the per-token hot path (router + attention + the small active-expert slice) on GPU via `n_gpu_layers=-1`, and bury the bulk expert weights in system RAM with `n_cpu_moe`. A dense model has **no expert weights to offload** — `n_cpu_moe` is a no-op. To fit dense Qwen3.6-27B in 12 GB you must offload *whole transformer layers* (`n_gpu_layers < 64`), and every CPU-resident layer runs its **full** compute on CPU per token — the slow path the CLAUDE.md warns about ("plain `n_gpu_layers=25` puts ~70% of every token's compute on CPU"). [S0]

## Finding 2 — Quant sizes (Unsloth `Qwen3.6-27B-GGUF`)

Closest match to the Gemma `UD-IQ4_XS` (13.6 GB [S1]) and the smaller options [S2]:

| Quant | Size | Fits fully in 12 GB VRAM? |
|---|---|---|
| `IQ4_XS` (Gemma tier) | **15.4 GB** | No — needs ~4 GB CPU offload |
| `UD-Q3_K_XL` | 14.5 GB | No |
| `Q3_K_S` | 12.4 GB | No (and no room for KV) |
| `UD-IQ3_XXS` | 12.0 GB | Borderline — fills card, no KV headroom |
| `UD-Q2_K_XL` | 11.8 GB | Barely; no KV headroom |
| `UD-IQ2_M` | 10.8 GB | Yes-ish (~1 GB for KV) |
| `UD-IQ2_XXS` | 9.39 GB | Yes (~2.5 GB for KV) |

Note: found a plain `Qwen3.6-27B-IQ4_XS.gguf` (15.4 GB) but **not** a file literally named `UD-IQ4_XS` in the listing — Unsloth's newer repos sometimes drop the `UD-` prefix on IQ4_XS even when it's dynamic. Verify the exact filename on the repo tree [S2] before scripting a download.

## Finding 3 — Unsloth's own VRAM guidance points away from 12 GB at Q4

- Unsloth docs: 4-bit needs **~18 GB** total (RAM+VRAM), 3-bit **~15 GB**; they note llama.cpp "can still run via SSD/HDD offloading, but inference will be slower," and the **minimum recommended is 15 GB for 3-bit**. No 12 GB guidance is given. [S5]
- One genuine upside for a dense model: **MTP** (Multi-Token Prediction) variants "accelerate dense models 1.4–2x vs MoE 1.15–1.25x" [S5]. So the `unsloth/Qwen3.6-27B-MTP-GGUF` repo benefits a dense Qwen *more* than MTP would benefit the MoE Gemma — this partly offsets the offload penalty. [S2][S5]

## Recommendation
- For the **same quality tier** as Gemma's IQ4_XS, the analogous `Qwen3.6-27B-IQ4_XS` (15.4 GB) exists, but on 12 GB it is a partial **layer** offload — expect materially slower tokens/s than the Gemma MoE setup, because there are no experts to keep the GPU hot path cheap. Pair it with the **MTP GGUF** to claw back ~1.4–2x.
- To **actually fit in VRAM** for fast inference: `UD-IQ2_M` (10.8 GB) or `UD-IQ2_XXS` (9.39 GB) — but Q2 on a *dense* 27B is a steep quality drop (very different from Q2 on a 4B-active MoE).
- `n_cpu_moe` from the facade is **irrelevant here** (dense model). Tune `n_gpu_layers` + `--fit` instead.

## Gaps and Uncertainties
- Could not confirm a file literally named `UD-IQ4_XS` for Qwen3.6-27B (only plain `IQ4_XS`); confirm on the repo tree.
- Did not measure KV-cache size for any target context length. The architecture is mostly Gated DeltaNet (linear attention; only 1-in-4 layers is full attention) [S4], which **[inference, unverified]** should make KV growth with long context much gentler than a standard dense model — a real plus for low-VRAM long-context, but no source benchmarks it on 12 GB.
- No source benchmarks actual tokens/s for dense Qwen3.6-27B with partial offload on a 12 GB card specifically.

## Sources
- [S0] `C:\Programming\LLMFacade\CLAUDE.md` — `n_cpu_moe` semantics; `A4B` = MoE 4B-active.
- [S1] https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf — baseline, 13.6 GB, MoE.
- [S2] https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main — quant file sizes.
- [S3] https://qwen.ai/blog?id=qwen3.6-27b ; https://simonwillison.net/2026/apr/22/qwen36-27b/ — release, dense, multimodal.
- [S4] https://huggingface.co/Qwen/Qwen3.6-27B — dense, 27B, 64 layers, DeltaNet hybrid.
- [S5] https://unsloth.ai/docs/models/qwen3.6 — VRAM guidance, MTP speedups.
