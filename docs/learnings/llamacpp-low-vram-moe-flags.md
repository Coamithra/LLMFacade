# Low-VRAM MoE on llama.cpp: the flag ladder, `--no-mmap` / `--mlock`, TurboQuant KV, and why speculative decoding backfires

_Learned 2026-06-04 from Codacus, "Running a 35B AI Model on 6GB VRAM, FAST (llama.cpp Guide)"
([video](https://www.youtube.com/watch?v=8F_5pdcD3HY), [companion article](https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c)).
Rig: GTX 1060 6 GB, PCIe Gen 3, i3-8100 (4c/no-HT), 24 GB DDR4 — a deliberate worst-case floor.
Model: Qwen 3.6 35B-A3B (35B total, ~3B active/token; 256 experts, 8 active; **30 of 40 layers are
SSM/state-space**)._

## TL;DR

Five launch flags take the same model+hardware from **3 → 17 tokens/s** at **4× the context**,
with no quality loss. Three of the five LLMFacade already exposes; two (`--no-mmap`, `--mlock`) it
does not, except via the `extra_args` escape hatch — see `plans/llamacpp-no-mmap-mlock-knobs.md`.
And one widely-recommended trick — **speculative decoding — actively makes this class of model
slower** (17 → 11 t/s). Don't add a draft-model knob for MoE/SSM models.

## The flag ladder (each step measured on the floor rig)

| Step | Flag | t/s | Why |
|---|---|---|---|
| baseline | `--n-gpu-layers 20` (naive half-split) | **3** | Every CPU-resident layer drags its *whole* expert block across PCIe per token. Bus chokes. |
| MoE offload | `--n-cpu-moe 41` | **10** | Pin only the **expert** blocks to CPU, keep the small fast-firing parts (router/attention/norms) on GPU. Per token the GPU asks for just the 8 needed experts. +230%. |
| RAM preload | `+ --no-mmap` | **13.5** | Default mmap demand-pages experts from **disk** mid-token (page fault → late token). `--no-mmap` reads the whole ~20 GB into RAM up front; every expert lookup is then predictable, no disk reads during inference. +35%. |
| fill VRAM | `--n-cpu-moe 41 → 35` | **17** | 2 GB of VRAM was still free at 13.5 t/s. Lowering `n-cpu-moe` pulls 6 layers' experts back onto the GPU. VRAM 4 → 5.5 GB. **Trade-off: less room for KV → context drops** (100k → 64k). |
| context | TurboQuant KV (below) | **17** | Reclaim context for free — compression doesn't slow decode. |
| stability | `+ --mlock` | **17** | Not a speed flag. Stops the kernel paging experts back out to disk after hours/idle (the "day-3 slowdown"). |

Final: **35B params, 6 GB VRAM, 256k context, 17 t/s, stable for a week.** "The hardware isn't the
bottleneck anymore. The defaults are."

## `--no-mmap` — the non-obvious +35%

mmap is the *right* default for a model that fits VRAM (lazy, low RAM). It is the *wrong* default
the moment expert weights live in **system RAM**, because "in RAM" is a lie until touched — the OS
pages each expert from disk on first use, and an MoE touches a fresh set every few tokens. `--no-mmap`
forces the full preload so there are no mid-inference disk reads.

**Caveat (why it must stay opt-in):** it requires the model to *fit* in RAM — it trades disk-backed
lazy loading for a full preload. On a tight-RAM box it backfires (or won't load). Never a default.

## `--mlock` — "works in a demo vs. survives Tuesday"

Even after `--no-mmap` preloads everything, the kernel still treats those pages as evictable file
cache; under memory pressure or idle it pages experts back to disk → next inference stutters with
page faults. `--mlock` pins them: "do not touch this RAM, it's mine." Speed is unchanged on a fresh
boot (experts already cached); what changes is the server runs for a **week without degrading**.
This matters specifically for **managed mode**, which *is* a long-lived supervised backend.

> **Docker gotcha:** `--mlock` needs three things aligned or it silently no-ops (no error, just a
> slow leak back to the bad behaviour): the container's `memlock` ulimit raised, the `IPC_LOCK`
> capability granted, **and** the `--mlock` flag. Miss any one and it falls back to default.

## TurboQuant KV — 4× context, same speed

KV cache stores keys+values per token per layer and grows **linearly** with context, so context is
paid in VRAM. The lever:

- **Q8 KV is the near-lossless baseline.** Going past it naively (Q4/Q3 symmetric) wrecks answers.
- **TurboQuant** (Google DeepMind, early 2026): random rotation, *then* aggressive quant — Q4 keys /
  Q3 values with quality indistinguishable from Q8. Flags: `--cache-type-k turbo4 --cache-type-v turbo3`.
- **The asymmetry (turbo4 K, turbo3 V) is not a typo.** This model uses GQA at an **8:1** ratio, so
  keys tolerate heavier compression than values. On a model with a different GQA ratio, revisit the split.
- Result on the rig: 64k → 256k context (5.9/6 GB) at the same 17 t/s — "the cache is small enough
  the lookup is essentially free." (Getting to 256k also needed `n-cpu-moe 35 → 36`, i.e. one more
  expert layer back to CPU, to free the last sliver of VRAM.)

**LLMFacade already supports this** — `cache_type_k`/`cache_type_v` are free-form pass-through
strings (no value whitelist), so `turbo4`/`turbo3` forward verbatim to a build that supports them.

## Speculative decoding **backfires** on MoE + SSM (the negative result worth keeping)

The intuitive next optimization — run a tiny drafter, let the big model verify 8 tokens in one batch
— is **net-negative here**: a Qwen3.5-0.8B drafter with a *healthy* 65% acceptance rate dropped
speed **17 → 11 t/s**. Two architectural reasons, both fatal for this model class:

1. **MoE breaks the batch.** Each of the 8 batched draft tokens picks its own 8-of-256 experts, so
   one verify step can need up to **64 different experts per layer**, each fetched fresh from CPU RAM
   over PCIe. "Verify in one batch" becomes a memory thrash — per-token verify time barely drops, and
   you still pay for running the draft model. Net loss.
2. **SSM can't be parallelized across a draft window.** 30 of 40 layers are state-space; each position
   depends on the previous step's state, so the "verify N tokens in one pass" trick simply doesn't
   apply. (Reproduced by others on a 3090 across 19 configs — same result.)

**Takeaway for the facade:** do **not** add a `draft_model`/speculative launch knob aimed at MoE or
SSM models — it's architecturally a no-op-to-regression. Speculative decoding still helps **dense
transformers**, so if such a knob is ever added it must be gated on architecture, not offered blindly.
(A diffusion-based drafter — "DLash" / block-diffusion, generating 8 tokens in one shot — was floated
as the thing that *might* work for the dense 27B sibling; unverified, future work.)

## Cross-references

- `plans/llamacpp-no-mmap-mlock-knobs.md` — the actionable facade work this entry motivates.
- `docs/learnings/qwen3.6-27b-12gb.md` — dense vs MoE; why `n_cpu_moe` is a no-op on dense models.
- `docs/learnings/llamacpp-reasoning-tool-calling.md` — `--jinja` / `enable_thinking` interplay.
- `CLAUDE.md` → llama.cpp provider quirks: `n_cpu_moe`, `cache_type_k/v`, `flash_attn`, autofit.
