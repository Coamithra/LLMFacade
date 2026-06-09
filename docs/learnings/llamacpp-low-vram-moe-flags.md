# Low-VRAM MoE on llama.cpp: the flag ladder, `--no-mmap` / `--mlock`, TurboQuant KV, and why speculative decoding backfires

_Learned 2026-06-04 from Codacus, "Running a 35B AI Model on 6GB VRAM, FAST (llama.cpp Guide)"
([video](https://www.youtube.com/watch?v=8F_5pdcD3HY), [companion article](https://mychen76.medium.com/run-qwen3-6-35b-a3b-on-6gb-vram-using-llama-cpp-30-tps-a89032e5a60c)).
Rig: GTX 1060 6 GB, PCIe Gen 3, i3-8100 (4c/no-HT), 24 GB DDR4 — a deliberate worst-case floor.
Model: Qwen 3.6 35B-A3B (35B total, ~3B active/token; 256 experts, 8 active; **30 of 40 layers are
SSM/state-space**)._

## TL;DR

Five launch flags take the same model+hardware from **3 → 17 tokens/s** at **4× the context**,
with no quality loss. All five map to LLMFacade knobs: `n_cpu_moe` / `cache_type_k` / `cache_type_v`
were already exposed, and `no_mmap` / `mlock` are now first-class managed-mode `LAUNCH_KNOBS` too.
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

> **`--mlock` + `--no-mmap` can CUDA-OOM at load — spill-fraction-dependent.** An external report
> (Reddit r/LocalLLaMA, heitortp0, 2026-06-06, Qwen3.6-35B-A3B Q4_K_M on an **8 GB** 4060 laptop /
> 32 GB RAM) hit `CUDA error: out of memory` when combining the two: with ~20 GB of experts pinned in
> host RAM, the pinned-host allocation exhausts CUDA's pinned-memory pool even though plain RAM is
> ample. **This does not contradict the 4070 Ti row below** (`no_mmap + mlock` → ~63 t/s, clean) — that
> rig is a 26B mostly *GPU-resident* on a 12 GB card, so only ~2 GB of inactive experts are host-resident
> and get pinned. The rule: the combo is safe when little spills, dangerous when most of the model lives
> in RAM. The facade now `warnings.warn`s at `new_model()` when both flags are set; fix is to drop
> `mlock`, keep `no_mmap`. (Unverified locally — believed because the mechanism is sound and matches the
> "pin 20 GB of host pages" math.)

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
**Build check (2026-06-05): our local llama-server `b9500` does NOT support them yet** -- its allowed
`--cache-type` values are `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` only. So this lever is
gated on a newer/custom build; passing `turbo4`/`turbo3` today fails at server launch (`q8_0` is our
near-lossless baseline in the meantime).

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

## On-hardware validation: the same levers on a 4070 Ti 12 GB (2026-06-05)

Re-tested these flags on a *much* better rig than the video's floor: RTX 4070 Ti (12 GB),
64 GB DDR4, NVMe. Model: `vlad-gemma4-26b-dynamic` (26B-A4B MoE, all-4-bit experts) via the
facade managed mode, 16k ctx, q8_0 KV, `--jinja`, `--flash-attn on`. Long prompt = 6877 tokens;
tok/s are llama-server `timings`. Numbers averaged over 2-3 runs each.

| Config | decode t/s | prompt-eval t/s | VRAM after load |
|---|---|---|---|
| `--fit on` (our deployed baseline) | ~62 | ~1697 | 11.27 GB |
| `+ no_mmap` | ~63 | **~2295 (+35%)** | 11.25 GB |
| `+ no_mmap + mlock` | ~63 | ~2258 | 11.28 GB |
| `n_cpu_moe=24` (fit off) | 44.5 | 946 | **6.30 GB** |

Three takeaways, two of them refinements of the video:

1. **`--no-mmap` gives ~+35% prompt-eval here -- even with 64 GB RAM and the model fully page-
   cached.** So on this rig the benefit is NOT (only) "avoid disk reads" as the video framed it
   (we have none -- it's cached). It's a **memory-access-pattern** effect: prompt-eval routes all
   6877 tokens through every layer, hitting a wide spread of experts in the mmap'd CPU-resident
   spill region, so even resident pages cost page-table/minor-fault overhead. `--no-mmap`'s
   contiguous malloc preload removes that. **Decode barely moves (+1-3%, within noise)** because
   each decoded token touches only its ~8 active experts. Net: `--no-mmap` is a near-free win
   for *prompt-eval-bound* MoE workloads (big single-pass inputs), not for chat decode.
2. **`--mlock` adds nothing to throughput** (as the video says -- its payoff is the long-horizon
   "day-3" anti-pageout stability, which a single bench can't show). Worth enabling on a
   long-lived managed backend anyway; it's free on a 64 GB box.
3. **`n_cpu_moe` is the context<->speed trade lever, and `--fit` already wins it on a 12 GB card.**
   The video *needed* `n_cpu_moe` to fit a 35B in 6 GB at all, then *lowered* it to fill spare
   VRAM for speed. On 12 GB, `--fit` already places experts to max the GPU (62 t/s, 11.3 GB used).
   Forcing `n_cpu_moe=24` pinned too many experts to CPU -- dropped to 44 t/s but freed ~5 GB.
   The knob's real use on this card is *deliberately* trading decode speed for a much larger
   context, not chasing speed (fit beats a naive manual split).

**Actionable for MTGAI:** the deployed 26B does prompt-eval-heavy theme extraction on large
(whole-PDF) inputs -- adding `no_mmap=True` (+ `mlock=True` for stability) to its managed-mode
registration should speed that stage ~1/3 for free, given the box has 64 GB RAM. Decode-bound
chat stages won't change. (That's an MTGAI config change, not an LLMFacade one.)

## Cross-references

- `no_mmap` / `mlock` are now implemented `LAUNCH_KNOBS` (settings.py) -- this entry's actionable facade work is DONE; the plan file was removed.
- `docs/learnings/qwen3.6-27b-12gb.md` — dense vs MoE; why `n_cpu_moe` is a no-op on dense models.
- `docs/learnings/llamacpp-reasoning-tool-calling.md` — `--jinja` / `enable_thinking` interplay.
- `CLAUDE.md` → llama.cpp provider quirks: `n_cpu_moe`, `cache_type_k/v`, `flash_attn`, autofit.
