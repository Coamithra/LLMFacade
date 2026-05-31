# Plan: homebrew an aggressively-compressed Gemma-4-26B that fits 12 GB VRAM

_Spec'd 2026-05-31 (MTGAI). Goal: a Gemma-4-26B-A4B GGUF that lives **entirely** on a 12 GB
RTX 4070 Ti — no expert spill to CPU — at a usable context, for a large decode-speed win._

## ⚠️ Before building anything: test a pre-built IQ2 first

Homebrewing needs a ~52 GB source download + an imatrix run + a quantize pass. **Don't pay
that until you've confirmed the core hypothesis** ("fits fully on GPU ⇒ massive speedup, at
acceptable quality") with an off-the-shelf quant:

1. Download `unsloth/gemma-4-26B-A4B-it-GGUF` → **`UD-IQ2_M`** (~8 GB) (or `UD-IQ3_XXS` ~10 GB).
2. Run it **fully GPU-resident**: `n_gpu_layers=-1`, **no `n_cpu_moe`**, `context_size=16384`,
   `--jinja`. Confirm via `nvidia-smi` that VRAM is flat per token (no host spill).
3. Measure **tokens/s vs the current setup** and **quality on a real MTG eval set**.
4. If unsloth's IQ2 template is stale, graft the fixed one (see Path A artifacts below) —
   `gguf_new_metadata --chat-template-file gemma4-fixed-template.jinja in.gguf out.gguf`.

If that wins on both speed and quality, **you may be done** — skip the homebrew entirely.
Only build your own if you need a better recipe than unsloth's (e.g. a different expert/
hot-path split, or your own imatrix calibrated on MTG + tool-calling traffic).

## Why the current setup is slow (measured this session)

- **Hardware:** RTX 4070 Ti, **12282 MiB** total, ~10.9 GB usable (≈10924 MiB free observed).
- **Model:** `gemma4` arch — **30 layers, 128 experts, 8 active/token**, embedding 2816,
  expert FFN 704. **~23 of the 26 B params are experts.** Sliding-window attention
  (`sliding_window=1024`); only the global layers' KV scales with context.
- **The bind:** at IQ4_XS the experts *alone* are ~12 GB — they cannot fit a 12 GB card.
  llama.cpp's `--fit` keeps the per-token hot path on GPU (`n_gpu_layers=31` ≈ all 30 layers
  + output) and **spills expert weights to CPU/RAM**. Every token then drags 8 experts × 30
  layers across PCIe and computes them on CPU. That is the entire slowdown.
- **Second villain — context.** The deployed `swap.yaml` ran `--ctx-size 128000` with q8_0 KV;
  KV at 128k is ~11.6 GB **by itself**. Most of that is unnecessary if 16k–32k is enough.

## The fit math (two levers)

**Lever 1 — shrink the experts to ~2-bit** (hot path stays high-precision):

| Expert quant | Experts | + hot path (Q5/Q6) | Total model | 12 GB? |
|---|---|---|---|---|
| IQ4_XS (today) | ~12.2 GB | +1.3 GB | ~13.2 GB | ❌ |
| IQ3_XXS | ~8.6 GB | +1.3 GB | ~9.9 GB | tight |
| **IQ2_M** | **~6.9 GB** | **+1.3 GB** | **~8.2 GB** | ✅ |

**Lever 2 — cut context** (SWA means only global layers scale; estimates, verify with
`llama-fit-params`):

| Context | KV cache (q8_0) |
|---|---|
| 128k (today) | ~11.6 GB |
| 32k | ~3.1 GB |
| 16k | ~1.7 GB |
| 8k | ~0.9 GB |

**Target combo:** experts `IQ2_M` + hot path `Q5_K`/`Q6_K` (~8.2 GB) + 16k context (~1.7 GB KV)
+ ~1 GB compute ≈ **~11 GB → fits, fully GPU-resident.** No expert spill, no PCIe per token ⇒
GPU-native decode (expect several ×). The IQ2 dequant cost on *GPU* is trivial next to
deleting the offload (this is exactly why unsloth's IQ3_S experts were slow on CPU but won't
be once everything's on-device).

**Risk:** IQ2 experts cost quality. A 128-expert/8-active MoE is the most forgiving case
(router + attention + embeddings stay sharp; only 8/128 experts fire per token), but it
*will* be measurably worse than IQ4 on hard prompts. **Eval, don't assume.** If too soft,
step experts up to IQ3_XXS and drop context to 8k to stay within budget.

## Homebrew recipe (only if step-0 prebuilt isn't good enough)

Toolchain is already present at `C:\Tools\llama.cpp\` (`llama-quantize`, `llama-imatrix`,
`llama-fit-params`, `llama-gguf-split`); the `gguf` python package is installed.

1. **Source weights (~52 GB):** download **bf16** `unsloth/gemma-4-26B-A4B-it` or
   `google/gemma-4-26B-A4B-it`. You **cannot** requantize *up* from the existing Q4 files —
   an i-quant needs an f16/bf16 source. (739 GB free on `C:` as of writing.) Convert to GGUF
   with `convert_hf_to_gguf.py --outtype bf16` (needs the llama.cpp *repo*, not just the
   release binaries), or grab a pre-made bf16/f16 GGUF and skip conversion.
2. **Imatrix (required for i-quants):**
   ```
   llama-imatrix -m gemma4-26b-bf16.gguf -f calib.txt -o g4.imatrix --n-gpu-layers 99
   ```
   Fold real **MTG-domain + tool-calling** traffic into `calib.txt` so the experts you
   actually hit are weighted well.
3. **Quantize with the MoE-aware split** (flags verified against the local binary):
   ```
   llama-quantize --imatrix g4.imatrix \
     --tensor-type ffn_down_exps=iq2_m --tensor-type ffn_gate_exps=iq2_m --tensor-type ffn_up_exps=iq2_m \
     --output-tensor-type q6_K --token-embedding-type q6_K \
     gemma4-26b-bf16.gguf gemma4-26b-moe-iq2m.gguf IQ3_XXS
   ```
   Base ftype sets the attention default; the `*_exps` overrides crush only the experts.
   Bump experts to `iq3_xxs` if `iq2_m` is too soft and it still fits the context budget.
4. **Template:** a fresh convert from latest weights already embeds the current (fixed)
   chat template. If you quantized from an older source, graft the fix:
   `gguf_new_metadata --chat-template-file C:\Models\gemma4-fixed-template.jinja in.gguf out.gguf`.
5. **Register in the facade (managed mode):**
   ```python
   provider.new_model(
       gguf="C:/Models/gemma4-26b-moe-iq2m.gguf",
       name="gemma4-26b-moe-iq2m",
       context_size=16384,        # not 128000
       n_gpu_layers=-1,           # full offload — the whole point
       # NO n_cpu_moe — we want experts ON the GPU
       cache_type_k="q8_0", cache_type_v="q8_0",
       jinja=True,
       flash_attn="on",           # q8_0 V already forces flash; explicit is clearer
       fit=True, fit_ctx=16384,   # keep --fit as a safety net but floor the context
   )
   ```

## Acceptance criteria

- `nvidia-smi` shows the model fully resident and **VRAM flat across decode** (no host spill).
- llama-server load log: all layers + experts offloaded to GPU; no expert-on-CPU lines.
- **tokens/s** materially higher than the current 128k/IQ4 spill setup (benchmark both).
- **Quality** holds on a real MTG eval set (the IQ2 risk — the gating check).
- Tool-calling + reasoning still correct (fixed template; no "applying compatibility
  workarounds" warning at load).

## Fallback knobs if it won't fit / quality too low

- Experts `IQ3_XXS` instead of `IQ2_M` (better quality, needs 8k ctx to fit).
- `cache_type_k/v=q4_0` to roughly halve KV (quality cost on long context).
- Drop context to 8k.
- Avoid `--prune-layers` — too lossy for this model; not worth it.

## Artifacts already on disk (from the 2026-05-31 session)

- `C:\Models\gemma4-fixed-template.jinja` (+`.json`) — the unsloth Apr-11+ fixed chat
  template (16,934 chars), extracted from `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`. Reuse for any
  future quant via `--chat-template-file` / grafting.
- `C:\Models\vlad-updated-gemma4-26b.gguf` (13.23 GB) — Path A result: vlad's fast all-4-bit
  recipe + the fixed template (byte-identical graft). This is the **current best** until a
  fits-on-GPU quant exists; it still spills experts to CPU.
- `C:\Models\_ggufmeta.py` — stdlib GGUF metadata reader (arch params + template extract).

## Local quant comparison (why "IQ4_XS" labels lie)

Both vlad and unsloth declare `general.file_type=30` (IQ4_XS) but differ where it counts:

| File | Size | Expert FFN quant |
|---|---|---|
| `vlad-gemma4-26b-dynamic.gguf` | 13.23 GB | IQ4_NL + IQ4_XS (all 4-bit) |
| `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` (unsloth) | 12.66 GB | IQ4_NL + **IQ3_S** (half 3-bit) |

unsloth's 3-bit experts made it *smaller but slower on CPU* — the lesson that motivates this
plan: on a CPU-spilled MoE, expert quant *type* drives speed; once fully on GPU, the
calculus flips and aggressive expert compression becomes a pure win.

## References

- `docs/learnings/llamacpp-reasoning-tool-calling.md` — the chat-template fix (Path A).
- `docs/learnings/qwen3.6-27b-12gb.md` — 12 GB-card fit analysis; MoE vs dense; `n_cpu_moe`.
- `CLAUDE.md` → llama.cpp provider quirks: `n_cpu_moe`, `flash_attn`, autofit, `jinja`.
- Local toolchain: `C:\Tools\llama.cpp\`.
