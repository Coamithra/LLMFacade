# Research: Gemma 4 12B on a 12 GB card — a dense model that *actually fits*

_Researched 2026-06-04. Trello card "Look at the new Gemma4 12B model" (recently released,
"might fit 12 GB with the right quant"). Baseline for comparison: `gemma-4-26B-A4B-it`
(MoE, ~4B active) run via expert-offload, and the homebrew-IQ2 plan that tries to cram it
fully onto a 12 GB RTX 4070 Ti (`plans/homebrew-gemma4-vram-fit.md`)._

## Questions
- Q1: Does a Gemma 4 **12B** model exist? What is it, exactly?
- Q2: Dense or MoE? (Decides whether the `n_cpu_moe` GPU-hot-path trick applies.)
- Q3: Which GGUF quant fits a 12 GB card, and with how much KV headroom?

## Bottom line
Yes — **Gemma 4 12B is real, released 2026-06-03, dense (not MoE), and multimodal**. Because
it is a *dense 12B* (~11.95 B params, all active), it sidesteps the entire 26B-A4B fit
problem: an off-the-shelf **Q5_K / Q6_K quant runs fully GPU-resident on a 12 GB card with
room left for KV** — no expert spill, no `n_cpu_moe`, no homebrew IQ2 quant. This is the
"fully on GPU ⇒ big speedup" outcome `plans/homebrew-gemma4-vram-fit.md` was chasing for the
26B, achieved for free by switching to the smaller dense model. Google also reports it
*beats Gemma 3 27B* on GPQA Diamond / MMLU Pro / DocVQA and stays close to the 26B MoE. [S1]

## Finding 1 — The model is real, dense, and encoder-free multimodal

- Released **2026-06-03**, Apache 2.0, on Hugging Face + Kaggle. First mid-size Gemma with
  **native audio**; it drops the separate vision/audio encoders most multimodal models use,
  feeding raw image patches and audio waveforms straight into the LLM. [S1]
- **Dense, NOT MoE.** Unsloth's own docs contrast the 12B (dense) with the
  *"26B-A4B ... MoE ... 4B active parameters"*; apxml and secondary writeups concur. [S2][S4][S5]
- This is the **same crux as the Qwen3.6-27B note**: on a dense model the facade's
  `n_cpu_moe` knob is a **no-op** (there are no expert weights to bury in RAM). The
  difference from Qwen is that here it *doesn't matter* — the whole model fits on the GPU,
  so you never need to offload anything. See `docs/learnings/qwen3.6-27b-12gb.md` and
  CLAUDE.md → llama.cpp quirks (`n_cpu_moe` semantics). [S0]

## Finding 2 — Architecture (12 GB-relevant bits)

From apxml's spec sheet [S2]:

- **11.95 B params**, **48 layers**, hidden **3,840**, FFN intermediate **15,360**.
- **Sliding-window attention, 1,024-token window** (Gemma-3/4 family trait) — so KV grows
  gently with context: only the global layers scale, not every layer. Good for low-VRAM
  long context.
- 16 query / **8 KV heads**, head dim 256 (GQA). Vocab **262,144**. RoPE theta 10,000.
- **Context up to 262,144 (256K)** [S2][S4] — but don't *run* at 256K on 12 GB; the KV
  cache blows the budget. Use 16k–32k.
- **Multimodal** (text / image / audio), encoder-free. [S1][S2]

## Finding 3 — Quant ladder (Unsloth `gemma-4-12b-it-GGUF`) and the 12 GB fit

Exact file sizes from the HF repo tree [S3]. Fit column assumes ~10.9 GB usable on a 12282 MiB
card (the same budget as the homebrew plan), leaving the remainder for KV + compute:

| Quant | Size | Fits fully GPU-resident in 12 GB? |
|---|---|---|
| `BF16` | 23.8 GB | No |
| `UD-Q8_K_XL` | 13.6 GB | No |
| `Q8_0` | 12.7 GB | No (over the card by itself) |
| `UD-Q6_K_XL` | 10.7 GB | Tight — only with quantized KV + small ctx |
| **`Q6_K`** | **9.79 GB** | ✅ ~1 GB for KV (modest ctx) |
| **`Q5_K_M` / `UD-Q5_K_XL`** | **8.41 / 8.61 GB** | ✅ ~2.3–2.5 GB for KV — comfortable |
| **`UD-Q4_K_XL`** (Unsloth pick) | **7.37 GB** | ✅ ~3.5 GB headroom |
| `Q4_K_M` / `Q4_K_S` | 7.12 / 6.76 GB | ✅ |
| `IQ4_NL` / `IQ4_XS` | 6.72 / 6.38 GB | ✅ lots of room (long ctx OK) |
| `UD-Q3_K_XL` | 6.02 GB | ✅ |
| `UD-Q2_K_XL` / `UD-IQ3_XXS` / `UD-IQ2_M` | 4.66 / 4.64 / 4.21 GB | ✅ (only if you need huge ctx) |
| `mmproj-F16.gguf` (vision projector) | **122 MB** | negligible add-on |

The projector for vision is tiny (122 MB F16 / 175 MB BF16 / 210 MB F32) [S3], so enabling
images costs almost nothing against the VRAM budget.

## Finding 4 — Vendor VRAM guidance

Unsloth's run-locally page [S4]:
- **4-bit: "7–8 GB"**, 8-bit: "13–14 GB", BF16: 25 GB. (These are total runtime
  footprint — weights + KV + compute overhead — so they run a bit above the raw GGUF file
  sizes in Finding 3; e.g. 8-bit's 13–14 GB vs the 12.7 GB `Q8_0` file.)
- Recommended starting quant: **`UD-Q4_K_XL`**.
- Thinking control: `--chat-template-kwargs '{"enable_thinking":false}'` (the
  `TEMPLATE_KWARG` style the facade auto-detects; needs `--jinja`).

So 4/5/6-bit all sit inside a 12 GB budget. Contrast the two prior 12 GB studies: the 26B-A4B
needed expert-offload to fit at all, and dense Qwen3.6-27B needed **Q2** to fit in VRAM —
this dense 12B fits at **Q5/Q6** with quality headroom to spare.

## Recommendation

- **Default: `Q5_K_M` (8.41 GB) or `UD-Q4_K_XL` (7.37 GB), fully GPU-resident.** Both leave
  2.5–3.5 GB for KV at 16k–32k context with room for the vision projector. Step up to `Q6_K`
  (9.79 GB) if quality matters more than context length; step down to `IQ4_XS` (6.38 GB) if
  you want maximum context.
- This likely **obsoletes the homebrew-IQ2 26B plan** for most local use: instead of crushing
  26B experts to IQ2 to fit, run the native 12B dense at Q5/Q6 — fewer total params but full
  precision on the hot path, fully on-GPU, and reportedly stronger than Gemma 3 27B. Keep the
  26B only where its extra capacity is demonstrably needed on a real eval.
- **Eval before committing** — the speed story is solid (dense + fits ⇒ GPU-native decode);
  the quality-vs-26B claim is Google's, not independently benchmarked here.

## Facade config (llamacpp managed mode)

```python
provider.new_model(
    gguf="C:/Models/gemma-4-12b-it-Q5_K_M.gguf",
    name="gemma4-12b-q5",
    context_size=16384,        # NOT 256000 — KV blowup on a 12 GB card
    n_gpu_layers=-1,           # dense + fits ⇒ full offload, no spill
    # NO n_cpu_moe — dense model, it's a no-op
    cache_type_k="q8_0", cache_type_v="q8_0",
    jinja=True,                # required for Gemma 4 enable_thinking + correct tool calls
    flash_attn="on",           # here also forced by the quantized V cache; explicit anyway
                               #   because Gemma 4 'auto' disables flash on f16 KV (~2x cost)
    mmproj_path="C:/Models/mmproj-F16.gguf",  # only if you want vision; ~122 MB
    fit=True, fit_ctx=16384,   # safety net, floored at the chosen context
)
```

Notes tying back to existing CLAUDE.md quirks:
- `n_cpu_moe` intentionally absent — dense model (same reason as Qwen3.6-27B).
- `mmproj_path` is the facade knob that actually turns on vision; without it llama-server
  serves text-only even though the GGUF advertises multimodal. Drop it (and narrow with
  `capability_override=provider.SUPPORTS - {"vision"}`) for a text-only deploy.
- `jinja=True` + `flash_attn="on"` are the same Gemma 4 gotchas documented for the 26B
  (`docs/learnings/llamacpp-reasoning-tool-calling.md`, CLAUDE.md → llama.cpp quirks).

## Gaps and Uncertainties

- **KV-cache size per context length not measured** on this model. Sliding-window attention
  *should* keep it small (only global layers scale), but no source benchmarks it. The fit
  table's KV headroom figures are size-budget arithmetic, not measured KV.
- **No independent tokens/s benchmark** on a 12 GB card specifically — the speed claim rests
  on "dense + fully resident ⇒ no PCIe per token," which is sound but unbenchmarked here.
- **Quality-vs-26B is Google's claim** [S1], not independently verified. Q5/Q6 on a dense 12B
  should hold up far better than Q2 on a dense 27B, but eval on a real task set before relying
  on it.
- Sizes are **as published by Unsloth** on the repo tree [S3]; I read the listing, not the
  files. The `UD-` dynamic quants use mixed per-tensor precision, so the realized quality at a
  given size differs from the plain quant of the same name.

## Sources
- [S0] `C:\Programming\LLMFacade\docs\learnings\qwen3.6-27b-12gb.md` — dense-vs-MoE crux; why `n_cpu_moe` is a no-op on a dense model.
- [S1] https://www.marktechpost.com/2026/06/03/google-deepmind-releases-gemma-4-12b-an-encoder-free-multimodal-model-with-native-audio-that-runs-on-a-16-gb-laptop/ ; https://techstartups.com/2026/06/03/google-deepmind-launches-gemma-4-12b-bringing-frontier-ai-model-to-everyday-laptops/ — release date, encoder-free multimodal, native audio, beats Gemma 3 27B.
- [S2] https://apxml.com/models/gemma-4-12b — architecture: 11.95B, 48 layers, hidden 3840, sliding-window 1024, 256K context, multimodal.
- [S3] https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/tree/main — exact GGUF quant file sizes + mmproj projector sizes.
- [S4] https://unsloth.ai/docs/models/gemma-4 — dense vs 26B-MoE, 4-bit "7–8 GB" VRAM guidance, `UD-Q4_K_XL` recommendation, `enable_thinking`.
- [S5] https://www.buildfastwithai.com/blogs/gemma-4-12b-guide — dense decoder-only confirmation.
