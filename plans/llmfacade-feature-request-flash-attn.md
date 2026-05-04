# Feature request: expose `--flash-attn` for managed-mode llamacpp

**Component**: `llmfacade` — managed-mode `llamacpp` provider
**Type**: Feature / launch-knob passthrough
**Motivated by**: MTGAI TC-2 benchmark, 2026-05-03. Strong empirical evidence that flash attention is *not* engaging on Gemma 4 in our llama-server runs, contradicting an inherited assumption. Forcing it on should reclaim a ~40 % wall and ~8× TTFT speedup based on TC-1f Ollama data.

## Summary

`llama-server` accepts `-fa, --flash-attn [on|off|auto]` at launch, defaulting to `auto`. Empirical TC-2 verification (2026-05-03) showed:

- **Quantized V cache forces flash on**: `--cache-type-v q8_0` with `--flash-attn off` crashes immediately with `"V cache quantization requires flash_attn"`. So our production Vlad q8_0 / q4_0 configs were already running with flash on (no upside available there).
- **f16 V cache + Gemma 4 + auto = flash off**: the `auto` heuristic disables flash for f16 KV on Gemma 4 (per TC-1f also: "Gemma 4 on pre-Turing GPUs is auto-disabled"; the heuristic is conservative on Ada Lovelace too). This explains TC-2's f16 row being 2.1× slower than Ollama's flash-on f16 baseline (711s vs 346s).

llmfacade currently exposes `cache_type_k`, `cache_type_v`, and `n_gpu_layers` on `provider.new_model(...)` but not `flash_attn`. There's no way for MTGAI to force flash on (or off) without dropping out of managed mode.

## Why this matters

### Empirical evidence

TC-2 verification on Vlad Gemma 4 26B / 128K / Dark Sun PDF:

| Config | Wall | TTFT | Note |
|---|---|---|---|
| q8_0 KV, default `auto` | 105.5 s | 42.1 s | Flash on (auto picked it because V quant forces it) |
| q8_0 KV, `--flash-attn on` | 103.6 s | 42.5 s | Identical. Confirms baseline already had flash on. |
| q8_0 KV, `--flash-attn off` | — | — | **Server refuses to start**: "V cache quantization requires flash_attn" |
| f16 KV, 35-layer offload, default `auto` | 711.6 s | 408.3 s | TC-2 baseline. Flash status not directly tested but matches the "off" signature: |
| (TC-1f reference) Ollama dynamic f16 flash ON | 346 s | 56 s | What our f16 row would look like *with* flash |
| (TC-1f reference) Ollama upstream f16 flash OFF | 865 s | 474 s | What our f16 row currently looks like *without* flash |

So:
- For quantized V cache: flash is hard-coupled on and we already get it.
- For f16 V cache on Gemma 4: `auto` quietly says no, costing us roughly 2× wall time. Forcing `--flash-attn on` should reclaim it.

### Inherited assumption (now corrected)

MTGAI's `CLAUDE.md` originally claimed *"Flash attention is on by default in llama-server for architectures that support it (no flag needed)"*. That sentence was paraphrased from the TC-1f Ollama 0.21+ verification and silently extrapolated to llama-server during the May 2026 migration without re-confirmation. TC-2 surfaced the half-truth: only with quantized V cache. CLAUDE.md is now corrected.

### Expected upside

Limited but real:

- **Production Vlad q8_0 / q4_0 configs** — already flash-on, no upside from this knob alone (still worth setting `flash_attn = "on"` explicitly so the registry is self-documenting and immune to future `auto` heuristic changes).
- **Any f16 KV use case on Gemma 4** — projected wall reduction roughly 2× (TC-1f data point: Ollama dynamic with flash on ran the same f16 corpus in 346 s vs flash-off 865 s on a similar config). MTGAI doesn't currently use f16 KV in production, but the option opens up for small models that don't need cache quantization, or hardware with enough VRAM that f16 is the natural choice.
- **Future non-Gemma models** — `auto` may pick differently per architecture; explicit control removes a source of silent slowdown at registration time.

## Proposed change to llmfacade

### `LLMModel` schema (mirrors `models.toml`)

```python
@dataclass(frozen=True)
class LLMModel:
    ...
    n_gpu_layers: int | None = None
    n_cpu_moe: int | None = None      # (separate feature request)
    mmproj_path: str | None = None    # (separate feature request)
    flash_attn: str | None = None     # NEW — "on" / "off" / "auto" / None
```

`None` means "don't pass the flag, let llama-server's `auto` decide". `"on"` / `"off"` / `"auto"` are forwarded verbatim. Keeping it as a string (rather than bool) preserves the "auto" option for cases where the heuristic does the right thing — e.g. small dense models on Hopper where `auto` is presumably accurate.

### `_llamacpp_new_model` thread-through

```python
launch_kwargs: dict[str, Any] = {
    "name": info.model_id,
    "gguf": info.gguf_path,
    "context_size": info.context_window,
}
if info.cache_type_k is not None:
    launch_kwargs["cache_type_k"] = info.cache_type_k
if info.cache_type_v is not None:
    launch_kwargs["cache_type_v"] = info.cache_type_v
if info.n_gpu_layers is not None:
    launch_kwargs["n_gpu_layers"] = info.n_gpu_layers
if info.flash_attn is not None:                     # NEW
    launch_kwargs["flash_attn"] = info.flash_attn
return provider.new_model(**launch_kwargs)
```

### llama-swap YAML output

```yaml
gemma4-26b-vram-dynamic:
  cmd: llama-server --model C:\Models\vlad-gemma4-26b-dynamic.gguf --port ${PORT}
    --ctx-size 128000 --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers -1
    --flash-attn on                             # NEW
    --parallel 1 --slot-save-path ...
  ttl: 0
```

Single new field, threaded straight through to llama-server's CLI.

## Acceptance for MTGAI

With this feature shipped:
1. Add `flash_attn = "on"` to every llamacpp entry in `models.toml` for self-documentation and to immunize against future `auto` heuristic changes (no runtime difference for q8_0 / q4_0 configs since flash is already forced on, but explicit is better).
2. Re-run TC-2 Phase B f16 row with `--flash-attn on` to confirm the projected ~2× wall reduction (711 s → ~340 s based on TC-1f reference).
3. If a future use case calls for f16 KV cache (e.g. small model that doesn't quantize, or hardware with VRAM headroom), the registry now controls flash explicitly rather than relying on `auto`.
4. Update CLAUDE.md and `learnings/llamacpp-tc2-benchmark.md` with the f16-with-flash measurement.

## References

- llama.cpp `--flash-attn` flag: `llama-server.exe --help` shows `-fa, --flash-attn [on|off|auto]` with default `auto` and env var `LLAMA_ARG_FLASH_ATTN`.
- TC-1f Ollama writeup attributing 83 % of speedup to flash attention: `MTGAI/learnings/gemma4-benchmark.md` lines 70–88.
- TC-2 evidence: Phase C `llama-bench` JSON `"flash_attn": false` on every model; Phase B f16 TTFT (408 s) matching TC-1f flash-OFF baseline (474 s) far better than flash-ON (56 s).
- Adjacent feature requests in this directory: `llmfacade-feature-request-n-cpu-moe.md`, `llmfacade-feature-request-vision.md`. The pattern is identical; if all three land, the `_llamacpp_new_model` patch in MTGAI is one small commit.
