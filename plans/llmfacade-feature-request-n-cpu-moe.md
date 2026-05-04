# Feature request: expose `--n-cpu-moe` for MoE-aware partial offload

**Component**: `llmfacade` — managed-mode `llamacpp` provider
**Type**: Feature / launch-knob passthrough
**Motivated by**: MTGAI TC-2 benchmark, 2026-05-03 — failed to bench Qwen 3.6 35B-A3B and Unsloth Gemma 4 26B at 128K because partial layer-offload was unusably slow on 12 GB VRAM, with no MoE-aware alternative reachable through llmfacade's current API.

## Summary

`llama-server` accepts `--n-cpu-moe N`, which keeps the **MoE expert weights** of the first `N` layers on CPU even when those layers' non-expert weights are placed on GPU. For a sparse-MoE model like Qwen 3.6 35B-A3B (3B active per token) or Gemma 4 26B-A4B (4B active), this is the architecturally correct way to fit a 17–22 GB-class model on a 12 GB card without paying full per-token PCIe cost.

llmfacade currently exposes `n_gpu_layers` on `provider.new_model(...)` but not `n_cpu_moe`. Without it, large MoE models on small GPUs are forced into dense-style partial offload (`--n-gpu-layers 25` of ~80), which spends most of every token's compute on CPU regardless of which experts the router selected.

## Why this matters

### Current state without `--n-cpu-moe`

`--n-gpu-layers N` is layer-wholesale: all weights of layers 0..N-1 (router + attention + every expert) go to GPU; layers N..L-1 go to CPU. For a model with 80 layers, 8 experts per layer, 3 active per token:

- With `n_gpu_layers=25`, ~30% of layers are on GPU.
- For each token, the router picks 3 experts per layer.
- On a CPU-side layer, all 3 chosen experts are on CPU → CPU-bound forward pass for that layer.
- On a GPU-side layer, all 3 chosen experts are on GPU → GPU-bound forward pass.
- Net effect: ~70% of every layer's forward pass runs on CPU. The MoE win (3B active vs 35B total) is fully eaten by the layer-wholesale split.

This matches what MTGAI TC-2 saw: Qwen 3.6 35B-A3B with `n_gpu_layers=25` didn't reach TTFT in 9 minutes on a 58K-token prompt and was killed.

### Current state *with* `--n-cpu-moe N`

`--n-cpu-moe N` keeps **only the expert weights** of the first N layers on CPU; the layers' non-expert weights (router + attention + LayerNorm) and every other layer's weights all stay on GPU. The combination `--n-gpu-layers -1 --n-cpu-moe N` means "all layers on GPU, but bury the experts of the first N layers in system RAM."

For Qwen 3.6 35B-A3B at 22 GB GGUF on a 12 GB card:
- Non-expert weights are small (router + attention + norms ≪ expert weights).
- 8 experts per layer × 80 layers = 640 expert weight blobs total.
- Choosing the right `N` leaves most experts on CPU but keeps the per-token hot path (router → attention → 3 chosen experts of the rest) mostly on GPU.
- Per-token PCIe cost is 3 expert lookups instead of 3 expert × 70% layers = ~170 expert lookups.

This is the path that makes `35B-A3B`-class models actually usable on consumer GPUs.

## Proposed change to llmfacade

### `LLMModel` schema (mirrors `models.toml`)

```python
@dataclass(frozen=True)
class LLMModel:
    ...
    n_gpu_layers: int | None = None
    n_cpu_moe: int | None = None     # NEW
```

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
if info.n_cpu_moe is not None:                  # NEW
    launch_kwargs["n_cpu_moe"] = info.n_cpu_moe
return provider.new_model(**launch_kwargs)
```

### `provider.new_model(...)` → swap.yaml

```yaml
qwen36-35b-a3b:
  cmd: llama-server --model C:\Models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port ${PORT}
    --ctx-size 128000 --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers -1
    --n-cpu-moe 60          # NEW
    --parallel 1 --slot-save-path ...
  ttl: 0
```

That's the entire change — one new field, threaded straight through to llama-server.

## Picking `N` (out of scope for llmfacade itself)

llama.cpp doesn't auto-pick `--n-cpu-moe N` — it's a manual knob. The right value depends on:
- Per-expert weight size (function of total params, expert count, and quant)
- VRAM budget after weights + KV cache
- Active experts per token (architecture constant)

This is the same family of "auto-placement" math as the open `n_gpu_layers` auto-placement TODO in MTGAI. If/when llmfacade grows an auto-placement helper (port of Ollama's estimator, or wrapper around `llama-fit-params.exe`), `n_cpu_moe` should be one of its outputs alongside `n_gpu_layers`. Until then, `models.toml` would carry a measured-good integer per MoE entry, sourced from a benchmark sweep — same pattern as `n_gpu_layers` today.

## Acceptance for MTGAI

With this feature shipped, TC-2 Phase D6 can be re-run:
- `qwen36-35b-a3b` with `n_gpu_layers=-1` + `n_cpu_moe=<measured>` + `q8_0` KV cache at 128K
- Measure: load time (flare), TTFT, wall, output chars, GPU placement
- Compare against Vlad's q8_0 winner (105.5 s wall, 42.1 s TTFT) on the same Dark Sun PDF
- Decide whether Qwen 3.6 35B-A3B replaces Vlad as the long-context default for theme extraction

Same applies to `gemma4-26b-unsloth-q4kxl` once we want to compare Unsloth UD-Q4_K_XL fairly (it's MoE too).

## References

- llama.cpp `--n-cpu-moe` flag: introduced as part of the wider MoE-offload work in llama.cpp; visible as `n_cpu_moe` field in `llama-bench`'s JSON output (TC-2 Phase C results show `"n_cpu_moe": 0` for the dense / unused case).
- MTGAI TC-2 writeup: `learnings/llamacpp-tc2-benchmark.md` — Phase D6 "killed at 9 min" data point and the "MoE-aware offload via `--n-cpu-moe`" follow-up section.
- Related but separate: auto-placement for `n_gpu_layers` itself, also tracked in TC-2 writeup.
