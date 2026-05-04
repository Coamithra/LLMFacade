# Feature request: managed-mode llamacpp vision support (mmproj + image content blocks)

**Component**: `llmfacade` — managed-mode `llamacpp` provider
**Type**: Feature
**Motivated by**: MTGAI `art_selector.py` (Haiku vision today), longer-term local-first vision use cases like screenshot/diagram analysis or local OCR. Surfaced during MTGAI TC-2 benchmark, 2026-05-03.

## Summary

llama-server has supported multimodal models for a while via `--mmproj <path>` (loads the multimodal projector alongside the main model) plus the OpenAI-compatible `image_url` content block on `/v1/chat/completions`. llmfacade's managed-mode llamacpp provider exposes neither end of that pipe today:

1. `provider.new_model(...)` doesn't accept an `mmproj_path` (or equivalent) launch knob, so the YAML llmfacade writes never includes `--mmproj`.
2. There's no marshalling for vision content blocks, so even if a vision-capable server *was* running, MTGAI's `ImageBlock`-style API wouldn't know how to send images to it.

Net effect: `supports_vision = false` on every llamacpp registry entry in MTGAI today, including Gemma 4 26B (which has a multimodal projector available). All vision work falls to Anthropic Haiku/Sonnet — fine for low volumes, increasingly inconvenient as use cases grow.

## Why this matters

### MTGAI's current vision usage

Just one stage today: `art_selector.py` (Haiku picks the best of N generated card arts per card). Haiku vision costs ~$0.006/card → ~$1.70 for a 280-card set. Trivial cost; the motivator isn't $$, it's:
- Removing the Anthropic dependency from the card-generation pipeline so it can fully self-host
- Avoiding round-trip latency on a per-card loop
- Keeping image content (sometimes private setting prose + custom art) out of a third-party API

### What llama.cpp actually supports today

- `llama-server --mmproj <path>` loads the projector. The same `--model <path>` keeps loading the LLM weights as usual.
- `/v1/chat/completions` accepts OpenAI-style content blocks of the shape `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` — base64 inline data URLs work; bare `data:` and `http(s):` URLs both supported.
- Some models ship as a single GGUF (vision tensors baked in); others ship as `<model>.gguf` + `mmproj-<quant>.gguf` pair from the same HF repo. Both patterns need the same `--mmproj` flag — even the single-GGUF case wants the multimodal projector path passed explicitly so llama-server enables the multimodal code path.
- Currently-known compatible families on the build at `C:\Tools\llama.cpp\b9010-d05fe1d7d`: Llava, MiniCPM-V, Qwen2-VL, Gemma 3, Qwen 3.6 (per Unsloth docs). Gemma 4 multimodal needs verification — the architecture is in the metadata but loader strict-mode rejects the Ollama-stripped variants (see TC-2 writeup); whether the build supports the *complete* multimodal Gemma 4 GGUF is untested.

### Why text-only repacks aren't enough

The existing Unsloth Gemma 4 26B GGUFs we use today (`gemma-4-26B-A4B-it-GGUF`) are **deliberately text-only repacks** — that's why they load in stock llama.cpp where Ollama's full multimodal blobs don't (TC-2 writeup, "Why the Ollama Gemma 4 blobs fail" section). Going multimodal means downloading the full multimodal version of each model + its mmproj file, which is a separate path from text-only theme-extraction work. Both can coexist as separate `models.toml` entries; they're not in conflict.

## Proposed change to llmfacade

### `LLMModel` schema (mirrors `models.toml`)

```python
@dataclass(frozen=True)
class LLMModel:
    ...
    n_gpu_layers: int | None = None
    n_cpu_moe: int | None = None     # (separate feature request)
    mmproj_path: str | None = None   # NEW — absolute path to mmproj-*.gguf
```

### `_llamacpp_new_model` thread-through (in MTGAI's wrapper, mirrors any change inside llmfacade)

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
if info.mmproj_path is not None:                # NEW
    launch_kwargs["mmproj_path"] = info.mmproj_path
return provider.new_model(**launch_kwargs)
```

### llama-swap YAML output

```yaml
qwen36-vision:
  cmd: llama-server --model C:\Models\Qwen3.6-VL-7B.gguf --port ${PORT}
    --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers -1
    --mmproj C:\Models\mmproj-F16-Qwen3.6-VL-7B.gguf      # NEW
    --parallel 1 --slot-save-path ...
  ttl: 0
```

### Vision content block marshalling

llmfacade's existing `ImageBlock.from_path(...)` (used for Anthropic via `art_selector.py`) needs to also work when the conversation's underlying provider is llamacpp. Two paths:

1. **Per-provider marshalling in the conversation layer**. Anthropic gets the existing native image block; llamacpp gets it converted to `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>"}}`. Conversion is purely mechanical — read file, MIME-detect, base64-encode.
2. **Provider-agnostic OpenAI-shaped wire format**. Always emit `image_url` blocks; let the Anthropic provider transcode them back to its native shape. Cleaner long-term but a larger refactor.

Path 1 is the smaller diff and matches the existing pattern (Anthropic-shaped tool schemas already get wrapped in `Tool(...)` and re-shaped per-provider).

### Capability flag

`Model.supports_vision: bool` (or equivalent) should reflect the truth — `True` when `mmproj_path` is set on the LLMModel, `False` otherwise. MTGAI already keys off `supports_vision` in `models.toml`; the registry currently hardcodes `False` for every llamacpp entry as a TODO, which would resolve once this lands.

## Out of scope (for this request)

- **Picking which vision-capable models to register** — that's MTGAI's responsibility, downstream of this feature. Likely first targets: Qwen 3.6 vision, MiniCPM-V (it has a dedicated `llama-minicpmv-cli.exe` binary in the toolchain, suggesting strong llama.cpp support).
- **Verifying Gemma 4 multimodal works in stock llama.cpp builds** — separate investigation; possibly needs a newer llama.cpp build than the b9010 currently shipped at `C:\Tools\llama.cpp\`.
- **Vision projector autodetection** — could llmfacade scan the GGUF directory and auto-find `mmproj-*.gguf` matching the main model's name? Nice ergonomics improvement but explicit `mmproj_path` is the safer and more debuggable starting point.

## Acceptance for MTGAI

With this feature shipped:
1. Add a vision-capable Local entry to `models.toml` (e.g. Qwen 3.6 VL) with `mmproj_path = "..."` and `supports_vision = true`.
2. Update `art_selector.py` to read the per-stage model setting (it already does for non-vision stages); when the configured model is llamacpp + vision, route through llmfacade's vision-marshalling path instead of Anthropic.
3. Re-run art selection on a small ASD card subset; compare quality + speed against Haiku baseline.
4. If acceptable, add a "Local-first" preset to `model_settings.py` that uses local for vision too.

## References

- llama.cpp `--mmproj` flag: documented in llama-server's `--help`. Multimodal models like LLaVA, MiniCPM-V, Qwen2-VL all use it. The `llama-mtmd-cli.exe`, `llama-llava-cli.exe`, `llama-minicpmv-cli.exe`, `llama-qwen2vl-cli.exe`, `llama-gemma3-cli.exe` binaries shipped in the toolchain at `C:\Tools\llama.cpp\` confirm broad family support.
- OpenAI image content block shape: <https://platform.openai.com/docs/guides/vision> — same shape llama-server consumes.
- MTGAI `art_selector.py` and the existing Anthropic vision path: `backend/mtgai/art/art_selector.py:142` (`facade_model = provider.new_model(model)` then `ImageBlock.from_path(...)`).
- TC-2 writeup, "Local vision is an llmfacade gap" line in MTGAI's CLAUDE.md, and the Phase D Gemma multimodal-tensor failure that surfaced this.
- Adjacent feature request: `llmfacade-feature-request-n-cpu-moe.md` — same-shape "expose a launch knob" change. If both land, the `_llamacpp_new_model` patch is one small commit.
