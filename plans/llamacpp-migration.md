# Replace Ollama with llama-server (llama.cpp HTTP)

## Context

LLMFacade ships an Ollama provider as its only local-model option. Ollama keeps surfacing bugs the user is unwilling to keep working around:

- **#15783** — Ollama's Go-native sampler silently accepts and ignores `repeat_penalty` / `frequency_penalty` / `presence_penalty` on newer models (Gemma 4 etc.). Open issue with PR pending.
- **No per-model KV cache quantization.** Ollama exposes only the global `OLLAMA_KV_CACHE_TYPE` env var. llama.cpp lets you set `--cache-type-k` / `--cache-type-v` per server instance, so per-model is achievable by running one server per config.
- **No introspection.** Ollama can't tell you "is the model currently generating?" or surface per-slot state.
- **No KV cache disk persistence.** Ollama loses warm caches on every restart. llama-server can save/restore per-slot KV state to disk via `POST /slots/{id}?action={save,restore,erase}` plus the `--slot-save-path` flag — the requested "warm-up on slow machines" win.
- **Ollama silently uses a smaller context than `num_ctx` on its OpenAI-compatible endpoint** (the user previously hit this and was forced onto Ollama's native API to get the requested context honored). llama-server has no such asymmetry: `/completion`, `/completions`, and `/v1/chat/completions` all share the single KV cache allocated at server launch via `--ctx-size N`. What you set on the CLI is what the chat-completions endpoint actually uses; with `--parallel K` it splits as `--ctx-size / K` per slot. Per-request you control *truncation* via `n_keep` / `n_discard` (forwarded through `extra_body`); you cannot grow the cache past launch-time allocation. Per-model ctx-size differences are handled by running separate servers — same pattern as the KV-quant case.

llama-server (`llama.cpp/tools/server`) addresses all five. Its `/v1/chat/completions` endpoint is OpenAI-compatible and additionally accepts llama.cpp-specific samplers (`min_p`, `mirostat*`, `dynatemp*`, `dry*`, `xtc*`, `n_keep`, `json_schema`, custom `samplers` ordering) in the request body.

The migration removes the Ollama provider entirely — single local provider going forward.

## Approach

Add a new **`LlamaCppServerProvider`** registered as `"llamacpp"`. It uses the `openai` Python SDK for transport (HTTP, streaming, tool plumbing) by pointing it at the llama-server `/v1` base URL. Adds a small `httpx`-based side channel for the non-OpenAI introspection endpoints. No new heavy deps — `httpx` already comes with `openai>=2.32`.

Why not just `OpenAIProvider(base_url="http://localhost:8080/v1")`? Because that drops every llama.cpp-specific sampler (the OpenAI SDK won't pass them) and gives no access to `/health` / `/slots` / `/slots/{id}?action=save`. The wedge for a thin dedicated provider is real.

Why not in-process `llama-cpp-python` bindings? Heavy compiled-wheel install (especially with CUDA on Windows), per-process model copies, sync-only library forces async wrappers, and tools require per-model chat-format selection. The user already runs a daemon with Ollama; running `llama-server` is the same operational shape with strictly more capability.

## File-level changes

### Add

- **`src/llmfacade/providers/llamacpp.py`** (~220 lines). New `LlamaCppServerProvider`:
  - `NAME = "llamacpp"`, `API_KEY_ENV = None`, lazy-imports `openai` (raises `ProviderNotInstalledError` with `pip install llmfacade[llamacpp]`).
  - `_init_client()` builds `openai.OpenAI(base_url=..., api_key="sk-noop")` and async equivalent. Stores `self._http_base` for raw introspection.
  - `_build_chat_kwargs(req)` mirrors the OpenAI provider's payload assembly, but **routes llama.cpp-specific knobs into `extra_body=`**: `min_p`, `repeat_penalty` (as `repetition_penalty` if needed — verify against current llama-server build), and any keys present in `req.settings["extra_body"]` (a dict knob, see settings cascade below).
  - `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw` call into the OpenAI SDK; reuse parsing helpers cribbed from `src/llmfacade/providers/openai.py` (do not import private symbols — copy the small helpers).
  - **Introspection methods (synchronous, plus async twins):**
    - `health() -> dict` — `GET /health`. Returns `{"status": "ok"}` or raises `ProviderError` on 503.
    - `slots() -> list[dict]` — `GET /slots`. Per-slot processing state, sampling params, token counts, generation speed.
    - `save_slot(id_slot: int, filename: str) -> dict` — `POST /slots/{id_slot}?action=save` body `{"filename": filename}`.
    - `restore_slot(id_slot: int, filename: str) -> dict` — `POST /slots/{id_slot}?action=restore` body `{"filename": filename}`.
    - `erase_slot(id_slot: int) -> dict` — `POST /slots/{id_slot}?action=erase`.
    - All five hit `self._http_base` directly via `httpx`. Do NOT add `props()` or `metrics()` for now — minimal surface per the user's choice.
  - Maps `openai.AuthenticationError` → no-op (server doesn't auth), `openai.RateLimitError` → `RateLimitError`, all else → `ProviderError`.

- **`tests/test_llamacpp.py`** — unit tests modelled on `tests/test_ollama_finish_reason.py`:
  - finish_reason translation (`length` → `"length"`, etc.)
  - `extra_body` passthrough (assert `min_p`/`mirostat` land in the request body)
  - tool-call parsing roundtrip with mocked OpenAI client
  - `health()` / `slots()` / `save_slot()` against a mocked `httpx` transport
- **`tests/integration/test_llamacpp_live.py`** — modelled on `tests/integration/test_ollama_live.py`:
  - tool roundtrip
  - one save_slot → erase → restore cycle to prove the warmup path
  - Skips if server unreachable.

### Modify

- **`src/llmfacade/providers/__init__.py`** — replace the `"ollama"` entry with `"llamacpp": ("llmfacade.providers.llamacpp", "LlamaCppServerProvider")`.
- **`src/llmfacade/settings.py`** — `RUNTIME_KNOBS`: **add** `min_p`, `extra_body`. **Remove** `keep_alive` (Ollama-only) and `context_size` (Ollama-only — llama-server takes ctx via `--ctx-size` at launch). The 14 → 14 net (drop 2, add 2) with `min_p` first-class because it's the standard recommended knob for local models, and `extra_body` for forward-compat with the rest of llama.cpp's sampler family without polluting the global knob list.
- **`src/llmfacade/provider.py`** — `Provider.__init__` kwargs: drop `keep_alive`, `context_size`; add `min_p`, `extra_body`. The `_validate_knobs` / `_filter_unsupported` paths Just Work because they read from `RUNTIME_KNOBS`.
- **`pyproject.toml`** — `[project.optional-dependencies]`: drop `ollama = [...]`, add `llamacpp = ["openai>=2.32"]`. Update `all = [...]` to include `llamacpp` and drop `ollama`. The current comment on line 35 ("Anthropic and Ollama have no offline tokenizer") should change to "Anthropic and llamacpp have no offline tokenizer".
- **`tests/test_tool_choice.py`** — replace the Ollama-fixture block (≈ lines 62–65, 182–237) with a llamacpp-fixture block. llama-server's `/v1/chat/completions` accepts standard OpenAI `tool_choice`, so unlike Ollama llamacpp's `SUPPORTS` will include `"tool_choice"` — the tests change shape, not just identity. Drop the "rejects tool_choice" tests; add equivalents asserting it passes through.
- **`tests/integration/conftest.py`** — replace `ollama_host`/`ollama_model` fixtures (lines 55–61) with `llamacpp_host` (env `LLAMACPP_HOST`, default `http://localhost:8080`) and `llamacpp_model` (env `LLAMACPP_MODEL`, default `qwen2.5-3b-instruct-q4_k_m`).
- **`README.md`** — update intro line 3 (drop Ollama, add llama.cpp); replace the Ollama row in the provider table (line 163) with a llama.cpp row pointing at `[llamacpp]` extra and listing the introspection methods. Add a short "Running llama-server" section with the canonical CLI invocation:

  ```
  llama-server -m models/qwen2.5-3b-q4.gguf --host 0.0.0.0 --port 8080 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --slot-save-path ./slot_cache --metrics
  ```

  …and a recipe for "one server per KV cache config".

- **`.env.example`** — replace the `OLLAMA_HOST` / `OLLAMA_TEST_MODEL` block with `LLAMACPP_HOST` / `LLAMACPP_MODEL`.
- **`CLAUDE.md`** — in the project overview update the provider list. In **Provider quirks** replace the Ollama bullet with a llama.cpp bullet covering: server-launch knobs (KV quant, `--slot-save-path`, `--ctx-size`), per-request samplers via `extra_body` and the first-class `min_p`, the `health` / `slots` / `save_slot` / `restore_slot` / `erase_slot` methods, that `tool_choice` is supported (unlike Ollama). Update the "Adding a new provider" example accordingly.
- **`testapp/dnd_gift.py`** — switch the demo (lines 433–437) from `ollama.new_provider("ollama", temperature=0.7)` + `"llama3.2:3b"` to `llm.new_provider("llamacpp", base_url="http://localhost:8080/v1", temperature=0.7)` and a model id like `"qwen2.5-3b-instruct-q4_k_m"`. Drop `context_size=16384` (it's now server-launch); replace `max_tokens=512` with the same.

### Delete

- **`src/llmfacade/providers/ollama.py`**
- **`tests/test_ollama_finish_reason.py`**
- **`tests/integration/test_ollama_live.py`**

## Capability surface (`SUPPORTS`)

```python
SUPPORTS: frozenset[str] = frozenset({
    "temperature", "max_tokens", "top_p", "top_k", "min_p",
    "repeat_penalty", "output_format", "extra_body",
    "tools", "tool_choice",
})
```

Notable absences (deliberate):
- `context_size` — server-launch only. Document.
- `keep_alive` — n/a (server holds the model for its lifetime).
- `thinking` / `effort` / `cache_ttl` / `auto_cache_*` / `beta_headers` / `user_metadata` — provider/cloud-specific, not relevant.

## Critical files / functions to reuse

- `src/llmfacade/providers/openai.py` — the implementation pattern for OpenAI-SDK-based providers. Copy its parsing helpers (`_message_to_api`, `_parse_response`, `_chunk_to_events`) into `llamacpp.py` rather than importing private symbols. Adapt the `_build_kwargs` to inject `extra_body`.
- `src/llmfacade/helpers.py::flatten_text_blocks` — for tool-result text flattening, same as Ollama uses today.
- `src/llmfacade/provider.py::Provider._validate_knobs` and `_filter_unsupported` — gate the cascade automatically once `SUPPORTS` is correct; no provider-side wiring needed beyond declaring it.

## Verification

1. **Unit tests:** `pytest tests/test_llamacpp.py tests/test_tool_choice.py` — pass without a live server.
2. **Type/lint:** `ruff check src/ tests/` and `ruff format --check src/ tests/`.
3. **Cross-provider regressions:** `pytest -k "not integration"` — confirms removing `keep_alive`/`context_size` from `RUNTIME_KNOBS` and the Provider __init__ doesn't break anything outside the deleted Ollama suite.
4. **Live integration (with explicit user approval — these tests are gated):**
   - Launch llama-server: `llama-server -m <gguf> --port 8080 --slot-save-path ./slot_cache --metrics`
   - `LLAMACPP_HOST=http://localhost:8080 LLAMACPP_MODEL=<id> pytest -m integration tests/integration/test_llamacpp_live.py`
   - Confirms: chat works, streaming works, tools work, `slots()` shows live state during a long generation, `save_slot()`/`restore_slot()`/`erase_slot()` round-trip a KV file on disk.
5. **Demo:** run `python testapp/dnd_gift.py` against the local llama-server and confirm it produces output equivalent to the Ollama version.
6. **Smoke for the headline pain points:**
   - Send a request with `repeat_penalty=1.3`, observe via `/slots` that the sampling param is applied (Ollama #15783 silently dropped it).
   - Run two servers with `--cache-type-k q8_0` and `--cache-type-k f16` respectively, instantiate two `LlamaCppServerProvider` objects with different `base_url`s, confirm both work — proves per-model KV quant.
   - During a multi-second generation, poll `provider.slots()` and observe the active slot's `is_processing=true` and decreasing `n_remain` — proves "model is working" introspection.
   - After a long prompt, call `save_slot(0, "warmup.bin")`, restart the server, call `restore_slot(0, "warmup.bin")`, send a follow-up message — confirm the prompt prefix didn't need re-evaluation.
   - **Context size honored on OpenAI-compat endpoint** (the bug that drove this migration — Ollama silently ignored `num_ctx` on its `/v1/chat/completions`): launch with `--ctx-size 32768 --parallel 1`, send a long prompt to `/v1/chat/completions`, then `provider.slots()` and verify the active slot's `n_ctx == 32768` and `n_past` reflects the actual prompt length. Should not silently fall back to a smaller default.

---

## Phase 2: server lifecycle via llama-swap

Phase 1 makes the user run `llama-server` themselves. The user wants the launch-time knobs (`ctx_size`, `cache_type_k/v`, `n_gpu_layers`, etc.) to participate in the cascade so they can be set per-model or per-conversation, with LLMFacade managing server start/stop and avoiding RAM/GPU contention.

**llama-swap (mostlygeek/llama-swap, MIT, 3.8k★, Go) already solves this** and is more battle-tested than anything we'd ship. It's a proxy in front of one or more llama-server instances: clients send OpenAI-compatible requests with a `model` field, and llama-swap loads the matching backend (per the YAML config) on demand, unloading the previous one. Per-model `ttl` handles idle eviction. Native endpoints `GET /running` and `POST /api/models/unload[/:model_id]` give the introspection / explicit-unload story.

This collapses our planned pool from "spawn subprocesses, port-allocate, refcount, evict" to "document that llama-swap is the recommended deployment, add a few thin client methods, optionally help generate the YAML."

### What Phase 2 actually delivers

#### Provider methods that target llama-swap's native API
- `LlamaCppServerProvider.running() -> list[dict]` — `GET /running`. Currently-loaded backends.
- `LlamaCppServerProvider.unload(model_id: str) -> None` — `POST /api/models/unload/{model_id}`.
- `LlamaCppServerProvider.unload_all() -> None` — `POST /api/models/unload`.
- All raise `UnsupportedFeature("llama-swap not detected at base_url")` on 404 (i.e., when pointed at bare llama-server).

#### Optional config helper
`src/llmfacade/providers/llamacpp_swap_config.py` (~120 lines): a Python builder that emits a llama-swap YAML from a list of `(model_id, gguf_path, launch_flags, ttl)` records. Lets users define variants in Python and dump the config — keeps the cascade ergonomics without LLMFacade owning the process.

#### Recommended YAML the user writes (or generates)
```yaml
healthCheckTimeout: 60
models:
  qwen-3b-fast:
    ttl: 600
    cmd: |
      llama-server --port ${PORT} --model models/qwen2.5-3b-q4.gguf
                   --ctx-size 8192 --cache-type-k f16 --cache-type-v f16
                   --slot-save-path ./slot_cache --metrics
  qwen-3b-quality:
    ttl: 600
    cmd: |
      llama-server --port ${PORT} --model models/qwen2.5-3b-q4.gguf
                   --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0
                   --slot-save-path ./slot_cache --metrics
  llama-8b-quality:
    ttl: 600
    cmd: |
      llama-server --port ${PORT} --model models/llama-8b-q4.gguf
                   --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q8_0
                   --metrics
```

User code:
```python
provider = llm.new_provider("llamacpp", base_url="http://localhost:9090/v1")
fast    = provider.new_model("qwen-3b-fast")
quality = provider.new_model("qwen-3b-quality")
big     = provider.new_model("llama-8b-quality")
# Switching between fast / quality / big triggers llama-swap's automatic unload+load.
```

The "model id" in Python becomes the YAML key, not the GGUF path. The cascade still works inside LLMFacade — different `Model` objects route to different llama-swap entries, which encode different launch settings.

### Resource governance (the user's RAM/GPU concern)

llama-swap defaults to **single-model-active** (loading model B unloads model A). That's exactly the safe default we wanted (`max_concurrent=1` in the abandoned pool design). For concurrent loads it has a `matrix` DSL, but no explicit RAM/VRAM budget caps — for single-GPU users this is moot, and for multi-GPU users they'd be tuning at the OS level anyway.

### Honest tradeoffs vs a custom in-Python pool

- **No dynamic per-conversation launch knobs**. Variants must be declared in YAML ahead of time. The config helper softens this (define them in Python, regenerate config) but it's not "set `cache_type_k` on the conversation and have it just work." If you need a new combination, edit YAML and reload llama-swap.
- **No GPU/RAM budget enforcement** beyond llama-swap's single-active-model default. We're trusting the user's YAML.
- **One more process to run** (llama-swap), but the user was already going to run llama-server, and llama-swap replaces it as the front door.

These are acceptable in exchange for not shipping (and maintaining) a Python subprocess pool, port allocator, refcount manager, eviction policy, and atexit cleanup.

### Open question to verify at implementation time

Does llama-swap proxy non-OpenAI paths through to the active backend? Specifically: does `GET /slots`, `POST /slots/{id}?action=save` work when the request goes through llama-swap's port? If not, document "run llama-swap for routing, add a second `base_url` pointing at the bare llama-server for introspection" as a workaround. This is a small risk and discoverable on first integration test.

### File-level changes for Phase 2

#### Add
- `src/llmfacade/providers/llamacpp_swap_config.py` — YAML builder.
- `tests/test_llamacpp_swap_config.py` — round-trip + schema tests; no live llama-swap needed.
- `tests/integration/test_llamacpp_swap_live.py` — drives a real `llama-swap -config <generated.yaml>` running over two models; asserts `running()` reports the active one, `unload(...)` works, and a switch between models triggers a swap.

#### Modify
- `src/llmfacade/providers/llamacpp.py` — add `running` / `unload` / `unload_all` (sync + async). Hit llama-swap's native paths via `httpx`. Map 404 → `UnsupportedFeature`.
- `README.md` — new "Recommended deployment: llama-swap" section with the YAML example and the config-helper one-liner. Note that bare llama-server is still supported.
- `CLAUDE.md` — provider-quirks bullet adds: "Designed to sit behind llama-swap for multi-model lifecycle; bare llama-server also supported. `running` / `unload` / `unload_all` available when behind llama-swap."

#### NOT doing (struck from the original Phase 2 sketch)
Custom in-Python `LlamaCppServerPool`, `ServerFingerprint`, port allocation, refcounting, eviction policy, atexit subprocess management, `LAUNCH_KNOBS` cascade machinery. llama-swap owns it.

### Verification additions for Phase 2
1. **Unit**: config helper emits a YAML that parses back to the same record list.
2. **Unit**: `running()` / `unload()` against a mocked HTTP transport.
3. **Live**: spin up llama-swap with two model entries, run alternating calls — observe one process at a time via `ps`.
4. **Live**: `slots()` / `save_slot()` smoke through llama-swap. Document if non-OpenAI paths require a direct llama-server connection.
