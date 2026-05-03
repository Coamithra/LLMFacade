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
  - `_build_chat_kwargs(req)` mirrors the OpenAI provider's payload assembly. The OpenAI Python SDK doesn't accept `top_k`, `min_p`, or `repeat_penalty` as typed kwargs, so all three are passed through the SDK's `extra_body=` parameter (the SDK forwards that dict verbatim onto the wire). Wire field names: `top_k`, `min_p`, `repeat_penalty` — all as-named on llama-server's `/v1/chat/completions`. `output_format=OutputFormat.JSON` reuses the OpenAI branch: `response_format={"type": "json_object"}`.
    - **No user-facing `extra_body` knob.** llama.cpp has a long tail of additional samplers (`mirostat*`, `dynatemp*`, `dry*`, `xtc*`, `n_keep`, `json_schema`, custom `samplers` ordering). We add them as first-class knobs as-and-when a real use case appears, rather than shipping a free-form escape hatch with undefined cascade-merge semantics. Anything not currently supported is simply not reachable from the facade.
  - `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw` call into the OpenAI SDK; reuse parsing helpers cribbed from `src/llmfacade/providers/openai.py` (do not import private symbols — copy the small helpers).
  - **Introspection methods (synchronous, plus async twins):**
    - `health() -> dict` — `GET /health`. Returns `{"status": "ok"}` or raises `ProviderError` on 503.
    - `slots() -> list[dict]` — `GET /slots`. Per-slot processing state, sampling params, token counts, generation speed.
    - `save_slot(id_slot: int, filename: str) -> dict` — `POST /slots/{id_slot}?action=save` body `{"filename": filename}`.
    - `restore_slot(id_slot: int, filename: str) -> dict` — `POST /slots/{id_slot}?action=restore` body `{"filename": filename}`.
    - `erase_slot(id_slot: int) -> dict` — `POST /slots/{id_slot}?action=erase`.
    - All five hit `self._http_base` directly via `httpx`. Do NOT add `props()` or `metrics()` for now — minimal surface per the user's choice.
    - **Field-name verification:** the live `/slots` response shape has shifted across llama-server builds. During first integration smoke, capture one real `/slots` payload and pin the assertion field names (`n_ctx`, `is_processing`, `n_remain`, sampling under `params` vs. flat) to whatever the running server actually returns. Update verification step 6 below to match.
  - **`count_tokens(text, *, model_id=None)` override.** llama-server exposes `POST /tokenize` (body `{"content": text}` → `{"tokens": [...]}`). Since the server is local by definition this stays in spirit with the "no external network call" rule. CLAUDE.md's existing carve-out (`"Always local — never makes a network call (except Anthropic with exact_count_tokens=True)"`) extends to "and llamacpp, which calls its local server's `/tokenize`". `tokenizer_name()` returns `"llama-server /tokenize"`. Falls back to `chars/4` on connection error so it never blocks logging.
  - Maps `openai.AuthenticationError` → no-op (server doesn't auth), `openai.RateLimitError` → `RateLimitError`, all else → `ProviderError`.

- **`tests/test_llamacpp.py`** — unit tests modelled on `tests/test_ollama_finish_reason.py`:
  - finish_reason translation (`length` → `"length"`, etc.)
  - `top_k` / `min_p` / `repeat_penalty` land on the wire via the SDK's `extra_body=` argument (assert against a mocked OpenAI client)
  - `output_format=OutputFormat.JSON` produces `response_format={"type": "json_object"}`
  - tool-call parsing roundtrip with mocked OpenAI client
  - `health()` / `slots()` / `save_slot()` against a mocked `httpx` transport
  - `count_tokens()` hits `/tokenize` and returns `len(tokens)`; falls back to `chars/4` when the mocked transport refuses connection
- **`tests/integration/test_llamacpp_live.py`** — modelled on `tests/integration/test_ollama_live.py`:
  - tool roundtrip
  - one save_slot → erase → restore cycle to prove the warmup path
  - Skips if server unreachable.

### Modify

- **`src/llmfacade/providers/__init__.py`** — replace the `"ollama"` entry with `"llamacpp": ("llmfacade.providers.llamacpp", "LlamaCppServerProvider")`.
- **`src/llmfacade/settings.py`** — `RUNTIME_KNOBS`: **add** `min_p`. **Remove** `keep_alive` (Ollama-only) and `context_size` (Ollama-only — llama-server takes ctx via `--ctx-size` at launch). Net 14 → 13. `min_p` is first-class because it's the standard recommended knob for local models; gating to llamacpp-only is handled by `SUPPORTS` (the same mechanism that gates `thinking` to Anthropic). `extra_body` is **not** added — see the no-extra-body decision above.
- **`src/llmfacade/provider.py`** — `Provider.__init__` kwargs: drop `keep_alive`, `context_size`; add `min_p`. The `_validate_knobs` / `_filter_unsupported` paths Just Work because they read from `RUNTIME_KNOBS`. **Backwards-compat note:** call sites that passed `keep_alive=` or `context_size=` to non-Ollama providers (which previously raised `UnsupportedFeature`) now raise `TypeError` instead. No public API has ever encouraged this, but document the shift in the migration notes.
- **`pyproject.toml`** — `[project.optional-dependencies]`: drop `ollama = [...]`, add `llamacpp = ["openai>=2.32"]`. Update `all = [...]` to include `llamacpp` and drop `ollama`. The current comment on line 35 ("Anthropic and Ollama have no offline tokenizer") should change to "Anthropic has no offline tokenizer; llamacpp uses the running server's `/tokenize`".
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
- **`testapp/dnd_gift.py`** — switch the demo (lines 433–437) from `ollama.new_provider("ollama", temperature=0.7)` + `"llama3.2:3b"` to `llm.new_provider("llamacpp", base_url="http://localhost:8080/v1", temperature=0.7)` and a model id like `"qwen2.5-3b-instruct-q4_k_m"`. Drop `context_size=16384` (it's now server-launch); keep `max_tokens=512` unchanged.

### Delete

- **`src/llmfacade/providers/ollama.py`**
- **`tests/test_ollama_finish_reason.py`**
- **`tests/integration/test_ollama_live.py`**

## Capability surface (`SUPPORTS`)

```python
SUPPORTS: frozenset[str] = frozenset({
    "temperature", "max_tokens", "top_p", "top_k", "min_p",
    "repeat_penalty", "output_format",
    "tools", "tool_choice",
})
```

Notable absences (deliberate):
- `context_size` — server-launch only. Document.
- `keep_alive` — n/a (server holds the model for its lifetime).
- `extra_body` — no free-form escape hatch. New llama.cpp samplers get individual first-class knobs as needed.
- `thinking` / `effort` / `cache_ttl` / `auto_cache_*` / `beta_headers` / `user_metadata` — provider/cloud-specific, not relevant.

## Critical files / functions to reuse

- `src/llmfacade/providers/openai.py` — the implementation pattern for OpenAI-SDK-based providers. Copy its parsing helpers (`_message_to_api`, `_parse_response`, `_chunk_to_events`) into `llamacpp.py` rather than importing private symbols. Adapt `_build_kwargs` to assemble an `extra_body=` dict at the SDK call level for `top_k` / `min_p` / `repeat_penalty`.
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
7. **Pin `/slots` field names.** During the first integration run, capture one real `/slots` payload (e.g. `print(provider.slots())`) and update verification step 6's assertion field names to match. The shape has shifted across builds; don't assert against fields you haven't seen.

---

## Phase 2: managed lifecycle of llama-server via llama-swap

### Goal

Phase 1 ships a thin HTTP client to a `llama-server` process that the user runs themselves. Phase 2 adds **managed mode**: LLMFacade owns the server lifecycle. The user passes layer-2 launch flags (context size, KV cache quantization, GPU layers, etc.) at `new_model` construction; the library generates llama-swap YAML and supervises a `llama-swap` subprocess that spawns/swaps/unloads `llama-server` instances on demand.

The user never sees YAML. The simplest valid call is `provider.new_model(gguf="models/qwen.gguf")` — Ollama-tier ergonomics with strictly more capability.

This explicitly **un-rejects** the prior Phase 2 sketch's "NOT doing" list (subprocess lifecycle, atexit cleanup, port management). The cost is paid once, inside the library, and `llama-swap` handles the actually-hard parts (port allocation, eviction, single-active-model invariants). We just supervise one process and write a YAML file.

### Two modes (mutually exclusive)

| Mode | Triggered by | Library does |
|---|---|---|
| **External** | `new_provider("llamacpp", base_url="http://...")` | Nothing — talks to whatever's there. (Phase 1 unchanged.) |
| **Managed** | `new_provider("llamacpp")` (no `base_url`) | Lazily owns a `llama-swap` subprocess + its YAML. |

There is no "I'll run llama-swap, you write the YAML" middle ground — keeps the mental model clean.

### Architecture: layers and ownership

```
  User Python ─── LLMFacade ──┬─→ External: bare llama-server (user-owned)
                              │
                              └─→ Managed: llama-swap subprocess (we own)
                                                ├── llama-server #A (per-model)
                                                ├── llama-server #B
                                                └── ... (llama-swap manages)
```

In managed mode, **the LlamaCppServerProvider instance owns**:
- A session directory (default `./.llmfacade/`, override via `llmfacade_dir=`)
- A `swap.yaml` config file inside it
- A `llama-swap` subprocess started lazily on first `convo.send()`
- A PID file at `<dir>/swap.pid` for orphan recovery

### Launch-knob cascade

A new `LAUNCH_KNOBS` set, separate from `RUNTIME_KNOBS`, lives in `src/llmfacade/settings.py`:

```python
LAUNCH_KNOBS: frozenset[str] = frozenset({
    "gguf",              # required at model scope
    "context_size",
    "cache_type_k",
    "cache_type_v",
    "n_gpu_layers",
    "parallel",
    "slot_save_path",
    "ttl",
    "extra_args",        # list[str] escape hatch for flags we don't surface
})
```

These knobs:
- Are valid at **provider** and **model** scope only — never per-call, never on `Conversation`.
- Cascade `provider < model` (later wins).
- Are `LlamaCppServerProvider`-only. Other providers reject them via the same `_validate_knobs` machinery (`LAUNCH_KNOBS` is checked in addition to `RUNTIME_KNOBS`; gating is by per-provider opt-in — see "Validation" below).
- Never appear in `CompletionRequest.settings`. They're consumed by the YAML generator only.
- Are ignored entirely in external mode.

```python
provider = llm.new_provider(
    "llamacpp",
    n_gpu_layers=32,                     # provider-level launch default
    slot_save_path="./.llmfacade/slots", # shared by every model
    default_ttl=0,
)
fast    = provider.new_model(gguf="models/qwen-3b-q4.gguf", context_size=8192,  cache_type_k="f16",  cache_type_v="f16")
quality = provider.new_model(gguf="models/qwen-3b-q4.gguf", context_size=32768, cache_type_k="q8_0", cache_type_v="q8_0")
```

### Validation

`_validate_knobs` (`src/llmfacade/provider.py`) currently checks `RUNTIME_KNOBS`. Extend it with a parallel `_validate_launch_knobs` path:

- The `LlamaCppServerProvider.__init__` and `LlamaCppServerProvider.new_model` accept the `LAUNCH_KNOBS` kwargs explicitly.
- Other providers' `__init__`/`new_model` do **not** accept them — passing them raises `TypeError` (unknown kwarg) at the provider's constructor, same mechanism as today.
- Within LlamaCppServerProvider in **external** mode, passing any `LAUNCH_KNOBS` value raises `UnsupportedFeature` (managed-mode only).
- `gguf` is required at model scope in managed mode and raises a clear `ValueError` if missing.

### Sensible defaults (the "Ollama ergonomic")

The simplest valid call is `provider.new_model(gguf="models/qwen.gguf")`. Defaults:

| Knob | Default | Notes |
|---|---|---|
| `context_size` | unset | falls back to `llama-server`'s per-model default |
| `cache_type_k` / `cache_type_v` | unset | `llama-server` default (fp16) |
| `n_gpu_layers` | unset | `llama-server` default (usually 0) |
| `parallel` | `1` | one slot |
| `slot_save_path` | `<llmfacade_dir>/slots` | provider-level default; created on demand |
| `ttl` | `0` | never unload (matches llama-swap default) |
| `extra_args` | `[]` | escape hatch list of `--flag value` strings appended verbatim |

Provider-level defaults cascade into model-level. Example: setting `slot_save_path` on the provider applies to every model.

We deliberately do **not** ship a Modelfile-style separate config file. The cascade plus hardcoded defaults already meet that ergonomic; adding a parallel config language would compete with the Python API.

### Model id naming and idempotency

```python
def derive_model_id(launch_config: dict, name: str | None) -> str:
    if name is not None:
        return name                                              # user-provided wins
    basename = Path(launch_config["gguf"]).stem
    h = hashlib.sha256(canonical_json(launch_config)).hexdigest()[:8]
    return f"{basename}-{h}"
```

- Default: `<gguf-stem>-<hash8>` (e.g. `qwen2.5-3b-q4-a1b2c3d4`). Readable in logs, uniquely identifies the launch config.
- User: `provider.new_model(name="qwen-fast", gguf=..., ...)` → `name` becomes the model id (and the llama-swap YAML key).
- The same model id is sent as the `model` field in `/v1/chat/completions` requests so llama-swap routes correctly.

**Idempotency**: same launch knobs → same hash → same YAML entry. Calling `new_model(gguf=X, context_size=8192)` twice returns two `Model` objects that refer to the same llama-swap entry; their non-launch defaults (`temperature`, etc.) live on the `Model` instance and don't conflict because `Model` is constructor-immutable. No aliasing risk.

**Name conflicts**: two `new_model` calls with the same `name=` but different launch knobs raise `ValueError("model name 'X' already registered with different launch params: ...")`.

### Lifecycle (lazy spawn, watch-config, shutdown)

1. **`new_provider("llamacpp")`** — no process, no temp files. Constructor records `llmfacade_dir` and provider-level launch defaults only.
2. **`new_model(gguf=..., ...)`** — registers a `_LaunchEntry(model_id, gguf, ...)` in an in-memory list on the provider. No process started. Validates `gguf` exists on disk; raises `FileNotFoundError` if not.
3. **First `convo.send()` on any model** triggers `_LlamaSwapSupervisor.ensure_started()`:
   - Materialise `<llmfacade_dir>/` (mkdir -p).
   - Sweep `<llmfacade_dir>/swap.pid` for stale orphan from a prior run; SIGTERM if alive.
   - Generate `<llmfacade_dir>/swap.yaml` from the registered entries.
   - Pick a free localhost port (bind to `127.0.0.1:0`, read back, release).
   - `subprocess.Popen("llama-swap -config <yaml> -watch-config -listen 127.0.0.1:<port>")` with OS-level kill-on-parent-death (see "Shutdown defense in depth").
   - Pipe stdout+stderr to `<llmfacade_dir>/logs/llamacpp-swap.log`.
   - Write `<llmfacade_dir>/swap.pid` (PID + port + a session UUID so we can detect "this PID file is from THIS process" vs an orphan).
   - Poll `http://127.0.0.1:<port>/health` for up to `health_check_timeout_seconds=60` seconds. Raise `ProviderError` with stderr tail on timeout.
   - Update `self._http_base` to the bound port; the existing OpenAI SDK client is reconfigured (or re-created) with the new base.
4. **Subsequent `new_model`** — appends to the in-memory entry list and rewrites `swap.yaml`. llama-swap's `-watch-config` picks it up automatically (no API call needed; documented behaviour of llama-swap).
5. **Shutdown** (atexit / signal / explicit `provider.shutdown()`):
   - SIGTERM llama-swap; wait up to 10 seconds.
   - SIGKILL on timeout.
   - Best-effort delete `<llmfacade_dir>/swap.pid` (but never delete `<llmfacade_dir>/` itself — user might be inspecting logs).

### Shutdown defense in depth

This is the hard part. atexit alone doesn't cover hard kills, OOM, power loss, etc. Three layers, with **layer 1 the load-bearing one**:

**Layer 1 — OS-level kill-on-parent-death.** Kernel guarantees that survive even `kill -9` of the Python process.

- **Windows**: create a Win32 Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, assign llama-swap's process handle to it. When the Python process's last handle to the job closes (any process exit including hard kill), the OS terminates llama-swap, which in turn terminates the `llama-server` children it spawned.
  - Use `ctypes.windll.kernel32.CreateJobObjectW` + `SetInformationJobObject` + `AssignProcessToJobObject`.
  - Pass `creationflags=subprocess.CREATE_BREAKAWAY_FROM_JOB` only if Python itself is in a job that prevents nesting; otherwise `CREATE_NEW_PROCESS_GROUP` is enough.
- **Linux**: pass `preexec_fn=lambda: prctl(PR_SET_PDEATHSIG, SIGTERM)` to `subprocess.Popen`. Kernel sends `SIGTERM` to the child when its parent dies.
- **macOS**: no kernel equivalent. Falls through to layers 2+3.

These are best-effort but deterministic on Windows and Linux (the user's primary). Implement in a small helper `_spawn_with_pdeathsig(cmd, **popen_kwargs)`.

**Layer 2 — signal handlers and atexit.**
- Install `atexit.register(self.shutdown)`.
- Install `signal.signal(SIGINT, ...)` and `signal.signal(SIGTERM, ...)` handlers that call `self.shutdown()` then re-raise to preserve user behaviour. Store and restore prior handlers so we don't break callers that have their own.
- `shutdown()` is idempotent — multiple calls are safe.

**Layer 3 — PID-file sweep on startup.**
- On `_LlamaSwapSupervisor.ensure_started()`, before spawning a new llama-swap, check `<llmfacade_dir>/swap.pid`.
- If present and parseable, query whether the PID is alive AND is a `llama-swap` process:
  - **Windows**: `tasklist /FI "PID eq <pid>" /NH /FO CSV`, parse, match image name.
  - **Linux/macOS**: read `/proc/<pid>/comm` (Linux) or `ps -p <pid> -o comm=` (portable).
- If yes → SIGTERM it (orphan from prior run that survived layer 1, e.g. macOS or a forced kill that bypassed the job object somehow). Wait, then SIGKILL.
- If no → just remove the stale file.
- Log the recovery to the swap log.

**Together**: layer 1 covers ~99% of failure modes on Windows/Linux. Layer 3 catches the remaining edge cases (macOS, exotic kills, IDE-driven kills that detach from the job). Layer 2 is the "polite" path for normal exits.

### Native llama-swap methods (sync + async)

Add to `LlamaCppServerProvider`:

```python
def running(self) -> list[dict[str, Any]]: ...        # GET /running
def unload(self, model_id: str) -> None: ...           # POST /api/models/unload/{id}
def unload_all(self) -> None: ...                      # POST /api/models/unload
async def arunning(self) -> list[dict[str, Any]]: ...
async def aunload(self, model_id: str) -> None: ...
async def aunload_all(self) -> None: ...
```

In **managed** mode they hit the supervised llama-swap directly. In **external** mode they hit the user-supplied `base_url`; if llama-swap is running there they work, if it's a bare llama-server they get 404 and we raise `UnsupportedFeature("llama-swap not detected at base_url")`.

### Introspection routing (`slots`, `save_slot`, etc.)

**Resolved.** Verified against a live `llama-swap` (see `testapp/probe_swap_introspection.py`): llama-swap does **not** proxy `/health` (returns its own plain-text "OK"), `/slots`, `/tokenize`, or `/slots/{id}?action=*` via the bare URL — all 404. **However**, llama-swap exposes `/upstream/<model_id>/<arbitrary-path>` which forwards to the named backend with on-demand load. The neither-mode-works-the-same outcome is therefore avoidable: we route managed-mode per-backend introspection through `/upstream/`, leaving external mode unchanged.

Implementation lives in the follow-up plan (`~/.claude/plans/starry-leaping-liskov.md`): adds `model: str | None = None` kwarg to provider introspection methods, mirrors them on `Model` (auto-binding the id, like `Model.count_tokens`), special-cases `provider.health()` no-arg to return the swap's own health normalized to `{"status": "ok"}`, and fixes `count_tokens` to actually use `model_id` in managed mode (was silently degrading to chars/4).

### Logging

`llama-swap` stdout+stderr → `<llmfacade_dir>/logs/llamacpp-swap.log` (truncate-on-startup or rotate; a simple truncate-per-run is fine for MVP).

If the LLMFacade `LLM` instance has `log_dir=False` and no explicit `llmfacade_dir=` is passed, fall back to `tempfile.mkdtemp(prefix="llmfacade-llamacpp-")` and clean up on shutdown.

### llama-swap binary discovery

Look up `llama-swap` on PATH (`shutil.which("llama-swap")`). If missing, raise `ProviderNotInstalledError` with:

> llama-swap binary not found on PATH. Install from https://github.com/mostlygeek/llama-swap (e.g. `go install github.com/mostlygeek/llama-swap@latest`) and ensure it's on PATH, or use external mode by passing base_url= to point at an existing llama-server.

Same idea for `llama-server` itself if we want to be helpful — but llama-swap will discover and report that on its own; we don't need to pre-check.

### File-level changes

#### Add

- **`src/llmfacade/providers/_swap_lifecycle.py`** (~200 lines):
  - `class _LlamaSwapSupervisor`: owns the YAML, the subprocess, the PID file, and shutdown plumbing.
  - `_spawn_with_pdeathsig(cmd, **popen_kwargs) -> subprocess.Popen` — Windows Job Object on Windows, `prctl(PR_SET_PDEATHSIG)` on Linux, plain Popen on macOS.
  - `_pid_file_sweep(path: Path) -> None` — read PID file, kill orphan if alive, remove file.
  - `_render_swap_yaml(entries: list[_LaunchEntry], *, global_ttl: int) -> str` — renders the YAML used by llama-swap.

- **`src/llmfacade/providers/_launch.py`** (~80 lines):
  - `LAUNCH_KNOBS: frozenset[str]` (re-exported via `settings.py`).
  - `@dataclass(frozen=True, slots=True) class _LaunchEntry` — the per-model launch config.
  - `derive_model_id(entry: _LaunchEntry, name: str | None) -> str`.
  - `default_provider_launch_defaults(llmfacade_dir: Path) -> dict` — the hardcoded defaults table from above.

- **`tests/test_llamacpp_swap_yaml.py`** — pure unit tests for `_render_swap_yaml`: minimum-viable entry, all-knobs-set entry, multiple entries, deterministic ordering, escaping of paths with spaces, `extra_args` passthrough.

- **`tests/test_llamacpp_supervisor.py`** — `_LlamaSwapSupervisor` tests against a stubbed subprocess + filesystem. Covers: lazy startup, PID-file write/sweep, idempotent shutdown, signal handler installation/restoration, orphan detection logic (mock `psutil`-style or platform-specific calls). No live `llama-swap` needed.

- **`tests/integration/test_llamacpp_swap_live.py`** — drives a real `llama-swap` subprocess (skipped if binary missing). Covers:
  - Two `new_model` calls with different launch params → both YAML entries present → first `send` triggers spawn → second `send` to the other model triggers swap.
  - `running()` reports the active model.
  - `unload(model_id)` works; subsequent `running()` reflects it.
  - Smoke test introspection: try `provider.slots()` while a model is loaded; record outcome (works / 404). Update README/CLAUDE.md based on result.
  - Process cleanup: assert the spawned `llama-swap` is gone after the test ends (use `psutil` to walk children, or just check the PID file).

#### Modify

- **`src/llmfacade/settings.py`** — add `LAUNCH_KNOBS` frozenset alongside `RUNTIME_KNOBS`.
- **`src/llmfacade/provider.py`** — extend `_validate_knobs` (or add `_validate_launch_knobs`) so launch-knob keys validate against a launch SUPPORTS set if the provider declares one. Keep cross-provider behaviour unchanged.
- **`src/llmfacade/providers/llamacpp.py`** — substantial:
  - Constructor grows `llmfacade_dir`, `default_ttl`, plus all provider-scope `LAUNCH_KNOBS` defaults.
  - `_init_client()` no longer requires `base_url`; if absent, switches to managed mode.
  - `new_model` accepts `gguf=`, `name=`, all `LAUNCH_KNOBS`, plus the existing `RUNTIME_KNOBS`.
  - Adds `running` / `unload` / `unload_all` (sync + async). Wire 404 → `UnsupportedFeature`.
  - Adds `shutdown()` method for explicit teardown.
  - Holds a `_LlamaSwapSupervisor | None` (None in external mode).
  - First call to `_complete_raw` / `_acomplete_raw` / `_stream_raw` / `_astream_raw` calls `supervisor.ensure_started()` before the request.
- **`src/llmfacade/model.py`** — `Model.new_model` and `Model.__init__` need to accept `LAUNCH_KNOBS` kwargs only when the provider opts in (or just accept them and let validation reject for non-llamacpp providers). Simpler path: extend `_validate_knobs` to know about `LAUNCH_KNOBS`, gated by a per-provider declaration.
- **`pyproject.toml`** — `llamacpp` extra grows `pyyaml>=6.0` for emitting the swap YAML deterministically. (Or roll our own YAML emitter for the small subset we use; prefer the lib since this is a one-line dep.)
- **`README.md`** — add "Managed mode" section showing the zero-YAML flow; keep the existing "Running llama-server" section as the external-mode docs.
- **`CLAUDE.md`** — provider-quirks bullet expands: managed vs external mode, lifecycle ownership, `running`/`unload`/`unload_all`, shutdown defense in depth, the `<llmfacade_dir>` layout.
- **`.env.example`** — add `LLAMACPP_USE_MANAGED=1` env-var hint? (Optional; the test fixture can be plain.)

### Verification

1. **Unit (`pytest tests/test_llamacpp_swap_yaml.py tests/test_llamacpp_supervisor.py`)**:
   - YAML rendering deterministic, idempotent, handles all knobs.
   - Supervisor: lazy start, PID-file sweep, idempotent shutdown, signal-handler installation/restoration, orphan kill on stale PID.
   - No live `llama-swap` needed.

2. **Lint**: `ruff check src/ tests/` and `ruff format --check src/ tests/` pass.

3. **Cross-provider regressions**: `pytest -k "not integration"` stays green — adding `LAUNCH_KNOBS` doesn't break other providers.

4. **Live integration (gated, requires user permission)**:
   - Binary present: `which llama-swap` and `which llama-server`.
   - `LLAMACPP_HOST` is **unset** so the test exercises managed mode.
   - Run `pytest -m integration tests/integration/test_llamacpp_swap_live.py`.
   - Confirms: launch, two-entry YAML, model swap, `running()`/`unload()`, introspection probe outcome, clean shutdown (no surviving llama-swap or llama-server processes).

5. **Smoke (manual, with user approval)**:
   - **Hard-kill recovery**: start a Python script that spawns the supervisor, then `kill -9` the Python process (or end-task on Windows). Verify llama-swap and its child are gone within seconds (layer-1 OS guarantee). On macOS, verify the next `new_provider("llamacpp")` call cleanly recovers via PID-file sweep.
   - **Two providers in same Python process**: each gets its own `llmfacade_dir` and llama-swap subprocess; no cross-contamination.

### Honest tradeoffs

- **One more dependency** (`pyyaml`) and one more required external binary (`llama-swap`). Documented as a managed-mode prerequisite.
- **Multi-process coordination is out of scope.** Two Python processes running managed-mode providers each spawn their own `llama-swap`; if they both try to use the same GPU, llama-swap's single-active-model invariant doesn't span instances. We don't try to solve this — document that managed mode is "one Python process, one provider, one llama-swap."
- **macOS shutdown is best-effort.** Layer 1 isn't available; we lean on layers 2+3. PID-file sweep on next start cleans up any orphan, so user-visible behaviour is "your old llama-swap might survive a hard kill until you next call `new_provider`, then it's reaped."
- **Introspection-through-llama-swap is unverified.** Phase 2 ships and observes; we adapt docs (and possibly add `introspection_base_url=`) only if real users hit the limitation.
- **Phase 1's external-mode use case stays first-class.** Anyone with their own llama-server keeps working.
