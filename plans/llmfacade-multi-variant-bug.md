# Bug: managed-mode llamacpp routes to first model only when multiple are registered in one process

**Component**: `llmfacade` — managed-mode `llamacpp` provider (with `llama-swap` supervisor)
**Severity**: Medium — silently breaks any benchmark / sweep / multi-tenant workload that touches more than one model in one process
**Discovered during**: MTGAI TC-2 benchmark, 2026-05-03
**Workaround**: run each model in its own Python process

## Summary

In a single process, calling `provider.new_model(...)` and dispatching a request works for the **first** model name. Subsequent models registered in the same process — even with distinct `name=` values — return HTTP 400 `"could not find suitable inference handler for <name>"` from llama-swap, despite being present in the on-disk `swap.yaml`.

## Reproduction

Minimal repro using llmfacade's managed mode (no `base_url=` on the provider):

```python
from llmfacade import LLM

provider = LLM.default().new_provider("llamacpp")

# Register and use model A — works.
model_a = provider.new_model(
    name="vlad-bench-f16",
    gguf="C:/Models/vlad-gemma4-26b-dynamic.gguf",
    context_size=128000,
    cache_type_k="f16",
    cache_type_v="f16",
    n_gpu_layers=35,
)
convo_a = model_a.new_conversation(log_dir=False)
print(convo_a.send("hi", max_tokens=4))   # 200 OK

# Register and use model B (same .gguf, different cache settings) — fails.
model_b = provider.new_model(
    name="vlad-bench-q8_0",
    gguf="C:/Models/vlad-gemma4-26b-dynamic.gguf",
    context_size=128000,
    cache_type_k="q8_0",
    cache_type_v="q8_0",
    n_gpu_layers=-1,
)
convo_b = model_b.new_conversation(log_dir=False)
print(convo_b.send("hi", max_tokens=4))
# llmfacade.exceptions.ProviderError: Error code: 400 -
#   {'error': 'could not find suitable inference handler for vlad-bench-q8_0'}

# Same pattern for any third model registered after.
```

Reproduces deterministically, both with successful first-model inference and with first-model crashes (e.g. when the first-model OOMs mid-request, second-model registration also returns 400).

Tested matrix that all hit the same failure:

| First model | Second model | Result |
|---|---|---|
| `*-bench-f16` (load OK, extract OOMed) | `*-bench-q8_0` | 400 |
| `*-bench-f16` (load OK, extract completed at n_gpu_layers=35) | `*-bench-q8_0` | 400 |
| `*-bench-f16` (load OK, extract completed) | `*-bench-q4_0` | 400 |

## What I observed

### `swap.yaml` on disk — contains all three entries

Verified by reading `<repo>/.llmfacade/swap.yaml` after the failure:

```yaml
healthCheckTimeout: 60
models:
  vlad-gemma4-26b-dynamic-bench-f16:
    cmd: llama-server --model C:\Models\vlad-gemma4-26b-dynamic.gguf --port ${PORT}
      --ctx-size 128000 --cache-type-k f16 --cache-type-v f16 --n-gpu-layers 35 --parallel
      1 --slot-save-path C:\Programming\MTGAI\backend\.llmfacade\slots
    ttl: 0
  vlad-gemma4-26b-dynamic-bench-q8_0:
    cmd: llama-server --model C:\Models\vlad-gemma4-26b-dynamic.gguf --port ${PORT}
      --ctx-size 128000 --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers -1
      --parallel 1 --slot-save-path C:\Programming\MTGAI\backend\.llmfacade\slots
    ttl: 0
  vlad-gemma4-26b-dynamic-bench-q4_0:
    cmd: llama-server --model C:\Models\vlad-gemma4-26b-dynamic.gguf --port ${PORT}
      --ctx-size 128000 --cache-type-k q4_0 --cache-type-v q4_0 --n-gpu-layers -1
      --parallel 1 --slot-save-path C:\Programming\MTGAI\backend\.llmfacade\slots
    ttl: 0
```

So the YAML write step **does** happen for every variant. The breakage is downstream.

### llama-swap access log — first model serves 200, subsequent models get 400 in 0 ms

```
Watching configuration for changes (poll-based, 2s interval)
[INFO] Request 127.0.0.1 "GET /health HTTP/1.1" 200 2 "python-httpx/0.28.1" 0s
[INFO] <vlad-gemma4-26b-dynamic-bench-f16> Health check passed on http://localhost:5800/health
[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 648 "OpenAI/Python 2.32.0" 10.65s   # f16 flare OK
... (heavy extraction request runs through without further log entries) ...
[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 400 92 "OpenAI/Python 2.32.0" 0s        # q8_0 flare
[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 400 92 "OpenAI/Python 2.32.0" 0s        # q4_0 flare
```

The 0 ms response time on the 400s rules out network timeouts and rules out llama-swap stalling on a model load. llama-swap is replying immediately that it doesn't recognise the model name.

### Workaround: per-process isolation works

Splitting the benchmark into three separate Python invocations (each with `--variants <single>`) succeeds for every variant. Each new process spawns a fresh `llama-swap` supervisor with a fresh `swap.yaml`, and the first model in that process always works.

This is what the MTGAI TC-2 benchmark does in practice — it's a real cost: each run pays the supervisor-startup overhead (~1–2 s) and the swap.yaml regeneration, and you can't share the running supervisor across variants for a sweep.

## Hypothesis

llama-swap's `-watch-config` polls every 2 s. The flare-probe request follows the `provider.new_model()` call essentially immediately (microseconds). My best guess at the root cause:

**`provider.new_model()` writes the YAML, but doesn't trigger or wait for llama-swap's config reload** before returning the `Model` object. The first call happens to work because llama-swap reloads the YAML during the supervisor's startup window. Subsequent calls land in the gap before the next 2 s poll fires, so llama-swap's in-memory routing table doesn't yet contain the new entry.

If correct, the fix is one of:
1. **Block on llama-swap reload after writing the YAML.** llama-swap likely exposes a `/reload` or `/upstream/refresh` endpoint — call it synchronously and wait for `200`. Or poll `/upstream/<name>` until it shows up. Both are O(50–500 ms) on a healthy supervisor.
2. **Drop poll-based watching and write to llama-swap via its admin API.** llama-swap's REST API supports `PUT /upstream/<name>` (or equivalent) for live model registration. Skips the YAML round-trip entirely.
3. **Pre-register all models at supervisor start time** via a single YAML write before the supervisor is spawned, and forbid `provider.new_model()` after the supervisor is live. (More restrictive but simpler.)

I haven't dug into llama-swap's source to confirm the right knob — option 1 is the smallest behavioural change, option 2 is the most robust.

## Counter-evidence and what I haven't ruled out

- The YAML on disk is correct, so it's not a serialisation bug in the YAML writer.
- The 0 ms 400 response is from llama-swap, not from llmfacade-side validation, so it's not a client-side cache issue.
- I haven't tested whether **waiting >2 s** between `provider.new_model()` and `convo.send()` makes the second model work. That would confirm the poll-race hypothesis. Easy to add to the repro.
- I haven't tested whether passing a `keep_alive=` or a wait-for-ready helper exists on the Model side — there might already be a fix that just isn't surfaced.

## Impact on MTGAI

- TC-2 benchmark scripts run per-variant in fresh processes (workaround). Adds ~10 s of supervisor cold-start per variant on a 4-variant sweep — annoying for development, irrelevant for production.
- Production MTGAI workflows always touch one model per process (theme extractor process holds Vlad-dynamic for its lifetime), so this bug never fires there. Pure dev-tooling pain right now.
- If MTGAI ever wants to load-balance between two registered models in one process (e.g. small model for triage, big model for hard cases), this bug becomes a blocker.

## Files in the repro

- `C:\Programming\MTGAI\backend\scripts\benchmark_llamacpp_tc2.py` — the original three-variant script. Run with default args reproduces the bug; run with `--variants <single>` works around it.
- `C:\Programming\MTGAI\backend\.llmfacade\swap.yaml` — the YAML state at the time of failure (snippets above).
- `C:\Programming\MTGAI\backend\.llmfacade\logs\llamacpp-swap.log` — llama-swap access log (snippets above).
- `C:\Programming\MTGAI\learnings\llamacpp-tc2-benchmark.md` — the broader benchmark writeup that this bug surfaced from.
