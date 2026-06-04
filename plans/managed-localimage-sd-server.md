# Managed-mode local image generation (sd-server supervisor)

Trello: https://trello.com/c/l13cKtE0 (card `6a21502c`). Follow-up from the
image-generation card (https://trello.com/c/yN1FyNzu).

## Context

The `localimage` provider today is **external mode only**: the caller points
`base_url` at an already-running OpenAI-compatible image server (stable-diffusion.cpp
`sd-server`, or LocalAI) and the provider reuses the `openai` SDK's Images
transport (`/v1/images/generations` + `/v1/images/edits`). This card adds a
**managed mode** that spawns and supervises an `sd-server` subprocess, mirroring
the llamacpp `llama-swap` supervisor: lazy spawn, port allocation, PID file, OS
kill-on-parent-death, atexit/signal handlers, health-wait, explicit shutdown.

## Research findings (sd-server, leejet/stable-diffusion.cpp, early 2026)

Verified against a shallow clone of the repo (`examples/server/{README.md,api.md,
CMakeLists.txt,main.cpp,routes_*.cpp}`):

- **Binary**: CMake target `sd-server` (single executable; not `sd --server`).
- **Listen flags**: `-l/--listen-ip` (default `127.0.0.1`), `--listen-port`
  (default `1234`). There is **no** `--host`/`--port`.
- **Model flags**: `-m/--model` (single-file checkpoint) OR the split-pipeline
  group `--diffusion-model`, `--vae`, `--clip_l`, `--clip_g`, `--t5xxl`, `--llm`,
  `--taesd`; plus `--lora-model-dir`, `-t/--threads`, `--max-vram <GiB>`,
  `--offload-to-cpu`, `--fa`, `--diffusion-fa`, etc.
- **OpenAI-compat API** at `/v1/...`: `POST /v1/images/generations`,
  `POST /v1/images/edits` (multipart), `GET /v1/models`. Response is always
  base64 in `data[].b64_json`. This is exactly the surface the existing
  `_openai_images.py` builds/parses — **no request/response changes needed**.
- **Single model per process**: the model is loaded once at startup (`new_sd_ctx`)
  *before* the server begins listening, guarded by one mutex. `GET /v1/models`
  returns a fixed id `sd-cpp-local`. There is **no llama-swap equivalent** — no
  on-demand multi-model routing.
- **No `/health` endpoint.** Readiness signal: `GET /v1/models` returns `<400`.
  Because the server only listens after the model loads, a 200 there means ready.

## Design decisions (confirmed with user)

1. **Single process, swap on demand.** One `sd-server` resident at a time. When a
   `generate_image` targets a *different* registered model than the one loaded,
   the supervisor tears down the current process and spawns the new one. Only one
   model holds VRAM at a time; cross-model calls serialize and a swap pays a full
   reload. (No idle-TTL auto-unload in this PR.)
2. **Full parity** with the llamacpp supervisor's lifecycle hardening: PID file,
   Win32 Job Object / POSIX `prctl(PR_SET_PDEATHSIG)` kill-on-parent-death,
   atexit + signal handlers, PID-file sweep for orphan recovery, lazy spawn,
   health-wait, idempotent `shutdown()`.

## Design

### New file: `src/llmfacade/providers/_sd_launch.py`

Image-launch bookkeeping, the analog of `_launch.py` (kept separate from the
supervisor so it's unit-testable without subprocess plumbing).

- `SD_LAUNCH_KNOBS: frozenset[str]` — the managed-mode launch knob names.
- `@dataclass(frozen=True, slots=True) _SdLaunchEntry`:
  `model_id`, `model`, `diffusion_model`, `vae`, `clip_l`, `clip_g`, `t5xxl`,
  `llm`, `taesd`, `lora_model_dir`, `threads: int|None`, `max_vram: float|None`,
  `offload_to_cpu: bool=False`, `fa: bool=False`, `diffusion_fa: bool=False`,
  `extra_args: tuple[str,...]=()`.
- `canonical_sd_launch_json(cfg)` — deterministic JSON for hashing; resolves the
  file-path knobs via `Path.resolve()`; drops `None`; tuples→lists.
- `derive_image_model_id(cfg, name)` — `name` wins; else `<stem>-<hash8>` where
  `stem = Path(model or diffusion_model).stem`. Raises if neither model source set.
- `default_provider_sd_defaults()` — hardcoded provider-scope launch defaults
  (all `None`/`False`/`()`), cascaded `provider < model` like llamacpp.
- `build_sd_server_argv(binary, entry, *, port, listen_ip="127.0.0.1")` — pure
  function returning the argv list (`--listen-ip`, `--listen-port`, then the
  model/perf flags, then `extra_args`). Unit-testable.

### New file: `src/llmfacade/providers/_sd_lifecycle.py`

`class _SdServerSupervisor` — owns at most one `sd-server` subprocess.

Reuses the generic OS-process primitives already in `_swap_lifecycle.py`
(imported at module top so the new supervisor's tests can monkeypatch them on
this module): `_spawn_with_pdeathsig`, `_pick_free_localhost_port`,
`_pid_file_sweep`, `_hard_kill_tree`. (These are llama-swap-agnostic; a comment
notes the shared origin. No change to `_swap_lifecycle.py` itself → zero blast
radius on llamacpp.)

State: `_entries: dict[str,_SdLaunchEntry]`, `_proc`, `_anchor`, `_port`,
`_current_model_id`, `_log_file`, an `RLock`, exit-hook bookkeeping, session uuid.

- `register(entry)` — store keyed by `model_id`; re-register with same id+params
  is a no-op; same id, different params raises `ValueError` (parity w/ llamacpp).
- `ensure_model(model_id) -> str` (base_url) — under the lock: validate
  registered; if the right model is already loaded and alive, return its base_url;
  otherwise gracefully stop the current process and spawn the requested one,
  health-wait, record `_current_model_id`, return base_url.
- `_spawn_locked(entry)` — `which(binary)` (else `ProviderNotInstalledError` with
  a build hint), mkdir session + logs dir, `_pid_file_sweep`, pick free port,
  build argv, spawn with kill-on-parent-death, write PID file
  (`pid|port|model_id|uuid`), install exit hooks, `_wait_for_ready` on
  `GET /v1/models`. On failure, clean up and raise `ProviderError` with the log tail.
- `_stop_current_locked(*, graceful=True)` — terminate + wait(timeout) + kill;
  reset proc/port/current; close log. Used by both swap and `shutdown()`.
- `shutdown()` — idempotent full teardown + PID-file unlink (atexit/signal funnel).
- Properties: `base_url`, `pid_file` (`sd-server.pid`), `log_path`
  (`logs/sd-server.log`), `current_model_id`, `is_started`.
- `STARTUP_TIMEOUT_SECONDS` (default 300 — image model loads are slow; overridable
  via the provider's `startup_timeout=`).

### `src/llmfacade/providers/localimage.py`

- Add `__init__(*, manager, api_key, base_url, log_dir, cache_dir, cache_mode,
  llmfacade_dir=None, binary="sd-server", startup_timeout=None, **launch_knobs)`.
  `self._managed = base_url is None`. Managed: build the supervisor + merge
  provider-level launch defaults (`default_provider_sd_defaults` < explicit).
  External: any launch knob present → `UnsupportedFeature` (parity w/ llamacpp);
  no supervisor. Forward only the base kwargs to `super().__init__`.
- `_init_client()` — **drop the hard `base_url` requirement.** External: build the
  openai sync/async clients now (as today). Managed: defer; create the locks
  (`_client_lock`, sync `_gen_lock`, async `_gen_alock`) and leave clients `None`.
- `new_image_model(model_id=None, *, name=None, capability_override=None,
  <launch knobs>, n=None, size=None, aspect_ratio=None, quality=None,
  background=None, output_format=None)`:
  - External: reject launch knobs / `name`; require positional `model_id`;
    behave like the base (return an `ImageModel`).
  - Managed: cascade provider-defaults < model overrides; positional `model_id`
    aliases `name` (conflict raises); require a model source (`model` or
    `diffusion_model`); existence-check every provided file-path knob
    (`FileNotFoundError`), dir-check `lora_model_dir`; derive the id; register the
    entry; return an `ImageModel` bound to the derived id (so
    `image_model.generate()` routes correctly).
- `generate_image` / `agenerate_image`:
  - External: unchanged.
  - Managed: hold `_gen_lock` (sync) / `_gen_alock` (async) across the whole
    op so a concurrent call can't swap the model out mid-request; resolve the
    target id (`model=`, or infer when exactly one is registered, else raise);
    `base = supervisor.ensure_model(id)`; `_ensure_image_client(base + "/v1")`
    (rebuild openai clients only when the port changed); then the existing
    `_image_kwargs` → `images.generate`/`images.edit` → `parse_images_response`
    → `_apply_save_dir` path.
- `shutdown()` — tears down the managed subprocess (no-op external). Idempotent.

### `SUPPORTS`

Unchanged: `frozenset({"image_generation"})`. Managed mode adds no runtime knobs.

### Cascade / capability behaviour

- Launch knobs cascade `provider < model` (managed only), mirroring llamacpp.
- `"image_generation"` capability flag unchanged; `ImageModel.capability_override`
  still narrows per model. A registered model is still nominally
  `is_available("image_generation")`.

## Tests (no integration / no real sd-server)

- `tests/test_sd_launch.py` — `_SdLaunchEntry` defaults; `derive_image_model_id`
  determinism + name-wins + raises with no model source; `build_sd_server_argv`
  emits the right flags in a stable order (incl. `extra_args` tail, bools as
  presence flags, `--listen-port`); provider-default cascade.
- `tests/test_localimage_supervisor.py` — mirror `test_llamacpp_supervisor.py`
  with monkeypatched `_pick_free_localhost_port` / `_spawn_with_pdeathsig` /
  health: `register` + conflict; `ensure_model` spawns + health-waits; requesting
  a *different* model stops the old proc and spawns the new (assert the old was
  terminated); `ensure_model` no-ops when the right model is already loaded;
  `shutdown` idempotent + unlinks PID file; PID-file written with the
  `pid|port|model|uuid` shape; ready-timeout raises `ProviderError` with log tail;
  missing binary raises `ProviderNotInstalledError`.
- Extend `tests/test_image_generation.py` — managed-mode `LocalImageProvider`
  with a fake supervisor (monkeypatched) and a fake openai client:
  `generate_image` routes through `ensure_model` and rebuilds the client against
  the returned port; single-registered-model inference; external mode unchanged;
  launch knobs / `name=` rejected in external mode; `new_image_model` managed
  registration derives an id and returns a bound `ImageModel`; missing model
  source / missing file raise.

## Out of scope

- **Video generation** (`/sdcpp/v1/vid_gen`), the native async `/sdcpp/v1/*` job
  API, and the `/sdapi/v1/*` WebUI family. OpenAI-compat `/v1/images/*` only.
- **Idle-TTL auto-unload** and **multi-process co-residency** (the two rejected
  supervisor models). Single-process swap-on-demand only.
- **`instant interrupt()`** (llamacpp's cross-thread abort) — image gen is a
  one-shot with no streaming consumer needing it; can be a follow-up.
- **Managed `LocalAI`** — managed mode targets `sd-server` specifically.
- `LLM.generate_image` one-shot does not gain registration; managed mode is used
  via the provider object (`new_provider("localimage") → new_image_model(...)`),
  same as managed llamacpp is used via `new_model`.

## Verification

- `ruff check src/` + `ruff format src/` clean; `python -c "import llmfacade"`.
- `pytest` (unit only) green.
- Manual (needs a real `sd-server` build + a model; flag for the user): register a
  model, `generate_image`, confirm a process spawns on an allocated port,
  `sd-server.pid` is written, the image returns; request a second registered model
  and confirm the first process is replaced; `provider.shutdown()` reaps it.

## CLAUDE.md updates

Rewrite the `localimage` provider-quirks bullet: external + managed modes, the
swap-on-demand supervisor, `new_image_model(model=.../diffusion_model=..., ...)`
registration, `SD_LAUNCH_KNOBS`, `shutdown()`. Remove the "managed mode is
planned" note from the **Future work** section.
