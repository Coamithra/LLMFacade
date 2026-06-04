# Plan: add `no_mmap` and `mlock` launch knobs (llamacpp managed mode)

_Spec'd 2026-06-04. Motivated by `docs/learnings/llamacpp-low-vram-moe-flags.md` (Codacus
"35B on 6 GB" video). Two of the five flags that take a low-VRAM MoE from 3 → 17 t/s are
not first-class facade knobs — only reachable via the `extra_args` escape hatch._

## Goal

Promote llama-server's `--no-mmap` and `--mlock` to first-class managed-mode launch knobs:

- **`no_mmap: bool = False`** → emits `--no-mmap`. Reads the whole model into RAM up front
  instead of mmap'ing it. On a low-VRAM MoE with experts in system RAM, the default mmap
  demand-pages experts from disk mid-token; `--no-mmap` removes those page faults. Measured
  **+35%** (10 → 13.5 t/s) in the source video.
- **`mlock: bool = False`** → emits `--mlock`. Pins the (already-RAM-resident) expert pages so
  the kernel can't page them back to disk under memory pressure/idle. Not a speed flag — it's
  what stops the "day-3 slowdown" on a long-running server. Directly relevant because **managed
  mode is a long-lived supervised backend**.

Both are **boolean flags** (like the existing `jinja`), default **off**, and **output-neutral**
(they change *where bytes live*, not *what the model emits*), so both must be **excluded from the
`<gguf-stem>-<hash8>` derivation** — toggling them must not shift `model_id` or break slot-cache
continuity. Same treatment as `fit`.

## Why opt-in, not default

- `--no-mmap` forces a full preload, so it **requires the model to fit in RAM**. On a tight-RAM
  box it backfires or fails to load. Document this clearly; never default it on.
- `--mlock` under Docker silently no-ops unless three things align: the container `memlock`
  ulimit is raised, the `IPC_LOCK` capability is granted, **and** `--mlock` is passed. We don't
  ship Docker, but the knob docstring should warn (no error is raised — it just leaks back to the
  default paging behaviour).

## Touch-points (file by file)

1. **`src/llmfacade/settings.py`** — add `"no_mmap"` and `"mlock"` to the `LAUNCH_KNOBS`
   frozenset. (This alone makes external mode reject them with `UnsupportedFeature`, since the
   external-mode guard rejects any `LAUNCH_KNOBS` member — verify that guard path covers them.)

2. **`src/llmfacade/providers/_launch.py`**
   - `_LaunchEntry` — add `no_mmap: bool = False` and `mlock: bool = False` fields.
   - `_HASH_EXCLUDED_KEYS` — add `"no_mmap"` and `"mlock"` (output-neutral, like `fit*`). Update
     the surrounding docstring to explain *why* (memory residency ≠ generation behaviour).
   - `default_provider_launch_defaults(...)` — add `"no_mmap": False, "mlock": False`.

3. **`src/llmfacade/providers/_swap_lifecycle.py::_build_llama_server_cmd`** — append `--no-mmap`
   / `--mlock` when the respective field is `True` (boolean-flag pattern, mirror the `jinja`
   block at lines 57-61).

4. **`src/llmfacade/providers/llamacpp.py`**
   - Provider `__init__` (provider-level defaults, ~L161-222): add both kwargs + dict entries.
   - `new_model(...)` (~L387-430): add both kwargs to the signature + the merged dict.
   - `_LaunchEntry(...)` construction (~L533): pass `no_mmap=merged.get("no_mmap")`,
     `mlock=merged.get("mlock")`.
   - `_maybe_estimate_fit` (~L1312-1332): **do NOT forward** `--no-mmap`/`--mlock` to
     `llama-fit-params`. They don't affect the VRAM fit, and `--no-mmap` would force the probe to
     load the whole model into RAM, blowing the sub-second/15 s-capped probe budget. Add a one-line
     comment next to the existing `extra_args`-not-forwarded rationale.

5. **`CLAUDE.md`** — add `no_mmap`, `mlock` to the `LAUNCH_KNOBS` enumeration in the llamacpp
   provider-quirks section, and add a short **"Memory residency (`no_mmap` / `mlock`)"** bullet
   covering: the low-VRAM-MoE motivation, the opt-in RAM-fit caveat, hash-exclusion, the
   not-forwarded-to-fit-params note, and the Docker `IPC_LOCK`/ulimit gotcha for `mlock`.

## Tests (`tests/` — all local/free, no integration)

- **Rendering:** `no_mmap=True` / `mlock=True` ⇒ `--no-mmap` / `--mlock` present in the rendered
  swap.yaml command; default (`False`) ⇒ absent. (Extend the existing `_build_llama_server_cmd` /
  `_render_swap_yaml` tests.)
- **Hash exclusion:** two `new_model` calls differing only in `no_mmap` (and only in `mlock`)
  derive the **same** `model_id`. Mirror the existing `fit`-exclusion test.
- **External-mode rejection:** constructing an external-mode provider (or `new_model`) with
  `no_mmap=`/`mlock=` raises `UnsupportedFeature`, like the other `LAUNCH_KNOBS`.
- **Cascade:** provider-level `no_mmap=True` propagates to a `new_model` that doesn't override it;
  a model-level `no_mmap=False` overrides a provider-level `True`.

## Acceptance criteria

- `provider.new_model(gguf=..., n_cpu_moe=41, no_mmap=True, mlock=True)` renders a swap.yaml whose
  llama-server line contains `--n-cpu-moe 41 --no-mmap --mlock`.
- Toggling either knob does not change `model_id`.
- External mode rejects both with `UnsupportedFeature`.
- `llama-fit-params` probe command omits both flags.
- `ruff check src/` clean; full (non-integration) `pytest` green.

## Out of scope

- No auto-enabling heuristic (e.g. "turn on `--no-mmap` when experts spill to CPU"). The RAM-fit
  caveat makes auto-enable unsafe without a RAM probe; keep it a manual opt-in for now.
- Speculative-decoding / draft-model knob: explicitly **not** pursued — see the learnings entry
  for why it's net-negative on MoE/SSM architectures.

## References

- `docs/learnings/llamacpp-low-vram-moe-flags.md` — the flag ladder + measurements that motivate this.
- `CLAUDE.md` → llama.cpp provider quirks (`LAUNCH_KNOBS`, `jinja` as the boolean-flag precedent).
- Trello: card on board `69f86428` (LLMFacade) pointing back at this file.
