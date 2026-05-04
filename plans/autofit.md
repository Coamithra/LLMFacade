# Track A — server-side `--fit` + `fit_estimate` log metadata

## Context

The llamacpp provider's managed mode currently spawns `llama-server` with whatever launch knobs the user supplied. When the user doesn't constrain `context_size` / `n_gpu_layers` / `parallel`, llama-server can OOM on small GPUs. llama-server itself supports `--fit on` (plus `--fit-target` per-device free margin and `--fit-ctx` minimum ctx floor) which adjusts unset args to fit available VRAM at spawn time. We bake `--fit on` into the rendered `swap.yaml` CLI by default so newly registered models are OOM-safe out of the box.

For visibility, `provider.new_model()` (managed mode only) also runs `llama-fit-params --fit on --fit-print on` once per registered entry, parses the printed estimate, and stashes it on the provider keyed by `model_id`. A new optional `Provider.log_metadata(model_id=...)` hook lets the llamacpp provider surface that estimate as a top-level `fit_estimate` block in the JSONL settings header. The block is labelled an *estimate* (not wire-truth) because llama-server re-fits at spawn against current VRAM, which can drift between registration and first send.

User-confirmed defaults: `fit` defaults to `True`; estimation runs eagerly and synchronously in `new_model()`.

## Design

### 1. Launch knobs (`src/llmfacade/settings.py`)

Add `"fit"`, `"fit_target"`, `"fit_ctx"` to `LAUNCH_KNOBS`.

### 2. `_LaunchEntry` fields (`src/llmfacade/providers/_launch.py`)

Add to the dataclass:

```python
fit: bool = True
fit_target: tuple[int, ...] | None = None   # MiB free margin, per-device
fit_ctx: int | None = None                   # min ctx floor for --fit
```

`default_provider_launch_defaults(...)` returns `fit=True`, others `None`.

Adding `fit=True` to the baseline changes `canonical_launch_json(...)` output for every entry, so the `<gguf-stem>-<hash8>` derived `model_id` shifts on first run after the upgrade. Llamacpp managed mode is recently merged code, so this is acceptable. Note in the commit message.

Add a small helper in `_launch.py`:

```python
def parse_fit_print(stdout: str, stderr: str) -> dict[str, Any] | None:
    """Defensive parser for `llama-fit-params --fit-print on` output. Returns
    {context_size, n_gpu_layers, parallel, est_vram_mib} on success, None
    on any unexpected shape. Empirically tuned against captured real output."""
```

The exact regexes have to be tuned against captured real-binary output during implementation — capture stdout/stderr from a real run and snapshot-test against it. Defensive on every key (each independently optional).

### 3. CLI rendering (`src/llmfacade/providers/_swap_lifecycle.py`)

`_build_llama_server_cmd` appends, in order:

- `--fit on` if `entry.fit is True`, else `--fit off` if explicitly `False`.
- `--fit-target <comma-joined-ints>` if `entry.fit_target` is set.
- `--fit-ctx <N>` if `entry.fit_ctx` is set.

Pass through `_needs_quoting` for safety.

### 4. Provider constructor + `new_model` (`src/llmfacade/providers/llamacpp.py`)

Add `fit: bool | None = None`, `fit_target: list[int] | tuple[int, ...] | None = None`, `fit_ctx: int | None = None` kwargs to:

- `LlamaCppServerProvider.__init__` — threaded into `_launch_defaults` via the same merge pattern as the existing knobs.
- `new_model()` — explicit per-model overrides, cascaded into `merged`, then included on `_LaunchEntry`.
- External-mode rejection — extend the `offending` list to include the three new knobs.

Provider holds `self._fit_estimates: dict[str, dict[str, Any] | None]` (mapping `model_id -> estimate dict or None`). Initialised in `__init__`.

After `_LaunchEntry` is constructed and `self._supervisor.register(entry)` is called in managed-mode `new_model()`, if `entry.fit is True` and `shutil.which("llama-fit-params") is not None`:

- Build the equivalent CLI argv (same flags `_build_llama_server_cmd` would emit, minus `--port ${PORT}`, plus `--fit-print on`).
- Run via `subprocess.run(..., capture_output=True, timeout=60, check=False)`.
- On any exception or non-zero exit or unparseable output, store `None`.
- On success, store the parsed dict.

This is best-effort metadata: failures never propagate to the caller.

### 5. `Provider.log_metadata()` hook (`src/llmfacade/provider.py`)

Add base method:

```python
def log_metadata(self, *, model_id: str) -> dict[str, Any] | None:
    """Optional: provider-specific top-level extras for the JSONL settings
    header. Default returns None. Returned dict's keys are merged into the
    settings record as siblings of `settings`, `system_blocks`, etc."""
    return None
```

`LlamaCppServerProvider.log_metadata()`:

- External mode → always `None`.
- Managed mode → `{"fit_estimate": est}` if `self._fit_estimates.get(model_id)` is not `None`, else `None`.

### 6. Conversation header invocation (`src/llmfacade/conversation.py`)

In `_write_settings_header()` (file:line `conversation.py:848-891`), immediately before `self._append_log(record)`:

```python
extra = provider.log_metadata(model_id=self._model.model_id)
if extra:
    record.update(extra)
```

Pass `extra` through to `self._html_logger.write_header(...)` as a new optional kwarg.

### 7. HTML log (`src/llmfacade/html_logger.py`)

`write_header` accepts `extra: dict[str, Any] | None = None`. When `extra` contains `fit_estimate`, render a small section (matches the existing settings-table styling) with the four fields. Best-effort; missing keys render blank.

## Files to modify

| File | Change |
|---|---|
| `src/llmfacade/settings.py` | Add 3 keys to `LAUNCH_KNOBS` |
| `src/llmfacade/provider.py` | Add `log_metadata()` base method |
| `src/llmfacade/conversation.py` | Call hook in `_write_settings_header`; pass extras to HTML logger |
| `src/llmfacade/html_logger.py` | Render `fit_estimate` section |
| `src/llmfacade/providers/_launch.py` | Add 3 fields to `_LaunchEntry`; new `parse_fit_print` helper; baseline includes `fit=True` |
| `src/llmfacade/providers/_swap_lifecycle.py` | Render `--fit` / `--fit-target` / `--fit-ctx` in `_build_llama_server_cmd` |
| `src/llmfacade/providers/llamacpp.py` | Constructor + `new_model` kwargs, eager fit-params subprocess, `_fit_estimates` dict, `log_metadata` override |
| `CLAUDE.md` | Document the new knobs and the `fit_estimate` log block under Provider quirks |

## Tests

| File | New tests |
|---|---|
| `tests/test_llamacpp_swap_yaml.py` | `test_render_default_includes_fit_on`; `test_render_fit_off_explicit`; `test_render_fit_target_and_ctx` |
| `tests/test_llamacpp.py` | `test_log_metadata_returns_none_when_no_estimate`; `test_log_metadata_returns_fit_estimate_when_cached` (manually populate `_fit_estimates`); `test_new_model_silently_skips_when_fit_params_missing` (monkeypatch `shutil.which` → None); `test_new_model_runs_fit_params_and_stores_estimate` (monkeypatch `subprocess.run` to return captured-real output) |
| `tests/test_conversation.py` | `test_settings_header_includes_provider_log_metadata` (fake provider returning `{"foo": 1}`; verify it lands as a sibling of `settings`) |
| `tests/test_llamacpp_supervisor.py` | No changes (estimates live on provider, not supervisor) |

## Verification

1. `ruff check src/` and `ruff format src/` clean.
2. `pytest` (default — no integration markers) passes.
3. Manual smoke:
   - Construct a managed-mode `LlamaCppServerProvider` with no `base_url`.
   - `provider.new_model(gguf=<real GGUF>, name="test")` — verify call returns within ~10s.
   - Inspect `<llmfacade_dir>/swap.yaml` — `cmd:` line contains `--fit on`.
   - Construct a `Conversation` and inspect the first JSONL log line — has top-level `fit_estimate: {context_size, n_gpu_layers, parallel, est_vram_mib}`.
   - First `convo.send("hi")` — server starts, no OOM.
4. Do **not** run any test under `tests/integration/` without explicit user permission (per `CLAUDE.md`).

## Out of scope

- Wire-truth fit values (Option 2 from the discussion: `/props` + `/slots` post-load). Can be layered on later as a `fit_actual` follow-up record.
- Any change to external-mode behaviour. External mode rejects all launch knobs (including the new three) as it does today.
- Ollama-style auto-restart on OOM. `--fit on` covers the prevention path.
