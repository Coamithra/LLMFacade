# Plumb DRY sampler through llmfacade for local loop escape

Trello card `6a219422` (https://trello.com/c/rCM97tL0) — follow-up from "Tame Gemma 4
local overthinking" (card `hCTe7FGv`).

## Context

llama.cpp's **DRY** ("Don't Repeat Yourself") sampler penalises *n-gram-level*
repetition — the right tool for paragraph/verbatim loops that token-level
`repeat_penalty` cannot break. The originating card's fix wanted the
truncation-retry path to escalate to anti-loop sampling (higher temp **+ DRY**)
rather than only bumping temperature. The temperature bump shipped; DRY is not
currently plumbable because `llamacpp`'s provider forwards only
`top_k` / `min_p` / `repeat_penalty` to llama-server via `extra_body`.

llama-server accepts the DRY family on its OpenAI-compat
`/v1/chat/completions` endpoint as top-level sampling params:

| wire param             | type        | llama.cpp default | meaning                          |
|------------------------|-------------|-------------------|----------------------------------|
| `dry_multiplier`       | float       | `0.0` (disabled)  | penalty strength; >0 enables DRY |
| `dry_base`             | float       | `1.75`            | exponential growth base          |
| `dry_allowed_length`   | int         | `2`               | max repeat length before penalty |
| `dry_penalty_last_n`   | int         | `-1`              | lookback (-1 = ctx, 0 = off)     |
| `dry_sequence_breakers`| list[str]   | `["\n",":","\"","*"]` | tokens that reset the n-gram  |

## Design decision (NEEDS USER SIGN-OFF)

The card's literal wording is "add `dry_*` to SUPPORTS". But every runtime knob in
this codebase is plumbed as an **explicit kwarg** at ~26 sites (base `Provider`
`__init__`/`new_model`, `Model` `__init__`/`new_conversation`, `Conversation`
`__init__` + all four `send`/`stream` signatures, and the `anthropic`/`llamacpp`
provider overrides — `openai` uses `**kwargs`, `google` uses the base). So:

- **Five flat knobs** (`dry_multiplier`, …) ≈ **130 mechanical edits** and adds 5
  dead kwargs to every non-llamacpp provider signature.
- **One structured `dry` knob** ≈ **26 edits** (same cost as adding one knob like
  `repeat_penalty`), and DRY params travel as a unit — which matches the use case
  (the consumer flips a *whole* DRY config on for one escalation retry).

**Recommendation: one structured `dry` knob**, value a frozen `DrySampler`
dataclass. Precedent for structured knob values already exists (`thinking` union,
`output_format` dict/enum, `user_metadata` dict, `beta_headers` list).

```python
@dataclass(frozen=True, slots=True)
class DrySampler:
    multiplier: float                       # required; the enabling param (>0)
    base: float = 1.75
    allowed_length: int = 2
    penalty_last_n: int = -1
    sequence_breakers: tuple[str, ...] | None = None   # None → omit, server default
```

Usage:
```python
convo.send("...", dry=DrySampler(multiplier=0.8))            # escalation retry
model.new_conversation(dry=DrySampler(multiplier=0.8, allowed_length=3))
```

## File-by-file changes (for the recommended single-knob design)

1. **`settings.py`**
   - Add `"dry"` to `RUNTIME_KNOBS`.
   - Define `DrySampler` frozen dataclass; export it; add to `__all__`.

2. **`provider.py`** — add `dry` kwarg to `Provider.__init__` (signature +
   `_validate_knobs` dict) and `Provider.new_model` (signature + defaults dict).

3. **`model.py`** — add `dry` to `Model.__init__` (signature + dict) and
   `Model.new_conversation` (signature + dict).

4. **`conversation.py`** — add `dry` to `Conversation.__init__`, `send`, `asend`,
   `stream`, `astream` (signature + forwarded dict at each).

5. **`providers/anthropic.py`** — add `dry` kwarg + `super().__init__` passthrough
   (mirrors `repeat_penalty`; it is *not* in Anthropic's `SUPPORTS`, so setting it
   raises `UnsupportedFeature`, consistent with the other samplers).

6. **`providers/llamacpp.py`**
   - Add `"dry"` to `SUPPORTS`.
   - Add `dry` kwarg to `__init__` (+ super), and to `new_model` (+ both the
     external-mode and managed-mode `Model(...)` constructions).
   - In `_build_kwargs`, after the `top_k`/`min_p`/`repeat_penalty` loop, read
     `req.settings.get("dry")` and merge its wire form into `extra` (omitting any
     `None` field so llama-server keeps its own default). A small
     `_dry_to_extra_body(value)` helper accepts a `DrySampler` (and, defensively, a
     plain mapping) and returns the `dry_*` dict.

7. **`cache.py`** `_normalize` + **`conversation.py`** `_logsafe` — teach both to
   unwrap a (non-type) dataclass via `dataclasses.asdict` so `DrySampler` renders
   as a clean nested dict in the fingerprint and the JSONL/HTML log (instead of a
   `repr()`/`str()` blob). One-line guard each; general win for any future
   dataclass-valued knob. (Without this it still *works* — repr/str are stable —
   just less readable.)

8. **`CLAUDE.md`** — document the `dry` knob + `DrySampler` under the llamacpp
   "Both modes" sampler note and the `settings.py` key-files entry; note it is
   llamacpp-only (not in other providers' `SUPPORTS`).

## Cascade / gating behaviour

- `dry` cascades `provider < model < convo < per_call` like any knob (verbatim
  value replacement — no per-field merge across scopes; a lower scope's
  `DrySampler` wholly replaces a higher one's).
- Only `llamacpp` declares `"dry"` in `SUPPORTS`. Setting `dry` on any other
  provider raises `UnsupportedFeature` at the layer it's set (`_validate_knobs`),
  or is silently dropped-with-warning if it cascades down from a higher scope onto
  a model that doesn't support it (`_filter_unsupported`) — identical to how
  `repeat_penalty` already behaves.
- Cache fingerprint: `dry` is in the merged effective `settings`, so it is hashed
  automatically by `fingerprint_request` → two different DRY configs produce
  different cache keys; flipping DRY off vs on does too.

## Tests (`tests/test_llamacpp.py` + `tests/test_settings_cascade.py` as fit)

- `DrySampler` → `extra_body` mapping: `multiplier` only (other fields omitted via
  server default); all fields set incl. `sequence_breakers`; `None` fields omitted.
- `dry` cascades and the per-call override wins.
- `dry` set on a non-llamacpp model raises `UnsupportedFeature`.
- `dry` set on llamacpp but cascaded onto a `capability_override` that drops it
  → dropped with one warning (mirror existing `_filter_unsupported` test).
- Cache fingerprint: two distinct `DrySampler` values hash differently; identical
  values hash the same; `dry=None` matches "no dry".
- (If touching `_normalize`/`_logsafe`) a `DrySampler` normalises to a sorted dict.

Find the nearest existing knob test (`repeat_penalty` / `min_p`) and mirror it.

## Out of scope (mtgai-side — separate repo, separate follow-up)

The card's scope items 2–3 (thread DRY through mtgai's `llm_client` convo_kwargs;
apply it on the gate truncation-retry in `gate_common.stream_flag_batch`) live in
the **mtgai** repo, not llmfacade. They consume this knob once it ships. Track as a
follow-up card against mtgai; not done here.

## Verification

- `ruff check src/` + `ruff format src/` clean.
- `python -c "import llmfacade"`.
- `pytest` (no integration).
- Manual (optional, needs a local GGUF + permission): register a managed model,
  `convo.send("repeat 'ha' forever", dry=DrySampler(multiplier=0.8))`, inspect the
  JSONL log shows `dry` in the settings header and the loop breaks.
