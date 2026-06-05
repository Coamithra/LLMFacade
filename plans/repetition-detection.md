# Plan: repetition-loop detection + auto-retry

_Spec'd 2026-06-05. Goal: let a caller enable detection of degenerate repetition loops
("repetition detection 3, 2 retries") so that when a model gets stuck emitting the same
line/paragraph over and over until it hits the length limit, llmfacade aborts the call, restarts it
up to N times, and finally raises a clear error instead of returning (or burning a full context of)
garbage._

## Motivation

Local models — Gemma especially, and low-bit quants most of all — routinely fall into a degenerate
state where they emit a single line, sentence, or paragraph repeatedly until `finish_reason: length`.
This is the same failure family we just saw with the truncated tool calls: a runaway that wastes the
entire token budget and returns useless output. `repeat_penalty` / the `dry` sampler help but do not
eliminate it.

We want an opt-in guard, configured ergonomically, e.g.:

```python
convo = model.new_conversation(repetition_detection=RepetitionGuard(retries=2))
# or shorthand:
convo.send(prompt, repetition_detection=3)   # enable with sensitivity 3, default retries
```

Semantics: if a repeating pattern is detected, **abort and restart the whole call**, up to `retries`
times; if the last retry still loops, raise `RepetitionLoopError`. Disabled by default (None).

## Steal from MTGAI (the user's own, proven implementation)

MTGAI already has a well-tuned detector and the surrounding retry machinery. Port it, with attribution.
Source files (`C:\Programming\MTGAI\backend\mtgai\pipeline\theme_extractor.py` unless noted):

- **`_detect_tandem_repeat(text)` / `_detect_repetition_loop(text)`** (~lines 1727-1767) — the core
  algorithm. Scans the last `_REPETITION_TAIL_CHARS` (4096) of the buffer for the **smallest period**
  `p` (1..`_REPETITION_MAX_PERIOD`=120) such that the suffix is ≥ `MIN_REPS[p]` consecutive copies of a
  `p`-char window totalling ≥ `MIN_TOTAL[p]` chars. Returns a human-readable hit description or `None`.
  Key design points worth preserving:
  - **Smallest-period canonical** (Fine & Wilf): iterate `p` upward, return on first hit, so
    `"thethethe"` is reported as period 3 `"the"`, not period 9 — otherwise the threshold check is wrong.
  - **Period-length-aware thresholds** (`_build_repetition_thresholds`, ~1706-1724): probability of a
    random tandem repeat falls geometrically with period length, so longer periods need fewer reps.
    The banded table (period → min_reps, min_total_chars):
    `1→(20,20)`, `2-4→(8,24)`, `5-10→(5,30)`, `11-25→(4,50)`, `26-60→(3,90)`, `61-120→(2,130)`.
  - **Alphanumeric guard**: the period window must contain ≥1 alphanumeric char, which suppresses
    false positives on ASCII-art separators, markdown rules (`"-"*N`), table borders (`"|---|"`),
    underscore fills, and whitespace runs. Real LLM loops always cycle through letters/digits.
- **Mid-stream wiring** (`_stream_single_call`, ~lines 1553-1614) — during streaming, every ≥64 chars
  of new text run the detector on the accumulated buffer; on a hit, `break`, **eagerly close the
  stream iterator** (`stream_iter.close()`) to release the HTTP connection, record the abort, and
  surface it. This is critical: without mid-stream detection a looping model fills the whole context
  before any post-hoc check runs. (See the comment at ~line 58.)
- **Retry + penalty escalation** — the subcall retry loop escalates `repeat_penalty` on each
  repetition retry; cf. `_repeat_penalty_for(attempt)` in `generation/skeleton_relabel.py` and the
  retry-aggregation loop (~line 2058). Worth porting the *idea*: on a repetition restart, nudge
  anti-repetition samplers (`repeat_penalty`, and/or enable `dry`) upward so the retry is less likely
  to loop the same way, rather than re-rolling the identical settings.

Constants to port: `_REPETITION_TAIL_CHARS = 4096`, `_REPETITION_MAX_PERIOD = 120`, check cadence 64.

## Design

### Config knob (`RepetitionGuard`)

A frozen dataclass (mirroring `DrySampler`), `from llmfacade import RepetitionGuard`:

```python
@dataclass(frozen=True, slots=True)
class RepetitionGuard:
    retries: int = 2                  # restarts before raising (the "2 retries")
    tail_chars: int = 4096            # MTGAI default
    max_period: int = 120
    check_every: int = 64             # mid-stream cadence (chars of new text)
    on_exhausted: str = "error"       # "error" -> raise; "return_last" -> return the last attempt
    escalate_repeat_penalty: float | None = 0.1  # added to repeat_penalty per retry; None = off
    escalate_dry: bool = False        # enable/strengthen `dry` on retries
```

Open question for the implementer: how the **bare-int shorthand** maps. The user phrased it as
"repetition detection 3 (2 retries)" — `3` reads as a sensitivity/strictness and `2` as retries.
Since the *real* sensitivity lives in MTGAI's period-aware bands (not a single number), the cleanest
mapping is probably: `repetition_detection=3` → `RepetitionGuard(retries=3)` is **wrong** (conflates
the two numbers); instead treat a bare int as the **min-reps floor** layered over the banded
thresholds (a global strictness dial: higher = needs more reps before firing = fewer false positives),
with `retries` defaulting to 2. Decide and document; default profile = MTGAI's bands unchanged.

### Where it lives

- **New module** `src/llmfacade/repetition.py` — port `detect_repetition_loop(text, *, tail_chars,
  max_period)` + the banded thresholds (attribution comment to MTGAI), plus `RepetitionGuard`.
- **New exception** `RepetitionLoopError(LLMError)` in `exceptions.py`, carrying the hit description,
  the attempt count, and the partial text of the final attempt.
- **Cascade resolution** — `repetition_detection` resolves like `cache_dir`/`cache_mode`
  (provider < model < convo < per_call), **not** as a wire `RUNTIME_KNOB` (it's a facade behavior, not
  a provider param). Add a resolver alongside `resolve_cache`. `repetition_detection=None` disables.

### `send` / `asend` (transparent restart — the primary surface)

Nothing is yielded to the caller until done, so retries are fully transparent. Implementation: run the
round-trip under the hood as a stream so mid-stream abort works even for `send` (or post-hoc detect on
the full text if streaming-internally is too invasive — mid-stream is strongly preferred to avoid
filling context). On a hit: discard the attempt, escalate samplers per `RepetitionGuard`, retry. After
`retries` exhausted: raise `RepetitionLoopError` (or return the last attempt if `on_exhausted ==
"return_last"`). A repetition-aborted attempt must **not** be written to the response cache.

### `stream` / `astream` (caller's stream — abort, with caveats)

Mid-stream detection runs every `check_every` chars (exactly MTGAI's pattern). On a hit the stream is
aborted and the iterator closed. Because deltas have *already been yielded* to the caller, transparent
retry is awkward — so:
- Default streaming behavior: **abort and raise `RepetitionLoopError`** (or yield a terminal error
  `StreamEvent`/sentinel — pick one and document) so the caller stops rendering garbage.
- Auto-retry on streaming is opt-in and means **restarting the stream from scratch**; document that the
  caller must reset any UI it rendered from the aborted attempt. (This composes naturally with the
  separate "incremental tool-arg streaming" card.)

### Interaction with tool calls

A repeating tool-argument stream is the same failure (see the just-merged `raw_arguments` work). The
detector should run on the accumulating tool-args buffer too, not just assistant text — so a looping
tool call is caught and retried like a looping prose answer.

## Tests

- **Detector unit tests** — port/recreate MTGAI's cases: exact-period loops at various period lengths
  hit at the banded thresholds; near-threshold negatives don't fire; alphanumeric guard suppresses
  `"-"*200`, `"|---|---|"`, underscore/whitespace fills; smallest-period canonicalization
  (`"thethethe"` → period 3).
- **Retry behavior** — fake provider that streams a looping body for the first K attempts then a clean
  body: assert it retries, succeeds, and the final `Response` is the clean one; assert
  `RepetitionLoopError` is raised after `retries` exhausted; assert aborted attempts aren't cached.
- **Cascade** — `repetition_detection` resolves provider < model < convo < per_call; `None` disables;
  per-call value overrides a convo default.
- **Backward-compat** — default (disabled) leaves `send`/`stream` behavior byte-for-byte unchanged.

## Touch list

- `src/llmfacade/repetition.py` — new: detector + thresholds + `RepetitionGuard` (ported from MTGAI).
- `src/llmfacade/exceptions.py` — `RepetitionLoopError`.
- `src/llmfacade/__init__.py` — export `RepetitionGuard`, `RepetitionLoopError`.
- `src/llmfacade/conversation.py` — wrap `send`/`asend` (transparent retry) and `stream`/`astream`
  (mid-stream abort + optional restart) with the guard; cascade resolution; cache-skip on abort.
- `src/llmfacade/settings.py` — register `repetition_detection` as a facade setting (cascade, not a
  wire knob), or wherever `cache_dir`/`cache_mode` are handled.
- `src/llmfacade/cache.py` — ensure aborted attempts are never stored.
- `CLAUDE.md` — document the guard, the cascade, the streaming caveat, and the MTGAI attribution.
- `tests/` — detector unit tests + retry-behavior + cascade tests.

## Done when

A caller can set `repetition_detection=RepetitionGuard(retries=2)` (or the int shorthand) and have a
Gemma-style repetition loop detected mid-stream, the call restarted up to the configured retries, and a
`RepetitionLoopError` raised when it still loops — with the default (disabled) path unchanged. Card
moved to Done and this file deleted per the repo convention.
