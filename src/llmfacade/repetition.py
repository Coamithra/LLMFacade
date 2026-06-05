"""Degenerate-repetition-loop detection plus the ``RepetitionGuard`` knob.

Local models — Gemma especially, low-bit quants most of all — routinely fall
into a degenerate state where they emit the same line / sentence / paragraph
over and over until they hit the length limit, wasting the whole token budget
on useless output. ``repeat_penalty`` and the ``dry`` sampler help but do not
eliminate it. This module detects that state so a caller can abort and restart
the call (see the retry machinery in ``conversation.py``).

The detector — the tandem-repeat scan, its period-length-aware confidence
thresholds, and the alphanumeric guard against ASCII-art false positives — is
ported from MTGAI's ``theme_extractor`` (``_detect_tandem_repeat`` /
``_build_repetition_thresholds``), the user's own proven implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tail size for the suffix-periodicity scan. Bounded so the detector cost stays
# trivial regardless of total stream length.
_DEFAULT_TAIL_CHARS = 4096

# Largest period we scan for. Beyond this the per-call cost grows and real LLM
# loops are vanishingly rare (degeneration cycles are short phrases).
_DEFAULT_MAX_PERIOD = 120

# Mid-stream check cadence: run the detector every this-many chars of new text.
_DEFAULT_CHECK_EVERY = 64

# Period-length-aware confidence bands: (period_lo, period_hi, min_reps,
# min_total_chars). A hit requires both at least ``min_reps`` consecutive copies
# of the period at the suffix and at least ``min_total_chars`` of repeated
# content. The probability of a random tandem repeat at the suffix falls
# geometrically with period length, so longer periods need fewer reps.
_BANDS: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 20, 20),
    (2, 4, 8, 24),
    (5, 10, 5, 30),
    (11, 25, 4, 50),
    (26, 60, 3, 90),
    (61, _DEFAULT_MAX_PERIOD, 2, 130),
)


def _build_thresholds(max_period: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reps = [0] * (max_period + 1)
    total = [0] * (max_period + 1)
    for lo, hi, r, t in _BANDS:
        for p in range(lo, min(hi, max_period) + 1):
            reps[p] = r
            total[p] = t
    # Any period above the last band's ceiling (when max_period > 120) reuses
    # the top band's thresholds.
    top_r, top_t = _BANDS[-1][2], _BANDS[-1][3]
    for p in range(1, max_period + 1):
        if reps[p] == 0:
            reps[p] = top_r
            total[p] = top_t
    return tuple(reps), tuple(total)


_THRESHOLD_CACHE: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}


def _thresholds_for(max_period: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cached = _THRESHOLD_CACHE.get(max_period)
    if cached is None:
        cached = _build_thresholds(max_period)
        _THRESHOLD_CACHE[max_period] = cached
    return cached


def detect_repetition_loop(
    text: str,
    *,
    tail_chars: int = _DEFAULT_TAIL_CHARS,
    max_period: int = _DEFAULT_MAX_PERIOD,
    min_reps_floor: int = 0,
) -> str | None:
    """Detect a degenerate tandem repeat at the suffix of ``text``.

    Scans the last ``tail_chars`` of ``text`` for the *smallest* period ``p``
    (1..``max_period``) such that the suffix consists of at least
    ``max(banded_min_reps[p], min_reps_floor)`` consecutive copies of a
    ``p``-char window, totalling at least the band's minimum characters.

    Iterating ``p`` upward and returning on the first hit yields the canonical
    smallest period (Fine & Wilf): reporting ``"thethethe"`` as period 9 rather
    than period 3 ``"the"`` would distort the threshold check.

    The period window must contain at least one alphanumeric character. This
    suppresses realistic non-loop patterns that look superficially periodic
    (ASCII-art separators, markdown rules ``"-"*N``, table borders
    ``"|---|---|"``, underscore fills, whitespace runs). Real LLM repetition
    loops always cycle through tokens with letters or digits.

    ``min_reps_floor`` is a global strictness dial layered over the banded
    thresholds (higher = needs more reps before firing = fewer false
    positives); ``0`` leaves the MTGAI bands unchanged.

    Returns a human-readable hit description, or ``None`` when no loop is found.
    """
    if not text:
        return None
    tail = text[-tail_chars:] if tail_chars > 0 else text
    n = len(tail)
    max_p = min(max_period, n // 2)
    if max_p < 1:
        return None
    min_reps, min_total = _thresholds_for(max_period)
    for p in range(1, max_p + 1):
        window = tail[n - p :]
        if not any(c.isalnum() for c in window):
            continue
        copies = 1
        while n - p * (copies + 1) >= 0 and tail[n - p * (copies + 1) : n - p * copies] == window:
            copies += 1
        need = max(min_reps[p], min_reps_floor)
        if copies >= need and p * copies >= min_total[p]:
            display = window if len(window) <= 40 else window[:37] + "..."
            return f"Period {display!r} (len={p}) repeated {copies}+ times at tail"
    return None


@dataclass(frozen=True, slots=True)
class RepetitionGuard:
    """Opt-in guard against degenerate repetition loops. The value of the
    facade-level ``repetition_detection`` setting (``from llmfacade import
    RepetitionGuard``).

    When set on a conversation (or any cascade scope), every ``send`` / ``asend``
    runs the model under the hood as a stream, runs ``detect_repetition_loop``
    on the accumulating text every ``check_every`` chars, and on a hit discards
    the attempt and restarts it — up to ``retries`` times — before raising
    ``RepetitionLoopError`` (or, with ``on_exhausted="return_last"``, returning
    the last attempt). ``stream`` / ``astream`` abort on the first hit and raise
    (no transparent retry, since deltas have already been yielded).

    Disabled by default (the setting resolves to ``None``). A bare int shorthand
    (e.g. ``repetition_detection=3``) maps to ``RepetitionGuard(min_reps_floor=3)``
    — it enables the guard with the MTGAI bands plus that strictness floor and
    the default ``retries``; ``False`` disables the guard at that scope.

    Fields:
        retries: restarts before giving up (the "2 retries"). ``send`` only.
        tail_chars: suffix window the detector scans.
        max_period: largest repeat period scanned.
        check_every: mid-stream cadence, in chars of new text, between detector
            runs.
        min_reps_floor: strictness floor layered over the banded thresholds.
        on_exhausted: ``"error"`` raises ``RepetitionLoopError``; ``"return_last"``
            returns the final (still-looping) attempt instead.
        escalate_repeat_penalty: added to ``repeat_penalty`` per retry so a
            restart is less likely to loop the same way (only when the model
            supports ``repeat_penalty``); ``None`` disables the escalation.
        escalate_dry: enable / strengthen the ``dry`` sampler on retries (only
            when the model supports ``dry``).
    """

    retries: int = 2
    tail_chars: int = _DEFAULT_TAIL_CHARS
    max_period: int = _DEFAULT_MAX_PERIOD
    check_every: int = _DEFAULT_CHECK_EVERY
    min_reps_floor: int = 0
    on_exhausted: str = "error"
    escalate_repeat_penalty: float | None = 0.1
    escalate_dry: bool = False

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError(f"RepetitionGuard.retries must be >= 0; got {self.retries!r}")
        if self.tail_chars < 1:
            raise ValueError(f"RepetitionGuard.tail_chars must be >= 1; got {self.tail_chars!r}")
        if self.max_period < 1:
            raise ValueError(f"RepetitionGuard.max_period must be >= 1; got {self.max_period!r}")
        if self.check_every < 1:
            raise ValueError(f"RepetitionGuard.check_every must be >= 1; got {self.check_every!r}")
        if self.min_reps_floor < 0:
            raise ValueError(
                f"RepetitionGuard.min_reps_floor must be >= 0; got {self.min_reps_floor!r}"
            )
        if self.on_exhausted not in {"error", "return_last"}:
            raise ValueError(
                f"RepetitionGuard.on_exhausted must be 'error' or 'return_last'; "
                f"got {self.on_exhausted!r}"
            )


def coerce_repetition_guard(value: object) -> RepetitionGuard | None:
    """Normalise a ``repetition_detection`` setting value to a guard or ``None``.

    ``None`` / ``False`` -> ``None`` (disabled). A ``RepetitionGuard`` passes
    through. A bare ``int`` -> ``RepetitionGuard(min_reps_floor=value)`` with
    default retries. ``True`` and other types raise ``TypeError``."""
    if value is None or value is False:
        return None
    if isinstance(value, RepetitionGuard):
        return value
    if isinstance(value, bool):
        raise TypeError(
            "repetition_detection=True is not valid; pass an int sensitivity "
            "(the min-reps floor) or a RepetitionGuard."
        )
    if isinstance(value, int):
        return RepetitionGuard(min_reps_floor=value)
    raise TypeError(
        "repetition_detection must be None, False, an int, or a RepetitionGuard; "
        f"got {type(value).__name__}."
    )


def resolve_repetition_guard(*, convo_repetition: object, model: object) -> RepetitionGuard | None:
    """Apply the provider < model < convo cascade and return the effective guard.

    Walks convo, then the model's override, then the provider's override; the
    first layer with an opinion wins (``None`` means "no opinion, defer to the
    next layer"; ``False`` is an opinion that disables)."""
    layers = (
        convo_repetition,
        getattr(model, "_repetition_override", None),
        getattr(getattr(model, "provider", None), "_repetition_override", None),
    )
    for v in layers:
        if v is not None:
            return coerce_repetition_guard(v)
    return None


__all__ = [
    "RepetitionGuard",
    "coerce_repetition_guard",
    "detect_repetition_loop",
    "resolve_repetition_guard",
]
