"""Merge SRT display cues into sentence-aligned training units.

An SRT cue is a *display* boundary — how much text fits on screen for how long.
It has nothing to do with where a sentence ends. Cutting one clip per cue is how
the h_tts_1 corpus ended up with 92.7% of its clips starting and ending
mid-phrase, which teaches a model that utterances have no beginnings or ends.

This module merges adjacent cues until a sentence actually terminates, then cuts
there. It is deliberately **pure**: no I/O, no ffmpeg, no numpy. Everything here
is unit-testable with literal Devanagari strings, because a bug in this logic
silently corrupts every clip in the corpus.

Composition note: run :func:`~hindi_tts_builder.data.srt_health.close_gaps`
*before* merging when the timeline has fabricated inter-cue gaps. That collapses
the gaps to zero, after which the tight gap thresholds here behave correctly. A
large ``max_gap_seconds`` is not the right fix — it admits unlabelled audio into
the middle of a clip, and `trim_silence` only trims the leading and trailing
edges, so interior gaps survive into training by construction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from hindi_tts_builder.utils.srt import SrtCue

#: Characters that end a sentence outright.
HARD_TERMINATORS: tuple[str, ...] = ("।", "॥", "?", "!", ".")
#: Ends a sentence only when a real pause follows — otherwise it is hesitation.
SOFT_TERMINATORS: tuple[str, ...] = ("…",)
#: Trailing characters that may sit *after* a terminator. Measured on real Hindi
#: narration: 18.5% of sentences end on a closing quote, so checking the last
#: character alone misses nearly a fifth of true sentence ends.
CLOSERS: tuple[str, ...] = ('"', "'", "”", "’", "»", ")", "]", "}", "॰")
#: Mid-sentence breathing points, used to choose where to split an over-long run.
CLAUSE_BREAKS: tuple[str, ...] = (",", ";", ":", "—", "–", "॥")

_OPEN_QUOTES = ('"', "“", "‘", "«")
_CLOSE_QUOTES = ('"', "”", "’", "»")


def strip_closers(text: str) -> str:
    """Remove trailing whitespace and closing punctuation, keeping terminators."""
    s = text.rstrip()
    while s and s[-1] in CLOSERS:
        s = s[:-1].rstrip()
    return s


def ends_with_terminator(text: str) -> bool:
    """True if ``text`` ends a sentence, tolerating trailing quotes/brackets."""
    s = strip_closers(text)
    return bool(s) and s[-1] in HARD_TERMINATORS


def ends_with_soft_terminator(text: str) -> bool:
    s = strip_closers(text)
    return bool(s) and s[-1] in SOFT_TERMINATORS


def ends_with_clause_break(text: str) -> bool:
    s = strip_closers(text)
    return bool(s) and s[-1] in CLAUSE_BREAKS


def quotes_balanced(text: str) -> bool:
    """Cheap check that we are not cutting inside a quotation.

    Straight quotes are counted for parity; curly quotes are matched as pairs.
    A sentence terminator inside an open quote is usually dialogue continuing
    across cues, and splitting there strands the closing quote on the next clip.
    """
    if text.count('"') % 2:
        return False
    opens = sum(text.count(q) for q in ("“", "‘", "«"))
    closes = sum(text.count(q) for q in ("”", "’", "»"))
    return opens <= closes


@dataclass
class MergedCue:
    """One training unit: a span of audio whose text is a complete thought."""

    start_sec: float
    end_sec: float
    text: str
    member_indices: list[int] = field(default_factory=list)
    #: Ended on a real sentence terminator (vs flushed by a gap or a length cap).
    terminated: bool = False
    #: Produced by the over-long splitter rather than a natural boundary.
    split_at_clause: bool = False
    #: A single cue that alone exceeded max_seconds — emitted whole, never cut.
    oversized: bool = False
    interior_gap_sec: float = 0.0
    max_interior_gap_sec: float = 0.0
    #: The originating SRT cue's own index number. Set only in 1:1 (cue) mode,
    #: where clip filenames must keep using it rather than a list position —
    #: an SRT whose indices are not 1..N would otherwise rename every clip.
    source_cue_index: int | None = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def first_index(self) -> int:
        return self.member_indices[0] if self.member_indices else -1

    @property
    def interior_silence_ratio(self) -> float:
        return self.interior_gap_sec / self.duration if self.duration > 0 else 0.0

    def to_srt_cue(self, index: int) -> SrtCue:
        return SrtCue(index=index, start_sec=self.start_sec, end_sec=self.end_sec, text=self.text)


def _join(texts: list[str]) -> str:
    """Join cue texts the way parse_srt joins wrapped lines: single spaces."""
    return " ".join(" ".join(t.split()) for t in texts if t.strip()).strip()


def _build(cues: list[SrtCue], members: list[int], *, terminated: bool) -> MergedCue:
    first, last = cues[members[0]], cues[members[-1]]
    interior = 0.0
    max_interior = 0.0
    for a, b in zip(members, members[1:]):
        gap = max(0.0, cues[b].start_sec - cues[a].end_sec)
        interior += gap
        max_interior = max(max_interior, gap)
    return MergedCue(
        start_sec=first.start_sec,
        end_sec=last.end_sec,
        text=_join([cues[i].text for i in members]),
        member_indices=list(members),
        terminated=terminated,
        interior_gap_sec=interior,
        max_interior_gap_sec=max_interior,
    )


def _split_overlong(
    cues: list[SrtCue],
    members: list[int],
    *,
    max_seconds: float,
) -> list[list[int]]:
    """Split a too-long run, choosing boundaries only where cue edges already are.

    Never interpolates a timestamp inside a cue — cue edges are the only instants
    whose timing we have any evidence for.
    """
    span = cues[members[-1]].end_sec - cues[members[0]].start_sec
    if span <= max_seconds or len(members) == 1:
        return [members]

    target = 0.8 * max_seconds
    best_k, best_score = None, float("-inf")
    for k in range(1, len(members)):
        left_span = cues[members[k - 1]].end_sec - cues[members[0]].start_sec
        if left_span > max_seconds:
            break  # every later k is longer still
        score = 0.0
        if ends_with_clause_break(cues[members[k - 1]].text):
            score += 2.0
        gap = max(0.0, cues[members[k]].start_sec - cues[members[k - 1]].end_sec)
        score += min(gap, 1.0)
        score -= abs(left_span - target) / max(target, 1e-6)
        if score > best_score:
            best_k, best_score = k, score

    if best_k is None:
        # Even the first cue alone busts the cap. Emit it whole and continue.
        return [[members[0]]] + (
            _split_overlong(cues, members[1:], max_seconds=max_seconds) if len(members) > 1 else []
        )

    left = members[:best_k]
    right = members[best_k:]
    return _split_overlong(cues, left, max_seconds=max_seconds) + _split_overlong(
        cues, right, max_seconds=max_seconds
    )


def merge_cues_to_sentences(
    cues: list[SrtCue],
    *,
    min_seconds: float = 2.0,
    max_seconds: float = 15.0,
    max_gap_seconds: float = 0.4,
    max_interior_gap_seconds: float = 0.6,
    max_interior_silence_ratio: float = 0.25,
    merge_short_forward: bool = True,
) -> tuple[list[MergedCue], dict]:
    """Merge cues into sentence-aligned units.

    ``max_gap_seconds`` flushes the buffer even mid-sentence when the pause is
    long enough to mean the speaker stopped. ``max_interior_*`` refuse a merge
    that would bury unlabelled audio inside a clip.

    Returns the units and a stats dict for reporting.
    """
    stats = {
        "input_cues": len(cues),
        "units": 0,
        "terminated": 0,
        "flushed_by_gap": 0,
        "flushed_by_length": 0,
        "split_overlong": 0,
        "oversized_single": 0,
        "dropped_short": 0,
        "attached_short": 0,
    }
    if not cues:
        return [], stats

    runs: list[tuple[list[int], bool]] = []
    buf: list[int] = []

    for i, cue in enumerate(cues):
        buf.append(i)
        is_last = i + 1 >= len(cues)
        gap_after = float("inf") if is_last else cues[i + 1].start_sec - cue.end_sec

        hard = ends_with_terminator(cue.text) and quotes_balanced(_join([cues[j].text for j in buf]))
        soft = ends_with_soft_terminator(cue.text) and gap_after > max_gap_seconds
        big_gap = gap_after > max_gap_seconds

        span = cue.end_sec - cues[buf[0]].start_sec
        would_be_long = span >= max_seconds

        if hard or soft:
            runs.append((buf, True))
            stats["terminated"] += 1
            buf = []
        elif big_gap:
            runs.append((buf, False))
            # End-of-stream is not a gap flush; only count real interior pauses.
            if not is_last:
                stats["flushed_by_gap"] += 1
            buf = []
        elif would_be_long:
            runs.append((buf, False))
            stats["flushed_by_length"] += 1
            buf = []
        else:
            # Refuse a merge that would bury too much unlabelled audio.
            nxt_gap = gap_after
            if nxt_gap > max_interior_gap_seconds:
                runs.append((buf, False))
                stats["flushed_by_gap"] += 1
                buf = []

    if buf:
        runs.append((buf, ends_with_terminator(cues[buf[-1]].text)))

    # --- length discipline ------------------------------------------------
    units: list[MergedCue] = []
    for members, terminated in runs:
        pieces = _split_overlong(cues, members, max_seconds=max_seconds)
        if len(pieces) > 1:
            stats["split_overlong"] += len(pieces) - 1
        for n, piece in enumerate(pieces):
            u = _build(cues, piece, terminated=terminated and n == len(pieces) - 1)
            u.split_at_clause = len(pieces) > 1
            if u.duration > max_seconds:
                u.oversized = True
                stats["oversized_single"] += 1
            if u.interior_silence_ratio > max_interior_silence_ratio:
                # Too much dead air inside — fall back to unmerged members.
                for m in piece:
                    units.append(_build(cues, [m], terminated=ends_with_terminator(cues[m].text)))
                continue
            units.append(u)

    # --- short fragments --------------------------------------------------
    if merge_short_forward:
        merged: list[MergedCue] = []
        i = 0
        while i < len(units):
            u = units[i]
            if u.duration >= min_seconds or i + 1 >= len(units):
                merged.append(u)
                i += 1
                continue
            nxt = units[i + 1]
            gap = nxt.start_sec - u.end_sec
            combined = nxt.end_sec - u.start_sec
            if gap <= max_gap_seconds and combined <= max_seconds:
                merged.append(
                    MergedCue(
                        start_sec=u.start_sec,
                        end_sec=nxt.end_sec,
                        text=_join([u.text, nxt.text]),
                        member_indices=u.member_indices + nxt.member_indices,
                        terminated=nxt.terminated,
                        split_at_clause=u.split_at_clause or nxt.split_at_clause,
                        interior_gap_sec=u.interior_gap_sec + nxt.interior_gap_sec + max(0.0, gap),
                        max_interior_gap_sec=max(u.max_interior_gap_sec, nxt.max_interior_gap_sec, gap),
                    )
                )
                stats["attached_short"] += 1
                i += 2
            else:
                stats["dropped_short"] += 1
                i += 1
        units = merged

    stats["units"] = len(units)
    return units, stats


def units_from_cues(cues: list[SrtCue]) -> list[MergedCue]:
    """1:1 adapter reproducing today's behaviour exactly — one unit per cue.

    Kept so ``segmentation_mode="cue"`` runs through the same code path as
    sentence mode and stays byte-identical to the legacy segmenter.
    """
    return [
        MergedCue(
            start_sec=c.start_sec,
            end_sec=c.end_sec,
            text=c.text,
            member_indices=[i],
            terminated=ends_with_terminator(c.text),
            source_cue_index=c.index,
        )
        for i, c in enumerate(cues)
    ]


def terminator_ratio(units: list[MergedCue]) -> float:
    """Share of units ending on a real sentence terminator. The headline metric."""
    if not units:
        return 0.0
    return sum(1 for u in units if u.terminated) / len(units)


def segmentation_fingerprint(**params) -> str:
    """Stable short hash over every parameter that affects where audio is cut.

    Recorded per source so a later run can tell whether existing clips were cut
    under the same policy, instead of silently mixing two policies in one corpus.
    """
    payload = "|".join(f"{k}={params[k]!r}" for k in sorted(params))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
