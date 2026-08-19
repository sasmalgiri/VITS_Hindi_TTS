"""Cut sentence-aligned units from word-level timings.

The counterpart to :mod:`~hindi_tts_builder.data.cue_merge`. Merging can only
cut where a cue already ends; when a cue *contains* several sentences, those
interior boundaries are unreachable — we have no timestamp for a danda sitting
in the middle of a 7-second cue.

Measured on the h_tts_1 corpus: sentences average ~3.0s while cues are 7.0s, and
each cue holds 2.09-2.29 sentence terminators. 12,202 sentence boundaries were
invisible to any cue-level operation. That is why clips ended mid-phrase, and no
amount of merging could have fixed it.

Word-level timings — from WhisperX forced alignment, or from faster-whisper with
``word_timestamps=True`` — make every boundary addressable. This module turns a
flat word stream into units that begin and end where sentences do.

    words = words_from_whisper_segments(segments)
    units, stats = split_words_into_sentences(words, max_seconds=15.0)
"""
from __future__ import annotations

from dataclasses import dataclass

from hindi_tts_builder.data.cue_merge import (
    MergedCue,
    ends_with_clause_break,
    ends_with_soft_terminator,
    ends_with_terminator,
    quotes_balanced,
)


@dataclass
class Word:
    """One timed token. ``start``/``end`` are seconds from the start of the audio."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def words_from_whisper_segments(segments) -> list[Word]:
    """Flatten whisper/WhisperX segments into a word stream.

    Accepts both dict-shaped segments (WhisperX) and attribute-shaped ones
    (faster-whisper). Words missing timings are skipped rather than guessed —
    a fabricated timestamp is what got this project into trouble in the first
    place.
    """
    out: list[Word] = []
    for seg in segments:
        words = seg.get("words") if isinstance(seg, dict) else getattr(seg, "words", None)
        if not words:
            continue
        for w in words:
            if isinstance(w, dict):
                text, start, end = w.get("word"), w.get("start"), w.get("end")
            else:
                text, start, end = getattr(w, "word", None), getattr(w, "start", None), getattr(w, "end", None)
            if text is None or start is None or end is None:
                continue
            text = text.strip()
            if not text:
                continue
            out.append(Word(text=text, start=float(start), end=float(end)))
    return out


def _build(words: list[Word], *, terminated: bool, pad_left: float, pad_right: float,
           limit_start: float, limit_end: float) -> MergedCue:
    start = max(limit_start, words[0].start - pad_left)
    end = min(limit_end, words[-1].end + pad_right)
    interior = 0.0
    max_interior = 0.0
    for a, b in zip(words, words[1:]):
        gap = max(0.0, b.start - a.end)
        interior += gap
        max_interior = max(max_interior, gap)
    return MergedCue(
        start_sec=start,
        end_sec=end,
        text=" ".join(w.text for w in words).strip(),
        member_indices=[],
        terminated=terminated,
        interior_gap_sec=interior,
        max_interior_gap_sec=max_interior,
    )


def _split_long(words: list[Word], *, max_seconds: float) -> list[list[Word]]:
    """Break a too-long run at the best interior point.

    Prefers, in order: a clause break, then the widest real pause. Unlike the
    cue-level splitter this can cut anywhere, because every word has a measured
    timestamp.
    """
    span = words[-1].end - words[0].start
    if span <= max_seconds or len(words) == 1:
        return [words]

    target = 0.8 * max_seconds
    best_k, best_score = None, float("-inf")
    for k in range(1, len(words)):
        left = words[k - 1].end - words[0].start
        if left > max_seconds:
            break
        score = 0.0
        if ends_with_clause_break(words[k - 1].text):
            score += 2.0
        score += min(max(0.0, words[k].start - words[k - 1].end), 1.0) * 2.0
        score -= abs(left - target) / max(target, 1e-6)
        if score > best_score:
            best_k, best_score = k, score

    if best_k is None:
        return [[words[0]]] + (_split_long(words[1:], max_seconds=max_seconds) if len(words) > 1 else [])
    return _split_long(words[:best_k], max_seconds=max_seconds) + _split_long(
        words[best_k:], max_seconds=max_seconds
    )


def split_words_into_sentences(
    words: list[Word],
    *,
    min_seconds: float = 2.0,
    max_seconds: float = 15.0,
    max_gap_seconds: float = 0.6,
    pad_left_ms: int = 50,
    pad_right_ms: int = 100,
    audio_duration: float | None = None,
    merge_short_forward: bool = True,
) -> tuple[list[MergedCue], dict]:
    """Cut a word stream into sentence-aligned units.

    A unit closes on a sentence terminator (tolerating trailing quotes), or on a
    pause wider than ``max_gap_seconds``. Padding mirrors the trust-SRT policy:
    a little before the first word to catch the consonant attack, a little after
    the last to catch the release — clamped so a unit never runs past the audio.
    """
    stats = {
        "input_words": len(words),
        "units": 0,
        "terminated": 0,
        "flushed_by_gap": 0,
        "split_overlong": 0,
        "dropped_short": 0,
        "attached_short": 0,
    }
    if not words:
        return [], stats

    pad_l = pad_left_ms / 1000.0
    pad_r = pad_right_ms / 1000.0
    lo = 0.0
    hi = audio_duration if audio_duration is not None else float("inf")

    runs: list[tuple[list[Word], bool]] = []
    buf: list[Word] = []
    for i, w in enumerate(words):
        buf.append(w)
        is_last = i + 1 >= len(words)
        gap_after = float("inf") if is_last else words[i + 1].start - w.end
        joined = " ".join(x.text for x in buf)
        hard = ends_with_terminator(w.text) and quotes_balanced(joined)
        soft = ends_with_soft_terminator(w.text) and gap_after > max_gap_seconds
        if hard or soft:
            runs.append((buf, True))
            stats["terminated"] += 1
            buf = []
        elif gap_after > max_gap_seconds:
            runs.append((buf, False))
            # End-of-stream is not a gap flush; only count real interior pauses.
            if not is_last:
                stats["flushed_by_gap"] += 1
            buf = []
    if buf:
        runs.append((buf, ends_with_terminator(buf[-1].text)))

    units: list[MergedCue] = []
    for run, terminated in runs:
        pieces = _split_long(run, max_seconds=max_seconds)
        if len(pieces) > 1:
            stats["split_overlong"] += len(pieces) - 1
        for n, piece in enumerate(pieces):
            u = _build(
                piece,
                terminated=terminated and n == len(pieces) - 1,
                pad_left=pad_l,
                pad_right=pad_r,
                limit_start=lo,
                limit_end=hi,
            )
            u.split_at_clause = len(pieces) > 1
            if u.duration > max_seconds:
                u.oversized = True
            units.append(u)

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
            if gap <= max_gap_seconds and (nxt.end_sec - u.start_sec) <= max_seconds:
                merged.append(
                    MergedCue(
                        start_sec=u.start_sec,
                        end_sec=nxt.end_sec,
                        text=f"{u.text} {nxt.text}".strip(),
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
