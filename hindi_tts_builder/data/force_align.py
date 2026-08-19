"""Stage 1.5b: force-align existing transcript text to get real word timings.

The counterpart to :mod:`~hindi_tts_builder.data.transcribe`. Use this when the
transcript **text** is good but its **timings** are not — which is the common
case for machine-regenerated SRTs.

Why this beats re-transcribing, measured on a 6-minute slice of this corpus:

| source of timed text            | sentence-terminated | speed        |
|---------------------------------|---------------------|--------------|
| original SRT cue boundaries      | 7.3%                | -            |
| Whisper large-v3 transcription   | 0.0%                | 2.4x realtime|
| forced alignment of the SRT text | **85.9%**           | **86.6x**    |

Whisper transcribing this Hindi narration emitted no sentence punctuation at
all, so splitting on terminators had nothing to split on. The SRTs already carry
15-18 terminators per 1000 characters. Keeping that text and recovering only the
timings is both far more accurate and ~36x faster, and it avoids baking ASR
errors into the training target permanently.

Output is the same ``aligned/<id>.words.json`` that ``transcribe`` writes, so the
``aligned_words`` segmentation mode consumes it unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.data.sentence_split import Word, split_words_into_sentences
from hindi_tts_builder.data.srt_health import analyze_timeline, close_gaps
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.project import ProjectPaths
from hindi_tts_builder.utils.srt import parse_srt, write_srt
from hindi_tts_builder.utils.text_compat import sanitize_for_training, untrainable_chars

TARGET_SR = 16000


def _load_audio_16k(path: Path, start: float, duration: float):
    """Read a window of audio as mono 16 kHz float32 (what the aligner expects)."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    sr = info.samplerate
    frames = int(duration * sr)
    x, _ = sf.read(str(path), start=int(start * sr), frames=frames, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != TARGET_SR:
        idx = (np.arange(int(len(x) * TARGET_SR / sr)) * sr / TARGET_SR).astype(int)
        x = x[np.clip(idx, 0, len(x) - 1)]
    return x


def _group_cues(cues, *, segment_seconds: float):
    """Batch consecutive cues into larger alignment segments.

    One segment per cue is WRONG when the cue timings are fabricated. WhisperX
    aligns with CTC *inside* each segment window, so a word whose true onset lies
    before ``segment["start"]`` cannot be given an earlier time — it is pinned at
    the boundary. Feeding per-cue windows off a whole-second grid produced 3.41%
    of words with exact-integer start times (~34x the chance rate) and left 31.7%
    of clips beginning ~0.8s after their text did.

    Concatenating cues into long segments removes those interior boundaries
    entirely: CTC then decides where every internal word starts, which is the
    whole point of forced alignment.
    """
    groups, cur = [], []
    for c in cues:
        if cur and (c.end_sec - cur[0].start_sec) > segment_seconds:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    return groups


def force_align_source(
    audio_path: Path,
    cues,
    *,
    align_model,
    metadata,
    device: str,
    audio_duration: float,
    chunk_minutes: float = 20.0,
    segment_seconds: float = 30.0,
    boundary_pad_sec: float = 1.5,
    log=None,
) -> list[Word]:
    """Align `cues`' text against the audio, returning globally-timed words.

    Processed in windows so a three-hour file does not have to sit in RAM at
    once. Cues are grouped into ``segment_seconds`` segments and each segment's
    window is widened by ``boundary_pad_sec`` on both sides, so no word is forced
    to start at a fabricated timestamp.
    """
    import whisperx

    words: list[Word] = []
    chunk = chunk_minutes * 60.0
    n_chunks = max(1, int(audio_duration / chunk) + (1 if audio_duration % chunk else 0))

    for k in range(n_chunks):
        c0 = k * chunk
        c1 = min(audio_duration, c0 + chunk)
        if c1 - c0 < 0.5:
            continue
        sel = [c for c in cues if c0 <= c.start_sec < c1]
        if not sel:
            continue
        # Extend the window to cover the last cue, so it is not cut off.
        w_end = min(audio_duration, max(c1, max(c.end_sec for c in sel) + 1.0))
        audio = _load_audio_16k(audio_path, c0, w_end - c0)

        groups = _group_cues(sel, segment_seconds=segment_seconds)
        segments = []
        for g in groups:
            # Pad outward so CTC may place words outside the nominal cue span.
            s = max(0.0, g[0].start_sec - boundary_pad_sec - c0)
            e = min(w_end - c0, g[-1].end_sec + boundary_pad_sec - c0)
            text = " ".join(" ".join(c.text.split()) for c in g).strip()
            if text and e > s:
                segments.append({"text": text, "start": s, "end": e})
        try:
            res = whisperx.align(segments, align_model, metadata, audio, device,
                                 return_char_alignments=False)
        except Exception as e:
            if log:
                log.warning(f"[align] chunk {k + 1}/{n_chunks} failed: {e}")
            continue

        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                words.append(Word(text=text, start=float(w["start"]) + c0, end=float(w["end"]) + c0))
        if log:
            log.info(f"[align] chunk {k + 1}/{n_chunks} ({c0 / 60:.0f}-{w_end / 60:.0f} min): "
                     f"{len(sel)} cues -> {len(words)} words so far")

    words.sort(key=lambda w: w.start)
    return words


def force_align_sources(
    paths: ProjectPaths,
    manifest: Manifest,
    *,
    language: str = "hi",
    min_seconds: float = 2.0,
    max_seconds: float = 15.0,
    max_gap_seconds: float = 0.6,
    chunk_minutes: float = 20.0,
    close_gaps_first: bool = True,
    skip_existing: bool = True,
    logger=None,
) -> dict:
    """Force-align every source that has transcript text and downloaded audio."""
    import soundfile as sf
    import torch
    import whisperx

    log = logger or get_logger("data.force_align", paths.logs / "force_align.log")
    summary = {
        "sources_aligned": 0, "sources_skipped": 0, "sources_failed": 0,
        "words_total": 0, "units_total": 0, "terminator_ratio": 0.0,
        "untrainable_chars": [], "reclaimed_min": 0.0,
    }

    todo = []
    for src in manifest.active():
        if not src.status.downloaded or not src.transcript_path:
            continue
        if skip_existing and (paths.aligned / f"{src.id}.words.json").exists():
            summary["sources_skipped"] += 1
            continue
        todo.append(src)

    if not todo:
        log.info("[align] nothing to force-align")
        return summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[align] loading alignment model for '{language}' on {device}")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)

    all_bad: set[str] = set()
    term_num = term_den = 0

    for src in todo:
        audio_path = paths.root / (src.audio_path or "")
        srt_path = paths.root / src.transcript_path
        if not audio_path.exists() or not srt_path.exists():
            log.warning(f"[align] {src.id}: missing audio or transcript")
            summary["sources_failed"] += 1
            continue
        try:
            cues = parse_srt(srt_path)
            health = analyze_timeline(cues)
            if close_gaps_first and (health.gaps_uniform or health.gap_fraction > 0.05):
                cues, reclaimed = close_gaps(cues)
                summary["reclaimed_min"] += reclaimed / 60.0
                log.info(f"[align] {src.id}: closed fabricated gaps, +{reclaimed / 60:.1f} min "
                         f"of coverage for the aligner")

            duration = float(sf.info(str(audio_path)).duration)
            log.info(f"[align] {src.id}: {len(cues)} cues over {duration / 60:.1f} min")
            words = force_align_source(
                audio_path, cues,
                align_model=align_model, metadata=metadata, device=device,
                audio_duration=duration, chunk_minutes=chunk_minutes, log=log,
            )
            if not words:
                raise RuntimeError("alignment produced no timed words")

            for w in words:
                w.text, bad = sanitize_for_training(w.text)
                all_bad |= bad
            words = [w for w in words if w.text]

            units, stats = split_words_into_sentences(
                words, min_seconds=min_seconds, max_seconds=max_seconds,
                max_gap_seconds=max_gap_seconds, audio_duration=duration,
            )

            (paths.aligned / f"{src.id}.words.json").write_text(
                json.dumps({
                    "source_id": src.id, "language": language, "model": "whisperx-forced-align",
                    "audio_duration": duration, "n_words": len(words),
                    "words": [asdict(w) for w in words], "unit_stats": stats,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            write_srt(paths.aligned / f"{src.id}.srt",
                      [u.to_srt_cue(i + 1) for i, u in enumerate(units)])

            term_num += sum(1 for u in units if u.terminated)
            term_den += len(units)

            src.status.aligned = True
            src.transcript_origin = "forced_align"
            manifest.save()

            summary["sources_aligned"] += 1
            summary["words_total"] += len(words)
            summary["units_total"] += len(units)
            ratio = sum(1 for u in units if u.terminated) / max(len(units), 1)
            log.info(f"[align] {src.id}: {len(words)} words -> {len(units)} units, "
                     f"sentence-terminated {100 * ratio:.1f}%")
        except Exception as e:
            log.error(f"[align fail] {src.id}: {type(e).__name__}: {e}")
            src.error = f"force_align: {e}"
            manifest.save()
            summary["sources_failed"] += 1

    summary["terminator_ratio"] = term_num / term_den if term_den else 0.0
    summary["untrainable_chars"] = sorted(all_bad)
    log.info(f"force-align complete: {summary['sources_aligned']} aligned, "
             f"{summary['sources_failed']} failed; overall sentence-terminated "
             f"{100 * summary['terminator_ratio']:.1f}%")
    return summary
