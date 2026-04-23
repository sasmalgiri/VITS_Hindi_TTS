"""Stage 2: Refine SRT timestamps against audio using WhisperX forced alignment.

User provides SRT files with their own timestamps. Those timestamps are usually
close but not audio-precise (human translators adjust for reading pace, not
phoneme timing). WhisperX transcribes the audio and aligns word-level
timestamps, then we snap each user SRT cue to the nearest alignment boundary.

If WhisperX is not available, we fall back to using the user's SRT timestamps
as-is. Quality suffers slightly but the pipeline still produces a working
model.

Output: one refined SRT per source at `aligned/<source_id>.srt`.

Tuning for VRAM-constrained GPUs (12 GB or less sharing with a Windows
desktop): three env vars override the WhisperX defaults without touching
code. These are read at alignment time, not import time.

    HTTS_WHISPERX_MODEL    — whisper model size (default: medium)
                             options: tiny, base, small, medium, large-v2, large-v3
    HTTS_WHISPERX_BATCH    — transcription batch size (default: 4)
    HTTS_WHISPERX_COMPUTE  — compute dtype on CUDA (default: int8_float16)
                             options: float32, float16, int8_float16, int8

On OOM the loader automatically steps down to smaller models before
falling back to the SRT-as-is path.
"""
from __future__ import annotations
from pathlib import Path
import os

from hindi_tts_builder.data.manifest import Manifest, Source
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.project import ProjectPaths
from hindi_tts_builder.utils.srt import SrtCue, parse_srt, write_srt


def _try_load_whisperx():
    """Attempt to import whisperx. Returns None if unavailable."""
    try:
        import whisperx  # type: ignore
        import torch  # type: ignore
        return whisperx, torch
    except ImportError:
        return None


def _transcribe_and_align(
    audio_path: Path,
    language: str = "hi",
    device: str | None = None,
    logger=None,
):
    """Run WhisperX transcription + alignment. Returns a list of word dicts:
        [{"word": "...", "start": float, "end": float}, ...]
    or None if WhisperX is unavailable.

    Honors HTTS_WHISPERX_{MODEL,BATCH,COMPUTE} env vars; on CUDA OOM it
    steps down to a smaller model rather than crashing the pipeline.
    """
    bundle = _try_load_whisperx()
    if bundle is None:
        return None
    whisperx, torch = bundle
    log = logger

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    default_compute = "int8_float16" if device == "cuda" else "int8"
    compute_type = os.environ.get("HTTS_WHISPERX_COMPUTE", default_compute)
    batch_size = int(os.environ.get("HTTS_WHISPERX_BATCH", "4"))
    requested = os.environ.get("HTTS_WHISPERX_MODEL", "medium")

    # Fallback chain: if the requested model OOMs, try progressively smaller
    # ones before falling back to the SRT-as-is path.
    fallback_order = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    try:
        start = fallback_order.index(requested)
    except ValueError:
        start = fallback_order.index("medium")
    candidates = list(reversed(fallback_order[: start + 1]))

    audio = whisperx.load_audio(str(audio_path))

    model = None
    used = None
    last_err = None
    for name in candidates:
        try:
            if log:
                log.info(f"[whisperx] loading {name} (batch={batch_size} compute={compute_type})")
            model = whisperx.load_model(name, device=device, compute_type=compute_type, language=language)
            used = name
            break
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                if log:
                    log.warning(f"[whisperx] {name} OOM or CUDA error; trying smaller model")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
    if model is None:
        if log:
            log.error(f"[whisperx] every model in {candidates} failed to load: {last_err}")
        return None

    try:
        result = model.transcribe(audio, batch_size=batch_size)
    except Exception as e:
        if log:
            log.error(f"[whisperx] transcribe failed with {used}: {e}")
        return None
    finally:
        # Free transcription model before loading the alignment model — they
        # both sit in VRAM otherwise.
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Align
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({"word": w.get("word", "").strip(), "start": float(w["start"]), "end": float(w["end"])})
    return words


def _snap_cues_to_words(cues: list[SrtCue], words: list[dict]) -> list[SrtCue]:
    """Snap each cue's start/end to the nearest word boundary from WhisperX."""
    if not words:
        return cues

    word_starts = [w["start"] for w in words]
    word_ends = [w["end"] for w in words]

    def nearest(value: float, options: list[float]) -> float:
        return min(options, key=lambda v: abs(v - value))

    refined: list[SrtCue] = []
    for cue in cues:
        # Find words overlapping this cue's time range
        overlap_starts = [s for s in word_starts if cue.start_sec - 1.0 <= s <= cue.end_sec + 1.0]
        overlap_ends = [e for e in word_ends if cue.start_sec - 1.0 <= e <= cue.end_sec + 1.0]
        new_start = nearest(cue.start_sec, overlap_starts) if overlap_starts else cue.start_sec
        new_end = nearest(cue.end_sec, overlap_ends) if overlap_ends else cue.end_sec
        if new_end <= new_start:
            new_end = new_start + max(0.3, cue.duration)
        refined.append(SrtCue(cue.index, new_start, new_end, cue.text))
    return refined


def align_transcripts(
    paths: ProjectPaths,
    manifest: Manifest,
    *,
    language: str = "hi",
    skip_existing: bool = True,
    use_whisperx: bool = True,
    logger=None,
) -> dict:
    """Refine each source's SRT timestamps and write to `aligned/<id>.srt`.

    Returns summary {aligned, skipped, failed, fallback_used}.
    """
    log = logger or get_logger("data.align", paths.logs / "align.log")
    summary = {"aligned": 0, "skipped": 0, "failed": 0, "fallback_used": 0}

    whisperx_available = use_whisperx and (_try_load_whisperx() is not None)
    if use_whisperx and not whisperx_available:
        log.warning("WhisperX not available; falling back to user SRT timestamps as-is.")

    for src in manifest:
        if not src.status.downloaded or not src.audio_path:
            log.warning(f"[skip] {src.id} not downloaded; cannot align")
            continue

        out_path = paths.aligned / f"{src.id}.srt"
        if skip_existing and out_path.exists():
            src.status.aligned = True
            summary["skipped"] += 1
            log.info(f"[skip] {src.id} already aligned")
            continue

        transcript_path = paths.root / src.transcript_path
        if not transcript_path.exists():
            log.error(f"[fail] {src.id}: transcript not found at {transcript_path}")
            src.error = f"transcript missing at {transcript_path}"
            summary["failed"] += 1
            continue

        try:
            cues = parse_srt(transcript_path)
            if not cues:
                raise ValueError("transcript has no cues")

            if whisperx_available:
                audio_path = paths.root / src.audio_path
                log.info(f"[whisperx] {src.id} ({len(cues)} cues)")
                words = _transcribe_and_align(audio_path, language=language, logger=log)
                if words:
                    cues = _snap_cues_to_words(cues, words)
                else:
                    log.warning(f"[whisperx] {src.id}: fell back to SRT timestamps")
                    summary["fallback_used"] += 1
            else:
                summary["fallback_used"] += 1

            write_srt(out_path, cues)
            src.status.aligned = True
            src.error = None
            summary["aligned"] += 1
            log.info(f"[ok] {src.id} → {out_path.name}")
        except Exception as e:
            src.error = f"alignment failed: {e}"
            src.status.aligned = False
            summary["failed"] += 1
            log.error(f"[fail] {src.id}: {e}")
        finally:
            manifest.save()

    log.info(
        f"align complete: {summary['aligned']} aligned, "
        f"{summary['skipped']} skipped, {summary['failed']} failed, "
        f"{summary['fallback_used']} used SRT-only fallback"
    )
    return summary
