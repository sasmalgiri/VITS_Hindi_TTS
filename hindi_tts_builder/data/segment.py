"""Stage 3: Segment raw audio into per-cue training clips.

Reads the aligned SRT for each source, cuts the audio at each cue's timestamps,
resamples to the project's target sample rate (default 24kHz), and writes:

    aligned/<source_id>/<clip_id>.wav
    aligned/<source_id>/<clip_id>.txt

The .txt file holds the raw Devanagari transcript (not yet frontend-processed
— that happens during dataset build).

Clips outside the configured min/max duration are dropped here with a logged
reason. They'll be gone from the manifest too.
"""
from __future__ import annotations
from pathlib import Path
import subprocess

from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.audio import trim_silence, write_wav, read_wav
from hindi_tts_builder.utils.project import ProjectPaths
from hindi_tts_builder.utils.srt import parse_srt


def _extract_clip(
    src_audio: Path,
    dst_audio: Path,
    start: float,
    duration: float,
    sample_rate: int,
    loudness_lufs: float,
) -> None:
    """Use ffmpeg to extract and resample a single clip."""
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(src_audio),
        "-af", f"loudnorm=I={loudness_lufs}:TP=-2:LRA=7,aresample={sample_rate}",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst_audio),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def segment_clips(
    paths: ProjectPaths,
    manifest: Manifest,
    *,
    sample_rate: int = 24000,
    loudness_lufs: float = -23.0,
    min_seconds: float = 1.5,
    max_seconds: float = 15.0,
    trim_silence_pad_ms: int = 50,
    skip_existing: bool = True,
    logger=None,
) -> dict:
    """Segment every aligned source into clips. Returns summary counts."""
    log = logger or get_logger("data.segment", paths.logs / "segment.log")
    summary = {"clips_created": 0, "clips_skipped_existing": 0, "clips_rejected": 0, "sources_processed": 0, "sources_failed": 0}

    for src in manifest:
        if not src.status.aligned:
            continue

        aligned_srt = paths.aligned / f"{src.id}.srt"
        audio_path = paths.root / (src.audio_path or "")
        if not aligned_srt.exists() or not audio_path.exists():
            log.warning(f"[skip] {src.id}: missing aligned SRT or audio")
            continue

        out_dir = paths.aligned / src.id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            cues = parse_srt(aligned_srt)
        except Exception as e:
            log.error(f"[fail] {src.id}: cannot parse aligned SRT: {e}")
            summary["sources_failed"] += 1
            continue

        n_created = 0
        n_rejected = 0
        n_skipped = 0
        total_cues = len(cues)
        progress_every = max(50, total_cues // 20)
        log.info(f"[segment] {src.id}: starting, {total_cues} cues to cut")

        for i, cue in enumerate(cues, 1):
            clip_id = f"{src.id}_c{cue.index:06d}"
            clip_wav = out_dir / f"{clip_id}.wav"
            clip_txt = out_dir / f"{clip_id}.txt"

            # Duration filter
            duration = cue.duration
            if duration < min_seconds or duration > max_seconds:
                n_rejected += 1
                continue

            if skip_existing and clip_wav.exists() and clip_txt.exists():
                n_skipped += 1
                continue

            try:
                _extract_clip(
                    src_audio=audio_path,
                    dst_audio=clip_wav,
                    start=cue.start_sec,
                    duration=duration,
                    sample_rate=sample_rate,
                    loudness_lufs=loudness_lufs,
                )
                # Optionally trim silence
                if trim_silence_pad_ms > 0:
                    audio, sr = read_wav(clip_wav)
                    trimmed = trim_silence(audio, sr, pad_ms=trim_silence_pad_ms)
                    if len(trimmed) >= int(min_seconds * sr):
                        write_wav(clip_wav, trimmed, sr)
                    else:
                        clip_wav.unlink(missing_ok=True)
                        n_rejected += 1
                        continue

                clip_txt.write_text(cue.text, encoding="utf-8")
                n_created += 1
            except Exception as e:
                log.warning(f"[clip fail] {clip_id}: {e}")
                n_rejected += 1

            if i % progress_every == 0 or i == total_cues:
                log.info(f"[segment] {src.id}: {i}/{total_cues} cues processed (created={n_created} skipped={n_skipped} rejected={n_rejected})")

        src.status.segmented = True
        summary["clips_created"] += n_created
        summary["clips_skipped_existing"] += n_skipped
        summary["clips_rejected"] += n_rejected
        summary["sources_processed"] += 1
        log.info(f"[ok] {src.id}: {n_created} new, {n_skipped} skipped, {n_rejected} rejected")
        manifest.save()

    log.info(
        f"segment complete: {summary['clips_created']} new clips, "
        f"{summary['clips_skipped_existing']} skipped, "
        f"{summary['clips_rejected']} rejected, "
        f"across {summary['sources_processed']} sources"
    )
    return summary
