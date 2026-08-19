"""Stage 3: Segment raw audio into training clips.

Writes, per source:

    aligned/<source_id>/<clip_id>.wav
    aligned/<source_id>/<clip_id>.txt

The .txt holds raw Devanagari (frontend processing happens at dataset build).

Three segmentation modes, chosen by ``segmentation_mode``:

``cue`` (default)
    One clip per SRT cue. Byte-identical to the original behaviour, kept as the
    default so no existing project changes shape without being asked.

``sentence``
    Merge adjacent cues until a sentence terminator. Correct when sentences are
    *longer* than cues. Useless when they are shorter — see below.

``aligned_words``
    Cut from word-level timings (``aligned/<id>.words.json``, written by
    :mod:`~hindi_tts_builder.data.transcribe`). The only mode that can reach a
    sentence boundary sitting inside a cue.

Choosing between the last two is a measurement, not a preference. On the
h_tts_1 corpus each 7-second cue contained 2.09-2.29 sentence terminators, so
12,202 boundaries were unreachable at cue level and ``sentence`` mode moved
alignment only 7.4% -> 8.7%. Run ``scripts/srt_dry_merge.py`` first; if
terminators-per-cue is above ~1.2, cue-level merging cannot help and
``aligned_words`` is the only mode that will.

Clips outside the configured min/max duration are dropped with a logged reason.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess

from hindi_tts_builder.data.cue_merge import (
    MergedCue,
    merge_cues_to_sentences,
    segmentation_fingerprint,
    terminator_ratio,
    units_from_cues,
)
from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.data.srt_health import analyze_timeline, close_gaps
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.audio import trim_silence, write_wav, read_wav
from hindi_tts_builder.utils.project import ProjectPaths
from hindi_tts_builder.utils.srt import parse_srt

MODE_CUE = "cue"
MODE_SENTENCE = "sentence"
MODE_ALIGNED_WORDS = "aligned_words"
VALID_MODES = (MODE_CUE, MODE_SENTENCE, MODE_ALIGNED_WORDS)


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


def _units_for_source(
    paths: ProjectPaths,
    src,
    srt_path: Path,
    *,
    mode: str,
    min_seconds: float,
    max_seconds: float,
    max_gap_seconds: float,
    max_interior_gap_seconds: float,
    max_interior_silence_ratio: float,
    close_synthetic_gaps: bool,
    refuse_untrusted_timeline: bool,
    log,
) -> tuple[list[MergedCue], dict]:
    """Produce the cut list for one source. Raises on an untrusted timeline."""
    info: dict = {"mode": mode}

    if mode == MODE_ALIGNED_WORDS:
        from hindi_tts_builder.data.sentence_split import split_words_into_sentences
        from hindi_tts_builder.data.transcribe import load_words

        words = load_words(paths, src.id)
        if not words:
            raise RuntimeError(
                f"no word timings at aligned/{src.id}.words.json — run "
                f"`hindi-tts-builder transcribe` first, or pick another mode"
            )
        units, stats = split_words_into_sentences(
            words,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        info["stats"] = stats
        info["padded"] = True  # sentence_split already applied padding
        return units, info

    cues = parse_srt(srt_path)
    health = analyze_timeline(cues)
    info["timeline_verdict"] = health.verdict

    if not health.trustworthy:
        for line in health.describe().splitlines():
            log.warning(f"[timeline] {src.id}: {line}")
        if refuse_untrusted_timeline and mode != MODE_CUE:
            raise RuntimeError(
                f"timeline verdict '{health.verdict}' — refusing to cut. Either "
                f"transcribe with ASR (mode={MODE_ALIGNED_WORDS}) or set "
                f"segmentation.refuse_untrusted_timeline=false to override."
            )

    if mode == MODE_CUE:
        return units_from_cues(cues), info

    if close_synthetic_gaps and (health.gaps_uniform or health.gap_fraction > 0.05):
        cues, reclaimed = close_gaps(cues)
        info["reclaimed_sec"] = reclaimed
        log.info(f"[timeline] {src.id}: reclaimed {reclaimed / 60:.1f} min from inter-cue gaps")

    units, stats = merge_cues_to_sentences(
        cues,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        max_gap_seconds=max_gap_seconds,
        max_interior_gap_seconds=max_interior_gap_seconds,
        max_interior_silence_ratio=max_interior_silence_ratio,
    )
    info["stats"] = stats
    info["padded"] = False
    return units, info


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
    trust_srt: bool = False,
    trust_srt_pad_left_ms: int = 50,
    trust_srt_pad_right_ms: int = 100,
    segmentation_mode: str = MODE_CUE,
    sentence_min_seconds: float = 2.0,
    sentence_max_gap_seconds: float = 0.4,
    sentence_max_interior_gap_seconds: float = 0.6,
    sentence_max_interior_silence_ratio: float = 0.25,
    close_synthetic_gaps: bool = True,
    refuse_untrusted_timeline: bool = True,
    workers: int | None = None,
    logger=None,
) -> dict:
    """Segment every ready source into clips. Returns summary counts.

    ``trust_srt=True`` reads the *original* user SRT rather than the WhisperX
    output, so Stage 2 can be skipped, and pads each cut by
    ``trust_srt_pad_left_ms`` / ``trust_srt_pad_right_ms`` to catch the leading
    consonant attack and trailing release.

    ``segmentation_mode`` selects how cut boundaries are chosen — see the module
    docstring. Each source records the mode and a parameter fingerprint, so a
    later run with different settings is detected instead of quietly producing a
    corpus cut two different ways.
    """
    # ffmpeg + numpy both release the GIL, so threads scale nearly linearly here.
    workers = workers or max(1, min(12, (os.cpu_count() or 4)))

    if segmentation_mode not in VALID_MODES:
        raise ValueError(f"segmentation_mode must be one of {VALID_MODES}, got {segmentation_mode!r}")

    log = logger or get_logger("data.segment", paths.logs / "segment.log")
    summary = {
        "clips_created": 0, "clips_skipped_existing": 0, "clips_rejected": 0,
        "sources_processed": 0, "sources_failed": 0, "sources_conflicted": 0,
        "mode": segmentation_mode, "terminator_ratio": 0.0,
    }

    fingerprint = segmentation_fingerprint(
        mode=segmentation_mode,
        trust_srt=trust_srt,
        sample_rate=sample_rate,
        loudness_lufs=loudness_lufs,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        trim_silence_pad_ms=trim_silence_pad_ms,
        pad_left_ms=trust_srt_pad_left_ms,
        pad_right_ms=trust_srt_pad_right_ms,
        sentence_min_seconds=sentence_min_seconds,
        max_gap_seconds=sentence_max_gap_seconds,
        max_interior_gap_seconds=sentence_max_interior_gap_seconds,
        max_interior_silence_ratio=sentence_max_interior_silence_ratio,
        close_synthetic_gaps=close_synthetic_gaps,
    )

    term_num = term_den = 0

    for src in manifest.active():
        # aligned_words needs a transcript, not an alignment pass.
        if segmentation_mode == MODE_ALIGNED_WORDS:
            if not src.status.downloaded:
                continue
        elif trust_srt:
            if not src.status.downloaded:
                continue
        elif not src.status.aligned:
            continue

        # Refuse to mix two cut policies in one corpus. Skip the source and flag
        # it; never abort the run, or `corpus-ingest --watch` loops forever.
        if (
            src.segmentation_fingerprint
            and src.segmentation_fingerprint != fingerprint
            and any(paths.aligned.joinpath(src.id).glob("*.wav"))
        ):
            src.segmentation_state = "conflict"
            manifest.save()
            summary["sources_conflicted"] += 1
            log.error(
                f"[conflict] {src.id}: existing clips were cut under policy "
                f"{src.segmentation_policy}/{src.segmentation_fingerprint}, now asked for "
                f"{segmentation_mode}/{fingerprint}. Run `resegment --force` to recut, "
                f"or restore the old settings. Skipping."
            )
            continue

        if trust_srt or segmentation_mode == MODE_ALIGNED_WORDS:
            srt_path = paths.root / (src.transcript_path or "")
        else:
            srt_path = paths.aligned / f"{src.id}.srt"
        audio_path = paths.root / (src.audio_path or "")

        if not audio_path.exists():
            log.warning(f"[skip] {src.id}: missing audio")
            continue
        if segmentation_mode != MODE_ALIGNED_WORDS and not srt_path.exists():
            log.warning(f"[skip] {src.id}: missing SRT ({srt_path.name})")
            continue

        out_dir = paths.aligned / src.id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            units, info = _units_for_source(
                paths, src, srt_path,
                mode=segmentation_mode,
                min_seconds=sentence_min_seconds,
                max_seconds=max_seconds,
                max_gap_seconds=sentence_max_gap_seconds,
                max_interior_gap_seconds=sentence_max_interior_gap_seconds,
                max_interior_silence_ratio=sentence_max_interior_silence_ratio,
                close_synthetic_gaps=close_synthetic_gaps,
                refuse_untrusted_timeline=refuse_untrusted_timeline,
                log=log,
            )
        except Exception as e:
            log.error(f"[fail] {src.id}: {e}")
            src.error = f"segment: {e}"
            manifest.save()
            summary["sources_failed"] += 1
            continue

        already_padded = bool(info.get("padded"))
        n_created = n_rejected = n_skipped = 0
        total = len(units)
        progress_every = max(50, total // 20)
        log.info(
            f"[segment] {src.id}: starting, mode={segmentation_mode}, {total} units to cut"
            + (f", terminated={100 * terminator_ratio(units):.1f}%" if segmentation_mode != MODE_CUE else "")
            + f", workers={workers}"
        )

        # Build the work list first, then cut in parallel. Each clip is an
        # independent ffmpeg subprocess plus a numpy trim — both release the GIL,
        # so threads give near-linear speedup. Cutting 16k clips serially left 11
        # of 12 cores idle for over an hour.
        jobs: list[tuple[int, str, Path, Path, float, float, str]] = []
        for i, unit in enumerate(units, 1):
            # cue mode keeps the legacy clip-id numbering — the SRT's own cue
            # index, not a list position — so existing corpora keep their
            # filenames. Other modes number units sequentially.
            ordinal = i
            if segmentation_mode == MODE_CUE and unit.source_cue_index is not None:
                ordinal = unit.source_cue_index
            clip_id = f"{src.id}_c{ordinal:06d}"
            clip_wav = out_dir / f"{clip_id}.wav"
            clip_txt = out_dir / f"{clip_id}.txt"

            cut_start = unit.start_sec
            cut_dur = unit.duration
            # Legacy: padding applied only in trust_srt mode. sentence mode cuts
            # from the same SRT timings so it pads too; aligned_words already did.
            if not already_padded and (trust_srt or segmentation_mode == MODE_SENTENCE):
                cut_start = max(0.0, unit.start_sec - trust_srt_pad_left_ms / 1000.0)
                cut_dur = unit.duration + (unit.start_sec - cut_start) + trust_srt_pad_right_ms / 1000.0

            if cut_dur < min_seconds or cut_dur > max_seconds:
                n_rejected += 1
                continue

            if skip_existing and clip_wav.exists() and clip_txt.exists():
                n_skipped += 1
                continue

            jobs.append((i, clip_id, clip_wav, clip_txt, cut_start, cut_dur, unit.text))

        def _cut_one(job) -> str:
            i, clip_id, clip_wav, clip_txt, cut_start, cut_dur, text = job
            try:
                _extract_clip(
                    src_audio=audio_path,
                    dst_audio=clip_wav,
                    start=cut_start,
                    duration=cut_dur,
                    sample_rate=sample_rate,
                    loudness_lufs=loudness_lufs,
                )
                if trim_silence_pad_ms > 0:
                    audio, sr = read_wav(clip_wav)
                    trimmed = trim_silence(audio, sr, pad_ms=trim_silence_pad_ms)
                    if len(trimmed) >= int(min_seconds * sr):
                        write_wav(clip_wav, trimmed, sr)
                    else:
                        clip_wav.unlink(missing_ok=True)
                        return "rejected"
                clip_txt.write_text(text, encoding="utf-8")
                return "created"
            except Exception as e:
                log.warning(f"[clip fail] {clip_id}: {e}")
                clip_wav.unlink(missing_ok=True)  # never leave a half-written wav
                return "rejected"

        if jobs:
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for outcome in pool.map(_cut_one, jobs):
                    if outcome == "created":
                        n_created += 1
                    else:
                        n_rejected += 1
                    done += 1
                    if done % progress_every == 0 or done == len(jobs):
                        log.info(
                            f"[segment] {src.id}: {done}/{len(jobs)} cut "
                            f"(created={n_created} skipped={n_skipped} rejected={n_rejected})"
                        )

        term_num += sum(1 for u in units if u.terminated)
        term_den += len(units)

        src.status.segmented = True
        src.segmentation_policy = segmentation_mode
        src.segmentation_fingerprint = fingerprint
        src.segmentation_state = None
        if src.transcript_origin is None:
            src.transcript_origin = "user_srt" if trust_srt else "whisperx_aligned"
        summary["clips_created"] += n_created
        summary["clips_skipped_existing"] += n_skipped
        summary["clips_rejected"] += n_rejected
        summary["sources_processed"] += 1
        log.info(f"[ok] {src.id}: {n_created} new, {n_skipped} skipped, {n_rejected} rejected")
        manifest.save()

    summary["terminator_ratio"] = term_num / term_den if term_den else 0.0
    log.info(
        f"segment complete [mode={segmentation_mode}]: {summary['clips_created']} new clips, "
        f"{summary['clips_skipped_existing']} skipped, {summary['clips_rejected']} rejected, "
        f"{summary['sources_conflicted']} conflicted, "
        f"across {summary['sources_processed']} sources; "
        f"sentence-terminated {100 * summary['terminator_ratio']:.1f}%"
    )
    return summary
