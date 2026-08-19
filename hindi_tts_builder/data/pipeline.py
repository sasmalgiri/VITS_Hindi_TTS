"""Top-level data pipeline orchestrator.

`run_pipeline(project_name)` runs all 5 stages in order, stopping on the
first failure. Each stage is idempotent so re-running recovers cleanly.

Usage from Python:

    from hindi_tts_builder.data.pipeline import run_pipeline
    run_pipeline("my_voice")

Or from CLI:

    hindi-tts-builder prepare my_voice
"""
from __future__ import annotations
from pathlib import Path

from hindi_tts_builder.data.align import align_transcripts
from hindi_tts_builder.data.dataset import build_training_set
from hindi_tts_builder.data.download import download_audio
from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.data.qc import quality_filter
from hindi_tts_builder.data.segment import segment_clips
from hindi_tts_builder.frontend.pipeline import HindiFrontend
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.lockfile import ProjectLock
from hindi_tts_builder.utils.project import ProjectPaths, load_config


def _manifest_path(paths: ProjectPaths) -> Path:
    return paths.sources / "manifest.json"


def add_sources_from_files(
    paths: ProjectPaths,
    urls_file: Path,
    transcripts_dir: Path,
) -> int:
    """Populate (or append to) the sources manifest from a URLs file and a
    transcripts directory. Transcripts are matched to URLs by line order:
    the Nth URL gets the Nth .srt from the sorted transcripts dir.

    Returns count of new sources added.
    """
    manifest = Manifest(_manifest_path(paths))
    existing_urls = {s.url for s in manifest.sources}

    urls = [u.strip() for u in urls_file.read_text(encoding="utf-8").splitlines() if u.strip() and not u.strip().startswith("#")]
    srts = sorted(transcripts_dir.glob("*.srt"))

    if len(urls) != len(srts):
        raise ValueError(
            f"Count mismatch: {len(urls)} URLs vs {len(srts)} SRT files. "
            "Line N in urls.txt must correspond to the Nth SRT when sorted by name."
        )

    # Copy transcripts into project's sources/transcripts dir so the project
    # is self-contained
    added = 0
    for i, (url, srt) in enumerate(zip(urls, srts)):
        if url in existing_urls:
            continue
        dst_srt = paths.transcripts / srt.name
        if not dst_srt.exists():
            dst_srt.write_bytes(srt.read_bytes())
        rel_srt = str(dst_srt.relative_to(paths.root))
        manifest.add(url=url, transcript_path=rel_srt, index=i)
        added += 1

    manifest.save()
    return added


def write_qc_meta(paths: ProjectPaths, qc_mode: str, thresholds: dict | None = None) -> None:
    """Record how QC ran, next to the report it describes.

    Without this a passthrough report is indistinguishable from a real one: the
    h_tts_1 corpus carried 10,280 rows of `passed=1` that had never been scored,
    and nothing downstream could tell. The pre-train gate reads this file.
    """
    import json
    from datetime import datetime, timezone

    paths.training_set.mkdir(parents=True, exist_ok=True)
    (paths.training_set / "qc_report_meta.json").write_text(
        json.dumps(
            {
                "qc_mode": qc_mode,
                "thresholds": thresholds or {},
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _passthrough_qc(paths: ProjectPaths, manifest: Manifest, log) -> dict:
    """Mark every segmented clip as passed *without scoring it*.

    Metric columns are left EMPTY rather than filled with zeros. Writing "0.00"
    for SNR made an unscored corpus look measured — the report claimed 0 dB SNR
    on 10,280 clips and every one still passed. Empty means "not measured", and
    `reason=qc_skipped` plus the meta sidecar make that machine-detectable.

    Duration is still read from each wav header so build_training_set, which
    parses it as a float, keeps working.
    """
    import csv
    import soundfile as sf  # type: ignore
    paths.training_set.mkdir(parents=True, exist_ok=True)
    report = paths.training_set / "qc_report.csv"
    summary = {"total": 0, "passed": 0, "failed_snr": 0, "failed_silence": 0,
               "failed_cer": 0, "failed_duration": 0, "qc_mode": "skipped"}
    with report.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "source_id", "duration", "snr_db", "silence_ratio",
                    "whisper_cer", "passed", "reason", "qc_mode"])
        for src in manifest.active():
            if not src.status.segmented:
                continue
            clip_dir = paths.aligned / src.id
            if not clip_dir.exists():
                continue
            for clip_wav in sorted(clip_dir.glob(f"{src.id}_c*.wav")):
                clip_id = clip_wav.stem
                # soundfile.info reads only the WAV header — no audio decode.
                try:
                    info = sf.info(str(clip_wav))
                    duration = info.frames / float(info.samplerate)
                except Exception as e:
                    log.warning(f"[qc-skip] cannot stat {clip_id}: {e}; using 0.0")
                    duration = 0.0
                summary["total"] += 1
                summary["passed"] += 1
                w.writerow([clip_id, src.id, f"{duration:.3f}", "", "", "", 1, "qc_skipped", "skipped"])
            src.status.qc_passed = True
            src.qc_mode = "skipped"
            manifest.save()
    write_qc_meta(paths, "skipped")
    log.warning(
        f"QC SKIPPED: {summary['passed']}/{summary['total']} clips marked passed without "
        f"being scored. This corpus will be blocked by the pre-train gate — see "
        f"`hindi-tts-builder gate`."
    )
    return summary


def run_pipeline(
    project_root: Path,
    *,
    use_whisperx: bool = True,
    use_whisper_qc: bool = True,
    skip_qc: bool = False,
    trust_srt: bool = False,
    segmentation_mode: str | None = None,
    transcribe: bool | None = None,
    logger=None,
) -> dict:
    """Run the full data pipeline end-to-end. Returns a combined summary.

    skip_qc=True bypasses Stage 4 entirely - every segmented clip is taken
    as-is. Use when your SRTs are known good and the default thresholds
    (calibrated for clean studio audio) reject too much narration content.

    trust_srt=True activates trusted_srt_cut mode: skips Stage 2 (WhisperX
    alignment) entirely and cuts directly from the original SRT timestamps
    with 50ms left + 100ms right padding to capture consonant attacks and
    trailing breath. Use when SRT timing is verified word-to-word and you
    don't want WhisperX to introduce drift.

    segmentation_mode overrides config's segmentation.mode ("cue" | "sentence" |
    "aligned_words"). "aligned_words" implies Stage 1.5 (ASR transcription),
    which is also what runs when a source has no SRT at all.

    transcribe=None auto-enables Stage 1.5 when the mode needs word timings.
    """
    paths = ProjectPaths(project_root)
    cfg = load_config(project_root)
    log = logger or get_logger("data.pipeline", paths.logs / "pipeline.log")
    paths.ensure_all()

    # Single-writer lock. Two concurrent pipelines interleave ffmpeg writes on the
    # same clip paths and produce corrupt WAVs that `skip_existing` then preserves
    # forever. This has happened; see utils/lockfile.py.
    with ProjectLock(project_root, "prepare", logger=log):
        return _run_pipeline_locked(
            project_root, paths, cfg, log,
            use_whisperx=use_whisperx, use_whisper_qc=use_whisper_qc,
            skip_qc=skip_qc, trust_srt=trust_srt,
            segmentation_mode=segmentation_mode, transcribe=transcribe,
        )


def _run_pipeline_locked(
    project_root: Path,
    paths: ProjectPaths,
    cfg: dict,
    log,
    *,
    use_whisperx: bool,
    use_whisper_qc: bool,
    skip_qc: bool,
    trust_srt: bool,
    segmentation_mode: str | None,
    transcribe: bool | None,
) -> dict:
    """Pipeline body. Runs only while the project lock is held."""
    manifest = Manifest(_manifest_path(paths))
    if len(manifest) == 0:
        raise RuntimeError(
            "No sources in manifest. Call `add_sources_from_files` first "
            "(or use `hindi-tts-builder add-sources`)."
        )

    log.info(f"=== Pipeline starting for '{cfg['name']}' ({len(manifest)} sources) ===")

    seg_cfg = cfg["segmentation"]
    mode = segmentation_mode or seg_cfg["mode"]
    asr_cfg = cfg["asr"]

    log.info("--- Stage 1: download ---")
    s1 = download_audio(paths, manifest, logger=log)

    # Word timings can come from two places, and picking the right one matters a
    # great deal. Measured on this corpus: force-aligning the EXISTING punctuated
    # transcript gave 85.9% sentence-terminated at 86.6x realtime, while
    # re-transcribing the same audio with Whisper gave 0.0% at 2.4x — Whisper
    # emitted no sentence punctuation at all, leaving nothing to split on.
    # So: force-align wherever a transcript exists; fall back to ASR only where
    # there is genuinely no text.
    needs_words = mode == "aligned_words"
    with_text = [s for s in manifest if s.transcript_path]
    without_text = [s for s in manifest if not s.transcript_path]
    do_words = needs_words or bool(without_text) if transcribe is None else transcribe

    s1a = {"skipped": True, "reason": "no forced alignment needed"}
    s1b = {"skipped": True, "reason": "no ASR needed"}

    if do_words:
        if with_text:
            log.info(f"--- Stage 1.5a: forced alignment ({len(with_text)} source(s) with text) ---")
            from hindi_tts_builder.data.force_align import force_align_sources

            s1a = force_align_sources(
                paths, manifest,
                language=cfg.get("language", "hi"),
                min_seconds=seg_cfg["min_seconds"],
                max_seconds=seg_cfg["max_seconds"],
                max_gap_seconds=seg_cfg["max_gap_seconds"],
                logger=log,
            )
        if without_text:
            log.info(f"--- Stage 1.5b: transcribe (ASR) ({len(without_text)} source(s) with no text) ---")
            from hindi_tts_builder.data.transcribe import transcribe_sources

            s1b = transcribe_sources(
                paths, manifest,
                language=cfg.get("language", "hi"),
                model=asr_cfg["model"],
                compute_type=asr_cfg["compute_type"],
                beam_size=asr_cfg["beam_size"],
                min_seconds=seg_cfg["min_seconds"],
                max_seconds=seg_cfg["max_seconds"],
                max_gap_seconds=seg_cfg["max_gap_seconds"],
                logger=log,
            )

    if trust_srt or mode == "aligned_words":
        why = "--trust-srt set" if trust_srt else "mode=aligned_words uses ASR word timings"
        log.info(f"--- Stage 2: align --- (SKIPPED, {why})")
        s2 = {"aligned": 0, "skipped": len(manifest), "failed": 0, "fallback_used": 0,
              "reason": "trust_srt" if trust_srt else "aligned_words"}
    else:
        log.info("--- Stage 2: align ---")
        s2 = align_transcripts(paths, manifest, language=cfg.get("language", "hi"), use_whisperx=use_whisperx, logger=log)

    log.info(f"--- Stage 3: segment (mode={mode}) ---")
    s3 = segment_clips(
        paths, manifest,
        sample_rate=cfg["target_sample_rate"],
        loudness_lufs=cfg["target_loudness_lufs"],
        min_seconds=cfg["clip_min_seconds"],
        max_seconds=cfg["clip_max_seconds"],
        trust_srt=trust_srt,
        trust_srt_pad_left_ms=seg_cfg["pad_left_ms"],
        trust_srt_pad_right_ms=seg_cfg["pad_right_ms"],
        segmentation_mode=mode,
        sentence_min_seconds=seg_cfg["min_seconds"],
        sentence_max_gap_seconds=seg_cfg["max_gap_seconds"],
        sentence_max_interior_gap_seconds=seg_cfg["max_interior_gap_seconds"],
        sentence_max_interior_silence_ratio=seg_cfg["max_interior_silence_ratio"],
        close_synthetic_gaps=seg_cfg["close_synthetic_gaps"],
        refuse_untrusted_timeline=seg_cfg["refuse_untrusted_timeline"],
        logger=log,
    )

    if skip_qc:
        log.info("--- Stage 4: quality filter --- (SKIPPED, --skip-qc set)")
        s4 = _passthrough_qc(paths, manifest, log)
    else:
        log.info("--- Stage 4: quality filter ---")
        qc_cfg = cfg["qc"]
        s4 = quality_filter(
            paths, manifest,
            min_snr_db=qc_cfg["min_snr_db"],
            max_cer_vs_whisper=qc_cfg["max_cer_vs_whisper"],
            max_silence_ratio=qc_cfg["max_silence_ratio"],
            min_seconds=cfg["clip_min_seconds"],
            max_seconds=cfg["clip_max_seconds"],
            use_whisper=use_whisper_qc,
            language=cfg.get("language", "hi"),
            logger=log,
        )
        qc_mode = "full" if use_whisper_qc else "no_whisper"
        for src in manifest.active():
            if src.status.segmented:
                src.qc_mode = qc_mode
        manifest.save()
        write_qc_meta(paths, qc_mode, thresholds=dict(qc_cfg))

    log.info("--- Stage 5: build training set ---")
    # v1: prosody tokens (<falling>, <p_short>, etc.) are disabled because
    # Coqui's character-level Graphemes tokenizer would split a multi-char
    # token like '<falling>' into 9 separate single-char tokens, losing
    # the atomic semantic meaning. Re-enable when an atomic-aware tokenizer
    # subclass is in place. See trainer.py / TASKS.md.
    frontend = HindiFrontend(apply_prosody=False)
    s5 = build_training_set(paths, frontend=frontend, logger=log)

    summary = {
        "download": s1,
        "force_align": s1a,
        "transcribe": s1b,
        "align": s2,
        "segment": s3,
        "qc": s4,
        "dataset": s5,
    }

    # Report the gate verdict rather than enforcing it here — `prepare` should
    # always finish and leave a corpus to inspect. `train` is where it blocks.
    try:
        from hindi_tts_builder.data.gate import check_corpus

        g = check_corpus(project_root, **{k: v for k, v in cfg["gate"].items()})
        summary["gate"] = {"ok": g.ok, "blockers": g.blockers, "warnings": g.warnings,
                           "stats": g.stats}
        for line in g.describe().splitlines():
            (log.error if g.blockers else log.info)(f"[gate] {line}")
    except Exception as e:
        log.warning(f"[gate] could not evaluate: {e}")

    log.info("=== Pipeline complete ===")
    return summary
