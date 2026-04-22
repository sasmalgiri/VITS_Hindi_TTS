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


def run_pipeline(
    project_root: Path,
    *,
    use_whisperx: bool = True,
    use_whisper_qc: bool = True,
    logger=None,
) -> dict:
    """Run the full data pipeline end-to-end. Returns a combined summary."""
    paths = ProjectPaths(project_root)
    cfg = load_config(project_root)
    log = logger or get_logger("data.pipeline", paths.logs / "pipeline.log")
    paths.ensure_all()

    manifest = Manifest(_manifest_path(paths))
    if len(manifest) == 0:
        raise RuntimeError(
            "No sources in manifest. Call `add_sources_from_files` first "
            "(or use `hindi-tts-builder add-sources`)."
        )

    log.info(f"=== Pipeline starting for '{cfg['name']}' ({len(manifest)} sources) ===")

    log.info("--- Stage 1: download ---")
    s1 = download_audio(paths, manifest, logger=log)

    log.info("--- Stage 2: align ---")
    s2 = align_transcripts(paths, manifest, language=cfg.get("language", "hi"), use_whisperx=use_whisperx, logger=log)

    log.info("--- Stage 3: segment ---")
    s3 = segment_clips(
        paths, manifest,
        sample_rate=cfg["target_sample_rate"],
        loudness_lufs=cfg["target_loudness_lufs"],
        min_seconds=cfg["clip_min_seconds"],
        max_seconds=cfg["clip_max_seconds"],
        logger=log,
    )

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

    log.info("--- Stage 5: build training set ---")
    frontend = HindiFrontend()
    s5 = build_training_set(paths, frontend=frontend, logger=log)

    summary = {
        "download": s1,
        "align": s2,
        "segment": s3,
        "qc": s4,
        "dataset": s5,
    }
    log.info(f"=== Pipeline complete ===")
    return summary
