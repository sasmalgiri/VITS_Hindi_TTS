"""Bulk corpus ingest — drop video+SRT pairs in a folder, get a clean project.

Runs the existing data pipeline (extract → segment → validate) per source,
but with a few production-grade additions for large corpora (50–150 h+):

  - Walks a *staging directory* and finds new (media, srt) pairs
  - Idempotent: re-running skips sources already in the project manifest
  - --watch mode: polls every N seconds, picks up newly-dropped pairs
  - Per-source progress + running-total report (hours collected, clips,
    accept/reject ratio, source-by-source breakdown)
  - CPU-only — explicitly does NOT run WhisperX alignment so it can run
    alongside a GPU training job without contention. Uses --trust-srt
    semantics: cuts directly from your SRT timestamps with 50ms left +
    100ms right pad.
  - Optional `--asr-qc` flag to run faster-whisper roundtrip QC after each
    source (uses GPU if available; off by default during training)

Discovery rules (auto-detected):

    staging_dir/                          # nested layout
        ep001/
            ep001.mp4
            ep001.srt
        ep002/
            ep002.mkv
            ep002.srt

  OR

    staging_dir/                          # flat layout
        ep001.mp4
        ep001.srt
        ep002.mp4
        ep002.srt

Source ID is derived from the basename of the SRT (`ep001.srt` -> `ep001`).
Media file matched by basename: any of .mp4 .mkv .mov .webm .m4a .mp3 .wav.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from hindi_tts_builder.data.dataset import build_training_set
from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.data.pipeline import _manifest_path
from hindi_tts_builder.data.segment import segment_clips
from hindi_tts_builder.frontend.pipeline import HindiFrontend
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.audio import write_wav  # noqa: F401  re-export
from hindi_tts_builder.utils.project import ProjectPaths, load_config


MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}


@dataclass
class IngestResult:
    source_id: str
    accepted: int
    rejected: int
    hours: float
    skipped_reason: str | None = None  # set when source was skipped (already done, missing file, etc.)


@dataclass
class IngestRunSummary:
    project: str
    sources_processed: int
    sources_skipped: int
    total_clips_added: int
    total_hours_added: float
    per_source: list[IngestResult] = field(default_factory=list)


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def _find_media_for(srt_path: Path) -> Path | None:
    """Find the media file that pairs with this srt — same basename, any
    known media extension. Looks first in the same directory as the SRT,
    then one level up (for nested layouts where srt + media are siblings).
    """
    stem = srt_path.stem
    for parent in (srt_path.parent, srt_path.parent.parent):
        for ext in MEDIA_EXTS:
            cand = parent / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def discover_pairs(staging_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return [(source_id, media_path, srt_path), …] for every pair found.

    SRT files are the anchor — discovery walks for *.srt and looks up its
    matching media file. Files with no matching media are silently skipped
    (logged elsewhere).
    """
    if not staging_dir.exists():
        return []
    pairs: list[tuple[str, Path, Path]] = []
    seen_ids: set[str] = set()
    for srt in sorted(staging_dir.rglob("*.srt")):
        media = _find_media_for(srt)
        if media is None:
            continue
        # source_id from SRT stem; if duplicate, suffix with parent dir
        sid = srt.stem
        if sid in seen_ids:
            sid = f"{srt.parent.name}_{srt.stem}"
        seen_ids.add(sid)
        pairs.append((sid, media, srt))
    return pairs


# ----------------------------------------------------------------------
# Per-source ingest (uses existing pipeline primitives)
# ----------------------------------------------------------------------
def _already_in_manifest(manifest: Manifest, source_id: str) -> bool:
    for s in manifest.sources:
        if s.id == source_id and s.status.segmented:
            return True
    return False


def _add_source_to_manifest(manifest: Manifest, source_id: str, media: Path,
                            srt: Path, project_root: Path) -> None:
    """Register the (media, srt) pair in the project manifest if not already
    there. Mirrors what `add_sources_from_files` does but for one source.
    """
    if any(s.id == source_id for s in manifest.sources):
        return
    # Copy SRT into project's transcripts dir if not already there
    transcripts = project_root / "sources" / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    dst_srt = transcripts / f"{source_id}.srt"
    if not dst_srt.exists():
        dst_srt.write_bytes(srt.read_bytes())
    # Copy media into project's audio dir
    audio_full = project_root / "audio_full"
    audio_full.mkdir(parents=True, exist_ok=True)
    dst_media = audio_full / media.name
    if not dst_media.exists():
        dst_media.write_bytes(media.read_bytes())
    rel_srt = str(dst_srt.relative_to(project_root))
    rel_media = str(dst_media.relative_to(project_root))
    manifest.add(url=f"local://{source_id}", transcript_path=rel_srt,
                 index=len(manifest.sources))
    # Mark this source as "already downloaded" since we have local media
    src = manifest.sources[-1]
    src.id = source_id
    src.audio_path = rel_media
    src.status.downloaded = True
    manifest.save()


def _process_one_source(
    paths: ProjectPaths,
    manifest: Manifest,
    cfg: dict,
    source_id: str,
    media: Path,
    srt: Path,
    log,
) -> IngestResult:
    if _already_in_manifest(manifest, source_id):
        log.info(f"[skip] {source_id}: already segmented in this project")
        return IngestResult(source_id=source_id, accepted=0, rejected=0,
                            hours=0.0, skipped_reason="already_present")

    _add_source_to_manifest(manifest, source_id, media, srt, paths.root)

    # Run segment with --trust-srt mode (skips alignment, uses SRT directly)
    log.info(f"[ingest] {source_id}: segmenting {media.name}…")
    s3 = segment_clips(
        paths, manifest,
        sample_rate=cfg["target_sample_rate"],
        loudness_lufs=cfg["target_loudness_lufs"],
        min_seconds=cfg["clip_min_seconds"],
        max_seconds=cfg["clip_max_seconds"],
        trust_srt=True,
        logger=log,
    )

    # Find this source's row in the just-updated metadata
    accepted = s3.get("clips_created", 0)
    rejected = s3.get("clips_rejected", 0)
    # Estimate hours added by reading clip files for this source
    aligned = paths.aligned / source_id
    hours = 0.0
    if aligned.exists():
        import soundfile as sf  # type: ignore
        for w in aligned.glob(f"{source_id}_c*.wav"):
            try:
                info = sf.info(str(w))
                hours += info.frames / float(info.samplerate)
            except Exception:
                pass
        hours /= 3600.0
    return IngestResult(source_id=source_id, accepted=accepted, rejected=rejected, hours=hours)


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def ingest_directory(
    project_root: Path,
    staging_dir: Path,
    *,
    rebuild_training_set_after: bool = True,
    logger=None,
) -> IngestRunSummary:
    """Walk staging_dir, ingest every (media, srt) pair not already in the
    project. Returns a summary report.

    Idempotent: skips sources already segmented in the project manifest.
    """
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    cfg = load_config(project_root)
    log = logger or get_logger("data.corpus_ingest", paths.logs / "corpus_ingest.log")

    manifest = Manifest(_manifest_path(paths))
    pairs = discover_pairs(staging_dir)
    log.info(f"=== corpus-ingest: {len(pairs)} pair(s) discovered in {staging_dir} ===")

    summary = IngestRunSummary(
        project=project_root.name,
        sources_processed=0,
        sources_skipped=0,
        total_clips_added=0,
        total_hours_added=0.0,
    )

    for source_id, media, srt in pairs:
        try:
            res = _process_one_source(paths, manifest, cfg, source_id, media, srt, log)
        except Exception as e:
            log.error(f"[fail] {source_id}: {e}")
            summary.per_source.append(IngestResult(
                source_id=source_id, accepted=0, rejected=0, hours=0.0,
                skipped_reason=f"error:{e}",
            ))
            summary.sources_skipped += 1
            continue
        summary.per_source.append(res)
        if res.skipped_reason:
            summary.sources_skipped += 1
        else:
            summary.sources_processed += 1
            summary.total_clips_added += res.accepted
            summary.total_hours_added += res.hours

    if rebuild_training_set_after and summary.sources_processed > 0:
        # Rebuild train/val/test CSVs to include the newly added sources.
        # Uses prosody-disabled frontend per project convention.
        log.info("rebuilding training_set CSVs to include new sources…")
        frontend = HindiFrontend(apply_prosody=False)
        s5 = build_training_set(paths, frontend=frontend, logger=log)
        log.info(f"training_set rebuilt: {s5}")

    log.info(
        f"=== ingest done: +{summary.sources_processed} sources, "
        f"+{summary.total_clips_added} clips, +{summary.total_hours_added:.2f}h ==="
    )
    return summary


def watch_directory(
    project_root: Path,
    staging_dir: Path,
    *,
    poll_interval_sec: float = 30.0,
    rebuild_training_set_after: bool = True,
    logger=None,
) -> Iterator[IngestRunSummary]:
    """Long-running watch mode: poll staging_dir every `poll_interval_sec`
    and ingest any newly-dropped pairs. Yields a summary after each pass.

    Use Ctrl-C to stop. Idempotent across passes.
    """
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    log = logger or get_logger("data.corpus_ingest", paths.logs / "corpus_ingest.log")
    log.info(f"watching {staging_dir} (poll every {poll_interval_sec}s; Ctrl-C to stop)")
    while True:
        try:
            summary = ingest_directory(
                project_root, staging_dir,
                rebuild_training_set_after=rebuild_training_set_after,
                logger=log,
            )
            yield summary
        except KeyboardInterrupt:
            log.info("watch interrupted by user")
            return
        time.sleep(poll_interval_sec)
