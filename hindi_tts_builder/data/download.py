"""Stage 1: Download audio from YouTube URLs using yt-dlp.

Resumable: re-runs skip sources whose WAV already exists with non-zero size.
Failures are recorded in the manifest so other sources can continue.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess

from hindi_tts_builder.data.manifest import Manifest, Source
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.project import ProjectPaths


def _find_yt_dlp() -> str:
    path = shutil.which("yt-dlp")
    if not path:
        raise RuntimeError(
            "yt-dlp not found on PATH. Install with: pip install yt-dlp"
        )
    return path


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. On WSL2 Ubuntu: sudo apt install ffmpeg"
        )
    return path


def _probe_duration(audio_path: Path) -> float:
    """Use ffprobe to read audio duration. Returns 0.0 on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            text=True,
            timeout=30,
        )
        return float(out.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def _download_one(url: str, dst: Path, yt_dlp: str) -> None:
    """Download audio as 48kHz mono WAV to `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # yt-dlp will write to dst directly; we use --no-part so partial files
    # from interrupted runs don't masquerade as complete downloads.
    tmp = dst.with_suffix(".partial.wav")
    cmd = [
        yt_dlp,
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 48000 -ac 1",
        "--no-part",
        "--no-playlist",
        "-o", str(tmp.with_suffix("")) + ".%(ext)s",
        url,
    ]
    # yt-dlp will actually produce tmp_path with .wav extension via postprocessor
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    # yt-dlp's -o template strips our .partial to .partial.wav; rename to dst
    produced = tmp.with_suffix(".wav")
    if produced.exists():
        produced.rename(dst)
    elif tmp.exists():
        tmp.rename(dst)
    else:
        raise RuntimeError(f"yt-dlp finished but no output WAV at {produced} or {tmp}")


def download_audio(
    paths: ProjectPaths,
    manifest: Manifest,
    *,
    skip_existing: bool = True,
    logger=None,
) -> dict:
    """Download audio for every source in the manifest.

    Returns a summary dict with counts: {downloaded, skipped, failed, total}.
    Manifest is updated in-place and saved after each download so crashes are
    recoverable.
    """
    log = logger or get_logger("data.download", paths.logs / "download.log")
    yt_dlp = _find_yt_dlp()
    _find_ffmpeg()  # verify available; yt-dlp needs it

    summary = {"downloaded": 0, "skipped": 0, "failed": 0, "total": len(manifest)}

    for src in manifest:
        target = paths.audio_raw / f"{src.id}.wav"
        if skip_existing and target.exists() and target.stat().st_size > 0:
            src.audio_path = str(target.relative_to(paths.root))
            src.status.downloaded = True
            if src.duration_sec is None:
                src.duration_sec = _probe_duration(target)
            summary["skipped"] += 1
            log.info(f"[skip] {src.id} already downloaded")
            continue

        log.info(f"[download] {src.id} ← {src.url}")
        try:
            _download_one(src.url, target, yt_dlp)
            src.audio_path = str(target.relative_to(paths.root))
            src.duration_sec = _probe_duration(target)
            src.status.downloaded = True
            src.error = None
            summary["downloaded"] += 1
            log.info(f"[ok] {src.id} duration={src.duration_sec:.1f}s")
        except Exception as e:
            src.error = f"download failed: {e}"
            src.status.downloaded = False
            summary["failed"] += 1
            log.error(f"[fail] {src.id}: {e}")
        finally:
            # Persist after every source so a crash doesn't lose progress
            manifest.save()

    log.info(
        f"download complete: {summary['downloaded']} new, "
        f"{summary['skipped']} skipped, {summary['failed']} failed "
        f"(of {summary['total']} total)"
    )
    return summary
