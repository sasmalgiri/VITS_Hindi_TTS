"""Data pipeline: YouTube download → alignment → segmentation → QC → training set.

Each stage is idempotent and resumable — re-running a stage skips work that is
already done. This lets you recover from interruptions cleanly.
"""
from __future__ import annotations

__all__ = [
    "download_audio",
    "align_transcripts",
    "segment_clips",
    "quality_filter",
    "build_training_set",
]


def __getattr__(name: str):
    if name == "download_audio":
        from hindi_tts_builder.data.download import download_audio
        return download_audio
    if name == "align_transcripts":
        from hindi_tts_builder.data.align import align_transcripts
        return align_transcripts
    if name == "segment_clips":
        from hindi_tts_builder.data.segment import segment_clips
        return segment_clips
    if name == "quality_filter":
        from hindi_tts_builder.data.qc import quality_filter
        return quality_filter
    if name == "build_training_set":
        from hindi_tts_builder.data.dataset import build_training_set
        return build_training_set
    raise AttributeError(f"module 'hindi_tts_builder.data' has no attribute {name!r}")
