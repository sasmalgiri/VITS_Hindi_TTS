"""Inference subsystem: load a trained engine and generate audio.

Public API:
    TTSEngine        — main engine class (load + generate)
    RoundTripValidator — Whisper-based word-omission check
    SRTRenderer      — SRT file → timed audio output
"""
from __future__ import annotations

__all__ = ["TTSEngine", "RoundTripValidator", "SRTRenderer"]


def __getattr__(name: str):
    if name == "TTSEngine":
        from hindi_tts_builder.inference.engine import TTSEngine
        return TTSEngine
    if name == "RoundTripValidator":
        from hindi_tts_builder.inference.roundtrip import RoundTripValidator
        return RoundTripValidator
    if name == "SRTRenderer":
        from hindi_tts_builder.inference.srt_renderer import SRTRenderer
        return SRTRenderer
    raise AttributeError(f"module 'hindi_tts_builder.inference' has no attribute {name!r}")
