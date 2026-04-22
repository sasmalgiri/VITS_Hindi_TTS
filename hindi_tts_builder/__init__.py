"""Hindi TTS Builder — private personal Hindi TTS engine trainer.

Primary public API:
    TTSEngine       — load a trained engine and generate audio
    HindiFrontend   — the text preprocessing pipeline (can be used standalone)

Both are exposed via lazy import so importing the top-level package never
triggers heavy dependencies (torch, whisperx, etc.) unless the user actually
asks for them.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["TTSEngine", "HindiFrontend"]


def __getattr__(name: str):
    if name == "TTSEngine":
        from hindi_tts_builder.inference.engine import TTSEngine
        return TTSEngine
    if name == "HindiFrontend":
        from hindi_tts_builder.frontend.pipeline import HindiFrontend
        return HindiFrontend
    raise AttributeError(f"module 'hindi_tts_builder' has no attribute {name!r}")
