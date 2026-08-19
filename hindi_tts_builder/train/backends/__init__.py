"""Trainer backends.

Each backend wraps one upstream model framework and exposes the same
prepare()/train()/export_engine() contract. Selection happens at CLI
call time via `--backend <name>`.

Registry:

    coqui-vits     current production trainer (`coqui_vits.CoquiVitsBackend`)
    indic-parler   v5 primary, AI4Bharat Indic-Parler-TTS fine-tune (stub)
    f5-hindi       v5 secondary, SPRINGLab F5-Hindi-24KHz fine-tune (stub)

Backends are imported lazily because each one pulls in heavy framework
dependencies (Coqui-TTS, parler-tts, F5-TTS) that we don't want to load
just to run an unrelated CLI command.
"""
from __future__ import annotations
from typing import Type

from hindi_tts_builder.train.backends.base import TrainerBackend


def get_backend(name: str) -> Type[TrainerBackend]:
    """Resolve a backend name to its class. Raises KeyError on unknown name."""
    if name == "coqui-vits":
        from hindi_tts_builder.train.backends.coqui_vits import CoquiVitsBackend
        return CoquiVitsBackend
    if name == "indic-parler":
        from hindi_tts_builder.train.backends.indic_parler import IndicParlerBackend
        return IndicParlerBackend
    if name == "indic-parler-lora":
        from hindi_tts_builder.train.backends.indic_parler_lora import IndicParlerLoraBackend
        return IndicParlerLoraBackend
    if name == "f5-hindi":
        from hindi_tts_builder.train.backends.f5_hindi import F5HindiBackend
        return F5HindiBackend
    raise KeyError(
        f"Unknown backend: {name!r}. "
        "Known: coqui-vits, indic-parler, indic-parler-lora, f5-hindi"
    )


__all__ = ["TrainerBackend", "get_backend"]
