"""Training subsystem: VITS training from scratch on single-speaker Hindi data.

Public API:
    TrainingConfig   — dataclass + YAML config
    HindiTokenizer   — character + prosody-token tokenizer
    TTSDataset       — PyTorch Dataset reading train.csv
    Trainer          — the training orchestrator (Coqui TTS under the hood)

All heavy imports (torch, coqui-tts) are lazy. Importing `hindi_tts_builder`
does not trigger torch imports.
"""
from __future__ import annotations

__all__ = ["TrainingConfig", "HindiTokenizer", "TTSDataset", "Trainer"]


def __getattr__(name: str):
    if name == "TrainingConfig":
        from hindi_tts_builder.train.config import TrainingConfig
        return TrainingConfig
    if name == "HindiTokenizer":
        from hindi_tts_builder.train.tokenizer import HindiTokenizer
        return HindiTokenizer
    if name == "TTSDataset":
        from hindi_tts_builder.train.dataset import TTSDataset
        return TTSDataset
    if name == "Trainer":
        from hindi_tts_builder.train.trainer import Trainer
        return Trainer
    raise AttributeError(f"module 'hindi_tts_builder.train' has no attribute {name!r}")
