"""Coqui VITS backend.

Thin adapter that delegates to the existing `train.trainer.Trainer`.
Kept as an adapter (not a re-implementation) so v4 and earlier runs
continue to work bit-for-bit.

When v5 lands and the indic-parler / f5-hindi backends are mature,
the trainer.Trainer logic can be lifted directly into this file and
the legacy module deleted.
"""
from __future__ import annotations
from pathlib import Path

from hindi_tts_builder.train.backends.base import TrainerBackend


class CoquiVitsBackend(TrainerBackend):
    name = "coqui-vits"
    supports_resume = True

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        # Heavy import: only when this backend is actually selected.
        from hindi_tts_builder.train.trainer import Trainer
        self._trainer = Trainer(self.project_root)

    def prepare(self) -> dict:
        return self._trainer.prepare()

    def train(self) -> None:
        self._trainer.train()

    def export_engine(self) -> Path:
        return self._trainer.export_engine()
