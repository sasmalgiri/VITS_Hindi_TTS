"""Trainer backend ABC.

The contract every backend must implement so the CLI can drive any of
them with the same code path. Keep it small — only the three operations
the CLI actually calls.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class TrainerBackend(ABC):
    """Abstract base for all training backends.

    Subclasses should be safe to instantiate without GPU/heavy framework
    imports — defer those to inside `prepare()` / `train()` so a CLI like
    `audit` doesn't pull in TTS/parler/f5 just to read a CSV.
    """

    name: str = "base"
    supports_resume: bool = False

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    @abstractmethod
    def prepare(self) -> dict:
        """Validate data, fit any per-project tokenizer, write training_config.

        Returns a small summary dict for the CLI to display.
        Must NOT touch the GPU.
        """

    @abstractmethod
    def train(self) -> None:
        """Long-running. Call only after prepare(). Resumable iff supports_resume."""

    @abstractmethod
    def export_engine(self) -> Path:
        """Bundle the trained model + tokenizer + frontend state into
        `<project>/engine/`. Returns the engine dir path.
        """
