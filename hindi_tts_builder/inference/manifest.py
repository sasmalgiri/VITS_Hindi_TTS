"""Engine manifest: describes what an exported engine folder contains.

Written by Trainer.export_engine() during training, read by TTSEngine.load()
during inference. Version mismatches cause a clear error rather than silent
degradation.

Engine folder layout:

    engine/
        manifest.json              # this file
        model.pt                   # trained VITS weights
        tokenizer.json             # HindiTokenizer vocab
        pronunciation_dict.json    # optional, for custom word pronunciations
        training_config.yaml       # snapshot of training config
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json


CURRENT_ENGINE_VERSION = 1


@dataclass
class EngineFrontendSpec:
    prosody_tokens: list[str] = field(default_factory=list)
    apply_schwa_deletion: bool = True
    apply_prosody: bool = True


@dataclass
class EngineManifest:
    engine_version: int = CURRENT_ENGINE_VERSION
    package_version: str = ""
    project_name: str = ""
    language: str = "hi"
    sample_rate: int = 24000
    model_type: str = "vits"
    frontend: EngineFrontendSpec = field(default_factory=EngineFrontendSpec)

    @classmethod
    def load(cls, path: Path) -> "EngineManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        frontend_data = data.pop("frontend", {})
        m = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        m.frontend = EngineFrontendSpec(
            **{k: v for k, v in frontend_data.items() if k in EngineFrontendSpec.__dataclass_fields__}
        )
        return m

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def check_compatible(self) -> None:
        """Raise ValueError if this engine isn't compatible with current code."""
        if self.engine_version != CURRENT_ENGINE_VERSION:
            raise ValueError(
                f"Engine version {self.engine_version} incompatible with this "
                f"installation (requires engine version {CURRENT_ENGINE_VERSION}). "
                "Re-export the engine with the matching package version, or "
                "upgrade/downgrade hindi-tts-builder."
            )
        if self.model_type != "vits":
            raise ValueError(
                f"Unknown model type: {self.model_type!r}. "
                "This version of hindi-tts-builder supports: vits"
            )
        if self.language != "hi":
            raise ValueError(
                f"Engine language is {self.language!r}, but this package is "
                "Hindi-only."
            )
