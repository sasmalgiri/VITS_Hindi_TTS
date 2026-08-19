"""SPRINGLab F5-Hindi-24KHz fine-tune backend.

Status: STUB. Implementation deferred until v5 (after Indic-Parler track
proves out). See ROADMAP.md.

Implementation plan (when wired up):

  1. Install: `pip install git+https://github.com/SWivid/F5-TTS.git`
  2. Base: `SPRINGLab/F5-Hindi-24KHz` (151 M params, CC-BY-4.0).
  3. Native 24 kHz Hindi — our pipeline already outputs 24 kHz, no resample.
  4. Important: do NOT call F5-TTS's default `convert_char_to_pinyin` for
     Hindi — pass Devanagari through unchanged. (Per F5-Hindi model card.)
  5. Per-clip metadata file with `audio_path|text` (similar to our format).
     A converter helper belongs in `data/exporters/f5_metadata.py`.
  6. Comfortable on 12 GB at batch=2-4 in bf16. Full fine-tune only — LoRA
     not yet supported per upstream maintainers.
  7. Verify IndicVoices-R training-data licenses before commercial release.
"""
from __future__ import annotations
from pathlib import Path

from hindi_tts_builder.train.backends.base import TrainerBackend


_NOT_IMPLEMENTED_MSG = (
    "f5-hindi backend is a stub (v5). Implementation plan in "
    "hindi_tts_builder/train/backends/f5_hindi.py and ROADMAP.md."
)


class F5HindiBackend(TrainerBackend):
    name = "f5-hindi"
    supports_resume = True

    def prepare(self) -> dict:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def train(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def export_engine(self) -> Path:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
