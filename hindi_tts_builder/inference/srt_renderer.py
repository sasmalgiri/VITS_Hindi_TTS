"""SRT renderer: translate an SRT file into a single WAV where each cue's
audio sits at that cue's timestamp.

Two timing modes:

1. `fit_to_cue` (default): generate audio for each cue, then either
   compress or extend it slightly to match the SRT's allotted duration.
   Uses a simple time-stretch algorithm (speed adjustment) via ffmpeg/sox
   if available, or silence padding as fallback.

2. `natural`: generate audio at natural pace, ignore cue durations, and
   lay cues back-to-back with small gaps. Original SRT timings are
   discarded. Use this when the SRT has rushed timestamps and you want
   clean narration.

The writer handles overlaps defensively: if two cues overlap in time,
the second starts as soon as the first ends.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import shutil
import subprocess
import tempfile

from hindi_tts_builder.inference.engine import GenerationResult, TTSEngine
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.srt import SrtCue, parse_srt


@dataclass
class RenderProgress:
    cue_index: int
    total_cues: int
    cue_text: str
    validation_passed: bool


class SRTRenderer:
    """Render an SRT file to a single timed WAV using a TTSEngine."""

    def __init__(
        self,
        engine: TTSEngine,
        *,
        mode: str = "fit_to_cue",
        gap_ms_between_cues: int = 200,
        max_speed_ratio: float = 1.3,
        logger=None,
    ):
        if mode not in ("fit_to_cue", "natural"):
            raise ValueError(f"Invalid mode: {mode!r}")
        self.engine = engine
        self.mode = mode
        self.gap_ms = gap_ms_between_cues
        self.max_speed_ratio = max_speed_ratio
        self.log = logger or get_logger("inference.srt_renderer")
        self._have_ffmpeg = shutil.which("ffmpeg") is not None

    # ------------------------------------------------------------------
    def render(
        self,
        srt_path: str | Path,
        output_path: str | Path,
        *,
        progress_callback: Callable[[RenderProgress], None] | None = None,
    ) -> dict:
        """Render `srt_path` to `output_path` (WAV). Returns summary."""
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        srt_path = Path(srt_path)
        output_path = Path(output_path)
        cues = parse_srt(srt_path)
        if not cues:
            raise ValueError(f"No cues found in {srt_path}")

        sr = self.engine.sample_rate
        total_validation_failures = 0

        if self.mode == "natural":
            audio = self._render_natural(cues, sr, progress_callback, counter_ref=[0])
        else:
            audio, total_validation_failures = self._render_fit_to_cue(
                cues, sr, progress_callback
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr, subtype="PCM_16")

        duration_sec = len(audio) / sr
        self.log.info(
            f"rendered {srt_path.name} → {output_path.name} "
            f"({duration_sec:.1f}s, {len(cues)} cues, "
            f"{total_validation_failures} validation failures)"
        )
        return {
            "cues_rendered": len(cues),
            "duration_sec": duration_sec,
            "validation_failures": total_validation_failures,
            "mode": self.mode,
        }

    # ------------------------------------------------------------------
    def _render_fit_to_cue(
        self,
        cues: list[SrtCue],
        sample_rate: int,
        progress_callback,
    ):
        import numpy as np  # type: ignore

        total_sec = max(c.end_sec for c in cues) + 0.5
        total_samples = int(total_sec * sample_rate)
        out = np.zeros(total_samples, dtype=np.float32)

        cursor_sec = 0.0
        validation_failures = 0

        for i, cue in enumerate(cues):
            start_sec = max(cue.start_sec, cursor_sec)
            allotted_sec = max(0.1, cue.end_sec - start_sec)
            result = self.engine.speak(cue.text)

            if result.validation and not result.validation.passed:
                validation_failures += 1

            generated_sec = len(result.audio) / sample_rate
            speed_ratio = generated_sec / allotted_sec
            if speed_ratio > self.max_speed_ratio:
                # Too long even with max speedup — let it run past the cue
                clip = result.audio
            elif speed_ratio > 1.0 + 1e-3:
                clip = self._time_stretch(result.audio, sample_rate, speed_ratio)
            else:
                # Shorter than allotted — pad with silence at end
                pad = int((allotted_sec - generated_sec) * sample_rate)
                clip = np.concatenate([result.audio, np.zeros(pad, dtype=np.float32)])

            # Place into output
            start_idx = int(start_sec * sample_rate)
            end_idx = start_idx + len(clip)
            if end_idx > total_samples:
                # Extend output array
                extra = end_idx - total_samples
                out = np.concatenate([out, np.zeros(extra, dtype=np.float32)])
                total_samples = len(out)
            out[start_idx:end_idx] += clip
            # Clamp any accidental overflow
            cursor_sec = end_idx / sample_rate

            if progress_callback:
                progress_callback(RenderProgress(
                    cue_index=i + 1, total_cues=len(cues),
                    cue_text=cue.text,
                    validation_passed=bool(result.validation and result.validation.passed),
                ))

        # Soft clip
        out = np.clip(out, -1.0, 1.0)
        return out, validation_failures

    def _render_natural(
        self,
        cues: list[SrtCue],
        sample_rate: int,
        progress_callback,
        counter_ref,
    ):
        """Sequential concatenation with silence gaps. Ignores SRT timing."""
        import numpy as np  # type: ignore

        gap_samples = int(self.gap_ms / 1000.0 * sample_rate)
        parts: list = []
        for i, cue in enumerate(cues):
            result = self.engine.speak(cue.text)
            parts.append(result.audio)
            if i < len(cues) - 1 and gap_samples > 0:
                parts.append(np.zeros(gap_samples, dtype=np.float32))
            if progress_callback:
                progress_callback(RenderProgress(
                    cue_index=i + 1, total_cues=len(cues),
                    cue_text=cue.text,
                    validation_passed=bool(result.validation and result.validation.passed),
                ))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return np.clip(audio, -1.0, 1.0)

    # ------------------------------------------------------------------
    def _time_stretch(self, audio, sample_rate: int, ratio: float):
        """Stretch audio by `ratio` (>1 = faster, <1 = slower).

        Uses ffmpeg's atempo filter for quality. Falls back to numpy resample
        (which changes pitch) if ffmpeg not available.
        """
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        ratio = max(0.5, min(2.0, float(ratio)))
        if not self._have_ffmpeg:
            # Fallback: pitch-shift-via-resample (lower quality but works)
            new_len = int(len(audio) / ratio)
            if new_len < 1:
                return audio
            indices = np.linspace(0, len(audio) - 1, new_len).astype(np.int64)
            return audio[indices]

        # Write → ffmpeg atempo → read
        with tempfile.TemporaryDirectory() as td:
            tin = Path(td) / "in.wav"
            tout = Path(td) / "out.wav"
            sf.write(str(tin), audio, sample_rate, subtype="PCM_16")

            # atempo accepts ratios 0.5..2.0 per stage; chain if outside
            stages = self._atempo_chain(ratio)
            afilter = ",".join(f"atempo={r:.4f}" for r in stages)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tin),
                   "-af", afilter, str(tout)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                stretched, _ = sf.read(str(tout))
                if stretched.ndim == 2:
                    stretched = stretched.mean(axis=1)
                return stretched.astype(np.float32)
            except subprocess.CalledProcessError as e:
                self.log.warning(f"ffmpeg time-stretch failed, falling back: {e}")
                new_len = int(len(audio) / ratio)
                indices = np.linspace(0, len(audio) - 1, new_len).astype(np.int64)
                return audio[indices]

    @staticmethod
    def _atempo_chain(ratio: float) -> list[float]:
        """Decompose ratio into a chain of atempo values each in [0.5, 2.0]."""
        stages: list[float] = []
        r = ratio
        while r > 2.0:
            stages.append(2.0)
            r /= 2.0
        while r < 0.5:
            stages.append(0.5)
            r /= 0.5
        stages.append(r)
        return stages
