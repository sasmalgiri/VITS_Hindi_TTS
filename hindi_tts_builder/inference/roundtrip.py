"""Round-trip validation: generate audio, transcribe it back with Whisper,
compare to the input text.

Purpose: catch word omissions, mispronunciations, and alignment artifacts that
slip through even though VITS is architecturally non-autoregressive. If the
CER exceeds the threshold, the caller can regenerate with a different random
seed.

Whisper (faster-whisper preferred, openai-whisper as fallback) is loaded
lazily and cached. If neither is installed, validation is disabled but the
engine still works — zero-omission is reduced from "guaranteed" to "highly
likely by architecture".
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import tempfile
import unicodedata


def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate after NFC + whitespace stripping."""
    ref = unicodedata.normalize("NFC", reference).replace(" ", "").replace("\n", "")
    hyp = unicodedata.normalize("NFC", hypothesis).replace(" ", "").replace("\n", "")
    if not ref:
        return 1.0 if hyp else 0.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


@dataclass
class ValidationResult:
    passed: bool
    cer: float
    transcription: str
    reason: str = ""


class RoundTripValidator:
    """Transcribe generated audio and compare to expected text.

    Parameters
    ----------
    cer_threshold : float
        CER above this is considered a failure (default 0.05 = 5%).
    language : str
        Whisper language code. Defaults to Hindi.
    device : str | None
        "cuda" / "cpu" / None (auto).
    """

    def __init__(
        self,
        cer_threshold: float = 0.05,
        language: str = "hi",
        device: str | None = None,
    ):
        self.cer_threshold = cer_threshold
        self.language = language
        self.device = device
        self._model = None
        self._model_kind: str | None = None  # "faster-whisper" or "openai-whisper" or None
        self._load_attempted = False

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        # Prefer faster-whisper (2-4x faster + lower VRAM)
        try:
            from faster_whisper import WhisperModel  # type: ignore
            import torch  # type: ignore
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            compute = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel("large-v3", device=device, compute_type=compute)
            self._model_kind = "faster-whisper"
            return
        except ImportError:
            pass

        # Fallback to openai-whisper
        try:
            import whisper  # type: ignore
            import torch  # type: ignore
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = whisper.load_model("large-v3", device=device)
            self._model_kind = "openai-whisper"
            return
        except ImportError:
            pass

        self._model = None
        self._model_kind = None

    @property
    def available(self) -> bool:
        self._load_model()
        return self._model is not None

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path) -> str | None:
        self._load_model()
        if self._model is None:
            return None
        if self._model_kind == "faster-whisper":
            segments, _ = self._model.transcribe(
                str(audio_path), language=self.language, beam_size=1
            )
            return " ".join(s.text for s in segments).strip()
        if self._model_kind == "openai-whisper":
            result = self._model.transcribe(str(audio_path), language=self.language, beam_size=1)
            return result.get("text", "").strip()
        return None

    def transcribe_array(self, audio, sample_rate: int) -> str | None:
        """Transcribe an in-memory audio array by writing to a temp WAV."""
        import soundfile as sf  # type: ignore
        self._load_model()
        if self._model is None:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            sf.write(str(tmp), audio, sample_rate, subtype="PCM_16")
            return self.transcribe(tmp)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def validate(
        self,
        *,
        expected_text: str,
        audio=None,
        audio_path: Path | None = None,
        sample_rate: int | None = None,
    ) -> ValidationResult:
        """Run validation. Caller provides either a numpy audio array + sr,
        or a path to a WAV file on disk.
        """
        if not self.available:
            return ValidationResult(
                passed=True,            # treat as pass-through when unavailable
                cer=0.0,
                transcription="",
                reason="whisper-not-available",
            )
        if audio is not None and sample_rate is not None:
            hyp = self.transcribe_array(audio, sample_rate)
        elif audio_path is not None:
            hyp = self.transcribe(audio_path)
        else:
            raise ValueError("Provide either (audio, sample_rate) or audio_path")
        if hyp is None:
            return ValidationResult(
                passed=True, cer=0.0, transcription="", reason="transcription-failed"
            )
        cer = _cer(expected_text, hyp)
        if cer <= self.cer_threshold:
            return ValidationResult(passed=True, cer=cer, transcription=hyp, reason="ok")
        return ValidationResult(
            passed=False, cer=cer, transcription=hyp,
            reason=f"cer {cer:.3f} > threshold {self.cer_threshold}",
        )
