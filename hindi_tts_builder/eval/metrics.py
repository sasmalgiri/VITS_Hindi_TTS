"""Evaluation metrics.

All metrics work on pure Python strings so they can be tested without torch.
Numeric-only helpers (UTMOS, speaker similarity) are optional — they require
torchmetrics / WavLM and fall back to None if those aren't installed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import unicodedata


def _normalize_for_compare(s: str) -> str:
    """NFC + lowercase + collapse whitespace."""
    return unicodedata.normalize("NFC", s).lower().strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. Whitespace is ignored for fair comparison."""
    ref = _normalize_for_compare(reference).replace(" ", "").replace("\n", "")
    hyp = _normalize_for_compare(hypothesis).replace(" ", "").replace("\n", "")
    return _levenshtein_ratio(ref, hyp)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate."""
    ref_tokens = _normalize_for_compare(reference).split()
    hyp_tokens = _normalize_for_compare(hypothesis).split()
    return _levenshtein_ratio(ref_tokens, hyp_tokens)


def _levenshtein_ratio(ref, hyp) -> float:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0 if m > 0 else 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


@dataclass
class ClipMetrics:
    item_id: str
    category: str
    cer: float | None
    wer: float | None
    utmos: float | None
    duration_sec: float
    rtf: float | None   # real-time factor: generation_time / audio_duration


def compute_metrics(
    *,
    item_id: str,
    category: str,
    reference_text: str,
    audio_path: Path,
    generation_time_sec: float | None = None,
    transcription: str | None = None,
    compute_utmos: bool = False,
) -> ClipMetrics:
    """Compute metrics for a single generated clip.

    Parameters
    ----------
    reference_text : the input text given to the TTS
    audio_path     : path to generated WAV
    transcription  : Whisper transcription of the audio (if already computed).
                     If None, CER/WER are skipped.
    compute_utmos  : if True, load UTMOS model and score the audio.
    """
    # Duration from WAV
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate
    except Exception:
        duration = 0.0

    cer_value: float | None = None
    wer_value: float | None = None
    if transcription is not None:
        cer_value = cer(reference_text, transcription)
        wer_value = wer(reference_text, transcription)

    utmos_value: float | None = None
    if compute_utmos:
        utmos_value = _compute_utmos(audio_path)

    rtf: float | None = None
    if generation_time_sec is not None and duration > 0:
        rtf = generation_time_sec / duration

    return ClipMetrics(
        item_id=item_id,
        category=category,
        cer=cer_value,
        wer=wer_value,
        utmos=utmos_value,
        duration_sec=duration,
        rtf=rtf,
    )


def _compute_utmos(audio_path: Path) -> float | None:
    """UTMOS is a neural MOS estimator (Saeki et al. 2022).

    Optional dependency. Returns None if the model can't be loaded.
    """
    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore
    except ImportError:
        return None

    # UTMOS has multiple implementations; we try the `sarulab-speech/UTMOS22`
    # style via torch.hub if available.
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        wave, sr = torchaudio.load(str(audio_path))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        score = predictor(wave, sr)
        return float(score.item())
    except Exception:
        return None


def aggregate_by_category(metrics: Iterable[ClipMetrics]) -> dict[str, dict[str, float]]:
    """Group metrics by category and return means."""
    per_cat: dict[str, list[ClipMetrics]] = {}
    for m in metrics:
        per_cat.setdefault(m.category, []).append(m)

    result: dict[str, dict[str, float]] = {}
    for cat, items in per_cat.items():
        row: dict[str, float] = {"n": len(items)}
        for key in ("cer", "wer", "utmos", "rtf", "duration_sec"):
            values = [getattr(m, key) for m in items if getattr(m, key) is not None]
            if values:
                row[f"{key}_mean"] = sum(values) / len(values)
        result[cat] = row
    return result
