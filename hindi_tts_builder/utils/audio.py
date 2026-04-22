"""Audio helpers: resampling, loudness normalization, VAD-based segmentation."""
from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf


def ffmpeg_resample(src: Path, dst: Path, sr: int = 24000, loudness_lufs: float = -23.0) -> None:
    """Resample to target sr, mono, and loudness-normalize in one ffmpeg pass."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", f"loudnorm=I={loudness_lufs}:TP=-2:LRA=7,aresample={sr}",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def compute_snr_db(audio: np.ndarray, frame_len: int = 2048) -> float:
    """Rough SNR estimate: top-10% RMS vs bottom-10% RMS of frames."""
    if len(audio) < frame_len * 2:
        return 0.0
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_sorted = np.sort(rms)
    noise = rms_sorted[: max(1, n_frames // 10)].mean()
    signal = rms_sorted[-max(1, n_frames // 10):].mean()
    if noise < 1e-8:
        return 80.0
    return 20.0 * float(np.log10(signal / noise))


def silence_ratio(audio: np.ndarray, threshold_db: float = -40.0, frame_len: int = 2048) -> float:
    """Fraction of frames whose RMS is below threshold_db."""
    if len(audio) < frame_len:
        return 1.0
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    return float((db < threshold_db).mean())


def trim_silence(audio: np.ndarray, sr: int, pad_ms: int = 50, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading/trailing silence, leaving `pad_ms` of padding."""
    frame_len = max(1, sr // 100)  # 10ms frames
    n_frames = len(audio) // frame_len
    if n_frames < 1:
        return audio
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    voiced = np.where(db > threshold_db)[0]
    if len(voiced) == 0:
        return audio
    start = max(0, voiced[0] * frame_len - int(pad_ms * sr / 1000))
    end = min(len(audio), (voiced[-1] + 1) * frame_len + int(pad_ms * sr / 1000))
    return audio[start:end]
