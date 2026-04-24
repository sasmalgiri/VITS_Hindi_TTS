"""Stage 4: Quality filtering for segmented clips.

Reads every clip under aligned/<source_id>/*.wav, scores it on multiple
quality axes, and records pass/fail + reason in a manifest file:

    training_set/qc_report.csv
      columns: clip_id, source_id, duration, snr_db, silence_ratio,
               whisper_cer, passed, reason

The CSV is the input to the next stage (dataset assembly), which only uses
clips where passed=true.

Whisper CER is optional (computed only if openai-whisper or faster-whisper
is installed). It measures "does Whisper transcribe this clip back to
something close to the written transcript" — a powerful check that catches
mispronunciations, source separation artifacts, and wrong-speaker clips.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import os
import time
import unicodedata

from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.audio import compute_snr_db, read_wav, silence_ratio
from hindi_tts_builder.utils.project import ProjectPaths


@dataclass
class ClipQC:
    clip_id: str
    source_id: str
    duration: float
    snr_db: float
    silence_ratio: float
    whisper_cer: float | None
    passed: bool
    reason: str


def _cer(reference: str, hypothesis: str) -> float:
    """Character error rate ignoring whitespace and NFC-normalized."""
    ref = unicodedata.normalize("NFC", reference).replace(" ", "").replace("\n", "")
    hyp = unicodedata.normalize("NFC", hypothesis).replace(" ", "").replace("\n", "")
    if not ref:
        return 1.0 if hyp else 0.0
    # Levenshtein distance
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


class _LazyWhisper:
    """Load faster-whisper once on first use; share across clips."""

    def __init__(self, language: str = "hi"):
        self.language = language
        self.model = None
        self._tried = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            from faster_whisper import WhisperModel  # type: ignore
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Defaults sized for 12 GB shared with a Windows desktop.
            # Override via env vars when more headroom is available.
            default_compute = "int8_float16" if device == "cuda" else "int8"
            compute = os.environ.get("HTTS_QC_WHISPER_COMPUTE", default_compute)
            model_name = os.environ.get("HTTS_QC_WHISPER_MODEL", "medium")
            self.model = WhisperModel(model_name, device=device, compute_type=compute)
        except ImportError:
            self.model = None

    def transcribe(self, wav_path: Path) -> str | None:
        self._load()
        if self.model is None:
            return None
        segments, _ = self.model.transcribe(str(wav_path), language=self.language, beam_size=1)
        return " ".join(s.text for s in segments).strip()


def quality_filter(
    paths: ProjectPaths,
    manifest: Manifest,
    *,
    min_snr_db: float = 30.0,
    max_cer_vs_whisper: float = 0.10,
    max_silence_ratio: float = 0.25,
    min_seconds: float = 1.5,
    max_seconds: float = 15.0,
    use_whisper: bool = True,
    language: str = "hi",
    logger=None,
) -> dict:
    """Run QC on every segmented clip. Writes qc_report.csv.

    Returns summary {total, passed, failed_snr, failed_silence, failed_cer,
    failed_duration}.
    """
    log = logger or get_logger("data.qc", paths.logs / "qc.log")
    paths.training_set.mkdir(parents=True, exist_ok=True)
    report_path = paths.training_set / "qc_report.csv"

    whisper = _LazyWhisper(language=language) if use_whisper else None

    summary = {
        "total": 0, "passed": 0,
        "failed_snr": 0, "failed_silence": 0,
        "failed_cer": 0, "failed_duration": 0,
    }
    rows: list[ClipQC] = []

    # Pre-count total clips so we can log progress as N/total with an ETA.
    total_clips = 0
    for src in manifest:
        if not src.status.segmented:
            continue
        clip_dir = paths.aligned / src.id
        if clip_dir.exists():
            total_clips += sum(1 for _ in clip_dir.glob(f"{src.id}_c*.wav"))
    log.info(f"[qc] starting on {total_clips} clip(s) across {len(manifest)} source(s)")

    progress_every = max(50, total_clips // 50) if total_clips else 50
    t_started = time.time()

    for src in manifest:
        if not src.status.segmented:
            continue
        clip_dir = paths.aligned / src.id
        if not clip_dir.exists():
            continue
        for clip_wav in sorted(clip_dir.glob(f"{src.id}_c*.wav")):
            clip_id = clip_wav.stem
            clip_txt = clip_wav.with_suffix(".txt")
            if not clip_txt.exists():
                continue
            summary["total"] += 1

            try:
                audio, sr = read_wav(clip_wav)
            except Exception as e:
                log.warning(f"[skip] {clip_id}: cannot read wav ({e})")
                continue

            duration = len(audio) / sr
            snr = compute_snr_db(audio)
            silr = silence_ratio(audio)

            passed = True
            reason = "ok"

            if duration < min_seconds or duration > max_seconds:
                passed = False
                reason = f"duration {duration:.2f}s out of [{min_seconds}, {max_seconds}]"
                summary["failed_duration"] += 1
            elif snr < min_snr_db:
                passed = False
                reason = f"snr {snr:.1f}dB < {min_snr_db}dB"
                summary["failed_snr"] += 1
            elif silr > max_silence_ratio:
                passed = False
                reason = f"silence {silr:.2f} > {max_silence_ratio}"
                summary["failed_silence"] += 1

            cer: float | None = None
            if passed and whisper is not None:
                reference = clip_txt.read_text(encoding="utf-8")
                hypothesis = whisper.transcribe(clip_wav)
                if hypothesis is not None:
                    cer = _cer(reference, hypothesis)
                    if cer > max_cer_vs_whisper:
                        passed = False
                        reason = f"cer {cer:.3f} > {max_cer_vs_whisper}"
                        summary["failed_cer"] += 1

            if passed:
                summary["passed"] += 1

            rows.append(ClipQC(
                clip_id=clip_id,
                source_id=src.id,
                duration=duration,
                snr_db=snr,
                silence_ratio=silr,
                whisper_cer=cer,
                passed=passed,
                reason=reason,
            ))

            # Periodic progress with rate + ETA
            if total_clips and (summary["total"] % progress_every == 0 or summary["total"] == total_clips):
                elapsed = time.time() - t_started
                rate = summary["total"] / elapsed if elapsed > 0 else 0
                remaining = (total_clips - summary["total"]) / rate if rate > 0 else 0
                log.info(
                    f"[qc] progress {summary['total']}/{total_clips} "
                    f"({summary['total']/total_clips*100:.1f}%) "
                    f"passed={summary['passed']} rate={rate:.1f}/s eta={int(remaining)}s"
                )

        src.status.qc_passed = True
        manifest.save()

    # Write report
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "source_id", "duration", "snr_db", "silence_ratio", "whisper_cer", "passed", "reason"])
        for r in rows:
            w.writerow([
                r.clip_id, r.source_id,
                f"{r.duration:.3f}", f"{r.snr_db:.2f}", f"{r.silence_ratio:.3f}",
                "" if r.whisper_cer is None else f"{r.whisper_cer:.4f}",
                int(r.passed), r.reason,
            ])

    log.info(
        f"qc complete: {summary['passed']}/{summary['total']} passed. "
        f"failures: snr={summary['failed_snr']} silence={summary['failed_silence']} "
        f"cer={summary['failed_cer']} duration={summary['failed_duration']}"
    )
    return summary
