"""Compute Character Error Rate (CER) for the existing 2h long-form render.

Uses faster-whisper to back-transcribe the produced WAV, then aligns each
transcription against the ground-truth Hindi text and reports per-sentence
+ overall CER. CER is the standard objective metric for "did the TTS
read the words correctly?"

Run from the main venv (transformers 4.57 + faster-whisper):
    /root/hindi-tts/venv/bin/python scripts/eval_2h_cer.py
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # whisper model may need download

import re
from pathlib import Path

import numpy as np
import soundfile as sf

# Inputs
TEXT_FILE = Path("/root/hindi-tts/eval/long_form_3min.txt")
PER_SENT_DIR = Path("/mnt/c/Users/USER/OneDrive/Desktop/hindi-tts-builder-v1.0.0/longform_ab_2h")
OUT_REPORT = Path("/root/hindi-tts/projects/h_tts_1/cer_report_2h.csv")
OUT_SUMMARY = Path("/root/hindi-tts/projects/h_tts_1/cer_summary_2h.txt")


def normalize_devanagari(text: str) -> str:
    """Strip punctuation + collapse whitespace so CER measures *words*,
    not punctuation conventions ASR may not reproduce."""
    text = re.sub(r"[।?!,—\.\"\']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_error_rate(ref: str, hyp: str) -> float:
    """Standard Levenshtein-distance CER, char-level."""
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1] / len(r)


def main():
    # Load reference text
    refs = [
        line.strip() for line in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(f"loaded {len(refs)} reference sentences")

    # Discover per-sentence WAVs (fall back to single full WAV split is
    # impossible without timestamps, so per-sentence is required)
    per_sent_wavs = sorted(PER_SENT_DIR.glob("s*.wav"))
    if len(per_sent_wavs) < len(refs):
        print(f"  WARNING: only found {len(per_sent_wavs)} per-sentence WAVs in {PER_SENT_DIR}; "
              f"need {len(refs)}. Aborting.")
        return
    per_sent_wavs = per_sent_wavs[:len(refs)]
    print(f"found {len(per_sent_wavs)} per-sentence WAVs")

    # Load faster-whisper (multilingual large-v3 is best for Hindi)
    from faster_whisper import WhisperModel  # type: ignore
    print("\nloading faster-whisper large-v3...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("  loaded.")

    # Transcribe and score each
    rows = [("idx", "ref", "hyp", "cer", "ref_len", "hyp_len", "missed_chars")]
    cer_total_num = 0
    cer_total_den = 0
    per_cer = []

    for i, (ref_text, wav_path) in enumerate(zip(refs, per_sent_wavs)):
        ref_norm = normalize_devanagari(ref_text)
        # transcribe (Hindi prompt for better script)
        segments, info = model.transcribe(str(wav_path), language="hi",
                                          beam_size=5, vad_filter=True)
        hyp_text = " ".join(seg.text.strip() for seg in segments)
        hyp_norm = normalize_devanagari(hyp_text)
        cer = char_error_rate(ref_norm, hyp_norm)
        per_cer.append(cer)
        cer_total_num += int(cer * len(ref_norm))
        cer_total_den += len(ref_norm)
        rows.append((i, ref_text, hyp_text, f"{cer:.3f}",
                     len(ref_norm), len(hyp_norm),
                     max(0, len(ref_norm) - len(hyp_norm))))
        print(f"  s{i:02d}: CER={cer:.3f}  ref_len={len(ref_norm)}  hyp_len={len(hyp_norm)}")

    overall_cer = cer_total_num / max(1, cer_total_den)

    # Write CSV
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("|".join(str(c) for c in row) + "\n")

    # Write summary
    summary = (
        f"=== CER Summary: 2h long-form render ===\n"
        f"sentences scored: {len(per_cer)}\n"
        f"overall CER (length-weighted): {overall_cer:.3f}\n"
        f"mean per-sentence CER: {np.mean(per_cer):.3f}\n"
        f"median per-sentence CER: {np.median(per_cer):.3f}\n"
        f"max per-sentence CER: {max(per_cer):.3f}\n"
        f"\n"
        f"Interpretation:\n"
        f"  CER < 0.05 — excellent (production-grade)\n"
        f"  0.05-0.10 — good (minor word issues)\n"
        f"  0.10-0.20 — acceptable for 16h training\n"
        f"  > 0.20 — significant intelligibility loss\n"
    )
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"detailed report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
