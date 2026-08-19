"""CER eval for the BASE AI4Bharat long-form render.

Identical methodology to eval_2h_cer.py but pointed at the base output
directory, so the CER number is directly comparable to the 2h CER (0.347).
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import re
from pathlib import Path

import numpy as np

TEXT_FILE = Path("/root/hindi-tts/eval/long_form_3min.txt")
PER_SENT_DIR = Path("/root/hindi-tts/projects/h_tts_1/longform_ab_BASE")
OUT_REPORT = Path("/root/hindi-tts/projects/h_tts_1/cer_report_base.csv")
OUT_SUMMARY = Path("/root/hindi-tts/projects/h_tts_1/cer_summary_base.txt")


def normalize_devanagari(text: str) -> str:
    text = re.sub(r"[।?!,—\.\"\']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_error_rate(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
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
    refs = [
        line.strip() for line in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(f"loaded {len(refs)} reference sentences")

    per_sent_wavs = sorted(PER_SENT_DIR.glob("s*.wav"))
    if len(per_sent_wavs) < len(refs):
        print(f"  WARNING: only {len(per_sent_wavs)} per-sentence WAVs in {PER_SENT_DIR}; need {len(refs)}. Aborting.")
        return
    per_sent_wavs = per_sent_wavs[:len(refs)]
    print(f"found {len(per_sent_wavs)} per-sentence WAVs")

    from faster_whisper import WhisperModel  # type: ignore
    print("\nloading faster-whisper large-v3...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("  loaded.")

    rows = [("idx", "ref", "hyp", "cer", "ref_len", "hyp_len", "missed_chars")]
    cer_total_num = 0
    cer_total_den = 0
    per_cer = []

    for i, (ref_text, wav_path) in enumerate(zip(refs, per_sent_wavs)):
        ref_norm = normalize_devanagari(ref_text)
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

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("|".join(str(c) for c in row) + "\n")

    summary = (
        f"=== CER Summary: BASE ai4bharat/indic-parler-tts long-form render ===\n"
        f"sentences scored: {len(per_cer)}\n"
        f"overall CER (length-weighted): {overall_cer:.3f}\n"
        f"mean per-sentence CER: {np.mean(per_cer):.3f}\n"
        f"median per-sentence CER: {np.median(per_cer):.3f}\n"
        f"max per-sentence CER: {max(per_cer):.3f}\n"
        f"\n"
        f"For comparison, 2h fine-tune scored: 0.347 overall CER.\n"
    )
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"detailed report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
