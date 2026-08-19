"""Pre-training dataset audit.

Run before training to catch silent-collapse bugs cheaper than a 16-hour
wasted run. Reports:

  - hours: raw vs accepted
  - clip-count + accept/reject reason histogram
  - duration distribution (mean / p50 / p95 / under-1s / over-10s)
  - loudness range (min / max LUFS proxy via dBFS)
  - text char coverage vs the trainer's CharactersConfig
  - speaker consistency proxy (mean RMS variance across N random clips —
    cheap stand-in for x-vector cosine; flags multi-speaker contamination)

Two entry points: `audit_project(project_root) -> dict` returns the full
report; `print_report(report)` formats it for the CLI.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv
import json
import math
import random

from hindi_tts_builder.utils.project import ProjectPaths


def _read_qc_reasons(qc_csv: Path) -> Counter:
    """Tally the `reason` column from qc_report.csv to surface why clips fail."""
    reasons: Counter = Counter()
    if not qc_csv.exists():
        return reasons
    with qc_csv.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("passed") == "1":
                continue
            reason = (row.get("reason") or "unknown").strip() or "unknown"
            reasons[reason] += 1
    return reasons


def _read_csv_records(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="|"))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _loudness_dbfs(audio_path: Path) -> float | None:
    """Cheap loudness proxy: peak dBFS via soundfile. Not LUFS, but good
    enough to flag wildly-different volume sources without pulling in
    pyloudnorm here."""
    try:
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
        data, _ = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if len(data) == 0:
            return None
        peak = float(np.max(np.abs(data)))
        if peak <= 0:
            return None
        return 20.0 * math.log10(peak)
    except Exception:
        return None


def _speaker_consistency_proxy(audio_paths: list[Path], sample_n: int = 30) -> float | None:
    """Cheap multi-speaker contamination check: stdev of RMS across N random
    clips, normalised to mean. Real x-vector cosine would be better but
    requires loading speechbrain — keep this dependency-free.

    Lower = more consistent. Roughly:
      < 0.20 -> probably one speaker
      0.20-0.40 -> some variation (mic distance, emotion)
      > 0.40 -> likely multiple speakers / inconsistent capture
    """
    try:
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
        paths = audio_paths if len(audio_paths) <= sample_n else random.sample(audio_paths, sample_n)
        rms_values: list[float] = []
        for p in paths:
            try:
                data, _ = sf.read(str(p), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if len(data) == 0:
                    continue
                rms = float(np.sqrt(np.mean(data ** 2)))
                if rms > 0:
                    rms_values.append(rms)
            except Exception:
                continue
        if len(rms_values) < 5:
            return None
        mean = sum(rms_values) / len(rms_values)
        if mean <= 0:
            return None
        var = sum((v - mean) ** 2 for v in rms_values) / len(rms_values)
        return math.sqrt(var) / mean
    except Exception:
        return None


def _text_char_coverage(records: list[dict]) -> dict:
    """Compare every char in processed_text against the trainer's
    CharactersConfig set. Returns {covered, uncovered_sample, n_unique}."""
    try:
        from hindi_tts_builder.train.trainer import _CHARS_FOR_COQUI, _PUNCT_FOR_COQUI  # type: ignore
    except Exception:
        return {"covered": None, "uncovered": [], "n_unique": 0,
                "note": "trainer module unavailable; run with coqui-tts installed"}
    valid = set(_CHARS_FOR_COQUI) | set(_PUNCT_FOR_COQUI)
    seen = set()
    for r in records:
        seen.update(r.get("processed_text", "") or r.get("raw_text", ""))
    missing = sorted(seen - valid)
    return {
        "covered": (len(seen - valid) == 0),
        "uncovered": [(c, f"U+{ord(c):04X}") for c in missing[:20]],
        "n_uncovered": len(missing),
        "n_unique_in_text": len(seen),
    }


def audit_project(project_root: Path, *, sample_n_for_speaker: int = 30) -> dict:
    """Run all audits. Returns a dict suitable for json.dumps + print_report.

    Pure read-only: opens CSVs and audio headers, never mutates.
    """
    paths = ProjectPaths(project_root)
    ts = paths.training_set
    qc_csv = ts / "qc_report.csv"
    train_csv = ts / "train.csv"
    val_csv = ts / "val.csv"
    test_csv = ts / "test.csv"

    train_recs = _read_csv_records(train_csv)
    val_recs = _read_csv_records(val_csv)
    test_recs = _read_csv_records(test_csv)
    all_recs = train_recs + val_recs + test_recs

    durations = [float(r["duration"]) for r in all_recs if r.get("duration")]
    total_clean_hours = sum(durations) / 3600.0
    under_1s = sum(1 for d in durations if d < 1.0)
    over_10s = sum(1 for d in durations if d > 10.0)
    mean_dur = sum(durations) / len(durations) if durations else 0.0
    p50 = _percentile(durations, 0.50)
    p95 = _percentile(durations, 0.95)

    qc_reasons = _read_qc_reasons(qc_csv)
    n_qc_total = sum(1 for _ in qc_csv.open(encoding="utf-8")) - 1 if qc_csv.exists() else len(all_recs)
    n_accepted = len(all_recs)
    n_rejected = max(0, n_qc_total - n_accepted) if n_qc_total else sum(qc_reasons.values())

    # Loudness sample (don't scan all 10k clips — sample 30)
    audio_files: list[Path] = []
    for r in all_recs:
        ap = (r.get("audio_path") or "").strip()
        if not ap:
            continue
        audio_files.append(project_root / ap)
    sample_paths = random.sample(audio_files, min(sample_n_for_speaker, len(audio_files))) if audio_files else []
    loudness_vals = [v for v in (_loudness_dbfs(p) for p in sample_paths) if v is not None]

    speaker_consistency = _speaker_consistency_proxy(audio_files, sample_n=sample_n_for_speaker)
    text_cov = _text_char_coverage(all_recs)

    return {
        "project": project_root.name,
        "totals": {
            "total_clean_hours": round(total_clean_hours, 3),
            "total_clips": len(all_recs),
            "accepted_clips": n_accepted,
            "rejected_clips": n_rejected,
            "qc_reject_reasons": dict(qc_reasons.most_common()),
            "splits": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
        },
        "duration": {
            "mean_seconds": round(mean_dur, 3),
            "p50_seconds": round(p50, 3),
            "p95_seconds": round(p95, 3),
            "clips_under_1s": under_1s,
            "clips_over_10s": over_10s,
        },
        "loudness_dbfs_sampled": {
            "n_samples": len(loudness_vals),
            "min": round(min(loudness_vals), 2) if loudness_vals else None,
            "max": round(max(loudness_vals), 2) if loudness_vals else None,
            "note": "peak dBFS proxy (negative = quieter); large min/max gap suggests inconsistent capture",
        },
        "speaker_consistency_proxy": {
            "value": round(speaker_consistency, 3) if speaker_consistency is not None else None,
            "interpretation": (
                "<0.20 = consistent; 0.20-0.40 = some variation; >0.40 = likely multi-speaker"
                if speaker_consistency is not None else "could not compute"
            ),
        },
        "text_char_coverage": text_cov,
    }


def print_report(report: dict) -> None:
    """Human-readable formatter. Click-friendly; uses plain prints."""
    print(f"\n=== Audit: {report['project']} ===\n")
    t = report["totals"]
    print(f"  total_clean_hours: {t['total_clean_hours']}")
    print(f"  total_clips:       {t['total_clips']}")
    print(f"  accepted / rejected: {t['accepted_clips']} / {t['rejected_clips']}")
    if t["qc_reject_reasons"]:
        print(f"  qc_reject_reasons:")
        for reason, n in t["qc_reject_reasons"].items():
            print(f"    - {reason}: {n}")
    print(f"  splits: {t['splits']}")

    d = report["duration"]
    print(f"\n  duration mean / p50 / p95: {d['mean_seconds']}s / {d['p50_seconds']}s / {d['p95_seconds']}s")
    print(f"  clips_under_1s: {d['clips_under_1s']}    clips_over_10s: {d['clips_over_10s']}")

    l = report["loudness_dbfs_sampled"]
    if l["n_samples"]:
        print(f"\n  loudness (peak dBFS, sampled n={l['n_samples']}): {l['min']} .. {l['max']}")
        print(f"    {l['note']}")

    s = report["speaker_consistency_proxy"]
    if s["value"] is not None:
        print(f"\n  speaker_consistency_proxy: {s['value']}  ({s['interpretation']})")

    c = report["text_char_coverage"]
    if c.get("covered") is True:
        print(f"\n  text_char_coverage: OK ({c['n_unique_in_text']} unique chars all covered)")
    elif c.get("covered") is False:
        print(f"\n  text_char_coverage: FAIL — {c['n_uncovered']} uncovered chars (first 20):")
        for ch, code in c["uncovered"]:
            print(f"    {ch!r}  {code}")
    else:
        print(f"\n  text_char_coverage: skipped ({c.get('note', 'unknown')})")
    print()
