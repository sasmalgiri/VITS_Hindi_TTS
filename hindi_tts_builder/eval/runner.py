"""Evaluate a trained TTSEngine against a TestSet.

Runs the engine over every item in the test set, transcribes the generated
audio with Whisper, computes CER/WER/RTF per item, and writes a CSV report
with per-category aggregates.

Typical usage:

    from hindi_tts_builder.eval.runner import evaluate
    from hindi_tts_builder import TTSEngine
    from hindi_tts_builder.eval import TestSet

    engine = TTSEngine.load("projects/my_voice/engine")
    test_set = TestSet.load(Path("eval/test_set.json"))
    summary = evaluate(engine, test_set, out_dir=Path("eval/my_voice"))
    print(summary)
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Callable
import csv
import json
import time

from hindi_tts_builder.eval.metrics import (
    ClipMetrics, aggregate_by_category, compute_metrics,
)
from hindi_tts_builder.eval.test_set import TestSet


def evaluate(
    engine,                                 # TTSEngine (duck-typed)
    test_set: TestSet,
    *,
    out_dir: Path,
    transcriber=None,                       # RoundTripValidator-like; if None, uses engine.validator
    compute_utmos: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run the engine over the test set. Write report.csv + report.json.

    Returns a summary dict with per-category aggregates.
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    transcriber = transcriber or getattr(engine, "validator", None)

    metrics_rows: list[ClipMetrics] = []
    total = len(test_set)

    for i, item in enumerate(test_set):
        out_wav = audio_dir / f"{item.id}.wav"

        t0 = time.monotonic()
        result = engine.speak(item.text, output=out_wav, validate=False)
        elapsed = time.monotonic() - t0

        transcription: str | None = None
        if transcriber is not None and getattr(transcriber, "available", False):
            try:
                transcription = transcriber.transcribe(out_wav)
            except Exception:
                transcription = None

        m = compute_metrics(
            item_id=item.id,
            category=item.category,
            reference_text=item.text,
            audio_path=out_wav,
            generation_time_sec=elapsed,
            transcription=transcription,
            compute_utmos=compute_utmos,
        )
        metrics_rows.append(m)

        if progress_callback:
            progress_callback(i + 1, total, item.id)

    # Per-item CSV
    csv_path = out_dir / "report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "category", "cer", "wer", "utmos", "duration_sec", "rtf"])
        for m in metrics_rows:
            w.writerow([
                m.item_id, m.category,
                "" if m.cer is None else f"{m.cer:.4f}",
                "" if m.wer is None else f"{m.wer:.4f}",
                "" if m.utmos is None else f"{m.utmos:.3f}",
                f"{m.duration_sec:.3f}",
                "" if m.rtf is None else f"{m.rtf:.3f}",
            ])

    # Per-category summary
    aggregates = aggregate_by_category(metrics_rows)
    overall = aggregate_by_category([m for m in metrics_rows])  # same input
    summary = {"per_category": aggregates, "n_items": len(metrics_rows)}

    # Overall aggregate (single row)
    overall_row = {}
    for key in ("cer", "wer", "utmos", "rtf", "duration_sec"):
        values = [getattr(m, key) for m in metrics_rows if getattr(m, key) is not None]
        if values:
            overall_row[f"{key}_mean"] = sum(values) / len(values)
    summary["overall"] = overall_row

    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
