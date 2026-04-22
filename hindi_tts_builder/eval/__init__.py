"""Evaluation harness: build locked test sets, compute metrics, compare against baselines."""
from __future__ import annotations

__all__ = ["TestSet", "compute_metrics", "cer", "wer"]


def __getattr__(name: str):
    if name == "TestSet":
        from hindi_tts_builder.eval.test_set import TestSet
        return TestSet
    if name == "compute_metrics":
        from hindi_tts_builder.eval.metrics import compute_metrics
        return compute_metrics
    if name == "cer":
        from hindi_tts_builder.eval.metrics import cer
        return cer
    if name == "wer":
        from hindi_tts_builder.eval.metrics import wer
        return wer
    raise AttributeError(f"module 'hindi_tts_builder.eval' has no attribute {name!r}")
