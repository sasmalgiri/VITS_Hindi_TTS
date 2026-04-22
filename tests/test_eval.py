"""Tests for hindi_tts_builder.eval."""
from pathlib import Path
import pytest

from hindi_tts_builder.eval.metrics import cer, wer, aggregate_by_category, ClipMetrics
from hindi_tts_builder.eval.test_set import TestSet, TestItem, VALID_CATEGORIES


class TestCERAndWER:
    def test_cer_identical(self):
        assert cer("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0

    def test_wer_identical(self):
        assert wer("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0

    def test_cer_word_order_preserved(self):
        # Different word order — WER counts every token wrong
        c = cer("abc def", "def abc")
        w = wer("abc def", "def abc")
        assert w >= c  # WER typically >= CER for reordering

    def test_cer_handles_empty(self):
        assert cer("", "") == 0.0
        assert cer("", "extra") == 1.0
        assert cer("extra", "") == 1.0


class TestItemValidation:
    def test_valid_category(self):
        it = TestItem(id="x", category="narration", text="त")
        assert it.category == "narration"

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            TestItem(id="x", category="bogus", text="त")


class TestTestSet:
    def test_build_from_category_dict(self):
        per_cat = {
            "narration": ["एक", "दो", "तीन"],
            "dialogue": ["क्या?"],
            "numeric": ["2026 में"],
        }
        ts = TestSet.from_category_dict(per_cat)
        counts = ts.count_by_category()
        assert counts["narration"] == 3
        assert counts["dialogue"] == 1
        assert counts["numeric"] == 1
        assert counts["pronunciation"] == 0
        assert len(ts) == 5

    def test_ids_are_deterministic(self):
        per_cat = {"narration": ["नमस्ते"]}
        ts1 = TestSet.from_category_dict(per_cat)
        ts2 = TestSet.from_category_dict(per_cat)
        assert ts1.items[0].id == ts2.items[0].id

    def test_skips_empty_strings(self):
        per_cat = {"narration": ["  ", "real", ""]}
        ts = TestSet.from_category_dict(per_cat)
        assert len(ts) == 1
        assert ts.items[0].text == "real"

    def test_save_and_load(self, tmp_path: Path):
        per_cat = {
            "narration": ["एक", "दो"],
            "dialogue": ["क्या?"],
        }
        ts1 = TestSet.from_category_dict(per_cat)
        path = tmp_path / "ts.json"
        ts1.save(path)
        ts2 = TestSet.load(path)
        assert len(ts2) == len(ts1)
        assert ts2.count_by_category() == ts1.count_by_category()

    def test_by_category(self):
        ts = TestSet.from_category_dict({
            "narration": ["a", "b"],
            "dialogue": ["c"],
        })
        assert len(ts.by_category("narration")) == 2
        assert len(ts.by_category("dialogue")) == 1


class TestAggregateByCategory:
    def test_basic(self):
        metrics = [
            ClipMetrics("a", "narration", cer=0.01, wer=0.02, utmos=None, duration_sec=2.0, rtf=0.3),
            ClipMetrics("b", "narration", cer=0.03, wer=0.04, utmos=None, duration_sec=3.0, rtf=0.4),
            ClipMetrics("c", "dialogue", cer=0.05, wer=0.06, utmos=None, duration_sec=1.5, rtf=0.5),
        ]
        result = aggregate_by_category(metrics)
        assert result["narration"]["n"] == 2
        assert result["dialogue"]["n"] == 1
        assert abs(result["narration"]["cer_mean"] - 0.02) < 1e-9

    def test_ignores_none(self):
        metrics = [
            ClipMetrics("a", "narration", cer=None, wer=0.01, utmos=None, duration_sec=1.0, rtf=None),
            ClipMetrics("b", "narration", cer=0.02, wer=None, utmos=None, duration_sec=1.0, rtf=None),
        ]
        result = aggregate_by_category(metrics)
        # cer_mean derived only from the non-None value
        assert result["narration"]["cer_mean"] == 0.02
        assert result["narration"]["wer_mean"] == 0.01
