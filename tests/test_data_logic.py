"""Tests for dataset split and CER."""
from collections import Counter
import pytest

from hindi_tts_builder.data.dataset import _split_for
from hindi_tts_builder.data.qc import _cer


class TestSplitFor:
    def test_deterministic(self):
        assert _split_for("clip_xyz", 0.04, 0.01) == _split_for("clip_xyz", 0.04, 0.01)

    def test_returns_valid_split(self):
        for i in range(100):
            s = _split_for(f"clip_{i}", 0.04, 0.01)
            assert s in ("train", "val", "test")

    def test_distribution_roughly_matches(self):
        counts = Counter()
        N = 10_000
        for i in range(N):
            counts[_split_for(f"clip_{i}", 0.04, 0.01)] += 1
        # Expect ~95/4/1 split; allow generous margin for small sample
        assert 0.93 * N < counts["train"] < 0.97 * N
        assert 0.02 * N < counts["val"] < 0.06 * N
        assert 0.00 * N < counts["test"] < 0.03 * N

    def test_zero_pcts(self):
        # If val_pct=0 and test_pct=0, everything goes to train
        for i in range(50):
            assert _split_for(f"x_{i}", 0.0, 0.0) == "train"


class TestCER:
    def test_perfect_match(self):
        assert _cer("हेलो", "हेलो") == 0.0

    def test_whitespace_ignored(self):
        assert _cer("हेलो दुनिया", "हेलोदुनिया") == 0.0

    def test_complete_mismatch(self):
        # totally different strings -> CER close to 1.0
        assert _cer("हेलो", "क्स्") >= 0.75

    def test_empty_reference(self):
        assert _cer("", "") == 0.0
        assert _cer("", "extra") == 1.0

    def test_single_substitution(self):
        # 1 error out of 3 chars = 1/3
        v = _cer("abc", "abd")
        assert abs(v - 1/3) < 1e-9

    def test_nfc_normalized(self):
        # Decomposed vs composed nukta should match
        decomposed = "क" + "\u093c"
        composed = "क़"
        assert _cer(decomposed, composed) == 0.0
