"""Tests for SRT renderer helpers that don't require the engine."""
import pytest

from hindi_tts_builder.inference.srt_renderer import SRTRenderer


class TestAtempoChain:
    def test_within_range_single_stage(self):
        # Ratios 0.5..2.0 produce one stage
        assert SRTRenderer._atempo_chain(1.0) == [1.0]
        assert SRTRenderer._atempo_chain(1.5) == [1.5]
        assert SRTRenderer._atempo_chain(2.0) == [2.0]
        assert SRTRenderer._atempo_chain(0.5) == [0.5]

    def test_above_range_chained(self):
        stages = SRTRenderer._atempo_chain(3.0)
        # Should include one 2.0 then the remainder
        assert stages[0] == 2.0
        # Product equals original ratio
        prod = 1.0
        for s in stages:
            prod *= s
        assert abs(prod - 3.0) < 1e-6

    def test_product_preserved(self):
        for r in [0.3, 0.7, 1.2, 2.5, 4.0]:
            stages = SRTRenderer._atempo_chain(r)
            prod = 1.0
            for s in stages:
                prod *= s
            assert abs(prod - r) < 1e-6
            for s in stages:
                assert 0.5 <= s <= 2.0
