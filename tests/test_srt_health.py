"""Tests for hindi_tts_builder.data.srt_health."""
import pytest

from hindi_tts_builder.data.srt_health import (
    VERDICT_DEGENERATE,
    VERDICT_NATURAL,
    VERDICT_QUANTIZED,
    VERDICT_SYNTHETIC,
    analyze_timeline,
    close_gaps,
)
from hindi_tts_builder.utils.srt import SrtCue


def natural_cues(n: int = 40) -> list[SrtCue]:
    """Fractional timestamps and varying gaps — what real alignment produces."""
    out, t = [], 0.0
    for i in range(n):
        dur = 3.0 + (i % 7) * 0.37
        gap = 0.13 + (i % 5) * 0.09
        out.append(SrtCue(index=i + 1, start_sec=round(t, 3), end_sec=round(t + dur, 3), text=f"वाक्य {i}।"))
        t += dur + gap
    return out


def gridded_cues(n: int = 40, *, gap: float = 1.0, dur: float = 7.0) -> list[SrtCue]:
    """Integer timestamps, constant gap — the h_tts_1 signature."""
    out, t = [], 0.0
    for i in range(n):
        out.append(SrtCue(index=i + 1, start_sec=t, end_sec=t + dur, text=f"खंड {i}"))
        t += dur + gap
    return out


class TestVerdicts:
    def test_natural_timeline_passes(self):
        rep = analyze_timeline(natural_cues())
        assert rep.verdict == VERDICT_NATURAL
        assert rep.trustworthy
        assert not rep.gaps_uniform

    def test_uniform_gap_grid_is_synthetic(self):
        rep = analyze_timeline(gridded_cues())
        assert rep.verdict == VERDICT_SYNTHETIC
        assert not rep.trustworthy
        assert rep.gaps_uniform
        assert rep.gap_median == pytest.approx(1.0)

    def test_quantized_without_uniform_gaps(self):
        c, t = [], 0.0
        for i in range(40):
            dur = float(3 + (i % 5))
            c.append(SrtCue(index=i + 1, start_sec=t, end_sec=t + dur, text=f"क {i}"))
            t += dur + float(1 + (i % 3))  # integer but varying gap
        rep = analyze_timeline(c)
        assert rep.verdict == VERDICT_QUANTIZED
        assert not rep.trustworthy

    def test_empty_is_degenerate(self):
        rep = analyze_timeline([])
        assert rep.verdict == VERDICT_DEGENERATE

    def test_out_of_order_is_degenerate(self):
        c = [
            SrtCue(1, 0.0, 3.0, "a"),
            SrtCue(2, 10.0, 13.0, "b"),
            SrtCue(3, 4.0, 7.0, "c"),
        ]
        assert analyze_timeline(c).verdict == VERDICT_DEGENERATE

    def test_short_uniform_run_is_not_enough_evidence(self):
        """Under the transition threshold, uniformity is coincidence."""
        rep = analyze_timeline(gridded_cues(5))
        assert not rep.gaps_uniform


class TestMeasurements:
    def test_gap_total_and_fraction(self):
        rep = analyze_timeline(gridded_cues(11, gap=1.0, dur=7.0))
        assert rep.n_transitions == 10
        assert rep.gap_total_sec == pytest.approx(10.0)
        # span = 11*7 + 10*1 = 87
        assert rep.gap_fraction == pytest.approx(10.0 / 87.0, rel=1e-3)

    def test_integer_ratio(self):
        assert analyze_timeline(gridded_cues()).integer_ts_ratio == 1.0
        assert analyze_timeline(natural_cues()).integer_ts_ratio < 0.5

    def test_describe_flags_untrustworthy(self):
        text = analyze_timeline(gridded_cues()).describe()
        assert "SYNTHETIC" in text
        assert "DO NOT CUT" in text


class TestPunctuationDensity:
    """Predicts, before any GPU work, whether a source can be sentence-segmented."""

    def test_healthy_narration(self):
        c = [SrtCue(i + 1, i * 8.0, i * 8.0 + 7.0, "यह एक वाक्य है। और यह दूसरा है।")
             for i in range(30)]
        rep = analyze_timeline(c)
        assert rep.sentence_segmentable
        assert rep.terminators_per_1k > 10
        assert rep.terminators_per_cue == pytest.approx(2.0)

    def test_unpunctuated_source_flagged(self):
        """src_6C-6RvEkqxg: 1 terminator in 1,285 cues, 0% after alignment."""
        c = [SrtCue(i + 1, i * 8.0, i * 8.0 + 7.0, "यह एक वाक्य है और यह दूसरा है")
             for i in range(30)]
        rep = analyze_timeline(c)
        assert not rep.sentence_segmentable
        assert "CANNOT be split" in " ".join(rep.warnings)
        assert "none" in rep.recommended_mode

    def test_recommends_aligned_words_when_sentences_shorter_than_cues(self):
        c = [SrtCue(i + 1, i * 8.0, i * 8.0 + 7.0, "एक। दो। तीन।") for i in range(30)]
        rep = analyze_timeline(c)
        assert rep.terminators_per_cue > 1.2
        assert "aligned_words" in rep.recommended_mode

    def test_recommends_sentence_mode_when_sentences_span_cues(self):
        cues = []
        for i in range(30):
            text = "यह एक लंबा वाक्य है जो कई क्यू तक चलता है" + ("।" if i % 4 == 3 else "")
            cues.append(SrtCue(i + 1, i * 8.0, i * 8.0 + 7.0, text))
        rep = analyze_timeline(cues)
        assert rep.sentence_segmentable
        assert rep.terminators_per_cue <= 1.2
        assert rep.recommended_mode.startswith("sentence")

    def test_describe_includes_punctuation_line(self):
        c = [SrtCue(i + 1, i * 8.0, i * 8.0 + 7.0, "वाक्य है।") for i in range(30)]
        assert "punctuation:" in analyze_timeline(c).describe()


class TestCloseGaps:
    def test_reclaims_every_gap(self):
        c = gridded_cues(10, gap=1.0, dur=7.0)
        out, reclaimed = close_gaps(c, max_close_seconds=2.0)
        assert reclaimed == pytest.approx(9.0)
        for a, b in zip(out, out[1:]):
            assert a.end_sec == pytest.approx(b.start_sec)

    def test_last_cue_untouched(self):
        c = gridded_cues(5)
        out, _ = close_gaps(c)
        assert out[-1].end_sec == c[-1].end_sec

    def test_does_not_mutate_input(self):
        c = gridded_cues(5)
        before = [(x.start_sec, x.end_sec) for x in c]
        close_gaps(c)
        assert [(x.start_sec, x.end_sec) for x in c] == before

    def test_wide_gaps_left_alone(self):
        c = [
            SrtCue(1, 0.0, 3.0, "a"),
            SrtCue(2, 20.0, 23.0, "b"),
        ]
        out, reclaimed = close_gaps(c, max_close_seconds=2.0)
        assert reclaimed == 0.0
        assert out[0].end_sec == 3.0

    def test_text_and_index_preserved(self):
        c = gridded_cues(4)
        out, _ = close_gaps(c)
        assert [x.text for x in out] == [x.text for x in c]
        assert [x.index for x in out] == [x.index for x in c]

    def test_closed_timeline_reads_as_natural_gaps(self):
        """After closing, the gap signature is gone so merging can proceed."""
        out, _ = close_gaps(gridded_cues(40))
        rep = analyze_timeline(out)
        assert rep.gap_total_sec == pytest.approx(0.0)


class TestCueGrouping:
    """Per-cue alignment windows pin words to fabricated boundaries; grouping fixes it."""

    def test_groups_respect_span_budget(self):
        from hindi_tts_builder.data.force_align import _group_cues

        cues = gridded_cues(20, gap=1.0, dur=7.0)   # 8s apart
        groups = _group_cues(cues, segment_seconds=30.0)
        assert len(groups) > 1
        for g in groups:
            assert g[-1].end_sec - g[0].start_sec <= 30.0 + 8.0

    def test_no_cue_lost_or_duplicated(self):
        from hindi_tts_builder.data.force_align import _group_cues

        cues = gridded_cues(37)
        flat = [c for g in _group_cues(cues, segment_seconds=30.0) for c in g]
        assert [c.index for c in flat] == [c.index for c in cues]

    def test_single_cue_group(self):
        from hindi_tts_builder.data.force_align import _group_cues

        cues = gridded_cues(1)
        assert len(_group_cues(cues, segment_seconds=30.0)) == 1

    def test_empty(self):
        from hindi_tts_builder.data.force_align import _group_cues

        assert _group_cues([], segment_seconds=30.0) == []
