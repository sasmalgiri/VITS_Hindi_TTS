"""Tests for hindi_tts_builder.data.cue_merge."""
import pytest

from hindi_tts_builder.data.cue_merge import (
    MergedCue,
    ends_with_clause_break,
    ends_with_terminator,
    merge_cues_to_sentences,
    quotes_balanced,
    segmentation_fingerprint,
    strip_closers,
    terminator_ratio,
    units_from_cues,
)
from hindi_tts_builder.utils.srt import SrtCue


def cues(*spec) -> list[SrtCue]:
    """Build cues from (start, end, text) triples."""
    return [SrtCue(index=i + 1, start_sec=s, end_sec=e, text=t) for i, (s, e, t) in enumerate(spec)]


class TestTerminatorDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "यह एक परीक्षण है।",
            "कैसे हो?",
            "रुको!",
            "ठीक है.",
            "वह बोला॥",
        ],
    )
    def test_hard_terminators(self, text: str):
        assert ends_with_terminator(text)

    @pytest.mark.parametrize(
        "text",
        [
            'उसने कहा "मैं जा रहा हूँ।"',
            "उसने कहा “मैं जा रहा हूँ।”",
            "(वह चला गया।)",
            "यह सही है।  ",
        ],
    )
    def test_terminator_behind_closing_punctuation(self, text: str):
        """18.5% of real Hindi sentences end on a closing quote, not a terminator."""
        assert ends_with_terminator(text)

    @pytest.mark.parametrize("text", ["धीरे-धीरे वे एक", "लेकिन जितनी बड़ी", "", "और"])
    def test_non_terminators(self, text: str):
        assert not ends_with_terminator(text)

    def test_strip_closers_keeps_terminator(self):
        assert strip_closers('कहा "ठीक।"') == 'कहा "ठीक।'

    def test_clause_break(self):
        assert ends_with_clause_break("पहले यह हुआ,")
        assert not ends_with_clause_break("पहले यह हुआ।")


class TestQuoteBalance:
    def test_balanced(self):
        assert quotes_balanced('उसने कहा "ठीक है।"')

    def test_unbalanced_straight(self):
        assert not quotes_balanced('उसने कहा "ठीक है।')

    def test_unbalanced_curly(self):
        assert not quotes_balanced("उसने कहा “ठीक है।")


class TestMergeBasic:
    def test_fragments_merge_to_one_sentence(self):
        c = cues(
            (0.0, 2.0, "धीरे-धीरे वे एक"),
            (2.0, 4.0, "डीसोलेट लैंडस्केप में"),
            (4.0, 6.0, "पहुंचते हैं।"),
        )
        units, stats = merge_cues_to_sentences(c, min_seconds=1.0, max_seconds=15.0)
        assert len(units) == 1
        assert units[0].text == "धीरे-धीरे वे एक डीसोलेट लैंडस्केप में पहुंचते हैं।"
        assert units[0].terminated
        assert units[0].member_indices == [0, 1, 2]
        assert units[0].start_sec == 0.0 and units[0].end_sec == 6.0
        assert stats["terminated"] == 1

    def test_two_sentences_split(self):
        c = cues(
            (0.0, 2.0, "पहला वाक्य है।"),
            (2.0, 4.0, "दूसरा वाक्य है।"),
        )
        units, _ = merge_cues_to_sentences(c, min_seconds=1.0, max_seconds=15.0)
        assert len(units) == 2
        assert all(u.terminated for u in units)

    def test_text_joined_with_single_space(self):
        c = cues((0.0, 2.0, "  पहला   भाग  "), (2.0, 4.0, "दूसरा भाग।"))
        units, _ = merge_cues_to_sentences(c, min_seconds=1.0)
        assert units[0].text == "पहला भाग दूसरा भाग।"

    def test_empty_input(self):
        units, stats = merge_cues_to_sentences([])
        assert units == []
        assert stats["input_cues"] == 0


class TestGapBoundary:
    def test_large_gap_forces_flush_without_terminator(self):
        c = cues(
            (0.0, 2.0, "बिना विराम के चलता"),
            (7.0, 9.0, "फिर दूसरा हिस्सा"),
        )
        units, stats = merge_cues_to_sentences(c, min_seconds=1.0, max_gap_seconds=0.4)
        assert len(units) == 2
        assert not units[0].terminated
        assert stats["flushed_by_gap"] >= 1

    def test_small_gap_still_merges(self):
        c = cues(
            (0.0, 2.0, "पहला हिस्सा"),
            (2.2, 4.0, "दूसरा हिस्सा।"),
        )
        units, _ = merge_cues_to_sentences(c, min_seconds=1.0, max_gap_seconds=0.4)
        assert len(units) == 1
        assert units[0].interior_gap_sec == pytest.approx(0.2)

    def test_interior_gap_is_reported(self):
        c = cues((0.0, 2.0, "एक"), (2.3, 4.0, "दो"), (4.3, 6.0, "तीन।"))
        units, _ = merge_cues_to_sentences(
            c, min_seconds=1.0, max_gap_seconds=0.5, max_interior_gap_seconds=0.5,
            max_interior_silence_ratio=0.9,
        )
        assert len(units) == 1
        assert units[0].interior_gap_sec == pytest.approx(0.6)
        assert units[0].max_interior_gap_sec == pytest.approx(0.3)


class TestOverlongSplit:
    def test_long_run_splits_under_cap(self):
        # 5s cues against a 12s cap: the buffer overshoots to 15s before the
        # length flush fires, so the splitter is genuinely exercised.
        c = cues(*[(float(i * 5), float(i * 5 + 5), f"खंड {i},") for i in range(10)])
        units, stats = merge_cues_to_sentences(c, min_seconds=1.0, max_seconds=12.0, max_gap_seconds=0.4)
        assert units, "expected at least one unit"
        assert all(u.duration <= 12.0 for u in units)
        assert stats["split_overlong"] >= 1

    def test_split_only_at_cue_boundaries(self):
        c = cues(*[(float(i * 5), float(i * 5 + 5), f"खंड {i},") for i in range(10)])
        edges = {x.start_sec for x in c} | {x.end_sec for x in c}
        units, _ = merge_cues_to_sentences(c, min_seconds=1.0, max_seconds=12.0)
        for u in units:
            assert u.start_sec in edges
            assert u.end_sec in edges

    def test_single_oversized_cue_emitted_whole(self):
        c = cues((0.0, 30.0, "एक बहुत लंबा वाक्य जो अकेले ही सीमा से बड़ा है।"))
        units, stats = merge_cues_to_sentences(c, min_seconds=1.0, max_seconds=15.0)
        assert len(units) == 1
        assert units[0].duration == 30.0
        assert units[0].oversized
        assert stats["oversized_single"] == 1


class TestShortFragments:
    def test_short_unit_attaches_forward(self):
        c = cues((0.0, 0.5, "हाँ।"), (0.6, 4.0, "फिर वह चला गया।"))
        units, stats = merge_cues_to_sentences(c, min_seconds=2.0, max_seconds=15.0, max_gap_seconds=0.4)
        assert len(units) == 1
        assert units[0].text == "हाँ। फिर वह चला गया।"
        assert stats["attached_short"] == 1

    def test_short_unit_dropped_when_gap_too_wide(self):
        c = cues((0.0, 0.5, "हाँ।"), (9.0, 12.0, "फिर वह चला गया।"))
        units, stats = merge_cues_to_sentences(c, min_seconds=2.0, max_gap_seconds=0.4)
        assert stats["dropped_short"] == 1
        assert len(units) == 1
        assert units[0].text == "फिर वह चला गया।"


class TestLegacyAdapter:
    def test_one_to_one(self):
        c = cues((0.0, 2.0, "एक"), (2.0, 4.0, "दो।"), (4.0, 6.0, "तीन"))
        units = units_from_cues(c)
        assert len(units) == len(c)
        for u, src in zip(units, c):
            assert u.start_sec == src.start_sec
            assert u.end_sec == src.end_sec
            assert u.text == src.text

    def test_terminated_flag_computed_but_timing_untouched(self):
        c = cues((0.0, 2.0, "दो।"))
        u = units_from_cues(c)[0]
        assert u.terminated
        assert (u.start_sec, u.end_sec) == (0.0, 2.0)

    def test_preserves_srt_index_not_list_position(self):
        """Clip filenames derive from this; a mismatch renames an entire corpus."""
        c = [SrtCue(index=7, start_sec=0.0, end_sec=2.0, text="एक"),
             SrtCue(index=99, start_sec=2.0, end_sec=4.0, text="दो।")]
        units = units_from_cues(c)
        assert [u.source_cue_index for u in units] == [7, 99]
        assert [u.first_index for u in units] == [0, 1]

    def test_merged_units_carry_no_cue_index(self):
        c = cues((0.0, 2.0, "एक"), (2.0, 4.0, "दो।"))
        units, _ = merge_cues_to_sentences(c, min_seconds=1.0)
        assert units[0].source_cue_index is None


class TestTerminatorRatio:
    def test_ratio(self):
        units = [
            MergedCue(0, 1, "a", terminated=True),
            MergedCue(1, 2, "b", terminated=False),
        ]
        assert terminator_ratio(units) == 0.5

    def test_empty(self):
        assert terminator_ratio([]) == 0.0


class TestFingerprint:
    def test_stable_across_calls(self):
        a = segmentation_fingerprint(mode="sentence", max_seconds=15.0, pad_left_ms=50)
        b = segmentation_fingerprint(mode="sentence", max_seconds=15.0, pad_left_ms=50)
        assert a == b

    def test_order_independent(self):
        a = segmentation_fingerprint(mode="sentence", max_seconds=15.0)
        b = segmentation_fingerprint(max_seconds=15.0, mode="sentence")
        assert a == b

    @pytest.mark.parametrize("key,val", [("mode", "cue"), ("max_seconds", 12.0), ("pad_left_ms", 60)])
    def test_any_param_change_changes_hash(self, key, val):
        base = dict(mode="sentence", max_seconds=15.0, pad_left_ms=50)
        changed = dict(base, **{key: val})
        assert segmentation_fingerprint(**base) != segmentation_fingerprint(**changed)
