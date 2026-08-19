"""Tests for hindi_tts_builder.data.sentence_split."""
import pytest

from hindi_tts_builder.data.sentence_split import (
    Word,
    split_words_into_sentences,
    words_from_whisper_segments,
)


def words(*spec) -> list[Word]:
    """Build a word stream from (text, start, end) triples."""
    return [Word(text=t, start=s, end=e) for t, s, e in spec]


def stream(texts, *, start=0.0, dur=0.4, gap=0.05) -> list[Word]:
    """Evenly spaced words — the common case for a fluent run of speech."""
    out, t = [], start
    for x in texts:
        out.append(Word(text=x, start=round(t, 3), end=round(t + dur, 3)))
        t += dur + gap
    return out


class TestWhisperAdapter:
    def test_dict_shaped_segments(self):
        segs = [{"words": [{"word": " नमस्ते", "start": 0.0, "end": 0.5},
                           {"word": " दुनिया।", "start": 0.5, "end": 1.0}]}]
        w = words_from_whisper_segments(segs)
        assert [x.text for x in w] == ["नमस्ते", "दुनिया।"]
        assert w[0].start == 0.0 and w[1].end == 1.0

    def test_attribute_shaped_segments(self):
        class W:
            def __init__(self, word, start, end):
                self.word, self.start, self.end = word, start, end

        class S:
            def __init__(self, words):
                self.words = words

        w = words_from_whisper_segments([S([W("एक", 0.0, 0.3), W("दो।", 0.3, 0.7)])])
        assert [x.text for x in w] == ["एक", "दो।"]

    def test_words_without_timings_are_skipped_not_guessed(self):
        segs = [{"words": [{"word": "एक", "start": 0.0, "end": 0.3},
                           {"word": "दो", "start": None, "end": None}]}]
        assert len(words_from_whisper_segments(segs)) == 1

    def test_segment_without_words_ignored(self):
        assert words_from_whisper_segments([{"text": "कुछ"}]) == []


class TestSentenceBoundaries:
    def test_splits_at_terminator_inside_a_run(self):
        """The case cue-merging cannot reach: two sentences in one span."""
        w = stream(["पहला", "वाक्य", "है।", "दूसरा", "वाक्य", "है।"])
        units, stats = split_words_into_sentences(w, min_seconds=0.5, max_seconds=15.0)
        assert len(units) == 2
        assert units[0].text == "पहला वाक्य है।"
        assert units[1].text == "दूसरा वाक्य है।"
        assert all(u.terminated for u in units)
        assert stats["terminated"] == 2

    def test_terminator_behind_quote(self):
        w = stream(["उसने", "कहा", '"ठीक।"', "फिर", "चला", "गया।"])
        units, _ = split_words_into_sentences(w, min_seconds=0.5)
        assert len(units) == 2

    def test_gap_flushes_without_terminator(self):
        w = words(("बिना", 0.0, 0.4), ("विराम", 0.45, 0.9), ("फिर", 5.0, 5.4))
        units, stats = split_words_into_sentences(w, min_seconds=0.1, max_gap_seconds=0.6)
        assert len(units) == 2
        assert not units[0].terminated
        assert stats["flushed_by_gap"] == 1

    def test_unterminated_tail_still_emitted(self):
        w = stream(["एक", "अधूरा", "वाक्य"])
        units, _ = split_words_into_sentences(w, min_seconds=0.1)
        assert len(units) == 1
        assert not units[0].terminated


class TestPadding:
    def test_pad_applied_both_sides(self):
        w = words(("नमस्ते।", 5.0, 5.5))
        units, _ = split_words_into_sentences(
            w, min_seconds=0.1, pad_left_ms=50, pad_right_ms=100
        )
        assert units[0].start_sec == pytest.approx(4.95)
        assert units[0].end_sec == pytest.approx(5.6)

    def test_pad_clamped_at_zero(self):
        w = words(("नमस्ते।", 0.01, 0.5))
        units, _ = split_words_into_sentences(w, min_seconds=0.1, pad_left_ms=200)
        assert units[0].start_sec == 0.0

    def test_pad_clamped_at_audio_end(self):
        w = words(("नमस्ते।", 5.0, 5.9))
        units, _ = split_words_into_sentences(
            w, min_seconds=0.1, pad_right_ms=500, audio_duration=6.0
        )
        assert units[0].end_sec == pytest.approx(6.0)


class TestLengthDiscipline:
    def test_long_run_split_under_cap(self):
        w = stream([f"शब्द{i}," for i in range(80)], dur=0.4, gap=0.05)
        units, stats = split_words_into_sentences(w, min_seconds=1.0, max_seconds=8.0)
        assert units
        assert all(u.duration <= 8.0 + 0.2 for u in units), [u.duration for u in units]
        assert stats["split_overlong"] >= 1

    def test_no_text_lost_when_splitting(self):
        w = stream([f"शब्द{i}," for i in range(40)])
        units, _ = split_words_into_sentences(w, min_seconds=0.5, max_seconds=6.0)
        joined = " ".join(u.text for u in units).split()
        assert joined == [x.text for x in w]

    def test_short_unit_attaches_forward(self):
        w = words(("हाँ।", 0.0, 0.4), ("फिर", 0.5, 0.9), ("वह", 0.95, 1.3), ("गया।", 1.35, 2.6))
        units, stats = split_words_into_sentences(w, min_seconds=1.5, max_seconds=15.0)
        assert len(units) == 1
        assert units[0].text == "हाँ। फिर वह गया।"
        assert stats["attached_short"] == 1

    def test_short_unit_is_kept_not_deleted_when_it_cannot_attach(self):
        """Silently deleting a short unit discards a complete sentence.

        It must survive to the duration filter in segment.py, which rejects it
        visibly, rather than vanishing into a counter nothing reads.
        """
        w = words(("हाँ।", 0.0, 0.4), ("बहुत", 30.0, 30.5), ("बाद", 30.6, 31.0), ("में।", 31.1, 32.5))
        units, stats = split_words_into_sentences(w, min_seconds=1.5, max_gap_seconds=0.6)
        assert stats.get("dropped_short", 0) == 0
        assert len(units) == 2
        assert "हाँ।" in " ".join(u.text for u in units)

    def test_no_text_is_ever_lost_to_the_short_pass(self):
        w, t = [], 0.0
        for _ in range(200):
            for j in range(4):
                w.append(Word(text=("शब्द" if j < 3 else "अंत।"), start=round(t, 3), end=round(t + 0.3, 3)))
                t += 0.35
            t += 0.6                      # pause wider than max_gap_seconds
        units, _ = split_words_into_sentences(w, min_seconds=2.0, max_seconds=15.0, max_gap_seconds=0.4)
        assert sum(len(u.text.split()) for u in units) == len(w)


class TestEdges:
    def test_empty_input(self):
        units, stats = split_words_into_sentences([])
        assert units == []
        assert stats["input_words"] == 0

    def test_units_are_monotonic_and_non_overlapping(self):
        w = stream(["एक", "दो।", "तीन", "चार।", "पाँच", "छह।"])
        units, _ = split_words_into_sentences(w, min_seconds=0.1, pad_left_ms=0, pad_right_ms=0)
        for a, b in zip(units, units[1:]):
            assert a.end_sec <= b.start_sec
            assert a.start_sec < a.end_sec
