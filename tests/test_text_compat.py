"""Tests for hindi_tts_builder.utils.text_compat."""
import pytest

from hindi_tts_builder.utils.text_compat import (
    PUNCT,
    TRAINABLE,
    fold_punctuation,
    latin_ratio,
    sanitize_for_training,
    text_is_trainable,
    untrainable_chars,
)


class TestMirrorsTrainer:
    def test_char_set_matches_trainer(self):
        """Guard against drift from trainer.py's CharactersConfig."""
        trainer = pytest.importorskip(
            "hindi_tts_builder.train.trainer",
            reason="coqui-tts not installed in this environment",
        )
        assert set(trainer._CHARS_FOR_COQUI) | set(trainer._PUNCT_FOR_COQUI) == set(TRAINABLE)
        assert trainer._PUNCT_FOR_COQUI == PUNCT


class TestFold:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("“ठीक”", '"ठीक"'),
            ("‘हाँ’", "'हाँ'"),
            ("रुको—अभी", "रुको-अभी"),
            ("अच्छा…", "अच्छा."),
            ("क्या।", "क्या।"),
        ],
    )
    def test_folds_typography(self, raw, expected):
        assert fold_punctuation(raw) == expected

    def test_folded_output_is_trainable(self):
        raw = "उसने कहा “ठीक है”—फिर चला गया…"
        assert not text_is_trainable(raw)
        assert text_is_trainable(fold_punctuation(raw))

    def test_zero_width_chars_removed(self):
        assert fold_punctuation("क​ख") == "कख"

    def test_devanagari_untouched(self):
        s = "नमस्ते दुनिया। क्या हाल है?"
        assert fold_punctuation(s) == s


class TestUntrainable:
    def test_detects_latin(self):
        assert untrainable_chars("hello नमस्ते") == set("helo")

    def test_clean_hindi_is_trainable(self):
        assert text_is_trainable("नमस्ते दुनिया। 123 ₹")

    def test_danda_is_trainable(self):
        assert text_is_trainable("यह ठीक है।")


class TestSanitize:
    def test_drops_unknown_but_keeps_latin(self):
        cleaned, bad = sanitize_for_training("नमस्ते ☺ hello")
        assert "☺" not in cleaned
        assert "hello" in cleaned          # Latin reported, never silently mangled
        assert "☺" in bad and "h" in bad

    def test_collapses_whitespace(self):
        cleaned, _ = sanitize_for_training("नमस्ते    दुनिया")
        assert cleaned == "नमस्ते दुनिया"

    def test_clean_text_unchanged(self):
        cleaned, bad = sanitize_for_training("यह ठीक है।")
        assert cleaned == "यह ठीक है।"
        assert bad == set()

    def test_no_drop_mode_reports_without_changing(self):
        cleaned, bad = sanitize_for_training("नमस्ते ☺", drop_unknown=False)
        assert "☺" in cleaned
        assert "☺" in bad


class TestLatinRatio:
    def test_pure_devanagari(self):
        assert latin_ratio("नमस्ते दुनिया") == 0.0

    def test_pure_latin(self):
        assert latin_ratio("hello world") == 1.0

    def test_no_letters(self):
        assert latin_ratio("123 !?") == 0.0

    def test_mixed(self):
        assert 0.0 < latin_ratio("नमस्ते hello") < 1.0
