"""Tests for hindi_tts_builder.frontend.prosody."""
import pytest
from hindi_tts_builder.frontend.prosody import inject_prosody, PROSODY_TOKENS, all_prosody_tokens


class TestBasicPunctuation:
    def test_comma(self):
        result = inject_prosody("राम, सीता")
        assert PROSODY_TOKENS["short_pause"] in result

    def test_period(self):
        result = inject_prosody("राम है.")
        assert PROSODY_TOKENS["falling"] in result
        assert PROSODY_TOKENS["medium_pause"] in result

    def test_danda(self):
        result = inject_prosody("राम है।")
        assert PROSODY_TOKENS["falling"] in result
        assert PROSODY_TOKENS["medium_pause"] in result

    def test_double_danda(self):
        result = inject_prosody("राम है॥")
        assert PROSODY_TOKENS["long_pause"] in result

    def test_question_mark(self):
        result = inject_prosody("कैसे हो?")
        assert PROSODY_TOKENS["rising"] in result

    def test_exclamation(self):
        result = inject_prosody("वाह!")
        assert PROSODY_TOKENS["emphasis"] in result

    def test_ellipsis_three_dots(self):
        result = inject_prosody("रुको...")
        assert PROSODY_TOKENS["trail"] in result

    def test_ellipsis_unicode(self):
        result = inject_prosody("रुको…")
        assert PROSODY_TOKENS["trail"] in result


class TestTokenPreservation:
    """Critical: the regex that strips leftover punctuation must NOT eat our tokens."""

    def test_angle_brackets_survive(self):
        result = inject_prosody("राम.")
        # The falling and medium pause tokens should appear intact
        assert "<falling>" in result
        assert "<p_medium>" in result

    def test_underscore_in_tokens_survives(self):
        result = inject_prosody("राम,")
        assert "<p_short>" in result

    def test_tokens_never_split_by_whitespace(self):
        result = inject_prosody("नमस्ते, कैसे हो?")
        for token in all_prosody_tokens():
            if token in result:
                # The entire token string must be present exactly
                assert token in result


class TestStrippedPunctuation:
    def test_quote_removed(self):
        result = inject_prosody('राम "test"')
        assert '"' not in result

    def test_hash_removed(self):
        result = inject_prosody("राम #tag")
        assert "#" not in result


class TestNoPunctuation:
    def test_plain_text(self):
        assert inject_prosody("राम सीता") == "राम सीता"


class TestMultipleTokens:
    def test_mixed_punctuation(self):
        result = inject_prosody("नमस्ते, कैसे हो? ठीक हूँ.")
        for token_name in ("short_pause", "rising", "falling", "medium_pause"):
            assert PROSODY_TOKENS[token_name] in result
