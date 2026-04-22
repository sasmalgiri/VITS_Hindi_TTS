"""Tests for hindi_tts_builder.frontend.schwa."""
from hindi_tts_builder.frontend.schwa import delete_schwa


HALANT = "\u094d"


class TestBasicDeletion:
    def test_ram_adds_halant(self):
        # राम -> राम् (Ram, not Rama)
        assert delete_schwa("राम") == "राम" + HALANT

    def test_kamal(self):
        # कमल -> कमल् (kamal, final schwa deleted)
        assert delete_schwa("कमल") == "कमल" + HALANT


class TestNoDeletionNeeded:
    def test_ending_with_matra_untouched(self):
        # कैसे already ends with matra ए — no schwa to delete
        assert delete_schwa("कैसे") == "कैसे"

    def test_ending_with_halant_untouched(self):
        # नमस्ते ends in matra ए; no change
        assert delete_schwa("नमस्ते") == "नमस्ते"

    def test_independent_vowel_only(self):
        assert delete_schwa("और") == "और" + HALANT


class TestPreservesNonDevanagari:
    def test_roman_text_unchanged(self):
        assert delete_schwa("hello") == "hello"

    def test_punctuation_unchanged(self):
        result = delete_schwa("राम, सीता")
        assert "," in result
        assert HALANT in result  # should add halant to राम and सीता's final consonant

    def test_empty(self):
        assert delete_schwa("") == ""

    def test_whitespace(self):
        assert delete_schwa("   ") == "   "


class TestIdempotent:
    def test_already_has_halant(self):
        # If schwa already deleted, don't double-delete
        text = "राम" + HALANT
        assert delete_schwa(text) == text
