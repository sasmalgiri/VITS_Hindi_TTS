"""Tests for hindi_tts_builder.frontend.transliterate."""
import json
import pytest
from pathlib import Path
from hindi_tts_builder.frontend.transliterate import Transliterator


class TestRuleBased:
    def test_simple_word(self):
        t = Transliterator()
        result = t.translit_word("Ram")
        # Rule-based is approximate; just confirm it produces Devanagari-ish output
        assert any(c == "र" for c in result)

    def test_empty_word(self):
        t = Transliterator()
        assert t.translit_word("") == ""


class TestDictionary:
    def test_dictionary_overrides_rules(self):
        t = Transliterator()
        t.add("ram", "राम")
        assert t.translit_word("ram") == "राम"
        # Case-insensitive lookup
        assert t.translit_word("Ram") == "राम"
        assert t.translit_word("RAM") == "राम"

    def test_save_and_load(self, tmp_path: Path):
        dict_path = tmp_path / "pron.json"
        t1 = Transliterator()
        t1.add("mana", "माना")
        t1.add("core", "कोर")
        t1.save(dict_path)

        t2 = Transliterator(dict_path)
        assert t2.translit_word("mana") == "माना"
        assert t2.translit_word("core") == "कोर"

    def test_missing_dict_ok(self, tmp_path: Path):
        # Non-existent path — should not error, just be empty
        dict_path = tmp_path / "missing.json"
        t = Transliterator(dict_path)
        assert t.dictionary == {}


class TestProcess:
    def test_passes_through_devanagari(self):
        t = Transliterator()
        text = "यह राम है"
        assert t.process(text) == text

    def test_replaces_latin_in_mixed(self):
        t = Transliterator()
        t.add("ram", "राम")
        result = t.process("यह ram है")
        assert "राम" in result
        assert "ram" not in result.lower()

    def test_preserves_punctuation_and_spaces(self):
        t = Transliterator()
        t.add("hello", "हेलो")
        result = t.process("hello, world!")
        assert "," in result
        assert "!" in result
